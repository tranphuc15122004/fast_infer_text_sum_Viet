"""Small, optional distributed-runtime helpers for the Finetuning package."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class DistributedContext:
    """Process identity and collectives used by one Finetuning worker."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    owns_process_group: bool = False

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be within [0, world_size)")
        if self.local_rank < 0:
            raise ValueError("local_rank must be non-negative")

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def global_batch_size(self, local_batch_size: int, accumulation_steps: int) -> int:
        if local_batch_size <= 0 or accumulation_steps <= 0:
            raise ValueError("batch size and accumulation steps must be positive")
        return local_batch_size * self.world_size * accumulation_steps

    def barrier(self) -> None:
        if self.is_distributed:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed barrier requested before process-group initialization")
            dist.barrier()

    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("all_reduce_sum expects a torch.Tensor")
        if not self.is_distributed:
            return value
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed reduction requested before process-group initialization")
        reduced = value.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

    def _all_reduce(self, value: torch.Tensor, op: dist.ReduceOp) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("distributed reduction expects a torch.Tensor")
        if not self.is_distributed:
            return value
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed reduction requested before process-group initialization")
        reduced = value.clone()
        dist.all_reduce(reduced, op=op)
        return reduced

    def all_reduce_min(self, value: torch.Tensor) -> torch.Tensor:
        """Return the element-wise minimum across all workers."""

        return self._all_reduce(value, dist.ReduceOp.MIN)

    def all_reduce_max(self, value: torch.Tensor) -> torch.Tensor:
        """Return the element-wise maximum across all workers."""

        return self._all_reduce(value, dist.ReduceOp.MAX)


def initialize_distributed(
    device: str | torch.device | None = None,
    *,
    backend: str | None = None,
) -> DistributedContext:
    """Initialize a torchrun process group, or return a single-process context."""

    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if world_size == 1:
        return DistributedContext()
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable but WORLD_SIZE > 1")
    if rank >= world_size:
        raise ValueError("RANK must be smaller than WORLD_SIZE")

    if device is None or str(device) == "auto":
        requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested_device = torch.device(device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("distributed CUDA training requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        requested_device = torch.device("cuda", local_rank)
    selected_backend = backend or ("nccl" if requested_device.type == "cuda" else "gloo")
    owns_process_group = False
    if not dist.is_initialized():
        init_method = os.environ.get("INIT_METHOD")
        kwargs = {
            "backend": selected_backend,
            "rank": rank,
            "world_size": world_size,
        }
        if init_method:
            kwargs["init_method"] = init_method
        dist.init_process_group(**kwargs)
        owns_process_group = True
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        owns_process_group=owns_process_group,
    )


def cleanup_distributed(context: DistributedContext | None) -> None:
    """Destroy only process groups created by this invocation."""

    if context is None or not context.owns_process_group:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def shard_indices(length: int, rank: int, world_size: int) -> Iterable[int]:
    """Yield deterministic round-robin indices for a rank-local data shard."""

    if not isinstance(length, int) or length < 0:
        raise ValueError("length must be a non-negative integer")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("rank must be within [0, world_size)")
    return range(rank, length, world_size)


def ranked_path(path: str | Path, rank: int, world_size: int) -> Path:
    """Return a rank-local sibling path while keeping single-process paths stable."""

    source = Path(path)
    if world_size == 1:
        return source
    return source.with_name(f"{source.name}.rank{rank:05d}")


def merge_ranked_jsonl(
    shard_paths: Iterable[str | Path],
    destination: str | Path,
) -> None:
    """Merge rank-local JSONL rows by their internal source index atomically."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"distributed merge destination already exists: {target}")
    rows: list[tuple[int, dict[str, object]]] = []
    seen: set[int] = set()
    for raw_path in shard_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"distributed shard not found: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"distributed row must be an object: {path}:{line_number}")
                source_index = payload.pop("_source_index", None)
                if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
                    raise ValueError(
                        f"distributed row lacks a non-negative _source_index: {path}:{line_number}"
                    )
                if source_index in seen:
                    raise ValueError(f"duplicate distributed source index: {source_index}")
                seen.add(source_index)
                rows.append((source_index, payload))
    rows.sort(key=lambda item: item[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for _source_index, payload in rows:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DistributedContext",
    "cleanup_distributed",
    "initialize_distributed",
    "merge_ranked_jsonl",
    "ranked_path",
    "shard_indices",
]
