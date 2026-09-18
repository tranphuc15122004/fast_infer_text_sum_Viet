"""CUDA preflight tuning for a safe, fixed per-GPU training batch size.

The tuner deliberately runs before DDP wrapping.  Every worker measures the
same real DFlash objective locally, then workers agree on the smallest batch
that fits their device.  The selected value is kept fixed for the run so the
optimizer, scheduler and checkpoint semantics remain deterministic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import capture_rng_state, restore_rng_state
from .distributed import DistributedContext
from .features import collate_features
from .strategy import DFlashTrainStrategy, TrainBatch


@dataclass(frozen=True)
class AdaptiveBatchSettings:
    """Controls one preflight search, not dynamic changes during training."""

    enabled: bool = True
    target_memory_fraction: float = 0.90
    min_batch_size: int = 1
    max_batch_size: int = 256
    probe_batches: int = 2


@dataclass(frozen=True)
class BatchProbe:
    batch_size: int
    peak_reserved_bytes: int
    target_memory_bytes: int
    fits: bool


@dataclass(frozen=True)
class AdaptiveBatchResult:
    """Serializable result recorded in every checkpoint."""

    batch_size: int
    peak_reserved_bytes: int
    target_memory_bytes: int
    target_memory_fraction: float
    world_size: int
    probe_batches: int
    max_batch_size: int
    hit_maximum: bool

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def validate_adaptive_batch_settings(settings: AdaptiveBatchSettings) -> None:
    if not 0.0 < settings.target_memory_fraction < 1.0:
        raise ValueError("target_memory_fraction must be between 0 and 1")
    if settings.min_batch_size <= 0:
        raise ValueError("min_batch_size must be positive")
    if settings.max_batch_size < settings.min_batch_size:
        raise ValueError("max_batch_size must be >= min_batch_size")
    if settings.probe_batches <= 0:
        raise ValueError("probe_batches must be positive")


def is_cuda_oom(error: BaseException) -> bool:
    """Recognize allocator OOMs without masking unrelated runtime failures."""

    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def search_largest_fitting_batch(
    *,
    minimum: int,
    maximum: int,
    probe: Callable[[int], bool],
) -> int:
    """Find the largest fitting integer using exponential then binary search."""

    if minimum <= 0:
        raise ValueError("minimum batch size must be positive")
    if maximum < minimum:
        raise ValueError("maximum batch size must be >= minimum")

    if not probe(minimum):
        raise RuntimeError(
            "adaptive batch-size preflight could not fit the minimum batch size "
            f"({minimum})"
        )
    best = minimum
    candidate = minimum * 2
    first_failure: int | None = None
    while candidate <= maximum:
        if probe(candidate):
            best = candidate
            candidate *= 2
            continue
        first_failure = candidate
        break

    if best == maximum:
        return best
    high = min(maximum, (first_failure - 1) if first_failure is not None else maximum)
    low = best + 1
    while low <= high:
        candidate = (low + high) // 2
        if probe(candidate):
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return best


def _clear_dflash_caches(strategy: DFlashTrainStrategy) -> None:
    model = strategy.dflash_model
    for name in ("_cached_block_mask", "_cached_seq_len", "_cached_bsz"):
        if hasattr(model, name):
            setattr(model, name, None)


def _probe_one_batch_size(
    strategy: DFlashTrainStrategy,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
    context: DistributedContext,
    target_memory_bytes: int,
    probe_batches: int,
) -> BatchProbe:
    """Run a bounded real forward/backward probe and restore all model state."""

    sampler = None
    if context.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_features,
        num_workers=0,
        pin_memory=False,
    )
    module = strategy.dflash_model
    draft = module.draft_model
    draft_state = {
        key: value.detach().cpu().clone()
        for key, value in draft.state_dict().items()
    }
    rng_state = capture_rng_state()
    was_training = module.training
    peak_reserved = 0
    fits = False
    try:
        module.train()
        module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        batches_seen = 0
        for index, raw_batch in enumerate(loader):
            if index >= probe_batches:
                break
            output = strategy.forward_loss(TrainBatch(tensors=raw_batch))
            if not torch.isfinite(output.loss.detach()).all():
                raise ValueError("adaptive batch-size probe produced a non-finite loss")
            output.loss.backward()
            module.zero_grad(set_to_none=True)
            batches_seen += 1
        if batches_seen == 0:
            raise ValueError(
                "adaptive batch-size probe received no complete feature batch"
            )
        torch.cuda.synchronize(device)
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        fits = peak_reserved <= target_memory_bytes
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        fits = False
        torch.cuda.empty_cache()
    finally:
        module.zero_grad(set_to_none=True)
        draft.load_state_dict(draft_state, strict=True)
        module.train(was_training)
        _clear_dflash_caches(strategy)
        restore_rng_state(rng_state)
        torch.cuda.empty_cache()
    return BatchProbe(
        batch_size=batch_size,
        peak_reserved_bytes=peak_reserved,
        target_memory_bytes=target_memory_bytes,
        fits=fits,
    )


def tune_batch_size(
    strategy: DFlashTrainStrategy,
    dataset: Dataset,
    *,
    device: torch.device,
    context: DistributedContext | None = None,
    settings: AdaptiveBatchSettings | None = None,
) -> AdaptiveBatchResult:
    """Select a fixed local batch size using real CUDA memory measurements."""

    context = context or DistributedContext()
    settings = settings or AdaptiveBatchSettings()
    validate_adaptive_batch_settings(settings)
    if not settings.enabled:
        raise ValueError("tune_batch_size requires adaptive batch sizing to be enabled")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("adaptive batch-size tuning requires a CUDA device")
    if len(dataset) < context.world_size * settings.min_batch_size:
        raise ValueError(
            "feature dataset is too small for adaptive batch-size tuning: "
            f"need at least {context.world_size * settings.min_batch_size} samples, "
            f"got {len(dataset)}"
        )

    device_properties = torch.cuda.get_device_properties(device)
    target_memory_bytes = int(
        int(device_properties.total_memory) * settings.target_memory_fraction
    )
    maximum = min(settings.max_batch_size, len(dataset) // context.world_size)
    local_probes: dict[int, BatchProbe] = {}

    def probe(batch_size: int) -> bool:
        result = _probe_one_batch_size(
            strategy,
            dataset,
            batch_size=batch_size,
            device=device,
            context=context,
            target_memory_bytes=target_memory_bytes,
            probe_batches=settings.probe_batches,
        )
        local_probes[batch_size] = result
        return result.fits

    local_selected = search_largest_fitting_batch(
        minimum=settings.min_batch_size,
        maximum=maximum,
        probe=probe,
    )
    selected = int(
        context.all_reduce_min(
            torch.tensor(local_selected, dtype=torch.int64, device=device)
        ).item()
    )
    local_peak = local_probes.get(selected)
    peak = local_peak.peak_reserved_bytes if local_peak is not None else 0
    global_peak = int(
        context.all_reduce_max(
            torch.tensor(peak, dtype=torch.int64, device=device)
        ).item()
    )
    return AdaptiveBatchResult(
        batch_size=selected,
        peak_reserved_bytes=global_peak,
        target_memory_bytes=target_memory_bytes,
        target_memory_fraction=settings.target_memory_fraction,
        world_size=context.world_size,
        probe_batches=settings.probe_batches,
        max_batch_size=maximum,
        hit_maximum=selected == maximum,
    )


__all__ = [
    "AdaptiveBatchResult",
    "AdaptiveBatchSettings",
    "BatchProbe",
    "is_cuda_oom",
    "search_largest_fitting_batch",
    "tune_batch_size",
    "validate_adaptive_batch_settings",
]
