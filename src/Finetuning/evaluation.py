"""Evaluation shared by in-memory and checkpoint-restored models."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

from .checkpoint import CheckpointManager
from .distributed import DistributedContext
from .metrics import aggregate_step_outputs
from .strategy import StepContext, TrainBatch


def _as_batch(value: Any) -> TrainBatch:
    if isinstance(value, TrainBatch):
        return value
    if isinstance(value, dict):
        return TrainBatch(tensors=value)
    raise TypeError(f"dataloader item must be a mapping or TrainBatch, got {type(value)!r}")


class Evaluator:
    """Run finite validation and reject empty/non-finite results."""

    def __init__(self, distributed_context: DistributedContext | None = None) -> None:
        self.distributed_context = distributed_context or DistributedContext()

    def evaluate_in_memory(
        self,
        model: Any,
        dataloader: Iterable[Any],
        device: torch.device,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        del device
        module = model.trainable_module() if hasattr(model, "trainable_module") else model
        was_training = bool(module.training) if hasattr(module, "training") else False
        if hasattr(module, "eval"):
            module.eval()
        outputs = []
        try:
            with torch.no_grad():
                for index, raw_batch in enumerate(dataloader):
                    if max_batches is not None and index >= max_batches:
                        break
                    outputs.append(model.forward_loss(_as_batch(raw_batch), StepContext()))
        finally:
            if was_training and hasattr(module, "train"):
                module.train()
        if not outputs:
            raise ValueError("empty validation loader")
        return aggregate_step_outputs(outputs, self.distributed_context)

    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        build_model: Callable[[], Any],
        dataloader_factory: Callable[[], Iterable[Any]],
        device: torch.device,
    ) -> dict[str, float]:
        model = build_model()
        module = model.trainable_module() if hasattr(model, "trainable_module") else model
        # Resolve/load without assuming the caller's output directory naming.
        from pathlib import Path

        root = Path(checkpoint_path)
        while root.name and not (root / "COMPLETE").is_file() and root.parent != root:
            root = root.parent
        if not (root / "COMPLETE").is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        manager = CheckpointManager(root.parent, root.name.rsplit("-step", 1)[0])
        state = manager.load(root, map_location=device)
        state_dict = state["draft_state_dict"]
        load_module = module
        # DFlash checkpoints intentionally contain draft-local keys while the
        # optimizer owns the wrapper.  Generic strategies keep their own
        # state-dict contract and load directly.
        if hasattr(model, "dflash_model"):
            load_module = model.dflash_model.draft_model
        load_module.load_state_dict(state_dict, strict=False)
        module.to(device)
        return self.evaluate_in_memory(model, dataloader_factory(), device)


__all__ = ["Evaluator"]
