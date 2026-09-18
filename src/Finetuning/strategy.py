"""Training strategy boundary for the self-contained DFlash port."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .distributed import DistributedContext


@dataclass(frozen=True)
class StepOutput:
    """A scalar loss plus detached metrics and additive loss terms."""

    loss: torch.Tensor
    metrics: Dict[str, Any]
    ratio_metrics: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    loss_terms: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


@dataclass(frozen=True)
class StepContext:
    """Optimizer-step state passed through the generic trainer boundary."""

    global_step: int = 0
    total_steps: Optional[int] = None


@dataclass
class TrainBatch:
    """Minimal tensor-carrying batch contract required by DFlash."""

    tensors: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DraftTrainStrategy(abc.ABC):
    """Common strategy contract without a dependency on SpecForge runtime."""

    name: str
    required_features: set[str]

    @abc.abstractmethod
    def trainable_module(self) -> nn.Module:
        """Return the module whose parameters the trainer owns."""

    def validate_batch(self, batch: TrainBatch) -> None:
        missing = self.required_features - set(batch.tensors)
        if missing:
            raise ValueError(
                f"{self.name} batch missing required features {sorted(missing)}; "
                f"present={sorted(batch.tensors)}"
            )

    @abc.abstractmethod
    def forward_loss(
        self,
        batch: TrainBatch,
        ctx: Optional[StepContext] = None,
    ) -> StepOutput:
        """Compute the strategy-specific loss for one batch."""

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Select the state that should be persisted as draft weights."""

        return state_dict


def _detach_ratio_metrics(
    ratio_metrics: Dict[str, Tuple[Any, Any]],
) -> Dict[str, Tuple[Any, Any]]:
    detached: Dict[str, Tuple[Any, Any]] = {}
    for name, pair in ratio_metrics.items():
        detached[name] = tuple(
            value.detach() if isinstance(value, torch.Tensor) else value
            for value in pair
        )  # type: ignore[assignment]
    return detached


class DFlashTrainStrategy(DraftTrainStrategy):
    """Wrap ``OnlineDFlashModel`` for the generic trainer lifecycle."""

    name = "dflash"
    required_features = {"input_ids", "hidden_states", "loss_mask"}

    def __init__(self, dflash_model: nn.Module) -> None:
        self.dflash_model = dflash_model
        self._forward_model: nn.Module = dflash_model

    @property
    def forward_model(self) -> nn.Module:
        """Return the local or DDP-wrapped module used for forward calls."""

        return self._forward_model

    def configure_distributed(self, context: DistributedContext) -> None:
        """Wrap the objective forward path in DDP when torchrun is active."""

        if not context.is_distributed:
            self._forward_model = self.dflash_model
            return
        if isinstance(self._forward_model, DistributedDataParallel):
            return
        device_ids = [context.local_rank] if self._device_type() == "cuda" else None
        self._forward_model = DistributedDataParallel(
            self.dflash_model,
            device_ids=device_ids,
            output_device=context.local_rank if device_ids is not None else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    def _device_type(self) -> str:
        for parameter in self.dflash_model.parameters():
            return parameter.device.type
        return "cpu"

    def trainable_module(self) -> nn.Module:
        # Target embedding and LM head are frozen by OnlineDFlashModel; keeping
        # the wrapper here preserves the upstream trainer/model boundary.
        return self.dflash_model

    def _device(self) -> torch.device:
        return next(self.dflash_model.parameters()).device

    def _draft_dtype(self) -> torch.dtype:
        draft_module = getattr(self.dflash_model, "draft_model", self.dflash_model)
        for parameter in draft_module.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        raise ValueError("DFlash draft module has no floating-point parameters")

    def forward_loss(
        self,
        batch: TrainBatch,
        ctx: Optional[StepContext] = None,
    ) -> StepOutput:
        del ctx
        self.validate_batch(batch)
        tensors = batch.tensors
        device = self._device()
        loss, accuracy, model_metrics = self._forward_model(
            input_ids=tensors["input_ids"].to(device),
            hidden_states=tensors["hidden_states"].to(
                device=device,
                dtype=self._draft_dtype(),
            ),
            loss_mask=tensors["loss_mask"].to(device),
        )
        metrics: Dict[str, Any] = {"accuracy": accuracy.detach()}
        if "accuracy_denom" in model_metrics:
            accuracy_denom = model_metrics["accuracy_denom"]
            metrics["accuracy_denom"] = (
                accuracy_denom.detach()
                if isinstance(accuracy_denom, torch.Tensor)
                else accuracy_denom
            )
        raw_loss_terms = model_metrics.get("loss_terms")
        loss_terms = None
        if raw_loss_terms is not None:
            loss_terms = (
                raw_loss_terms[0],
                raw_loss_terms[1].detach()
                if isinstance(raw_loss_terms[1], torch.Tensor)
                else raw_loss_terms[1],
            )
        return StepOutput(
            loss=loss.reshape(()),
            metrics=metrics,
            ratio_metrics=_detach_ratio_metrics(
                model_metrics.get("ratio_metrics", {})
            ),
            loss_terms=loss_terms,
        )

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Persist only the draft namespace, with its upstream local keys."""

        return {
            key.replace("draft_model.", ""): value
            for key, value in state_dict.items()
            if "draft_model." in key
        }


__all__ = [
    "DFlashTrainStrategy",
    "DraftTrainStrategy",
    "StepContext",
    "StepOutput",
    "TrainBatch",
]
