"""Optimizer-step schedules for the self-contained DFlash trainer."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def resolve_total_steps(
    total_steps: int | None,
    max_steps: int | None,
    num_samples: int,
    batch_size: int,
    accumulation_steps: int,
    num_epochs: int,
) -> int:
    """Resolve the number of completed optimizer updates.

    The fixed-data contract intentionally uses complete micro-batches only.
    Callers should run :func:`validate_fixed_accumulation_plan` first when a
    partial final batch must be rejected instead of silently dropped.
    """

    for name, value in (
        ("num_samples", num_samples),
        ("batch_size", batch_size),
        ("accumulation_steps", accumulation_steps),
        ("num_epochs", num_epochs),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if total_steps is not None and (
        not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps <= 0
    ):
        raise ValueError("total_steps must be a positive integer when provided")
    if max_steps is not None and (
        not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer when provided")

    natural_micro_batches = num_samples // batch_size
    natural_steps = (natural_micro_batches * num_epochs) // accumulation_steps
    if natural_steps <= 0:
        raise ValueError(
            "training data does not contain one complete optimizer update"
        )

    resolved = natural_steps
    if total_steps is not None:
        resolved = min(resolved, total_steps)
    if max_steps is not None:
        resolved = min(resolved, max_steps)
    if resolved <= 0:
        raise ValueError("resolved total_steps must be positive")
    return resolved


def validate_fixed_accumulation_plan(
    num_samples: int,
    batch_size: int,
    accumulation_steps: int,
    num_epochs: int,
    max_steps: int | None,
) -> None:
    """Reject a fixed loader whose tail cannot form a complete update."""

    for name, value in (
        ("num_samples", num_samples),
        ("batch_size", batch_size),
        ("accumulation_steps", accumulation_steps),
        ("num_epochs", num_epochs),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_samples % batch_size:
        raise ValueError(
            "incomplete gradient accumulation: the final batch is smaller than "
            f"batch_size={batch_size}"
        )

    micro_batches = (num_samples // batch_size) * num_epochs
    complete_steps, remainder = divmod(micro_batches, accumulation_steps)
    stops_before = max_steps is not None and max_steps <= complete_steps
    if remainder and not stops_before:
        raise ValueError(
            "incomplete gradient accumulation: "
            f"{micro_batches} micro-batches do not divide evenly by "
            f"accumulation_steps={accumulation_steps}"
        )


def _warmup_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps]")
    if warmup_steps and step < warmup_steps:
        # LambdaLR performs its first update at logical step 0 during
        # construction.  Treat that as the first warmup point so a 50% /
        # two-step warmup starts at 0.5*base_lr and reaches base_lr after the
        # first explicit scheduler.step().
        return float(step + 1) / float(warmup_steps)
    return 1.0


def _schedule_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    scheduler_type: str,
) -> float:
    warmup = _warmup_factor(step, total_steps, warmup_steps)
    if step < warmup_steps:
        return warmup
    if scheduler_type == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


class WarmupLambdaLR(LambdaLR):
    """LambdaLR that also restores the optimizer's current LR on load."""

    def load_state_dict(self, state_dict):  # type: ignore[override]
        super().load_state_dict(state_dict)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * self.lr_lambdas[0](self.last_epoch)


def build_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int | None = None,
    warmup_ratio: float | None = None,
    scheduler_type: str = "cosine",
) -> WarmupLambdaLR:
    """Build a warmup + cosine/constant scheduler over optimizer steps."""

    if scheduler_type not in {"cosine", "constant"}:
        raise ValueError("scheduler_type must be 'cosine' or 'constant'")
    if not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if warmup_steps is not None and warmup_ratio is not None:
        raise ValueError("provide warmup_steps or warmup_ratio, not both")
    if warmup_ratio is not None:
        if not 0.0 <= warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be in [0, 1]")
        warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps is None:
        warmup_steps = 0
    if not isinstance(warmup_steps, int) or not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps]")

    fn: Callable[[int], float] = lambda step: _schedule_factor(
        step, total_steps, warmup_steps, scheduler_type
    )
    return WarmupLambdaLR(optimizer, lr_lambda=fn, last_epoch=-1)


def current_lr(optimizer: Optimizer) -> float:
    """Return the first parameter group's current learning rate."""

    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


__all__ = [
    "WarmupLambdaLR",
    "build_scheduler",
    "current_lr",
    "resolve_total_steps",
    "validate_fixed_accumulation_plan",
]
