from __future__ import annotations

import pytest
import torch

try:
    from Finetuning.schedule import (
        build_scheduler,
        resolve_total_steps,
        validate_fixed_accumulation_plan,
    )
except ModuleNotFoundError as exc:  # Red phase: lifecycle is not ported yet.
    _IMPORT_ERROR = exc


def _require_api() -> None:
    if "_IMPORT_ERROR" in globals():
        pytest.fail(f"schedule API is not implemented: {_IMPORT_ERROR}")


def test_total_steps_matches_optimizer_updates() -> None:
    _require_api()
    assert resolve_total_steps(None, None, 12, 2, 3, 2) == 4


def test_partial_accumulation_is_rejected() -> None:
    _require_api()
    with pytest.raises(ValueError, match="incomplete gradient accumulation"):
        validate_fixed_accumulation_plan(
            num_samples=5,
            batch_size=2,
            accumulation_steps=2,
            num_epochs=1,
            max_steps=None,
        )


def test_warmup_cosine_scheduler_state_restores_next_learning_rate() -> None:
    _require_api()
    first = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([first], lr=0.1)
    scheduler = build_scheduler(
        optimizer,
        total_steps=6,
        warmup_steps=2,
        scheduler_type="cosine",
    )
    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])

    second = torch.nn.Parameter(torch.ones(()))
    restored_optimizer = torch.optim.AdamW([second], lr=0.1)
    restored_scheduler = build_scheduler(
        restored_optimizer,
        total_steps=6,
        warmup_steps=2,
        scheduler_type="cosine",
    )
    restored_scheduler.load_state_dict(scheduler.state_dict())

    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(observed[-1])
    scheduler.step()
    restored_scheduler.step()
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )


def test_constant_scheduler_warms_then_holds_base_rate() -> None:
    _require_api()
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.2)
    scheduler = build_scheduler(
        optimizer,
        total_steps=4,
        warmup_ratio=0.5,
        scheduler_type="constant",
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
