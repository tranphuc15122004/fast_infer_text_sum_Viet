from __future__ import annotations

import random

import pytest
import torch
from torch import nn

try:
    from Finetuning.checkpoint import (
        CheckpointManager,
        capture_rng_state,
        export_draft,
        load_draft_initialization,
        restore_rng_state,
    )
    from Finetuning.schedule import build_scheduler
except ModuleNotFoundError as exc:  # Red phase: lifecycle is not ported yet.
    _IMPORT_ERROR = exc


def _require_api() -> None:
    if "_IMPORT_ERROR" in globals():
        pytest.fail(f"checkpoint API is not implemented: {_IMPORT_ERROR}")


class _DraftContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.draft_model = nn.Linear(2, 2, bias=False)
        self.target = nn.Linear(2, 2, bias=False)
        self.target.requires_grad_(False)


class _Strategy:
    name = "dflash"
    required_features = {"x"}

    def __init__(self) -> None:
        self.module = _DraftContainer()

    def trainable_module(self) -> nn.Module:
        return self.module

    def checkpoint_state_filter(self, state_dict: dict[str, torch.Tensor]):
        return {
            key.removeprefix("draft_model."): value
            for key, value in state_dict.items()
            if key.startswith("draft_model.")
        }


def test_checkpoint_round_trip_restores_training_state(tmp_path) -> None:
    _require_api()
    strategy = _Strategy()
    optimizer = torch.optim.AdamW(strategy.trainable_module().parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, total_steps=4, warmup_steps=1)
    manager = CheckpointManager(tmp_path, "tiny")

    path = manager.save(
        step=3,
        model=strategy,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"global_step": 3, "micro_step": 7, "seed": 42},
        extra={"strategy": "dflash", "resolved_config": {"lr": 0.1}},
    )
    restored = manager.load(path)

    assert restored["trainer_state"]["global_step"] == 3
    assert restored["trainer_state"]["micro_step"] == 7
    assert restored["extra"]["strategy"] == "dflash"
    assert set(restored["draft_state_dict"]) == {"weight"}
    assert (tmp_path / "tiny-latest").exists()
    assert manager.latest_dir() == path
    assert manager.resolve_resume_dir(tmp_path) == path
    assert "optimizer" in restored and "scheduler" in restored
    assert "rng_state" in restored


def test_checkpoint_restores_optimizer_scheduler_and_rng_state(tmp_path) -> None:
    _require_api()
    strategy = _Strategy()
    optimizer = torch.optim.AdamW(strategy.trainable_module().parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, total_steps=5, warmup_steps=1)
    loss = strategy.module.draft_model.weight.square().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    manager = CheckpointManager(tmp_path, "restore")

    random.seed(19)
    torch.manual_seed(19)
    path = manager.save(
        step=1,
        model=strategy,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"global_step": 1},
        extra={},
    )
    state = manager.load(path)
    expected_python = random.random()
    expected_torch = torch.rand(3)

    new_strategy = _Strategy()
    new_optimizer = torch.optim.AdamW(
        new_strategy.trainable_module().parameters(), lr=0.1
    )
    new_scheduler = build_scheduler(new_optimizer, total_steps=5, warmup_steps=1)
    new_optimizer.load_state_dict(state["optimizer"])
    new_scheduler.load_state_dict(state["scheduler"])
    new_strategy.module.draft_model.load_state_dict(state["draft_state_dict"])
    restore_rng_state(state["rng_state"])

    assert new_optimizer.state_dict()["state"]
    assert new_scheduler.last_epoch == scheduler.last_epoch
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)


def test_checkpoint_rotation_keeps_only_requested_complete_steps(tmp_path) -> None:
    _require_api()
    strategy = _Strategy()
    optimizer = torch.optim.AdamW(strategy.trainable_module().parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, total_steps=4)
    manager = CheckpointManager(tmp_path, "rotate", max_checkpoints=2)
    for step in range(1, 4):
        manager.save(step, strategy, optimizer, scheduler, {"global_step": step}, {})
    assert sorted(path.name for path in tmp_path.glob("rotate-step*")) == [
        "rotate-step2",
        "rotate-step3",
    ]


def test_portable_draft_export_warm_starts_only_matching_metadata(tmp_path) -> None:
    _require_api()
    source = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        source.weight.fill_(0.25)
    metadata = {
        "target_model_path": "/models/qwen3",
        "target_layer_ids": [1, 9, 17, 25, 33],
        "block_size": 16,
        "mask_token_id": 151669,
    }
    export_path = tmp_path / "portable"

    export_draft(export_path, source, metadata)
    restored = nn.Linear(2, 2, bias=False)
    load_draft_initialization(export_path, restored, metadata)

    assert torch.equal(restored.weight, source.weight)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_draft_initialization(
            export_path,
            nn.Linear(2, 2, bias=False),
            {**metadata, "block_size": 8},
        )


def test_checkpoint_can_embed_portable_draft_export(tmp_path) -> None:
    _require_api()
    strategy = _Strategy()
    optimizer = torch.optim.AdamW(strategy.trainable_module().parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, total_steps=1)
    metadata = {"target_model_path": "/models/qwen3", "block_size": 16}

    path = CheckpointManager(tmp_path, "export").save(
        1,
        strategy,
        optimizer,
        scheduler,
        {"global_step": 1},
        {},
        draft_export_metadata=metadata,
    )
    restored = nn.Linear(2, 2, bias=False)
    load_draft_initialization(path / "draft_export", restored, metadata)

    assert torch.equal(restored.weight, strategy.module.draft_model.weight)
