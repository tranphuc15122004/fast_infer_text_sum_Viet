from __future__ import annotations

import json

import pytest
import torch
from torch import nn

try:
    from Finetuning.strategy import StepOutput, TrainBatch
    from Finetuning.trainer import Trainer
except ModuleNotFoundError as exc:  # Red phase: lifecycle is not ported yet.
    _IMPORT_ERROR = exc


def _require_api() -> None:
    if "_IMPORT_ERROR" in globals():
        pytest.fail(f"trainer API is not implemented: {_IMPORT_ERROR}")


class _CountingStrategy:
    name = "tiny"
    required_features = {"x", "target"}

    def __init__(self, *, nonfinite: bool = False) -> None:
        self.module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(0.0)
        self.nonfinite = nonfinite

    def trainable_module(self) -> nn.Module:
        return self.module

    def forward_loss(self, batch: TrainBatch, ctx=None) -> StepOutput:
        del ctx
        if self.nonfinite:
            loss = torch.tensor(float("nan"), requires_grad=True)
            return StepOutput(loss=loss, metrics={})
        x = batch.tensors["x"].float()
        target = batch.tensors["target"].float()
        error = self.module(x) - target
        numerator = error.square().sum()
        denominator = torch.tensor(float(error.numel()))
        correct = (self.module(x).round() == target).float().sum()
        return StepOutput(
            loss=numerator / denominator,
            metrics={"accuracy": correct / denominator, "accuracy_denom": denominator},
            ratio_metrics={"acc": (correct.detach(), denominator)},
            loss_terms=(numerator, denominator),
        )

    def checkpoint_state_filter(self, state_dict):
        return dict(state_dict)


def _train_loader():
    return [
        {"x": torch.tensor([[1.0]]), "target": torch.tensor([[1.0]])},
        {"x": torch.tensor([[1.0]]), "target": torch.tensor([[1.0]])},
        {"x": torch.tensor([[1.0]]), "target": torch.tensor([[1.0]])},
        {"x": torch.tensor([[1.0]]), "target": torch.tensor([[1.0]])},
    ]


def test_trainer_steps_only_at_accumulation_boundary_and_honors_max_steps(tmp_path) -> None:
    _require_api()
    strategy = _CountingStrategy()
    trainer = Trainer(
        strategy=strategy,
        train_dataloader=_train_loader(),
        output_dir=tmp_path,
        run_id="tiny",
        batch_size=1,
        accumulation_steps=2,
        num_epochs=1,
        max_steps=2,
        learning_rate=0.1,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
    )
    calls = 0
    original_step = trainer.optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counted_step
    assert trainer.fit() == 2
    assert calls == 2

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2]
    for record in records:
        assert {"loss", "grad_norm", "lr", "step_time_s", "tokens_per_s", "mfu"} <= set(record)
        assert record["mfu"] is None
    assert "step=1" in (tmp_path / "train.log").read_text()
    assert trainer.checkpoint_manager.latest_dir().name == "tiny-step2"


def test_trainer_fails_with_descriptive_nonfinite_loss(tmp_path) -> None:
    _require_api()
    trainer = Trainer(
        strategy=_CountingStrategy(nonfinite=True),
        train_dataloader=_train_loader(),
        output_dir=tmp_path,
        run_id="nan",
        batch_size=1,
        accumulation_steps=1,
        num_epochs=1,
        max_steps=1,
    )
    with pytest.raises(ValueError, match="non-finite loss"):
        trainer.fit()


def test_trainer_rejects_empty_validation_before_training(tmp_path) -> None:
    _require_api()
    trainer = Trainer(
        strategy=_CountingStrategy(),
        train_dataloader=_train_loader(),
        validation_dataloader=[],
        output_dir=tmp_path,
        run_id="empty-val",
        batch_size=1,
        accumulation_steps=1,
        num_epochs=1,
        max_steps=1,
    )
    with pytest.raises(ValueError, match="empty validation"):
        trainer.fit()


def test_trainer_always_saves_final_checkpoint_and_can_resume(tmp_path) -> None:
    _require_api()
    first = Trainer(
        strategy=_CountingStrategy(),
        train_dataloader=_train_loader(),
        output_dir=tmp_path,
        run_id="resume",
        batch_size=1,
        accumulation_steps=1,
        num_epochs=1,
        max_steps=1,
        save_interval=100,
        learning_rate=0.1,
    )
    assert first.fit() == 1
    checkpoint = first.checkpoint_manager.latest_dir()

    resumed = Trainer(
        strategy=_CountingStrategy(),
        train_dataloader=_train_loader(),
        output_dir=tmp_path,
        run_id="resume-continued",
        batch_size=1,
        accumulation_steps=1,
        num_epochs=1,
        max_steps=2,
        save_interval=100,
        learning_rate=0.1,
        resume_from=checkpoint,
    )
    assert resumed.fit() == 2


def test_trainer_reports_mfu_only_when_peak_is_configured(tmp_path) -> None:
    _require_api()
    trainer = Trainer(
        strategy=_CountingStrategy(),
        train_dataloader=_train_loader(),
        output_dir=tmp_path,
        run_id="mfu",
        batch_size=1,
        accumulation_steps=1,
        num_epochs=1,
        max_steps=1,
        hardware_peak_tflops=100.0,
    )
    trainer.fit()
    record = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert record["mfu"] == 0.0


def test_trainer_accumulation_normalizes_additive_loss_terms(tmp_path) -> None:
    _require_api()

    class WeightedStrategy:
        name = "weighted"

        def __init__(self) -> None:
            self.parameter = nn.Parameter(torch.tensor(0.0))
            self.module = nn.Module()
            self.module.register_parameter("weight", self.parameter)
            self.calls = 0

        def trainable_module(self) -> nn.Module:
            return self.module

        def forward_loss(self, _batch, _ctx) -> StepOutput:
            coefficient, denominator = ((2.0, 1.0), (-6.0, 10.0))[self.calls]
            self.calls += 1
            numerator = self.parameter * coefficient
            return StepOutput(
                loss=numerator / denominator,
                metrics={},
                loss_terms=(numerator, torch.tensor(denominator)),
            )

    strategy = WeightedStrategy()
    trainer = Trainer(
        strategy=strategy,
        train_dataloader=[{}, {}],
        output_dir=tmp_path,
        run_id="weighted",
        batch_size=1,
        accumulation_steps=2,
        num_epochs=1,
        max_steps=1,
        learning_rate=0.1,
        warmup_ratio=0.0,
    )

    trainer.fit()

    # d((2w - 6w) / (1 + 10))/dw < 0, so AdamW must move w upward.
    assert strategy.parameter.item() > 0


def test_trainer_rejects_resume_with_incompatible_draft_provenance(tmp_path) -> None:
    _require_api()
    metadata = {"target_model_path": "/models/qwen3", "block_size": 16}
    first = Trainer(
        strategy=_CountingStrategy(),
        train_dataloader=_train_loader(),
        output_dir=tmp_path,
        run_id="source",
        batch_size=1,
        accumulation_steps=1,
        num_epochs=1,
        max_steps=1,
        draft_export_metadata=metadata,
    )
    first.fit()

    with pytest.raises(ValueError, match="metadata mismatch"):
        Trainer(
            strategy=_CountingStrategy(),
            train_dataloader=_train_loader(),
            output_dir=tmp_path,
            run_id="target",
            batch_size=1,
            accumulation_steps=1,
            num_epochs=1,
            max_steps=2,
            draft_export_metadata={**metadata, "block_size": 8},
            resume_from=first.checkpoint_manager.latest_dir(),
        )
