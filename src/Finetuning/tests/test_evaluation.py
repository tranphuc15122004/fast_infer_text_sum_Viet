from __future__ import annotations

import pytest
import torch
from torch import nn

try:
    from Finetuning.checkpoint import CheckpointManager
    from Finetuning.evaluation import Evaluator
    from Finetuning.schedule import build_scheduler
    from Finetuning.strategy import StepOutput, TrainBatch
except ModuleNotFoundError as exc:  # Red phase: lifecycle is not ported yet.
    _IMPORT_ERROR = exc


def _require_api() -> None:
    if "_IMPORT_ERROR" in globals():
        pytest.fail(f"evaluation API is not implemented: {_IMPORT_ERROR}")


class _EvaluationStrategy:
    name = "tiny"
    required_features = {"x", "target"}

    def __init__(self) -> None:
        self.module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(1.0)

    def trainable_module(self) -> nn.Module:
        return self.module

    def forward_loss(self, batch: TrainBatch, ctx=None) -> StepOutput:
        del ctx
        x = batch.tensors["x"].float()
        target = batch.tensors["target"].float()
        prediction = self.module(x)
        squared_error = (prediction - target).square()
        numerator = squared_error.sum()
        denominator = torch.tensor(float(squared_error.numel()))
        correct = (prediction.round() == target).float().sum()
        return StepOutput(
            loss=numerator / denominator,
            metrics={"accuracy": correct / denominator, "accuracy_denom": denominator},
            ratio_metrics={"acc": (correct.detach(), denominator)},
            loss_terms=(numerator, denominator),
        )

    def checkpoint_state_filter(self, state_dict):
        return dict(state_dict)


def _loader():
    return [
        {"x": torch.tensor([[1.0]]), "target": torch.tensor([[1.0]])},
        {"x": torch.tensor([[2.0]]), "target": torch.tensor([[2.0]])},
    ]


def test_checkpoint_and_in_memory_evaluation_share_results(tmp_path) -> None:
    _require_api()
    strategy = _EvaluationStrategy()
    evaluator = Evaluator()
    in_memory = evaluator.evaluate_in_memory(strategy, _loader(), torch.device("cpu"))

    optimizer = torch.optim.AdamW(strategy.trainable_module().parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, total_steps=2)
    manager = CheckpointManager(tmp_path, "eval")
    checkpoint = manager.save(1, strategy, optimizer, scheduler, {}, {})
    from_checkpoint = evaluator.evaluate_checkpoint(
        checkpoint,
        _EvaluationStrategy,
        lambda: _loader(),
        torch.device("cpu"),
    )

    assert from_checkpoint.keys() == in_memory.keys()
    for key in in_memory:
        assert from_checkpoint[key] == pytest.approx(in_memory[key], rel=1e-6)


def test_evaluation_rejects_empty_or_nonfinite_validation(tmp_path) -> None:
    _require_api()
    evaluator = Evaluator()
    with pytest.raises(ValueError, match="empty validation"):
        evaluator.evaluate_in_memory(_EvaluationStrategy(), [], torch.device("cpu"))

    class _NaNStrategy(_EvaluationStrategy):
        def forward_loss(self, batch, ctx=None):
            del batch, ctx
            nan = torch.tensor(float("nan"))
            return StepOutput(
                loss=nan,
                metrics={"accuracy": nan, "accuracy_denom": torch.tensor(1.0)},
            )

    with pytest.raises(ValueError, match="non-finite"):
        evaluator.evaluate_in_memory(
            _NaNStrategy(), _loader(), torch.device("cpu"), max_batches=1
        )
