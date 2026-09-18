from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

try:
    from Finetuning.dflash_family_model import OnlineDFlashModel
    from Finetuning.model import DFlashDraftModel
    from Finetuning.strategy import (
        DFlashTrainStrategy,
        StepContext,
        StepOutput,
        TrainBatch,
    )
except ModuleNotFoundError as exc:  # Red phase: the strategy is not ported yet.
    _STRATEGY_IMPORT_ERROR = exc


def _require_strategy_api() -> None:
    if "_STRATEGY_IMPORT_ERROR" in globals():
        pytest.fail(f"DFlash strategy API is not implemented: {_STRATEGY_IMPORT_ERROR}")


def tiny_online_dflash(dtype: torch.dtype = torch.float32) -> OnlineDFlashModel:
    _require_strategy_api()
    config = Qwen3Config(
        architectures=["DFlashDraftModel"],
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_target_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        block_size=4,
        layer_types=["full_attention", "full_attention"],
        dflash_config={"target_layer_ids": [1, 2], "mask_token_id": 96},
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    draft_model = DFlashDraftModel(config)
    model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=nn.Linear(32, 97, bias=False),
        target_embed_tokens=nn.Embedding(97, 32),
        mask_token_id=96,
        block_size=4,
        attention_backend="eager",
        num_anchors=2,
        objective_chunk_blocks=1,
    )
    return model.to(dtype)


@dataclass
class SimpleBatch:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


def test_strategy_requires_all_dflash_features_and_filters_draft_keys() -> None:
    _require_strategy_api()
    strategy = DFlashTrainStrategy(tiny_online_dflash())
    with pytest.raises(ValueError, match="missing required features"):
        strategy.validate_batch(SimpleBatch({"input_ids": torch.ones(1, 4)}))

    filtered = strategy.checkpoint_state_filter(
        {
            "draft_model.fc.weight": torch.ones(2),
            "lm_head.weight": torch.ones(2),
        }
    )
    assert set(filtered) == {"fc.weight"}


def test_strategy_returns_step_output_with_detached_metrics_and_loss_terms() -> None:
    _require_strategy_api()
    model = tiny_online_dflash()
    strategy = DFlashTrainStrategy(model)
    batch = TrainBatch(
        tensors={
            "input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]),
            "hidden_states": torch.randn(1, 8, 64),
            "loss_mask": torch.tensor([[0, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.float32),
        }
    )

    torch.manual_seed(13)
    output = strategy.forward_loss(batch, StepContext(global_step=2, total_steps=10))

    assert isinstance(output, StepOutput)
    assert output.loss.ndim == 0
    assert output.metrics["accuracy"].requires_grad is False
    assert output.metrics["accuracy_denom"].requires_grad is False
    assert output.loss_terms is not None
    assert output.loss_terms[0].requires_grad
    assert output.loss_terms[1].requires_grad is False
    assert output.ratio_metrics["acc"][0].requires_grad is False
    assert output.ratio_metrics["acc"][1].requires_grad is False


def test_strategy_optimizer_boundary_exposes_dflash_wrapper() -> None:
    _require_strategy_api()
    model = tiny_online_dflash()
    strategy = DFlashTrainStrategy(model)

    assert strategy.trainable_module() is model
    assert strategy.name == "dflash"
    assert strategy.required_features == {"input_ids", "hidden_states", "loss_mask"}


@pytest.mark.parametrize("draft_dtype", [torch.float32, torch.bfloat16])
def test_strategy_casts_offline_hidden_states_to_draft_dtype(draft_dtype) -> None:
    _require_strategy_api()
    model = tiny_online_dflash(draft_dtype)
    strategy = DFlashTrainStrategy(model)
    batch = TrainBatch(
        tensors={
            "input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.long),
            "hidden_states": torch.randn(1, 8, 64, dtype=torch.float32),
            "loss_mask": torch.tensor([[0, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.float32),
        }
    )

    torch.manual_seed(19)
    output = strategy.forward_loss(batch)

    assert output.loss.is_floating_point()
    assert torch.isfinite(output.loss)
