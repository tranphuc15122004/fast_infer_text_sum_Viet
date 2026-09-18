from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils import checkpoint as checkpoint_utils
from transformers import Qwen3Config

try:
    from Finetuning.dflash_family_model import (
        FLEX_ATTENTION_AVAILABLE,
        OnlineDFlashModel,
        _sum_chunk_terms,
        compute_accept_len,
        create_dflash_block_mask,
        create_dflash_sdpa_mask,
    )
    from Finetuning.model import DFlashDraftModel
except ModuleNotFoundError as exc:  # Red phase: the objective is not ported yet.
    _OBJECTIVE_IMPORT_ERROR = exc
    FLEX_ATTENTION_AVAILABLE = False


def _require_objective_api() -> None:
    if "_OBJECTIVE_IMPORT_ERROR" in globals():
        pytest.fail(f"DFlash objective API is not implemented: {_OBJECTIVE_IMPORT_ERROR}")


def tiny_qwen3_config() -> Qwen3Config:
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
    return config


def tiny_online_dflash(
    *,
    loss_decay_gamma: float | None = None,
    objective_chunk_blocks: int = 128,
    loss_type: str = "dflash",
    attention_backend: str = "eager",
    num_anchors: int = 2,
    dpace_alpha: float = 0.5,
    block_size: int = 4,
) -> OnlineDFlashModel:
    _require_objective_api()
    torch.manual_seed(7)
    draft_model = DFlashDraftModel(tiny_qwen3_config())
    target_lm_head = nn.Linear(32, 97, bias=False)
    target_embed_tokens = nn.Embedding(97, 32)
    return OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_lm_head,
        target_embed_tokens=target_embed_tokens,
        mask_token_id=96,
        block_size=block_size,
        attention_backend=attention_backend,
        num_anchors=num_anchors,
        loss_decay_gamma=loss_decay_gamma,
        objective_chunk_blocks=objective_chunk_blocks,
        loss_type=loss_type,
        dpace_alpha=dpace_alpha,
    )


def test_compute_accept_len_respects_ragged_block_validity() -> None:
    _require_objective_api()
    predicted = torch.tensor([[[1, 2, 9, 4], [5, 0, 7, 8]]])
    target = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    valid = torch.tensor([[[True, True, True, False], [False, True, True, True]]])

    result = compute_accept_len(predicted, target, valid)

    assert torch.equal(result, torch.tensor([[2.0, 0.0]]))


def test_sdpa_mask_distinguishes_full_and_sliding_attention() -> None:
    _require_objective_api()
    anchors = torch.tensor([[2, 6]])
    keep = torch.tensor([[True, True]])
    full_mask = create_dflash_sdpa_mask(
        anchors,
        keep,
        S=8,
        block_size=4,
        device=torch.device("cpu"),
    )
    sliding_mask = create_dflash_sdpa_mask(
        anchors,
        keep,
        S=8,
        block_size=4,
        device=torch.device("cpu"),
        sliding_window=4,
    )

    full_row = full_mask[0, 0, 1].bool()
    sliding_row = sliding_mask[0, 0, 1].bool()
    assert full_row[0] and full_row[1]
    assert not full_row[2]
    assert full_row[8] and full_row[9] and full_row[10] and full_row[11]
    assert not full_row[12]
    assert sliding_row[0] and sliding_row[1]
    assert not sliding_row[2]
    assert sliding_row[8] and sliding_row[9]
    assert not sliding_row[10] and not sliding_row[11]
    assert not sliding_row[12]


@pytest.mark.skipif(not FLEX_ATTENTION_AVAILABLE, reason="flex_attention unavailable")
def test_flex_block_mask_distinguishes_full_and_sliding_attention() -> None:
    _require_objective_api()
    full_mask = create_dflash_block_mask(
        torch.tensor([[2, 6]]),
        torch.tensor([[True, True]]),
        S=8,
        block_size=4,
        device=torch.device("cpu"),
    )
    sliding_mask = create_dflash_block_mask(
        torch.tensor([[2, 6]]),
        torch.tensor([[True, True]]),
        S=8,
        block_size=4,
        device=torch.device("cpu"),
        sliding_window=4,
    )
    # CPU Flex BlockMask uses implementation-defined block metadata; its
    # callable mask modifier is the stable semantic contract.
    scalar = lambda value: torch.tensor(value)
    args = (scalar(0), scalar(0), scalar(1))
    assert full_mask.mask_mod(*args, scalar(0))
    assert full_mask.mask_mod(*args, scalar(1))
    assert not full_mask.mask_mod(*args, scalar(2))
    assert full_mask.mask_mod(*args, scalar(8))
    assert full_mask.mask_mod(*args, scalar(10))
    assert sliding_mask.mask_mod(*args, scalar(8))
    assert sliding_mask.mask_mod(*args, scalar(9))
    assert not sliding_mask.mask_mod(*args, scalar(10))


def test_chunk_reduction_checkpoints_only_grad_enabled_chunks(monkeypatch) -> None:
    _require_objective_api()
    calls = []
    real_checkpoint = checkpoint_utils.checkpoint

    def recording_checkpoint(function, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(checkpoint_utils, "checkpoint", recording_checkpoint)

    base = torch.arange(6.0, requires_grad=True)
    values = base.reshape(2, 3)

    def terms(chunk):
        return chunk.square().sum(), chunk.sum()

    chunked = _sum_chunk_terms(terms, (values,), chunk_size=1)
    assert calls == [False, False, False]
    chunked[0].backward()
    assert torch.equal(base.grad, 2 * base.detach())

    calls.clear()
    with torch.no_grad():
        _sum_chunk_terms(terms, (values,), chunk_size=1)
    assert calls == []

    calls.clear()
    unchunked = _sum_chunk_terms(terms, (values,), chunk_size=0)
    assert calls == []
    assert torch.isfinite(unchunked[0])


def test_anchor_sampling_requires_consecutive_supervised_tokens_and_caps_width() -> None:
    _require_objective_api()
    model = tiny_online_dflash(num_anchors=2)
    loss_mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0]], dtype=torch.float32)

    torch.manual_seed(11)
    anchors, keep = model._sample_anchor_positions(8, loss_mask, torch.device("cpu"))

    assert anchors.shape == (1, 2)
    assert keep.tolist() == [[True, True]]
    assert anchors.tolist()[0] == sorted(anchors.tolist()[0])
    assert set(anchors.tolist()[0]).issubset({1, 4, 5})

    with pytest.raises(ValueError, match="two consecutive supervised tokens"):
        model._sample_anchor_positions(
            8,
            torch.zeros(1, 8),
            torch.device("cpu"),
        )


def test_target_labels_are_same_position_and_weight_mask_excludes_anchor() -> None:
    _require_objective_api()
    model = tiny_online_dflash(num_anchors=1)
    input_ids = torch.tensor([[4, 5, 6, 7, 8]])
    loss_mask = torch.tensor([[0, 1, 1, 0, 1]], dtype=torch.float32)
    anchors = torch.tensor([[1]])
    block_keep = torch.tensor([[True]])

    target_ids, weight_mask = model._build_targets_and_weight_mask(
        input_ids, loss_mask, anchors, block_keep
    )

    assert torch.equal(target_ids, torch.tensor([[[5, 6, 7, 8]]]))
    assert torch.equal(weight_mask, torch.tensor([[[0.0, 1.0, 0.0, 1.0]]]))


def test_dflash_loss_excludes_anchor_and_keeps_target_modules_frozen() -> None:
    _require_objective_api()
    model = tiny_online_dflash(loss_decay_gamma=None)
    input_ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    hidden = torch.randn(1, 8, 64)
    loss_mask = torch.tensor([[0, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.float32)

    torch.manual_seed(5)
    loss, accuracy, metrics = model(input_ids, hidden, loss_mask)

    assert torch.isfinite(loss)
    assert torch.isfinite(accuracy)
    assert metrics["accuracy_denom"].item() > 0
    assert metrics["loss_terms"][0].requires_grad
    loss.backward()
    assert all(parameter.grad is None for parameter in model.embed_tokens.parameters())
    assert all(parameter.grad is None for parameter in model.lm_head.parameters())
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.draft_model.parameters()
        if parameter.requires_grad
    )


def test_positional_decay_changes_objective_denominator() -> None:
    _require_objective_api()
    model = tiny_online_dflash(loss_decay_gamma=1.0)
    with torch.no_grad():
        model.lm_head.weight.zero_()
    hidden = torch.zeros(1, 1, 4, 32)
    target_ids = torch.tensor([[[1, 2, 3, 4]]])
    weight_mask = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

    loss_num, loss_den, _, _ = model._dflash_objective_chunk_terms(
        hidden, target_ids, weight_mask
    )
    expected_den = 1.0 + math.exp(-1.0) + math.exp(-2.0)
    expected_num = math.log(97.0) * expected_den

    assert loss_den.item() == pytest.approx(expected_den)
    assert loss_num.item() == pytest.approx(expected_num, rel=1e-5)


def test_dpace_weight_modes_select_cumulative_and_continuation_values() -> None:
    _require_objective_api()
    model = tiny_online_dflash(dpace_alpha=0.0)
    prob = torch.tensor([[[0.5, 0.5, 0.5]]])
    mask = torch.ones_like(prob, dtype=torch.float32)
    binary = mask > 0

    cumulative = model._dpace_weight(
        prob, mask, binary, "dpace-cumulative-confidence-only"
    )
    combined = model._dpace_weight(prob, mask, binary, "dpace")
    continuation = model._dpace_weight(
        prob, mask, binary, "dpace-continuation-value-only"
    )

    assert torch.allclose(cumulative, torch.tensor([[[0.5, 0.25, 0.125]]]))
    assert torch.allclose(combined, torch.tensor([[[0.875, 0.375, 0.125]]]))
    assert torch.allclose(continuation, torch.tensor([[[1.75, 1.5, 1.0]]]))


@pytest.mark.parametrize(
    "loss_type",
    ["dpace", "dpace-cumulative-confidence-only", "dpace-continuation-value-only"],
)
def test_dpace_objective_modes_return_finite_loss(loss_type: str) -> None:
    _require_objective_api()
    model = tiny_online_dflash(loss_type=loss_type)
    input_ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    hidden = torch.randn(1, 8, 64)
    loss_mask = torch.tensor([[0, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.float32)

    torch.manual_seed(5)
    loss, _, metrics = model(input_ids, hidden, loss_mask)

    assert torch.isfinite(loss)
    assert metrics["loss_terms"][1].item() == pytest.approx(1.0)


def test_objective_chunking_preserves_loss_and_metrics() -> None:
    _require_objective_api()
    unchunked = tiny_online_dflash(objective_chunk_blocks=0)
    chunked = tiny_online_dflash(objective_chunk_blocks=1)
    chunked.load_state_dict(unchunked.state_dict())
    input_ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    hidden = torch.randn(1, 8, 64)
    loss_mask = torch.tensor([[0, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.float32)

    torch.manual_seed(17)
    unchunked_result = unchunked(input_ids, hidden, loss_mask)
    torch.manual_seed(17)
    chunked_result = chunked(input_ids, hidden, loss_mask)

    assert torch.allclose(unchunked_result[0], chunked_result[0])
    assert torch.allclose(unchunked_result[1], chunked_result[1])
    assert torch.allclose(
        unchunked_result[2]["accuracy_denom"], chunked_result[2]["accuracy_denom"]
    )


def test_invalid_loss_type_and_attention_backend_are_rejected() -> None:
    _require_objective_api()
    with pytest.raises(ValueError, match="loss_type"):
        tiny_online_dflash(loss_type="not-a-dflash-loss")
    with pytest.raises(ValueError, match="attention_backend"):
        tiny_online_dflash(attention_backend="not-an-attention-backend")


def test_block_size_one_is_rejected_before_empty_loss_denominator() -> None:
    _require_objective_api()

    with pytest.raises(ValueError, match="block_size.*2"):
        tiny_online_dflash(block_size=1)
