from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config

try:
    from Finetuning.model import (
        DFlashDraftModel,
        build_target_layer_ids,
        extract_context_feature,
        normalize_draft_head_checkpoint_keys,
    )
except ModuleNotFoundError as exc:  # Red phase: make absence a test failure.
    _MODEL_IMPORT_ERROR = exc


def _require_model_api():
    if "_MODEL_IMPORT_ERROR" in globals():
        pytest.fail(f"DFlash model API is not implemented: {_MODEL_IMPORT_ERROR}")


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


def test_target_layer_ids_and_context_feature_contract():
    _require_model_api()
    assert build_target_layer_ids(28, 1) == [14]
    states = [torch.full((1, 5, 3), float(i)) for i in range(6)]
    result = extract_context_feature(states, [1, 3])
    assert result.shape == (1, 5, 6)
    assert torch.equal(result[..., :3], states[2])
    assert torch.equal(result[..., 3:], states[4])


def test_dflash_forward_shape_and_gradient():
    _require_model_api()
    torch.manual_seed(0)
    model = DFlashDraftModel(tiny_qwen3_config())
    target_hidden = torch.randn(1, 5, 64)
    noise = torch.randn(1, 8, 32)
    positions = torch.arange(8).view(1, -1)
    output = model(
        position_ids=positions,
        noise_embedding=noise,
        target_hidden=target_hidden,
    )
    assert output.shape == (1, 8, 32)
    output.square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_invalid_layer_layout_is_rejected():
    _require_model_api()
    config = tiny_qwen3_config()
    config.layer_types = ["full_attention"]
    with pytest.raises(ValueError, match="num_hidden_layers"):
        DFlashDraftModel(config)


def test_state_dict_contains_dflash_backbone_components():
    _require_model_api()
    state_keys = set(DFlashDraftModel(tiny_qwen3_config()).state_dict())
    assert "fc.weight" in state_keys
    assert "hidden_norm.weight" in state_keys
    assert "norm.weight" in state_keys
    assert "layers.0.self_attn.q_proj.weight" in state_keys
    assert "layers.1.mlp.gate_proj.weight" in state_keys


def test_checkpoint_hook_normalizes_legacy_head_keys_only():
    _require_model_api()
    normal = torch.ones(2)
    legacy = torch.zeros(2)
    state_dict = {
        "fc.weight": normal,
        "logit_head.embed_proj.weight": legacy,
    }

    normalize_draft_head_checkpoint_keys(
        None,
        state_dict,
        "",
        {},
        False,
        [],
        [],
        [],
    )

    assert torch.equal(state_dict["fc.weight"], normal)
    assert torch.equal(state_dict["embed_proj.weight"], legacy)
    assert "logit_head.embed_proj.weight" not in state_dict
