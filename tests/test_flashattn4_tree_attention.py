import sys
import types

import pytest
import torch

from Benchmark.common.flashattn4_tree_attention import (
    additive_mask_to_keep,
    pad_keep_mask_for_fa4,
)


def test_additive_tree_mask_becomes_boolean_keep_mask():
    floor = torch.finfo(torch.float32).min
    additive_mask = torch.tensor(
        [[[[0.0, floor, 0.0], [0.0, 0.0, floor]]]], dtype=torch.float32
    )

    keep_mask = additive_mask_to_keep(additive_mask)

    assert keep_mask.dtype == torch.bool
    assert keep_mask.tolist() == [[[True, False, True], [True, True, False]]]


def test_additive_tree_mask_rejects_wrong_rank():
    with pytest.raises(ValueError, match="4D"):
        additive_mask_to_keep(torch.zeros(1, 2, 3))


def test_additive_tree_mask_rejects_multiple_attention_heads():
    with pytest.raises(ValueError, match="head dimension must be 1"):
        additive_mask_to_keep(torch.zeros(1, 2, 3, 4))


def test_fa4_aux_mask_pads_edges_for_kernel_tiles():
    keep_mask = torch.ones((1, 2, 622), dtype=torch.bool)
    keep_mask[0, 0, -1] = False

    padded = pad_keep_mask_for_fa4(keep_mask)

    assert padded.dtype == torch.uint8
    assert padded.shape == (1, 256, 768)
    assert torch.equal(padded[:, :2, :622].bool(), keep_mask)
    assert not padded[:, 2:, :].any()
    assert not padded[:, :, 622:].any()


def test_flashattn4_attention_returns_output_tensor_from_fa4_tuple(monkeypatch):
    from Benchmark.common.flashattn4_tree_attention import flashattn4_attention

    cute_module = types.ModuleType("flash_attn.cute")
    expected = torch.randn(1, 3, 2, 8)
    calls = []

    def fake_flash_attn(*args, **kwargs):
        calls.append(kwargs)
        return expected, torch.zeros(1, 2, 3)

    cute_module.flash_attn_func = fake_flash_attn
    flash_module = types.ModuleType("flash_attn")
    flash_module.__path__ = []
    flash_module.cute = cute_module
    monkeypatch.setitem(sys.modules, "flash_attn", flash_module)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute_module)

    actual = flashattn4_attention(
        torch.zeros(1, 2, 3, 8),
        torch.zeros(1, 2, 4, 8),
        torch.zeros(1, 2, 4, 8),
        None,
        scaling=8**-0.5,
        is_causal=False,
    )

    assert actual is expected
    assert calls[0]["causal"] is False
