from types import SimpleNamespace

import torch

from Benchmark.dflash_fa4_attention import (
    dflash_fa4_attention_stats,
    install_dflash_fa4_attention,
)
from Benchmark.common.flashattn_runtime import AttentionDispatchTracker


class _DFlashAttention:
    def __init__(self, backend):
        self.config = SimpleNamespace(_attn_implementation=backend)
        self.scaling = 0.25
        self.sliding_window = None
        self.is_causal = False


def test_dflash_attention_dispatches_through_fa4_and_tracks_call_count():
    fa4_calls = []

    def original_sdpa(*args, **kwargs):
        raise AssertionError("DFlash FA4 call unexpectedly used SDPA")

    def fake_fa4(query, key, value, attention_mask, **kwargs):
        fa4_calls.append((query, key, value, attention_mask, kwargs))
        return query.transpose(1, 2)

    module = SimpleNamespace(
        Qwen3DFlashAttention=_DFlashAttention,
        ALL_ATTENTION_FUNCTIONS={"sdpa": original_sdpa},
    )
    install_dflash_fa4_attention(module, attention_fn=fake_fa4)

    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)
    mask = None
    attention = _DFlashAttention("flash_attention_4")
    tracker = AttentionDispatchTracker(
        SimpleNamespace(modules=lambda: []),
        SimpleNamespace(modules=lambda: [attention]),
    )
    with tracker.recording():
        output, weights = module.ALL_ATTENTION_FUNCTIONS["sdpa"](
            attention,
            query,
            key,
            value,
            mask,
            dropout=0.0,
            scaling=0.25,
            sliding_window=None,
        )

    assert output.shape == (1, 3, 2, 4)
    assert weights is None
    assert len(fa4_calls) == 1
    assert fa4_calls[0][3] is mask
    assert fa4_calls[0][4] == {
        "scaling": 0.25,
        "sliding_window": None,
        "is_causal": False,
    }
    assert dflash_fa4_attention_stats(module) == {
        "fa4_dispatch_calls": 1,
        "sdpa_fallback_calls": 0,
    }
    assert tracker.snapshot()["draft_attention_dispatch_calls"] == 1
    assert tracker.snapshot()["draft_fallback_attention_calls"] == 0


def test_dflash_attention_keeps_sdpa_for_non_fa4_backend():
    calls = []

    def original_sdpa(*args, **kwargs):
        calls.append((args, kwargs))
        return "sdpa-output", "sdpa-weights"

    def fake_fa4(*args, **kwargs):
        raise AssertionError("non-FA4 DFlash call must stay on SDPA")

    module = SimpleNamespace(
        Qwen3DFlashAttention=_DFlashAttention,
        ALL_ATTENTION_FUNCTIONS={"sdpa": original_sdpa},
    )
    install_dflash_fa4_attention(module, attention_fn=fake_fa4)

    result = module.ALL_ATTENTION_FUNCTIONS["sdpa"](
        _DFlashAttention("sdpa"), torch.empty(0), torch.empty(0), torch.empty(0), None
    )

    assert result == ("sdpa-output", "sdpa-weights")
    assert len(calls) == 1
    assert dflash_fa4_attention_stats(module) == {
        "fa4_dispatch_calls": 0,
        "sdpa_fallback_calls": 1,
    }
