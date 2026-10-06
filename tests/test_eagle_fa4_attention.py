from types import SimpleNamespace

import torch
from torch import nn

from Benchmark.eagle_fa4_attention import (
    install_eagle_draft_fa4_attention,
    install_eagle_fa4_attention,
)
from Benchmark.common.flashattn_runtime import AttentionDispatchTracker


def _linear_identity(size):
    layer = nn.Linear(size, size, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(size))
    return layer


class _FakeQwen3Attention:
    def __init__(self, backend):
        self.config = SimpleNamespace(_attn_implementation=backend)
        self.layer_idx = 0
        self.head_dim = 2
        self.num_key_value_groups = 1
        self.scaling = 0.5
        self.attention_dropout = 0.0
        self.sliding_window = None
        self.q_proj = _linear_identity(2)
        self.k_proj = _linear_identity(2)
        self.v_proj = _linear_identity(2)
        self.o_proj = _linear_identity(2)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.training = False

    def forward(self, *args, **kwargs):
        return "original-result"


def test_eagle_qwen3_dispatches_custom_tree_mask_through_fa4():
    calls = []

    def fake_fa4(query, key, value, attention_mask, **kwargs):
        calls.append((query, key, value, attention_mask, kwargs))
        return query.transpose(1, 2)

    module = SimpleNamespace(
        Qwen3Attention=_FakeQwen3Attention,
        apply_rotary_pos_emb=lambda q, k, cos, sin: (q, k),
    )
    install_eagle_fa4_attention(module, attention_fn=fake_fa4)
    attention = _FakeQwen3Attention("flash_attention_4")
    tracker = AttentionDispatchTracker(
        SimpleNamespace(modules=lambda: [attention]),
        SimpleNamespace(modules=lambda: []),
    )
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    tree_mask = torch.zeros((1, 1, 2, 2))

    with tracker.recording():
        output, weights, cache = attention.forward(hidden, (None, None), tree_mask)

    assert output.shape == hidden.shape
    assert weights is None
    assert cache is None
    assert len(calls) == 1
    assert calls[0][3] is tree_mask
    assert calls[0][4] == {"scaling": 0.5, "sliding_window": None}
    assert tracker.snapshot()["target_attention_dispatch_calls"] == 1
    assert tracker.snapshot()["target_fallback_attention_calls"] == 0


class _FakeKVCache:
    def __init__(self, length):
        self.shape = (1, 1, length, 2)

    def cat(self, tensor, dim=2):
        return tensor


def test_eagle_qwen3_uses_native_causal_fa4_for_prefill_and_single_token_decode():
    calls = []

    class _FakePrefillAttention(_FakeQwen3Attention):
        def forward(self, *args, **kwargs):
            return "original-result"

    def fake_fa4(query, key, value, attention_mask, **kwargs):
        calls.append((attention_mask, kwargs))
        return query.transpose(1, 2)

    module = SimpleNamespace(
        Qwen3Attention=_FakePrefillAttention,
        apply_rotary_pos_emb=lambda q, k, cos, sin: (q, k),
    )
    install_eagle_fa4_attention(module, attention_fn=fake_fa4)
    attention = _FakePrefillAttention("flash_attention_4")
    prefill_cache = (_FakeKVCache(0), _FakeKVCache(0))
    prefill_mask = torch.zeros((1, 1, 2, 2))
    attention.forward(
        torch.ones((1, 2, 2)), (None, None), prefill_mask, past_key_value=prefill_cache
    )

    decode_cache = (_FakeKVCache(2), _FakeKVCache(2))
    decode_mask = torch.zeros((1, 1, 1, 3))
    attention.forward(
        torch.ones((1, 1, 2)), (None, None), decode_mask, past_key_value=decode_cache
    )

    assert [call[0] for call in calls] == [None, None]


def test_eagle_qwen3_patch_preserves_non_fa4_attention_path():
    calls = []

    def original(self, *args, **kwargs):
        calls.append("original")
        return "original-result"

    def fake_fa4(*args, **kwargs):
        calls.append("fa4")
        return torch.empty((1, 1, 1, 1))

    module = SimpleNamespace(
        Qwen3Attention=_FakeQwen3Attention,
        apply_rotary_pos_emb=lambda q, k, cos, sin: (q, k),
    )
    module.Qwen3Attention.forward = original
    install_eagle_fa4_attention(module, attention_fn=fake_fa4)
    attention = _FakeQwen3Attention("eager")

    result = attention.forward(torch.zeros((1, 1, 2)), (None, None), None)

    assert result == "original-result"
    assert calls == ["original"]


class _FakeDraftAttention:
    def __init__(self, backend):
        self.config = SimpleNamespace(
            _attn_implementation=backend,
            pretraining_tp=1,
        )
        self.hidden_size = 2
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = 2
        self.max_position_embeddings = 8
        self.q_proj = nn.Linear(4, 2, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(2, 2, bias=False)
        self.rotary_emb = lambda value, seq_len: (None, None)

    def forward(self, *args, **kwargs):
        return "original-result"


def test_eagle_draft_attention_uses_fa4_for_its_causal_or_tree_mask():
    calls = []

    def fake_fa4(query, key, value, attention_mask, **kwargs):
        calls.append((query, key, value, attention_mask, kwargs))
        return query.transpose(1, 2)

    module = SimpleNamespace(
        LlamaAttention=_FakeDraftAttention,
        apply_rotary_pos_emb=lambda q, k, cos, sin, position_ids: (q, k),
    )
    install_eagle_draft_fa4_attention(module, attention_fn=fake_fa4)
    attention = _FakeDraftAttention("flash_attention_4")
    tracker = AttentionDispatchTracker(
        SimpleNamespace(modules=lambda: []),
        SimpleNamespace(modules=lambda: [attention]),
    )
    hidden_states = torch.ones((1, 2, 4))
    tree_mask = torch.zeros((1, 1, 2, 2))

    with tracker.recording():
        output, weights, cache = attention.forward(
            hidden_states,
            attention_mask=tree_mask,
            position_ids=torch.tensor([[0, 1]]),
            use_cache=True,
        )

    assert output.shape == (1, 2, 2)
    assert weights is None
    assert cache is not None and len(cache) == 2
    assert len(calls) == 1
    assert calls[0][3] is tree_mask
    assert calls[0][4] == {"scaling": 2**-0.5, "sliding_window": None}
    assert tracker.snapshot()["draft_attention_dispatch_calls"] == 1
    assert tracker.snapshot()["draft_fallback_attention_calls"] == 0
