from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_eagle_compat_installs_missing_loss_kwargs(monkeypatch):
    import transformers.utils as transformers_utils

    monkeypatch.delattr(transformers_utils, "LossKwargs", raising=False)

    from Benchmark.eagle_compat import install_eagle_transformers_compat

    installed = install_eagle_transformers_compat()

    assert installed is True
    assert hasattr(transformers_utils, "LossKwargs")
    assert "labels" in getattr(transformers_utils.LossKwargs, "__annotations__", {})


def test_eagle_compat_does_not_replace_existing_loss_kwargs(monkeypatch):
    import transformers.utils as transformers_utils

    class ExistingLossKwargs:
        pass

    monkeypatch.setattr(
        transformers_utils,
        "LossKwargs",
        ExistingLossKwargs,
        raising=False,
    )

    from Benchmark.eagle_compat import install_eagle_transformers_compat

    installed = install_eagle_transformers_compat()

    assert installed is False
    assert transformers_utils.LossKwargs is ExistingLossKwargs


def test_eagle_compat_restores_default_rope_initializer(monkeypatch):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    monkeypatch.delitem(ROPE_INIT_FUNCTIONS, "default", raising=False)

    from Benchmark.eagle_compat import install_eagle_transformers_compat

    install_eagle_transformers_compat()

    initializer = ROPE_INIT_FUNCTIONS["default"]
    assert callable(initializer)


def test_eagle_draft_attention_accepts_transformers5_llama3_rope():
    """The vendored EAGLE draft block must accept Transformers 5 RoPE keys."""

    import torch

    sys.path.insert(0, str(ROOT / "externals" / "EAGLE"))
    from eagle.model.cnets import LlamaAttention

    class Config:
        hidden_size = 8
        num_attention_heads = 2
        num_key_value_heads = 2
        max_position_embeddings = 16
        pretraining_tp = 1
        rope_scaling = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8,
        }
        rope_parameters = {**rope_scaling, "rope_theta": 500000.0}
        rope_theta = 500000.0

        @staticmethod
        def standardize_rope_params():
            return None

    attention = LlamaAttention(Config())
    assert type(attention.rotary_emb).__name__ == "Llama3RotaryEmbedding"
    cos, sin = attention.rotary_emb(torch.zeros(1, 2, 2, 4), seq_len=2)
    assert cos.shape == (1, 1, 2, 4)
    assert sin.shape == (1, 1, 2, 4)
