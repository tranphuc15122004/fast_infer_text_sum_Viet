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


def test_eagle_load_options_do_not_request_transformers_loading_info_tuple():
    from Benchmark.eagle_compat import eagle_model_load_options

    options = eagle_model_load_options()

    assert options["output_loading_info"] is False
    assert options["low_cpu_mem_usage"] is False
    assert "device_map" not in options


def test_eagle_uses_bfloat16_on_hopper_and_fp16_on_older_devices():
    from types import SimpleNamespace

    from Benchmark.eagle_compat import eagle_target_dtype

    class Cuda:
        def __init__(self, capability, bf16):
            self.capability = capability
            self.bf16 = bf16

        def get_device_capability(self, _device_index):
            return self.capability

        def is_bf16_supported(self):
            return self.bf16

    torch_stub = SimpleNamespace(
        cuda=Cuda((9, 0), True), bfloat16="bf16", float16="fp16"
    )
    assert eagle_target_dtype(torch_stub) == "bf16"

    torch_stub.cuda = Cuda((7, 5), False)
    assert eagle_target_dtype(torch_stub) == "fp16"


def test_eagle_rotary_diagnostics_supports_transformers5_model_level_rope():
    from types import SimpleNamespace

    from Benchmark.eagle_compat import eagle_rotary_diagnostics

    rotary = SimpleNamespace(inv_freq=[0.5, 0.25], rope_type="default")
    core = SimpleNamespace(rotary_emb=rotary)
    attention = SimpleNamespace(_uses_llama3_rope=False)

    report = eagle_rotary_diagnostics(core, attention)

    assert report["rotary_impl"] == "SimpleNamespace"
    assert report["rotary_type"] == "default"
    assert report["inv_freq_head"] == [0.5, 0.25]


def test_eagle_acceptance_uses_all_tree_nodes_as_proposals():
    from Benchmark.eagle_compat import normalize_eagle_acceptance_metrics

    metrics = normalize_eagle_acceptance_metrics(
        [1, 2, 4], draft_tokens_accepted=4, draft_tokens_per_step=31
    )

    assert metrics["verification_steps"] == 3
    assert metrics["avg_accept_length"] == 2.3333
    assert metrics["accepted_draft_tokens_per_step"] == 1.3333
    assert metrics["draft_tokens_accepted"] == 4
    assert metrics["draft_tokens_proposed"] == 93
    assert metrics["draft_proposal_unit"] == "draft_tree_node"
    assert metrics["acceptance_rate"] == round(4 / 93, 6)
    assert metrics["acceptance_rate_percent"] == round(400 / 93, 4)
    assert metrics["rejected_draft_ratio"] == round(1 - 4 / 93, 6)



def test_shared_acceptance_metrics_keep_small_eagle_rate_visible():
    from Benchmark.common.speculative_metrics import normalize_speculative_acceptance

    metrics = normalize_speculative_acceptance(
        verification_steps=2048,
        draft_tokens_accepted=1,
        draft_tokens_proposed=5515,
    )

    assert metrics["avg_accept_length"] == 1.0005
    assert metrics["acceptance_rate"] == round(1 / 5515, 6)
    assert metrics["acceptance_rate_percent"] == round(100 / 5515, 4)
    assert metrics["accepted_draft_tokens_per_step"] == round(1 / 2048, 4)


def test_eagle_checkpoint_reload_copies_safetensors_into_custom_target(tmp_path):
    import torch
    from safetensors.torch import save_file

    from Benchmark.eagle_compat import reload_eagle_target_weights

    source = torch.nn.Linear(4, 3, bias=False)
    target = torch.nn.Linear(4, 3, bias=False)
    checkpoint_dir = tmp_path / "target"
    checkpoint_dir.mkdir()
    save_file(
        {"weight": source.weight.detach().clone()},
        str(checkpoint_dir / "model.safetensors"),
    )

    report = reload_eagle_target_weights(target, checkpoint_dir)

    assert report["missing_parameter_count"] == 0
    assert report["loaded_tensor_count"] == 1
    assert torch.equal(target.weight, source.weight)


def test_eagle_rope_repair_recomputes_default_frequency_buffer():
    from types import SimpleNamespace

    import torch

    from Benchmark.eagle_compat import repair_eagle_rotary_embeddings

    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
                head_dim=8,
                hidden_size=8,
                num_attention_heads=1,
                max_position_embeddings=64,
            )
            self.rope_type = "default"
            self.register_buffer("inv_freq", torch.zeros(4), persistent=False)
            self.original_inv_freq = self.inv_freq
            self.attention_scaling = 1.0

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.rotary_emb = Rotary()

    core = Core()
    repair_eagle_rotary_embeddings(core)

    assert torch.allclose(
        core.rotary_emb.inv_freq,
        torch.tensor([1.0, 0.1, 0.01, 0.001]),
    )


def test_eagle_timing_pair_remains_valid_when_greedy_tokens_differ():
    from Benchmark.eagle_compat import eagle_paired_speedup_fields

    matched = eagle_paired_speedup_fields(
        [5, 8, 13], [5, 8, 13], eagle_time_s=2.0, naive_time_s=3.0
    )
    mismatched = eagle_paired_speedup_fields(
        [5, 8, 13], [5, 8, 14], eagle_time_s=2.0, naive_time_s=3.0
    )

    assert matched["target_greedy_match"] is True
    assert matched["paired_speedup_valid"] is True
    assert matched["paired_speedup"] == 1.5
    assert mismatched["target_greedy_match"] is False
    assert mismatched["paired_speedup_valid"] is True
    assert mismatched["paired_speedup_invalid_reason"] is None
    assert mismatched["paired_first_mismatch_index"] == 2
    assert mismatched["paired_eagle_mismatch_token"] == 13
    assert mismatched["paired_naive_mismatch_token"] == 14
    assert mismatched["paired_eagle_output_tokens"] == 3
    assert mismatched["paired_naive_output_tokens"] == 3


def test_eagle_output_truncates_tokens_after_first_eos_and_repairs_final_trace():
    import torch

    from Benchmark.eagle_compat import truncate_eagle_generation_at_stop

    output = torch.tensor([[10, 11, 20, 21, 99, 99, 22]])
    truncated, trace, removed = truncate_eagle_generation_at_stop(
        output,
        prompt_length=2,
        acceptance_lengths=[2, 1, 2],
        stop_token_ids=[99],
    )

    assert truncated.tolist() == [[10, 11, 20, 21, 99]]
    assert trace == [1, 1, 1]
    assert removed == 2
    assert sum(trace) == truncated.shape[1] - 2


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


def test_versioned_eagle_timing_patch_is_idempotent():
    from Benchmark.eagle_timing_patch import patch_eagle_sources

    root = Path(__file__).resolve().parents[1]
    utils_source = (root / "externals/EAGLE/eagle/model/utils.py").read_text()
    model_source = (root / "externals/EAGLE/eagle/model/ea_model.py").read_text()

    patched_utils, patched_model = patch_eagle_sources(utils_source, model_source)
    patched_again_utils, patched_again_model = patch_eagle_sources(patched_utils, patched_model)

    assert patched_again_utils == patched_utils
    assert patched_again_model == patched_model
    assert patched_utils.count("_fast_infer_first_token_commit_time") == 1
    assert patched_model.count("strict_decode_active_ms") == 2
    assert patched_model.count("strict_decode_token_count") == 2
    assert "self._fast_infer_first_token_commit_time = time.perf_counter()" in patched_model
    compile(patched_utils, "utils.py", "exec")
    compile(patched_model, "ea_model.py", "exec")


def test_versioned_dflash_timing_patch_is_idempotent():
    import hashlib

    from Benchmark.dflash_timing_patch import (
        PATCH_VERSION,
        SOURCE_SHA256,
        patch_dflash_source,
    )

    source = (ROOT / "externals/dflash/dflash/model.py").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA256
    patched = patch_dflash_source(source)
    assert PATCH_VERSION in patched
    assert patch_dflash_source(patched) == patched
    assert "strict_decode_active_ms=total_decode_time * 1e3" in patched
    compile(patched, "externals/dflash/dflash/model.py", "exec")
