from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_native_defaults_restore_eagle_ar_and_domino_graph():
    from Benchmark.native_flashattn import NativeInferenceConfig

    config = NativeInferenceConfig()
    assert config.eagle_tree() == {"total_token": 17, "depth": 16, "top_k": 1}
    assert config.domino_cuda_graph is True
    assert config.phase_timing_mode == "separate"
    assert config.strict_greedy_parity is False
    assert config.require_speedup is False


@pytest.mark.parametrize("kwargs", [
    {"eagle_total_token": 0}, {"eagle_depth": -1}, {"eagle_top_k": 0},
    {"eagle_total_token": 100, "eagle_depth": 1, "eagle_top_k": 1},
    {"phase_timing_mode": "unknown"}, {"dspark_confidence_threshold": 1.1},
])
def test_invalid_native_configuration_fails_before_loading_models(kwargs):
    from Benchmark.native_flashattn import NativeInferenceConfig

    with pytest.raises(ValueError):
        NativeInferenceConfig(**kwargs)


def test_fa4_pin_matches_the_stack_already_verified_on_b200():
    from Benchmark.native_flashattn import FA4_VERSION, validate_fa4_version

    assert FA4_VERSION == "4.0.0b32"
    validate_fa4_version({"flash-attn-4": FA4_VERSION})
    with pytest.raises(ValueError, match="4.0.0b32"):
        validate_fa4_version({"flash-attn-4": "4.0.0b19"})


def test_source_manifest_detects_native_inference_changes_for_safe_resume(tmp_path):
    from Benchmark.native_flashattn import native_source_manifest

    paths = ("src/Benchmark/native_flashattn.py", "src/Benchmark/common/flashattn4_tree_attention.py",
             "src/Benchmark/common/flashattn_runtime.py", "src/Benchmark/common/vanilla_inference.py",
             "src/Benchmark/dflash_compat.py",
             "src/Benchmark/dflash_fa4_attention.py", "externals/dflash/dflash/model.py")
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("version 1", encoding="utf-8")
    original = native_source_manifest(tmp_path, ("vanilla_hf", "dflash"))
    assert set(original) == set(paths)
    (tmp_path / paths[-1]).write_text("changed native decoder", encoding="utf-8")
    updated = native_source_manifest(tmp_path, ("vanilla_hf", "dflash"))
    assert updated[paths[-1]] != original[paths[-1]]
    assert updated[paths[0]] == original[paths[0]]


def test_source_manifest_resolves_the_actual_vendored_entrypoints():
    from pathlib import Path
    from Benchmark.native_flashattn import native_source_manifest

    root = Path(__file__).resolve().parents[1]
    if not (root / "externals/DeepSpec").is_dir():
        pytest.skip("vendored externals unavailable in this checkout")
    manifest = native_source_manifest(root, ("vanilla_hf", "eagle3", "dflash", "domino", "dspark"))
    assert len(manifest) >= 20
    assert all(len(value) == 64 for value in manifest.values())


def test_dflash_timing_scope_restores_native_timer_even_after_failure():
    from Benchmark.native_flashattn import native_timing_scope

    original_calls = []
    def original():
        original_calls.append(True)
        return 42.0
    module = SimpleNamespace(_cuda_time=original)
    context = {"dflash_model_module": module}
    with pytest.raises(RuntimeError):
        with native_timing_scope("dflash", context, synchronized=False) as stats:
            assert isinstance(module._cuda_time(), float)
            assert original_calls == []
            assert stats.skipped_synchronizations == 1
            raise RuntimeError("generation failed")
    assert module._cuda_time is original
    with native_timing_scope("dflash", context, synchronized=True):
        assert module._cuda_time() == 42.0
    assert original_calls == [True]


def test_eagle_timing_proxy_does_not_change_torch_or_tensor_operations():
    from Benchmark.native_flashattn import native_timing_scope

    calls = []
    real_cuda = SimpleNamespace(synchronize=lambda *a, **k: calls.append((a, k)))
    tensor_op = object()
    real_torch = SimpleNamespace(cuda=real_cuda, argmax=tensor_op)
    module = SimpleNamespace(torch=real_torch)
    with native_timing_scope("eagle3", {"eagle_model_module": module}, synchronized=False) as stats:
        assert module.torch.argmax is tensor_op
        module.torch.cuda.synchronize("cuda:0")
        assert stats.skipped_synchronizations == 1
        assert calls == []
        real_torch.cuda.synchronize("cuda:0")
        assert len(calls) == 1
    assert module.torch is real_torch
    assert real_torch.cuda is real_cuda


def test_domino_timing_scope_preserves_device_argument_and_restores_clock():
    from Benchmark.native_flashattn import native_timing_scope

    original = lambda device=None: 123.0
    module = SimpleNamespace(cuda_time=original)
    with native_timing_scope("domino", {"domino_model_module": module}, synchronized=False):
        assert isinstance(module.cuda_time("cuda:0"), float)
    assert module.cuda_time is original


def test_measured_payload_discards_unsynchronized_phase_clocks_but_keeps_acceptance():
    from Benchmark.native_flashattn import prepare_measured_payload

    payload = {
        "elapsed_ms": 120.0, "ttft_ms": 0.03, "output_ids": [5, 6],
        "draft_tokens_accepted": 1, "draft_tokens_proposed": 15,
        "verification_steps": 1, "acceptance_lengths": [2],
        "phases": {"draft_latency_ms": 0.04, "verification_latency_ms": 0.01,
                   "draft_tokens_accepted": 1, "stop_tokens_trimmed": 0},
    }
    measured = prepare_measured_payload(payload, first_forward_ms=10.0, mode="separate")
    assert measured["elapsed_ms"] == 120.0
    assert measured["ttft_ms"] == 10.0
    assert measured["draft_tokens_accepted"] == 1
    assert measured["phases"]["draft_latency_ms"] is None
    assert measured["phases"]["draft_tokens_accepted"] == 1
    assert payload["ttft_ms"] == 0.03


def test_profile_phases_are_attached_only_when_output_and_acceptance_match():
    from Benchmark.native_flashattn import attach_profile_phases

    measured = {"elapsed_ms": 100, "ttft_ms": 10, "output_ids": [5, 6],
                "draft_tokens_accepted": 1, "draft_tokens_proposed": 15,
                "verification_steps": 1, "acceptance_lengths": [2], "phases": {}}
    profile = {**measured, "elapsed_ms": 180, "ttft_ms": 20,
               "phases": {"draft_latency_ms": 30, "verification_latency_ms": 100}}
    metadata = attach_profile_phases(measured, profile)
    assert metadata["output_and_acceptance_match"] is True
    assert metadata["profiling_e2e_ms"] == 180
    assert measured["elapsed_ms"] == 100
    assert measured["ttft_ms"] == 10
    assert measured["phases"]["draft_latency_ms"] == 30

    other = {**profile, "output_ids": [5, 7]}
    measured["phases"] = {}
    metadata = attach_profile_phases(measured, other)
    assert metadata["output_and_acceptance_match"] is False
    assert measured["phases"] == {}
    other = {**profile, "draft_tokens_accepted": 0}
    assert attach_profile_phases(measured, other)["output_and_acceptance_match"] is False


@pytest.mark.parametrize("mode,method", [
    ("off", "dflash"), ("inline", "eagle3"),
    ("separate", "vanilla_hf"), ("separate", "domino"), ("separate", "dspark"),
])
def test_phase_collection_does_not_run_an_unused_profiling_generation(mode, method):
    from Benchmark.native_flashattn import NativeInferenceConfig, collect_native_phase_profile

    def unexpected_call():
        raise AssertionError("should not run another generation")
    measured = {"elapsed_ms": 100, "ttft_ms": 10}
    metadata = collect_native_phase_profile(
        NativeInferenceConfig(phase_timing_mode=mode), method, measured, unexpected_call,
    )
    assert metadata["source"] in {"disabled", "inline_generation", "unavailable_native_phase_timers"}
    assert measured == {"elapsed_ms": 100, "ttft_ms": 10}


def test_separate_profiling_never_overwrites_primary_latency_or_acceptance():
    from Benchmark.native_flashattn import NativeInferenceConfig, collect_native_phase_profile

    measured = {"elapsed_ms": 100, "ttft_ms": 10, "output_ids": [5, 6],
                "draft_tokens_accepted": 1, "draft_tokens_proposed": 15,
                "verification_steps": 1, "acceptance_lengths": [2], "phases": {}}
    profile = {**measured, "elapsed_ms": 180, "ttft_ms": 20,
               "phases": {"draft_latency_ms": 30, "verification_latency_ms": 100}}
    calls = []
    def profile_call():
        calls.append(True)
        return profile
    metadata = collect_native_phase_profile(NativeInferenceConfig(), "dflash", measured, profile_call)
    assert calls == [True]
    assert metadata["source"] == "separate_generation"
    assert measured["elapsed_ms"] == 100
    assert measured["ttft_ms"] == 10
    assert measured["draft_tokens_accepted"] == 1
    assert measured["phases"] == {"draft_latency_ms": 30, "verification_latency_ms": 100}


@pytest.mark.parametrize("shift_label,steps", [(False, 13), (True, 14)])
def test_domino_graph_uses_the_native_checkpoint_dimensions(shift_label, steps):
    from Benchmark.native_flashattn import build_domino_graph_runner

    draft = SimpleNamespace(block_size=16, pure_draft_prefix_len=2,
        prefix_gru=SimpleNamespace(hidden_size=512),
        config=SimpleNamespace(dflash_config={"shift_label": shift_label}))
    target = SimpleNamespace(lm_head=SimpleNamespace(weight=SimpleNamespace(shape=(1000, 128))))
    seen = {}
    def factory(**kwargs):
        seen.update(kwargs)
        return "native-graph"
    graph = build_domino_graph_runner(draft, target, "cuda:0", factory=factory)
    assert graph == "native-graph"
    assert seen == {"draft_model": draft, "target_model": target, "batch_size": 1,
        "steps": steps, "hidden_dim": 128, "gru_hidden_dim": 512,
        "vocab_size": 1000, "prefix_token_count": 3, "device": "cuda:0"}


def test_benchmark_gates_do_not_relabel_measured_slowdown_as_execution_failure():
    from Benchmark.native_flashattn import NativeInferenceConfig, native_run_status

    checks = dict(runtime_pass=True, execution_complete=True, failure_count=0,
                  quality_pass=True, exact_match_all=False, speedup_all_over_one=False,
                  schema_valid=True)
    assert native_run_status(NativeInferenceConfig(), **checks) == "success"
    assert native_run_status(NativeInferenceConfig(strict_greedy_parity=True), **checks) == "greedy_parity_failure"
    assert native_run_status(NativeInferenceConfig(require_speedup=True), **checks) == "speedup_not_above_one"
    assert native_run_status(NativeInferenceConfig(), **{**checks, "quality_pass": False}) == "quality_failure"
    assert native_run_status(NativeInferenceConfig(), **{**checks, "runtime_pass": False}) == "runtime_failure"
