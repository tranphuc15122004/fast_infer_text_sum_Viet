from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_sglang_server_args_keep_official_speculative_contract() -> None:
    from Benchmark.infer_sglang_spec import build_server_args

    args = build_server_args(
        method="domino",
        model="/models/Qwen3-4B",
        draft_model="/models/Qwen3-4B-Domino",
        port=39001,
        batch_size=8,
        tp_size=2,
        mem_fraction_static=0.9,
    )
    assert "--speculative-algorithm" in args
    assert "DFLASH" in args
    batch_sizes = ["1", "2", "3", "4", "5", "6", "7", "8"]
    decode_index = args.index("--cuda-graph-bs-decode")
    assert args[decode_index + 1 : decode_index + 9] == batch_sizes
    assert args[args.index("--cuda-graph-max-bs-decode") + 1] == "8"
    assert "--cuda-graph-bs-prefill" not in args
    assert "--cuda-graph-max-bs-prefill" not in args
    assert "--cuda-graph-bs" not in args
    assert "--cuda-graph-max-bs" not in args
    assert args[args.index("--speculative-draft-model-path") + 1] == "/models/Qwen3-4B-Domino"
    assert args[args.index("--tp-size") + 1] == "2"


def test_sglang_payload_extracts_phase_and_acceptance_metrics() -> None:
    from Benchmark.infer_sglang_spec import extract_response_metrics

    payload = {
        "text": "bản tóm tắt",
        "meta_info": {
            "prompt_tokens": 120,
            "completion_tokens": 12,
            "prompt_latency": 0.03,
            "completion_latency": 0.04,
            "e2e_latency": 0.08,
            "queue_time": 0.002,
            "batch_wait_ms": 1.5,
            "spec_accept_length": 3.5,
            "spec_accept_rate": 0.25,
            "spec_verify_ct": 4,
        },
    }
    metrics = extract_response_metrics(payload, request_elapsed_ms=90.0)
    assert metrics["input_tokens"] == 120
    assert metrics["output_tokens"] == 12
    assert metrics["prefill_ms"] == 30.0
    assert metrics["decode_ms"] == 40.0
    assert metrics["queue_wait_ms"] == 2.0
    assert metrics["batch_wait_ms"] == 1.5
    assert metrics["avg_accept_length"] == 3.5
    assert metrics["acceptance_rate"] == 0.25
    assert metrics["verification_steps"] == 4
    assert metrics["e2e_ms"] == 90.0
    assert metrics["measurement_scope"] == "full_e2e"


def test_sglang_payload_downgrades_honestly_when_phase_timing_is_unavailable() -> None:
    from Benchmark.infer_sglang_spec import extract_response_metrics

    payload = {
        "text": "bản tóm tắt",
        "meta_info": {
            "prompt_tokens": 120,
            "completion_tokens": 8,
            "e2e_latency": 0.08,
            "spec_accept_length": 1.0,
            "spec_accept_histogram": [0, 7],
            "spec_verify_ct": 7,
        },
    }

    metrics = extract_response_metrics(payload, request_elapsed_ms=90.0)

    assert metrics["measurement_scope"] == "e2e_only"
    assert metrics["prefill_ms"] is None
    assert metrics["decode_ms"] is None
    assert metrics["acceptance_histogram"] == [0, 7]


def test_auto_batch_size_uses_b200_default_and_explicit_override(monkeypatch) -> None:
    from Benchmark.infer_sglang_spec import resolve_batch_size

    monkeypatch.delenv("LONG_BENCH_AUTO_BATCH_SIZE", raising=False)
    assert resolve_batch_size("auto", total_memory_gb=180.0) == 8
    monkeypatch.setenv("LONG_BENCH_AUTO_BATCH_SIZE", "5")
    assert resolve_batch_size("auto", total_memory_gb=180.0) == 5


def test_vanilla_fa_parser_accepts_blackwell_backend() -> None:
    from Benchmark.common.vanilla_inference import build_parser

    parser = build_parser("flash_attention_2", "test")
    args = parser.parse_args(
        ["--attention-backend", "flash_attention_4", "--output", "/tmp/out.jsonl"]
    )
    assert args.attention_backend == "flash_attention_4"


def test_vanilla_fa_resolves_fa2_to_fa4_on_blackwell() -> None:
    from Benchmark.common.vanilla_inference import _resolve_flash_attention_backend

    assert (
        _resolve_flash_attention_backend(
            "flash_attention_2",
            compute_capability=(10, 0),
            flash_attention_4_available=True,
        )
        == "flash_attention_4"
    )


def test_vanilla_fa_rejects_fa2_on_blackwell_without_fa4() -> None:
    import pytest

    from Benchmark.common.vanilla_inference import _resolve_flash_attention_backend

    with pytest.raises(RuntimeError, match="FA2.*unsupported"):
        _resolve_flash_attention_backend(
            "flash_attention_2",
            compute_capability=(10, 0),
            flash_attention_4_available=False,
        )


def test_baseline_config_uses_attention_backend_separate_from_sglang() -> None:
    from Benchmark.common.longbench_adapter import baseline_config_from_env

    config = baseline_config_from_env(
        "vanilla_fa",
        {
            "LONG_BENCH_ATTENTION_BACKEND": "flash_attention_4",
            "LONG_BENCH_SGLANG_ATTENTION_BACKEND": "triton",
        },
    )

    assert config["attention_backend"] == "flash_attention_4"


def test_vanilla_fa_preflight_uses_fa4_on_blackwell_without_importing_fa2(
    monkeypatch,
) -> None:
    from Benchmark.common import longbench_adapter

    probed_modules = []

    def module_importable(name):
        probed_modules.append(name)
        if name == "flash_attn":
            return False, "ImportError: undefined PyTorch C++ ABI symbol"
        if name == "flash_attn.cute":
            return True, None
        raise AssertionError(f"unexpected module probe: {name}")

    monkeypatch.setattr(
        longbench_adapter, "_local_requirement", lambda _path: (True, None)
    )
    monkeypatch.setattr(longbench_adapter, "_cuda_compute_capability", lambda: (10, 0))
    monkeypatch.setattr(longbench_adapter, "_module_importable", module_importable)

    result = longbench_adapter.preflight_baseline(
        "vanilla_fa",
        {"model": "/models/Qwen3-4B"},
        cuda_available=True,
    )

    assert result["status"] == "ready"
    assert result["requirements"]["flash_attention_4"]["available"] is True
    assert result["requirements"]["flash_attention_4"]["auto_selected_for_blackwell"]
    assert probed_modules == ["flash_attn.cute"]


def test_vanilla_fa_preflight_classifies_fa4_import_failure_as_dependency_error(
    monkeypatch,
) -> None:
    from Benchmark.common import longbench_adapter

    fa4_error = (
        "ImportError: /venv/site-packages/flash_attn_2_cuda.so: "
        "undefined symbol: materialize_cow_storage"
    )
    monkeypatch.setattr(
        longbench_adapter, "_local_requirement", lambda _path: (True, None)
    )
    monkeypatch.setattr(longbench_adapter, "_cuda_compute_capability", lambda: (10, 0))
    monkeypatch.setattr(
        longbench_adapter,
        "_module_importable",
        lambda name: (False, fa4_error) if name == "flash_attn.cute" else (True, None),
    )

    result = longbench_adapter.preflight_baseline(
        "vanilla_fa",
        {"model": "/models/Qwen3-4B"},
        cuda_available=True,
    )

    assert result["status"] == "missing_dependency"
    assert fa4_error in result["reason"]
