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
    assert args[args.index("--cuda-graph-bs") + 1 : args.index("--cuda-graph-max-bs")] == ["1", "2", "3", "4", "5", "6", "7", "8"]
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
    assert metrics["verification_steps"] == 4
    assert metrics["e2e_ms"] == 90.0


def test_auto_batch_size_uses_b200_default_and_explicit_override(monkeypatch) -> None:
    from Benchmark.infer_sglang_spec import resolve_batch_size

    monkeypatch.delenv("LONG_BENCH_AUTO_BATCH_SIZE", raising=False)
    assert resolve_batch_size("auto", total_memory_gb=180.0) == 8
    monkeypatch.setenv("LONG_BENCH_AUTO_BATCH_SIZE", "5")
    assert resolve_batch_size("auto", total_memory_gb=180.0) == 5
