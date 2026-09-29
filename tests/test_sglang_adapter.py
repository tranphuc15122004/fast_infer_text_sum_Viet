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
    assert args[1:3] == ["-m", "sglang.launch_server"]
    assert "--speculative-algorithm" in args
    assert "--enable-metrics" in args
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


def test_sglang_target_only_reference_omits_speculative_server_flags() -> None:
    from Benchmark.infer_sglang_spec import build_server_args

    args = build_server_args(
        method="target_only",
        model="/models/Qwen3-4B",
        draft_model=None,
        port=39002,
        batch_size=1,
        tp_size=1,
        mem_fraction_static=0.75,
        attention_backend="flashinfer",
    )

    assert "--model-path" in args
    assert "--enable-metrics" in args
    assert "--speculative-algorithm" not in args
    assert "--speculative-draft-model-path" not in args


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
            "spec_accept_length": 2.25,
            "spec_accept_rate": 0.25,
            "spec_verify_ct": 4,
            "spec_num_correct_drafts": 5,
            "spec_num_proposed_drafts": 20,
            "spec_correct_drafts_histogram": [1, 1, 2, 0, 0],
        },
    }
    metrics = extract_response_metrics(payload, request_elapsed_ms=90.0)
    assert metrics["input_tokens"] == 120
    assert metrics["output_tokens"] == 12
    assert metrics["prefill_ms"] == 30.0
    assert metrics["decode_ms"] == 40.0
    assert metrics["queue_wait_ms"] == 2.0
    assert metrics["batch_wait_ms"] == 1.5
    assert metrics["avg_accept_length"] == 2.25
    assert metrics["acceptance_rate"] == 0.25
    assert metrics["acceptance_rate_percent"] == 25.0
    assert metrics["accepted_draft_tokens_per_step"] == 1.25
    assert metrics["verification_steps"] == 4
    assert metrics["draft_tokens_accepted"] == 5
    assert metrics["draft_tokens_proposed"] == 20
    assert metrics["draft_proposal_unit"] == "runtime_draft_candidate"
    assert metrics["acceptance_histogram"] == [1, 1, 2, 0, 0]
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
    assert metrics["strict_decode_active_ms"] is None
    assert metrics["strict_decode_phase_verified"] is False
    assert metrics["acceptance_histogram"] == [0, 7]


def test_sglang_sampling_request_explicitly_stops_on_target_eos() -> None:
    from Benchmark.infer_sglang_spec import build_sampling_params

    params = build_sampling_params(
        temperature=0.0,
        max_new_tokens=2048,
        stop_token_ids=[151645],
    )

    assert params == {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 2048,
        "stop_token_ids": [151645],
        "ignore_eos": False,
    }


def test_resolve_sglang_stop_ids_requires_and_normalizes_eos_ids() -> None:
    import pytest

    from Benchmark.infer_sglang_spec import resolve_stop_token_ids

    class Tokenizer:
        eos_token_id = [151645, 151643, 151645]

    assert resolve_stop_token_ids(Tokenizer()) == [151645, 151643]

    with pytest.raises(ValueError, match="EOS token"):
        resolve_stop_token_ids(type("MissingEOS", (), {"eos_token_id": None})())


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


def test_sglang_cache_can_be_disabled_for_the_paper_latency_profile() -> None:
    from Benchmark.infer_sglang_spec import build_server_args

    args = build_server_args(
        method="target_only",
        model="/models/Qwen3-4B",
        draft_model=None,
        port=39003,
        batch_size=1,
        tp_size=1,
        mem_fraction_static=0.75,
        disable_radix_cache=True,
    )
    assert "--disable-radix-cache" in args


def test_target_only_sidecar_load_requires_complete_matching_sample_identity(tmp_path) -> None:
    import json
    import pytest

    from Benchmark.infer_sglang_spec import load_target_only_reference

    path = tmp_path / "target_only.jsonl"
    row = {
        "contract_version": 2,
        "sample_id": "a",
        "dataset": "vietnews",
        "status": "success",
        "prompt_token_sha256": "prompt-a",
        "generation_config_sha256": "config",
        "hardware_fingerprint": "gpu",
        "gpu_count": 1,
        "tp_size": 1,
        "batch_size": 1,
        "concurrency": 1,
        "cache_policy": "disabled",
        "actual_input_tokens": 3,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    expected = {"a": {key: row[key] for key in (
        "prompt_token_sha256", "generation_config_sha256", "hardware_fingerprint",
        "gpu_count", "tp_size", "batch_size", "concurrency", "cache_policy", "actual_input_tokens",
    )}}

    loaded = load_target_only_reference(path, expected_sample_ids=["a"], expected_identity_by_id=expected)
    assert loaded["a"]["contract_version"] == 2

    expected["a"]["prompt_token_sha256"] = "different"
    with pytest.raises(ValueError, match="prompt_token_sha256"):
        load_target_only_reference(path, expected_sample_ids=["a"], expected_identity_by_id=expected)

    expected["a"]["prompt_token_sha256"] = "prompt-a"
    expected["a"]["actual_input_tokens"] = 4
    with pytest.raises(ValueError, match="actual_input_tokens"):
        load_target_only_reference(path, expected_sample_ids=["a"], expected_identity_by_id=expected)


def test_sglang_strict_decode_uses_direct_completion_latency() -> None:
    from Benchmark.infer_sglang_spec import extract_response_metrics

    payload = {
        "meta_info": {
            "completion_tokens": 5,
            "completion_latency": 0.05,
            # SGLang 0.5.20 computes this as (completion_tokens - 1) /
            # (finished_time - first_token_time); it is not itself a timer.
            "decode_throughput": 80.0,
        }
    }

    metrics = extract_response_metrics(payload, request_elapsed_ms=100.0)

    assert metrics["strict_decode_active_ms"] == 50.0
    assert metrics["strict_decode_token_count"] == 4
    assert metrics["strict_decode_phase_verified"] is True
    assert metrics["strict_decode_phase_definition"] == (
        "after_first_token_committed_to_final_token"
    )


def test_sglang_rate_alone_does_not_verify_a_decode_phase() -> None:
    from Benchmark.infer_sglang_spec import extract_response_metrics

    metrics = extract_response_metrics(
        {
            "meta_info": {
                "completion_tokens": 8,
                "decode_throughput": 4_152_592.6895027626,
                "prefill_finished_time": 1790699361.8854835,
                "request_finished_ts": 1790699361.9060307,
            }
        },
        request_elapsed_ms=33.909,
    )

    assert metrics["measurement_scope"] == "e2e_only"
    assert metrics["strict_decode_token_count"] == 7
    assert metrics["strict_decode_active_ms"] is None
    assert metrics["strict_decode_phase_verified"] is False


def test_sglang_streaming_parser_measures_first_output_to_final_output() -> None:
    import io

    import pytest

    from Benchmark.infer_sglang_spec import _parse_sglang_sse_response

    times = iter([10.0, 10.1, 10.25])
    lines = io.BytesIO(
        b'data: {"text":"a","output_ids":[11],"meta_info":{"completion_tokens":1,"finish_reason":null}}\n\n'
        b'data: {"text":"b","output_ids":[12],"meta_info":{"completion_tokens":2,"finish_reason":null}}\n\n'
        b'data: {"text":"c","output_ids":[13],"meta_info":{"completion_tokens":3,"finish_reason":{"type":"stop"}}}\n\n'
        b'data: [DONE]\n\n'
    )

    payload, timing = _parse_sglang_sse_response(lines, clock=lambda: next(times))

    assert payload["text"] == "abc"
    assert payload["output_ids"] == [11, 12, 13]
    assert payload["meta_info"]["completion_tokens"] == 3
    assert timing["decode_ms"] == 250.0
    assert timing["output_chunks"] == 3
    assert timing["decode_phase_verified"] is True


def test_sglang_streaming_parser_fails_closed_without_distinct_phase_events() -> None:
    import io

    from Benchmark.infer_sglang_spec import _parse_sglang_sse_response

    lines = io.BytesIO(
        b'data: {"text":"all","output_ids":[11,12],"meta_info":{"completion_tokens":2,"finish_reason":{"type":"stop"}}}\n\n'
    )

    _payload, timing = _parse_sglang_sse_response(lines, clock=lambda: 1.0)

    assert timing["decode_ms"] is None
    assert timing["decode_phase_verified"] is False


def test_sglang_api_stats_patch_exposes_direct_monotonic_duration() -> None:
    from Benchmark.sglang_timing_patch import patch_api_stats_class

    class FakeApiStats:
        first_token_time = 1.0
        finished_time = 1.25

        def convert_to_output_meta_info(
            self, scheduler_time_stats=None, completion_tokens=0
        ):
            return {"decode_throughput": 12.0}

    patch_api_stats_class(FakeApiStats)
    meta = FakeApiStats().convert_to_output_meta_info(completion_tokens=4)

    assert meta["completion_latency"] == 0.25
    assert meta["completion_latency_source"] == (
        "sglang_api_server_monotonic_first_token_to_finished"
    )


def test_sglang_stream_phase_requires_direct_server_timer_and_distinct_chunks() -> None:
    from Benchmark.infer_sglang_spec import extract_response_metrics

    payload = {
        "meta_info": {
            "completion_tokens": 4,
            "completion_latency": 0.2,
            "completion_latency_source": (
                "sglang_api_server_monotonic_first_token_to_finished"
            ),
        }
    }
    metrics = extract_response_metrics(
        payload,
        request_elapsed_ms=400.0,
        stream_timing={
            "decode_ms": 240.0,
            "output_chunks": 3,
            "decode_phase_verified": True,
        },
    )

    assert metrics["decode_ms"] == 200.0
    assert metrics["strict_decode_active_ms"] == 200.0
    assert metrics["strict_decode_phase_verified"] is True
    assert metrics["measurement_scope"] == "e2e_plus_decode"
    assert metrics["stream_decode_client_ms"] == 240.0

    metrics = extract_response_metrics(
        payload,
        request_elapsed_ms=400.0,
        stream_timing={
            "decode_ms": None,
            "output_chunks": 1,
            "decode_phase_verified": False,
        },
    )
    assert metrics["strict_decode_active_ms"] is None
    assert metrics["strict_decode_phase_verified"] is False
    assert metrics["measurement_scope"] == "e2e_only"


def test_sglang_strict_decode_is_unavailable_for_one_token_or_missing_latency() -> None:
    from Benchmark.infer_sglang_spec import extract_response_metrics

    one_token = extract_response_metrics(
        {"meta_info": {"completion_tokens": 1, "decode_throughput": 80.0}},
        request_elapsed_ms=10.0,
    )
    missing = extract_response_metrics(
        {"meta_info": {"completion_tokens": 5, "decode_throughput": 80.0}},
        request_elapsed_ms=10.0
    )

    assert one_token["strict_decode_active_ms"] is None
    assert one_token["strict_decode_token_count"] == 0
    assert one_token["strict_decode_phase_verified"] is False
    assert missing["strict_decode_active_ms"] is None


def test_paper_sglang_request_sends_the_exact_hashed_prompt_token_ids(monkeypatch):
    from types import SimpleNamespace

    from Benchmark import infer_sglang_spec as adapter

    class Tokenizer:
        name_or_path = "tokenizer"

        def __call__(self, _prompt, **_kwargs):
            class Row(list):
                def tolist(self):
                    return list(self)
            return SimpleNamespace(input_ids=[Row([11, 12, 13])])

    captured = {}
    monkeypatch.setattr(adapter, "_prepare_prompt", lambda prompt, _tok, _cap: f"formatted:{prompt}")

    def fake_http_stream(_url, body, timeout):
        captured.update(body=body, timeout=timeout)
        return (
            {"text": "summary", "meta_info": {"prompt_tokens": 3, "completion_tokens": 2}},
            {"decode_ms": 25.0, "output_chunks": 2, "decode_phase_verified": True},
        )

    monkeypatch.setattr(adapter, "_http_json_stream", fake_http_stream)
    args = SimpleNamespace(
        paper_speedup=True, temperature=0.0, max_new_tokens=8, seed=42,
        max_input_tokens=0, request_timeout=5.0,
    )
    sample, result = adapter._request_one(
        "http://localhost", {"id": "x", "prompt": "hello"}, args, Tokenizer(), [2]
    )

    assert sample["id"] == "x"
    assert captured["body"]["input_ids"] == [11, 12, 13]
    assert "text" not in captured["body"]
    assert captured["body"]["stream"] is True
    assert result["metrics"]["prompt_token_count_match"] is True
    assert result["metrics"]["prompt_token_sha256"] == adapter.token_ids_sha256([11, 12, 13])
