from __future__ import annotations

import math

import pytest


def _row(sample_id, *, wall, decode, tokens, text, prompt="p", status="success"):
    return {
        "contract_version": 2,
        "dataset": "vietnews",
        "sample_id": sample_id,
        "target_revision": "model",
        "tokenizer_revision": "tokenizer",
        "runtime_config_sha256": "runtime",
        "actual_input_tokens": 16,
        "method": "vanilla_fa",
        "status": status,
        "prompt_token_sha256": prompt,
        "pairing_config_sha256": "cfg",
        "generation_config_sha256": "cfg",
        "hardware_fingerprint": "gpu",
        "gpu_count": 1,
        "tp_size": 1,
        "batch_size": 1,
        "concurrency": 1,
        "cache_policy": "disabled",
        "request_wall_ms": wall,
        "native_elapsed_ms": wall,
        "native_timing_scope": "request_wall",
        "measurement_scope": "full_e2e",
        "decode_ms": decode,
        "decode_active_ms": decode,
        "decode_token_count": tokens,
        "decode_phase_definition": "after_first_token_to_final_token",
        "decode_phase_verified": True,
        "timed_generated_tokens": tokens + 1 if tokens is not None else None,
        "visible_output_tokens": tokens + 1 if tokens is not None else None,
        "text": text,
    }


def test_common_speedups_keep_mismatched_output_pairs_and_use_ratio_of_sums():
    from Benchmark.common.paired_reference import aggregate_pair

    reference = [
        _row("a", wall=100, decode=60, tokens=30, text="reference A"),
        _row("b", wall=200, decode=60, tokens=30, text="reference B"),
    ]
    method = [
        {**_row("a", wall=50, decode=30, tokens=24, text="different A"), "method": "dflash"},
        {**_row("b", wall=100, decode=30, tokens=24, text="different B"), "method": "dflash"},
    ]

    result = aggregate_pair(reference, method, scope="common")

    assert result["common_esr"] == pytest.approx(2.0)
    assert result["common_decode_time_ratio"] == pytest.approx(2.0)
    assert result["common_decode_rate_ratio"] == pytest.approx(1.6)
    assert result["common_output_rate_ratio"] == pytest.approx((50 / 150) / (62 / 300))
    assert result["common_esr_valid_sample_ids"] == ["a", "b"]
    assert result["common_quality"]["exact_text_match_rate"] == 0.0
    assert result["common_quality"]["valid_pair_count"] == 2
    assert result["common_quality"]["mean_method_visible_output_tokens"] == pytest.approx(25.0)
    assert result["common_quality"]["exact_token_count_match_rate"] == 0.0


def test_failed_missing_phase_and_single_output_exclusions_are_metric_specific():
    from Benchmark.common.paired_reference import aggregate_pair

    reference = [
        _row("a", wall=100, decode=0, tokens=0, text="a"),
        _row("b", wall=100, decode=50, tokens=20, text="b"),
        _row("c", wall=100, decode=50, tokens=20, text="c", status="failed"),
    ]
    method = [
        {**_row("a", wall=50, decode=0, tokens=0, text="a"), "method": "eagle3"},
        {**_row("b", wall=50, decode=None, tokens=None, text="b"), "method": "eagle3", "decode_phase_verified": False},
        {**_row("c", wall=None, decode=None, tokens=None, text="c", status="failed"), "method": "eagle3"},
    ]

    result = aggregate_pair(reference, method, scope="common")

    assert result["common_esr"] == pytest.approx(2.0)
    assert result["common_decode_time_ratio"] is None
    assert result["common_decode_time_exclusions"] == {
        "nonpositive_time": 1,
        "missing_decode_evidence": 1,
        "failed_status": 1,
    }
    assert result["common_decode_rate_ratio"] is None
    assert result["common_decode_rate_exclusions"]["zero_decode_tokens"] == 1


def test_decode_metrics_reject_e2e_only_rate_reconstruction_even_if_flagged_verified():
    from Benchmark.common.paired_reference import aggregate_pair

    reference = [_row("a", wall=100, decode=60, tokens=30, text="same")]
    method = {
        **_row("a", wall=50, decode=0.0016, tokens=7, text="same"),
        "method": "domino",
        "measurement_scope": "e2e_only",
        "decode_ms": None,
        "decode_active_ms": 0.0016,
        "decode_phase_verified": True,
        "timing_source": "sglang_0.5.20_api_server_request_time_stats",
    }

    result = aggregate_pair(reference, [method], scope="common")

    assert result["common_esr"] == pytest.approx(2.0)
    assert result["common_decode_time_ratio"] is None
    assert result["common_decode_rate_ratio"] is None
    assert result["common_decode_time_exclusions"] == {"missing_decode_evidence": 1}


def test_pairing_rejects_prompt_resource_and_duplicate_identity_mismatches():
    from Benchmark.common.paired_reference import aggregate_pair

    reference = [
        _row("a", wall=100, decode=60, tokens=30, text="a"),
        _row("b", wall=100, decode=60, tokens=30, text="b", prompt="p2"),
        _row("c", wall=100, decode=60, tokens=30, text="c"),
    ]
    method = [
        {**_row("a", wall=50, decode=30, tokens=20, text="a", prompt="other"), "method": "dspark"},
        {**_row("b", wall=50, decode=30, tokens=20, text="b", prompt="p2"), "method": "dspark", "gpu_count": 2},
        {**_row("c", wall=50, decode=30, tokens=20, text="c"), "method": "dspark"},
        {**_row("c", wall=40, decode=20, tokens=20, text="duplicate"), "method": "dspark"},
    ]

    result = aggregate_pair(reference, method, scope="common")

    assert result["common_esr"] is None
    assert result["common_esr_exclusions"] == {
        "prompt_mismatch": 1,
        "unequal_resource": 1,
        "duplicate_sample_id": 2,
    }



def test_pairing_rejects_native_rows_from_different_runs_or_without_scope():
    from Benchmark.common.paired_reference import aggregate_pair

    reference = _row("a", wall=100, decode=60, tokens=30, text="native")
    method = {**_row("a", wall=50, decode=30, tokens=20, text="method"), "method": "dflash"}
    reference.update(native_elapsed_ms=80, native_timing_scope="generation", run_id="run-a")
    method.update(native_elapsed_ms=40, native_timing_scope="generation", run_id="run-b")

    result = aggregate_pair([reference], [method], scope="native")
    assert result["native_esr"] is None
    assert result["native_esr_exclusions"] == {"config_mismatch": 1}

    method["run_id"] = "run-a"
    method.pop("native_timing_scope")
    result = aggregate_pair([reference], [method], scope="native")
    assert result["native_esr"] is None
    assert result["native_esr_exclusions"] == {"missing_identity": 1}

def test_nonfinite_and_nonpositive_times_are_excluded_without_zero_filling():
    from Benchmark.common.paired_reference import aggregate_pair

    reference = [_row("a", wall=100, decode=60, tokens=30, text="a")]
    method = [{**_row("a", wall=math.inf, decode=0, tokens=30, text="a"), "method": "domino"}]

    result = aggregate_pair(reference, method, scope="common")

    assert result["common_esr"] is None
    assert result["common_esr_exclusions"] == {"nonfinite_time": 1}
    assert result["common_decode_time_exclusions"] == {"nonpositive_time": 1}


def test_native_scope_uses_native_elapsed_fields_and_shared_bootstrap_is_seeded():
    from Benchmark.common.paired_reference import aggregate_pair, paired_bootstrap_ci

    reference = [_row("a", wall=1000, decode=60, tokens=30, text="native",)]
    reference[0].update(native_elapsed_ms=80, native_timing_scope="generation")
    method = [{**_row("a", wall=900, decode=30, tokens=24, text="method"), "method": "dflash"}]
    method[0].update(native_elapsed_ms=40, native_timing_scope="generation")

    result = aggregate_pair(reference, method, scope="native")

    assert result["native_esr"] == pytest.approx(2.0)
    pairs = [(2.0, 1.0), (4.0, 2.0), (6.0, 3.0)]
    assert paired_bootstrap_ci(pairs, seed=7, resamples=100) == paired_bootstrap_ci(
        pairs, seed=7, resamples=100
    )


def test_v2_record_fields_fingerprint_prompts_generation_and_hardware():
    from Benchmark.common.paired_reference import build_v2_record_fields

    first = build_v2_record_fields(
        prompt_token_ids=[11, 12],
        generation_config={"temperature": 0, "max_new_tokens": 64, "stop_token_ids": [2]},
        hardware={"gpu_name": "B200", "gpu_count": 1},
        request_wall_ms=80,
        native_elapsed_ms=70,
        native_timing_scope="generation",
        timed_generated_tokens=8,
        visible_output_tokens=7,
        decode_active_ms=40,
        decode_token_count=7,
        decode_phase_definition="after_first_token_to_final_token",
        decode_phase_verified=True,
    )
    same = build_v2_record_fields(
        prompt_token_ids=[11, 12],
        generation_config={"stop_token_ids": [2], "max_new_tokens": 64, "temperature": 0},
        hardware={"gpu_count": 1, "gpu_name": "B200"},
    )
    changed_prompt = build_v2_record_fields(
        prompt_token_ids=[11, 13],
        generation_config={"temperature": 0, "max_new_tokens": 64, "stop_token_ids": [2]},
        hardware={"gpu_name": "B200", "gpu_count": 1},
    )
    changed_config = build_v2_record_fields(
        prompt_token_ids=[11, 12],
        generation_config={"temperature": 0, "max_new_tokens": 65, "stop_token_ids": [2]},
        hardware={"gpu_name": "B200", "gpu_count": 1},
    )

    assert first["contract_version"] == 2
    assert first["prompt_token_sha256"] == same["prompt_token_sha256"]
    assert first["generation_config_sha256"] == same["generation_config_sha256"]
    assert first["hardware_fingerprint"] == same["hardware_fingerprint"]
    assert first["prompt_token_sha256"] != changed_prompt["prompt_token_sha256"]
    assert first["generation_config_sha256"] != changed_config["generation_config_sha256"]
    assert first["request_wall_ms"] == 80
    assert first["decode_phase_verified"] is True


def test_paper_report_emits_pairwise_shared_and_stratified_tables(tmp_path):
    import json

    from Benchmark.common.paired_reference import (
        PAPER_BASELINES,
        PAPER_DATASETS,
        build_paper_report,
        write_paper_report,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "contract_version": 2,
        "paper_speedup": True,
        "mode": "smoke",
        "common_reference": "vanilla_fa",
        "baselines": list(PAPER_BASELINES),
        "datasets": list(PAPER_DATASETS),
        "sample_count": 2,
        "bootstrap_resamples": 20,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def record(sample_id, method, wall, text, *, native=None):
        row = _row(sample_id, wall=wall, decode=40, tokens=20, text=text)
        row.update(method=method, prompt_token_count_match=True,
                   request_wall_ms=wall, native_elapsed_ms=wall,
                   native_timing_scope="generation",
                   target_revision="model", tokenizer_revision="tokenizer",
                   generation_config_sha256="gen", pairing_config_sha256="gen",
                   rouge1_f=0.5, rougeL_f=0.4)
        if native is not None:
            row["native_reference"] = native
        return row

    def native_reference(sample_id, method, elapsed):
        row = _row(sample_id, wall=elapsed, decode=50, tokens=20, text="native")
        row.update(method=method, native_elapsed_ms=elapsed,
                   native_timing_scope="generation", target_revision="model",
                   tokenizer_revision="tokenizer", generation_config_sha256="gen",
                   pairing_config_sha256="gen")
        return row

    for dataset in PAPER_DATASETS:
        reference_rows = [
            record("a", "vanilla_fa", 100, "ref-a"),
            record("b", "vanilla_fa", 200, "ref-b"),
        ]
        for baseline in PAPER_BASELINES:
            method_rows = []
            for index, sample_id in enumerate(("a", "b")):
                wall = 50 if baseline != "vanilla_fa" else (100 if index == 0 else 200)
                text = f"different-{sample_id}" if baseline not in {"vanilla_fa"} else f"ref-{sample_id}"
                native = None
                if baseline in {"dflash", "eagle3"}:
                    native = native_reference(sample_id, f"{baseline}_native", 80)
                method_rows.append(record(sample_id, baseline, wall, text, native=native))
            if baseline == "vanilla_fa":
                method_rows = reference_rows
            for row in method_rows:
                row["dataset"] = dataset
                if isinstance(row.get("native_reference"), dict):
                    row["native_reference"]["dataset"] = dataset
            output = run_dir / baseline / f"{dataset}.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(json.dumps(row) + "\n" for row in method_rows), encoding="utf-8")
        sidecar = run_dir / "references" / "sglang_target_only" / f"{dataset}.jsonl"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar_rows = [native_reference(sid, "target_only", 80) for sid in ("a", "b")]
        for row in sidecar_rows:
            row["dataset"] = dataset
        sidecar.write_text("".join(json.dumps(row) + "\n" for row in sidecar_rows), encoding="utf-8")

    report = build_paper_report(run_dir)
    dflash = next(row for row in report["rows"] if row["dataset"] == "vietnews" and row["method"] == "dflash")
    assert dflash["common_esr"] == pytest.approx(3.0)
    assert dflash["common_quality"]["exact_text_match_rate"] == 0.0
    assert len(report["shared_rows"]) == 4 * 6 * 4 + 6 * 4
    assert report["audit"]["paper_gate"]["status"] == "smoke_pass", report["audit"]["paper_gate"]
    paths = write_paper_report(run_dir)
    before = {key: path.read_bytes() for key, path in paths.items()}
    write_paper_report(run_dir)
    assert before == {key: path.read_bytes() for key, path in paths.items()}
