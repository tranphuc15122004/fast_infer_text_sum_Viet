from __future__ import annotations

from Benchmark.common.fa4_benchmark import (
    aggregate_fa4_records,
    finalize_fa4_records,
    parse_selection,
    resolve_sample_limit,
    select_length_spread,
    token_lcs_overlap,
)


DATASETS = ("vietnews", "wikilingua", "vims", "vlsp")
METHODS = ("vanilla_hf", "dflash")


def test_parse_selection_supports_all_and_orders_by_canonical_values():
    assert parse_selection("vims, vietnews", DATASETS, "datasets") == (
        "vietnews",
        "vims",
    )
    assert parse_selection("all", DATASETS, "datasets") == DATASETS


def test_parse_selection_rejects_unknown_duplicate_and_mixed_all_values():
    for value in ("unknown", "vietnews,vietnews", "all,vims"):
        try:
            parse_selection(value, DATASETS, "datasets")
        except ValueError:
            continue
        raise AssertionError(f"expected invalid selection to fail: {value}")


def test_resolve_sample_limit_uses_small_representative_and_all_full_defaults():
    assert resolve_sample_limit("smoke", None, available=100) == 2
    assert resolve_sample_limit("representative", None, available=100) == 20
    assert resolve_sample_limit("full", None, available=100) == 100
    assert resolve_sample_limit("full", 7, available=100) == 7


def test_select_length_spread_is_deterministic_and_covers_length_extremes():
    rows = [{"sample_id": str(i), "input_tokens": i + 1} for i in range(10)]
    selected = select_length_spread(rows, 4)
    assert [row["sample_id"] for row in selected] == ["0", "3", "6", "9"]
    assert select_length_spread(rows, 0) == rows


def test_token_lcs_overlap_uses_vanilla_output_as_denominator():
    assert token_lcs_overlap([1, 2, 3, 4], [1, 3, 4, 5]) == 0.75
    assert token_lcs_overlap([], [1]) is None


def test_aggregate_fa4_records_reports_paired_speed_quality_parity_and_acceptance():
    records = [
        {
            "method": method,
            "dataset": "vietnews",
            "sample_id": f"sample-{sample_index}",
            "repeat_index": 0,
            "status": "success",
            "output_token_ids": output_ids,
            "input_tokens": 100,
            "retained_tokens": 100,
            "output_tokens": len(output_ids),
            "prefill_ms": 100.0,
            "ttft_ms": 100.0,
            "decode_ms": decode_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "throughput_tok_s": len(output_ids) / (e2e_ms / 1000.0),
            "peak_memory_gb": 4.0,
            "text": text,
            "reference_output": "tóm tắt tham chiếu",
            "rouge1": 0.5,
            "rouge2": 0.25,
            "rougeL": 0.4,
            "quality_valid": True,
            "repetition_flag": False,
            "draft_tokens_accepted": accepted,
            "draft_tokens_proposed": proposed,
            "verification_steps": 2,
            "avg_accept_length": 1.5,
            "acceptance_rate": accepted / proposed if proposed else None,
        }
        for method, sample_index, output_ids, decode_ms, tpot_ms, e2e_ms, text, accepted, proposed in (
            ("vanilla_hf", 1, [1, 2, 3], 20.0, 10.0, 120.0, "vanilla text", None, None),
            ("dflash", 1, [1, 2, 3], 10.0, 5.0, 110.0, "draft text", 2, 4),
            ("vanilla_hf", 2, [4, 5, 6, 7], 30.0, 10.0, 130.0, "vanilla two", None, None),
            ("dflash", 2, [4, 8, 6, 7], 15.0, 5.0, 115.0, "draft two", 3, 5),
        )
    ]

    summary = aggregate_fa4_records(records, methods=METHODS)
    metrics = summary["metrics_by_dataset"]["vietnews"]["dflash"]

    assert metrics["successful_samples"] == 2
    assert metrics["greedy_exact_matches"] == 1
    assert metrics["token_lcs_overlap_with_vanilla"] == 0.857143
    assert metrics["paired_speed_metrics"]["paired_samples"] == 2
    assert metrics["paired_speed_metrics"]["dsr"] == 2.0
    assert metrics["paired_speed_metrics"]["esr"] > 1.0
    assert metrics["mean_acceptance_rate"] == 0.55
    assert metrics["speed_statistics"]["e2e_ms"]["median"] == 112.5


def test_finalize_fa4_records_compacts_resume_retries_and_sets_run_gates():
    common = {
        "dataset": "vietnews",
        "sample_id": "vietnews:sample-1",
        "repeat_index": 0,
        "input_tokens": 64,
        "retained_tokens": 64,
        "output_tokens": 4,
        "prefill_ms": 20.0,
        "ttft_ms": 20.0,
        "decode_ms": 30.0,
        "tpot_ms": 10.0,
        "e2e_ms": 50.0,
        "throughput_tok_s": 80.0,
        "peak_memory_gb": 4.0,
        "text": "một bản tóm tắt hợp lệ",
        "reference_output": "một bản tóm tắt",
        "rouge1": 0.5,
        "rouge2": 0.25,
        "rougeL": 0.5,
        "quality_valid": True,
        "repetition_flag": False,
        "output_token_ids": [1, 2, 3, 4],
    }
    rows = [
        {**common, "method": "vanilla_hf", "status": "success"},
        {**common, "method": "dflash", "status": "runtime_error", "reason": "interrupted"},
        {**common, "method": "dflash", "status": "success", "e2e_ms": 40.0, "decode_ms": 20.0, "tpot_ms": 6.6667, "throughput_tok_s": 100.0},
    ]

    result = finalize_fa4_records(
        rows,
        methods=METHODS,
        datasets=("vietnews",),
        repetitions=1,
        expected_samples=1,
    )

    assert len(result["records"]) == 2
    assert result["execution_complete"] is True
    assert result["execution_pass"] is True
    assert result["quality_pass"] is True
    assert result["exact_match_all"] is True
    assert result["speedup_by_method"]["dflash"] > 1.0
    assert result["failure_count"] == 0
