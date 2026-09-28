from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _complete_vanilla_record() -> dict:
    return {
        "method": "vanilla_hf",
        "dataset": "vietnews",
        "sample_id": "a",
        "status": "success",
        "measurement_scope": "full_e2e",
        "input_tokens": 10,
        "output_tokens": 4,
        "retained_tokens": 10,
        "batch_size": 1,
        "model_load_ms": 1.0,
        "peak_memory_gb": 2.0,
        "device": "cuda",
        "e2e_ms": 20.0,
        "prefill_ms": 3.0,
        "ttft_ms": 3.0,
        "decode_ms": 17.0,
        "throughput_tok_s": 200.0,
        "text": "tóm tắt",
        "reference_output": "tóm tắt",
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
        "rouge1_p": 1.0,
        "rouge1_r": 1.0,
        "rouge1_f": 1.0,
        "rouge2_p": 1.0,
        "rouge2_r": 1.0,
        "rouge2_f": 1.0,
        "rougeL_p": 1.0,
        "rougeL_r": 1.0,
        "rougeL_f": 1.0,
        "rougeLsum_p": 1.0,
        "rougeLsum_r": 1.0,
        "rougeLsum_f": 1.0,
        "bleu1": 1.0,
        "bleu2": 1.0,
        "bleu3": 1.0,
        "bleu4": 1.0,
        "length_ratio": 1.0,
    }


def test_viet_baseline_contract_is_persisted_on_summary(tmp_path) -> None:
    from Benchmark.common.metric_audit import audit_output_file

    output = tmp_path / "vietnews.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps(_complete_vanilla_record(), ensure_ascii=False),
                json.dumps({"type": "summary", "status": "success"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = audit_output_file(
        output,
        baseline="vanilla_hf",
        dataset="vietnews",
        audit_path=tmp_path / "audit.json",
        expected_output_tokens=8,
        expected_samples=1,
    )

    assert summary["metric_contract"]["status"] == "complete"
    persisted = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert persisted[-1]["metric_contract"]["status"] == "complete"


def test_normalize_dflash_backfills_full_e2e_scope(tmp_path) -> None:
    from Benchmark.run_longbench_200 import _normalize_child_output

    output = tmp_path / "dflash.jsonl"
    output.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "status": "success",
                "input_tokens": 10,
                "output_tokens": 4,
                "e2e_ms": 20.0,
                "prefill_ms": 3.0,
                "ttft_ms": 3.0,
                "decode_ms": 17.0,
                "text": "tóm tắt",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    _normalize_child_output(
        output,
        baseline="dflash",
        dataset="vietnews",
        source_records=[
            {
                "id": "a",
                "reference_output": "tóm tắt",
                "task_type": "summarization",
            }
        ],
        config={"model": "m", "device": "cuda", "max_new_tokens": 8},
        run_id="r1",
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["measurement_scope"] == "full_e2e"


def test_reference_selection_keeps_pinned_reference_when_output_is_degenerate(tmp_path) -> None:
    from Benchmark.run_longbench_200 import _select_external_reference

    flash = tmp_path / "vanilla_fa" / "vietnews.jsonl"
    flash.parent.mkdir(parents=True)
    flash.write_text(
        json.dumps(
            {
                "status": "success",
                "sample_id": "a",
                "e2e_ms": 10.0,
                "output_quality_guard": {"degenerate_repetition": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hf = tmp_path / "vanilla_hf" / "vietnews.jsonl"
    hf.parent.mkdir(parents=True)
    hf.write_text(
        json.dumps({"status": "success", "sample_id": "a", "e2e_ms": 12.0})
        + "\n",
        encoding="utf-8",
    )

    selected = _select_external_reference(
        tmp_path,
        "vietnews",
        ["vanilla_fa", "vanilla_hf", "eagle3"],
    )

    assert selected == flash


def test_external_speedup_keeps_degenerate_output_timing_and_reports_quality(tmp_path) -> None:
    from Benchmark.run_longbench_200 import _attach_external_reference_metrics

    reference = tmp_path / "vanilla_fa.jsonl"
    reference.write_text(
        json.dumps({"sample_id": "a", "status": "success", "output_tokens": 40,
                    "e2e_ms": 80.0, "decode_ms": 70.0,
                    "text": "bản tin hôm nay có nội dung đầy đủ và không lặp lại."}) + "\n",
        encoding="utf-8",
    )
    speculative = tmp_path / "dspark.jsonl"
    speculative.write_text(
        json.dumps({"sample_id": "a", "status": "success", "output_tokens": 40,
                    "e2e_ms": 30.0, "decode_ms": 20.0,
                    "text": " ".join(["bản tin hôm nay"] * 40)}) + "\n",
        encoding="utf-8",
    )

    _attach_external_reference_metrics(
        speculative, reference, reference_baseline="vanilla_fa"
    )
    row = json.loads(speculative.read_text(encoding="utf-8").splitlines()[0])

    assert row["speedup_valid"] is True
    assert row["external_e2e_speedup"] == 2.6667
    assert row["output_degenerate"] is True


def test_external_speedup_record_contains_the_e2e_ratio(tmp_path) -> None:
    from Benchmark.run_longbench_200 import _attach_external_reference_metrics

    reference = tmp_path / "vanilla_hf.jsonl"
    reference.write_text(
        json.dumps({"sample_id": "a", "status": "success", "output_tokens": 40,
                    "e2e_ms": 80.0, "decode_ms": 70.0,
                    "text": "bản tin hôm nay có nội dung đầy đủ và không lặp lại."}) + "\n",
        encoding="utf-8",
    )
    speculative = tmp_path / "dflash.jsonl"
    speculative.write_text(
        json.dumps({"sample_id": "a", "status": "success", "output_tokens": 40,
                    "e2e_ms": 30.0, "decode_ms": 20.0,
                    "text": "bản tin hôm nay có nội dung đầy đủ và không lặp lại."}) + "\n",
        encoding="utf-8",
    )

    _attach_external_reference_metrics(
        speculative, reference, reference_baseline="vanilla_hf"
    )
    row = json.loads(speculative.read_text(encoding="utf-8").splitlines()[0])

    assert row["speedup_valid"] is True
    assert row["speedup"] == 2.6667
    assert row["external_decode_speedup"] == 3.5


def test_external_speedup_keeps_different_text_timing_and_reports_quality(tmp_path) -> None:
    from Benchmark.run_longbench_200 import _attach_external_reference_metrics

    reference = tmp_path / "vanilla_fa.jsonl"
    reference.write_text(
        json.dumps({"sample_id": "a", "status": "success", "output_tokens": 40,
                    "e2e_ms": 80.0, "decode_ms": 70.0,
                    "text": "bản tóm tắt dense khác nội dung"}) + "\n",
        encoding="utf-8",
    )
    speculative = tmp_path / "dspark.jsonl"
    speculative.write_text(
        json.dumps({"sample_id": "a", "status": "success", "output_tokens": 40,
                    "e2e_ms": 30.0, "decode_ms": 20.0,
                    "text": "bản tóm tắt speculative khác nội dung"}) + "\n",
        encoding="utf-8",
    )

    _attach_external_reference_metrics(
        speculative, reference, reference_baseline="vanilla_fa"
    )
    row = json.loads(speculative.read_text(encoding="utf-8").splitlines()[0])

    assert row["speedup_valid"] is True
    assert row["speedup"] == 2.6667
    assert row["external_reference_output_exact_match"] is False


def test_collect_metrics_labels_only_the_valid_eagle_speedup_scope() -> None:
    from Benchmark.collect_metrics import compute_group

    result = compute_group(
        [
            {
                "method": "eagle3",
                "speedup_valid": False,
                "speedup_scope": "external_reference",
                "external_reference_baseline": "vanilla_fa",
                "paired_speedup_valid": True,
                "naive_time": 10.0,
                "eagle_time": 11.0,
                "e2e_ms": 11.0,
                "text": "summary",
            }
        ],
        {},
    )

    assert result["speedup"] == {"paired_eagle_speedup": 0.9091}
    assert result["speedup_scope"] == "paired_eagle_target_greedy"
    assert "external_reference_baselines" not in result


def test_aggregate_speedup_keeps_paired_eagle_when_external_pair_is_invalid() -> None:
    from Benchmark.common.metrics import aggregate_speedup

    result = aggregate_speedup(
        [
            {
                "method": "eagle3",
                "speedup_valid": False,
                "paired_speedup_valid": True,
                "naive_time": 2.8,
                "eagle_time": 2.8,
                "dense_e2e_ms": 8.0,
                "e2e_ms": 4.0,
            }
        ]
    )

    assert result == {"paired_eagle_speedup": 1.0}


def test_speculative_repetition_is_audited_even_when_other_metrics_are_missing() -> None:
    from Benchmark.common.metric_audit import validate_cell_metric_contract

    record = {
        "method": "dspark",
        "sample_id": "a",
        "status": "success",
        "measurement_scope": "e2e_only",
        "input_tokens": 12,
        "output_tokens": 120,
        "batch_size": 1,
        "e2e_ms": 900.0,
        "throughput_tok_s": 133.3,
        "device": "cuda",
        "text": " ".join(["bản tin hôm nay"] * 40),
        "reference_output": "tóm tắt bản tin",
        "rouge1": 0.1,
    }

    contract = validate_cell_metric_contract(
        [record], baseline="dspark", expected_samples=1, expected_output_tokens=2048
    )

    assert contract["status"] == "metric_incomplete"
    assert contract["issue_counts"]["degenerate_repetition"] == 1



def test_degenerate_output_quality_warning_does_not_invalidate_timing_contract() -> None:
    from Benchmark.common.metric_audit import validate_cell_metric_contract

    record = {
        "method": "dspark",
        "sample_id": "a",
        "status": "success",
        "measurement_scope": "e2e_only",
        "input_tokens": 12,
        "retained_tokens": 12,
        "output_tokens": 120,
        "batch_size": 1,
        "e2e_ms": 900.0,
        "throughput_tok_s": 133.3,
        "device": "cuda",
        "avg_accept_length": 1.0,
        "acceptance_rate": 0.0,
        "text": " ".join(["bản tin hôm nay"] * 40),
    }

    contract = validate_cell_metric_contract(
        [record], baseline="dspark", expected_samples=1, expected_output_tokens=2048
    )

    assert contract["status"] == "complete"
    assert contract["issue_counts"]["degenerate_repetition"] == 1
    assert contract["invalid_sample_ids"] == []

def test_sglang_speculative_contract_requires_valid_acceptance_metrics() -> None:
    from Benchmark.common.metric_audit import validate_cell_metric_contract

    record = {
        "method": "dspark",
        "sample_id": "a",
        "status": "success",
        "measurement_scope": "e2e_only",
        "input_tokens": 12,
        "retained_tokens": 12,
        "output_tokens": 120,
        "batch_size": 1,
        "e2e_ms": 900.0,
        "throughput_tok_s": 133.3,
        "device": "cuda",
        "avg_accept_length": 6.1,
        "acceptance_rate": 1.2,
    }

    contract = validate_cell_metric_contract(
        [record], baseline="dspark", expected_samples=1, expected_output_tokens=2048
    )

    assert contract["status"] == "metric_incomplete"
    assert contract["issue_counts"]["invalid_acceptance_rate"] == 1


def test_sglang_acceptance_histogram_and_counters_are_cross_checked() -> None:
    from Benchmark.common.metric_audit import audit_record

    record = {
        "method": "dspark",
        "status": "success",
        "output_tokens": 10,
        "avg_accept_length": 1.5,
        "acceptance_rate": 0.5,
        "acceptance_rate_percent": 50.0,
        "accepted_draft_tokens_per_step": 0.5,
        "verification_steps": 4,
        "acceptance_histogram": [2, 2],
        "draft_tokens_accepted": 2,
        "draft_tokens_proposed": 4,
    }

    audit = audit_record(record)

    assert not any(issue.startswith("acceptance_") for issue in audit["issues"])
    assert "accepted_draft_tokens_per_step_mismatch" not in audit["issues"]


def test_acceptance_percentage_and_per_step_metrics_are_cross_checked() -> None:
    from Benchmark.common.metric_audit import audit_record

    record = {
        "method": "dspark",
        "status": "success",
        "output_tokens": 10,
        "avg_accept_length": 2.0,
        "acceptance_rate": 0.5,
        "acceptance_rate_percent": 0.5,
        "accepted_draft_tokens_per_step": 1.0,
        "verification_steps": 4,
        "draft_tokens_accepted": 2,
        "draft_tokens_proposed": 4,
    }

    issues = audit_record(record)["issues"]

    assert "acceptance_rate_percent_mismatch" in issues
    assert "accepted_draft_tokens_per_step_mismatch" in issues
    assert "acceptance_length_counter_mismatch" in issues


def test_sglang_histogram_allows_single_eos_token_count_difference() -> None:
    from Benchmark.common.metric_audit import audit_record

    record = {
        "method": "dspark",
        "status": "success",
        "output_tokens": 94,
        "avg_accept_length": 94 / 45,
        "acceptance_rate": 50 / 315,
        "verification_steps": 45,
        "acceptance_histogram": [17, 13, 11, 2, 1, 1],
        "draft_tokens_accepted": 50,
        "draft_tokens_proposed": 315,
    }

    audit = audit_record(record)

    assert "acceptance_histogram_length_mismatch" not in audit["issues"]


def test_eagle_acceptance_summary_matches_iteration_trace() -> None:
    from Benchmark.common.metric_audit import audit_record

    record = {
        "method": "eagle3",
        "status": "success",
        "measurement_scope": "full_e2e",
        "output_tokens": 4,
        "avg_accept_length": 2.0,
        "acceptance_rate": 0.4,
        "acceptance_lengths": [1, 3],
    }

    audit = audit_record(record)

    assert "acceptance_trace_average_mismatch" not in audit["issues"]
    assert "invalid_acceptance_length" not in audit["issues"]


def test_eagle_acceptance_counters_are_cross_checked_against_tree_budget() -> None:
    from Benchmark.common.metric_audit import audit_record

    record = {
        "method": "eagle3",
        "status": "success",
        "output_tokens": 4,
        "avg_accept_length": 2.0,
        "acceptance_rate": 0.4,
        "verification_steps": 2,
        "acceptance_lengths": [1, 3],
        "draft_tokens_accepted": 4,
        "draft_tokens_proposed": 93,
    }

    audit = audit_record(record)

    assert "acceptance_rate_counter_mismatch" in audit["issues"]


def test_eagle_greedy_output_mismatch_is_reported_as_quality_signal() -> None:
    from Benchmark.common.metric_audit import audit_record

    record = {
        "method": "eagle3",
        "status": "success",
        "output_tokens": 4,
        "avg_accept_length": 1.0,
        "acceptance_rate": 0.0,
        "acceptance_lengths": [1, 1, 1, 1],
        "verification_steps": 4,
        "draft_tokens_accepted": 0,
        "draft_tokens_proposed": 12,
        "target_greedy_check_applicable": True,
        "target_greedy_match": False,
    }

    audit = audit_record(record)

    assert audit["quality"]["target_greedy_match"] is False
    assert "target_greedy_output_mismatch" not in audit["issues"]


def test_sample_records_default_retained_tokens_to_input_tokens() -> None:
    from Benchmark.common.benchmark_runtime import build_sample_record

    record = build_sample_record(
        method="vanilla_hf",
        dataset="vietnews",
        sample_id="a",
        model="m",
        input_tokens=10,
        output_tokens=4,
        timing={
            "prefill_ms": 3.0,
            "ttft_ms": 3.0,
            "decode_ms": 17.0,
            "e2e_ms": 20.0,
            "peak_memory_gb": 2.0,
        },
        config={"device": "cuda", "max_new_tokens": 8},
    )

    assert record["retained_tokens"] == 10


def test_dflash_acceptance_summary_uses_draft_budget() -> None:
    from Benchmark.infer_dflash import summarize_acceptance

    summary = summarize_acceptance([3, 2], block_size=4)

    assert summary["avg_accept_length"] == 2.5
    assert summary["acceptance_rate"] == 0.5
    assert summary["rejected_draft_ratio"] == 0.5


def test_dflash_acceptance_rate_uses_runtime_counters_when_available() -> None:
    from Benchmark.infer_dflash import summarize_acceptance

    summary = summarize_acceptance(
        [1, 3, 2, 1],
        block_size=16,
        draft_tokens_accepted=3,
        draft_tokens_proposed=60,
    )

    assert summary["verification_steps"] == 4
    assert summary["avg_accept_length"] == 1.75
    assert summary["acceptance_rate"] == 0.05
    assert summary["acceptance_rate_percent"] == 5.0
    assert summary["accepted_draft_tokens_per_step"] == 0.75
    assert summary["draft_proposal_unit"] == "linear_draft_slot"
    assert summary["rejected_draft_ratio"] == 0.95


def test_domino_e2e_only_contract_does_not_require_unavailable_phase_metrics(
    tmp_path,
) -> None:
    from Benchmark.common.metric_audit import audit_output_file

    record = {
        "method": "domino",
        "dataset": "vietnews",
        "sample_id": "a",
        "status": "success",
        "measurement_scope": "e2e_only",
        "input_tokens": 10,
        "output_tokens": 4,
        "retained_tokens": 10,
        "batch_size": 1,
        "device": "cuda",
        "e2e_ms": 20.0,
        "throughput_tok_s": 200.0,
        "avg_accept_length": 1.0,
        "acceptance_rate": 0.0,
        "text": "tóm tắt",
        "reference_output": "tóm tắt",
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
        "rouge1_p": 1.0,
        "rouge1_r": 1.0,
        "rouge1_f": 1.0,
        "rouge2_p": 1.0,
        "rouge2_r": 1.0,
        "rouge2_f": 1.0,
        "rougeL_p": 1.0,
        "rougeL_r": 1.0,
        "rougeL_f": 1.0,
        "rougeLsum_p": 1.0,
        "rougeLsum_r": 1.0,
        "rougeLsum_f": 1.0,
        "bleu1": 1.0,
        "bleu2": 1.0,
        "bleu3": 1.0,
        "bleu4": 1.0,
        "length_ratio": 1.0,
    }
    output = tmp_path / "domino.jsonl"
    output.write_text(
        json.dumps(record, ensure_ascii=False)
        + "\n"
        + json.dumps({"type": "summary", "status": "success"})
        + "\n",
        encoding="utf-8",
    )

    summary = audit_output_file(
        output,
        baseline="domino",
        dataset="vietnews",
        audit_path=tmp_path / "audit.json",
        expected_output_tokens=8,
        expected_samples=1,
    )

    assert summary["metric_contract"]["status"] == "complete"
