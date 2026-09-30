from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import Benchmark.vllm_all_baselines as evaluator
from Benchmark.vllm_all_baselines import (
    build_speculative_config,
    mean_itl_ms,
    order_methods,
    token_lcs_overlap,
)


def test_json_artifacts_end_with_real_newlines(tmp_path):
    row_path = tmp_path / "rows.jsonl"
    json_path = tmp_path / "report.json"

    evaluator._append_jsonl(row_path, {"sample_id": "a"})
    evaluator._append_jsonl(row_path, {"sample_id": "b"})
    evaluator._write_json(json_path, {"status": "ok"})

    assert row_path.read_text(encoding="utf-8").splitlines() == [
        '{"sample_id": "a"}',
        '{"sample_id": "b"}',
    ]
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"status": "ok"}


def test_normalize_vllm_spec_metrics_derives_counters_and_rates():
    metrics = {
        "num_drafts": 2,
        "num_draft_tokens": 8,
        "num_accepted_tokens": 6,
        "num_accepted_tokens_per_pos": [2, 2, 1, 1],
    }

    normalized = evaluator._normalize_vllm_spec_metrics(metrics, method="dflash")

    assert normalized["verification_steps"] == 2
    assert normalized["draft_tokens_accepted"] == 6
    assert normalized["draft_tokens_proposed"] == 8
    assert normalized["acceptance_rate"] == pytest.approx(0.75)
    assert normalized["acceptance_rate_percent"] == pytest.approx(75.0)
    assert normalized["avg_accept_length"] == pytest.approx(4.0)
    assert normalized["draft_proposal_unit"] == "draft_tokens"


def test_normalize_vllm_spec_metrics_preserves_tree_proposal_unit():
    normalized = evaluator._normalize_vllm_spec_metrics(
        {"acceptance_rate": 0.25, "avg_accept_length": 2.0}, method="eagle3"
    )

    assert normalized["acceptance_rate"] == pytest.approx(0.25)
    assert normalized["draft_proposal_unit"] == "draft_tree_nodes"


def test_event_jsonl_is_timestamped_and_keeps_details(tmp_path):
    event_path = tmp_path / "events.jsonl"
    started = evaluator.time.perf_counter()

    evaluator._append_event(
        event_path, "request_finished", started, method="dflash", sample_id="a", output_tokens=7
    )

    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["event"] == "request_finished"
    assert event["elapsed_ms"] >= 0
    assert event["method"] == "dflash"
    assert event["sample_id"] == "a"
    assert event["output_tokens"] == 7


def test_nvidia_smi_snapshot_parses_device_telemetry(monkeypatch):
    class Completed:
        returncode = 0
        stdout = (
            "0, NVIDIA H100, 81920, 12345, 69475, 50, 40, 61, 550.54.15\n"
        )
        stderr = ""

    monkeypatch.setattr(evaluator.subprocess, "run", lambda *args, **kwargs: Completed())

    snapshot = evaluator._nvidia_smi_snapshot()

    assert snapshot["available"] is True
    assert snapshot["devices"] == [
        {
            "index": "0",
            "name": "NVIDIA H100",
            "memory_total_mib": "81920",
            "memory_used_mib": "12345",
            "memory_free_mib": "69475",
            "gpu_utilization_percent": "50",
            "memory_utilization_percent": "40",
            "temperature_c": "61",
            "driver_version": "550.54.15",
        }
    ]


def test_method_metrics_summarizes_speculative_counters():
    records = [
        {
            "method": "eagle3",
            "status": "success",
            "quality_valid": True,
            "repetition_flag": False,
            "tpot_ms": 10.0,
            "raw_prefill_ms": 20.0,
            "rouge1": 0.3,
            "rouge2": 0.2,
            "rougeL": 0.25,
            "speculative_metrics_available": True,
            "acceptance_rate": 0.5,
            "avg_accept_length": 2.0,
            "draft_tokens_accepted": 2,
            "draft_tokens_proposed": 4,
            "draft_proposal_unit": "draft_tree_nodes",
        },
        {
            "method": "eagle3",
            "status": "success",
            "quality_valid": True,
            "repetition_flag": False,
            "tpot_ms": 12.0,
            "raw_prefill_ms": 22.0,
            "rouge1": 0.5,
            "rouge2": 0.4,
            "rougeL": 0.45,
            "speculative_metrics_available": True,
            "acceptance_rate": 0.75,
            "avg_accept_length": 4.0,
            "draft_tokens_accepted": 6,
            "draft_tokens_proposed": 8,
            "draft_proposal_unit": "draft_tree_nodes",
        },
    ]

    metrics = evaluator._method_metrics(records, ("eagle3",))["eagle3"]

    assert metrics["mean_acceptance_rate"] == 0.625
    assert metrics["mean_acceptance_rate_percent"] == 62.5
    assert metrics["mean_avg_accept_length"] == 3.0
    assert metrics["total_draft_tokens_accepted_observed"] == 8
    assert metrics["total_draft_tokens_proposed_observed"] == 12
    assert metrics["draft_proposal_units"] == ["draft_tree_nodes"]


def test_prepared_sample_preserves_source_row_for_offline_recalculation():
    class StubTokenizer:
        chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

    source = {
        "id": "vn-1",
        "dataset": "vietnews",
        "document": "Bài báo cần tóm tắt.",
        "reference": "Tóm tắt gốc.",
        "document_words": 5,
    }

    samples, excluded = evaluator._prepare_samples(
        [source],
        StubTokenizer(),
        max_input_tokens=4096,
        max_total_tokens=4096,
        max_samples=0,
    )

    assert not excluded
    assert samples[0]["source_record"] == source
    assert samples[0]["reference"] == source["reference"]
    assert samples[0]["document_words"] == source["document_words"]


def test_full_cli_flag_is_available_and_wins_over_smoke_flag():
    args = evaluator._parser().parse_args(
        ["--model", "m", "--data-file", "d", "--output-dir", "o", "--full", "--smoke"]
    )

    assert evaluator._resolve_sample_limit(
        full=args.full,
        smoke=args.smoke,
        max_samples=args.max_samples,
    ) == 0


def test_full_mode_ignores_smoke_and_max_sample_caps():
    assert evaluator._resolve_sample_limit(
        full=True,
        smoke=True,
        max_samples=2,
    ) == 0


def test_smoke_mode_still_limits_samples_when_full_is_not_requested():
    assert evaluator._resolve_sample_limit(
        full=False,
        smoke=True,
        max_samples=0,
    ) == 2


def test_order_methods_keeps_vanilla_first_and_preserves_requested_order():
    assert order_methods("domino,vanilla_vllm,eagle3") == (
        "vanilla_vllm",
        "domino",
        "eagle3",
    )


def test_order_methods_requires_vanilla_and_rejects_duplicates():
    with pytest.raises(ValueError, match="must include reference"):
        order_methods("eagle3,domino")
    with pytest.raises(ValueError, match="duplicates"):
        order_methods("vanilla_vllm,eagle3,eagle3")


def test_domino_uses_the_vllm_dflash_adapter_and_keeps_block_size():
    config, metadata = build_speculative_config(
        "domino",
        "/checkpoints/domino",
        {
            "block_size": 16,
            "target_layer_ids": [1, 9, 17, 25, 33],
            "dflash_config": {"projector_type": "domino", "shift_label": True},
        },
        target_num_hidden_layers=36,
    )

    assert config == {
        "method": "dflash",
        "model": "/checkpoints/domino",
        "num_speculative_tokens": 16,
    }
    assert metadata["vllm_method"] == "dflash"
    assert metadata["projector_type"] == "domino"


def test_mean_itl_uses_inter_token_intervals_not_token_count():
    assert mean_itl_ms(first_token_ts=10.0, last_token_ts=28.0, output_tokens=4) == 6.0
    assert mean_itl_ms(first_token_ts=10.0, last_token_ts=10.0, output_tokens=1) is None


def test_token_lcs_overlap_is_ordered_and_uses_vanilla_length_as_denominator():
    assert token_lcs_overlap([1, 2, 3, 4], [1, 3, 4]) == pytest.approx(0.75)
    assert token_lcs_overlap([1, 2, 3, 4], [1, 2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert token_lcs_overlap([], []) is None
