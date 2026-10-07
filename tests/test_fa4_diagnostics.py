from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _row(method, sample="a", *, dataset="vietnews", repeat=0, **kwargs):
    return {
        "method": method,
        "dataset": dataset,
        "sample_id": sample,
        "repeat_index": repeat,
        "status": "success",
        "input_tokens": 100,
        "source_input_tokens": 100,
        "output_tokens": 10,
        "output_token_ids": list(range(10)),
        "e2e_ms": 100.0,
        **kwargs,
    }


def _run(path, rows, **summary):
    path.mkdir()
    records = rows + [{"record_type": "summary", "run_id": path.name, **summary}]
    (path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    return path


def _analyze(*args):
    return importlib.import_module("Benchmark.fa4_diagnostics").analyze_runs(*args)


def test_pairs_ignore_failures_orphans_and_old_retry_timings(tmp_path):
    run = _run(tmp_path / "full", [
        _row("vanilla_hf", source_input_tokens=1000, input_was_truncated=True),
        _row("dflash", e2e_ms=40.0),
        _row("dflash", e2e_ms=50.0, source_input_tokens=1000,
             input_was_truncated=True, draft_tokens_accepted=2, draft_tokens_proposed=10),
        _row("dflash", "orphan", e2e_ms=1.0),
        _row("dflash", "failed", status="runtime_error", e2e_ms=1.0),
    ])
    result = _analyze(run)
    assert result["duplicate_rows"] == 1
    assert result["failed_rows"] == 1
    assert result["truncated_samples"] == 1
    metric = result["overall"][0]
    assert metric["pairs"] == 1
    assert metric["ratio_of_total_e2e"] == 2.0
    assert metric["faster_pairs"] == 1
    assert metric["weighted_acceptance_percent"] == 20.0
    assert metric["draft_latency_ms_count"] == 0
    assert metric["mean_draft_latency_ms"] is None


def test_cross_run_join_uses_dataset_sample_and_repeat(tmp_path):
    current = _run(tmp_path / "full", [
        _row("vanilla_hf", "same", e2e_ms=500.0),
        _row("vanilla_hf", "same", dataset="vims", e2e_ms=300.0),
        _row("vanilla_hf", "same", repeat=1, e2e_ms=400.0),
    ])
    previous = _run(tmp_path / "representative", [
        _row("vanilla_hf", "same", dataset="vims", e2e_ms=3000.0),
        _row("vanilla_hf", "same", e2e_ms=5000.0),
        _row("vanilla_hf", "other", e2e_ms=9000.0),
    ])
    comparison = _analyze(current, previous)["comparison"]
    assert len(comparison["common_samples"]) == 2
    assert comparison["unmatched_current_rows"] == 1
    assert comparison["unmatched_previous_rows"] == 1
    assert comparison["by_method"][0]["previous_over_current_e2e"] == 10.0
    assert {r["dataset"] for r in comparison["common_samples"]} == {"vietnews", "vims"}


def test_comparison_exposes_config_and_output_changes(tmp_path):
    current = _run(tmp_path / "new", [
        _row("vanilla_hf", e2e_ms=50.0, output_tokens=12, output_token_ids=[1, 2]),
    ], versions={"flash-attn-4": "new"}, max_input_tokens=8192)
    previous = _run(tmp_path / "old", [
        _row("vanilla_hf", e2e_ms=100.0),
    ], versions={"flash-attn-4": "old"}, max_input_tokens=8192)
    comparison = _analyze(current, previous)["comparison"]
    assert "versions" in comparison["config_differences"]
    assert "max_input_tokens" not in comparison["config_differences"]
    cell = comparison["common_samples"][0]
    assert cell["input_tokens_match"] is True
    assert cell["output_tokens_match"] is False
    assert cell["output_ids_match"] is False


def test_length_groups_use_measured_prompt_tokens_and_reject_unpaired_inputs(tmp_path):
    run = _run(tmp_path / "lengths", [
        _row("vanilla_hf", input_tokens=128, source_input_tokens=10000),
        _row("dflash", input_tokens=128, source_input_tokens=10000, e2e_ms=50.0),
        _row("vanilla_hf", "long", input_tokens=4000, source_input_tokens=4000),
        _row("dflash", "long", input_tokens=4000, source_input_tokens=4000, e2e_ms=200.0),
        _row("vanilla_hf", "wrong", input_tokens=500),
        _row("dflash", "wrong", input_tokens=600, e2e_ms=1.0),
    ])
    result = _analyze(run)
    assert result["input_mismatched_pairs"] == 1
    assert {r["group"]: r["ratio_of_total_e2e"] for r in result["by_input_length"]} == {
        "1-512": 2.0, "2049-4096": 0.5,
    }


def test_cli_runs_without_site_packages_and_preserves_source_artifact(tmp_path):
    run = _run(tmp_path / "run", [
        _row("vanilla_hf"), _row("dflash", e2e_ms=200.0),
    ], status="greedy_parity_failure")
    source = (run / "results.jsonl").read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_fa4_benchmark.py"
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--run-dir", str(run)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((run / "diagnostics/diagnostics.json").read_text(encoding="utf-8"))
    assert report["overall"][0]["ratio_of_total_e2e"] == 0.5
    assert (run / "diagnostics/report_vi.md").is_file()
    assert (run / "results.jsonl").read_bytes() == source


def test_invalid_artifact_reports_line_number(tmp_path):
    run = tmp_path / "bad"
    run.mkdir()
    (run / "results.jsonl").write_text('{}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="results.jsonl:2"):
        _analyze(run)
