from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_parse_baselines_accepts_comma_separated_values_and_deduplicates() -> None:
    from modal_benchmark_debug import parse_baselines

    assert parse_baselines(" dspark, vanilla_hf dspark ") == [
        "dspark",
        "vanilla_hf",
    ]


def test_validate_smoke_result_requires_successful_audited_coverage() -> None:
    from modal_benchmark_debug import validate_smoke_result

    manifest = {
        "failure_count": 0,
        "cells": [
            {
                "baseline": "dspark",
                "dataset": "vietnews",
                "status": "success",
                "sample_count": 1,
                "preflight": {"status": "ready", "requirements": {}},
                "metric_audit_summary": {
                    "num_records": 1,
                    "status_counts": {"success": 1},
                    "issue_counts": {},
                },
                "metric_contract": {
                    "status": "complete",
                    "observed_samples": 1,
                    "expected_samples": 1,
                    "issue_counts": {},
                },
            }
        ],
    }

    assert validate_smoke_result(0, manifest, ["dspark"]) == {
        "status": "passed",
        "baselines": {"dspark": {"status": "passed", "issues": []}},
        "issues": [],
    }

    manifest["cells"][0]["metric_contract"]["observed_samples"] = 0
    result = validate_smoke_result(0, manifest, ["dspark"])
    assert result["status"] == "failed"
    assert "coverage mismatch" in " ".join(result["baselines"]["dspark"]["issues"])


def test_validate_smoke_result_rejects_missing_dependency_even_on_zero_exit() -> None:
    from modal_benchmark_debug import validate_smoke_result

    manifest = {
        "failure_count": 0,
        "cells": [
            {
                "baseline": "dspark",
                "dataset": "vietnews",
                "status": "missing_dependency",
                "preflight": {
                    "status": "missing_dependency",
                    "reason": "missing dependency: sglang",
                    "requirements": {"sglang": False},
                },
                "metric_contract": {"status": "metric_incomplete"},
            }
        ],
    }

    result = validate_smoke_result(0, manifest, ["dspark"])

    assert result["status"] == "failed"
    assert any("missing_dependency" in issue for issue in result["issues"])


def test_run_persists_complete_output_log(tmp_path: Path) -> None:
    from modal_benchmark_debug import _run

    log_path = tmp_path / "baseline.log"
    result = _run(
        [sys.executable, "-c", "print('first-line'); print('last-line')"],
        env={},
        cwd=tmp_path,
        log_path=log_path,
        timeout_seconds=10,
    )

    assert result["returncode"] == 0
    assert result["log_path"] == str(log_path)
    assert log_path.read_text(encoding="utf-8") == "first-line\nlast-line\n"
    assert "first-line" in result["output_tail"]



def test_modal_runtime_pins_match_sglang_019_runtime() -> None:
    from modal_benchmark_debug import MODAL_REQUIRED_VERSIONS

    assert MODAL_REQUIRED_VERSIONS["torch"] == "2.13.0"
    assert MODAL_REQUIRED_VERSIONS["sglang"] == "0.5.19"
    assert MODAL_REQUIRED_VERSIONS["sglang-kernel"] == "0.4.6.post1"
    assert MODAL_REQUIRED_VERSIONS["flashinfer-python"] == "0.6.18"
    assert MODAL_REQUIRED_VERSIONS["flash-attn-4"] == "4.0.0b19"

