from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


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
    from common.metric_audit import audit_output_file

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
    from run_longbench_200 import _normalize_child_output

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


def test_reference_selection_skips_degenerate_vanilla_output(tmp_path) -> None:
    from run_longbench_200 import _select_external_reference

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

    assert selected == hf


def test_sample_records_default_retained_tokens_to_input_tokens() -> None:
    from common.benchmark_runtime import build_sample_record

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
    from infer_dflash import summarize_acceptance

    summary = summarize_acceptance([3, 2], block_size=4)

    assert summary["avg_accept_length"] == 2.5
    assert summary["acceptance_rate"] == 0.5
    assert summary["rejected_draft_ratio"] == 0.5
