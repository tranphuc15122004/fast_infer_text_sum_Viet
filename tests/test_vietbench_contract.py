from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_vietbench_registry_and_prompt_contract() -> None:
    from Benchmark.common import benchmark_data

    assert benchmark_data.DATASETS == ("vietnews", "wikilingua", "vims", "vlsp")
    row = {
        "dataset": "vietnews",
        "document": "Một văn bản tiếng Việt.",
        "reference": "Bản tóm tắt.",
    }
    prompt = benchmark_data.render_prompt(row)
    assert "tóm tắt" in prompt.lower()
    assert row["document"] in prompt


def test_eval_profile_has_four_datasets_with_one_hundred_rows() -> None:
    manifest = json.loads(
        (ROOT / "datasets" / "eval_100" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(manifest["datasets"]) == {"vietnews", "wikilingua", "vims", "vlsp"}
    for dataset in ("vietnews", "wikilingua", "vims", "vlsp"):
        path = ROOT / "datasets" / "eval_100" / f"{dataset}_100.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 100
        assert all(row["dataset"] == dataset for row in rows)
        assert all(row["task_type"] == "summarization" for row in rows)
        assert all(str(row["reference"]).strip() for row in rows)


def test_vietbench_selection_is_deterministic() -> None:
    from Benchmark.common import benchmark_data

    rows = benchmark_data.read_jsonl(ROOT / "datasets" / "eval_100" / "vims_100.jsonl")
    first = benchmark_data.select_rows(rows, dataset="vims", n=20, seed=42)
    second = benchmark_data.select_rows(rows, dataset="vims", n=20, seed=42)
    assert [row["id"] for row in first] == [row["id"] for row in second]


def test_loader_renders_vietnamese_prompt_and_reference() -> None:
    from Benchmark.common.data_loader import load_records

    rows = load_records(ROOT / "datasets/eval_100/vietnews_100.jsonl", max_samples=1)
    assert "tóm tắt" in rows[0]["prompt"].lower()
    assert rows[0]["raw"]["document"] in rows[0]["prompt"]
    assert rows[0]["reference_output"] == rows[0]["raw"]["reference"]
