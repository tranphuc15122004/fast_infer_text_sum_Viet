from __future__ import annotations

import json
from pathlib import Path

import pytest

from Finetuning.data import load_summary_jsonl
from Finetuning.prepare_data import (
    iter_input_output_jsonl,
    prepare_finetune_dataset,
    resolve_source_path,
)


def _write_source(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_resolve_source_path_supports_relative_path_files(tmp_path: Path) -> None:
    source = tmp_path / "data" / "source.jsonl"
    source.parent.mkdir()
    source.write_text("{}\n", encoding="utf-8")
    path_file = tmp_path / "path.txt"
    path_file.write_text("data/source.jsonl\n", encoding="utf-8")

    assert resolve_source_path(path_file) == source.resolve()


def test_iter_input_output_jsonl_normalizes_to_summary_records(tmp_path: Path) -> None:
    source = tmp_path / "sample.jsonl"
    _write_source(
        source,
        [{"input": "  Văn bản Việt Nam  ", "output": "  Tóm tắt  "}],
    )

    records = list(iter_input_output_jsonl(source))

    assert len(records) == 1
    assert records[0].id == "finetune-00000001"
    assert records[0].document == "Văn bản Việt Nam"
    assert records[0].summary == "Tóm tắt"
    assert records[0].metadata == {
        "source_line": 1,
        "source_schema": "input_output",
    }


def test_iter_input_output_jsonl_rejects_missing_or_empty_fields(tmp_path: Path) -> None:
    source = tmp_path / "invalid.jsonl"
    _write_source(source, [{"input": "văn bản", "output": "   "}])

    with pytest.raises(ValueError, match=r"line 1.*output"):
        list(iter_input_output_jsonl(source))


def test_prepare_finetune_dataset_is_deterministic_and_keeps_duplicate_documents_together(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {"input": "document trùng", "output": "summary một"},
        {"input": "document trùng", "output": "summary hai"},
    ] + [
        {"input": f"document {index}", "output": f"summary {index}"}
        for index in range(2, 102)
    ]
    _write_source(source, rows)

    first = prepare_finetune_dataset(
        source,
        tmp_path / "prepared-a",
        eval_ratio=0.5,
        progress_interval=25,
    )
    second = prepare_finetune_dataset(
        source,
        tmp_path / "prepared-b",
        eval_ratio=0.5,
        progress_interval=25,
    )

    assert first == second
    assert first["total"] == 102
    assert first["train"] + first["eval"] == 102
    assert first["train"] > 0
    assert first["eval"] > 0

    train_a = (tmp_path / "prepared-a" / "train.jsonl").read_text(encoding="utf-8")
    eval_a = (tmp_path / "prepared-a" / "eval.jsonl").read_text(encoding="utf-8")
    assert train_a == (tmp_path / "prepared-b" / "train.jsonl").read_text(encoding="utf-8")
    assert eval_a == (tmp_path / "prepared-b" / "eval.jsonl").read_text(encoding="utf-8")

    train_records = load_summary_jsonl(tmp_path / "prepared-a" / "train.jsonl")
    eval_records = load_summary_jsonl(tmp_path / "prepared-a" / "eval.jsonl")
    split_by_summary = {
        record.summary: "train" for record in train_records
    } | {record.summary: "eval" for record in eval_records}
    assert split_by_summary["summary một"] == split_by_summary["summary hai"]

    manifest = json.loads(
        (tmp_path / "prepared-a" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "dflash_summary_v1"
    assert manifest["source_schema"] == "input_output"
    assert manifest["total"] == 102


def test_prepare_finetune_dataset_does_not_overwrite_without_force(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source, [{"input": "văn bản", "output": "tóm tắt"}])
    output_dir = tmp_path / "prepared"
    prepare_finetune_dataset(source, output_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_finetune_dataset(source, output_dir)
