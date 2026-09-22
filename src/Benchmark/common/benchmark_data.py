"""Shared schema and deterministic utilities for the Vietnamese benchmark."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("vietnews", "wikilingua", "vims", "vlsp")
CODE_DATASETS = frozenset()
EXPECTED_SOURCE_COUNTS = {dataset: 100 for dataset in DATASETS}
REQUIRED_FIELDS = frozenset(
    ("id", "dataset", "source_split", "source_index", "task_type", "document", "reference")
)
PROMPT_TEMPLATE = (
    "Hãy tóm tắt văn bản sau bằng tiếng Việt. Chỉ trả lời bằng bản tóm tắt:\n\n"
    "{document}"
)


def load_prompt_templates(path: Path | None = None) -> dict[str, str]:
    """Return one stable Vietnamese summarization prompt for every dataset."""

    return {dataset: PROMPT_TEMPLATE for dataset in DATASETS}


def render_prompt(record: Mapping[str, Any], templates: Mapping[str, str] | None = None) -> str:
    dataset = str(record.get("dataset", ""))
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported Vietnamese dataset: {dataset!r}")
    document = str(record.get("document") or "")
    return (templates or load_prompt_templates())[dataset].format(document=document)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _length(row: Mapping[str, Any]) -> int:
    for key in ("input_tokens", "document_words"):
        value = row.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return len(str(row.get("document") or "").split())


def _balanced_bins(rows: Sequence[dict[str, Any]], n_bins: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (_length(row), str(row.get("id", ""))))
    base, remainder = divmod(len(ordered), n_bins)
    bins: list[list[dict[str, Any]]] = []
    start = 0
    for index in range(n_bins):
        size = base + (1 if index < remainder else 0)
        bins.append(ordered[start : start + size])
        start += size
    return bins


def stratified_sample(
    rows: Sequence[dict[str, Any]], n: int, n_bins: int = 5, seed: int = 42
) -> list[dict[str, Any]]:
    if n <= 0 or n > len(rows):
        raise ValueError(f"Cannot select {n} rows from {len(rows)} source rows")
    if n_bins <= 0 or n % n_bins:
        raise ValueError(f"Selection count {n} must be divisible by {n_bins} bins")
    rng = random.Random(f"vietbench:{seed}")
    selected: list[dict[str, Any]] = []
    per_bin = n // n_bins
    for bin_index, candidates in enumerate(_balanced_bins(rows, n_bins)):
        if len(candidates) < per_bin:
            raise ValueError(f"Length bin {bin_index} has too few rows")
        for row in rng.sample(candidates, per_bin):
            selected.append(dict(row, length_bin=bin_index))
    return sorted(selected, key=lambda row: str(row.get("id", "")))


def select_rows(rows: Sequence[dict[str, Any]], dataset: str, n: int, seed: int) -> list[dict[str, Any]]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported Vietnamese dataset: {dataset}")
    if n == len(rows):
        return [dict(row) for row in rows]
    if n == 1:
        return [dict(rows[0])]
    return stratified_sample(rows, n=n, n_bins=5, seed=seed)


def validate_record(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        return [f"missing fields: {sorted(missing)}"]
    if not str(record["id"]).strip():
        problems.append("id must be non-empty")
    if record["dataset"] not in DATASETS:
        problems.append(f"unknown dataset: {record['dataset']!r}")
    if record["task_type"] != "summarization":
        problems.append("task_type must be 'summarization'")
    if not str(record["document"]).strip():
        problems.append("document must be non-empty")
    if not str(record["reference"]).strip():
        problems.append("reference must be non-empty")
    return problems


def token_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    values = [_length(row) for row in rows]
    if not values:
        return {"num_samples": 0, "min": 0, "median": 0, "mean": 0.0, "max": 0}
    ordered = sorted(values)
    return {
        "num_samples": len(values),
        "min": min(values),
        "median": ordered[len(ordered) // 2],
        "mean": round(sum(values) / len(values), 2),
        "max": max(values),
    }


def validate_output_dir(output_dir: Path, expected_count: int = 100) -> dict[str, Any]:
    """Validate the committed eval_100 profile and its recorded checksums."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", {})
    if set(datasets) != set(DATASETS):
        raise ValueError(f"manifest datasets must be {list(DATASETS)}")
    summary: dict[str, Any] = {"datasets": {}, "total": 0}
    for dataset in DATASETS:
        path = output_dir / f"{dataset}_100.jsonl"
        rows = read_jsonl(path)
        if len(rows) != expected_count:
            raise ValueError(f"{dataset}: expected {expected_count} rows, got {len(rows)}")
        expected_hash = datasets[dataset].get("eval_sha256")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash and digest != expected_hash:
            raise ValueError(f"checksum mismatch for {dataset}")
        for row in rows:
            problems = validate_record(row)
            if problems:
                raise ValueError(f"{dataset}: {'; '.join(problems)}")
            if row["dataset"] != dataset:
                raise ValueError(f"{dataset}: record has dataset {row['dataset']!r}")
        summary["datasets"][dataset] = token_stats(rows)
        summary["total"] += len(rows)
    return summary


def metric_family(task_type: str) -> str:
    if task_type != "summarization":
        raise ValueError(f"Unknown task type: {task_type!r}")
    return "rouge"

