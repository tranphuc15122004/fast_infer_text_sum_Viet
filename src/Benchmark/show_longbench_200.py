#!/usr/bin/env python3
"""Đọc và trình bày nhanh các record trong bộ LongBench canonical."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]

from Benchmark.common.benchmark_data import (  # noqa: E402
    DATASETS,
    read_jsonl,
    render_prompt,
    token_stats,
)


DEFAULT_DATA_DIR = ROOT / "datasets" / "eval_100"
CODE_DATASETS: set[str] = set()


def preview(value: object, limit: int) -> str:
    """Rút gọn text thành một dòng nhưng vẫn đánh dấu xuống dòng."""
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "↵")
    if not text:
        return "(empty)"
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def format_record(
    row: Mapping[str, object],
    sample_number: int,
    *,
    context_chars: int = 240,
    field_chars: int = 160,
) -> str:
    """Format một canonical record để người đọc hiểu input/target."""
    context = str(row.get("document") or "")
    input_text = render_prompt(row) if row.get("dataset") in DATASETS else context
    reference = str(row.get("reference") or "")
    answers = row.get("answers") or [reference]
    answers_count = len(answers) if isinstance(answers, list) else 0
    return "\n".join(
        (
            f"  Sample {sample_number}: id={row.get('id')} "
            f"task_type={row.get('task_type')}",
            f"    document_words={row.get('document_words')} "
            f"length_bin={row.get('length_bin')} answers={answers_count}",
            f"    document ({len(context)} chars): "
            f"{preview(context, context_chars)}",
            f"    prompt ({len(input_text)} chars): "
            f"{preview(input_text, field_chars)}",
            f"    reference ({len(reference)} chars): "
            f"{preview(reference, field_chars)}",
        )
    )


def _dataset_names(datasets: Iterable[str] | None) -> list[str]:
    selected = list(DATASETS if datasets is None else datasets)
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("At least one dataset must be selected")
    return selected


def render_report(
    data_dir: Path,
    *,
    datasets: Iterable[str] | None = None,
    samples: int = 2,
    context_chars: int = 240,
    field_chars: int = 160,
) -> str:
    """Tạo báo cáo text gồm bảng tổng quan và sample preview."""
    if samples < 0:
        raise ValueError("samples must be non-negative")
    if context_chars <= 0 or field_chars <= 0:
        raise ValueError("preview limits must be positive")

    data_dir = Path(data_dir)
    selected = _dataset_names(datasets)
    loaded: list[tuple[str, list[dict]]] = []
    for dataset in selected:
        loaded.append((dataset, read_jsonl(data_dir / f"{dataset}_100.jsonl")))

    lines = [
        "LongBench canonical dataset preview",
        f"data_dir: {data_dir}",
        "",
        "Overview:",
        "dataset       task_type         records  document_words (min/median/max)  metric",
        "-------------  ----------------  -------  ------------------------------  --------------",
    ]
    for dataset, rows in loaded:
        stats = token_stats(rows)
        task_type = str(rows[0].get("task_type", "unknown")) if rows else "unknown"
        metric = "code-completion" if dataset in CODE_DATASETS else "ROUGE/BLEU"
        token_range = (
            f"{stats['min']}/{stats['median']}/{stats['max']}"
            if rows
            else "-"
        )
        lines.append(
            f"{dataset:<13} {task_type:<17} {len(rows):>7}  "
            f"{token_range:>28}  {metric}"
        )

    lines.extend(("", "Samples:"))
    for dataset, rows in loaded:
        bins = Counter(row.get("length_bin") for row in rows)
        bin_text = ", ".join(f"{key}:{bins[key]}" for key in sorted(bins, key=str))
        lines.extend((f"\n[{dataset}] bins={{{bin_text}}}",))
        for index, row in enumerate(rows[:samples], start=1):
            lines.append(
                format_record(
                    row,
                    index,
                    context_chars=context_chars,
                    field_chars=field_chars,
                )
            )
    lines.extend(("", f"total displayed datasets: {len(loaded)}"))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="Chỉ hiển thị dataset này; lặp option để chọn nhiều dataset.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Số record mẫu mỗi dataset (mặc định: 2).",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=240,
        help="Số ký tự context tối đa hiển thị (mặc định: 240).",
    )
    parser.add_argument(
        "--field-chars",
        type=int,
        default=160,
        help="Số ký tự input/reference tối đa hiển thị (mặc định: 160).",
    )
    args = parser.parse_args()
    try:
        print(
            render_report(
                args.data_dir,
                datasets=args.dataset,
                samples=args.samples,
                context_chars=args.context_chars,
                field_chars=args.field_chars,
            )
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
