#!/usr/bin/env python3
"""Phân tích phân phối của 4 bộ dữ liệu tóm tắt tiếng Việt.

Script chỉ **đọc** dữ liệu, không sửa và không sinh bộ eval. Kết quả dùng để
chọn tham số lấy mẫu (ngưỡng cắt OOD, số bin) và để kiểm tra bộ test sau khi
build có bám đúng phân phối nguồn hay không.

    python3 scripts/data/analyze_distribution.py
    python3 scripts/data/analyze_distribution.py --dataset vims vlsp
    python3 scripts/data/analyze_distribution.py --include-train-counts

Sinh ra:
    datasets/distribution_report.json   -- số liệu máy đọc
    datasets/distribution_report.md     -- báo cáo người đọc, có histogram ASCII

Không cần tokenizer hay thư viện ngoài: độ dài tính bằng số từ
(``str.split()``) và số ký tự, chạy được cả trên máy CPU local.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from viet_data import (  # noqa: E402
    DATASET_ORDER,
    DEFAULT_RAW_DIR,
    POOLS,
    iter_vietnews,
    length_stats,
    load_pool,
    word_count,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "datasets"


# --------------------------------------------------------------------------- #
# Histogram ASCII (không phụ thuộc matplotlib)
# --------------------------------------------------------------------------- #

def ascii_histogram(
    values: list[int], *, bins: int = 12, width: int = 44
) -> list[str]:
    """Vẽ histogram ASCII để báo cáo đọc được trên terminal."""

    if not values:
        return ["  (rỗng)"]
    lo, hi = min(values), max(values)
    if lo == hi:
        return [f"  tất cả giá trị = {lo}"]

    step = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        idx = min(int((value - lo) / step), bins - 1)
        counts[idx] += 1
    peak = max(counts) or 1

    lines = []
    for idx, count in enumerate(counts):
        start = lo + step * idx
        end = start + step
        bar = "#" * max(1 if count else 0, round(count / peak * width))
        lines.append(f"  {start:>9.0f}-{end:<9.0f} {count:>7d} |{bar}")
    return lines


def compact_counter(counter: collections.Counter, limit: int = 12) -> dict[str, int]:
    """Đưa Counter về dict đã sắp xếp, gộp phần đuôi thành ``khác``."""

    items = counter.most_common()
    out: dict[str, int] = {}
    for key, count in items[:limit]:
        out[str(key)] = count
    tail = items[limit:]
    if tail:
        out["(khác)"] = sum(c for _, c in tail)
    return out


# --------------------------------------------------------------------------- #
# Phân tích
# --------------------------------------------------------------------------- #

def analyze_dataset(
    dataset: str, raw_dir: Path, *, punct_clean: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Phân tích một bộ; trả về ``(report, records, skipped)``."""

    records, skipped = load_pool(
        dataset, raw_dir, punct_clean=punct_clean
    )

    # Bổ sung field độ dài ngay trên pool để phần render dùng chung được với
    # tầng normalize của build_eval_100.py.
    for row in records:
        row["document_words"] = word_count(row["document"])
        row["reference_words"] = word_count(row["reference"])

    doc_words = [r["document_words"] for r in records]
    ref_words = [r["reference_words"] for r in records]
    doc_chars = [len(r["document"]) for r in records]
    ref_chars = [len(r["reference"]) for r in records]
    ratio = [
        round(r / d, 4) if d else 0.0 for r, d in zip(ref_words, doc_words)
    ]
    doc_counts = collections.Counter(int(r["num_source_docs"]) for r in records)

    # Khử trùng lặp: cùng reference hoặc cùng document xuất hiện nhiều lần.
    dup_doc = collections.Counter(r["document"] for r in records)
    dup_ref = collections.Counter(r["reference"] for r in records)
    exact_dups = sum(c - 1 for c in dup_doc.values() if c > 1)
    ref_dups = sum(c - 1 for c in dup_ref.values() if c > 1)

    categories = collections.Counter(
        str(r["metadata"].get("category") or "(không có)")
        for r in records
    )

    report: dict[str, Any] = {
        "dataset": dataset,
        "pool_split": POOLS[dataset]["split"],
        "source_split": POOLS[dataset]["source_split"],
        "multi_document": POOLS[dataset]["multi_doc"],
        "records": len(records),
        "skipped_records": len(skipped),
        "skipped_reasons": compact_counter(
            collections.Counter(str(s.get("reason", "?")) for s in skipped)
        ),
        "document_words": length_stats(doc_words),
        "document_chars": length_stats(doc_chars),
        "reference_words": length_stats(ref_words),
        "reference_chars": length_stats(ref_chars),
        "compression_ratio_words": {
            "mean": round(statistics.mean(ratio), 4) if ratio else 0.0,
            "median": round(statistics.median(ratio), 4) if ratio else 0.0,
            "min": round(min(ratio), 4) if ratio else 0.0,
            "max": round(max(ratio), 4) if ratio else 0.0,
        },
        "source_docs_per_record": compact_counter(doc_counts),
        "duplicate_documents": exact_dups,
        "duplicate_references": ref_dups,
        "category_distribution": compact_counter(categories),
    }
    if dataset == "vims":
        report["clusters"] = len(records)
    return report, records, skipped


def count_other_splits(raw_dir: Path) -> dict[str, Any]:
    """Đếm file của các split không dùng làm pool (không đọc nội dung)."""

    out: dict[str, Any] = {"vietnews": {}, "wikilingua": {}}
    vn = raw_dir / "vietnews-master" / "data"
    for split in ("train", "val", "test"):
        folder = vn / f"{split}_tokenized"
        if folder.is_dir():
            out["vietnews"][split] = len(list(folder.glob("*.txt.seg")))
    wl = raw_dir / "wikilingua"
    for split in ("train", "val", "test"):
        path = wl / f"{split}.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                out["wikilingua"][split] = sum(1 for line in handle if line.strip())
    return out


# --------------------------------------------------------------------------- #
# Render markdown
# --------------------------------------------------------------------------- #

def render_markdown(
    reports: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    other_splits: dict[str, Any] | None,
) -> str:
    lines: list[str] = [
        "# Báo cáo phân phối dữ liệu — 4 bộ tóm tắt tiếng Việt",
        "",
        "Sinh tự động bởi `scripts/data/analyze_distribution.py`.",
        "Độ dài tính bằng số từ (`str.split()`), không dùng tokenizer.",
        "",
        "## Tổng quan",
        "",
        "| Bộ | Pool | Record | Bỏ qua | Doc (từ) median | Doc (từ) p95 | Ref (từ) median | Tỷ lệ nén |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for rep in reports:
        d = rep["document_words"]
        r = rep["reference_words"]
        lines.append(
            f"| `{rep['dataset']}` | `{rep['pool_split']}` | {rep['records']:,} | "
            f"{rep['skipped_records']} | {d.get('median', 0):,.0f} | {d.get('p95', 0):,.0f} | "
            f"{r.get('median', 0):,.0f} | {rep['compression_ratio_words']['median']:.3f} |"
        )

    for rep in reports:
        lines += ["", f"## `{rep['dataset']}`", ""]
        lines.append(
            f"- Pool: `{rep['pool_split']}` · source_split: `{rep['source_split']}` · "
            f"đa văn bản: {'có' if rep['multi_document'] else 'không'}"
        )
        lines.append(
            f"- Record hợp lệ: **{rep['records']:,}** · bỏ qua: {rep['skipped_records']}"
        )
        if rep["skipped_reasons"]:
            lines.append(f"- Lý do bỏ qua: `{rep['skipped_reasons']}`")

        for field, label in (
            ("document_words", "Độ dài document (từ)"),
            ("reference_words", "Độ dài reference (từ)"),
        ):
            st = rep[field]
            if not st:
                continue
            lines += ["", f"### {label}", ""]
            lines.append(
                f"min **{st['min']:,.0f}** · p1 {st['p1']:,.0f} · p5 {st['p5']:,.0f} · "
                f"p25 {st['p25']:,.0f} · median **{st['median']:,.0f}** · mean {st['mean']:,.0f} · "
                f"p75 {st['p75']:,.0f} · p95 {st['p95']:,.0f} · p99 {st['p99']:,.0f} · "
                f"max **{st['max']:,.0f}**"
            )
            lines += ["", "```text"]
            key = "document_words" if field == "document_words" else "reference_words"
            lines += ascii_histogram([int(r[key]) for r in pools[rep["dataset"]]])
            lines += ["```"]

        if rep["multi_document"]:
            lines += [
                "",
                f"### Số văn bản / record",
                "",
                f"`{rep['source_docs_per_record']}`",
            ]
        if rep["category_distribution"] and "(không có)" not in rep["category_distribution"]:
            lines += [
                "",
                "### Phân bố category",
                "",
                f"`{rep['category_distribution']}`",
            ]
        if rep["duplicate_documents"] or rep["duplicate_references"]:
            lines += [
                "",
                f"- ⚠️ Document trùng: **{rep['duplicate_documents']}** · "
                f"reference trùng: **{rep['duplicate_references']}**",
            ]

    if other_splits:
        lines += ["", "## Split không dùng làm pool", "", "| Bộ | Split | Số record |", "|---|---|---:|"]
        for ds, splits in other_splits.items():
            for split, count in splits.items():
                marker = " (pool)" if split == POOLS[ds]["split"] else ""
                lines.append(f"| `{ds}` | `{split}`{marker} | {count:,} |")

    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phân tích phân phối 4 bộ dữ liệu tóm tắt tiếng Việt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DATASET_ORDER),
        help="Chỉ phân tích các bộ này (mặc định: tất cả).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Thư mục dữ liệu thô (mặc định: {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Thư mục ghi báo cáo (mặc định: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--no-punct-clean",
        action="store_true",
        help="Không dọn khoảng trắng quanh dấu câu khi de-segment VietNews.",
    )
    parser.add_argument(
        "--include-train-counts",
        action="store_true",
        help="Đếm thêm số record của các split train/val (không đọc nội dung).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    punct_clean = not args.no_punct_clean

    reports: list[dict[str, Any]] = []
    pools: dict[str, list[dict[str, Any]]] = {}

    for dataset in args.dataset:
        print(f"[analyze] {dataset} ...", flush=True)
        try:
            report, records, _ = analyze_dataset(
                dataset, args.raw_dir, punct_clean=punct_clean
            )
        except FileNotFoundError as exc:
            print(f"  !! bỏ qua {dataset}: {exc}", file=sys.stderr)
            continue
        exists = sum(1 for r in records if r)
        print(
            f"  -> {report['records']:,} record hợp lệ "
            f"(bỏ qua {report['skipped_records']}), "
            f"doc median {report['document_words'].get('median', 0):,.0f} từ"
            + ("" if exists else " [rỗng]"),
            flush=True,
        )
        reports.append(report)
        pools[dataset] = records

    if not reports:
        print("Không có bộ nào được phân tích.", file=sys.stderr)
        return 1

    other = count_other_splits(args.raw_dir) if args.include_train_counts else None
    payload = {
        "datasets": reports,
        "other_splits": other,
        "config": {
            "raw_dir": str(args.raw_dir),
            "punct_clean": punct_clean,
            "length_metric": "words",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "distribution_report.json"
    md_path = args.out_dir / "distribution_report.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(
        render_markdown(reports, pools, other), encoding="utf-8"
    )
    print(f"\n[analyze] ghi {json_path}")
    print(f"[analyze] ghi {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
