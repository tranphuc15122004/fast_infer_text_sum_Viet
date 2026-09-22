#!/usr/bin/env python3
"""Chuẩn hoá 4 bộ dữ liệu tiếng Việt về MỘT format và trích bộ eval ~100 mẫu/bộ.

Hai tầng, giữ đúng convention của project tham chiếu (``prepare_data.py`` +
``extract_representative_samples.py``):

1. **normalize** -- đọc ``datasets/raw/``, ghi pool đầy đủ đã chuẩn hoá vào
   ``datasets/normalized/<dataset>.jsonl``. Có thể tái tạo bộ test từ đây mà
   không cần đọc lại 158k file raw.
2. **sample** -- chọn ~100 mẫu/bộ bằng ``stratified_sample``: **loại OOD** theo
   percentile rồi phân bổ theo phân phối độ dài, nên bộ test cân bằng và không
   lệch về tài liệu dài. Ghi ``datasets/eval_100/<dataset>_100.jsonl``.

    python3 scripts/data/build_eval_100.py
    python3 scripts/data/build_eval_100.py --stage sample --samples 100
    python3 scripts/data/build_eval_100.py --dataset vims vlsp --bins 5
    python3 scripts/data/build_eval_100.py --dry-run      # chỉ in thống kê

Chỉ dùng thư viện chuẩn; chạy được trên CPU local, không cần tokenizer/network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from Benchmark.data.viet_data import (
    DATASET_ORDER,
    DEFAULT_RAW_DIR,
    POOLS,
    filter_task_valid,
    load_pool,
    stratified_sample,
    word_count,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "datasets"


# --------------------------------------------------------------------------- #
# Chuẩn hoá
# --------------------------------------------------------------------------- #

def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bổ sung field độ dài và sắp xếp field theo schema thống nhất."""

    out: list[dict[str, Any]] = []
    for row in records:
        document = row["document"]
        reference = row["reference"]
        out.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "source_split": row["source_split"],
                "source_index": row["source_index"],
                "source_id": row["source_id"],
                "task_type": row["task_type"],
                "document": document,
                "reference": reference,
                "answers": row["answers"],
                "num_source_docs": row["num_source_docs"],
                "document_words": word_count(document),
                "reference_words": word_count(reference),
                "metadata": row.get("metadata", {}),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Kiểm tra kết quả
# --------------------------------------------------------------------------- #

def validate(records: list[dict[str, Any]], report: dict[str, Any]) -> list[str]:
    """Kiểm tra bộ eval có hợp lệ để chấm ROUGE và đúng ràng buộc OOD."""

    problems: list[str] = []
    ids: set[str] = set()
    low, high = report["clip_bounds_words"]

    for row in records:
        rid = row.get("id")
        if not rid:
            problems.append("record thiếu id")
            continue
        if rid in ids:
            problems.append(f"id trùng: {rid}")
        ids.add(rid)

        for field in ("document", "reference"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{rid}: {field} rỗng -> ROUGE sẽ bị bỏ qua")

        if not isinstance(row.get("answers"), list) or not row["answers"]:
            problems.append(f"{rid}: answers rỗng")

        length = row.get("document_words", 0)
        if not (low <= length <= high):
            problems.append(
                f"{rid}: document_words={length} ngoài khoảng chống OOD "
                f"[{low:.0f}, {high:.0f}]"
            )

        if row.get("length_bin") is None:
            problems.append(f"{rid}: thiếu length_bin")

        # Ràng buộc hợp lệ của task: tóm tắt phải ngắn hơn nguồn.
        if row.get("reference_words", 0) >= row.get("document_words", 0):
            problems.append(
                f"{rid}: reference_words={row.get('reference_words')} >= "
                f"document_words={row.get('document_words')} (mẫu suy biến)"
            )

    if len(records) != report["selected_samples"]:
        problems.append(
            f"selected_samples={report['selected_samples']} "
            f"nhưng ghi {len(records)} record"
        )
    return problems


def length_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Thống kê độ dài của bộ eval để so với phân phối pool."""

    from viet_data import length_stats

    doc = [int(r["document_words"]) for r in records]
    ref = [int(r["reference_words"]) for r in records]
    bins: dict[str, int] = {}
    for row in records:
        bins[str(row["length_bin"])] = bins.get(str(row["length_bin"]), 0) + 1
    return {
        "document_words": length_stats(doc),
        "reference_words": length_stats(ref),
        "per_bin_counts": bins,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chuẩn hoá + trích bộ eval 100 mẫu cho 4 bộ tóm tắt tiếng Việt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DATASET_ORDER),
        help="Chỉ xử lý các bộ này (mặc định: tất cả).",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "normalize", "sample"),
        default="all",
        help="Tầng cần chạy (mặc định: all).",
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Số mẫu mỗi bộ (mặc định: 100)."
    )
    parser.add_argument(
        "--bins", type=int, default=5, help="Số bin độ dài (mặc định: 5)."
    )
    parser.add_argument(
        "--clip-pct",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(1.0, 99.0),
        help="Percentile cắt OOD (mặc định: 1 99).",
    )
    parser.add_argument(
        "--min-per-bin",
        type=int,
        default=10,
        help="Sàn số mẫu mỗi bin (mặc định: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ghi vào manifest; thuật toán vốn deterministic (mặc định: 42).",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Thư mục datasets/raw."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Thư mục datasets/."
    )
    parser.add_argument(
        "--no-punct-clean",
        action="store_true",
        help="Giữ khoảng trắng quanh dấu câu khi de-segment VietNews.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in thống kê, không ghi file.",
    )
    parser.add_argument(
        "--min-doc-words",
        type=int,
        default=100,
        help=(
            "Sàn độ dài document (từ) để loại mẫu suy biến; 0 = tắt (mặc định: 100). "
            "Record có reference >= document luôn bị loại bất kể giá trị này."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    punct_clean = not args.no_punct_clean

    normalized_dir = args.out_dir / "normalized"
    eval_dir = args.out_dir / "eval_100"
    manifest: dict[str, Any] = {
        "config": {
            "samples_per_dataset": args.samples,
            "bins": args.bins,
            "clip_pct": list(args.clip_pct),
            "min_per_bin": args.min_per_bin,
            "min_doc_words": args.min_doc_words,
            "seed": args.seed,
            "length_metric": "words",
            "punct_clean": punct_clean,
            "raw_dir": str(args.raw_dir),
        },
        "datasets": {},
    }

    failed = False
    for dataset in args.dataset:
        print(f"[build] {dataset} ...", flush=True)
        try:
            records, skipped = load_pool(
                dataset, args.raw_dir, punct_clean=punct_clean
            )
        except FileNotFoundError as exc:
            print(f"  !! bỏ qua {dataset}: {exc}", file=sys.stderr)
            failed = True
            continue

        normalised = normalize_records(records)
        entry: dict[str, Any] = {
            "pool_records": len(normalised),
            "skipped_records": len(skipped),
            "skipped_reasons": {},
            "pool_split": POOLS[dataset]["split"],
            "source_split": POOLS[dataset]["source_split"],
        }
        for item in skipped:
            reason = str(item.get("reason", "?"))
            entry["skipped_reasons"][reason] = entry["skipped_reasons"].get(reason, 0) + 1

        # Loại mẫu suy biến (document quá ngắn, hoặc reference không ngắn hơn
        # document) TRƯỚC khi lấy mẫu. Cắt percentile một mình không đủ với các
        # bộ lệch mạnh như WikiLingua.
        valid, degenerate = filter_task_valid(
            normalised, min_doc_words=args.min_doc_words
        )
        entry["valid_records"] = len(valid)
        entry["degenerate_records"] = len(degenerate)
        entry["degenerate_reasons"] = {}
        for item in degenerate:
            reason = str(item.get("reason", "?"))
            entry["degenerate_reasons"][reason] = (
                entry["degenerate_reasons"].get(reason, 0) + 1
            )

        if args.stage in ("all", "normalize"):
            if args.dry_run:
                print(f"  [dry-run] sẽ ghi {len(normalised)} record normalized")
            else:
                path = normalized_dir / f"{dataset}.jsonl"
                checksum = write_jsonl(path, normalised)
                entry["normalized_file"] = str(path.relative_to(REPO_ROOT))
                entry["normalized_sha256"] = checksum
                print(f"  -> normalized {len(normalised):,} record: {path.name}")

        if args.stage in ("all", "sample"):
            result = stratified_sample(
                [dict(r) for r in valid],
                args.samples,
                bins=args.bins,
                clip_pct=(args.clip_pct[0], args.clip_pct[1]),
                min_per_bin=args.min_per_bin,
            )
            selected = result["selected"]
            report = result["report"]

            problems = validate(selected, report)
            entry["sampling"] = report
            entry["eval_summary"] = length_summary(selected)
            if problems:
                entry["validation_problems"] = problems
                failed = True
                print(f"  !! {len(problems)} lỗi validation", file=sys.stderr)
                for problem in problems[:5]:
                    print(f"     - {problem}", file=sys.stderr)

            if args.dry_run:
                print(
                    f"  [dry-run] sẽ ghi {len(selected)} mẫu "
                    f"(loại {report['outlier_count']} OOD)"
                )
            else:
                path = eval_dir / f"{dataset}_{args.samples}.jsonl"
                checksum = write_jsonl(path, selected)
                entry["eval_file"] = str(path.relative_to(REPO_ROOT))
                entry["eval_sha256"] = checksum
                entry["eval_records"] = len(selected)
                print(
                    f"  -> eval {len(selected)} mẫu "
                    f"(pool {report['pool_size']:,}, loại {report['outlier_count']} OOD): "
                    f"{path.name}"
                )

        manifest["datasets"][dataset] = entry

    if not args.dry_run:
        eval_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = eval_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n[build] ghi {manifest_path}")

    if failed:
        print("[build] CÓ LỖI — xem log phía trên.", file=sys.stderr)
        return 1
    print("[build] xong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
