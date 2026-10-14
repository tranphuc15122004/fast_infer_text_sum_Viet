#!/usr/bin/env python3
"""Build Vietnamese summarization evaluation sets partitioned by prompt length.

Input lengths are measured after rendering the benchmark's Vietnamese prompt
and the target tokenizer's chat template. Only the tokenizer is loaded; model
weights are not required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from Benchmark.common.benchmark_data import DATASETS, render_prompt, validate_record
from Benchmark.common.prompt_format import format_chat_prompt
from Benchmark.common.benchmark_data import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = REPO_ROOT / "datasets" / "normalized"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "length_bins"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"
CHECKPOINT_NAME = ".build_checkpoint.json"

# Half-open intervals, in full formatted prompt tokens.
BINS = (
    ("bin1_00k_01k", 0, 1_000),
    ("bin2_01k_02k", 1_000, 2_000),
    ("bin3_02k_04k", 2_000, 4_000),
    ("bin4_04k_06k", 4_000, 6_000),
    ("bin5_06k_12k", 6_000, 12_000),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_atomic(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _source_fingerprint(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    ]


def _candidate_score(seed: int, bin_name: str, dataset: str, record_id: str) -> str:
    payload = f"{seed}\0{bin_name}\0{dataset}\0{record_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _insert_reservoir(
    reservoir: list[dict[str, Any]], item: dict[str, Any], capacity: int
) -> None:
    """Keep the capacity lowest stable hash scores, independent of file order."""
    reservoir.append(item)
    reservoir.sort(key=lambda candidate: candidate["_selection_score"])
    if len(reservoir) > capacity:
        reservoir.pop()


def _new_state(
    *, fingerprints: list[dict[str, Any]], tokenizer_name: str, seed: int,
    samples_per_bin: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "fingerprints": fingerprints,
        "tokenizer": tokenizer_name,
        "seed": seed,
        "samples_per_bin": samples_per_bin,
        "resume_file_index": 0,
        "resume_line_number": 0,
        "source_counts": {},
        "population_counts": {name: 0 for name, _, _ in BINS},
        "outside_range_count": 0,
        "reservoirs": {name: {} for name, _, _ in BINS},
        "processed_rows": 0,
    }


def _load_or_create_state(
    checkpoint: Path, *, fingerprints: list[dict[str, Any]], tokenizer_name: str,
    seed: int, samples_per_bin: int, resume: bool,
) -> dict[str, Any]:
    expected = _new_state(
        fingerprints=fingerprints,
        tokenizer_name=tokenizer_name,
        seed=seed,
        samples_per_bin=samples_per_bin,
    )
    if not resume or not checkpoint.is_file():
        return expected

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    for key in ("version", "fingerprints", "tokenizer", "seed", "samples_per_bin"):
        if state.get(key) != expected[key]:
            raise ValueError(
                f"Checkpoint không khớp cấu hình ở trường {key!r}; "
                "hãy xóa checkpoint hoặc chạy với --no-resume."
            )
    return state


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    _json_atomic(path, state)


def _token_count(tokenizer: Any, record: dict[str, Any]) -> int:
    plain_prompt = render_prompt(record)
    full_prompt = format_chat_prompt(tokenizer, plain_prompt)
    return len(tokenizer.encode(full_prompt, add_special_tokens=False))


def _progress(dataset: str, seen: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = seen / elapsed
    remaining = max(total - seen, 0) / rate if rate else 0
    pct = 100.0 * seen / total if total else 100.0
    print(
        f"[scan] {dataset}: {seen:,}/{total:,} ({pct:5.1f}%) · "
        f"{rate:,.0f} mẫu/s · còn khoảng {remaining / 60:.1f} phút",
        flush=True,
    )


def _scan(
    *, source_files: list[Path], output_dir: Path, tokenizer: Any,
    tokenizer_name: str, samples_per_bin: int, seed: int,
    checkpoint_every: int, resume: bool,
) -> dict[str, Any]:
    fingerprints = _source_fingerprint(source_files)
    checkpoint = output_dir / CHECKPOINT_NAME
    state = _load_or_create_state(
        checkpoint,
        fingerprints=fingerprints,
        tokenizer_name=tokenizer_name,
        seed=seed,
        samples_per_bin=samples_per_bin,
        resume=resume,
    )
    for file_index in range(int(state["resume_file_index"]), len(source_files)):
        source_file = source_files[file_index]
        dataset = source_file.stem
        if dataset not in DATASETS:
            raise ValueError(f"Dataset không được hỗ trợ: {dataset!r} ({source_file})")
        rows = read_jsonl(source_file)
        line_start = int(state["resume_line_number"]) if file_index == int(state["resume_file_index"]) else 0
        if line_start > len(rows):
            raise ValueError(f"Checkpoint line vượt quá số record trong {source_file}")

        state["source_counts"].setdefault(dataset, 0)
        file_started = time.monotonic()
        for row_index in range(line_start, len(rows)):
            record = rows[row_index]
            problems = validate_record(record)
            if problems:
                raise ValueError(f"{source_file}:{row_index + 1}: {'; '.join(problems)}")

            input_tokens = _token_count(tokenizer, record)
            record["input_tokens"] = input_tokens
            selected_bin = next(
                ((name, index) for index, (name, low, high) in enumerate(BINS)
                 if low <= input_tokens < high),
                None,
            )
            state["source_counts"][dataset] += 1
            state["processed_rows"] += 1

            if selected_bin is None:
                state["outside_range_count"] += 1
            else:
                bin_name, bin_index = selected_bin
                state["population_counts"][bin_name] += 1
                record["length_bin"] = bin_index
                record["length_bin_name"] = bin_name
                score = _candidate_score(seed, bin_name, dataset, str(record["id"]))
                record["_selection_score"] = score
                bucket = state["reservoirs"][bin_name].setdefault(dataset, [])
                _insert_reservoir(bucket, record, samples_per_bin)

            completed_rows = row_index + 1
            if state["processed_rows"] % checkpoint_every == 0:
                state["resume_file_index"] = file_index
                state["resume_line_number"] = completed_rows
                _save_checkpoint(checkpoint, state)
                _progress(dataset, completed_rows, len(rows), file_started)

        state["resume_file_index"] = file_index + 1
        state["resume_line_number"] = 0
        _save_checkpoint(checkpoint, state)
        print(
            f"[done] {dataset}: {state['source_counts'][dataset]:,} record",
            flush=True,
        )

    return state


def _select_bin(
    reservoirs: dict[str, list[dict[str, Any]]], samples_per_bin: int
) -> list[dict[str, Any]]:
    dataset_names = sorted(reservoirs)
    pools = {
        name: sorted(reservoirs[name], key=lambda row: row["_selection_score"])
        for name in dataset_names
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < samples_per_bin and any(pools.values()):
        for name in dataset_names:
            if pools[name] and len(selected) < samples_per_bin:
                selected.append(pools[name].pop(0))
    selected.sort(key=lambda row: (int(row["input_tokens"]), str(row["id"])))
    for row in selected:
        row.pop("_selection_score", None)
    return selected


def build_length_bins(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    samples_per_bin: int = 50,
    seed: int = 42,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    checkpoint_every: int = 500,
    resume: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    if samples_per_bin <= 0:
        raise ValueError("--samples-per-bin phải lớn hơn 0")
    if checkpoint_every <= 0:
        raise ValueError("--checkpoint-every phải lớn hơn 0")
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_files = sorted(source_dir.glob("*.jsonl"))
    if not source_files:
        raise FileNotFoundError(f"Không có file JSONL trong {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_files = [output_dir / f"{name}.jsonl" for name, _, _ in BINS]
    manifest_path = output_dir / "manifest.json"
    has_checkpoint = (output_dir / CHECKPOINT_NAME).exists()
    if not overwrite and not has_checkpoint and (
        manifest_path.exists() or any(path.exists() for path in final_files)
    ):
        raise FileExistsError(
            f"Đã có output trong {output_dir}; truyền --overwrite để tạo lại."
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu transformers. Dùng venv mô phỏng B200 hoặc môi trường server có tokenizer."
        ) from exc

    print(f"[load] tokenizer={tokenizer_name} (local_files_only=True)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    state = _scan(
        source_files=source_files,
        output_dir=output_dir,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        samples_per_bin=samples_per_bin,
        seed=seed,
        checkpoint_every=checkpoint_every,
        resume=resume,
    )

    try:
        source_dir_display = source_dir.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source_dir_display = str(source_dir)
    manifest: dict[str, Any] = {
        "version": "1.0",
        "description": "Bộ đánh giá tóm tắt tiếng Việt phân vùng theo độ dài prompt.",
        "source_dir": source_dir_display,
        "source_counts": dict(sorted(state["source_counts"].items())),
        "source_files": [
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "records": state["source_counts"].get(path.stem, 0),
            }
            for path in source_files
        ],
        "tokenizer": tokenizer_name,
        "token_metric": "Số token Qwen3-4B của prompt đầy đủ sau prompt tiếng Việt và chat template; add_special_tokens=false",
        "prompt_template": "Hãy tóm tắt văn bản sau bằng tiếng Việt. Chỉ trả lời bằng bản tóm tắt:\\n\\n{document}",
        "seed": seed,
        "samples_per_bin_requested": samples_per_bin,
        "selection_strategy": "Xếp hạng ổn định bằng SHA-256 trong từng dataset, lấy vòng tròn giữa các dataset có dữ liệu, rồi sắp tăng dần theo input_tokens.",
        "excluded_outside_ranges": int(state["outside_range_count"]),
        "bins": {},
    }

    for index, (bin_name, low, high) in enumerate(BINS):
        reservoirs = state["reservoirs"].get(bin_name, {})
        selected = _select_bin(reservoirs, samples_per_bin)
        output_path = output_dir / f"{bin_name}.jsonl"
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(output_path)

        tokens = [int(row["input_tokens"]) for row in selected]
        dataset_dist = Counter(str(row["dataset"]) for row in selected)
        manifest["bins"][bin_name] = {
            "file": output_path.name,
            "range_tokens": [low, high],
            "population_count": int(state["population_counts"][bin_name]),
            "selected_count": len(selected),
            "min_tokens": min(tokens) if tokens else None,
            "median_tokens": sorted(tokens)[len(tokens) // 2] if tokens else None,
            "mean_tokens": round(sum(tokens) / len(tokens), 2) if tokens else None,
            "max_tokens": max(tokens) if tokens else None,
            "dataset_distribution": dict(sorted(dataset_dist.items())),
            "sha256": sha256_file(output_path),
            "sample_ids": [str(row["id"]) for row in selected],
        }
        if len(selected) < samples_per_bin:
            print(
                f"[warn] {bin_name}: chỉ có {len(selected)}/{samples_per_bin} mẫu; "
                "đã ghi toàn bộ mẫu có sẵn.",
                file=sys.stderr,
                flush=True,
            )

    _json_atomic(manifest_path, manifest)
    checkpoint = output_dir / CHECKPOINT_NAME
    checkpoint.unlink(missing_ok=True)
    print(f"[done] tạo {sum(v['selected_count'] for v in manifest['bins'].values())} mẫu tại {output_dir}", flush=True)
    print(f"[done] manifest: {manifest_path}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tạo bộ benchmark tiếng Việt theo length bins.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-bin", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("MODEL_TARGET", DEFAULT_TOKENIZER),
        help="Đường dẫn tokenizer/model Qwen3-4B local; chỉ đọc local cache.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--no-resume", action="store_true", help="Bỏ checkpoint cũ và quét lại từ đầu.")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè các file length bins đã tồn tại.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_length_bins(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        samples_per_bin=args.samples_per_bin,
        seed=args.seed,
        tokenizer_name=args.tokenizer,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
