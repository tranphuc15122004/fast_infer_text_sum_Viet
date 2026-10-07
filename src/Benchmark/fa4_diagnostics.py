"""Chẩn đoán latency FA4 từ artifact có sẵn, chỉ dùng Python standard library."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _mean(values) -> float | None:
    clean = [number for value in values if (number := _number(value)) is not None]
    return round(statistics.fmean(clean), 6) if clean else None


def _key(row: dict) -> tuple[str, str, str, int]:
    return (str(row["method"]), str(row["dataset"]), str(row["sample_id"]),
            int(row.get("repeat_index", 0)))


def _read_run(directory: Path) -> tuple[dict, list[dict], int]:
    path = directory / "results.jsonl"
    latest: dict[tuple, dict] = {}
    summary: dict = {}
    duplicates = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record phải là object")
                if row.get("record_type") == "summary" or row.get("scope") == "summary":
                    summary = row
                elif "sample_id" in row:
                    key = _key(row)
                    duplicates += int(key in latest)
                    latest[key] = row
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not latest:
        raise ValueError(f"Không có sample record trong {path}")
    if not summary and (directory / "run_report.json").is_file():
        summary = json.loads((directory / "run_report.json").read_text(encoding="utf-8"))
    return summary, list(latest.values()), duplicates


def _input_matches(left: dict, right: dict) -> bool:
    size = _number(left.get("input_tokens"))
    if not size or size != _number(right.get("input_tokens")):
        return False
    source_left = _number(left.get("source_input_tokens"))
    source_right = _number(right.get("source_input_tokens"))
    return source_left is None or source_right is None or source_left == source_right


def _length_bin(value: float) -> str:
    lower = 1
    for upper in (512, 2048, 4096, 8192):
        if value <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return "8193+"


def _summarize(method: str, group: str, pairs: list[tuple[dict, dict]]) -> dict:
    references, candidates = zip(*pairs)
    reference_total = sum(float(row["e2e_ms"]) for row in references)
    candidate_total = sum(float(row["e2e_ms"]) for row in candidates)
    counters = [
        (accepted, proposed)
        for row in candidates
        if (accepted := _number(row.get("draft_tokens_accepted"))) is not None
        and (proposed := _number(row.get("draft_tokens_proposed"))) is not None
        and proposed > 0
    ]
    proposed = sum(item[1] for item in counters)
    result = {
        "method": method, "group": group, "pairs": len(pairs),
        "ratio_of_total_e2e": round(reference_total / candidate_total, 6),
        "faster_pairs": sum(float(ref["e2e_ms"]) > float(row["e2e_ms"]) for ref, row in pairs),
        "mean_reference_e2e_ms": _mean(row["e2e_ms"] for row in references),
        "mean_method_e2e_ms": _mean(row["e2e_ms"] for row in candidates),
        "mean_output_tokens": _mean(row.get("output_tokens") for row in candidates),
        "mean_avg_accept_length": _mean(row.get("avg_accept_length") for row in candidates),
        "weighted_acceptance_percent": round(100 * sum(item[0] for item in counters) / proposed, 6)
        if proposed else None,
        "acceptance_counters_count": len(counters),
    }
    for phase in ("draft_latency_ms", "verification_latency_ms"):
        values = [_number(row.get(phase)) for row in candidates]
        result[f"mean_{phase}"] = _mean(values)
        result[f"{phase}_count"] = sum(value is not None for value in values)
    return result


def _compare(current_summary, current_rows, previous_summary, previous_rows) -> dict:
    current = {_key(row): row for row in current_rows if row.get("status") == "success"}
    previous = {_key(row): row for row in previous_rows if row.get("status") == "success"}
    cells = []
    by_method: dict[str, list[dict]] = defaultdict(list)
    for key in sorted(current.keys() & previous.keys()):
        new, old = current[key], previous[key]
        new_ms, old_ms = _number(new.get("e2e_ms")), _number(old.get("e2e_ms"))
        if not new_ms or not old_ms:
            continue
        cell = {
            "method": key[0], "dataset": key[1], "sample_id": key[2], "repeat_index": key[3],
            "previous_input_tokens": old.get("input_tokens"),
            "current_input_tokens": new.get("input_tokens"),
            "input_tokens_match": _input_matches(old, new),
            "previous_output_tokens": old.get("output_tokens"),
            "current_output_tokens": new.get("output_tokens"),
            "output_tokens_match": old.get("output_tokens") == new.get("output_tokens"),
            "output_ids_match": old["output_token_ids"] == new["output_token_ids"]
            if "output_token_ids" in old and "output_token_ids" in new else None,
            "previous_e2e_ms": old_ms, "current_e2e_ms": new_ms,
            "previous_over_current_e2e": round(old_ms / new_ms, 6),
        }
        cells.append(cell)
        by_method[key[0]].append(cell)
    fields = ("gpu", "attention_backend", "batch_size", "dtype", "seed", "max_new_tokens",
              "max_input_tokens", "warmup_tokens", "models", "versions", "data_sha256")
    differences = {
        field: {"previous": previous_summary.get(field), "current": current_summary.get(field)}
        for field in fields if previous_summary.get(field) != current_summary.get(field)
    }
    return {
        "previous_run_id": previous_summary.get("run_id"),
        "previous_sample_count": previous_summary.get("sample_count"),
        "config_differences": differences,
        "unmatched_current_rows": len(current.keys() - previous.keys()),
        "unmatched_previous_rows": len(previous.keys() - current.keys()),
        "common_samples": cells,
        "by_method": [
            {"method": method, "common_rows": len(rows),
             "previous_over_current_e2e": round(
                 sum(row["previous_e2e_ms"] for row in rows) / sum(row["current_e2e_ms"] for row in rows), 6),
             "changed_inputs": sum(not row["input_tokens_match"] for row in rows),
             "changed_output_lengths": sum(not row["output_tokens_match"] for row in rows)}
            for method, rows in by_method.items()
        ],
    }


def analyze_runs(run_dir: Path, compare_dir: Path | None = None) -> dict:
    """Ghép theo dataset/sample/repeat; không coi thiếu phase timing là 0 ms."""
    summary, rows, duplicates = _read_run(Path(run_dir))
    successful = [row for row in rows if row.get("status") == "success"]
    references = {_key(row)[1:]: row for row in successful if row["method"] == "vanilla_hf"}
    groups: dict[tuple[str, str, str], list[tuple[dict, dict]]] = defaultdict(list)
    mismatched = invalid_timing = unpaired = 0
    for row in successful:
        if row["method"] == "vanilla_hf":
            continue
        reference = references.get(_key(row)[1:])
        if reference is None:
            unpaired += 1
            continue
        if not _input_matches(reference, row):
            mismatched += 1
            continue
        if not _number(reference.get("e2e_ms")) or not _number(row.get("e2e_ms")):
            invalid_timing += 1
            continue
        for level, name in (("overall", "all"), ("by_dataset", str(row["dataset"])),
                            ("by_input_length", _length_bin(float(reference["input_tokens"])))):
            groups[(level, str(row["method"]), name)].append((reference, row))
    truncated = {
        (row["dataset"], row["sample_id"])
        for row in successful
        if row.get("input_was_truncated") or (
            (source := _number(row.get("source_input_tokens"))) is not None
            and (actual := _number(row.get("input_tokens"))) is not None and source > actual
        )
    }
    result = {
        "run_dir": str(Path(run_dir).resolve()),
        "run_id": summary.get("run_id", Path(run_dir).name),
        "run_status": summary.get("status"),
        "warmup_tokens": summary.get("warmup_tokens"),
        "sample_count": summary.get("sample_count"),
        "successful_rows": len(successful), "failed_rows": len(rows) - len(successful),
        "duplicate_rows": duplicates, "truncated_samples": len(truncated),
        "input_mismatched_pairs": mismatched, "invalid_timing_pairs": invalid_timing,
        "unpaired_rows": unpaired, "overall": [], "by_dataset": [], "by_input_length": [],
        "comparison": None,
    }
    for (level, method, name), pairs in groups.items():
        result[level].append(_summarize(method, name, pairs))
    if compare_dir is not None:
        previous_summary, previous_rows, _ = _read_run(Path(compare_dir))
        result["comparison"] = _compare(summary, rows, previous_summary, previous_rows)
    return result


def render_report(result: dict) -> str:
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    lines = [
        "# Chẩn đoán hiệu năng FA4 từ artifact",
        "", f"Run: {result['run_id']}; trạng thái benchmark: {result['run_status']}.",
        f"Sample records thành công: {result['successful_rows']}; lỗi: {result['failed_rows']}; "
        f"retry trùng đã gộp: {result['duplicate_rows']}; mẫu bị cắt input: {result['truncated_samples']}.",
        f"Warmup token/run: {result['warmup_tokens']}; số mẫu nguồn: {result['sample_count']}.",
        f"Cặp bị loại: input khác={result['input_mismatched_pairs']}, "
        f"timing thiếu/không hợp lệ={result['invalid_timing_pairs']}, thiếu Vanilla={result['unpaired_rows']}.",
        "", "E2E speedup = tổng thời gian Vanilla / tổng thời gian method trên các cặp cùng sample/repeat. "
        "Giá trị dưới 1 nghĩa là method chậm hơn; đây là tỷ số E2E đo trực tiếp, không phải DSR/ESR.",
    ]
    for level, title in (("overall", "Tổng thể"), ("by_dataset", "Theo dataset"),
                         ("by_input_length", "Theo số token input sau truncation")):
        lines += ["", f"## {title}", "",
                  "| Method | Nhóm | Cặp | E2E speedup | Mẫu nhanh hơn | Accept có trọng số (%) | Avg accept length | Draft TB ms (n) | Verify TB ms (n) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in result[level]:
            lines.append(
                f"| {row['method']} | {row['group']} | {row['pairs']} | {fmt(row['ratio_of_total_e2e'])} "
                f"| {row['faster_pairs']}/{row['pairs']} | {fmt(row['weighted_acceptance_percent'])} "
                f"| {fmt(row['mean_avg_accept_length'])} "
                f"| {fmt(row['mean_draft_latency_ms'])} ({row['draft_latency_ms_count']}) "
                f"| {fmt(row['mean_verification_latency_ms'])} ({row['verification_latency_ms_count']}) |"
            )
    comparison = result["comparison"]
    if comparison is not None:
        lines += ["", "## Cùng sample giữa hai run", "",
                  f"Run trước: {comparison['previous_run_id']}; số mẫu: {comparison['previous_sample_count']}.",
                  "Tỷ số trước/hiện tại > 1 cho biết cùng sample được đo chậm hơn ở run trước.",
                  "", "| Method | Cặp trùng | E2E trước/hiện tại | Input khác | Độ dài output khác |",
                  "|---|---:|---:|---:|---:|"]
        for row in comparison["by_method"]:
            lines.append(f"| {row['method']} | {row['common_rows']} | {fmt(row['previous_over_current_e2e'])} "
                         f"| {row['changed_inputs']} | {row['changed_output_lengths']} |")
        lines += ["", "Khác biệt cấu hình/version/checksum: "
                  + json.dumps(comparison["config_differences"], ensure_ascii=False),
                  "", "| Method | Dataset/sample | Repeat | Input trước/hiện tại | Output trước/hiện tại | E2E trước ms | E2E hiện tại ms | Trước/hiện tại |",
                  "|---|---|---:|---|---|---:|---:|---:|"]
        for row in comparison["common_samples"][:60]:
            lines.append(f"| {row['method']} | {row['dataset']}/{row['sample_id']} | {row['repeat_index']} "
                         f"| {row['previous_input_tokens']}/{row['current_input_tokens']} "
                         f"| {row['previous_output_tokens']}/{row['current_output_tokens']} "
                         f"| {fmt(row['previous_e2e_ms'])} | {fmt(row['current_e2e_ms'])} "
                         f"| {fmt(row['previous_over_current_e2e'])} |")
        if len(comparison["common_samples"]) > 60:
            lines += ["", "Bảng hiển thị 60 dòng đầu; CSV chứa toàn bộ cặp trùng."]
    lines += ["", "## Giới hạn diễn giải", "",
              "- Input có cùng độ dài chưa chứng minh cùng nội dung: kiểm tra cả checksum dữ liệu/model/version.",
              "- Output dài/ngắn khác nhau ảnh hưởng E2E; ROUGE/BLEU và token parity vẫn nằm trong artifact benchmark gốc.",
              "- Phase timing thiếu được ghi là null, không coi là 0. Không cộng các phase vào E2E nếu khác phạm vi hoặc đồng hồ đo.",
              "- Acceptance thấp là dấu hiệu cần profile, chưa xác nhận nguyên nhân chậm. Bảng này không đo GPU contention, clock, JIT hay CUDA synchronization.",
              "- Khác biệt giữa các run trên cùng sample cần kiểm tra warmup, tải GPU và môi trường trước khi quy cho phân phối dữ liệu."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phân tích artifact FA4 bằng CPU, không nạp model.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--compare-dir", type=Path, help="Run cũ để đối chiếu cùng sample ID và repeat")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze_runs(args.run_dir, args.compare_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    destination = args.output_dir or args.run_dir / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = render_report(result)
    (destination / "report_vi.md").write_text(report, encoding="utf-8")
    cells = (result["comparison"] or {}).get("common_samples", [])
    if cells:
        with (destination / "common_samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
            writer.writeheader()
            writer.writerows(cells)
    print(report)
    print(f"Báo cáo chẩn đoán: {destination.resolve()}")
    return 0
