"""Chẩn đoán latency FA4 từ artifact có sẵn, chỉ dùng Python standard library."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


def _signed_number(value: Any) -> float | None:
    """Parse finite values including negative derived deltas."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _distribution(values) -> dict[str, float | int | None]:
    clean = sorted(number for value in values if (number := _signed_number(value)) is not None)
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    # Nearest-rank p90 is deterministic and does not need NumPy.
    p90 = clean[max(0, math.ceil(0.90 * len(clean)) - 1)]
    return {
        "count": len(clean),
        "mean": round(statistics.fmean(clean), 6),
        "median": round(statistics.median(clean), 6),
        "p90": round(p90, 6),
        "min": round(clean[0], 6),
        "max": round(clean[-1], 6),
    }


_COMPONENT_TIMINGS = (
    "e2e_ms", "prefill_ms", "post_prefill_wall_ms", "decode_ms_recorded",
    "draft_latency_ms", "verification_latency_ms", "profiling_e2e_ms",
    "model_load_ms", "selector_latency_ms", "queue_wait_ms", "batch_wait_ms",
)


def _component_rows(summary: dict, rows: list[dict]) -> list[dict]:
    """Flatten each sample record and pair its timing with Vanilla on the same sample."""
    vanilla = {
        _key(row)[1:]: row
        for row in rows
        if row.get("method") == "vanilla_hf" and row.get("status") == "success"
    }
    flattened = []
    for row in rows:
        extra = row.get("extra_metrics") if isinstance(row.get("extra_metrics"), dict) else {}
        dispatch = extra.get("attention_dispatch") if isinstance(extra.get("attention_dispatch"), dict) else {}
        profile = extra.get("phase_profile") if isinstance(extra.get("phase_profile"), dict) else {}
        native_config = extra.get("native_inference_config") if isinstance(extra.get("native_inference_config"), dict) else {}
        vanilla_record = vanilla.get(_key(row)[1:]) if row.get("status") == "success" else None
        input_matches = _input_matches(row, vanilla_record) if vanilla_record else None
        reference = vanilla_record if input_matches else None

        e2e = _number(row.get("e2e_ms"))
        prefill = _number(row.get("prefill_ms"))
        post_prefill = e2e - prefill if e2e is not None and prefill is not None else None
        profiling_e2e = _number(profile.get("profiling_e2e_ms"))
        reference_e2e = _number(reference.get("e2e_ms")) if reference else None
        reference_prefill = _number(reference.get("prefill_ms")) if reference else None
        reference_decode = _number(reference.get("decode_ms")) if reference else None
        reference_post_prefill = (
            reference_e2e - reference_prefill
            if reference_e2e is not None and reference_prefill is not None else None
        )
        decode = _number(row.get("decode_ms"))
        target_dispatch_calls = extra.get(
            "target_attention_dispatch_calls", dispatch.get("target_attention_dispatch_calls")
        )
        draft_dispatch_calls = extra.get(
            "draft_attention_dispatch_calls", dispatch.get("draft_attention_dispatch_calls")
        )
        target_fallback_calls = extra.get(
            "target_fallback_attention_calls", dispatch.get("target_fallback_attention_calls")
        )
        draft_fallback_calls = extra.get(
            "draft_fallback_attention_calls", dispatch.get("draft_fallback_attention_calls")
        )
        eagle_tree = extra.get("eagle_tree") if isinstance(extra.get("eagle_tree"), dict) else {}

        flattened.append({
            "run_id": summary.get("run_id"),
            "method": row.get("method"),
            "dataset": row.get("dataset"),
            "sample_id": row.get("sample_id"),
            "repeat_index": row.get("repeat_index", 0),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "input_tokens": row.get("input_tokens"),
            "source_input_tokens": row.get("source_input_tokens"),
            "input_was_truncated": row.get("input_was_truncated"),
            "output_tokens": row.get("output_tokens"),
            "e2e_ms": e2e,
            "prefill_ms": prefill,
            "prefill_percent_of_e2e": round(prefill * 100.0 / e2e, 6)
            if prefill is not None and e2e is not None and e2e > 0 else None,
            "post_prefill_wall_ms": round(post_prefill, 6) if post_prefill is not None else None,
            "decode_ms_recorded": decode,
            "tpot_ms": _number(row.get("tpot_ms")),
            "throughput_tok_s": _number(row.get("throughput_tok_s")),
            "decode_throughput_tok_s": _number(row.get("decode_throughput_tok_s")),
            "draft_latency_ms": _number(row.get("draft_latency_ms")),
            "verification_latency_ms": _number(row.get("verification_latency_ms")),
            "selector_latency_ms": _number(row.get("selector_latency_ms")),
            "profiling_e2e_ms": profiling_e2e,
            "profiling_e2e_delta_vs_measured_ms": round(profiling_e2e - e2e, 6)
            if profiling_e2e is not None and e2e is not None else None,
            "profiling_ttft_ms": _number(profile.get("profiling_ttft_ms")),
            "phase_timing_mode": native_config.get("phase_timing_mode", summary.get("phase_timing_mode")),
            "phase_profile_source": profile.get("source"),
            "phase_profile_output_acceptance_match": profile.get("output_and_acceptance_match"),
            "skipped_profiling_synchronizations": extra.get("skipped_profiling_synchronizations"),
            "model_load_ms": _number(row.get("model_load_ms")),
            "queue_wait_ms": _number(row.get("queue_wait_ms")),
            "batch_wait_ms": _number(row.get("batch_wait_ms")),
            "server_startup_ms": _number(row.get("server_startup_ms")),
            "server_reported_e2e_ms": _number(row.get("server_reported_e2e_ms")),
            "peak_memory_gb": _number(row.get("peak_memory_gb")),
            "avg_accept_length": _number(row.get("avg_accept_length")),
            "acceptance_rate_percent": _number(row.get("acceptance_rate_percent")),
            "draft_tokens_accepted": row.get("draft_tokens_accepted"),
            "draft_tokens_proposed": row.get("draft_tokens_proposed"),
            "verification_steps": row.get("verification_steps"),
            "greedy_token_match": row.get("greedy_token_match"),
            "first_mismatch_token": row.get("first_mismatch_token"),
            "token_lcs_overlap_with_vanilla": _number(row.get("token_lcs_overlap_with_vanilla")),
            "quality_valid": row.get("quality_valid"),
            "attention_backend": row.get("attention_backend"),
            "target_attention": extra.get("target_attention", extra.get("target_attention_dispatch")),
            "draft_attention": extra.get("draft_attention", extra.get("draft_attention_dispatch")),
            "target_dispatch_calls": target_dispatch_calls,
            "target_fallback_attention_calls": target_fallback_calls,
            "draft_dispatch_calls": draft_dispatch_calls,
            "draft_fallback_attention_calls": draft_fallback_calls,
            "unattributed_attention_calls": json.dumps(
                dispatch.get("unattributed_attention_calls", extra.get("unattributed_attention_calls", {})),
                ensure_ascii=False, sort_keys=True,
            ),
            "dflash_block_size": extra.get("dflash_block_size"),
            "draft_sdpa_fallback_calls": extra.get("draft_sdpa_fallback_calls"),
            "eagle_tree_total_token": eagle_tree.get("total_token"),
            "eagle_tree_max_total_token": eagle_tree.get("max_total_token"),
            "measurement_scope": row.get("measurement_scope"),
            "vanilla_record_found": vanilla_record is not None,
            "vanilla_input_matches": input_matches,
            "paired_vanilla_e2e_ms": reference_e2e,
            "e2e_delta_vs_vanilla_ms": round(e2e - reference_e2e, 6)
            if e2e is not None and reference_e2e is not None else None,
            "vanilla_over_method_e2e_ratio": round(reference_e2e / e2e, 6)
            if reference_e2e is not None and e2e is not None and e2e > 0 else None,
            "paired_vanilla_prefill_ms": reference_prefill,
            "prefill_delta_vs_vanilla_ms": round(prefill - reference_prefill, 6)
            if prefill is not None and reference_prefill is not None else None,
            "paired_vanilla_post_prefill_wall_ms": round(reference_post_prefill, 6)
            if reference_post_prefill is not None else None,
            "post_prefill_delta_vs_vanilla_ms": round(post_prefill - reference_post_prefill, 6)
            if post_prefill is not None and reference_post_prefill is not None else None,
            "paired_vanilla_decode_ms": reference_decode,
            "decode_delta_vs_vanilla_ms": round(decode - reference_decode, 6)
            if decode is not None and reference_decode is not None else None,
            "paired_vanilla_output_tokens": reference.get("output_tokens") if reference else None,
        })
    return sorted(flattened, key=lambda row: (
        str(row.get("dataset", "")), str(row.get("sample_id", "")),
        int(row.get("repeat_index", 0)), str(row.get("method", "")),
    ))


def _component_summaries(component_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in component_rows:
        if row.get("dataset") is not None and row.get("method") is not None:
            groups[("all", str(row["method"]))].append(row)
            groups[(str(row["dataset"]), str(row["method"]))].append(row)

    result = []
    for (dataset, method), records in sorted(groups.items(), key=lambda item: (item[0][0] != "all", item[0])):
        successful = [row for row in records if row.get("status") == "success"]
        result.append({
            "dataset": dataset,
            "method": method,
            "records": len(records),
            "successful_records": len(successful),
            "failed_records": len(records) - len(successful),
            "phase_timing_mode_counts": dict(sorted(Counter(
                row["phase_timing_mode"] for row in successful if row.get("phase_timing_mode")
            ).items())),
            "phase_profile_source_counts": dict(sorted(Counter(
                row["phase_profile_source"] or "missing" for row in successful
            ).items())),
            "timings": {
                field: _distribution(row.get(field) for row in successful)
                for field in _COMPONENT_TIMINGS
            },
        })
    return result


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
    component_rows = _component_rows(summary, rows)
    result["component_summary"] = _component_summaries(component_rows)
    result["inference_components"] = component_rows
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
    def component_fmt(row, field):
        distribution = row["timings"][field]
        if not distribution["count"]:
            return "— (0)"
        return f"{fmt(distribution['mean'])} / {fmt(distribution['p90'])} ({distribution['count']})"

    lines += [
        "", "## Thời gian thành phần theo dataset và method", "",
        "Mỗi ô thời gian là mean / p90 (số lượt có số đo). Bảng chi tiết từng lượt nằm trong `inference_components.csv`; phân phối đầy đủ nằm trong `component_summary.csv`.",
        "",
        "- Trong native FA4 runner, `decode_ms` được tính từ `E2E - prefill`; `E2E - prefill` là phần wall time còn lại sau prefill proxy, không phải timer độc lập cho kernel decode.",
        "- `draft_latency_ms` và `verification_latency_ms` có thể đến từ lượt profiling riêng. Chỉ coi cặp pha là tương ứng khi `phase_profile_output_acceptance_match=true`; không cộng chúng vào E2E. Domino/DSpark có thể không cung cấp timer pha native.",
        "- `model_load_ms` được giữ trong CSV để đối chiếu nhưng nằm ngoài `e2e_ms` của lượt infer.",
        "",
        "| Dataset | Method | Thành công / records | E2E mean / p90 ms (n) | Prefill mean / p90 ms (n) | E2E−prefill ms (n) | decode_ms ghi nhận (n) | Draft profile ms (n) | Verify profile ms (n) | Nguồn phase profile |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result.get("component_summary", []):
        dataset = "Tổng thể" if row["dataset"] == "all" else row["dataset"]
        sources = ", ".join(
            f"{name}:{count}" for name, count in row["phase_profile_source_counts"].items()
        ) or "—"
        lines.append(
            f"| {dataset} | {row['method']} | {row['successful_records']}/{row['records']} "
            f"| {component_fmt(row, 'e2e_ms')} | {component_fmt(row, 'prefill_ms')} "
            f"| {component_fmt(row, 'post_prefill_wall_ms')} | {component_fmt(row, 'decode_ms_recorded')} "
            f"| {component_fmt(row, 'draft_latency_ms')} | {component_fmt(row, 'verification_latency_ms')} | {sources} |"
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
    inference_components = result.get("inference_components", [])
    if inference_components:
        with (destination / "inference_components.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(inference_components[0]))
            writer.writeheader()
            writer.writerows(inference_components)

    component_summaries = result.get("component_summary", [])
    if component_summaries:
        summary_fields = [
            "dataset", "method", "records", "successful_records", "failed_records",
            "phase_timing_mode_counts", "phase_profile_source_counts",
        ]
        summary_fields += [
            f"{field}_{stat}"
            for field in _COMPONENT_TIMINGS
            for stat in ("count", "mean", "median", "p90", "min", "max")
        ]
        with (destination / "component_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            for row in component_summaries:
                flattened = {
                    key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                    if isinstance(row[key], dict) else row[key]
                    for key in summary_fields[:7]
                }
                for field in _COMPONENT_TIMINGS:
                    for stat in ("count", "mean", "median", "p90", "min", "max"):
                        flattened[f"{field}_{stat}"] = row["timings"][field][stat]
                writer.writerow(flattened)

    # Keep per-inference observations in CSV; diagnostics.json remains an aggregate summary.
    result.pop("inference_components", None)
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
