"""Pure-Python selection and aggregation helpers for the native FA4 runner."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from Benchmark.common.metrics import aggregate_semantic, aggregate_speed, aggregate_speculative
from Benchmark.common.rouge import aggregate_rouge


def parse_selection(value: str | Iterable[str], choices: Sequence[str], label: str) -> tuple[str, ...]:
    """Parse a comma/space separated selection, preserving canonical order."""
    if isinstance(value, str):
        requested = [part for part in value.replace(",", " ").split() if part]
    else:
        requested = [str(part).strip() for part in value if str(part).strip()]
    if not requested:
        raise ValueError(f"{label} selection cannot be empty")
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError(f"'all' cannot be combined with named {label}")
        return tuple(choices)
    duplicates = sorted({item for item in requested if requested.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}: {duplicates}")
    unknown = [item for item in requested if item not in choices]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; allowed: {list(choices)}")
    selected = set(requested)
    return tuple(item for item in choices if item in selected)


def resolve_sample_limit(mode: str, requested: int | None, *, available: int) -> int:
    """Resolve per-dataset sample cap; zero means every eligible sample."""
    defaults = {"smoke": 2, "representative": 20, "full": 0}
    if mode not in defaults:
        raise ValueError(f"unknown benchmark mode: {mode}")
    limit = defaults[mode] if requested is None else int(requested)
    if limit < 0:
        raise ValueError("samples-per-dataset must be >= 0 (0 means all)")
    if limit == 0:
        return int(available)
    return min(limit, int(available))


def select_length_spread(
    rows: Sequence[Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Select deterministic length quantiles, including short and long rows."""
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (int(row.get("input_tokens", 0)), str(row.get("sample_id", ""))),
    )
    count = int(limit)
    if count < 0:
        raise ValueError("sample limit must be >= 0")
    if count == 0 or count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[(len(ordered) - 1) // 2]]
    # Integer interpolation is deterministic and does not require NumPy.
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def token_lcs_overlap(reference_ids: Sequence[int], candidate_ids: Sequence[int]) -> float | None:
    """Ordered-token LCS divided by the reference token count."""
    if not reference_ids:
        return None
    left = list(reference_ids)
    right = list(candidate_ids)
    if len(right) > len(left):
        left, right = right, left
    row = [0] * (len(right) + 1)
    for token_left in left:
        previous = 0
        for index, token_right in enumerate(right, start=1):
            saved = row[index]
            row[index] = previous + 1 if token_left == token_right else max(row[index], row[index - 1])
            previous = saved
    return row[-1] / len(reference_ids)


def annotate_greedy_parity(
    records: list[dict[str, Any]], *, reference_method: str = "vanilla_hf"
) -> dict[str, dict[str, Any]]:
    """Attach exact-match and token-LCS results to rows paired by sample/repeat."""
    reference_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in records:
        if row.get("method") != reference_method or row.get("status") != "success":
            continue
        key = (str(row.get("dataset", "")), str(row.get("sample_id", "")), int(row.get("repeat_index", 0)))
        reference_rows[key] = row

    output: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"compared_samples": 0, "exact_matches": 0, "lcs_numerator": 0, "lcs_denominator": 0}
    )
    for row in records:
        if row.get("status") != "success":
            continue
        key = (str(row.get("dataset", "")), str(row.get("sample_id", "")), int(row.get("repeat_index", 0)))
        reference = reference_rows.get(key)
        if reference is None:
            continue
        reference_ids = [int(token) for token in reference.get("output_token_ids", [])]
        candidate_ids = [int(token) for token in row.get("output_token_ids", [])]
        exact = reference_ids == candidate_ids
        overlap = token_lcs_overlap(reference_ids, candidate_ids)
        row["greedy_token_match"] = exact
        row["token_lcs_overlap_with_vanilla"] = overlap
        row["first_mismatch_token"] = first_token_mismatch(reference_ids, candidate_ids)
        stats = output[str(row["method"])]
        stats["compared_samples"] += 1
        stats["exact_matches"] += int(exact)
        if overlap is not None:
            stats["lcs_numerator"] += overlap * len(reference_ids)
            stats["lcs_denominator"] += len(reference_ids)

    return {
        method: {
            "compared_samples": value["compared_samples"],
            "exact_matches": value["exact_matches"],
            "exact_match_rate": (
                round(value["exact_matches"] / value["compared_samples"], 6)
                if value["compared_samples"] else None
            ),
            "token_lcs_overlap": (
                round(value["lcs_numerator"] / value["lcs_denominator"], 6)
                if value["lcs_denominator"] else None
            ),
        }
        for method, value in output.items()
    }


def first_token_mismatch(reference_ids: Sequence[int], candidate_ids: Sequence[int]) -> int | None:
    for index, (reference, candidate) in enumerate(zip(reference_ids, candidate_ids)):
        if int(reference) != int(candidate):
            return index
    if len(reference_ids) != len(candidate_ids):
        return min(len(reference_ids), len(candidate_ids))
    return None


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return round(statistics.mean(clean), 6) if clean else None


def _paired_metrics(
    rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_by_key = {
        (str(row.get("dataset", "")), str(row.get("sample_id", "")), int(row.get("repeat_index", 0))): row
        for row in reference_rows
        if row.get("status") == "success"
    }
    paired = []
    for row in rows:
        if row.get("status") != "success":
            continue
        key = (str(row.get("dataset", "")), str(row.get("sample_id", "")), int(row.get("repeat_index", 0)))
        reference = reference_by_key.get(key)
        if reference is None:
            continue
        paired.append((reference, row))
    common = [
        (reference, row)
        for reference, row in paired
        if all(
            value is not None and float(value) > 0
            for value in (reference.get("prefill_ms"), reference.get("tpot_ms"), row.get("tpot_ms"))
        )
        and int(reference.get("output_tokens") or 0) > 0
        and int(row.get("output_tokens") or 0) > 0
    ]
    reference_prefill = _mean([float(reference["prefill_ms"]) for reference, _ in common])
    reference_tpot = _mean([float(reference["tpot_ms"]) for reference, _ in common])
    method_tpot = _mean([float(row["tpot_ms"]) for _, row in common])
    min_tokens = _mean([
        min(int(reference["output_tokens"]), int(row["output_tokens"]))
        for reference, row in common
    ])
    dsr = reference_tpot / method_tpot if reference_tpot and method_tpot else None
    esr = None
    if reference_prefill and reference_tpot and method_tpot and min_tokens:
        esr = (reference_prefill + reference_tpot * min_tokens) / (
            reference_prefill + method_tpot * min_tokens
        )
    e2e_pairs = [
        (float(reference["e2e_ms"]), float(row["e2e_ms"]))
        for reference, row in paired
        if reference.get("e2e_ms") and row.get("e2e_ms")
        and float(reference["e2e_ms"]) > 0 and float(row["e2e_ms"]) > 0
    ]
    return {
        "paired_samples": len(paired),
        "paired_timing_samples": len(common),
        "reference_prefill_ms": reference_prefill,
        "reference_mean_tpot_ms": reference_tpot,
        "method_mean_tpot_ms": method_tpot,
        "mean_min_output_tokens": min_tokens,
        "dsr": round(dsr, 6) if dsr is not None else None,
        "esr": round(esr, 6) if esr is not None else None,
        "mean_paired_e2e_speedup": _mean([reference / method for reference, method in e2e_pairs]),
        "ratio_of_mean_e2e": (
            round(_mean([reference for reference, _ in e2e_pairs]) / _mean([method for _, method in e2e_pairs]), 6)
            if e2e_pairs and _mean([method for _, method in e2e_pairs]) else None
        ),
    }


def _summarize_group(
    records: Sequence[dict[str, Any]],
    *,
    method: str,
    reference_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    successful = [row for row in records if row.get("status") == "success"]
    rouge = aggregate_rouge(successful)
    return {
        "method": method,
        "samples": len(records),
        "successful_samples": len(successful),
        "failed_samples": sum(row.get("status") != "success" for row in records),
        "quality_valid_outputs": sum(bool(row.get("quality_valid")) for row in successful),
        "quality_valid_rate": (
            round(sum(bool(row.get("quality_valid")) for row in successful) / len(successful), 6)
            if successful else None
        ),
        "repetition_flags": sum(bool(row.get("repetition_flag")) for row in successful),
        **{f"mean_{key}": value for key, value in rouge.items()},
        "semantic_metrics": aggregate_semantic(successful),
        "speed_statistics": aggregate_speed(successful),
        "speculative_statistics": aggregate_speculative(successful),
        "mean_acceptance_rate": _mean([
            float(row["acceptance_rate"]) for row in successful
            if row.get("acceptance_rate") is not None
        ]),
        "mean_acceptance_rate_percent": _mean([
            float(row["acceptance_rate_percent"]) for row in successful
            if row.get("acceptance_rate_percent") is not None
        ]),
        "mean_avg_accept_length": _mean([
            float(row["avg_accept_length"]) for row in successful
            if row.get("avg_accept_length") is not None
        ]),
        "total_draft_tokens_accepted": sum(
            int(row["draft_tokens_accepted"]) for row in successful
            if row.get("draft_tokens_accepted") is not None
        ),
        "total_draft_tokens_proposed": sum(
            int(row["draft_tokens_proposed"]) for row in successful
            if row.get("draft_tokens_proposed") is not None
        ),
        "paired_speed_metrics": _paired_metrics(successful, reference_rows),
    }


def aggregate_fa4_records(
    records: list[dict[str, Any]],
    *,
    methods: Sequence[str],
    datasets: Sequence[str] | None = None,
    reference_method: str = "vanilla_hf",
) -> dict[str, Any]:
    """Aggregate paired latency, speed, quality, and speculative metrics."""
    parity = annotate_greedy_parity(records, reference_method=reference_method)
    dataset_names = tuple(datasets or sorted({str(row.get("dataset")) for row in records}))
    metrics_by_dataset: dict[str, dict[str, Any]] = {}
    overall: dict[str, Any] = {}
    for dataset in dataset_names:
        subset = [row for row in records if row.get("dataset") == dataset]
        ref_rows = [row for row in subset if row.get("method") == reference_method]
        metrics_by_dataset[dataset] = {}
        for method in methods:
            method_rows = [row for row in subset if row.get("method") == method]
            method_metrics = _summarize_group(
                method_rows, method=method, reference_rows=ref_rows
            )
            compared = [
                row for row in method_rows
                if isinstance(row.get("greedy_token_match"), bool)
            ]
            exact = sum(row["greedy_token_match"] is True for row in compared)
            lcs_numerator = sum(
                float(row["token_lcs_overlap_with_vanilla"])
                * int(next(
                    (
                        ref.get("output_tokens", 0)
                        for ref in ref_rows
                        if str(ref.get("sample_id")) == str(row.get("sample_id"))
                        and int(ref.get("repeat_index", 0)) == int(row.get("repeat_index", 0))
                    ),
                    0,
                ))
                for row in compared
                if row.get("token_lcs_overlap_with_vanilla") is not None
            )
            lcs_denominator = sum(
                int(ref.get("output_tokens", 0))
                for ref in ref_rows
                if any(
                    str(ref.get("sample_id")) == str(row.get("sample_id"))
                    and int(ref.get("repeat_index", 0)) == int(row.get("repeat_index", 0))
                    for row in compared
                )
            )
            method_metrics.update(
                {
                    "greedy_exact_matches": exact,
                    "greedy_compared_samples": len(compared),
                    "greedy_exact_match_rate": round(exact / len(compared), 6)
                    if compared else None,
                    "token_lcs_overlap_with_vanilla": round(
                        lcs_numerator / lcs_denominator, 6
                    ) if lcs_denominator else None,
                }
            )
            metrics_by_dataset[dataset][method] = method_metrics
    reference_rows = [row for row in records if row.get("method") == reference_method]
    for method in methods:
        method_rows = [row for row in records if row.get("method") == method]
        overall[method] = _summarize_group(
            method_rows, method=method, reference_rows=reference_rows
        )
        overall[method]["greedy_exact_matches"] = parity.get(method, {}).get("exact_matches", 0)
        overall[method]["greedy_compared_samples"] = parity.get(method, {}).get("compared_samples", 0)
        overall[method]["greedy_exact_match_rate"] = parity.get(method, {}).get("exact_match_rate")
        overall[method]["token_lcs_overlap_with_vanilla"] = parity.get(method, {}).get("token_lcs_overlap")
    return {
        "method_metrics": overall,
        "metrics_by_dataset": metrics_by_dataset,
        "parity": parity,
    }


def finalize_fa4_records(
    records: Sequence[dict[str, Any]],
    *,
    methods: Sequence[str],
    datasets: Sequence[str],
    repetitions: int,
    expected_samples: int,
    reference_method: str = "vanilla_hf",
) -> dict[str, Any]:
    """Deduplicate resumed cells, annotate pairs, and compute run gates."""
    latest: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in records:
        key = (
            str(row.get("method", "")),
            str(row.get("dataset", "")),
            str(row.get("sample_id", "")),
            int(row.get("repeat_index", 0)),
        )
        latest[key] = row
    canonical_records = list(latest.values())
    parity = annotate_greedy_parity(
        canonical_records, reference_method=reference_method
    )
    references = {
        (str(row.get("dataset")), str(row.get("sample_id")), int(row.get("repeat_index", 0))): row
        for row in canonical_records
        if row.get("method") == reference_method and row.get("status") == "success"
    }
    for row in canonical_records:
        if row.get("status") != "success":
            continue
        key = (str(row.get("dataset")), str(row.get("sample_id")), int(row.get("repeat_index", 0)))
        reference = references.get(key)
        if reference is None:
            continue
        reference_ms = float(reference.get("e2e_ms") or 0.0)
        method_ms = float(row.get("e2e_ms") or 0.0)
        reference_tps = float(reference.get("throughput_tok_s") or 0.0)
        method_tps = float(row.get("throughput_tok_s") or 0.0)
        extra = row.setdefault("extra_metrics", {})
        extra.update(
            {
                "paired_latency_speedup": round(reference_ms / method_ms, 6)
                if reference_ms > 0 and method_ms > 0 else None,
                "paired_throughput_speedup": round(method_tps / reference_tps, 6)
                if reference_tps > 0 and method_tps > 0 else None,
                "output_token_ids_match_reference_vanilla": reference.get("output_token_ids")
                == row.get("output_token_ids"),
            }
        )

    repeat_outputs: dict[tuple[str, str, str], list[list[int]]] = defaultdict(list)
    for row in canonical_records:
        if row.get("status") == "success":
            repeat_outputs[(str(row.get("method")), str(row.get("dataset")), str(row.get("sample_id")))].append(
                [int(token) for token in row.get("output_token_ids", [])]
            )
    for row in canonical_records:
        if row.get("status") == "success":
            key = (str(row.get("method")), str(row.get("dataset")), str(row.get("sample_id")))
            outputs = repeat_outputs[key]
            row.setdefault("extra_metrics", {})["repeat_greedy_stable"] = (
                len(outputs) <= 1 or all(output == outputs[0] for output in outputs[1:])
            )

    bundle = aggregate_fa4_records(
        canonical_records,
        methods=methods,
        datasets=datasets,
        reference_method=reference_method,
    )
    expected = len(methods) * int(expected_samples) * int(repetitions)
    execution_complete = len(canonical_records) == expected
    execution_pass = execution_complete and all(
        row.get("status") == "success" for row in canonical_records
    )
    quality_pass = execution_pass and all(
        row.get("quality_valid") is True
        for row in canonical_records
        if row.get("status") == "success"
    )
    exact_match_by_method = {
        method: (
            bundle["parity"].get(method, {}).get("compared_samples", 0)
            == int(expected_samples) * int(repetitions)
            and bundle["parity"].get(method, {}).get("exact_matches", 0)
            == int(expected_samples) * int(repetitions)
        )
        for method in methods
    }
    exact_match_all = all(exact_match_by_method.values())
    speedup_by_method = {
        method: bundle["method_metrics"].get(method, {})
        .get("paired_speed_metrics", {}).get("esr")
        for method in methods
        if method != reference_method
    }
    speedup_all_over_one = all(
        value is not None and float(value) > 1.0
        for value in speedup_by_method.values()
    )
    return {
        "records": canonical_records,
        **bundle,
        "expected_records": expected,
        "execution_complete": execution_complete,
        "execution_pass": execution_pass,
        "quality_pass": quality_pass,
        "exact_match_by_method": exact_match_by_method,
        "exact_match_all": exact_match_all,
        "speedup_by_method": speedup_by_method,
        "speedup_all_over_one": speedup_all_over_one,
        "failure_count": sum(row.get("status") != "success" for row in canonical_records),
    }
