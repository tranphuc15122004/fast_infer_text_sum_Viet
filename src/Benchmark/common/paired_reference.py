"""Paired, versioned speedup metrics for LongBench baseline observations.

The estimator intentionally treats timing and output quality as separate
signals. A text mismatch does not remove a successful latency pair.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from Benchmark.common.quality_guard import is_degenerate_output


COMMON_REFERENCE_DEFAULT = "vanilla_fa"
PAPER_BASELINES = ("vanilla_hf", "vanilla_fa", "eagle3", "dflash", "domino", "dspark")
PAPER_DATASETS = ("vietnews", "wikilingua", "vims", "vlsp")


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def _counter(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], set[str], Counter]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        sid = row.get("sample_id", row.get("id"))
        if sid is None:
            continue
        grouped.setdefault(str(sid), []).append(row)
    duplicates = {sid for sid, values in grouped.items() if len(values) != 1}
    indexed = {sid: values[0] for sid, values in grouped.items() if len(values) == 1}
    duplicate_counts = Counter({"duplicate_sample_id": sum(len(grouped[sid]) for sid in duplicates)})
    if not duplicates:
        duplicate_counts.clear()
    return indexed, duplicates, duplicate_counts


def _pair_identity_issue(reference: Mapping[str, Any], method: Mapping[str, Any], *, scope: str) -> str | None:
    if reference.get("contract_version") != 2 or method.get("contract_version") != 2:
        return "contract_version_mismatch"
    required_identity = (
        "dataset", "prompt_token_sha256", "generation_config_sha256",
        "target_revision", "tokenizer_revision", "hardware_fingerprint",
        "gpu_count", "tp_size", "batch_size", "concurrency", "cache_policy",
    )
    for required in required_identity:
        if reference.get(required) is None or method.get(required) is None:
            return "missing_identity"
    if scope == "native" and (
        reference.get("runtime_config_sha256") is None
        or method.get("runtime_config_sha256") is None
        or reference.get("native_timing_scope") is None
        or method.get("native_timing_scope") is None
    ):
        return "missing_identity"
    reference_run = reference.get("run_id")
    method_run = method.get("run_id")
    if reference_run is not None and method_run is not None and reference_run != method_run:
        return "config_mismatch"
    ref_input_tokens = reference.get("actual_input_tokens")
    method_input_tokens = method.get("actual_input_tokens")
    if ref_input_tokens is None or method_input_tokens is None:
        return "missing_identity"
    if ref_input_tokens != method_input_tokens:
        return "prompt_mismatch"
    for key in ("dataset",):
        left, right = reference.get(key), method.get(key)
        if left is not None and right is not None and str(left) != str(right):
            return "config_mismatch"

    prompt_a = reference.get("prompt_token_sha256")
    prompt_b = method.get("prompt_token_sha256")
    if prompt_a is not None and prompt_b is not None and prompt_a != prompt_b:
        return "prompt_mismatch"

    config_keys = (
        "pairing_config_sha256",
        "generation_config_sha256",
        "target_checkpoint_sha256",
        "target_revision",
        "tokenizer_revision",
    )
    for key in config_keys:
        left, right = reference.get(key), method.get(key)
        if left is not None and right is not None and left != right:
            return "config_mismatch"
    if scope == "native":
        left, right = reference.get("runtime_config_sha256"), method.get("runtime_config_sha256")
        if left is not None and right is not None and left != right:
            return "config_mismatch"

    resource_keys = (
        "hardware_fingerprint",
        "gpu_count",
        "tp_size",
        "batch_size",
        "concurrency",
        "cache_policy",
    )
    for key in resource_keys:
        left, right = reference.get(key), method.get(key)
        if left is not None and right is not None and left != right:
            return "unequal_resource"

    if scope == "native":
        left = reference.get("native_timing_scope")
        right = method.get("native_timing_scope")
        if left is not None and right is not None and left != right:
            return "config_mismatch"
    return None


def _status_issue(reference: Mapping[str, Any], method: Mapping[str, Any]) -> str | None:
    if reference.get("status", "success") != "success" or method.get("status", "success") != "success":
        return "failed_status"
    return None


def _paired_rows(
    reference_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> tuple[list[tuple[str, Mapping[str, Any], Mapping[str, Any]]], Counter]:
    ref, ref_dupes, ref_duplicate_counts = _index_rows(reference_rows)
    method, method_dupes, method_duplicate_counts = _index_rows(method_rows)
    exclusions = Counter(ref_duplicate_counts)
    exclusions.update(method_duplicate_counts)
    pairs = []
    for sid in sorted(set(ref) | set(method) | ref_dupes | method_dupes):
        if sid in ref_dupes or sid in method_dupes:
            continue
        if sid not in ref:
            exclusions["missing_reference"] += 1
            continue
        if sid not in method:
            exclusions["missing_method"] += 1
            continue
        issue = _status_issue(ref[sid], method[sid]) or _pair_identity_issue(ref[sid], method[sid], scope=scope)
        if issue:
            exclusions[issue] += 1
            continue
        pairs.append((sid, ref[sid], method[sid]))
    return pairs, exclusions


def _has_direct_decode_duration(row: Mapping[str, Any]) -> bool:
    value = row.get("decode_ms")
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _metric_pairs(
    pairs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    reference_field: str,
    method_field: str,
    kind: str,
) -> tuple[list[tuple[str, float, float]], Counter]:
    values: list[tuple[str, float, float]] = []
    exclusions: Counter = Counter()
    for sid, reference, method in pairs:
        if kind.startswith("decode"):
            direct_evidence_present = (
                _has_direct_decode_duration(reference)
                and _has_direct_decode_duration(method)
            )
            if not direct_evidence_present:
                exclusions["missing_decode_evidence"] += 1
                continue
            definitions_match = (
                reference.get("decode_phase_verified") is True
                and method.get("decode_phase_verified") is True
                and reference.get("decode_phase_definition") is not None
                and reference.get("decode_phase_definition") == method.get("decode_phase_definition")
            )
            if not definitions_match:
                exclusions["phase_unverified"] += 1
                continue
        if kind == "decode_rate":
            ref_count = _counter(reference.get("decode_token_count"))
            method_count = _counter(method.get("decode_token_count"))
            if ref_count is None or method_count is None:
                exclusions["missing_decode_tokens"] += 1
                continue
            if ref_count == 0 or method_count == 0:
                exclusions["zero_decode_tokens"] += 1
                continue
        else:
            ref_count = method_count = None

        ref_value = _finite_positive(reference.get(reference_field))
        method_value = _finite_positive(method.get(method_field))
        if ref_value is None or method_value is None:
            raw_ref = reference.get(reference_field)
            raw_method = method.get(method_field)
            if any(isinstance(v, (float, int)) and not isinstance(v, bool) and not math.isfinite(float(v)) for v in (raw_ref, raw_method)):
                exclusions["nonfinite_time"] += 1
            elif raw_ref is None or raw_method is None:
                exclusions["missing_time"] += 1
            else:
                exclusions["nonpositive_time"] += 1
            continue
        if kind == "decode_rate":
            values.append((sid, float(ref_count), float(method_count)))
        elif kind == "output_rate":
            ref_count = _counter(reference.get("timed_generated_tokens"))
            method_count = _counter(method.get("timed_generated_tokens"))
            if ref_count is None or method_count is None:
                exclusions["missing_timed_tokens"] += 1
                continue
            values.append((sid, float(ref_count), float(method_count)))
        else:
            values.append((sid, ref_value, method_value))
    return values, exclusions

def _ratio_of_sums(values: Sequence[tuple[str, float, float]]) -> float | None:
    if not values:
        return None
    denominator = sum(method for _, _, method in values)
    numerator = sum(reference for _, reference, _ in values)
    if denominator <= 0 or not math.isfinite(denominator) or not math.isfinite(numerator):
        return None
    return numerator / denominator


def _rate_ratio(
    values: Sequence[tuple[str, float, float]],
    pairs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    time_field: str,
) -> float | None:
    if not values:
        return None
    by_id = {sid: (reference, method) for sid, reference, method in pairs}
    reference_tokens = sum(reference for _, reference, _ in values)
    method_tokens = sum(method for _, _, method in values)
    reference_ms = sum(_finite_positive(by_id[sid][0].get(time_field)) or 0.0 for sid, _, _ in values)
    method_ms = sum(_finite_positive(by_id[sid][1].get(time_field)) or 0.0 for sid, _, _ in values)
    if reference_tokens <= 0 or method_tokens <= 0 or reference_ms <= 0 or method_ms <= 0:
        return None
    return (method_tokens / method_ms) / (reference_tokens / reference_ms)


def _row_is_degenerate(row: Mapping[str, Any]) -> bool:
    guard = row.get("output_quality_guard")
    text = row.get("text") or row.get("answer") or ""
    return bool(
        row.get("degenerate_repetition")
        or (isinstance(guard, Mapping) and guard.get("degenerate_repetition"))
        or is_degenerate_output(str(text))
    )


def _quality_summary(pairs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    matched = 0
    compared = 0
    token_matched = 0
    token_compared = 0
    token_ratios = []
    method_lengths = []
    reference_lengths = []
    method_degenerate = 0
    reference_degenerate = 0
    text_quality_pairs = 0
    for _, reference, method in pairs:
        ref_text, method_text = reference.get("text"), method.get("text")
        if isinstance(ref_text, str) and isinstance(method_text, str):
            compared += 1
            matched += int(ref_text == method_text)
            text_quality_pairs += 1
            method_degenerate += int(_row_is_degenerate(method))
            reference_degenerate += int(_row_is_degenerate(reference))
        ref_tokens = _counter(reference.get("visible_output_tokens", reference.get("output_tokens")))
        method_tokens = _counter(method.get("visible_output_tokens", method.get("output_tokens")))
        if ref_tokens is not None and method_tokens is not None:
            token_compared += 1
            token_matched += int(ref_tokens == method_tokens)
            reference_lengths.append(ref_tokens)
            method_lengths.append(method_tokens)
            if ref_tokens > 0:
                token_ratios.append(method_tokens / ref_tokens)
    return {
        "valid_pair_count": len(pairs),
        "text_comparison_count": compared,
        "exact_text_match_rate": matched / compared if compared else None,
        "token_count_comparison_count": token_compared,
        "exact_token_count_match_rate": token_matched / token_compared if token_compared else None,
        "mean_visible_token_ratio": sum(token_ratios) / len(token_ratios) if token_ratios else None,
        "mean_method_visible_output_tokens": sum(method_lengths) / len(method_lengths) if method_lengths else None,
        "mean_reference_visible_output_tokens": sum(reference_lengths) / len(reference_lengths) if reference_lengths else None,
        "text_quality_pair_count": text_quality_pairs,
        "method_degenerate_output_count": method_degenerate,
        "method_degenerate_output_rate": method_degenerate / text_quality_pairs if text_quality_pairs else None,
        "reference_degenerate_output_count": reference_degenerate,
        "reference_degenerate_output_rate": reference_degenerate / text_quality_pairs if text_quality_pairs else None,
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value) and value >= 0)
    if not clean:
        return None
    position = (len(clean) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    return clean[lower] * (upper - position) + clean[upper] * (position - lower)


def aggregate_pair(
    reference_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    *,
    scope: str = "common",
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Aggregate one reference/method pair using paired ratios of sums."""
    if scope not in {"common", "native"}:
        raise ValueError("scope must be 'common' or 'native'")
    pairs, join_exclusions = _paired_rows(reference_rows, method_rows, scope=scope)
    prefix = f"{scope}_"
    time_field = "request_wall_ms" if scope == "common" else "native_elapsed_ms"

    e2e_pairs, e2e_exclusions = _metric_pairs(
        pairs, reference_field=time_field, method_field=time_field, kind="time"
    )
    decode_pairs, decode_exclusions = _metric_pairs(
        pairs, reference_field="decode_active_ms", method_field="decode_active_ms", kind="decode_time"
    )
    decode_rate_pairs, decode_rate_exclusions = _metric_pairs(
        pairs, reference_field="decode_active_ms", method_field="decode_active_ms", kind="decode_rate"
    )
    quality_pairs = list(pairs)

    e2e = _ratio_of_sums(e2e_pairs)
    decode_time = _ratio_of_sums(decode_pairs)
    decode_rate = _rate_ratio(decode_rate_pairs, pairs, time_field="decode_active_ms")
    result: dict[str, Any] = {
        f"{prefix}esr": round(e2e, 8) if e2e is not None else None,
        f"{prefix}esr_n": len(e2e_pairs),
        f"{prefix}esr_valid_sample_ids": [sid for sid, _, _ in e2e_pairs],
        f"{prefix}esr_exclusions": dict(sorted((join_exclusions + e2e_exclusions).items())),
        f"{prefix}decode_time_ratio": round(decode_time, 8) if decode_time is not None else None,
        f"{prefix}decode_time_n": len(decode_pairs),
        f"{prefix}decode_time_valid_sample_ids": [sid for sid, _, _ in decode_pairs],
        f"{prefix}decode_time_exclusions": dict(sorted((join_exclusions + decode_exclusions).items())),
        f"{prefix}decode_rate_ratio": round(decode_rate, 8) if decode_rate is not None else None,
        f"{prefix}decode_rate_n": len(decode_rate_pairs),
        f"{prefix}decode_rate_valid_sample_ids": [sid for sid, _, _ in decode_rate_pairs],
        f"{prefix}decode_rate_exclusions": dict(sorted((join_exclusions + decode_rate_exclusions).items())),
        f"{prefix}quality": _quality_summary(quality_pairs),
    }
    metric_seed = {"common": 0, "native": 1000}[scope]
    _add_ci_fields(result, f"{prefix}esr", paired_bootstrap_ci([(ref, method) for _, ref, method in e2e_pairs], seed=20260928 + metric_seed, resamples=bootstrap_resamples))
    _add_ci_fields(result, f"{prefix}decode_time_ratio", paired_bootstrap_ci([(ref, method) for _, ref, method in decode_pairs], seed=20260929 + metric_seed, resamples=bootstrap_resamples))
    _add_ci_fields(result, f"{prefix}decode_rate_ratio", paired_rate_bootstrap_ci(_rate_observations(decode_rate_pairs, pairs, time_field="decode_active_ms"), seed=20260930 + metric_seed, resamples=bootstrap_resamples))
    if scope == "common":
        output_rate_pairs, output_rate_exclusions = _metric_pairs(
            pairs, reference_field="request_wall_ms", method_field="request_wall_ms", kind="output_rate"
        )
        rate = _rate_ratio(output_rate_pairs, pairs, time_field="request_wall_ms")
        result.update(
            common_output_rate_ratio=round(rate, 8) if rate is not None else None,
            common_output_rate_n=len(output_rate_pairs),
            common_output_rate_valid_sample_ids=[sid for sid, _, _ in output_rate_pairs],
            common_output_rate_exclusions=dict(sorted((join_exclusions + output_rate_exclusions).items())),
        )
        _add_ci_fields(result, "common_output_rate_ratio", paired_rate_bootstrap_ci(_rate_observations(output_rate_pairs, pairs, time_field="request_wall_ms"), seed=20260931, resamples=bootstrap_resamples))
    return result

def paired_bootstrap_ci(
    pairs: Sequence[tuple[float, float]],
    *,
    seed: int = 20260928,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile CI for ratio-of-sums, resampling paired sample IDs."""
    clean = []
    for reference, method in pairs:
        ref_value, method_value = _finite_positive(reference), _finite_positive(method)
        if ref_value is not None and method_value is not None:
            clean.append((ref_value, method_value))
    if not clean:
        return None
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("confidence must be between 0 and 1 and resamples must be positive")
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        denominator = sum(method for _, method in sample)
        estimates.append(sum(reference for reference, _ in sample) / denominator)
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, min(len(estimates) - 1, math.floor(alpha * len(estimates))))
    upper_index = max(0, min(len(estimates) - 1, math.ceil((1.0 - alpha) * len(estimates)) - 1))
    return estimates[lower_index], estimates[upper_index]


def paired_rate_bootstrap_ci(
    observations: Sequence[tuple[float, float, float, float]],
    *,
    seed: int = 20260928,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile CI for a paired ratio of token rates.

    Each observation is ``(reference_tokens, method_tokens, reference_ms,
    method_ms)`` for one shared sample ID.
    """
    clean = []
    for ref_tokens, method_tokens, ref_ms, method_ms in observations:
        values = tuple(_finite_positive(value) for value in (ref_tokens, method_tokens, ref_ms, method_ms))
        if all(value is not None for value in values):
            clean.append(values)
    if not clean:
        return None
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("confidence must be between 0 and 1 and resamples must be positive")
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        reference_rate = sum(row[0] for row in sample) / sum(row[2] for row in sample)
        method_rate = sum(row[1] for row in sample) / sum(row[3] for row in sample)
        estimates.append(method_rate / reference_rate)
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return (
        estimates[max(0, min(len(estimates) - 1, math.floor(alpha * len(estimates))))],
        estimates[max(0, min(len(estimates) - 1, math.ceil((1.0 - alpha) * len(estimates)) - 1))],
    )


def _rate_observations(
    values: Sequence[tuple[str, float, float]],
    pairs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    time_field: str,
) -> list[tuple[float, float, float, float]]:
    by_id = {sid: (reference, method) for sid, reference, method in pairs}
    observations = []
    for sid, reference_tokens, method_tokens in values:
        reference_ms = _finite_positive(by_id[sid][0].get(time_field))
        method_ms = _finite_positive(by_id[sid][1].get(time_field))
        if reference_ms is not None and method_ms is not None:
            observations.append((reference_tokens, method_tokens, reference_ms, method_ms))
    return observations


def _add_ci_fields(result: dict[str, Any], name: str, interval: tuple[float, float] | None) -> None:
    result[f"{name}_ci95_low"] = round(interval[0], 8) if interval else None
    result[f"{name}_ci95_high"] = round(interval[1], 8) if interval else None


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    """Hash an ordered token sequence with a stable, versioned binary format."""
    values = [int(value) for value in token_ids]
    if any(value < 0 or value > 0xFFFFFFFF for value in values):
        raise ValueError("token IDs must fit unsigned 32-bit integers")
    payload = b"fast-infer-token-ids-v1\0" + len(values).to_bytes(8, "big")
    payload += b"".join(value.to_bytes(4, "big") for value in values)
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_v2_record_fields(
    *,
    prompt_token_ids: Iterable[int],
    generation_config: Mapping[str, Any],
    hardware: Mapping[str, Any],
    request_wall_ms: Any = None,
    native_elapsed_ms: Any = None,
    native_timing_scope: str | None = None,
    timed_generated_tokens: Any = None,
    visible_output_tokens: Any = None,
    decode_active_ms: Any = None,
    decode_token_count: Any = None,
    decode_phase_definition: str | None = None,
    decode_phase_verified: bool = False,
    timing_source: str | None = None,
    target_revision: str | None = None,
    tokenizer_revision: str | None = None,
    gpu_count: int | None = None,
    tp_size: int = 1,
    batch_size: int = 1,
    concurrency: int = 1,
    cache_policy: str = "disabled",
    runtime_config: Mapping[str, Any] | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Build identity and timing fields shared by benchmark adapters.

    ``generation_config`` must contain only parameters that define the target
    generation task (for example temperature, output budget, seed and stop
    IDs); speculative algorithm parameters belong in the run manifest.
    """
    def optional_finite(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    token_values = [int(value) for value in prompt_token_ids]
    normalized_generation = dict(generation_config)
    stop_ids = normalized_generation.get("stop_token_ids")
    if isinstance(stop_ids, int) and not isinstance(stop_ids, bool):
        normalized_generation["stop_token_ids"] = [stop_ids]
    elif isinstance(stop_ids, tuple):
        normalized_generation["stop_token_ids"] = list(stop_ids)
    normalized_hardware = dict(hardware)
    return {
        "contract_version": 2,
        "prompt_token_sha256": token_ids_sha256(token_values),
        "generation_config_sha256": canonical_sha256(normalized_generation),
        "pairing_config_sha256": canonical_sha256(normalized_generation),
        "hardware_fingerprint": canonical_sha256(normalized_hardware),
        "runtime_config_sha256": canonical_sha256(dict(runtime_config or {})),
        "target_revision": target_revision,
        "tokenizer_revision": tokenizer_revision,
        "gpu_count": gpu_count,
        "tp_size": int(tp_size),
        "batch_size": int(batch_size),
        "concurrency": int(concurrency),
        "cache_policy": str(cache_policy),
        "actual_input_tokens": len(token_values),
        "timed_generated_tokens": _counter(timed_generated_tokens),
        "visible_output_tokens": _counter(visible_output_tokens),
        "stop_reason": stop_reason,
        "request_wall_ms": optional_finite(request_wall_ms),
        "native_elapsed_ms": optional_finite(native_elapsed_ms),
        "native_timing_scope": native_timing_scope,
        "decode_active_ms": optional_finite(decode_active_ms),
        "decode_token_count": _counter(decode_token_count),
        "decode_phase_definition": decode_phase_definition,
        "decode_phase_verified": bool(decode_phase_verified),
        "timing_source": timing_source,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("type") != "summary":
            rows.append(row)
        elif not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
    return rows


def _dataset_records(run_dir: Path, baseline: str, dataset: str) -> list[dict[str, Any]]:
    path = run_dir / baseline / f"{dataset}.jsonl"
    return read_jsonl(path) if path.is_file() else []


def _metric_observations(
    reference_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[tuple[Any, ...]]:
    pairs, _ = _paired_rows(reference_rows, method_rows, scope="common")
    if metric == "common_esr":
        values, _ = _metric_pairs(pairs, reference_field="request_wall_ms", method_field="request_wall_ms", kind="time")
        return [(reference, method) for _, reference, method in values]
    if metric == "common_decode_time_ratio":
        values, _ = _metric_pairs(pairs, reference_field="decode_active_ms", method_field="decode_active_ms", kind="decode_time")
        return [(reference, method) for _, reference, method in values]
    if metric == "common_decode_rate_ratio":
        values, _ = _metric_pairs(pairs, reference_field="decode_active_ms", method_field="decode_active_ms", kind="decode_rate")
        return _rate_observations(values, pairs, time_field="decode_active_ms")
    if metric == "common_output_rate_ratio":
        values, _ = _metric_pairs(pairs, reference_field="request_wall_ms", method_field="request_wall_ms", kind="output_rate")
        return _rate_observations(values, pairs, time_field="request_wall_ms")
    raise ValueError(f"unknown common metric: {metric}")


def _estimate_observations(observations: Sequence[tuple[Any, ...]], metric: str) -> float | None:
    if metric in {"common_esr", "common_decode_time_ratio"}:
        return _ratio_of_sums([(str(index), pair[0], pair[1]) for index, pair in enumerate(observations)])
    if not observations:
        return None
    ref_tokens = sum(row[0] for row in observations)
    method_tokens = sum(row[1] for row in observations)
    ref_time = sum(row[2] for row in observations)
    method_time = sum(row[3] for row in observations)
    if min(ref_tokens, method_tokens, ref_time, method_time) <= 0:
        return None
    return (method_tokens / method_time) / (ref_tokens / ref_time)


def _stratified_geomean_ci(
    dataset_observations: Mapping[str, Sequence[tuple[Any, ...]]],
    metric: str,
    *,
    seed: int,
    resamples: int = 10_000,
) -> tuple[float, float] | None:
    if not dataset_observations or any(not rows for rows in dataset_observations.values()):
        return None
    rng = random.Random(seed)
    estimates = []
    datasets = sorted(dataset_observations)
    for _ in range(resamples):
        values = []
        for dataset in datasets:
            rows = dataset_observations[dataset]
            sample = [rows[rng.randrange(len(rows))] for _ in rows]
            estimate = _estimate_observations(sample, metric)
            if estimate is None or estimate <= 0:
                values = []
                break
            values.append(estimate)
        if values:
            estimates.append(math.exp(sum(math.log(value) for value in values) / len(values)))
    if not estimates:
        return None
    estimates.sort()
    return estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]


def _mean(values: Sequence[Any]) -> float | None:
    clean = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return sum(clean) / len(clean) if clean else None


def build_paper_report(run_dir: Path) -> dict[str, Any]:
    """Rebuild deterministic v2 pairwise, shared-set, and audit outputs."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    common_reference = str(manifest.get("common_reference", COMMON_REFERENCE_DEFAULT))
    datasets = tuple(manifest.get("datasets") or PAPER_DATASETS)
    baselines = tuple(manifest.get("baselines") or PAPER_BASELINES)
    report_rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    metric_names = ("common_esr", "common_decode_time_ratio", "common_decode_rate_ratio", "common_output_rate_ratio")
    bootstrap_resamples = int(manifest.get("bootstrap_resamples", 10_000))
    audit: dict[str, Any] = {
        "contract_version": 2,
        "bootstrap_resamples": bootstrap_resamples,
        "common_reference": common_reference,
        "datasets": {},
        "pairwise_exclusions": {},
    }
    all_shared_observations: dict[str, dict[str, dict[str, list[tuple[Any, ...]]]]] = {
        metric: {baseline: {} for baseline in baselines} for metric in metric_names
    }

    for dataset in datasets:
        dataset_rows = {baseline: _dataset_records(run_dir, baseline, dataset) for baseline in baselines}
        reference = dataset_rows.get(common_reference, [])
        expected = int(manifest.get("sample_count") or len(reference))
        audit["datasets"][dataset] = {"expected_samples": expected, "baselines": {}}
        pairwise: dict[str, dict[str, Any]] = {}
        for baseline in baselines:
            method = dataset_rows.get(baseline, [])
            common = aggregate_pair(reference, method, scope="common", bootstrap_resamples=bootstrap_resamples) if reference else {}
            native_reference: list[dict[str, Any]] = []
            native_path = None
            if baseline in {"domino", "dspark"}:
                native_path = run_dir / "references" / "sglang_target_only" / f"{dataset}.jsonl"
                native_reference = read_jsonl(native_path) if native_path.is_file() else []
            elif baseline in {"dflash", "eagle3"}:
                native_method = "dflash_block_size_1" if baseline == "dflash" else "eagle_naivegenerate"
                native_reference = [
                    {**row["native_reference"], "sample_id": row.get("sample_id"), "dataset": dataset,
                     "run_id": row.get("run_id", row["native_reference"].get("run_id")),
                     "method": native_method, "contract_version": 2}
                    for row in method if isinstance(row.get("native_reference"), Mapping)
                ]
            native = aggregate_pair(native_reference, method, scope="native", bootstrap_resamples=bootstrap_resamples) if native_reference else {}
            cell = {"report_scope": "pairwise", "dataset": dataset, "method": baseline, "common_reference": common_reference}
            cell.update(common)
            cell.update(native)
            success = [row for row in method if row.get("status", "success") == "success"]
            cell["success_count"] = len(success)
            cell["expected_count"] = expected
            cell["success_coverage"] = len(success) / expected if expected else None
            cell["mean_rouge1_f"] = _mean([row.get("rouge1_f") for row in success])
            cell["mean_rougeL_f"] = _mean([row.get("rougeL_f") for row in success])
            request_times = [
                value for row in success
                if (value := _finite_positive(row.get("request_wall_ms"))) is not None
            ]
            cell["median_request_wall_ms"] = _percentile(request_times, 0.50)
            cell["p90_request_wall_ms"] = _percentile(request_times, 0.90)
            cell["mean_visible_output_tokens"] = _mean([
                _counter(row.get("visible_output_tokens", row.get("output_tokens")))
                for row in success
            ])
            cell["degenerate_output_count"] = sum(_row_is_degenerate(row) for row in success)
            cell["degenerate_output_rate"] = (
                cell["degenerate_output_count"] / len(success) if success else None
            )
            cell["native_reference"] = (
                "sglang_target_only" if baseline in {"domino", "dspark"}
                else "dflash_block_size_1" if baseline == "dflash"
                else "eagle_naivegenerate" if baseline == "eagle3" else None
            )
            report_rows.append(cell)
            pairwise[baseline] = common
            audit["datasets"][dataset]["baselines"][baseline] = {
                "records": len(method), "success": len(success),
                "success_coverage": len(success) / expected if expected else None,
                "common": common, "native": native,
                "native_reference_records": len(native_reference),
                "prompt_token_count_mismatches": sum(row.get("prompt_token_count_match") is False for row in method),
            }
            for metric in metric_names:
                all_shared_observations[metric][baseline][dataset] = _metric_observations(reference, method, metric) if reference else []
        audit["pairwise_exclusions"][dataset] = {
            baseline: pairwise.get(baseline, {}).get("common_esr_exclusions", {})
            for baseline in baselines
        }

        # A metric-specific shared set keeps all six baselines on identical IDs.
        for metric in metric_names:
            valid_ids = []
            for baseline in baselines:
                result = pairwise.get(baseline, {})
                id_key = {
                    "common_esr": "common_esr_valid_sample_ids",
                    "common_decode_time_ratio": "common_decode_time_valid_sample_ids",
                    "common_decode_rate_ratio": "common_decode_rate_valid_sample_ids",
                    "common_output_rate_ratio": "common_output_rate_valid_sample_ids",
                }[metric]
                valid_ids.append(set(result.get(id_key) or []))
            shared_ids = sorted(set.intersection(*valid_ids)) if valid_ids and all(valid_ids) else []
            metric_audit = {"sample_ids": shared_ids, "n": len(shared_ids), "baselines": {}}
            ref_by_id, _, _ = _index_rows(reference)
            filtered_reference = [ref_by_id[sid] for sid in shared_ids if sid in ref_by_id]
            for baseline in baselines:
                method_by_id, _, _ = _index_rows(dataset_rows.get(baseline, []))
                filtered_method = [method_by_id[sid] for sid in shared_ids if sid in method_by_id]
                shared_result = aggregate_pair(filtered_reference, filtered_method, scope="common", bootstrap_resamples=bootstrap_resamples) if shared_ids else {}
                value = shared_result.get(metric)
                low = shared_result.get(f"{metric}_ci95_low")
                high = shared_result.get(f"{metric}_ci95_high")
                metric_audit["baselines"][baseline] = {"value": value, "n": shared_result.get(f"{metric}_n", 0)}
                shared_rows.append({
                    "report_scope": "six_way_shared", "dataset": dataset, "method": baseline,
                    "metric": metric, "value": value, "ci95_low": low, "ci95_high": high,
                    "n": shared_result.get(f"{metric}_n", 0), "shared_sample_count": len(shared_ids),
                })
                all_shared_observations[metric][baseline][dataset] = _metric_observations(filtered_reference, filtered_method, metric) if shared_ids else []
            audit.setdefault("six_way_shared", {}).setdefault(dataset, {})[metric] = metric_audit

    # Equal-weight geometric mean across the four dataset-level shared-set ratios.
    geomean_audit: dict[str, Any] = {}
    for metric in metric_names:
        geomean_audit[metric] = {}
        for baseline in baselines:
            per_dataset = all_shared_observations[metric][baseline]
            estimates = {dataset: _estimate_observations(rows, metric) for dataset, rows in per_dataset.items()}
            complete = len(datasets) == 4 and all(estimates.get(dataset) is not None and estimates[dataset] > 0 for dataset in datasets)
            value = math.exp(sum(math.log(estimates[dataset]) for dataset in datasets) / len(datasets)) if complete else None
            seed_text = f"{baseline}:{metric}:20260928"
            seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
            interval = _stratified_geomean_ci(per_dataset, metric, seed=seed, resamples=bootstrap_resamples) if complete else None
            low, high = interval or (None, None)
            geomean_audit[metric][baseline] = {"value": value, "ci95_low": low, "ci95_high": high, "dataset_values": estimates}
            shared_rows.append({"report_scope": "four_dataset_geomean", "dataset": "ALL", "method": baseline,
                                "metric": metric, "value": value, "ci95_low": low, "ci95_high": high,
                                "n": sum(len(per_dataset.get(dataset, [])) for dataset in datasets) if complete else 0,
                                "shared_sample_count": None})
    audit["four_dataset_geomean"] = geomean_audit

    strict_decode_all = all(
        audit["datasets"].get(dataset, {}).get("baselines", {}).get(baseline, {}).get("common", {}).get("common_decode_rate_n", 0) > 0
        for dataset in datasets for baseline in baselines
    )
    prompt_counts_match = all(
        detail.get("prompt_token_count_mismatches", 0) == 0
        for dataset in audit["datasets"].values() for baseline, detail in dataset["baselines"].items()
    )
    if manifest.get("mode") == "smoke":
        smoke_native_ok = all(
            audit["datasets"].get(dataset, {}).get("baselines", {}).get(baseline, {}).get("native", {}).get("native_esr_n", 0) > 0
            and audit["datasets"].get(dataset, {}).get("baselines", {}).get(baseline, {}).get("native", {}).get("native_decode_rate_n", 0) > 0
            for dataset in datasets for baseline in ("eagle3", "dflash", "domino", "dspark")
        )
        gate_ok = strict_decode_all and prompt_counts_match and smoke_native_ok and set(baselines) == set(PAPER_BASELINES)
        audit["paper_gate"] = {"status": "smoke_pass" if gate_ok else "fail",
                               "strict_common_decode_all_cells": strict_decode_all,
                               "prompt_token_counts_match": prompt_counts_match,
                               "native_references_available": smoke_native_ok,
                               "native_decode_available": smoke_native_ok,
                               "reason": None if gate_ok else "smoke must verify strict common/native decode, exact prompt token counts, and all native references"}
    elif manifest.get("paper_speedup"):
        coverages = [detail.get("success_coverage") or 0 for dataset in audit["datasets"].values() for detail in dataset["baselines"].values()]
        shared_min = min((metric.get("n", 0) for dataset in audit.get("six_way_shared", {}).values() for metric in dataset.values()), default=0)
        coverage_ok = len(coverages) == len(datasets) * len(baselines) and all(value >= 0.95 for value in coverages)
        shared_ok = shared_min >= 0.90 * int(manifest.get("sample_count") or 0)
        native_counts = [
            min(
                audit["datasets"].get(dataset, {}).get("baselines", {}).get(baseline, {}).get("native", {}).get("native_esr_n", 0),
                audit["datasets"].get(dataset, {}).get("baselines", {}).get(baseline, {}).get("native", {}).get("native_decode_rate_n", 0),
            )
            for dataset in datasets for baseline in ("eagle3", "dflash", "domino", "dspark")
        ]
        native_coverage_ok = (
            len(native_counts) == len(datasets) * 4
            and all(value >= 0.90 * int(manifest.get("sample_count") or 0) for value in native_counts)
        )
        calibration = manifest.get("anchor_calibration") or {}
        calibration_ok = calibration.get("status") == "pass"
        gate_ok = strict_decode_all and prompt_counts_match and coverage_ok and shared_ok and native_coverage_ok and calibration_ok
        reasons = []
        if not strict_decode_all: reasons.append("strict common decode coverage is incomplete")
        if not prompt_counts_match: reasons.append("server/local prompt token counts differ")
        if not coverage_ok: reasons.append("success coverage is below 95% for at least one cell")
        if not shared_ok: reasons.append("six-way shared coverage is below 90% for at least one metric/dataset")
        if not native_coverage_ok: reasons.append("native e2e/decode coverage is below 90% for a speculative baseline/dataset")
        if not calibration_ok: reasons.append("anchor drift calibration is missing or failed")
        audit["paper_gate"] = {"status": "paper_ready" if gate_ok else "not_qualified",
                               "strict_common_decode_all_cells": strict_decode_all,
                               "prompt_token_counts_match": prompt_counts_match,
                               "coverage_at_least_95_percent": coverage_ok,
                               "six_way_shared_at_least_90_percent": shared_ok,
                               "native_e2e_and_decode_at_least_90_percent": native_coverage_ok,
                               "anchor_calibration": calibration,
                               "reason": "; ".join(reasons) if reasons else None}
    else:
        audit["paper_gate"] = {"status": "not_paper_profile", "reason": "run was not created with --paper-speedup"}
    return {"audit": audit, "rows": report_rows, "shared_rows": shared_rows}


def write_paper_report(run_dir: Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    report = build_paper_report(run_dir)
    audit_path = run_dir / "audit_v2.json"
    csv_path = run_dir / "paper_speedup.csv"
    md_path = run_dir / "paper_speedup.md"
    audit_path.write_text(json.dumps(report["audit"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = sorted({key for row in report["rows"] + report["shared_rows"] for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"] + report["shared_rows"]:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    lines = [
        "# Paired speedup results (contract v2)", "",
        f"Common reference: `{report['audit']['common_reference']}`. Output mismatches remain timing pairs and are reported as quality signals.", "",
        "## Pairwise results",
        "",
        "| Dataset | Baseline | Common ESR (95% CI) | Decode time ratio | Decode rate ratio | Output rate ratio | n ESR | n decode | Native ESR | Native decode rate | Common exact text | Native exact text | Mean token ratio | Mean output tokens | Degenerate rate | p50/p90 request ms | ROUGE-1 F | ROUGE-L F |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def show(value: Any) -> str:
        return "—" if value is None else str(value)
    for row in report["rows"]:
        quality = row.get("common_quality") or {}
        native_quality = row.get("native_quality") or {}
        esr = row.get("common_esr")
        ci = (row.get("common_esr_ci95_low"), row.get("common_esr_ci95_high"))
        esr_text = show(esr) + (f" [{ci[0]}, {ci[1]}]" if ci[0] is not None else "")
        request_percentiles = f"{show(row.get('median_request_wall_ms'))}/{show(row.get('p90_request_wall_ms'))}"
        cells = [row.get("dataset"), row.get("method"), esr_text,
                 row.get("common_decode_time_ratio"), row.get("common_decode_rate_ratio"),
                 row.get("common_output_rate_ratio"), row.get("common_esr_n", 0),
                 row.get("common_decode_rate_n", 0), row.get("native_esr"),
                 row.get("native_decode_rate_ratio"), quality.get("exact_text_match_rate"),
                 native_quality.get("exact_text_match_rate"), quality.get("mean_visible_token_ratio"),
                 row.get("mean_visible_output_tokens"), row.get("degenerate_output_rate"),
                 request_percentiles, row.get("mean_rouge1_f"), row.get("mean_rougeL_f")]
        lines.append("| " + " | ".join(show(value) for value in cells) + " |")
    lines.extend(["", "## Six-way shared-set and four-dataset geometric mean", "",
                  "The shared-set table uses identical successful sample IDs across all six baselines for each metric. Geometric means weight each dataset equally; bootstrap CIs are stratified by dataset.", "",
                  "| Scope | Dataset | Baseline | Metric | Ratio | 95% paired CI | n |",
                  "|---|---|---|---|---:|---:|---:|"])
    for row in report["shared_rows"]:
        ci = (row.get("ci95_low"), row.get("ci95_high"))
        ci_text = f"[{ci[0]}, {ci[1]}]" if ci[0] is not None else "—"
        lines.append("| " + " | ".join(show(value) for value in (
            row.get("report_scope"), row.get("dataset"), row.get("method"), row.get("metric"),
            row.get("value"), ci_text, row.get("n"))) + " |")
    lines.extend(["", "Decode ratios are emitted only for matching, verified first-token-to-final-token boundaries. `common_output_rate_ratio` is end-to-end output rate, not decode speedup.",
                  f"Paper gate: `{report['audit'].get('paper_gate', {}).get('status')}`. See `audit_v2.json` for exclusions, shared sample IDs, coverage, and qualification reasons.", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"audit": audit_path, "csv": csv_path, "markdown": md_path}
