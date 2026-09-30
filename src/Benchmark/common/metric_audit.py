"""Machine-readable metric coverage and failure diagnostics.

This module is deliberately post-processing only.  It inspects normalized
benchmark records and never changes model inputs, generation settings, or
timings.  The runner uses it to make missing metrics visible instead of
silently presenting a sparse table.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from Benchmark.common.quality_guard import is_degenerate_output


AUDIT_SCHEMA_VERSION = 1

# Measurement scope is part of the comparison contract.  A row without an
# explicit scope must be normalized from this registry before it can enter a
# strict report; otherwise a legacy adapter can look complete merely because it
# happened to emit an ``e2e_ms`` field.
BASELINE_MEASUREMENT_SCOPE = {
    "vanilla_hf": "full_e2e",
    "vanilla_fa": "full_e2e",
    "eagle3": "full_e2e",
    "dflash": "full_e2e",
    # SGLang-backed baselines expose request E2E plus a direct server-side
    # decode interval when the benchmark's streaming timing patch is active.
    "domino": "e2e_plus_decode",
    "dspark": "e2e_plus_decode",
}

_TIMING_BY_SCOPE = {
    "full_e2e": (
        "input_tokens",
        "output_tokens",
        "prefill_ms",
        "ttft_ms",
        "decode_ms",
        "e2e_ms",
        "throughput_tok_s",
    ),
    "e2e_only": (
        "input_tokens",
        "output_tokens",
        "e2e_ms",
        "throughput_tok_s",
    ),
    # The endpoint does not return a trustworthy prefill/TTFT phase, but its
    # patched streaming request stats expose a direct first-token-to-finish
    # decode duration alongside end-to-end request time.
    "e2e_plus_decode": (
        "input_tokens",
        "output_tokens",
        "decode_ms",
        "e2e_ms",
        "throughput_tok_s",
    ),
    "decode_only": (
        "input_tokens",
        "output_tokens",
        "decode_ms",
        "decode_throughput_tok_s",
    ),
}

_QUALITY_FIELDS = (
    "rouge1",
    "rouge2",
    "rougeL",
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "code_exact_match",
    "code_edit_similarity",
)

# Raw fields are emitted by the inference process.  Derived throughput and
# speedup values are intentionally excluded because the collector can
# recompute them from the raw timings.
_COMMON_DIRECT_FIELDS = (
    "input_tokens",
    "retained_tokens",
    "output_tokens",
    "batch_size",
    "e2e_ms",
    "device",
)
_MODEL_MEMORY_DIRECT_FIELDS = ("model_load_ms", "peak_memory_gb")
_FULL_E2E_DIRECT_FIELDS = ("prefill_ms", "ttft_ms", "decode_ms")
_SPECULATIVE_DIRECT_FIELDS = (
    "acceptance_lengths",
    "draft_latency_ms",
    "verification_latency_ms",
)
_TEXT_QUALITY_FIELDS = (
    "rouge1_p",
    "rouge1_r",
    "rouge1_f",
    "rouge2_p",
    "rouge2_r",
    "rouge2_f",
    "rougeL_p",
    "rougeL_r",
    "rougeL_f",
    "rougeLsum_p",
    "rougeLsum_r",
    "rougeLsum_f",
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "length_ratio",
)
_DERIVED_ISSUES = {
    "speedup_invalid",
    # Quality warning only: retain and report the latency observation.
    "degenerate_repetition",
    "missing_tpot_ms",
    "missing_throughput_tok_s",
    "missing_decode_throughput_tok_s",
}
_QUALITY_FIELDS = tuple(dict.fromkeys(_QUALITY_FIELDS + _TEXT_QUALITY_FIELDS))
_SPECULATIVE_BASELINES = frozenset({"eagle3", "dflash", "domino", "dspark"})


def _acceptance_metric_issues(record: Mapping[str, Any]) -> list[str]:
    """Check speculative acceptance fields and their directly available traces."""

    baseline = str(record.get("method") or "")
    if baseline not in _SPECULATIVE_BASELINES:
        return []

    issues: list[str] = []
    avg_accept_length = record.get("avg_accept_length")
    if avg_accept_length is None and baseline == "eagle3":
        avg_accept_length = record.get("accept_length")
    if avg_accept_length is None:
        issues.append("missing_avg_accept_length")
    else:
        try:
            avg_accept_length = float(avg_accept_length)
        except (TypeError, ValueError):
            issues.append("invalid_avg_accept_length")
        else:
            if not math.isfinite(avg_accept_length) or avg_accept_length < 1.0:
                issues.append("invalid_avg_accept_length")

    acceptance_rate = record.get("acceptance_rate")
    if acceptance_rate is None:
        issues.append("missing_acceptance_rate")
    else:
        try:
            acceptance_rate = float(acceptance_rate)
        except (TypeError, ValueError):
            issues.append("invalid_acceptance_rate")
        else:
            if not math.isfinite(acceptance_rate) or not 0.0 <= acceptance_rate <= 1.0:
                issues.append("invalid_acceptance_rate")

    if baseline in {"eagle3", "dflash"}:
        lengths = record.get("acceptance_lengths")
        if not isinstance(lengths, (list, tuple)) or not lengths:
            issues.append("missing_acceptance_lengths")
        else:
            try:
                normalized_lengths = [float(value) for value in lengths]
            except (TypeError, ValueError):
                issues.append("invalid_acceptance_length")
            else:
                if any(
                    not math.isfinite(value) or value < 1.0
                    for value in normalized_lengths
                ):
                    issues.append("invalid_acceptance_length")
                elif (
                    avg_accept_length is not None
                    and "invalid_avg_accept_length" not in issues
                    and abs(
                        sum(normalized_lengths) / len(normalized_lengths)
                        - avg_accept_length
                    ) > max(0.02, 0.005 * avg_accept_length)
                ):
                    issues.append("acceptance_trace_average_mismatch")
            verification_steps = record.get("verification_steps")
            if verification_steps is not None:
                try:
                    if len(lengths) != int(verification_steps):
                        issues.append("acceptance_trace_step_mismatch")
                except (TypeError, ValueError):
                    issues.append("invalid_verification_steps")

    histogram = record.get("acceptance_histogram")
    verification_steps = record.get("verification_steps")
    if histogram is not None:
        if not isinstance(histogram, (list, tuple)) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in histogram
        ):
            issues.append("invalid_acceptance_histogram")
        else:
            if (
                verification_steps is not None
                and sum(histogram) != int(verification_steps)
            ):
                issues.append("acceptance_histogram_step_mismatch")
            if (
                histogram
                and verification_steps
                and avg_accept_length is not None
                and "invalid_avg_accept_length" not in issues
            ):
                histogram_average = 1.0 + sum(
                    index * count for index, count in enumerate(histogram)
                ) / float(verification_steps)
                # A final EOS may be counted as one completion token by the
                # request metric but omitted from the accepted-draft histogram.
                # Allow that single-token discrepancy, plus small float noise.
                histogram_tolerance = max(
                    0.02,
                    0.005 * avg_accept_length,
                    1.0 / float(verification_steps) + 0.001,
                )
                if abs(histogram_average - avg_accept_length) > histogram_tolerance:
                    issues.append("acceptance_histogram_length_mismatch")

    correct = record.get("draft_tokens_accepted")
    proposed = record.get("draft_tokens_proposed")
    if (
        baseline in {"eagle3", "dflash", "domino", "dspark"}
        and correct is not None
        and proposed is not None
    ):
        try:
            correct_value = float(correct)
            proposed_value = float(proposed)
        except (TypeError, ValueError):
            issues.append("invalid_acceptance_counters")
        else:
            if (
                not math.isfinite(correct_value)
                or not math.isfinite(proposed_value)
                or proposed_value <= 0
                or not 0 <= correct_value <= proposed_value
            ):
                issues.append("invalid_acceptance_counters")
            elif (
                acceptance_rate is not None
                and "invalid_acceptance_rate" not in issues
                and abs(correct_value / proposed_value - acceptance_rate) > 1e-4
            ):
                issues.append("acceptance_rate_counter_mismatch")
            if acceptance_rate is not None:
                rate_percent = record.get("acceptance_rate_percent")
                if rate_percent is not None:
                    try:
                        rate_percent_value = float(rate_percent)
                    except (TypeError, ValueError):
                        issues.append("invalid_acceptance_rate_percent")
                    else:
                        if (
                            not math.isfinite(rate_percent_value)
                            or abs(rate_percent_value - acceptance_rate * 100.0)
                            > 0.00011
                        ):
                            issues.append("acceptance_rate_percent_mismatch")
                accepted_per_step = record.get("accepted_draft_tokens_per_step")
                verification_steps = record.get("verification_steps")
                if accepted_per_step is not None and verification_steps:
                    try:
                        accepted_per_step_value = float(accepted_per_step)
                        steps_value = float(verification_steps)
                    except (TypeError, ValueError):
                        issues.append("invalid_accepted_draft_tokens_per_step")
                    else:
                        if (
                            not math.isfinite(accepted_per_step_value)
                            or steps_value <= 0
                            or abs(accepted_per_step_value - correct_value / steps_value)
                            > 1e-4
                        ):
                            issues.append("accepted_draft_tokens_per_step_mismatch")
                        avg_accept_length = record.get("avg_accept_length")
                        if (
                            avg_accept_length is not None
                            and "invalid_avg_accept_length" not in issues
                            and steps_value > 0
                        ):
                            try:
                                avg_accept_length_value = float(avg_accept_length)
                            except (TypeError, ValueError):
                                issues.append("invalid_avg_accept_length")
                            else:
                                expected_length = 1.0 + correct_value / steps_value
                                length_tolerance = max(
                                    0.02, 1.0 / steps_value + 0.001
                                )
                                if (
                                    not math.isfinite(avg_accept_length_value)
                                    or abs(avg_accept_length_value - expected_length)
                                    > length_tolerance
                                ):
                                    issues.append(
                                        "acceptance_length_counter_mismatch"
                                    )

    return issues


def required_direct_metrics(
    baseline: str, *, measurement_scope: str | None = None
) -> tuple[str, ...]:
    """Return raw fields required by the Vietnamese baseline contract.

    SGLang-backed Domino/DSpark expose request E2E and a direct decode phase
    through the benchmark's version-pinned streaming timing patch.  They still
    do not expose a comparable prefill/TTFT phase or model-load/process memory
    fields, so those remain outside this adapter's required contract.
    """

    scope = measurement_scope or BASELINE_MEASUREMENT_SCOPE.get(baseline)
    if scope is None:
        return ()
    fields = list(_COMMON_DIRECT_FIELDS)
    if baseline in {"vanilla_hf", "vanilla_fa", "eagle3", "dflash"}:
        fields.extend(_MODEL_MEMORY_DIRECT_FIELDS)
    if scope == "full_e2e":
        fields.extend(_FULL_E2E_DIRECT_FIELDS)
    elif scope == "e2e_plus_decode":
        fields.extend(("decode_ms", "decode_phase_verified"))
    # EAGLE/DFlash expose per-iteration draft/verification timing directly.
    # SGLang server adapters retain these fields when upstream provides them,
    # but must not be marked incomplete merely because stock SGLang omits them.
    if baseline in {"eagle3", "dflash"}:
        fields.extend(_SPECULATIVE_DIRECT_FIELDS)
    return tuple(fields)


def _missing_quality_fields(record: Mapping[str, Any]) -> list[str]:
    reference_present = bool(str(record.get("reference_output") or "").strip())
    text_present = bool(str(record.get("text") or record.get("answer") or "").strip())
    if not reference_present or not text_present:
        return []
    return [field for field in _TEXT_QUALITY_FIELDS if not _is_present(record.get(field))]


def _is_present(value: object) -> bool:
    return value is not None


def _timing_source(record: Mapping[str, Any]) -> str:
    explicit = record.get("timing_source")
    if explicit:
        return str(explicit)
    method = str(record.get("method") or "")
    if record.get("eagle_phase_timings"):
        return "eagle_phase_timer"
    if "specextend" in method or method.startswith("fafo"):
        return "adapter_sidecar_or_upstream"
    if record.get("kv_cache_backend") or (
        isinstance(record.get("extra_metrics"), Mapping)
        and (record.get("extra_metrics") or {}).get("kv_cache_backend")
    ):
        return "manual_phase_timer"
    if record.get("timing"):
        return "upstream_timing_record"
    if record.get("eagle_time") is not None:
        return "legacy_decode_timer"
    return "legacy_or_adapter_record"


def audit_record(
    record: Mapping[str, Any], *, expected_output_tokens: int | None = None
) -> dict[str, Any]:
    """Return an audit object for one normalized sample/status record."""

    status = str(record.get("status", "success"))
    scope = str(record.get("measurement_scope") or "unknown")
    known_scope = scope in _TIMING_BY_SCOPE
    required = list(_TIMING_BY_SCOPE.get(scope, ()))
    present = [field for field in required if _is_present(record.get(field))]
    missing = [field for field in required if field not in present]
    if not known_scope:
        timing_status = "unknown"
    elif not missing:
        timing_status = "complete"
    elif present:
        timing_status = "partial"
    else:
        timing_status = "missing"

    reference_present = bool(str(record.get("reference_output") or "").strip())
    text_present = bool(str(record.get("text") or record.get("answer") or "").strip())
    quality_present = [
        field for field in _QUALITY_FIELDS if _is_present(record.get(field))
    ]
    if not reference_present:
        quality_status = "not_available"
    elif not text_present:
        quality_status = "missing_text"
    elif not quality_present:
        quality_status = "missing_metric"
    else:
        quality_status = "complete"

    budget = expected_output_tokens
    if budget is None:
        try:
            budget = int(record["max_new_tokens"])
        except (KeyError, TypeError, ValueError):
            budget = None
    output_tokens: int | None
    try:
        output_tokens = int(record["output_tokens"])
    except (KeyError, TypeError, ValueError):
        output_tokens = None
    budget_valid: bool | None = None
    if budget is not None and output_tokens is not None:
        budget_valid = 0 < output_tokens <= max(1, int(budget))

    issues: list[str] = []
    if status != "success":
        issues.append("failed_record")
    if status == "success":
        if not known_scope:
            issues.append("missing_measurement_scope")
        issues.extend(f"missing_{field}" for field in missing)
        if scope == "e2e_plus_decode":
            if record.get("decode_phase_verified") is not True:
                issues.append("unverified_decode_phase")
            if record.get("decode_phase_definition") != "after_first_token_committed_to_final_token":
                issues.append("missing_decode_phase_definition")
        if output_tokens is None or output_tokens <= 0:
            issues.append("missing_output_tokens")
        if budget_valid is False:
            issues.append("output_budget_invalid")
        if quality_status == "missing_text":
            issues.append("missing_text")
        elif quality_status == "missing_metric":
            issues.append("missing_quality_metric")
    if record.get("speedup_valid") is False:
        issues.append("speedup_invalid")
    if record.get("text_decode_matches_token_ids") is False:
        issues.append("text_token_ids_mismatch")
    issues.extend(_acceptance_metric_issues(record))
    guard = record.get("output_quality_guard")
    output_text = record.get("text") or record.get("answer")
    if (
        record.get("degenerate_repetition")
        or (isinstance(guard, Mapping) and guard.get("degenerate_repetition"))
        or is_degenerate_output(str(output_text or ""))
    ):
        issues.append("degenerate_repetition")

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "sample_id": record.get("sample_id"),
        "status": status,
        "measurement_scope": scope,
        "timing_source": _timing_source(record),
        "timing": {
            "status": timing_status,
            "required": required,
            "present": present,
            "missing": missing,
        },
        "quality": {
            "status": quality_status,
            "reference_present": reference_present,
            "text_present": text_present,
            "present": quality_present,
            "target_greedy_match": record.get("target_greedy_match"),
            "external_reference_output_exact_match": record.get("external_reference_output_exact_match"),
            "external_reference_output_token_count_match": record.get("external_reference_output_token_count_match"),
            "output_degenerate": bool(
                record.get("degenerate_repetition")
                or (isinstance(guard, Mapping) and guard.get("degenerate_repetition"))
                or is_degenerate_output(str(output_text or ""))
            ),
        },
        "tokens": {
            "input_tokens": record.get("input_tokens"),
            "output_tokens": output_tokens,
            "budget": budget,
            "budget_valid": budget_valid,
        },
        "speedup": {
            "valid": record.get("speedup_valid"),
            "scope": record.get("speedup_scope"),
        },
        "issues": issues,
    }


def summarize_audits(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-record audits into a compact cell-level summary."""

    status_counts = Counter(str(a.get("status", "unknown")) for a in audits)
    scope_counts = Counter(str(a.get("measurement_scope", "unknown")) for a in audits)
    issue_counts: Counter[str] = Counter()
    quality_status_counts: Counter[str] = Counter()
    timing_status_counts: Counter[str] = Counter()
    timing_source_counts: Counter[str] = Counter()
    field_present: Counter[str] = Counter()
    field_observed: Counter[str] = Counter()
    budget_valid_counts: Counter[str] = Counter()
    samples_with_issues: list[Any] = []
    for audit in audits:
        issue_counts.update(str(issue) for issue in audit.get("issues", []))
        quality = audit.get("quality") or {}
        timing = audit.get("timing") or {}
        quality_status_counts[str(quality.get("status", "unknown"))] += 1
        timing_status_counts[str(timing.get("status", "unknown"))] += 1
        timing_source_counts[str(audit.get("timing_source", "unknown"))] += 1
        for field in timing.get("required", []):
            field_observed[str(field)] += 1
        for field in timing.get("present", []):
            field_present[str(field)] += 1
        budget = (audit.get("tokens") or {}).get("budget_valid")
        if budget is not None:
            budget_valid_counts[str(bool(budget)).lower()] += 1
        if audit.get("issues"):
            samples_with_issues.append(audit.get("sample_id"))

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "num_records": len(audits),
        "status_counts": dict(status_counts),
        "scope_counts": dict(scope_counts),
        "timing_status_counts": dict(timing_status_counts),
        "timing_source_counts": dict(timing_source_counts),
        "timing_field_coverage": {
            field: {
                "present": field_present.get(field, 0),
                "observed": count,
            }
            for field, count in sorted(field_observed.items())
        },
        "quality_status_counts": dict(quality_status_counts),
        "quality_records": quality_status_counts.get("complete", 0),
        "budget_valid_counts": dict(budget_valid_counts),
        "issue_counts": dict(issue_counts),
        "samples_with_issues": samples_with_issues[:20],
    }


def _has_verified_decode_phase(record: Mapping[str, Any]) -> bool:
    if (
        record.get("status", "success") != "success"
        or record.get("decode_phase_verified") is not True
        or record.get("decode_phase_definition") != "after_first_token_committed_to_final_token"
    ):
        return False
    value = record.get("decode_ms")
    if value is None or isinstance(value, bool):
        return False
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(duration) and duration > 0.0


def validate_cell_metric_contract(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    expected_samples: int,
    expected_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Decide whether one baseline/dataset cell is eligible for strict use."""

    expected_scope = BASELINE_MEASUREMENT_SCOPE.get(baseline)
    sample_records = [
        record for record in records if record.get("type") != "summary"
    ]
    audits = [
        audit_record(record, expected_output_tokens=expected_output_tokens)
        for record in sample_records
    ]
    sample_ids = [
        str(record.get("sample_id"))
        for record in sample_records
        if record.get("sample_id") is not None
    ]
    duplicate_ids = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    )
    missing_field_counts: Counter[str] = Counter()
    missing_direct_field_counts: Counter[str] = Counter()
    missing_quality_field_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    scopes: Counter[str] = Counter()
    invalid_records: list[Any] = []
    valid_speedup_pairs = 0

    for record, audit in zip(sample_records, audits):
        scope = str(record.get("measurement_scope") or "unknown")
        scopes[scope] += 1
        for field in audit["timing"].get("missing", []):
            missing_field_counts[field] += 1
        for issue in audit.get("issues", []):
            issue_counts[str(issue)] += 1

        hard_issues = [
            issue
            for issue in audit.get("issues", [])
            if issue not in _DERIVED_ISSUES
        ]
        if record.get("status", "success") == "success":
            direct_missing = [
                field
                for field in required_direct_metrics(
                    baseline, measurement_scope=scope
                )
                if not _is_present(record.get(field))
            ]
            for field in direct_missing:
                missing_direct_field_counts[field] += 1
                issue_counts[f"missing_{field}"] += 1
                hard_issues.append(f"missing_{field}")

            quality_missing = _missing_quality_fields(record)
            for field in quality_missing:
                missing_quality_field_counts[field] += 1
            if quality_missing and "missing_quality_metric" not in audit.get(
                "issues", []
            ):
                issue_counts["missing_quality_metric"] += 1
                hard_issues.append("missing_quality_metric")

        legacy_sglang_e2e_only = (
            baseline in {"domino", "dspark"}
            and expected_scope == "e2e_plus_decode"
            and scope == "e2e_only"
        )
        if (
            expected_scope is not None
            and scope != expected_scope
            and not legacy_sglang_e2e_only
        ):
            issue_counts["unexpected_measurement_scope"] += 1
            hard_issues.append("unexpected_measurement_scope")
        if hard_issues:
            invalid_records.append(record.get("sample_id"))
        if record.get("speedup_valid") is True:
            valid_speedup_pairs += 1

    observed_count = len(sample_records)
    if observed_count != int(expected_samples):
        issue_counts["sample_count_mismatch"] += 1
    if len(set(sample_ids)) != len(sample_ids):
        issue_counts["duplicate_sample_id"] += len(duplicate_ids)
    if expected_scope is None:
        issue_counts["unknown_baseline_scope"] += 1

    hard_failure = bool(
        invalid_records
        or observed_count != int(expected_samples)
        or duplicate_ids
        or expected_scope is None
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "metric_incomplete" if hard_failure else "complete",
        "baseline": baseline,
        "expected_scope": expected_scope,
        "observed_samples": observed_count,
        "expected_samples": int(expected_samples),
        "unique_sample_ids": len(set(sample_ids)),
        "duplicate_sample_ids": duplicate_ids,
        "scope_counts": dict(scopes),
        "missing_field_counts": dict(missing_field_counts),
        "required_direct_metrics": list(
            required_direct_metrics(baseline, measurement_scope=expected_scope)
        ),
        "missing_direct_field_counts": dict(missing_direct_field_counts),
        "missing_quality_field_counts": dict(missing_quality_field_counts),
        "issue_counts": dict(issue_counts),
        "invalid_sample_ids": invalid_records[:20],
        "valid_speedup_pairs": valid_speedup_pairs,
        "speedup_pair_ratio": round(valid_speedup_pairs / observed_count, 4)
        if observed_count
        else 0.0,
        "decode_metrics_available": bool(sample_records) and all(
            _has_verified_decode_phase(record)
            for record in sample_records
        ),
        "records": audits,
    }


def audit_output_file(
    output_path: Path,
    *,
    baseline: str,
    dataset: str,
    audit_path: Path,
    expected_output_tokens: int | None = None,
    expected_samples: int | None = None,
) -> dict[str, Any]:
    """Attach audits to a JSONL output and write a sidecar audit JSON file."""

    output_path = Path(output_path)
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audits: list[dict[str, Any]] = []
    for row in rows:
        if row.get("type") == "summary":
            continue
        audit = audit_record(row, expected_output_tokens=expected_output_tokens)
        row["metric_audit"] = audit
        audits.append(audit)
    summary = summarize_audits(audits)
    metric_contract = None
    if expected_samples is not None:
        metric_contract = validate_cell_metric_contract(
            rows,
            baseline=baseline,
            expected_samples=expected_samples,
            expected_output_tokens=expected_output_tokens,
        )
    for row in rows:
        if row.get("type") == "summary":
            row["metric_audit_summary"] = summary
            if metric_contract is not None:
                row["metric_contract"] = metric_contract
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "baseline": baseline,
        "dataset": dataset,
        "output": str(output_path),
        "summary": summary,
        "records": audits,
    }
    if metric_contract is not None:
        payload["metric_contract"] = metric_contract
        summary["metric_contract"] = metric_contract
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def format_audit_log(
    *, baseline: str, dataset: str, summary: Mapping[str, Any], audit_path: Path
) -> str:
    """Format one grep-friendly cell line for the live benchmark log."""

    scopes = ",".join(
        f"{key}:{value}" for key, value in sorted((summary.get("scope_counts") or {}).items())
    ) or "none"
    timing = summary.get("timing_field_coverage") or {}
    timing_text = ",".join(
        f"{key}:{value.get('present', 0)}/{value.get('observed', 0)}"
        for key, value in sorted(timing.items())
    ) or "none"
    quality = summary.get("quality_records", 0)
    issue_total = sum((summary.get("issue_counts") or {}).values())
    sources = ",".join(
        f"{key}:{value}"
        for key, value in sorted((summary.get("timing_source_counts") or {}).items())
    ) or "unknown"
    return (
        f"[metrics-audit] {baseline}/{dataset} "
        f"records={summary.get('num_records', 0)} "
        f"status={summary.get('status_counts', {})} scopes={scopes} "
        f"timing={timing_text} quality={quality} "
        f"source={sources} issues={issue_total} audit={audit_path}"
    )
