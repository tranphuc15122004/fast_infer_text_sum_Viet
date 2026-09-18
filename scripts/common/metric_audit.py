"""Machine-readable metric coverage and failure diagnostics.

This module is deliberately post-processing only.  It inspects normalized
benchmark records and never changes model inputs, generation settings, or
timings.  The runner uses it to make missing metrics visible instead of
silently presenting a sparse table.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA_VERSION = 1

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


def audit_output_file(
    output_path: Path,
    *,
    baseline: str,
    dataset: str,
    audit_path: Path,
    expected_output_tokens: int | None = None,
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
    for row in rows:
        if row.get("type") == "summary":
            row["metric_audit_summary"] = summary
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
