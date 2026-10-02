"""Quality and anomaly audit for teacher trajectory JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .run_logging import ProgressReporter, count_jsonl_records

logger = logging.getLogger(__name__)


def _tokens(text: str) -> list[str]:
    return text.casefold().split()


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if overlap <= 0 or predicted <= 0 or reference <= 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2.0 * precision * recall / (precision + recall)


def _rouge_n(prediction: list[str], reference: list[str], n: int) -> float:
    if len(prediction) < n or len(reference) < n:
        return 0.0
    predicted_ngrams = Counter(tuple(prediction[i : i + n]) for i in range(len(prediction) - n + 1))
    reference_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
    overlap = sum((predicted_ngrams & reference_ngrams).values())
    return _f1(overlap, sum(predicted_ngrams.values()), sum(reference_ngrams.values()))


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    return {
        "rouge1_f": _rouge_n(pred_tokens, ref_tokens, 1),
        "rouge2_f": _rouge_n(pred_tokens, ref_tokens, 2),
        "rougeL_f": _f1(
            _lcs_length(pred_tokens, ref_tokens),
            len(pred_tokens),
            len(ref_tokens),
        ),
    }


def compute_distinct_n(tokens: list[str], n: int) -> float:
    """Compute Distinct-N metric (ratio of unique n-grams to total n-grams)."""
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def compute_repetition_rate(tokens: list[str], n: int = 3) -> float:
    """Compute n-gram repetition rate: 1.0 - Distinct-N."""
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def detect_mojibake(text: str) -> bool:
    """Detect presence of replacement character or obvious mojibake markers."""
    if "\ufffd" in text:
        return True
    suspicious_substrings = ("Ã¡", "Ã ", "Ã£", "Ã©", "Ã¨", "Ã­", "Ã³", "Ã²", "Ãº", "Ã¹", "Ä‘", "áº", "á»")
    return any(sub in text for sub in suspicious_substrings)


@dataclass
class AnomalyCriteria:
    """Configurable thresholds for detecting target degeneration or corruption."""
    min_summary_words: int = 5
    max_summary_to_doc_ratio: float = 1.0
    max_repetition_rate_3gram: float = 0.35
    max_repetition_rate_4gram: float = 0.25
    check_mojibake: bool = True
    flag_zero_rouge: bool = True


@dataclass
class RecordAudit:
    """Audit result for a single teacher JSONL record."""
    id: str
    is_valid: bool
    reasons: list[str] = field(default_factory=list)
    summary_words: int = 0
    document_words: int = 0
    reference_words: int = 0
    rouge1_f: float = 0.0
    rouge2_f: float = 0.0
    rougeL_f: float = 0.0
    repetition_rate_3gram: float = 0.0
    repetition_rate_4gram: float = 0.0


def audit_record(
    record: Mapping[str, Any],
    criteria: AnomalyCriteria | None = None,
) -> RecordAudit:
    """Audit one record against anomaly and degeneration criteria."""
    if criteria is None:
        criteria = AnomalyCriteria()

    rec_id = str(record.get("id", "unknown"))
    document = str(record.get("document", ""))
    summary = str(record.get("summary", ""))
    reference = str(record.get("reference_summary", record.get("summary_gold", "")))

    reasons: list[str] = []
    
    # Check missing or empty fields
    if not summary or not summary.strip():
        reasons.append("empty_summary")
    if not document or not document.strip():
        reasons.append("empty_document")

    doc_tokens = _tokens(document)
    sum_tokens = _tokens(summary)
    ref_tokens = _tokens(reference)

    sum_words = len(sum_tokens)
    doc_words = len(doc_tokens)
    ref_words = len(ref_tokens)

    # Word count checks
    if sum_words > 0 and sum_words < criteria.min_summary_words:
        reasons.append(f"too_short ({sum_words} < {criteria.min_summary_words} words)")

    if doc_words > 0 and (sum_words / doc_words) > criteria.max_summary_to_doc_ratio:
        reasons.append(f"too_long (summary/document ratio {sum_words/doc_words:.2f} > {criteria.max_summary_to_doc_ratio})")

    # Repetition loop checks
    rep_3g = compute_repetition_rate(sum_tokens, 3) if sum_words >= 3 else 0.0
    rep_4g = compute_repetition_rate(sum_tokens, 4) if sum_words >= 4 else 0.0

    if sum_words >= 10 and rep_3g > criteria.max_repetition_rate_3gram:
        reasons.append(f"repetition_loop_3gram ({rep_3g:.2f} > {criteria.max_repetition_rate_3gram})")
    elif sum_words >= 10 and rep_4g > criteria.max_repetition_rate_4gram:
        reasons.append(f"repetition_loop_4gram ({rep_4g:.2f} > {criteria.max_repetition_rate_4gram})")

    # Mojibake check
    if criteria.check_mojibake and (detect_mojibake(summary) or detect_mojibake(document)):
        reasons.append("mojibake_detected")

    # ROUGE metrics vs gold reference (if reference exists)
    rouge_res = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    if sum_words > 0 and ref_words > 0:
        rouge_res = compute_rouge(summary, reference)
        if criteria.flag_zero_rouge and rouge_res["rouge1_f"] == 0.0:
            reasons.append("zero_rouge1_with_reference")

    is_valid = len(reasons) == 0
    return RecordAudit(
        id=rec_id,
        is_valid=is_valid,
        reasons=reasons,
        summary_words=sum_words,
        document_words=doc_words,
        reference_words=ref_words,
        rouge1_f=rouge_res["rouge1_f"],
        rouge2_f=rouge_res["rouge2_f"],
        rougeL_f=rouge_res["rougeL_f"],
        repetition_rate_3gram=rep_3g,
        repetition_rate_4gram=rep_4g,
    )


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def _calc_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p10": 0.0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 2),
        "p10": round(_percentile(values, 10), 2),
        "p25": round(_percentile(values, 25), 2),
        "median": round(_percentile(values, 50), 2),
        "mean": round(sum(values) / len(values), 2),
        "p75": round(_percentile(values, 75), 2),
        "p90": round(_percentile(values, 90), 2),
        "max": round(max(values), 2),
    }


@dataclass
class ValidationReport:
    """Full aggregate validation report."""
    input_file: str
    total_records: int
    valid_records: int
    anomalous_records: int
    anomaly_rate: float
    passed_gate: bool
    gate_max_anomaly_rate: float
    gate_min_rouge1: float
    avg_rouge1_f: float
    avg_rouge2_f: float
    avg_rougeL_f: float
    summary_words_stats: dict[str, float]
    document_words_stats: dict[str, float]
    summary_to_doc_ratio_stats: dict[str, float]
    anomaly_breakdown: dict[str, int]
    sample_anomalies: list[dict[str, Any]]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_teacher_jsonl(
    input_path: str | Path,
    *,
    report_path: str | Path | None = None,
    clean_output_path: str | Path | None = None,
    filter_anomalies: bool = False,
    max_anomaly_rate: float = 0.05,
    min_rouge1: float = 0.0,
    criteria: AnomalyCriteria | None = None,
    warn_only: bool = False,
    progress: ProgressReporter | None = None,
) -> ValidationReport:
    """Validate a teacher JSONL file, produce metrics, and optionally filter out anomalies."""
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"teacher JSONL not found: {source}")

    if criteria is None:
        criteria = AnomalyCriteria()

    started = time.perf_counter()
    audits: list[RecordAudit] = []
    anomalous_samples: list[dict[str, Any]] = []
    anomaly_counter: Counter[str] = Counter()

    clean_records: list[dict[str, Any]] = []
    
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if progress is not None:
                progress.update()
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                audit = RecordAudit(
                    id=f"line_{line_number}",
                    is_valid=False,
                    reasons=[f"json_decode_error: {exc.msg}"],
                )
                audits.append(audit)
                anomaly_counter["json_decode_error"] += 1
                anomalous_samples.append({"line": line_number, "id": audit.id, "reasons": audit.reasons})
                continue

            audit = audit_record(payload, criteria)
            audits.append(audit)

            if not audit.is_valid:
                for reason in audit.reasons:
                    category = reason.split()[0]
                    anomaly_counter[category] += 1
                if len(anomalous_samples) < 50:
                    anomalous_samples.append({
                        "line": line_number,
                        "id": audit.id,
                        "reasons": audit.reasons,
                        "summary_preview": str(payload.get("summary", ""))[:120],
                    })
            else:
                if filter_anomalies or clean_output_path is not None:
                    clean_records.append(payload)

    total = len(audits)
    if total == 0:
        raise ValueError(f"input file is empty: {source}")

    valid_count = sum(1 for a in audits if a.is_valid)
    anomalous_count = total - valid_count
    anomaly_rate = anomalous_count / total

    # Compute aggregate metrics
    rouge1_list = [a.rouge1_f for a in audits if a.reference_words > 0]
    rouge2_list = [a.rouge2_f for a in audits if a.reference_words > 0]
    rougeL_list = [a.rougeL_f for a in audits if a.reference_words > 0]

    avg_rouge1 = float(sum(rouge1_list) / len(rouge1_list)) if rouge1_list else 0.0
    avg_rouge2 = float(sum(rouge2_list) / len(rouge2_list)) if rouge2_list else 0.0
    avg_rougeL = float(sum(rougeL_list) / len(rougeL_list)) if rougeL_list else 0.0

    sum_words = [float(a.summary_words) for a in audits]
    doc_words = [float(a.document_words) for a in audits]
    ratios = [
        float(a.summary_words / a.document_words)
        for a in audits
        if a.document_words > 0
    ]

    passed_anomaly_gate = anomaly_rate <= max_anomaly_rate
    passed_rouge_gate = avg_rouge1 >= min_rouge1
    passed_gate = passed_anomaly_gate and passed_rouge_gate

    elapsed = time.perf_counter() - started

    report = ValidationReport(
        input_file=str(source),
        total_records=total,
        valid_records=valid_count,
        anomalous_records=anomalous_count,
        anomaly_rate=round(anomaly_rate, 4),
        passed_gate=passed_gate,
        gate_max_anomaly_rate=max_anomaly_rate,
        gate_min_rouge1=min_rouge1,
        avg_rouge1_f=round(avg_rouge1, 4),
        avg_rouge2_f=round(avg_rouge2, 4),
        avg_rougeL_f=round(avg_rougeL, 4),
        summary_words_stats=_calc_stats(sum_words),
        document_words_stats=_calc_stats(doc_words),
        summary_to_doc_ratio_stats=_calc_stats(ratios),
        anomaly_breakdown=dict(anomaly_counter),
        sample_anomalies=anomalous_samples,
        elapsed_seconds=round(elapsed, 2),
    )

    # Write clean output if requested
    target_clean = Path(clean_output_path) if clean_output_path else None
    if filter_anomalies and target_clean is None:
        target_clean = source

    if target_clean is not None:
        target_clean.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target_clean.name}.", suffix=".tmp", dir=target_clean.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with tmp_path.open("w", encoding="utf-8") as out_h:
                for rec in clean_records:
                    out_h.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(tmp_path, target_clean)
            logger.info("Exported %d clean records to %s", len(clean_records), target_clean)
        finally:
            tmp_path.unlink(missing_ok=True)

    # Save report
    if report_path is not None:
        rep_dest = Path(report_path)
        rep_dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{rep_dest.name}.", suffix=".tmp", dir=rep_dest.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp_path, rep_dest)
        finally:
            tmp_path.unlink(missing_ok=True)

    # Log summary
    logger.info(
        "Teacher Trajectory Validation: %d total, %d valid, %d anomalous (rate: %.2f%%) [Gate: %s]",
        total, valid_count, anomalous_count, anomaly_rate * 100, "PASSED" if passed_gate else "FAILED"
    )
    if anomaly_counter:
        logger.warning("Anomaly breakdown: %s", dict(anomaly_counter))

    if not passed_gate and not warn_only:
        reasons = []
        if not passed_anomaly_gate:
            reasons.append(f"anomaly_rate {anomaly_rate:.2%} > max {max_anomaly_rate:.2%}")
        if not passed_rouge_gate:
            reasons.append(f"avg_rouge1 {avg_rouge1:.4f} < min {min_rouge1:.4f}")
        raise RuntimeError(
            f"Teacher trajectory validation failed gate ({', '.join(reasons)}). "
            f"Review {report_path or 'logs'} for detailed sample errors."
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and audit teacher trajectory JSONL files.")
    parser.add_argument("--input", required=True, help="Path to teacher JSONL file")
    parser.add_argument("--report-path", default=None, help="Path to save JSON validation report")
    parser.add_argument("--clean-output", default=None, help="Path to save sanitized/filtered JSONL")
    parser.add_argument("--filter-anomalies", action="store_true", help="Filter out anomalous records")
    parser.add_argument("--max-anomaly-rate", type=float, default=0.05, help="Maximum allowed anomaly rate before failure")
    parser.add_argument("--min-rouge1", type=float, default=0.0, help="Minimum required average ROUGE-1 F1")
    parser.add_argument("--min-summary-words", type=int, default=5, help="Minimum words in a summary")
    parser.add_argument("--warn-only", action="store_true", help="Log warnings without throwing exit errors")
    parser.add_argument(
        "--progress-total",
        type=int,
        help="known input record count supplied by the run launcher",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    criteria = AnomalyCriteria(
        min_summary_words=args.min_summary_words,
    )

    progress = ProgressReporter(
        f"validate:{Path(args.input).stem}",
        args.progress_total
        if args.progress_total is not None
        else count_jsonl_records(args.input),
    )
    succeeded = False
    try:
        validate_teacher_jsonl(
            input_path=args.input,
            report_path=args.report_path,
            clean_output_path=args.clean_output,
            filter_anomalies=args.filter_anomalies,
            max_anomaly_rate=args.max_anomaly_rate,
            min_rouge1=args.min_rouge1,
            criteria=criteria,
            warn_only=args.warn_only,
            progress=progress,
        )
        succeeded = True
    finally:
        progress.close(complete=succeeded)


if __name__ == "__main__":
    main()
