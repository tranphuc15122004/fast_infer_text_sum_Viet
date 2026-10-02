"""Tests for teacher trajectory validation and quality gate."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from Finetuning.validate_targets import (
    AnomalyCriteria,
    audit_record,
    compute_distinct_n,
    compute_repetition_rate,
    compute_rouge,
    detect_mojibake,
    validate_teacher_jsonl,
)


def test_rouge_and_metrics_calculation() -> None:
    pred = "Chính phủ vừa ban hành nghị định mới về phát triển công nghệ cao."
    ref = "Chính phủ ban hành nghị định mới liên quan đến công nghệ cao."
    rouge = compute_rouge(pred, ref)
    assert 0.0 < rouge["rouge1_f"] <= 1.0
    assert 0.0 < rouge["rouge2_f"] <= 1.0
    assert 0.0 < rouge["rougeL_f"] <= 1.0

    tokens = ["a", "b", "c", "a", "b", "c"]
    dist2 = compute_distinct_n(tokens, 2)
    assert 0.0 < dist2 < 1.0

    rep_rate = compute_repetition_rate(tokens, 3)
    assert rep_rate > 0.0


def test_detect_mojibake() -> None:
    assert detect_mojibake("Văn bản bình thường không lỗi") is False
    assert detect_mojibake("Văn bản chứa ký tự \ufffd bị lỗi") is True
    assert detect_mojibake("Văn bản Ã¡ há»™i") is True


def test_audit_record_clean() -> None:
    clean = {
        "id": "1",
        "document": "Bộ Giáo dục và Đào tạo vừa công bố phương án thi tốt nghiệp THPT từ năm 2025 với nhiều điểm đổi mới quan trọng.",
        "summary": "Bộ Giáo dục công bố phương án thi tốt nghiệp THPT 2025 với các thay đổi mới.",
        "reference_summary": "Phương án thi tốt nghiệp THPT năm 2025 được Bộ Giáo dục ban hành.",
    }
    audit = audit_record(clean)
    assert audit.is_valid is True
    assert len(audit.reasons) == 0
    assert audit.rouge1_f > 0.2


def test_audit_record_anomalies() -> None:
    # 1. Empty summary
    rec1 = {"id": "1", "document": "Văn bản nguồn", "summary": ""}
    audit1 = audit_record(rec1)
    assert audit1.is_valid is False
    assert any("empty_summary" in r for r in audit1.reasons)

    # 2. Too short
    rec2 = {"id": "2", "document": "Văn bản rất dài có nhiều nội dung quan trọng", "summary": "Ngắn"}
    audit2 = audit_record(rec2, AnomalyCriteria(min_summary_words=5))
    assert audit2.is_valid is False
    assert any("too_short" in r for r in audit2.reasons)

    # 3. Repetition loop
    rec3 = {
        "id": "3",
        "document": "Văn bản nguồn dài đầy đủ",
        "summary": "lặp lại lặp lại lặp lại lặp lại lặp lại lặp lại lặp lại lặp lại",
    }
    audit3 = audit_record(rec3)
    assert audit3.is_valid is False
    assert any("repetition_loop" in r for r in audit3.reasons)

    # 4. Too long (summary > document)
    rec4 = {
        "id": "4",
        "document": "Một câu ngắn.",
        "summary": "Một câu tóm tắt dài hơn rất nhiều so với văn bản gốc ban đầu.",
    }
    audit4 = audit_record(rec4)
    assert audit4.is_valid is False
    assert any("too_long" in r for r in audit4.reasons)


def test_validate_teacher_jsonl_pass_and_report(tmp_path: Path) -> None:
    input_file = tmp_path / "teacher.jsonl"
    report_file = tmp_path / "report.json"
    clean_file = tmp_path / "clean.jsonl"

    records = [
        {
            "id": f"rec_{i}",
            "document": f"Đây là tài liệu số {i} với nội dung thông tin đầy đủ và chi tiết về sự kiện hôm nay.",
            "summary": f"Tài liệu {i} tóm tắt thông tin sự kiện hôm nay.",
            "reference_summary": f"Sự kiện hôm nay trong tài liệu {i}.",
        }
        for i in range(20)
    ]
    # Add 1 anomaly (5% of 21 records)
    records.append({
        "id": "bad_rec",
        "document": "Tài liệu bị lỗi",
        "summary": "",
        "reference_summary": "Tài liệu",
    })

    with input_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = validate_teacher_jsonl(
        input_path=input_file,
        report_path=report_file,
        clean_output_path=clean_file,
        filter_anomalies=True,
        max_anomaly_rate=0.10,  # 1/21 ~ 4.76% <= 10% -> PASS
    )

    assert report.total_records == 21
    assert report.valid_records == 20
    assert report.anomalous_records == 1
    assert report.passed_gate is True
    assert report_file.is_file()
    assert clean_file.is_file()

    # Clean file should only have 20 records
    with clean_file.open("r", encoding="utf-8") as f:
        clean_lines = [json.loads(line) for line in f if line.strip()]
    assert len(clean_lines) == 20
    assert not any(r["id"] == "bad_rec" for r in clean_lines)


def test_validate_teacher_jsonl_fail_gate(tmp_path: Path) -> None:
    input_file = tmp_path / "bad_teacher.jsonl"
    records = [
        {
            "id": f"rec_{i}",
            "document": f"Tài liệu {i}",
            "summary": "quá ngắn",
            "reference_summary": f"Tóm tắt {i}",
        }
        for i in range(10)
    ]
    with input_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 100% anomaly -> exceeds max_anomaly_rate 0.05 -> should raise RuntimeError
    with pytest.raises(RuntimeError, match="Teacher trajectory validation failed gate"):
        validate_teacher_jsonl(
            input_path=input_file,
            max_anomaly_rate=0.05,
            warn_only=False,
        )

    # With warn_only=True, it should not raise
    report = validate_teacher_jsonl(
        input_path=input_file,
        max_anomaly_rate=0.05,
        warn_only=True,
    )
    assert report.passed_gate is False
