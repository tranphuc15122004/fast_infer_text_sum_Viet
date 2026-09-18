from __future__ import annotations

import pytest
import torch


def test_normalize_capture_rows_preserves_order_and_converts_dtype() -> None:
    from Finetuning.offline_sglang_capture import normalize_capture_rows

    aux_rows = [torch.ones(3, 4, dtype=torch.float32), torch.zeros(2, 4, dtype=torch.float32)]
    last_rows = [torch.ones(3, 2, dtype=torch.float32), torch.zeros(2, 2, dtype=torch.float32)]

    normalized = normalize_capture_rows(
        aux_rows,
        last_rows,
        expected_lengths=[3, 2],
        expected_aux_width=4,
        expected_last_width=2,
        dtype=torch.bfloat16,
    )

    assert [row.shape for row in normalized.aux_rows] == [(3, 4), (2, 4)]
    assert [row.shape for row in normalized.last_rows] == [(3, 2), (2, 2)]
    assert all(row.dtype == torch.bfloat16 for row in normalized.aux_rows)
    assert torch.equal(normalized.aux_rows[0].float(), aux_rows[0])


def test_normalize_capture_rows_rejects_cross_sample_shape_mismatch() -> None:
    from Finetuning.offline_sglang_capture import normalize_capture_rows

    with pytest.raises(ValueError, match="row 1.*length"):
        normalize_capture_rows(
            [torch.zeros(3, 4), torch.zeros(1, 4)],
            [torch.zeros(3, 2), torch.zeros(2, 2)],
            expected_lengths=[3, 2],
            expected_aux_width=4,
            expected_last_width=2,
            dtype=torch.float32,
        )


def test_hidden_state_parity_report_passes_for_identical_rows() -> None:
    from Finetuning.offline_sglang_capture import (
        ParityThresholds,
        compare_hidden_rows,
    )

    rows = [torch.randn(3, 8), torch.randn(2, 8)]
    report = compare_hidden_rows(rows, [row.clone() for row in rows], thresholds=ParityThresholds())

    assert report.passed is True
    assert report.max_abs_error == 0.0
    assert report.min_cosine_similarity == pytest.approx(1.0)


def test_hidden_state_parity_report_fails_for_wrong_values() -> None:
    from Finetuning.offline_sglang_capture import (
        ParityThresholds,
        compare_hidden_rows,
    )

    reference = [torch.ones(2, 4)]
    candidate = [torch.zeros(2, 4)]
    report = compare_hidden_rows(
        reference,
        candidate,
        thresholds=ParityThresholds(max_abs_error=1e-5, min_cosine_similarity=0.99),
    )

    assert report.passed is False
    assert report.max_abs_error == pytest.approx(1.0)
    assert report.min_cosine_similarity == pytest.approx(0.0)


def test_capture_backend_argument_is_exposed() -> None:
    from Finetuning.capture_features import _parser

    args = _parser().parse_args(
        [
            "--input",
            "input.jsonl",
            "--output",
            "features",
            "--target-model-path",
            "model",
            "--target-layer-ids",
            "0",
            "--max-length",
            "16",
            "--max-source-tokens",
            "8",
            "--max-summary-tokens",
            "4",
            "--capture-backend",
            "sglang",
            "--capture-method",
            "dflash",
            "--parity-samples",
            "2",
        ]
    )

    assert args.capture_backend == "sglang"
    assert args.capture_method == "dflash"
    assert args.parity_samples == 2
