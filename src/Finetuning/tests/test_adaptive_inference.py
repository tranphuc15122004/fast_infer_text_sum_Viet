from __future__ import annotations

import pytest

from Finetuning.adaptive_inference import (
    AdaptiveInferenceSettings,
    PreparedExample,
    length_bucket_batches,
    run_with_oom_backoff,
    validate_adaptive_inference_settings,
)


def test_length_bucket_batches_respects_window_and_token_budget() -> None:
    examples = [PreparedExample(index, length) for index, length in enumerate([9, 2, 8, 3])]

    batches = list(
        length_bucket_batches(
            examples,
            batch_size=3,
            max_tokens=10,
            window=4,
        )
    )

    assert [[item.index for item in batch] for batch in batches] == [[1, 3], [2], [0]]


def test_length_bucket_batches_preserves_source_order_across_windows() -> None:
    examples = [PreparedExample(index, length) for index, length in enumerate([5, 1, 4, 2])]

    batches = list(length_bucket_batches(examples, batch_size=2, window=2))

    assert [item.index for batch in batches for item in batch] == [1, 0, 3, 2]


def test_oom_backoff_retries_with_half_batch() -> None:
    attempts: list[int] = []

    def work(batch_size: int) -> int:
        attempts.append(batch_size)
        if batch_size > 2:
            raise RuntimeError("CUDA out of memory")
        return batch_size

    result = run_with_oom_backoff(
        work,
        initial_batch_size=8,
        minimum_batch_size=1,
    )

    assert result.value == 2
    assert result.batch_size == 2
    assert result.oom_retries == 2
    assert attempts == [8, 4, 2]


def test_oom_backoff_does_not_hide_other_runtime_errors() -> None:
    def work(_batch_size: int) -> None:
        raise RuntimeError("model shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        run_with_oom_backoff(work, initial_batch_size=4, minimum_batch_size=1)


def test_adaptive_settings_validate_ranges() -> None:
    with pytest.raises(ValueError, match="target_memory_fraction"):
        validate_adaptive_inference_settings(
            AdaptiveInferenceSettings(target_memory_fraction=1.0)
        )
