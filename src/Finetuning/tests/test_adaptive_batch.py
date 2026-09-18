from __future__ import annotations

import pytest

from Finetuning.adaptive_batch import (
    AdaptiveBatchSettings,
    is_cuda_oom,
    search_largest_fitting_batch,
    validate_adaptive_batch_settings,
)


def test_search_finds_largest_fitting_batch_with_binary_refinement() -> None:
    attempted: list[int] = []

    def probe(batch_size: int) -> bool:
        attempted.append(batch_size)
        return batch_size <= 5

    selected = search_largest_fitting_batch(
        minimum=1,
        maximum=16,
        probe=probe,
    )

    assert selected == 5
    assert attempted[:4] == [1, 2, 4, 8]
    assert 5 in attempted


def test_adaptive_settings_reject_invalid_memory_fraction() -> None:
    with pytest.raises(ValueError, match="target_memory_fraction"):
        validate_adaptive_batch_settings(
            AdaptiveBatchSettings(target_memory_fraction=1.0)
        )


def test_cuda_oom_detection_does_not_hide_other_runtime_errors() -> None:
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_cuda_oom(RuntimeError("backend compiler failed"))

