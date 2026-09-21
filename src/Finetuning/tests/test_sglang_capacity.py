from __future__ import annotations


def test_sglang_capacity_follows_adaptive_batch_when_not_overridden() -> None:
    from Finetuning.capture_features import resolve_sglang_capacity

    assert resolve_sglang_capacity(
        max_length=4096,
        adaptive_max_batch_size=64,
        requested_max_running_requests=0,
        requested_max_total_tokens=0,
    ) == (64, 64 * 4096)


def test_sglang_capacity_keeps_explicit_token_budget() -> None:
    from Finetuning.capture_features import resolve_sglang_capacity

    assert resolve_sglang_capacity(
        max_length=4096,
        adaptive_max_batch_size=64,
        requested_max_running_requests=32,
        requested_max_total_tokens=100000,
    ) == (32, 100000)

