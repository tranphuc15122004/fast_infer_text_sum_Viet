"""Shared normalization for speculative decoding acceptance counters."""

from __future__ import annotations

import math
from typing import Any


def normalize_speculative_acceptance(
    *,
    verification_steps: Any,
    draft_tokens_accepted: Any,
    draft_tokens_proposed: Any,
    fallback_acceptance_rate: Any = None,
    fallback_avg_accept_length: Any = None,
) -> dict[str, int | float | None]:
    """Normalize acceptance counters shared by all speculative baselines.

    avg_accept_length means accepted draft tokens plus one target token per
    verification, averaged across verify steps. When the accepted-token counter
    is unavailable, use a native trace/runtime value.

    acceptance_rate is accepted draft candidates divided by all proposed draft
    candidates (a fraction in [0, 1]). EAGLE counts every node in its draft
    tree, while linear/block methods count their proposed draft slots.
    accepted_draft_tokens_per_step is the topology-independent comparison.

    acceptance_rate_percent carries the same rate in percent units with four
    decimals so small but nonzero EAGLE rates remain visible.
    """

    def integer_counter(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    steps = integer_counter(verification_steps)
    accepted = integer_counter(draft_tokens_accepted)
    proposed = integer_counter(draft_tokens_proposed)

    if (
        accepted is not None
        and proposed is not None
        and proposed > 0
        and accepted <= proposed
    ):
        raw_rate = accepted / proposed
    elif draft_tokens_accepted is None and draft_tokens_proposed is None:
        try:
            candidate_rate = float(fallback_acceptance_rate)
        except (TypeError, ValueError, OverflowError):
            candidate_rate = math.nan
        raw_rate = (
            candidate_rate
            if math.isfinite(candidate_rate) and 0.0 <= candidate_rate <= 1.0
            else None
        )
    else:
        raw_rate = None

    if accepted is not None and steps is not None and steps > 0:
        average_accept_length = 1.0 + accepted / steps
    else:
        try:
            candidate_average = float(fallback_avg_accept_length)
        except (TypeError, ValueError, OverflowError):
            candidate_average = math.nan
        average_accept_length = (
            candidate_average
            if math.isfinite(candidate_average) and candidate_average >= 1.0
            else None
        )

    return {
        "verification_steps": steps,
        "draft_tokens_accepted": accepted,
        "draft_tokens_proposed": proposed,
        "avg_accept_length": round(average_accept_length, 4)
        if average_accept_length is not None
        else None,
        "accepted_draft_tokens_per_step": round(accepted / steps, 4)
        if accepted is not None and steps is not None and steps > 0
        else None,
        "acceptance_rate": round(raw_rate, 6) if raw_rate is not None else None,
        "acceptance_rate_percent": round(raw_rate * 100.0, 4)
        if raw_rate is not None
        else None,
        "rejected_draft_ratio": round(1.0 - raw_rate, 6)
        if raw_rate is not None
        else None,
    }
