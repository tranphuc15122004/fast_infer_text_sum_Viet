"""Input-token utilities shared by model inference adapters."""

from __future__ import annotations

from typing import Any

import torch


def truncate_input_ids(
    input_ids: torch.Tensor,
    max_tokens: int,
    *,
    suffix_tokens: int = 256,
) -> torch.Tensor:
    """Cap a token sequence while preserving the prompt suffix.

    LongBench prompts put the task instruction after the document (for
    example ``Summary:``, ``Answer:`` or ``Next line of code:``). A plain
    right-side truncation silently removes that instruction. Keep a head
    segment for document context and a tail segment for the instruction.
    """

    if input_ids.ndim != 2:
        raise ValueError(
            f"expected [batch, sequence] input_ids, got {tuple(input_ids.shape)}"
        )
    limit = int(max_tokens)
    if limit <= 0 or input_ids.shape[1] <= limit:
        return input_ids
    if limit < 2:
        return input_ids[:, -limit:]

    tail = min(max(int(suffix_tokens), 1), max(limit // 2, 1))
    head = limit - tail
    return torch.cat((input_ids[:, :head], input_ids[:, -tail:]), dim=1)


def truncate_encoded(
    encoded: Any,
    max_tokens: int,
    *,
    suffix_tokens: int = 256,
) -> Any:
    """Apply :func:`truncate_input_ids` to a tokenizer BatchEncoding."""

    if max_tokens <= 0 or not hasattr(encoded, "input_ids"):
        return encoded
    ids = encoded.input_ids
    if ids.ndim != 2 or ids.shape[1] <= max_tokens:
        return encoded
    encoded.input_ids = truncate_input_ids(
        ids, max_tokens, suffix_tokens=suffix_tokens
    )
    if hasattr(encoded, "attention_mask") and encoded.attention_mask is not None:
        encoded.attention_mask = torch.ones_like(encoded.input_ids)
    return encoded


__all__ = ["truncate_input_ids", "truncate_encoded"]
