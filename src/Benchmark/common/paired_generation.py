"""Shared timing helper for paired dense/optimized inference."""

from __future__ import annotations

import time
from typing import Any

import torch


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    max_new_tokens: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, float]:
    """Run greedy generation and return ``(output_ids, elapsed_ms)``.

    Synchronization is explicit so asynchronous CUDA launches cannot make an
    optimized method appear faster merely because timing stopped early.
    """

    _synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
            **kwargs,
        )
    _synchronize(device)
    return output, (time.perf_counter() - start) * 1000.0


__all__ = ["timed_generate"]
