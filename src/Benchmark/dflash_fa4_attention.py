"""Route the vendored DFlash Qwen3 draft attention through FlashAttention-4."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

import torch

from Benchmark.common.flashattn4_tree_attention import flashattn4_attention
from Benchmark.common.flashattn_runtime import record_attention_dispatch


_PATCH_MARKER = "_fast_infer_dflash_fa4_dispatch"


def install_dflash_fa4_attention(
    dflash_model: Any,
    *,
    attention_fn: Callable[..., torch.Tensor] = flashattn4_attention,
) -> bool:
    """Replace DFlash's hard-coded SDPA call with FA4 for its draft layers.

    DFlash's custom ``Qwen3DFlashAttention`` calls the Transformers ``sdpa``
    registry entry directly, even when its config says ``flash_attention_4``.
    This narrowly scoped registry wrapper redirects only that custom class;
    every other SDPA consumer keeps the original implementation.
    """

    registry = dflash_model.ALL_ATTENTION_FUNCTIONS
    current = registry["sdpa"]
    if getattr(current, _PATCH_MARKER, False):
        return False

    original_sdpa = current
    attention_class = dflash_model.Qwen3DFlashAttention
    counts = {"fa4_dispatch_calls": 0, "sdpa_fallback_calls": 0}

    @wraps(original_sdpa)
    def dispatch(module, query, key, value, attention_mask, *args, **kwargs):
        if isinstance(module, attention_class):
            config = getattr(module, "config", None)
            backend = getattr(config, "_attn_implementation", None)
            if backend == "flash_attention_4":
                if kwargs.get("output_attentions", False):
                    raise ValueError("DFlash FA4 attention does not return attention weights")
                dropout = float(kwargs.get("dropout", 0.0) or 0.0)
                if dropout != 0.0:
                    raise ValueError("DFlash FA4 attention requires dropout=0")
                scaling = kwargs.get("scaling", getattr(module, "scaling", None))
                if scaling is None:
                    raise ValueError("DFlash FA4 attention requires a scaling factor")
                record_attention_dispatch(
                    "flash_attention_4", module=module, role="draft"
                )
                counts["fa4_dispatch_calls"] += 1
                output = attention_fn(
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling=float(scaling),
                    sliding_window=kwargs.get(
                        "sliding_window", getattr(module, "sliding_window", None)
                    ),
                    is_causal=bool(getattr(module, "is_causal", True)),
                )
                return output, None
            counts["sdpa_fallback_calls"] += 1
            record_attention_dispatch(
                str(backend or "sdpa"), module=module, role="draft"
            )
        return original_sdpa(module, query, key, value, attention_mask, *args, **kwargs)

    setattr(dispatch, _PATCH_MARKER, True)
    setattr(dispatch, "_fast_infer_dflash_fa4_counts", counts)
    setattr(dispatch, "_fast_infer_dflash_fa4_original_sdpa", original_sdpa)
    registry["sdpa"] = dispatch
    return True


def dflash_fa4_attention_stats(dflash_model: Any) -> Mapping[str, int]:
    """Return runtime dispatch counts for the installed DFlash attention shim."""

    dispatch = dflash_model.ALL_ATTENTION_FUNCTIONS["sdpa"]
    counts = getattr(dispatch, "_fast_infer_dflash_fa4_counts", None)
    if counts is None:
        return {"fa4_dispatch_calls": 0, "sdpa_fallback_calls": 0}
    return {key: int(value) for key, value in counts.items()}
