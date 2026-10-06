"""FlashAttention-4 adapter for additive custom attention masks.

EAGLE's target verification uses a per-query tree mask.  The regular
Transformers FlashAttention adapter accepts padding masks, so this module
passes the same allowed-position mask to FA4's CuTe ``mask_mod`` interface.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch


_FA4_MASK_TILE_BOUND = 256


def additive_mask_to_keep(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert a `[batch, 1, query, key]` additive mask into FA4 keep bits."""

    if attention_mask.ndim != 4:
        raise ValueError(
            "FA4 custom attention requires a 4D additive attention mask"
        )
    if attention_mask.shape[1] != 1:
        raise ValueError(
            "FA4 custom attention mask head dimension must be 1"
        )
    if attention_mask.dtype == torch.bool:
        return attention_mask[:, 0].contiguous()
    return (attention_mask[:, 0] >= 0).contiguous()


def pad_keep_mask_for_fa4(keep_mask: torch.Tensor) -> torch.Tensor:
    """Pad a keep-mask with masked positions for FA4's padded tile coordinates."""

    if keep_mask.ndim != 3:
        raise ValueError("FA4 keep mask must have shape [batch, query, key]")
    batch, query_length, key_length = keep_mask.shape
    tile_bound = _FA4_MASK_TILE_BOUND
    padded_query_length = max(
        tile_bound,
        ((query_length + tile_bound - 1) // tile_bound) * tile_bound,
    )
    padded_key_length = max(
        tile_bound,
        ((key_length + tile_bound - 1) // tile_bound) * tile_bound,
    )
    padded = torch.zeros(
        (batch, padded_query_length, padded_key_length),
        dtype=torch.uint8,
        device=keep_mask.device,
    )
    padded[:, :query_length, :key_length] = keep_mask.to(torch.uint8)
    return padded


@lru_cache(maxsize=1)
def _fa4_mask_mod() -> Any:
    """Build the CuTe mask function only inside a FA4-enabled runtime."""

    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    @cute.jit
    def tree_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        # FA4 passes scalar mask coordinates as one-element TensorSSA vectors.
        # CuTe tensor indexing needs scalar coordinates, so extract lane zero
        # before indexing the auxiliary keep-mask (same pattern as FA4's own
        # document-mask example).
        # FA4 calls mask_mod for padded rows/columns too. The caller pads the
        # auxiliary tensor to at least 256x256 and fills padding with zeros, so
        # every coordinate in an FA4 tile remains a valid load.
        allowed = utils.scalar_to_ssa(
            aux_tensors[0][batch_idx[0], q_idx[0], kv_idx[0]], cutlass.Int32
        )
        return allowed != utils.scalar_to_ssa(0, cutlass.Int32)

    return tree_mask


def flashattn4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    sliding_window: int | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """Run FA4 on Q/K/V in `[batch, heads, sequence, dim]` layout.

    For a custom tree mask, values `>= 0` are visible and negative additive
    values are masked. With no custom mask, ``is_causal`` selects causal or
    bidirectional attention.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("FA4 Q/K/V tensors must have four dimensions")
    if key.shape != value.shape:
        raise ValueError("FA4 key and value tensor shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("FA4 query and key batch/head dimensions are incompatible")

    from flash_attn.cute import flash_attn_func

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    window = (sliding_window, 0) if sliding_window is not None else (None, None)
    if attention_mask is None:
        result = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=scaling,
            causal=is_causal,
            window_size=window,
        )
    else:
        expected_shape = (query.shape[0], 1, query.shape[2], key.shape[2])
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                "FA4 custom attention mask shape must match "
                f"[batch, 1, query, key]={expected_shape}, got {tuple(attention_mask.shape)}"
            )
        keep_mask = pad_keep_mask_for_fa4(additive_mask_to_keep(attention_mask))
        try:
            result = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=scaling,
                causal=False,
                window_size=window,
                mask_mod=_fa4_mask_mod(),
                aux_tensors=[keep_mask],
            )
        except Exception as exc:
            raise RuntimeError(
                "FA4 custom-mask attention failed: "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, "
                f"mask={tuple(keep_mask.shape)}, dtype={q.dtype}, "
                f"window={window}, scale={scaling}"
            ) from exc
    # FA4's public function returns (output, logsumexp) even when return_lse
    # is false.  Transformers attention modules consume only the output tensor.
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("FA4 returned an empty result tuple")
        return result[0]
    return result
