"""Self-contained SpecForge-compatible DFlash training objective."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed torch build.
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None

# Flex Attention is not available on Ascend NPU.
if hasattr(torch, "npu") and torch.npu.is_available():  # pragma: no cover
    FLEX_ATTENTION_AVAILABLE = False


_VALID_ATTENTION_BACKENDS = {"eager", "sdpa", "flex_attention"}
_VALID_LOSS_TYPES = {
    "dflash",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_DPACE_LOSS_TYPES = _VALID_LOSS_TYPES - {"dflash"}


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute the accepted prefix length for every parallel draft block."""

    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def create_dflash_sdpa_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Construct the dense boolean attention mask used by eager/SDPA.

    The first ``S`` key/value positions are the real context. The remaining
    positions are parallel draft blocks. Context visibility is strict
    (``kv_idx < anchor``). Full-attention layers allow every draft offset in
    the query's own block; sliding-window layers additionally use causal draft
    offsets and a bounded context window.
    """

    batch_size, num_blocks = anchor_positions.shape
    query_length = num_blocks * block_size
    key_value_length = S + num_blocks * block_size

    q_indices = torch.arange(query_length, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(key_value_length, device=device).view(1, 1, 1, -1)
    q_block_ids = q_indices // block_size
    q_block_offsets = q_indices % block_size

    anchor_expanded = anchor_positions.view(batch_size, 1, num_blocks, 1)
    anchor_expanded = anchor_expanded.repeat_interleave(block_size, dim=2)

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    if sliding_window is not None:
        context_lower_bound = (
            anchor_expanded + q_block_offsets - (sliding_window - 1)
        )
        mask_context = mask_context & (kv_indices >= context_lower_bound)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)
    if sliding_window is not None:
        kv_block_offsets = (kv_indices - S) % block_size
        mask_draft = mask_draft & (kv_block_offsets <= q_block_offsets)

    valid_block = block_keep_mask.view(batch_size, 1, num_blocks, 1)
    valid_block = valid_block.repeat_interleave(block_size, dim=2)
    return (mask_context | mask_draft) & valid_block


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> Any:
    """Construct a Flex Attention ``BlockMask`` with DFlash visibility."""

    if not FLEX_ATTENTION_AVAILABLE or create_block_mask is None:
        raise ValueError(
            "flex_attention is not available on this device; use sdpa/eager."
        )

    batch_size, num_blocks = anchor_positions.shape
    query_length = num_blocks * block_size
    key_value_length = S + num_blocks * block_size

    def dflash_mask_mod(b, _h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        q_block_offset = q_idx % block_size
        safe_q_block_id = q_block_id.clamp(max=num_blocks - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        mask_context = is_context & (kv_idx < anchor_pos)
        if sliding_window is not None:
            context_lower_bound = (
                anchor_pos + q_block_offset - (sliding_window - 1)
            )
            mask_context = mask_context & (kv_idx >= context_lower_bound)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        if sliding_window is not None:
            kv_block_offset = (kv_idx - S) % block_size
            mask_draft = mask_draft & (kv_block_offset <= q_block_offset)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < num_blocks
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    return create_block_mask(
        dflash_mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_value_length,
        device=device,
    )


def _sum_chunk_terms(
    term_fn: Callable[..., Tuple[torch.Tensor, ...]],
    args: Tuple[Optional[torch.Tensor], ...],
    chunk_size: int,
) -> Tuple[torch.Tensor, ...]:
    """Sum aligned objective terms with upstream checkpointing semantics."""

    if chunk_size < 0:
        raise ValueError(f"chunk_size must be >= 0, got {chunk_size}")
    tensors = tuple(value for value in args if value is not None)
    if not tensors:
        raise ValueError("chunked reduction requires at least one tensor")

    first = tensors[0]
    if first.ndim <= 1:
        raise ValueError("DFlash chunk reduction requires a block dimension at dim 1")
    num_blocks = first.shape[1]
    if num_blocks == 0:
        raise ValueError("chunked reduction received an empty dimension")
    for tensor in tensors[1:]:
        if tensor.ndim <= 1:
            raise ValueError(
                "DFlash chunk reduction requires a block dimension at dim 1"
            )
        if tensor.shape[1] != num_blocks:
            raise ValueError(
                "chunked reduction inputs must be aligned: "
                f"expected dimension length {num_blocks}, got {tensor.shape[1]}"
            )

    effective_chunk_size = chunk_size or num_blocks
    totals: Optional[Tuple[torch.Tensor, ...]] = None
    for start in range(0, num_blocks, effective_chunk_size):
        width = min(effective_chunk_size, num_blocks - start)
        chunk_args = tuple(
            value.narrow(1, start, width) if value is not None else None
            for value in args
        )
        should_checkpoint = (
            chunk_size > 0
            and torch.is_grad_enabled()
            and any(value is not None and value.requires_grad for value in chunk_args)
        )
        if should_checkpoint:
            from torch.utils.checkpoint import checkpoint

            chunk_terms = checkpoint(
                term_fn,
                *chunk_args,
                use_reentrant=False,
            )
        else:
            chunk_terms = term_fn(*chunk_args)

        if not isinstance(chunk_terms, tuple) or not all(
            isinstance(term, torch.Tensor) for term in chunk_terms
        ):
            raise TypeError("chunk function must return a tuple of tensors")
        if totals is None:
            totals = chunk_terms
            continue
        if len(totals) != len(chunk_terms):
            raise ValueError("chunk function returned a different number of terms")
        totals = tuple(left + right for left, right in zip(totals, chunk_terms))

    assert totals is not None
    return totals


class OnlineDFlashModel(nn.Module):
    """DFlash block-parallel training wrapper with optional D-PACE modes."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
            )
        if attention_backend not in _VALID_ATTENTION_BACKENDS:
            raise ValueError(
                "attention_backend must be one of "
                f"{sorted(_VALID_ATTENTION_BACKENDS)}, got {attention_backend!r}"
            )
        if block_size < 2:
            raise ValueError(
                "block_size must be at least 2 because offset 0 is excluded "
                f"from the loss, got {block_size}"
            )
        if num_anchors <= 0:
            raise ValueError(f"num_anchors must be positive, got {num_anchors}")
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks must be >= 0")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self.attention_backend = attention_backend
        self.num_anchors = int(num_anchors)
        self.loss_decay_gamma = loss_decay_gamma
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.loss_type = loss_type
        self.dpace_alpha = dpace_alpha

        # The target is a teacher. Its parameters must not be part of the
        # optimization graph or a draft checkpoint, while its outputs remain
        # differentiable with respect to draft hidden states.
        self.lm_head.requires_grad_(False)
        self.embed_tokens.requires_grad_(False)

        self._cached_block_mask = None
        self._cached_seq_len = None
        self._cached_bsz = None

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchors whose token and immediate target are supervised."""

        num_candidates = max(seq_len - 1, 0)
        valid = (loss_mask[:, :num_candidates] > 0.5) & (
            loss_mask[:, 1 : num_candidates + 1] > 0.5
        )
        valid_counts = valid.sum(dim=1)
        width = min(self.num_anchors, int(valid_counts.max().item()))
        if width == 0:
            raise ValueError(
                "DFlash-family training requires two consecutive supervised tokens"
            )

        random_values = torch.rand(valid.shape, device=device)
        random_values.masked_fill_(~valid, 2.0)
        candidates = random_values.argsort(dim=1)[:, :width]
        keep_mask = torch.arange(width, device=device).unsqueeze(0) < (
            valid_counts.clamp(max=width).unsqueeze(1)
        )

        sentinel = valid.shape[1]
        anchors = torch.where(
            keep_mask,
            candidates,
            torch.full_like(candidates, sentinel),
        )
        anchors = anchors.sort(dim=1).values
        keep_mask = anchors < sentinel
        return torch.where(keep_mask, anchors, 0), keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute positions for all parallel draft blocks."""

        batch_size, _num_blocks = anchor_positions.shape
        offsets = torch.arange(
            self.block_size,
            device=anchor_positions.device,
        ).view(1, 1, -1)
        position_ids = anchor_positions.unsqueeze(-1) + offsets
        return position_ids.view(batch_size, -1)

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        num_blocks = anchor_positions.shape[1]
        device = input_ids.device

        noise_ids = torch.full(
            (batch_size, num_blocks * self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        block_starts = torch.arange(num_blocks, device=device) * self.block_size
        block_starts = block_starts.unsqueeze(0).expand(batch_size, -1)

        safe_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, safe_anchor_positions)
        batch_indices = torch.arange(device=device, end=batch_size).unsqueeze(1)
        batch_indices = batch_indices.expand(batch_size, num_blocks)
        noise_ids[batch_indices, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.full_like(anchor_tokens, self.mask_token_id),
        )
        return self.embed_tokens(noise_ids)

    def _build_targets_and_weight_mask(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build same-position labels and the exact DFlash loss weights."""

        seq_len = input_ids.shape[1]
        offsets = torch.arange(
            self.block_size,
            device=input_ids.device,
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        weight_mask = block_keep_mask.unsqueeze(-1).expand(
            -1,
            -1,
            self.block_size,
        ).float()
        weight_mask = weight_mask * valid_label_mask.float()
        position_offsets = torch.arange(
            self.block_size,
            device=input_ids.device,
        ).view(1, 1, -1)
        # Offset 0 supplies the anchor embedding but is never supervised.
        weight_mask = weight_mask * (position_offsets > 0).float()

        original_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask
        return target_ids, weight_mask

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Compute detached D-PACE cumulative confidence/value weights."""

        smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
        smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)

        if loss_type == "dpace-cumulative-confidence-only":
            return prefix

        suffix = torch.flip(
            torch.cumsum(
                torch.flip(prefix * binary_mask, dims=[-1]),
                dim=-1,
            ),
            dims=[-1],
        )
        if loss_type == "dpace":
            return suffix
        if loss_type == "dpace-continuation-value-only":
            return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
        )
        noise_embedding = self._create_noise_embed(
            input_ids,
            anchor_positions,
            block_keep_mask,
        )

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        context_position_ids = context_position_ids.expand(batch_size, -1)
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat(
            [context_position_ids, draft_position_ids],
            dim=1,
        )

        if self.attention_backend == "flex_attention":
            mask_builder = create_dflash_block_mask
        else:
            mask_builder = create_dflash_sdpa_mask
        mask_args = {
            "anchor_positions": anchor_positions,
            "block_keep_mask": block_keep_mask,
            "S": seq_len,
            "block_size": self.block_size,
            "device": device,
        }
        full_attention_mask = mask_builder(**mask_args)
        sliding_window = getattr(self.draft_model, "sliding_window", None)
        dflash_attention_mask = full_attention_mask
        if sliding_window is not None:
            dflash_attention_mask = {
                "full_attention": full_attention_mask,
                "sliding_attention": mask_builder(
                    **mask_args,
                    sliding_window=sliding_window,
                ),
            }

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attention_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden

    def _dflash_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Return additive loss and metric terms for a block chunk."""

        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        neg_log_q = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)

        if self.loss_type == "dflash":
            loss_weights = weight_mask
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                positions = torch.arange(
                    self.block_size,
                    device=hidden.device,
                ).view(1, 1, -1)
                decay_weights = torch.exp(
                    -(positions - 1).clamp(min=0).float()
                    / self.loss_decay_gamma
                )
                loss_weights = loss_weights * decay_weights
            loss_num = (neg_log_q * loss_weights).sum()
            loss_den = loss_weights.sum()
        elif self.loss_type in _DPACE_LOSS_TYPES:
            with torch.no_grad():
                target_probability = torch.exp(-neg_log_q)
                dpace_weights = self._dpace_weight(
                    target_probability,
                    weight_mask,
                    weight_mask > 0,
                    self.loss_type,
                )
            loss_num = (neg_log_q * weight_mask * dpace_weights).sum()
            loss_den = loss_num.new_zeros(())
        else:  # Defensive; __init__ rejects this path.
            raise ValueError(f"unknown loss_type {self.loss_type!r}")

        with torch.no_grad():
            predicted_ids = logits.argmax(dim=-1)
            correct_num = (
                ((predicted_ids == target_ids) & (weight_mask > 0.5))
                .sum()
                .float()
            )
            accuracy_den = weight_mask.sum()
        return loss_num, loss_den, correct_num, accuracy_den

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Run one block-parallel DFlash objective."""

        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        batch_size, _seq_len = input_ids.shape
        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )
        target_ids, weight_mask = self._build_targets_and_weight_mask(
            input_ids,
            loss_mask,
            anchor_positions,
            block_keep_mask,
        )
        hidden_4d = output_hidden.reshape(
            batch_size,
            anchor_positions.shape[1],
            self.block_size,
            -1,
        )
        loss_num, loss_den, correct_num, accuracy_den = _sum_chunk_terms(
            self._dflash_objective_chunk_terms,
            (hidden_4d, target_ids, weight_mask),
            self.objective_chunk_blocks,
        )

        ratio_metrics = {
            "acc": (correct_num.detach(), accuracy_den.detach()),
        }
        loss_denominator = (
            loss_den
            if self.loss_type == "dflash"
            else loss_num.new_tensor(float(batch_size))
        )
        loss = loss_num / loss_denominator
        metrics: Dict[str, object] = {
            "accuracy_denom": accuracy_den.detach(),
            "ratio_metrics": ratio_metrics,
            "loss_terms": (loss_num, loss_denominator.detach()),
        }
        accuracy = correct_num / accuracy_den
        return loss, accuracy, metrics


__all__ = [
    "FLEX_ATTENTION_AVAILABLE",
    "OnlineDFlashModel",
    "compute_accept_len",
    "create_dflash_block_mask",
    "create_dflash_sdpa_mask",
]
