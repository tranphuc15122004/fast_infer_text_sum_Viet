"""Runtime patch for EAGLE's Qwen3 target tree-verification attention."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from Benchmark.common.flashattn4_tree_attention import flashattn4_attention
from Benchmark.common.flashattn_runtime import record_attention_dispatch


def install_eagle_fa4_attention(
    modeling_module: Any | None = None,
    *,
    attention_fn: Callable[..., torch.Tensor] = flashattn4_attention,
) -> None:
    """Route EAGLE Qwen3 attention through FA4 while preserving its tree mask."""

    if modeling_module is None:
        from eagle.model import modeling_qwen3_kv as modeling_module

    attention_class = modeling_module.Qwen3Attention
    current_forward = attention_class.forward
    if getattr(current_forward, "_fast_infer_fa4_tree_mask", False):
        return
    original_forward = current_forward
    apply_rotary_pos_emb = modeling_module.apply_rotary_pos_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        if getattr(self.config, "_attn_implementation", None) != "flash_attention_4":
            record_attention_dispatch(
                str(getattr(self.config, "_attn_implementation", None) or "eager"),
                module=self,
                role="target",
            )
            return original_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
        if kwargs.get("output_attentions", False):
            raise ValueError("EAGLE FA4 attention does not return attention weights")
        record_attention_dispatch("flash_attention_4", module=self, role="target")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        cached_length = 0
        if past_key_value is not None:
            cached_length = int(past_key_value[0].shape[-2])
            key_states = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)

        # Match Transformers' fast FA4 path for ordinary prefill and one-token
        # decode. EAGLE's tree mask is needed only when a multi-token verify
        # block follows a non-empty cached prefix. Inputs in this batch-one
        # runner are unpadded, so prefill's mask is pure causal and q_len=1 can
        # attend to the full prefix through bottom-right causal semantics.
        if attention_mask is not None and (
            (past_key_value is not None and input_shape[-1] > 1 and cached_length == 0)
            or (past_key_value is not None and input_shape[-1] == 1)
        ):
            attention_mask = None

        attention_output = attention_fn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        attention_output = self.o_proj(attention_output)
        return attention_output, None, None

    forward._fast_infer_fa4_tree_mask = True
    attention_class.forward = forward


def install_eagle_draft_fa4_attention(
    modeling_module: Any | None = None,
    *,
    attention_fn: Callable[..., torch.Tensor] = flashattn4_attention,
) -> None:
    """Route EAGLE-3's vendored draft attention through the same FA4 mask path."""

    if modeling_module is None:
        from eagle.model import cnets as modeling_module

    attention_class = modeling_module.LlamaAttention
    current_forward = attention_class.forward
    if getattr(current_forward, "_fast_infer_fa4_tree_mask", False):
        return
    original_forward = current_forward
    apply_rotary_pos_emb = modeling_module.apply_rotary_pos_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        if getattr(self.config, "_attn_implementation", None) != "flash_attention_4":
            record_attention_dispatch(
                str(getattr(self.config, "_attn_implementation", None) or "eager"),
                module=self,
                role="draft",
            )
            return original_forward(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        if output_attentions:
            raise ValueError("EAGLE FA4 attention does not return attention weights")
        record_attention_dispatch("flash_attention_4", module=self, role="draft")

        batch_size, query_length, _ = hidden_states.shape
        if self.config.pretraining_tp > 1:
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp,
                dim=0,
            )
            key_slices = self.k_proj.weight.split(
                (self.num_key_value_heads * self.head_dim)
                // self.config.pretraining_tp,
                dim=0,
            )
            value_slices = self.v_proj.weight.split(
                (self.num_key_value_heads * self.head_dim)
                // self.config.pretraining_tp,
                dim=0,
            )
            query_states = torch.cat(
                [torch.nn.functional.linear(hidden_states, part) for part in query_slices],
                dim=-1,
            )
            key_states = torch.cat(
                [torch.nn.functional.linear(hidden_states, part) for part in key_slices],
                dim=-1,
            )
            value_states = torch.cat(
                [torch.nn.functional.linear(hidden_states, part) for part in value_slices],
                dim=-1,
            )
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        key_length = key_states.shape[-2]
        if past_key_value is not None:
            key_length += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=key_length)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        present_key_value = (
            (key_states, value_states) if use_cache else None
        )

        attention_output = attention_fn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.head_dim**-0.5,
            sliding_window=None,
        )
        attention_output = attention_output.reshape(
            batch_size, query_length, self.num_heads * self.head_dim
        )
        if self.config.pretraining_tp > 1:
            projection_width = self.num_heads * self.head_dim
            split_width = projection_width // self.config.pretraining_tp
            output_slices = attention_output.split(split_width, dim=2)
            projection_slices = self.o_proj.weight.split(split_width, dim=1)
            attention_output = sum(
                torch.nn.functional.linear(output_slices[index], projection_slices[index])
                for index in range(self.config.pretraining_tp)
            )
        else:
            attention_output = self.o_proj(attention_output)
        return attention_output, None, present_key_value

    forward._fast_infer_fa4_tree_mask = True
    attention_class.forward = forward
