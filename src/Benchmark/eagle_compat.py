"""Compatibility helpers for the vendored EAGLE model.

The Qwen3 EAGLE model is generated from a newer Transformers source tree than
the offline server wheel.  Its generated typing-only model code imports
``LossKwargs`` from ``transformers.utils``.  Some supported Transformers
versions do not export that name, even though the class is only used to build
the ``**kwargs`` type annotation for the causal-LM forward method.
"""

import json
from pathlib import Path
from typing import TypedDict


def eagle_model_load_options() -> dict[str, object]:
    """Return model-loading options compatible with the vendored EaModel.

    Transformers returns ``(model, loading_info)`` when
    ``output_loading_info=True``. The vendored ``EaModel.from_pretrained``
    forwards that tuple as if it were the target model, so keep this flag off.
    A CUDA ``device_map`` also sends this custom class through Transformers'
    dispatch/meta-loading path, which can leave its parameters at random
    initialization values. Load on CPU first, then move the finished module to
    CUDA after ``EaModel.from_pretrained`` returns.
    """

    return {
        "low_cpu_mem_usage": False,
        "output_loading_info": False,
    }


def eagle_target_dtype(torch_module, *, device_index: int = 0):
    """Match the dense benchmark's preferred dtype on CUDA accelerators.

    The vendored adapter used FP16 unconditionally while Vanilla and SGLang
    use BF16 on Hopper/Blackwell. That can change greedy argmax choices even
    when checkpoint weights and the initial prompt logits are correct. Keep
    FP16 for pre-Ampere devices and for GPUs without BF16 support.
    """

    capability = torch_module.cuda.get_device_capability(device_index)
    bf16_supported = getattr(torch_module.cuda, "is_bf16_supported", None)
    if (
        capability[0] >= 8
        and callable(bf16_supported)
        and bf16_supported()
    ):
        return torch_module.bfloat16
    return torch_module.float16


def truncate_eagle_generation_at_stop(
    output_ids,
    *,
    prompt_length: int,
    acceptance_lengths,
    stop_token_ids,
):
    """Drop speculative tokens committed after the first generated stop token.

    EAGLE can accept a whole tree branch that contains EOS and tokens after
    EOS in the same verification step.  Its loop notices EOS only after the
    branch is appended, so those trailing special/ordinary tokens otherwise
    leak into output length and paired-greedy metrics.
    """

    import torch

    if output_ids.ndim != 2 or output_ids.shape[0] != 1:
        raise ValueError("EAGLE stop-token truncation expects one 2D sequence")
    if prompt_length < 0 or prompt_length > output_ids.shape[1]:
        raise ValueError("prompt_length is outside the output sequence")
    if isinstance(stop_token_ids, int) and not isinstance(stop_token_ids, bool):
        raw_stop_ids = [stop_token_ids]
    elif isinstance(stop_token_ids, (list, tuple, set)):
        raw_stop_ids = list(stop_token_ids)
    else:
        raw_stop_ids = []
    normalized_stop_ids = {
        int(token_id)
        for token_id in raw_stop_ids
        if isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and token_id >= 0
    }
    trace = [max(1, int(value)) for value in acceptance_lengths]
    if not normalized_stop_ids:
        return output_ids, trace, 0

    generated_ids = output_ids[0, prompt_length:]
    first_stop_offset = None
    for token_id in normalized_stop_ids:
        matches = torch.nonzero(generated_ids == token_id, as_tuple=False)
        if matches.numel():
            offset = int(matches[0, 0].item())
            if first_stop_offset is None or offset < first_stop_offset:
                first_stop_offset = offset
    if first_stop_offset is None:
        return output_ids, trace, 0

    end = prompt_length + first_stop_offset + 1
    truncated = output_ids[:, :end]
    removed = int(output_ids.shape[1] - end)
    retained_output_tokens = end - prompt_length
    excess_trace_tokens = max(0, sum(trace) - retained_output_tokens)
    for index in range(len(trace) - 1, -1, -1):
        if excess_trace_tokens <= 0:
            break
        removable = max(0, trace[index] - 1)
        removed_from_step = min(removable, excess_trace_tokens)
        trace[index] -= removed_from_step
        excess_trace_tokens -= removed_from_step
    # If the trace contains steps with no retained generated token, remove
    # those terminal steps too; a step after EOS cannot count as output.
    while excess_trace_tokens > 0 and trace and trace[-1] <= 1:
        trace.pop()
        excess_trace_tokens -= 1
    if excess_trace_tokens > 0:
        raise ValueError(
            "EAGLE acceptance trace cannot be reconciled with stop-truncated output"
        )
    return truncated, trace, removed


def reload_eagle_target_weights(target_model, checkpoint_path) -> dict[str, int | str]:
    """Copy checkpoint tensors into the vendored Qwen target model explicitly.

    Transformers 5.12.1 reports a clean load for the vendored Qwen3 class but
    leaves many linear/embedding parameters initialized randomly. Loading each
    checkpoint shard through ``load_state_dict`` avoids that silent mismatch.
    """

    import torch

    path = Path(checkpoint_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(str(checkpoint_path)))

    if path.is_file():
        weight_files = [path]
    else:
        index_path = next(
            (
                candidate
                for candidate in (
                    path / "model.safetensors.index.json",
                    path / "pytorch_model.bin.index.json",
                )
                if candidate.is_file()
            ),
            None,
        )
        if index_path is not None:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files = [
                path / name
                for name in sorted(set(index.get("weight_map", {}).values()))
            ]
        else:
            weight_files = [
                candidate
                for candidate in (
                    path / "model.safetensors",
                    path / "pytorch_model.bin",
                )
                if candidate.is_file()
            ]
            if not weight_files:
                weight_files = sorted(path.glob("model-*.safetensors"))
            if not weight_files:
                weight_files = sorted(path.glob("pytorch_model-*.bin"))

    if not weight_files or any(not file.is_file() for file in weight_files):
        raise FileNotFoundError(
            f"No complete safetensors/PyTorch checkpoint found under {path}"
        )

    loaded_keys: set[str] = set()
    for weight_file in weight_files:
        if weight_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(weight_file), device="cpu")
        else:
            state_dict = torch.load(
                weight_file, map_location="cpu", weights_only=True
            )
        if isinstance(state_dict, dict) and isinstance(
            state_dict.get("state_dict"), dict
        ):
            state_dict = state_dict["state_dict"]
        loaded_keys.update(str(key) for key in state_dict)
        target_model.load_state_dict(state_dict, strict=False)
        del state_dict

    parameter_names = {name for name, _ in target_model.named_parameters()}
    missing_parameters = sorted(parameter_names - loaded_keys)
    if missing_parameters:
        raise RuntimeError(
            "EAGLE target checkpoint did not load all model parameters: "
            + ", ".join(missing_parameters[:20])
        )

    return {
        "loaded_tensor_count": len(loaded_keys),
        "loaded_parameter_count": len(parameter_names),
        "missing_parameter_count": 0,
        "weight_shard_count": len(weight_files),
    }


def repair_eagle_rotary_embeddings(target_core) -> None:
    """Rebuild nonpersistent RoPE buffers lost by the Transformers 5 loader."""

    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rotary = getattr(target_core, "rotary_emb", None)
    if rotary is None:
        raise AttributeError("EAGLE Qwen target has no model-level rotary_emb")
    config = rotary.config
    rope_type = getattr(rotary, "rope_type", "default")
    device = next(target_core.parameters()).device
    initializer = (
        _default_rope_parameters
        if rope_type == "default"
        else ROPE_INIT_FUNCTIONS.get(rope_type)
    )
    if initializer is None:
        raise KeyError(f"Unsupported Qwen3 RoPE type {rope_type!r}")
    inv_freq, attention_scaling = initializer(config, device=device)
    inv_freq = inv_freq.to(device=device, dtype=torch.float32)
    rotary.register_buffer("inv_freq", inv_freq, persistent=False)
    rotary.original_inv_freq = inv_freq.clone()
    rotary.attention_scaling = attention_scaling


def eagle_rotary_diagnostics(target_core, target_attention) -> dict[str, object]:
    """Describe the target RoPE module across Transformers model layouts.

    Transformers 5 moved Qwen3's rotary embedding from each attention layer to
    the decoder model. The vendored EAGLE target follows that layout, while
    older Transformers builds keep it on the attention module.
    """

    refresh = getattr(target_attention, "_refresh_llama3_rope_buffers", None)
    if callable(refresh):
        refresh()

    rotary = getattr(target_attention, "rotary_emb", None)
    if rotary is None:
        rotary = getattr(target_core, "rotary_emb", None)

    inv_freq = getattr(rotary, "inv_freq", None)
    if inv_freq is not None:
        inv_freq_head = inv_freq[:4]
        if hasattr(inv_freq_head, "detach"):
            inv_freq_head = inv_freq_head.detach().float().cpu().tolist()
            inv_freq_stats = {
                "dtype": str(inv_freq.dtype),
                "min": float(inv_freq.detach().float().min().cpu().item()),
                "max": float(inv_freq.detach().float().max().cpu().item()),
            }
        else:
            inv_freq_head = list(inv_freq_head)
            inv_freq_stats = {"dtype": type(inv_freq).__name__}
    else:
        inv_freq_head = None
        inv_freq_stats = None

    return {
        "rotary_impl": type(rotary).__name__ if rotary is not None else None,
        "rotary_module": type(rotary).__module__ if rotary is not None else None,
        "rotary_type": getattr(rotary, "rope_type", None),
        "uses_llama3_rope": getattr(
            target_attention, "_uses_llama3_rope", None
        ),
        "inv_freq_head": inv_freq_head,
        "inv_freq_stats": inv_freq_stats,
    }


def normalize_eagle_acceptance_metrics(
    acceptance_lengths,
    *,
    draft_tokens_accepted: int | None,
    draft_tokens_per_step: int,
) -> dict[str, int | float | str | None]:
    """Normalize EAGLE trace and acceptance counters for the shared schema.

    EAGLE verifies multiple root-to-leaf paths, so its proposal count includes
    every draft-tree node per verification step. The shared candidate rate can
    therefore be small; accepted tokens per step is topology-independent.
    """
    from Benchmark.common.speculative_metrics import normalize_speculative_acceptance

    lengths = [int(value) for value in acceptance_lengths]
    steps = len(lengths)
    proposed = steps * int(draft_tokens_per_step)
    trace_average = sum(lengths) / steps if steps else None
    normalized = normalize_speculative_acceptance(
        verification_steps=steps,
        draft_tokens_accepted=draft_tokens_accepted,
        draft_tokens_proposed=proposed if steps else None,
        fallback_avg_accept_length=trace_average,
    )
    return {**normalized, "draft_proposal_unit": "draft_tree_node"}


def eagle_paired_speedup_fields(
    eagle_output_tokens,
    naive_output_tokens,
    *,
    eagle_time_s: float | None,
    naive_time_s: float | None,
) -> dict[str, object]:
    """Keep output agreement separate from a successful timing pair."""
    if naive_output_tokens is None:
        return {
            "target_greedy_match": None,
            "paired_output_exact_match": None,
            "paired_output_token_ids_match": None,
            "paired_output_token_count_match": None,
            "paired_first_mismatch_index": None,
            "paired_eagle_mismatch_token": None,
            "paired_naive_mismatch_token": None,
            "paired_eagle_output_tokens": len(list(eagle_output_tokens)),
            "paired_naive_output_tokens": None,
            "paired_speedup": None,
            "paired_speedup_valid": False,
            "paired_speedup_invalid_reason": "paired_naive_output_missing",
        }

    eagle_tokens = list(eagle_output_tokens)
    naive_tokens = list(naive_output_tokens)
    mismatch_index = next(
        (
            index
            for index, (eagle_token, naive_token) in enumerate(
                zip(eagle_tokens, naive_tokens)
            )
            if eagle_token != naive_token
        ),
        None,
    )
    if mismatch_index is None and len(eagle_tokens) != len(naive_tokens):
        mismatch_index = min(len(eagle_tokens), len(naive_tokens))
    output_match = mismatch_index is None
    diagnostics = {
        "paired_output_token_ids_match": output_match,
        "paired_output_token_count_match": len(eagle_tokens) == len(naive_tokens),
        "paired_first_mismatch_index": mismatch_index,
        "paired_eagle_mismatch_token": (
            eagle_tokens[mismatch_index]
            if mismatch_index is not None and mismatch_index < len(eagle_tokens)
            else None
        ),
        "paired_naive_mismatch_token": (
            naive_tokens[mismatch_index]
            if mismatch_index is not None and mismatch_index < len(naive_tokens)
            else None
        ),
        "paired_eagle_output_tokens": len(eagle_tokens),
        "paired_naive_output_tokens": len(naive_tokens),
    }
    timing_valid = (
        eagle_time_s is not None
        and naive_time_s is not None
        and float(eagle_time_s) > 0
        and float(naive_time_s) > 0
    )
    if not timing_valid:
        return {
            "target_greedy_match": output_match,
            **diagnostics,
            "paired_speedup": None,
            "paired_speedup_valid": False,
            "paired_speedup_invalid_reason": "paired_timing_missing_or_invalid",
        }
    return {
        "target_greedy_match": output_match,
        **diagnostics,
        "paired_speedup": round(float(naive_time_s) / float(eagle_time_s), 4),
        "paired_speedup_valid": True,
        "paired_speedup_invalid_reason": None,
    }

def _default_rope_parameters(config, device=None, seq_len=None):
    """Compute the original, unscaled RoPE parameters.

    Transformers 5.8+ removed the ``"default"`` entry from
    ``ROPE_INIT_FUNCTIONS`` and moved this implementation into the generated
    model class.  The vendored EAGLE Qwen3 class still calls the registry, so
    keep the equivalent calculation available for that older generated code.
    """

    import torch

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        base = rope_parameters.get("rope_theta")
    else:
        base = None
    base = base or getattr(config, "rope_theta", 10000.0)
    dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
            / dim
        )
    )
    return inv_freq, 1.0


def install_eagle_transformers_compat() -> bool:
    """Install missing Transformers symbols required by the vendored EAGLE.

    Returns ``True`` when this helper added a compatibility symbol and
    ``False`` when the installed Transformers package already provides it.
    The fallback is deliberately a ``TypedDict``: EAGLE only consumes this
    class while constructing a generated typing annotation, not as a runtime
    loss implementation.
    """

    import transformers.utils as transformers_utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    installed = False
    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            labels: object

        transformers_utils.LossKwargs = LossKwargs
        installed = True

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _default_rope_parameters
        installed = True

    return installed
