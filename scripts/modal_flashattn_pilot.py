#!/usr/bin/env python3
"""Batch-one native Transformers FA4 comparison of all baselines on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum_Viet")
REMOTE_SRC = REMOTE_ROOT / "src"
DATASETS = ("vietnews", "wikilingua", "vims", "vlsp")
METHODS = ("vanilla_hf", "eagle3", "dflash", "domino", "dspark")
LOCAL_DATA_FILES = {
    name: PROJECT_ROOT / "datasets" / "eval_100" / f"{name}_100.jsonl"
    for name in DATASETS
}
REMOTE_DATA_FILES = {
    name: REMOTE_ROOT / "datasets" / "eval_100" / f"{name}_100.jsonl"
    for name in DATASETS
}

TARGET_MODEL = os.environ.get("MODAL_QWEN3_MODEL", "Qwen/Qwen3-4B")
MODEL_REPOS = {
    "vanilla_hf": TARGET_MODEL,
    "eagle3": os.environ.get(
        "MODAL_EAGLE3_MODEL_REPO", "AngelSlim/Qwen3-4B_eagle3"
    ),
    "dflash": os.environ.get(
        "MODAL_DFLASH_MODEL_REPO", "z-lab/Qwen3-4B-DFlash-b16"
    ),
    "domino": os.environ.get(
        "MODAL_DOMINO_MODEL_REPO", "Huang2020/Qwen3-4B-Domino-b16"
    ),
    "dspark": os.environ.get(
        "MODAL_DSPARK_MODEL_REPO", "deepseek-ai/dspark_qwen3_4b_block7"
    ),
}
GPU = os.environ.get("MODAL_GPU", "B200")
MAX_INPUT_TOKENS = 8192
DEFAULT_MAX_NEW_TOKENS = 256
DFLASH_BLOCK_SIZE = 16
ATTENTION = "flash_attention_4"

PINNED_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.12.1",
    "tokenizers": "0.22.2",
    "accelerate": "1.15.0",
    "huggingface-hub": "1.31.0",
    "flash-attn-4": "4.0.0b19",
    "quack-kernels": "0.6.5",
    "triton": "3.7.1",
    "apache-tvm-ffi": "0.1.11",
    "nvidia-cutlass-dsl": "4.7.1",
}

app = modal.App("fast-infer-viet-native-flashattn-pilot")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.13.0",
        extra_options=(
            "--index-url https://download.pytorch.org/whl/cu130 "
            "--extra-index-url https://pypi.org/simple"
        ),
    )
    .pip_install(
        "flash-attn-4==4.0.0b19",
        "nvidia-cutlass-dsl==4.7.1",
        "nvidia-cutlass-dsl-libs-base==4.7.1",
        "nvidia-cutlass-dsl-libs-core==4.7.1",
        "nvidia-cutlass-dsl-libs-cu13==4.7.1",
        "cuda-bindings==13.4.1",
        "cuda-python==13.4.1",
        "cuda-core==1.2.0",
        "cuda-pathfinder==1.8.1",
        "cuda-toolkit==13.0.3.0",
        "apache-tvm-ffi==0.1.11",
        "quack-kernels==0.6.5",
        "torch_c_dlpack_ext==0.1.5",
        "triton==3.7.1",
        "einops==0.8.2",
        "ninja==1.13.0",
        extra_options="--pre",
    )
    .pip_install(
        "transformers==5.12.1",
        "tokenizers==0.22.2",
        "accelerate==1.15.0",
        "huggingface_hub==1.31.0",
        "safetensors==0.8.0",
        "numpy==2.3.5",
        "regex==2026.9.10",
        "sentencepiece==0.2.2",
        "tqdm==4.70.1",
        "psutil==7.2.2",
        "packaging==26.3",
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path=str(REMOTE_SRC), copy=True)
    .add_local_dir(
        str(PROJECT_ROOT / "externals" / "dflash"),
        remote_path=str(REMOTE_ROOT / "externals" / "dflash"),
        copy=True,
    )
    .add_local_dir(
        str(PROJECT_ROOT / "externals" / "Domino" / "code"),
        remote_path=str(REMOTE_ROOT / "externals" / "Domino" / "code"),
        copy=True,
    )
    .add_local_dir(
        str(PROJECT_ROOT / "externals" / "DeepSpec"),
        remote_path=str(REMOTE_ROOT / "externals" / "DeepSpec"),
        copy=True,
    )
    .add_local_dir(
        str(PROJECT_ROOT / "externals" / "EAGLE"),
        remote_path=str(REMOTE_ROOT / "externals" / "EAGLE"),
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_SRC),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
)
for _dataset, _local_file in LOCAL_DATA_FILES.items():
    image = image.add_local_file(
        str(_local_file), remote_path=str(REMOTE_DATA_FILES[_dataset]), copy=True
    )


def _install_fa4_compat() -> None:
    from Benchmark.common.vanilla_inference import (
        _install_flash_attention_4_cutlass_compat,
    )

    _install_flash_attention_4_cutlass_compat()

    # FA4 4.0.0b19 still imports this primitive from QuACK, while QuACK 0.6.5
    # removed that export.  The upstream compatibility fix calls the same
    # CuTe primitive directly; provide that alias without modifying site-packages.
    import cutlass.cute.arch as cute_arch
    import quack.activation as quack_activation

    if not hasattr(quack_activation, "sub_packed_f32x2"):
        quack_activation.sub_packed_f32x2 = cute_arch.sub_packed_f32x2


def _canonical_dist_name(value: str) -> str:
    return str(value).lower().replace("_", "-").split("==", 1)[0]


def _version_snapshot() -> dict[str, str | None]:
    from importlib import metadata

    actual: dict[str, str | None] = {}
    for name in PINNED_VERSIONS:
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual[name] = None
    return actual


def _validate_pins(actual: dict[str, str | None]) -> None:
    mismatches = {}
    for package, expected in PINNED_VERSIONS.items():
        value = actual.get(package)
        normalized = str(value).split("+", 1)[0] if value else None
        if normalized != expected:
            mismatches[package] = {"expected": expected, "actual": value}
    if mismatches:
        raise RuntimeError(f"Modal package pins differ from reference: {mismatches}")


def _runtime_base(
    torch,
    actual_versions: dict[str, str | None],
    *,
    methods: tuple[str, ...],
) -> dict[str, Any]:
    from importlib import metadata
    import sys

    installed = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed.append(_canonical_dist_name(name))
    imported = [name for name in sys.modules if name == "flash_attn" or name.startswith("flash_attn.")]
    imported.extend(
        name for name in sys.modules if name == "vllm" or name.startswith("vllm.")
    )
    return {
        "installed_distributions": installed,
        "imported_modules": imported,
        "batch_size": 1,
        "versions": actual_versions,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": torch.cuda.get_device_capability(0),
        "methods": {
            method: {
                "target_attention": ATTENTION,
                "draft_attention": None if method == "vanilla_hf" else ATTENTION,
            }
            for method in methods
        },
    }


def _fa4_tree_mask_gpu_probe(torch) -> dict[str, Any]:
    import torch.nn.functional as F

    from Benchmark.common.flashattn4_tree_attention import flashattn4_attention

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    q = torch.randn((1, 4, 4, 64), device=device, dtype=dtype)
    k = torch.randn((1, 2, 8, 64), device=device, dtype=dtype)
    v = torch.randn((1, 2, 8, 64), device=device, dtype=dtype)
    keep = torch.tensor(
        [[
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0],
        ]],
        dtype=torch.bool,
        device=device,
    )
    additive_mask = torch.where(
        keep[:, None],
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), torch.finfo(torch.float32).min, device=device),
    )
    actual = flashattn4_attention(
        q,
        k,
        v,
        additive_mask,
        scaling=64**-0.5,
    ).transpose(1, 2)
    reference = F.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(2, dim=1),
        v.float().repeat_interleave(2, dim=1),
        attn_mask=keep[:, None],
        is_causal=False,
        scale=64**-0.5,
    )
    torch.cuda.synchronize(device)
    max_abs_error = float((actual.float() - reference).abs().max().item())
    if not torch.allclose(actual.float(), reference, atol=0.06, rtol=0.03):
        raise RuntimeError(
            "FA4 custom tree-mask result disagrees with SDPA reference: "
            f"max_abs_error={max_abs_error:.6g}"
        )

    # EAGLE's Qwen3 draft attention uses two query nodes, GQA (32 Q heads / 8
    # KV heads), and a long cached prefix. Exercise that geometry explicitly;
    # the small probe above alone cannot detect padded-tile mask indexing bugs.
    q_len, kv_len = 2, 2726
    q = torch.randn((1, 32, q_len, 128), device=device, dtype=dtype)
    k = torch.randn((1, 8, kv_len, 128), device=device, dtype=dtype)
    v = torch.randn((1, 8, kv_len, 128), device=device, dtype=dtype)
    keep = torch.ones((1, q_len, kv_len), dtype=torch.bool, device=device)
    keep[:, 0, -2:] = torch.tensor([True, False], device=device)
    keep[:, 1, -2:] = torch.tensor([True, True], device=device)
    additive_mask = torch.where(
        keep[:, None],
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), torch.finfo(torch.float32).min, device=device),
    )
    actual = flashattn4_attention(
        q,
        k,
        v,
        additive_mask,
        scaling=128**-0.5,
    ).transpose(1, 2)
    reference = F.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=keep[:, None],
        is_causal=False,
        scale=128**-0.5,
        enable_gqa=True,
    )
    torch.cuda.synchronize(device)
    eagle_draft_max_abs_error = float(
        (actual.float() - reference).abs().max().item()
    )
    if not torch.allclose(actual.float(), reference, atol=0.08, rtol=0.04):
        raise RuntimeError(
            "FA4 EAGLE-draft tree-mask result disagrees with SDPA reference: "
            f"max_abs_error={eagle_draft_max_abs_error:.6g}"
        )
    short_kv_len = 622
    short_keep = torch.ones(
        (1, q_len, short_kv_len), dtype=torch.bool, device=device
    )
    short_keep[:, 0, -2:] = torch.tensor([True, False], device=device)
    short_keep[:, 1, -2:] = torch.tensor([True, True], device=device)
    short_additive_mask = torch.where(
        short_keep[:, None],
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), torch.finfo(torch.float32).min, device=device),
    )
    short_actual = flashattn4_attention(
        q,
        k[:, :, :short_kv_len],
        v[:, :, :short_kv_len],
        short_additive_mask,
        scaling=128**-0.5,
    ).transpose(1, 2)
    short_reference = F.scaled_dot_product_attention(
        q.float(),
        k[:, :, :short_kv_len].float(),
        v[:, :, :short_kv_len].float(),
        attn_mask=short_keep[:, None],
        is_causal=False,
        scale=128**-0.5,
        enable_gqa=True,
    )
    torch.cuda.synchronize(device)
    eagle_draft_short_max_abs_error = float(
        (short_actual.float() - short_reference).abs().max().item()
    )
    if not torch.allclose(short_actual.float(), short_reference, atol=0.08, rtol=0.04):
        raise RuntimeError(
            "FA4 EAGLE-draft short tree-mask result disagrees with SDPA reference: "
            f"max_abs_error={eagle_draft_short_max_abs_error:.6g}"
        )
    return {
        "passed": True,
        "max_abs_error": max_abs_error,
        "eagle_draft_max_abs_error": eagle_draft_max_abs_error,
        "eagle_draft_short_max_abs_error": eagle_draft_short_max_abs_error,
        "eagle_draft_q_len": q_len,
        "eagle_draft_kv_len": kv_len,
        "eagle_draft_short_kv_len": short_kv_len,
        "eagle_draft_query_heads": 32,
        "eagle_draft_kv_heads": 8,
    }


def _config_attention(model: Any) -> str | None:
    config = getattr(model, "config", None)
    value = getattr(config, "_attn_implementation", None)
    return str(value) if value is not None else None


def _assert_fa4(model: Any, method: str, role: str) -> str:
    value = _config_attention(model)
    if value != ATTENTION:
        raise RuntimeError(
            f"{method} {role} attention resolved to {value!r}, expected {ATTENTION!r}"
        )
    return value


def _dispatch_draft_model(method: str, context: dict[str, Any]):
    if method == "eagle3":
        return context["eagle_model"].ea_layer
    return context.get("draft")


def _assert_method_dispatch(method: str, stats: dict[str, Any]) -> None:
    roles = ("target",) if method == "vanilla_hf" else ("target", "draft")
    for role in roles:
        calls = int(stats.get(f"{role}_attention_dispatch_calls", 0) or 0)
        backend = stats.get(f"{role}_attention_dispatch")
        fallbacks = int(stats.get(f"{role}_fallback_attention_calls", 0) or 0)
        if backend != ATTENTION or calls <= 0 or fallbacks:
            raise RuntimeError(
                f"{method} {role} did not exclusively dispatch through FA4: "
                f"backend={backend!r}, fa4_calls={calls}, fallback_calls={fallbacks}, "
                f"fallback_backends={stats.get(f'{role}_fallback_attention_backends', {})}"
            )


def _new_dispatch_totals() -> dict[str, Any]:
    return {
        "target_fa4_calls": 0,
        "target_fallback_backends": {},
        "draft_fa4_calls": 0,
        "draft_fallback_backends": {},
        "unattributed_attention_calls": {},
    }


def _add_dispatch_stats(totals: dict[str, Any], stats: dict[str, Any]) -> None:
    for role in ("target", "draft"):
        totals[f"{role}_fa4_calls"] += int(
            stats.get(f"{role}_attention_dispatch_calls", 0) or 0
        )
        key = f"{role}_fallback_backends"
        for backend, count in stats.get(f"{role}_fallback_attention_backends", {}).items():
            totals[key][backend] = totals[key].get(backend, 0) + int(count)
    for backend, count in stats.get("unattributed_attention_calls", {}).items():
        key = "unattributed_attention_calls"
        totals[key][backend] = totals[key].get(backend, 0) + int(count)


def _dispatch_config_fields(totals: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for role in ("target", "draft"):
        fa4_calls = int(totals[f"{role}_fa4_calls"])
        fallback_backends = dict(totals[f"{role}_fallback_backends"])
        fields.update(
            {
                f"{role}_attention_dispatch": ATTENTION if fa4_calls else None,
                f"{role}_attention_dispatch_calls": fa4_calls,
                f"{role}_fallback_attention_calls": sum(fallback_backends.values()),
                f"{role}_fallback_attention_backends": fallback_backends,
            }
        )
    fields["unattributed_attention_calls"] = dict(
        totals["unattributed_attention_calls"]
    )
    return fields


def _prepare_samples(tokenizer, *, sample_count: int, max_input_tokens: int) -> list[dict[str, Any]]:
    from Benchmark.common.benchmark_data import read_jsonl, render_prompt
    from Benchmark.common.flashattn_runtime import (
        select_median_samples,
        select_smoke_datasets,
    )
    from Benchmark.common.input_utils import truncate_input_ids
    from Benchmark.common.prompt_format import format_chat_prompt

    selected_datasets = select_smoke_datasets(DATASETS, sample_count)
    candidates: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for row in read_jsonl(REMOTE_DATA_FILES[dataset]):
            prompt = format_chat_prompt(tokenizer, render_prompt(row))
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            input_ids = truncate_input_ids(input_ids, max_input_tokens).contiguous()
            answers = row.get("answers")
            reference = row.get("reference")
            if not reference and isinstance(answers, list) and answers:
                reference = answers[0]
            if not reference and isinstance(answers, str):
                reference = answers
            candidates.append(
                {
                    "dataset": dataset,
                    "sample_id": str(row.get("id", row.get("source_index", ""))),
                    "input_tokens": int(input_ids.shape[1]),
                    "input_ids": input_ids,
                    "reference": str(reference or ""),
                    "prompt": prompt,
                    "source_sha256": hashlib.sha256(
                        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
                    ).hexdigest(),
                }
            )
    selected = select_median_samples(
        candidates,
        datasets=selected_datasets,
        max_input_tokens=max_input_tokens,
    )
    return sorted(selected, key=lambda row: (DATASETS.index(row["dataset"]), row["sample_id"]))


def _target_model(model_name: str, *, dtype, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=ATTENTION,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    model.generation_config.do_sample = False
    return model


def _stop_ids(model, tokenizer) -> list[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(model.config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = [int(item) for item in value]
    else:
        values = []
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in values:
            values.append(token_id)
    return values


def _time_call(torch, fn):
    torch.cuda.synchronize("cuda:0")
    start = time.perf_counter()
    with torch.inference_mode():
        result = fn()
    torch.cuda.synchronize("cuda:0")
    return result, (time.perf_counter() - start) * 1000.0


def _warm_method(torch, method: str, context: dict[str, Any], samples: list[dict[str, Any]], max_new_tokens: int):
    for sample in samples:
        _call_method(
            torch,
            method,
            context,
            sample["input_ids"].to("cuda:0"),
            max_new_tokens=min(8, max_new_tokens),
            warmup=True,
        )
    torch.cuda.synchronize("cuda:0")


def _call_method(
    torch,
    method: str,
    context: dict[str, Any],
    input_ids,
    *,
    max_new_tokens: int,
    warmup: bool = False,
):
    from transformers import set_seed

    set_seed(42)
    model = context["target"]
    stop_token_ids = context["stop_token_ids"]
    if method == "vanilla_hf":
        return model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=stop_token_ids or None,
            pad_token_id=context["tokenizer"].pad_token_id,
        )
    if method == "dflash":
        return context["dflash_generate"](
            context["draft"],
            target=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids or None,
            temperature=0.0,
            block_size=DFLASH_BLOCK_SIZE,
            return_stats=True,
        )
    if method == "domino":
        return context["draft"].spec_generate(
            input_ids,
            target=model,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            stop_token_ids=stop_token_ids or None,
            block_size=context["draft"].block_size,
            use_bias=True,
            return_dict=True,
        )
    if method == "dspark":
        context["evaluator"].args.max_new_tokens = int(max_new_tokens)
        return context["evaluator"].generate_one_sample(
            input_ids=input_ids,
            stop_token_ids=stop_token_ids or None,
        )
    if method == "eagle3":
        from Benchmark.eagle3_infer_qwen3 import timed_generate

        return timed_generate(
            context["eagle_model"],
            input_ids,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            total_token=int(context["eagle_tree"]["total_token"]),
            spec=True,
            is_llama3=False,
            include_phase_timings=True,
            stop_token_ids=stop_token_ids or None,
        )
    raise ValueError(f"unknown method {method!r}")


def _direct_target_greedy(
    torch,
    method: str,
    context: dict[str, Any],
    input_ids,
    *,
    max_new_tokens: int,
) -> list[int]:
    """Generate target-only tokens outside timing to isolate verifier parity."""

    if method == "eagle3":
        from Benchmark.eagle3_infer_qwen3 import timed_generate

        result = timed_generate(
            context["eagle_model"],
            input_ids,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            total_token=int(context["eagle_tree"]["total_token"]),
            spec=False,
            is_llama3=False,
            include_phase_timings=True,
            stop_token_ids=context["stop_token_ids"] or None,
        )
        output_ids = result[0]
    else:
        output_ids = context["target"].generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=context["stop_token_ids"] or None,
            pad_token_id=context["tokenizer"].pad_token_id,
        )
    return [int(value) for value in output_ids[0, input_ids.shape[1] :].tolist()]


def _audit_target_verifier_predictions(
    torch,
    method: str,
    context: dict[str, Any],
    input_ids,
    *,
    max_new_tokens: int,
    reference_output_ids: list[int],
) -> dict[str, Any]:
    """Compare verifier-batch target argmaxes with the Vanilla token stream.

    This untimed diagnostic distinguishes speculative acceptance/correction
    errors from numerical differences between batched verification and
    one-token-at-a-time target decoding.
    """

    if method not in {"domino", "dspark"}:
        return {}
    from Benchmark.common.flashattn_runtime import first_token_mismatch

    target = context["target"]
    original_forward = target.forward
    observations: list[dict[str, Any]] = []
    verify_calls: list[dict[str, Any]] = []

    def capture_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        logits = getattr(output, "logits", None)
        positions = kwargs.get("position_ids")
        if logits is None or positions is None or logits.ndim != 3:
            return output
        query_length = int(logits.shape[-2])
        position_start = int(positions[0, 0].item())
        positions = positions[:, -query_length:]
        input_token_ids = kwargs.get("input_ids")
        if input_token_ids is None and args:
            input_token_ids = args[0]
        input_token_values = (
            input_token_ids[0].detach().cpu().tolist()
            if input_token_ids is not None and input_token_ids.ndim == 2
            else []
        )
        top2 = torch.topk(logits.float(), k=2, dim=-1)
        # Match the verifier's exact greedy tie-breaking: Domino/DSpark use
        # torch.argmax, while topk may return a different token on equal logits.
        predicted_ids = logits.argmax(dim=-1)
        margins = top2.values[..., 0] - top2.values[..., 1]
        position_values = positions[0].detach().cpu().tolist()
        predicted_values = predicted_ids[0].detach().cpu().tolist()
        margin_values = margins[0].detach().cpu().tolist()
        if query_length > 1 and len(input_token_values) == query_length:
            proposal_ids = [int(value) for value in input_token_values[1:]]
            accepted_draft_tokens = 0
            for proposal_id, predicted_id in zip(
                proposal_ids, predicted_values[:-1]
            ):
                if int(proposal_id) != int(predicted_id):
                    break
                accepted_draft_tokens += 1
            verify_calls.append(
                {
                    "start_position": position_start,
                    "query_length": query_length,
                    "input_token_ids": [int(value) for value in input_token_values],
                    "proposal_token_ids": proposal_ids,
                    "target_argmax_ids": [int(value) for value in predicted_values],
                    "target_top1_margins": [float(value) for value in margin_values],
                    "computed_accepted_draft_tokens": accepted_draft_tokens,
                }
            )
        for query_index, absolute_position in enumerate(position_values):
            absolute_position = int(absolute_position)
            output_offset = absolute_position - int(input_ids.shape[1]) + 1
            if 0 <= output_offset < len(reference_output_ids):
                predicted_id = int(predicted_values[query_index])
                expected_id = int(reference_output_ids[output_offset])
                input_position = absolute_position - position_start
                proposal_token_id = (
                    int(input_token_values[input_position + 1])
                    if 0 <= input_position + 1 < len(input_token_values)
                    else None
                )
                observations.append(
                    {
                        "output_offset": output_offset,
                        "expected_token_id": expected_id,
                        "verifier_argmax_token_id": predicted_id,
                        "verifier_proposal_token_id": proposal_token_id,
                        "top1_margin": float(margin_values[query_index]),
                    }
                )
        return output

    target.forward = capture_forward
    try:
        result = _call_method(
            torch,
            method,
            context,
            input_ids,
            max_new_tokens=max_new_tokens,
        )
    finally:
        target.forward = original_forward

    candidate_ids = [
        int(value)
        for value in result.output_ids[0, input_ids.shape[1] :].tolist()
    ]
    mismatch_offset = first_token_mismatch(reference_output_ids, candidate_ids)
    reported_acceptance_lengths = list(
        getattr(result, "acceptance_lengths", []) or []
    )
    emitting_step_diagnostic = None
    if mismatch_offset is not None:
        for call_index, call in enumerate(verify_calls):
            start_output_offset = int(call["start_position"]) - int(input_ids.shape[1])
            local_output_index = mismatch_offset - start_output_offset - 1
            accepted = int(call["computed_accepted_draft_tokens"])
            if not 0 <= local_output_index <= accepted:
                continue
            output_source = (
                "accepted_draft" if local_output_index < accepted else "target_correction"
            )
            proposal_token_id = (
                call["proposal_token_ids"][local_output_index]
                if local_output_index < len(call["proposal_token_ids"])
                else None
            )
            verifier_argmax_token_id = int(
                call["target_argmax_ids"][local_output_index]
            )
            algorithm_token_id = (
                proposal_token_id
                if output_source == "accepted_draft"
                else verifier_argmax_token_id
            )
            reported_length = (
                int(reported_acceptance_lengths[call_index])
                if call_index < len(reported_acceptance_lengths)
                else None
            )
            emitting_step_diagnostic = {
                "verify_call_index": call_index,
                "start_generated_offset": start_output_offset,
                "reported_acceptance_length": reported_length,
                "reported_accepted_draft_tokens": (
                    max(0, reported_length - 1) if reported_length is not None else None
                ),
                "recomputed_accepted_draft_tokens": accepted,
                "output_source": output_source,
                "vanilla_reference_token_id": int(reference_output_ids[mismatch_offset]),
                "verifier_argmax_token_id": verifier_argmax_token_id,
                "target_top1_margin": float(
                    call["target_top1_margins"][local_output_index]
                ),
                "verifier_proposal_token_id": proposal_token_id,
                "algorithm_token_id_from_this_step": algorithm_token_id,
                "actual_candidate_token_id": int(candidate_ids[mismatch_offset]),
                "algorithm_token_matches_candidate": (
                    algorithm_token_id == int(candidate_ids[mismatch_offset])
                ),
            }
            break
    at_first_mismatch = [
        item
        for item in observations
        if mismatch_offset is not None
        and item["output_offset"] == mismatch_offset
    ]
    candidate_token_id = (
        candidate_ids[mismatch_offset]
        if mismatch_offset is not None and mismatch_offset < len(candidate_ids)
        else None
    )
    mismatches_through_first_output_mismatch = [
        item
        for item in observations
        if mismatch_offset is not None
        and item["output_offset"] <= mismatch_offset
        and item["verifier_argmax_token_id"] != item["expected_token_id"]
    ]
    return {
        "target_verifier_audit_position_count": len(observations),
        "target_verifier_argmax_mismatch_count_through_first_output_mismatch": len(
            mismatches_through_first_output_mismatch
        ),
        "target_verifier_argmax_at_first_output_mismatch": at_first_mismatch,
        "emitting_verify_step_diagnostic": emitting_step_diagnostic,
        "candidate_token_id_at_first_output_mismatch": candidate_token_id,
        "candidate_matches_any_verifier_argmax_at_first_mismatch": (
            any(
                item["verifier_argmax_token_id"] == candidate_token_id
                for item in at_first_mismatch
            )
            if candidate_token_id is not None
            else None
        ),
        "candidate_matches_any_verifier_proposal_at_first_mismatch": (
            any(
                item["verifier_proposal_token_id"] == candidate_token_id
                for item in at_first_mismatch
            )
            if candidate_token_id is not None
            else None
        ),
    }


def _load_method(torch, method: str, tokenizer, device) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM

    target_name = TARGET_MODEL
    context: dict[str, Any] = {
        "target": None,
        "tokenizer": tokenizer,
        "stop_token_ids": [],
        "draft": None,
        "evaluator": None,
        "eagle_model": None,
        "dflash_generate": None,
        "dflash_model_module": None,
        "model_name": target_name,
        "draft_name": None,
    }
    if method == "vanilla_hf":
        target = _target_model(target_name, dtype=torch.bfloat16, device=device)
        _assert_fa4(target, method, "target")
        context.update(target=target, stop_token_ids=_stop_ids(target, tokenizer))
        return context

    draft_name = MODEL_REPOS[method]
    context["draft_name"] = draft_name

    if method == "dflash":
        from Benchmark.dflash_compat import (
            install_dflash_cache_crop_compat,
            install_dflash_transformers_compat,
        )
        from Benchmark.infer_dflash import normalize_generation_token_ids
        from Benchmark.dflash_fa4_attention import install_dflash_fa4_attention

        dflash_root = REMOTE_ROOT / "externals" / "dflash"
        if str(dflash_root) not in os.sys.path:
            os.sys.path.insert(0, str(dflash_root))
        install_dflash_transformers_compat()
        import dflash.model as dflash_model

        install_dflash_cache_crop_compat(dflash_model)
        install_dflash_fa4_attention(dflash_model)
        target_config = AutoConfig.from_pretrained(target_name)
        draft_config = AutoConfig.from_pretrained(draft_name)
        normalize_generation_token_ids(target_config, tokenizer)
        normalize_generation_token_ids(draft_config, tokenizer)
        target = AutoModelForCausalLM.from_pretrained(
            target_name,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION,
            low_cpu_mem_usage=True,
            config=target_config,
        ).to(device).eval()
        draft = dflash_model.DFlashDraftModel.from_pretrained(
            draft_name,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION,
            low_cpu_mem_usage=True,
            config=draft_config,
        ).to(device).eval()
        if int(draft.block_size) != DFLASH_BLOCK_SIZE:
            raise RuntimeError(
                f"DFlash checkpoint block_size={draft.block_size}, "
                f"expected {DFLASH_BLOCK_SIZE}"
            )
        _assert_fa4(target, method, "target")
        _assert_fa4(draft, method, "draft")
        context.update(
            target=target,
            draft=draft,
            stop_token_ids=_stop_ids(target, tokenizer),
            dflash_generate=dflash_model.dflash_generate,
            dflash_model_module=dflash_model,
        )
        return context

    if method == "domino":
        import importlib.util

        domino_path = REMOTE_ROOT / "externals" / "Domino" / "code" / "dflash.py"
        spec = importlib.util.spec_from_file_location("domino_native_dflash", domino_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import Domino native code: {domino_path}")
        domino_module = importlib.util.module_from_spec(spec)
        os.sys.modules[spec.name] = domino_module
        spec.loader.exec_module(domino_module)
        draft_config = AutoConfig.from_pretrained(draft_name)
        domino_config = dict(getattr(draft_config, "dflash_config", {}) or {})
        if domino_config.get("projector_type") == "causal_v5":
            domino_config["projector_type"] = "domino"
        if "emb_dim" not in domino_config and getattr(draft_config, "emb_dim", None):
            domino_config["emb_dim"] = draft_config.emb_dim
        if "gru_hidden_dim" not in domino_config:
            domino_config["gru_hidden_dim"] = getattr(
                draft_config, "gru_hidden_dim", domino_config.get("emb_dim")
            )
        draft_config.dflash_config = domino_config
        target = AutoModelForCausalLM.from_pretrained(
            target_name,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        draft = domino_module.DFlashDraftModel.from_pretrained(
            draft_name,
            config=draft_config,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        _assert_fa4(target, method, "target")
        _assert_fa4(draft, method, "draft")
        context.update(
            target=target,
            draft=draft,
            stop_token_ids=_stop_ids(target, tokenizer),
        )
        return context

    if method == "dspark":
        deepspec_root = REMOTE_ROOT / "externals" / "DeepSpec"
        if str(deepspec_root) not in os.sys.path:
            os.sys.path.insert(0, str(deepspec_root))
        from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator
        from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
        from deepspec.eval.base_evaluator import assert_no_final_target_layer

        target = AutoModelForCausalLM.from_pretrained(
            target_name,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        draft = Qwen3DSparkModel.from_pretrained(
            draft_name,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        assert_no_final_target_layer(target, draft.target_layer_ids)
        _assert_fa4(target, method, "target")
        _assert_fa4(draft, method, "draft")
        from types import SimpleNamespace

        evaluator = object.__new__(Qwen3DSparkEvaluator)
        evaluator.args = SimpleNamespace(
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=0.0,
            confidence_threshold=0.0,
        )
        evaluator.device = device
        evaluator.target_model = target
        evaluator.draft_model = draft
        evaluator.tokenizer = tokenizer
        evaluator.confidence_head_recorder = None
        context.update(
            target=target,
            draft=draft,
            evaluator=evaluator,
            stop_token_ids=_stop_ids(target, tokenizer),
        )
        return context

    if method == "eagle3":
        eagle_root = REMOTE_ROOT / "externals" / "EAGLE"
        if str(eagle_root) not in os.sys.path:
            os.sys.path.insert(0, str(eagle_root))
        from Benchmark.eagle3_infer_qwen3 import resolve_eagle_tree_config
        from Benchmark.eagle_compat import (
            eagle_model_load_options,
            reload_eagle_target_weights,
            repair_eagle_rotary_embeddings,
        )
        from Benchmark.eagle_fa4_attention import (
            install_eagle_draft_fa4_attention,
            install_eagle_fa4_attention,
        )
        from Benchmark.eagle_compat import install_eagle_transformers_compat

        install_eagle_transformers_compat()
        from eagle.model import cnets, modeling_qwen3_kv

        install_eagle_fa4_attention(modeling_qwen3_kv)
        install_eagle_draft_fa4_attention(cnets)
        from eagle.model.ea_model import EaModel
        from eagle.model.kv_cache import initialize_past_key_values

        # This vendored EAGLE tree builder can retain a child while pruning
        # its parent when total_token truncates the candidate list.  Use the
        # complete small tree (top_k + depth * top_k^2 = 18) so every parent
        # remains addressable and FA4 verifies all tree branches correctly.
        tree = resolve_eagle_tree_config(total_token=18, depth=4, top_k=2)
        eagle_model = EaModel.from_pretrained(
            base_model_path=target_name,
            ea_model_path=draft_name,
            total_token=int(tree["total_token"]),
            depth=4,
            top_k=2,
            threshold=1.0,
            torch_dtype=torch.bfloat16,
            **eagle_model_load_options(),
            use_eagle3=True,
        )
        target_load_audit = reload_eagle_target_weights(
            eagle_model.base_model, target_name
        )
        eagle_model.to(device)
        eagle_model.ea_layer.init_tree()
        repair_eagle_rotary_embeddings(eagle_model.base_model.model)
        eagle_model.eval()
        eagle_model.tokenizer = tokenizer
        # This vendored EAGLE fork hardcodes eager attention in its original
        # Qwen3 forward.  The patched forward routes to FA4 only when this
        # config flag is explicit, so set it after loading the custom model.
        eagle_model.base_model.config._attn_implementation = ATTENTION
        _assert_fa4(eagle_model.base_model, method, "target")
        eagle_model.ea_layer.config._attn_implementation = ATTENTION
        draft_attentions = [
            module
            for module in eagle_model.ea_layer.modules()
            if hasattr(module, "config")
            and hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
        ]
        if not draft_attentions:
            raise RuntimeError("EAGLE draft attention modules were not found")
        for attention_module in draft_attentions:
            attention_module.config._attn_implementation = ATTENTION
        context.update(
            target=eagle_model.base_model,
            eagle_model=eagle_model,
            stop_token_ids=_stop_ids(eagle_model.base_model, tokenizer),
            target_load_audit=target_load_audit,
            eagle_tree=tree,
        )
        return context

    raise ValueError(f"unknown method {method!r}")


def _output_payload(
    method: str,
    result: Any,
    input_ids,
    elapsed_ms: float,
    context: dict[str, Any],
) -> dict[str, Any]:
    output_ids = result
    phases: dict[str, Any] = {}
    acceptance_lengths: list[int] = []
    accepted = proposed = steps = None
    ttft_ms = None
    if method in {"dflash", "domino"}:
        output_ids = result.output_ids
        ttft_ms = float(getattr(result, "time_to_first_token", 0.0) or 0.0) * 1000.0
        acceptance_lengths = list(getattr(result, "acceptance_lengths", []) or [])
        steps = len(acceptance_lengths)
        if method == "dflash":
            accepted = int(getattr(result, "draft_tokens_accepted", 0) or 0)
            proposed = int(getattr(result, "draft_tokens_proposed", 0) or 0)
        else:
            accepted = sum(max(0, int(length) - 1) for length in acceptance_lengths)
            block_size = int(context["draft"].block_size)
            shift_label = bool(
                getattr(context["draft"].config, "dflash_config", {}).get(
                    "shift_label", False
                )
            )
            proposed = steps * (block_size if shift_label else block_size - 1)
    elif method == "dspark":
        output_ids = result.output_ids
        acceptance_lengths = list(getattr(result, "acceptance_lengths", []) or [])
        steps = int(getattr(result, "verify_count", len(acceptance_lengths)))
        accepted = sum(int(value) for value in (getattr(result, "accepted_draft_lengths", []) or []))
        proposed = sum(int(value) for value in (getattr(result, "proposal_lengths", []) or []))
    elif method == "eagle3":
        output_ids, _, _, _elapsed_s, acceptance_lengths, phases = result
        phases = dict(phases or {})
        ttft_ms = float(phases.get("prefill_ms", 0.0) or 0.0)
        accepted = int(phases.get("draft_tokens_accepted", 0) or 0)
        proposed = int(phases.get("draft_tokens_proposed", 0) or 0)
        steps = len(acceptance_lengths)
    generated = output_ids[:, input_ids.shape[1] :]
    return {
        "output_ids": [int(value) for value in generated[0].tolist()],
        "elapsed_ms": float(elapsed_ms),
        "ttft_ms": ttft_ms,
        "acceptance_lengths": acceptance_lengths,
        "draft_tokens_accepted": accepted,
        "draft_tokens_proposed": proposed,
        "verification_steps": steps,
        "phases": phases,
    }


def _safe_mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


@app.function(image=image, gpu=GPU, cpu=16, memory=65536, timeout=14400, max_containers=1)
def run_flashattn_smoke(
    *,
    sample_count: int = 2,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    methods: str = "all",
    debug_cuda_launch_blocking: bool = False,
) -> dict[str, Any]:
    """Compare selected methods on the same batch-one prompts on one GPU."""
    import gc
    import sys

    sys.path.insert(0, str(REMOTE_SRC))
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if debug_cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    _install_fa4_compat()

    import torch
    from importlib import metadata
    from transformers import AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from Benchmark.common.benchmark_runtime import (
        build_sample_record,
        build_status_record,
        runtime_metadata,
    )
    from Benchmark.common.flashattn_runtime import (
        AttentionDispatchTracker,
        first_token_mismatch,
        install_attention_dispatch_tracking,
        is_cuda_context_failure,
        resolve_flashattn_methods,
        validate_flashattn_runtime,
    )
    from Benchmark.common.io_util import JsonlWriter
    from Benchmark.common.quality_guard import repetition_metrics
    from Benchmark.common.rouge import add_rouge, aggregate_rouge

    if not torch.cuda.is_available():
        raise RuntimeError("Modal worker did not expose a CUDA GPU")
    if torch.cuda.get_device_capability(0)[0] < 10:
        raise RuntimeError("FA4 smoke requires a Blackwell-class GPU (SM100+)")
    if sample_count < 1 or sample_count > len(DATASETS):
        raise ValueError(f"sample_count must be in [1, {len(DATASETS)}]")
    selected_methods = resolve_flashattn_methods(methods)
    if max_new_tokens < 32:
        raise ValueError("max_new_tokens must be at least 32 for a meaningful smoke")
    actual_versions = _version_snapshot()
    _validate_pins(actual_versions)
    try:
        import flash_attn.cute  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"FA4 CuTe runtime failed to import: {exc}") from exc
    install_attention_dispatch_tracking(ALL_ATTENTION_FUNCTIONS)
    runtime = _runtime_base(
        torch, actual_versions, methods=selected_methods
    )
    baseline_config = validate_flashattn_runtime(
        runtime, methods=selected_methods
    )
    runtime["fa4_tree_mask_gpu_probe"] = _fa4_tree_mask_gpu_probe(torch)

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = _prepare_samples(
        tokenizer,
        sample_count=sample_count,
        max_input_tokens=max_input_tokens,
    )
    runtime["selected_samples"] = [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "input_tokens": row["input_tokens"],
            "source_sha256": row["source_sha256"],
        }
        for row in samples
    ]

    records: list[dict[str, Any]] = []
    raw: dict[tuple[str, str], dict[str, Any]] = {}
    target_only_raw: dict[tuple[str, str], list[int]] = {}
    verifier_audit_raw: dict[tuple[str, str], dict[str, Any]] = {}
    method_configs: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    gpu_context_lost = False

    for method in selected_methods:
        print(f"[FA4] loading {method}: {MODEL_REPOS[method]}", flush=True)
        context = None
        tracker = None
        dispatch_totals = _new_dispatch_totals()
        method_record_start = len(records)
        try:
            context = _load_method(torch, method, tokenizer, device)
            method_configs[method] = {
                "target_attention": _assert_fa4(context["target"], method, "target"),
                "draft_attention": (
                    None
                    if method == "vanilla_hf"
                    else (
                        _assert_fa4(
                            context["eagle_model"].ea_layer,
                            method,
                            "draft",
                        )
                        if method == "eagle3"
                        else _assert_fa4(context["draft"], method, "draft")
                    )
                ),
            }
            if method == "eagle3":
                context["eagle_model"].ea_layer.config._attn_implementation = ATTENTION
                max_prompt = max(int(row["input_tokens"]) for row in samples)
                eagle_model = context["eagle_model"]
                from eagle.model.kv_cache import initialize_past_key_values

                past_kv, past_kv_data, current_length = initialize_past_key_values(
                    eagle_model.base_model,
                    max_length=max_prompt + max_new_tokens + 96,
                )
                eagle_model.past_key_values = past_kv
                eagle_model.past_key_values_data = past_kv_data
                eagle_model.current_length_data = current_length

            tracker = AttentionDispatchTracker(
                context["target"], _dispatch_draft_model(method, context)
            )
            with tracker.recording():
                _warm_method(torch, method, context, samples, max_new_tokens)
            warm_dispatch = tracker.snapshot()
            _assert_method_dispatch(method, warm_dispatch)
            tracker.reset()
            if method == "dflash":
                from Benchmark.dflash_fa4_attention import dflash_fa4_attention_stats

                dispatch = dflash_fa4_attention_stats(
                    context["dflash_model_module"]
                )
                if (
                    dispatch["fa4_dispatch_calls"] <= 0
                    or dispatch["sdpa_fallback_calls"] != 0
                ):
                    raise RuntimeError(
                        "DFlash warmup did not prove exclusive FA4 draft dispatch: "
                        f"{dispatch}"
                    )
                method_configs[method].update(
                    {
                        "draft_attention_dispatch": ATTENTION,
                        "draft_attention_dispatch_calls": dispatch[
                            "fa4_dispatch_calls"
                        ],
                        "draft_sdpa_fallback_calls": dispatch[
                            "sdpa_fallback_calls"
                        ],
                    }
                )
            print(f"[FA4] warmup complete: {method}", flush=True)
            for sample in samples:
                input_ids = sample["input_ids"].to(device)
                torch.cuda.reset_peak_memory_stats(device)
                tracker.reset()
                dflash_dispatch_before = None
                if method == "dflash":
                    dflash_dispatch_before = dflash_fa4_attention_stats(
                        context["dflash_model_module"]
                    )
                with tracker.recording():
                    result, elapsed_ms = _time_call(
                        torch,
                        lambda method=method, context=context, input_ids=input_ids: _call_method(
                            torch,
                            method,
                            context,
                            input_ids,
                            max_new_tokens=max_new_tokens,
                        ),
                    )
                attention_dispatch = tracker.snapshot()
                _assert_method_dispatch(method, attention_dispatch)
                _add_dispatch_stats(dispatch_totals, attention_dispatch)
                payload = _output_payload(
                    method, result, input_ids, elapsed_ms, context
                )
                payload["attention_dispatch"] = attention_dispatch
                sample_id = f"{sample['dataset']}:{sample['sample_id']}"
                if method == "dflash":
                    dflash_dispatch_after = dflash_fa4_attention_stats(
                        context["dflash_model_module"]
                    )
                    payload["dflash_fa4_dispatch_calls"] = (
                        dflash_dispatch_after["fa4_dispatch_calls"]
                        - dflash_dispatch_before["fa4_dispatch_calls"]
                    )
                    payload["dflash_sdpa_fallback_calls"] = (
                        dflash_dispatch_after["sdpa_fallback_calls"]
                        - dflash_dispatch_before["sdpa_fallback_calls"]
                    )
                    if (
                        payload["dflash_fa4_dispatch_calls"] <= 0
                        or payload["dflash_sdpa_fallback_calls"] != 0
                    ):
                        raise RuntimeError(
                            "measured DFlash request did not use exclusive FA4 draft attention: "
                            f"fa4={payload['dflash_fa4_dispatch_calls']}, "
                            f"sdpa={payload['dflash_sdpa_fallback_calls']}"
                        )
                generated_ids = payload["output_ids"]
                text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                num_output_tokens = len(generated_ids)
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                ttft = payload["ttft_ms"]
                timing = {
                    "prefill_ms": round(float(ttft), 3) if ttft is not None else None,
                    "ttft_ms": round(float(ttft), 3) if ttft is not None else None,
                    # Vanilla HF exposes no direct TTFT hook. Keep comparisons
                    # on measured end-to-end latency instead of mixing a full
                    # generation time with method-specific decode estimates.
                    "decode_ms": None,
                    "e2e_ms": round(elapsed_ms, 3),
                    "peak_memory_gb": round(peak_gb, 4),
                    "draft_latency_ms": round(float(payload["phases"].get("draft_latency_ms", 0.0)), 3)
                    if payload["phases"].get("draft_latency_ms") is not None
                    else None,
                    "verification_latency_ms": round(float(payload["phases"].get("verification_latency_ms", 0.0)), 3)
                    if payload["phases"].get("verification_latency_ms") is not None
                    else None,
                    "verification_steps": payload["verification_steps"],
                    "draft_tokens_accepted": payload["draft_tokens_accepted"],
                    "draft_tokens_proposed": payload["draft_tokens_proposed"],
                    "draft_proposal_unit": "eagle_tree_node" if method == "eagle3" else "linear_draft_slot",
                }
                if method != "vanilla_hf":
                    from Benchmark.common.speculative_metrics import normalize_speculative_acceptance

                    acceptance = normalize_speculative_acceptance(
                        verification_steps=payload["verification_steps"],
                        draft_tokens_accepted=payload["draft_tokens_accepted"],
                        draft_tokens_proposed=payload["draft_tokens_proposed"],
                        fallback_avg_accept_length=(
                            statistics.mean(payload["acceptance_lengths"])
                            if payload["acceptance_lengths"]
                            else None
                        ),
                    )
                    timing.update(acceptance)
                metadata = runtime_metadata()
                config = {
                    "device": str(device),
                    "gpu_name": metadata.get("gpu_name"),
                    "dtype": "bfloat16",
                    "attention_backend": ATTENTION,
                    "seed": 42,
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "warmup_runs": len(samples),
                    "batch_size": 1,
                    "measurement_scope": "e2e_only",
                    "extra_metrics": {
                        "draft_model": context["draft_name"],
                        "target_attention": method_configs[method]["target_attention"],
                        "draft_attention": method_configs[method]["draft_attention"],
                        "attention_dispatch": attention_dispatch,
                        "target_attention_dispatch": attention_dispatch.get(
                            "target_attention_dispatch"
                        ),
                        "target_attention_dispatch_calls": attention_dispatch.get(
                            "target_attention_dispatch_calls"
                        ),
                        "target_fallback_attention_calls": attention_dispatch.get(
                            "target_fallback_attention_calls"
                        ),
                        "draft_attention_dispatch": attention_dispatch.get(
                            "draft_attention_dispatch"
                        ),
                        "draft_attention_dispatch_calls": attention_dispatch.get(
                            "draft_attention_dispatch_calls"
                        ),
                        "draft_fallback_attention_calls": attention_dispatch.get(
                            "draft_fallback_attention_calls"
                        ),
                        "unattributed_attention_calls": attention_dispatch.get(
                            "unattributed_attention_calls"
                        ),
                        "dflash_block_size": (
                            DFLASH_BLOCK_SIZE if method == "dflash" else None
                        ),
                        "draft_sdpa_fallback_calls": payload.get(
                            "dflash_sdpa_fallback_calls"
                        ),
                        "eagle_tree": context.get("eagle_tree"),
                        "eagle_target_weight_audit": context.get("target_load_audit"),
                        "greedy": True,
                        "tree_mask_probe_passed": runtime["fa4_tree_mask_gpu_probe"]["passed"],
                    },
                }
                record = build_sample_record(
                    method=method,
                    dataset=sample["dataset"],
                    sample_id=sample_id,
                    model=TARGET_MODEL,
                    input_tokens=sample["input_tokens"],
                    output_tokens=num_output_tokens,
                    timing=timing,
                    config=config,
                    text=text,
                    reference_output=sample["reference"],
                )
                if payload["acceptance_lengths"]:
                    record["acceptance_lengths"] = [
                        int(value) for value in payload["acceptance_lengths"]
                    ]
                add_rouge(record, text, sample["reference"])
                quality = repetition_metrics(text)
                record["extra_metrics"].update(
                    {
                        "quality": quality,
                        "quality_nonrepetitive": (
                            int(quality["word_count"]) >= 4
                            and not bool(quality["repetition_flag"])
                        ),
                        "output_token_ids_match_reference_vanilla": None,
                    }
                )
                records.append(record)
                raw[(method, sample_id)] = payload
                print(
                    f"[FA4] {method} {sample_id}: {num_output_tokens} tokens, "
                    f"{elapsed_ms:.1f} ms, {record['throughput_tok_s']} tok/s",
                    flush=True,
                )
            if method != "vanilla_hf":
                for sample in samples:
                    sample_id = f"{sample['dataset']}:{sample['sample_id']}"
                    target_only_raw[(method, sample_id)] = _direct_target_greedy(
                        torch,
                        method,
                        context,
                        sample["input_ids"].to(device),
                        max_new_tokens=max_new_tokens,
                    )
                    vanilla_payload = raw.get(("vanilla_hf", sample_id))
                    if method in {"domino", "dspark"} and vanilla_payload is not None:
                        try:
                            verifier_audit_raw[(method, sample_id)] = (
                                _audit_target_verifier_predictions(
                                    torch,
                                    method,
                                    context,
                                    sample["input_ids"].to(device),
                                    max_new_tokens=max_new_tokens,
                                    reference_output_ids=vanilla_payload["output_ids"],
                                )
                            )
                        except Exception as audit_exc:
                            verifier_audit_raw[(method, sample_id)] = {
                                "target_verifier_audit_error": (
                                    f"{type(audit_exc).__name__}: {audit_exc}"
                                )
                            }
            method_configs[method].update(_dispatch_config_fields(dispatch_totals))
        except Exception as exc:
            failures[method] = f"{type(exc).__name__}: {exc}"
            gpu_context_lost = is_cuda_context_failure(exc)
            print(f"[FA4] ERROR {method}: {failures[method]}", flush=True)
            print(traceback.format_exc(), flush=True)
            for sample in samples:
                sample_id = f"{sample['dataset']}:{sample['sample_id']}"
                if any(
                    row.get("method") == method
                    and row.get("sample_id") == sample_id
                    and row.get("status") == "success"
                    for row in records[method_record_start:]
                ):
                    continue
                records.append(
                    build_status_record(
                        method=method,
                        dataset=sample["dataset"],
                        sample_id=sample_id,
                        status="runtime_error",
                        reason=failures[method],
                        model=TARGET_MODEL,
                        config={
                            "device": str(device),
                            "gpu_name": torch.cuda.get_device_name(0),
                            "dtype": "bfloat16",
                            "attention_backend": ATTENTION,
                            "seed": 42,
                            "temperature": 0.0,
                            "max_new_tokens": max_new_tokens,
                            "batch_size": 1,
                        },
                    )
                )
        finally:
            if context is not None:
                for value in context.values():
                    del value
                del context
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_exc:
                gpu_context_lost = True
                print(
                    "[FA4] CUDA cleanup failed; marking worker GPU context unusable: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                    flush=True,
                )
        if gpu_context_lost:
            reason = f"skipped after CUDA context failure in {method}"
            for skipped_method in selected_methods[selected_methods.index(method) + 1 :]:
                failures[skipped_method] = reason
                for sample in samples:
                    sample_id = f"{sample['dataset']}:{sample['sample_id']}"
                    records.append(
                        build_status_record(
                            method=skipped_method,
                            dataset=sample["dataset"],
                            sample_id=sample_id,
                            status="runtime_error",
                            reason=reason,
                            model=TARGET_MODEL,
                            config={
                                "device": str(device),
                                "gpu_name": runtime.get("gpu_name"),
                                "dtype": "bfloat16",
                                "attention_backend": ATTENTION,
                                "seed": 42,
                                "temperature": 0.0,
                                "max_new_tokens": max_new_tokens,
                                "batch_size": 1,
                            },
                        )
                    )
            break

    for sample in samples:
        sample_id = f"{sample['dataset']}:{sample['sample_id']}"
        vanilla = raw.get(("vanilla_hf", sample_id))
        vanilla_record = next(
            (
                row
                for row in records
                if row.get("method") == "vanilla_hf"
                and row.get("sample_id") == sample_id
                and row.get("status") == "success"
            ),
            None,
        )
        for method in selected_methods:
            record = next(
                (
                    row
                    for row in records
                    if row.get("method") == method
                    and row.get("sample_id") == sample_id
                    and row.get("status") == "success"
                ),
                None,
            )
            payload = raw.get((method, sample_id))
            if record is None or payload is None or vanilla is None or vanilla_record is None:
                continue
            mismatch_index = first_token_mismatch(
                vanilla["output_ids"], payload["output_ids"]
            )
            exact_match = mismatch_index is None
            vanilla_ms = float(vanilla_record["e2e_ms"])
            method_ms = float(record["e2e_ms"])
            record["extra_metrics"].update(
                {
                    "paired_greedy_exact_match": exact_match,
                    "paired_greedy_first_mismatch_token": mismatch_index,
                    "paired_greedy_reference_token_id_at_mismatch": (
                        vanilla["output_ids"][mismatch_index]
                        if mismatch_index is not None
                        and mismatch_index < len(vanilla["output_ids"])
                        else None
                    ),
                    "paired_greedy_candidate_token_id_at_mismatch": (
                        payload["output_ids"][mismatch_index]
                        if mismatch_index is not None
                        and mismatch_index < len(payload["output_ids"])
                        else None
                    ),
                    "direct_target_greedy_exact_match": (
                        first_token_mismatch(
                            vanilla["output_ids"],
                            target_only_raw[(method, sample_id)],
                        )
                        is None
                        if (method, sample_id) in target_only_raw
                        else None
                    ),
                    "direct_target_greedy_first_mismatch_token": (
                        first_token_mismatch(
                            vanilla["output_ids"],
                            target_only_raw[(method, sample_id)],
                        )
                        if (method, sample_id) in target_only_raw
                        else None
                    ),
                    "paired_latency_speedup": round(vanilla_ms / method_ms, 4)
                    if method_ms > 0
                    else None,
                    "paired_throughput_speedup": round(
                        float(record["throughput_tok_s"])
                        / float(vanilla_record["throughput_tok_s"]),
                        4,
                    )
                    if float(vanilla_record["throughput_tok_s"] or 0) > 0
                    and float(record["throughput_tok_s"] or 0) > 0
                    else None,
                }
            )
            record["extra_metrics"].update(
                verifier_audit_raw.get((method, sample_id), {})
            )
            record["extra_metrics"]["output_token_ids_match_reference_vanilla"] = exact_match

    successful = [row for row in records if row.get("status") == "success"]
    summaries: dict[str, dict[str, Any]] = {}
    for method in selected_methods:
        method_rows = [row for row in successful if row.get("method") == method]
        method_summaries = {
            "successful_samples": len(method_rows),
            "mean_e2e_ms": _safe_mean([float(row["e2e_ms"]) for row in method_rows]),
            "mean_throughput_tok_s": _safe_mean(
                [
                    float(row["throughput_tok_s"])
                    for row in method_rows
                    if row.get("throughput_tok_s") is not None
                ]
            ),
            "mean_output_tokens": _safe_mean(
                [float(row["output_tokens"]) for row in method_rows]
            ),
            **aggregate_rouge(method_rows),
            "quality_nonrepetitive_samples": sum(
                bool(row.get("extra_metrics", {}).get("quality_nonrepetitive"))
                for row in method_rows
            ),
            "mean_latency_speedup_vs_vanilla": _safe_mean(
                [
                    float(row["extra_metrics"]["paired_latency_speedup"])
                    for row in method_rows
                    if row.get("extra_metrics", {}).get("paired_latency_speedup") is not None
                ]
            ),
            "mean_throughput_speedup_vs_vanilla": _safe_mean(
                [
                    float(row["extra_metrics"]["paired_throughput_speedup"])
                    for row in method_rows
                    if row.get("extra_metrics", {}).get("paired_throughput_speedup") is not None
                ]
            ),
            "greedy_exact_match_samples": sum(
                bool(row.get("extra_metrics", {}).get("paired_greedy_exact_match"))
                for row in method_rows
            ),
        }
        summaries[method] = method_summaries

    all_quality_valid = len(successful) == len(selected_methods) * len(samples) and all(
        row.get("extra_metrics", {}).get("quality_nonrepetitive")
        for row in successful
    )
    speedup_all_over_one = all(
        summaries[method]["mean_latency_speedup_vs_vanilla"] is not None
        and summaries[method]["mean_latency_speedup_vs_vanilla"] > 1.0
        for method in selected_methods
        if method != "vanilla_hf"
    ) and len(successful) == len(selected_methods) * len(samples)
    actual_runtime = dict(runtime)
    actual_runtime["methods"] = method_configs
    if len(method_configs) == len(selected_methods):
        try:
            runtime_validation = validate_flashattn_runtime(
                actual_runtime,
                methods=selected_methods,
                require_dispatch_proof=True,
            )
        except ValueError as exc:
            runtime_validation = {
                "passed": False,
                "batch_size": 1,
                "attention_backend": ATTENTION,
                "error": f"{type(exc).__name__}: {exc}",
                "methods": method_configs,
            }
    else:
        runtime_validation = {
            **baseline_config,
            "passed": False,
            "methods": method_configs,
            "error": "one or more native baseline runtimes failed to load",
        }
    exact_match_by_method = {
        method: (
            len([row for row in successful if row.get("method") == method])
            == len(samples)
            and all(
                row.get("extra_metrics", {}).get("paired_greedy_exact_match") is True
                for row in successful
                if row.get("method") == method
            )
        )
        for method in selected_methods
        if method != "vanilla_hf"
    }
    exact_match_all = (
        len(successful) == len(selected_methods) * len(samples)
        and all(exact_match_by_method.values())
    )
    run_passed = (
        all_quality_valid
        and exact_match_all
        and runtime_validation.get("passed") is True
        and not failures
    )
    if not runtime_validation.get("passed") or failures:
        run_status = "runtime_failure"
    elif not all_quality_valid:
        run_status = "quality_failure"
    elif not speedup_all_over_one:
        run_status = "speedup_not_above_one"
    elif not exact_match_all:
        run_status = "greedy_parity_failure"
    else:
        run_status = "success"
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_path = Path(f"/tmp/modal_flashattn_{run_id}.jsonl")
    writer = JsonlWriter(output_path)
    for record in records:
        writer.add(record)
    try:
        summary_metadata = runtime_metadata()
    except Exception:
        summary_metadata = {}
    summary_runtime = {**summary_metadata, **runtime, "methods": method_configs}
    summary = {
        "scope": "summary",
        "run_id": run_id,
        "method": "modal_native_fa4_all_baselines",
        "status": run_status,
        "runtime_validation": runtime_validation,
        "runtime": summary_runtime,
        "versions": actual_versions,
        "models": {method: MODEL_REPOS[method] for method in selected_methods},
        "gpu": runtime.get("gpu_name"),
        "attention_backend": ATTENTION,
        "batch_size": 1,
        "max_new_tokens": max_new_tokens,
        "methods": list(selected_methods),
        "dflash_block_size": DFLASH_BLOCK_SIZE if "dflash" in selected_methods else None,
        "max_input_tokens": max_input_tokens,
        "selected_samples": runtime["selected_samples"],
        "sample_count": len(samples),
        "method_summaries": summaries,
        "failures": failures,
        "quality_all_nonrepetitive": all_quality_valid,
        "greedy_exact_match_all_speculative_methods": exact_match_all,
        "greedy_exact_match_by_method": exact_match_by_method,
        "dflash_greedy_exact_match_all_samples": exact_match_by_method.get("dflash"),
        "latency_speedup_above_one_all_speculative": speedup_all_over_one,
        "note": "Smoke trên 2 mẫu là kiểm tra pipeline, không phải số liệu paper.",
    }
    writer.finalize(summary)
    return {
        "run_id": run_id,
        "records": records,
        "summary": summary,
        "jsonl": output_path.read_text(encoding="utf-8"),
    }


def _render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Smoke benchmark các baseline native FlashAttention-4 trên Modal",
        "",
        f"- Mã lượt chạy: `{summary['run_id']}`",
        f"- GPU: {summary['gpu']}; attention: `{summary['attention_backend']}`; batch size: 1",
        f"- Số mẫu dùng chung: {summary['sample_count']}; sinh tối đa {summary['max_new_tokens']} token/mẫu",
        f"- Runtime pin khớp: `{summary['runtime_validation']['passed']}`; kiểm chứng mask FA4: `{summary['runtime'].get('fa4_tree_mask_gpu_probe', {}).get('passed')}`",
    ]
    for method, dispatch in summary["runtime_validation"].get("methods", {}).items():
        lines.append(
            f"- `{method}`: target FA4={dispatch.get('target_attention_dispatch_calls')}, "
            f"target fallback={dispatch.get('target_fallback_attention_calls')}; "
            f"draft FA4={dispatch.get('draft_attention_dispatch_calls')}, "
            f"draft fallback={dispatch.get('draft_fallback_attention_calls')}"
        )
    lines.extend(
        [
            "",
            "| Phương pháp | Mẫu thành công | E2E ms (TB) | E2E token/s (TB) | Speedup độ trễ | Throughput speedup | ROUGE-L | Không lặp | Khớp greedy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, metrics in summary["method_summaries"].items():
        lines.append(
            "| {method} | {count} | {latency} | {throughput} | {lat_speedup} | {tok_speedup} | {rouge} | {quality}/{count} | {exact}/{count} |".format(
                method=method,
                count=metrics["successful_samples"],
                latency=metrics["mean_e2e_ms"],
                throughput=metrics["mean_throughput_tok_s"],
                lat_speedup=metrics["mean_latency_speedup_vs_vanilla"],
                tok_speedup=metrics["mean_throughput_speedup_vs_vanilla"],
                rouge=metrics.get("rougeL"),
                quality=metrics["quality_nonrepetitive_samples"],
                exact=metrics["greedy_exact_match_samples"],
            )
        )
    lines.extend(
        [
            "",
            "## Mẫu đã chọn",
            "",
        ]
    )
    for sample in summary["selected_samples"]:
        lines.append(
            f"- `{sample['dataset']}:{sample['sample_id']}` — {sample['input_tokens']} input token, SHA-256 `{sample['source_sha256']}`"
        )
    lines.extend(["", "## Kiểm tra chất lượng theo mẫu", ""])
    for row in result["records"]:
        if row.get("status") != "success":
            lines.append(
                f"- `{row['method']}` / `{row['sample_id']}`: lỗi `{row.get('reason')}`"
            )
            continue
        extra = row.get("extra_metrics", {})
        quality = extra.get("quality", {})
        lines.append(
            f"- `{row['method']}` / `{row['sample_id']}`: {row['output_tokens']} token, "
            f"ROUGE-L={row.get('rougeL')}, trigram lặp={quality.get('repeated_trigram_ratio')}, "
            f"cờ lặp={quality.get('repetition_flag')}, speedup độ trễ={extra.get('paired_latency_speedup')}, "
            f"khớp greedy={extra.get('paired_greedy_exact_match')}, "
            f"token lệch đầu={extra.get('paired_greedy_first_mismatch_token')}, "
            f"target-only khớp={extra.get('direct_target_greedy_exact_match')}, "
            f"target-only token lệch đầu={extra.get('direct_target_greedy_first_mismatch_token')}, "
            f"vị trí argmax đã audit={extra.get('target_verifier_audit_position_count')}, "
            f"argmax verifier lệch tại token đầu tiên={extra.get('target_verifier_argmax_at_first_output_mismatch')}, "
            f"candidate khớp argmax verifier tại divergence={extra.get('candidate_matches_any_verifier_argmax_at_first_mismatch')}, "
            f"candidate xuất hiện trong proposal verifier tại divergence={extra.get('candidate_matches_any_verifier_proposal_at_first_mismatch')}, "
            f"bước verify phát token lệch={extra.get('emitting_verify_step_diagnostic')}, "
            f"lỗi audit verifier={extra.get('target_verifier_audit_error')}"
        )
    audited_divergences = [
        row
        for row in result["records"]
        if row.get("status") == "success"
        and row.get("extra_metrics", {}).get("paired_greedy_exact_match") is False
        and row.get("extra_metrics", {}).get("emitting_verify_step_diagnostic")
    ]
    if audited_divergences:
        lines.extend(["", "## Audit sai khác greedy", ""])
        all_emissions_match_verifier = True
        for row in audited_divergences:
            extra = row["extra_metrics"]
            diagnostic = extra["emitting_verify_step_diagnostic"]
            target_top1_margin = diagnostic.get("target_top1_margin")
            if target_top1_margin is None:
                target_top1_margin = next(
                    (
                        item.get("top1_margin")
                        for item in extra.get(
                            "target_verifier_argmax_at_first_output_mismatch", []
                        )
                        if item.get("verifier_argmax_token_id")
                        == diagnostic.get("verifier_argmax_token_id")
                        and item.get("verifier_proposal_token_id")
                        == diagnostic.get("verifier_proposal_token_id")
                    ),
                    None,
                )
            all_emissions_match_verifier &= bool(
                diagnostic.get("algorithm_token_matches_candidate")
                and diagnostic.get("reported_accepted_draft_tokens")
                == diagnostic.get("recomputed_accepted_draft_tokens")
            )
            lines.append(
                f"- `{row['method']}` / `{row['sample_id']}` token "
                f"{extra.get('paired_greedy_first_mismatch_token')}: Vanilla="
                f"{diagnostic.get('vanilla_reference_token_id')}, verifier argmax="
                f"{diagnostic.get('verifier_argmax_token_id')}, actual="
                f"{diagnostic.get('actual_candidate_token_id')}, "
                f"margin top1-top2={target_top1_margin}, "
                f"nguồn={diagnostic.get('output_source')}, "
                f"draft accepted báo cáo/tính lại="
                f"{diagnostic.get('reported_accepted_draft_tokens')}/"
                f"{diagnostic.get('recomputed_accepted_draft_tokens')}"
            )
        if all_emissions_match_verifier:
            lines.append(
                "Các token lệch trong audit khớp với quyết định argmax/correction "
                "của chính bước verify FA4, và số draft được nhận báo cáo khớp phép "
                "tính lại. Ở các điểm margin bằng 0, argmax của block verification "
                "khác token của Vanilla autoregressive; vì vậy đây là sai khác "
                "greedy ở tie số học giữa hai cách gọi target, không phải fallback "
                "attention hay lỗi acceptance/correction trong bước đã audit."
            )
    lines.extend(
        [
            "",
            f"Tất cả speculative baseline khớp greedy Vanilla: **{summary['greedy_exact_match_all_speculative_methods']}**.",
            f"Các speculative baseline đều có speedup độ trễ > 1: **{summary['latency_speedup_above_one_all_speculative']}**.",
            f"Mọi baseline dùng chung Qwen3-4B, prompt, BF16, greedy, batch size 1, FA4 và ngân sách token; DFlash dùng block_size={DFLASH_BLOCK_SIZE}. Smoke ít mẫu chỉ kiểm tra pipeline, chưa đủ làm kết luận paper.",
        ]
    )
    if summary["failures"]:
        lines.extend(["", "## Lỗi runtime", ""])
        for method, error in summary["failures"].items():
            lines.append(f"- `{method}`: `{error}`")
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(
    sample_count: int = 2,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    methods: str = "all",
    debug_cuda_launch_blocking: bool = False,
) -> None:
    result = run_flashattn_smoke.remote(
        sample_count=sample_count,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        methods=methods,
        debug_cuda_launch_blocking=debug_cuda_launch_blocking,
    )
    run_dir = PROJECT_ROOT / "outputs" / "modal_flashattn_smoke" / result["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.jsonl").write_text(result["jsonl"], encoding="utf-8")
    (run_dir / "report_vi.md").write_text(_render_report(result), encoding="utf-8")
    print(_render_report(result))
    print(f"\nĐã lưu artifact: {run_dir}")
