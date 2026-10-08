#!/usr/bin/env python3
"""Full batch-one native Transformers FA4 benchmark for all Vietnamese baselines."""

from __future__ import annotations

import hashlib
import json
import csv
from contextlib import contextmanager
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

SERVER_MODE = os.environ.get("FA4_EXECUTION_BACKEND") == "server"
if SERVER_MODE:
    modal = None
else:
    try:
        import modal
    except ImportError:  # The server-native runner does not require the Modal SDK.
        modal = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USE_MODAL = modal is not None and not SERVER_MODE
REMOTE_ROOT = (
    PROJECT_ROOT
    if SERVER_MODE
    else Path("/workspace/fast_infer_text_sum_Viet")
)
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

TARGET_MODEL = (
    os.environ.get("MODEL_TARGET", "")
    if SERVER_MODE
    else os.environ.get("MODAL_QWEN3_MODEL", "Qwen/Qwen3-4B")
)
if SERVER_MODE:
    MODEL_REPOS = {
        "vanilla_hf": TARGET_MODEL,
        "eagle3": os.environ.get("MODEL_EAGLE_DRAFT", ""),
        "dflash": os.environ.get("MODEL_DFLASH_DRAFT", ""),
        "domino": os.environ.get("MODEL_DOMINO_DRAFT", ""),
        "dspark": os.environ.get("MODEL_DSPARK_DRAFT", ""),
    }
else:
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
MAX_INPUT_TOKENS = int(os.environ.get("MODAL_MAX_INPUT_TOKENS", "8192"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MODAL_MAX_NEW_TOKENS", "512"))
DFLASH_BLOCK_SIZE = 16
ATTENTION = "flash_attention_4"
OUTPUT_VOLUME_NAME = os.environ.get(
    "MODAL_FA4_OUTPUT_VOLUME", "fast-infer-viet-fa4-results"
)
REMOTE_OUTPUT_ROOT = (
    Path(os.environ.get("FA4_OUTPUT_ROOT", PROJECT_ROOT / "outputs/fa4_native_benchmark"))
    if SERVER_MODE
    else Path("/fa4-results")
)
ARTIFACT_FILENAMES = (
    "results.jsonl",
    "run_report.json",
    "report_vi.md",
    "metrics_summary.csv",
    "warmup.jsonl",
    "events.jsonl",
    "samples.jsonl",
    "excluded_samples.jsonl",
    "progress.json",
    "state.json",
    "results.partial.jsonl",
)

PINNED_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.12.1",
    "tokenizers": "0.22.2",
    "accelerate": "1.15.0",
    "huggingface-hub": "1.31.0",
    "flash-attn-4": "4.0.0b32",
    "quack-kernels": "0.6.5",
    "triton": "3.7.1",
    "apache-tvm-ffi": "0.1.11",
    "nvidia-cutlass-dsl": "4.7.1",
}

class _LocalApp:
    def function(self, **_kwargs):
        return lambda function: function

    def local_entrypoint(self, **_kwargs):
        return lambda function: function


class _LocalVolume:
    def commit(self) -> None:
        return None


if USE_MODAL:
    app = modal.App("fast-infer-viet-native-flashattn-benchmark")
    OUTPUT_VOLUME = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
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
            "flash-attn-4[cu13]==4.0.0b32",
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
        .add_local_dir(
            str(PROJECT_ROOT / "src"),
            remote_path=str(REMOTE_SRC),
            copy=True,
            ignore=["**/*vllm*"],
        )
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
    image = image.add_local_file(
        str(PROJECT_ROOT / "datasets" / "eval_100" / "manifest.json"),
        remote_path=str(REMOTE_ROOT / "datasets" / "eval_100" / "manifest.json"),
        copy=True,
    )
else:
    app = _LocalApp()
    OUTPUT_VOLUME = _LocalVolume()
    image = None


def _install_fa4_compat() -> None:
    from Benchmark.common.vanilla_inference import (
        _install_flash_attention_4_cutlass_compat,
    )

    _install_flash_attention_4_cutlass_compat()

    # Preserve the process-local compatibility alias for CuTe/QuACK combinations
    # that still use the removed export; this does not select a CUDA backend.
    try:
        import cutlass.cute.arch as cute_arch
        import quack.activation as quack_activation
    except ImportError:
        return
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


def _prepare_samples(
    tokenizer,
    *,
    datasets: tuple[str, ...],
    samples_per_dataset: int,
    max_input_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from Benchmark.common.benchmark_data import read_jsonl, render_prompt
    from Benchmark.common.fa4_benchmark import select_length_spread
    from Benchmark.common.input_utils import truncate_input_ids
    from Benchmark.common.prompt_format import format_chat_prompt

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for dataset in datasets:
        candidates: list[dict[str, Any]] = []
        data_path = REMOTE_DATA_FILES[dataset]
        dataset_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
        for row in read_jsonl(data_path):
            prompt = format_chat_prompt(tokenizer, render_prompt(row))
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            source_input_tokens = int(input_ids.shape[1])
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
                    "source_input_tokens": source_input_tokens,
                    "was_truncated": int(input_ids.shape[1]) < source_input_tokens,
                    "input_ids": input_ids,
                    "reference": str(reference or ""),
                    "prompt": prompt,
                    "source_index": row.get("source_index"),
                    "length_bin": row.get("length_bin"),
                    "source_sha256": hashlib.sha256(
                        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
                    ).hexdigest(),
                    "dataset_sha256": dataset_sha256,
                }
            )
        if not candidates:
            raise ValueError(f"no samples found for dataset={dataset}")
        dataset_selected = select_length_spread(candidates, samples_per_dataset)
        selected.extend(dataset_selected)
        selected_ids = {row["sample_id"] for row in dataset_selected}
        for candidate in candidates:
            if candidate["sample_id"] not in selected_ids:
                excluded.append(
                    {
                        "dataset": dataset,
                        "sample_id": candidate["sample_id"],
                        "input_tokens": candidate["input_tokens"],
                        "reason": "outside_deterministic_length_spread_sample",
                    }
                )
    return sorted(
        selected,
        key=lambda row: (DATASETS.index(row["dataset"]), row["input_tokens"], row["sample_id"]),
    ), excluded


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


def _warm_method(
    torch,
    method: str,
    context: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    warmup_tokens: int,
) -> list[dict[str, Any]]:
    warmup_records = []
    for sample in samples:
        input_ids = sample["input_ids"].to("cuda:0")
        result, elapsed_ms = _time_call(
            torch,
            lambda: _call_method(
                torch,
                method,
                context,
                input_ids,
                max_new_tokens=min(warmup_tokens, max_new_tokens),
                warmup=True,
            ),
        )
        payload = _output_payload(method, result, input_ids, elapsed_ms, context)
        warmup_records.append(
            {
                "method": method,
                "dataset": sample["dataset"],
                "sample_id": f"{sample['dataset']}:{sample['sample_id']}",
                "warmup_tokens_requested": min(warmup_tokens, max_new_tokens),
                "warmup_tokens_generated": len(payload["output_ids"]),
                "warmup_e2e_ms": round(elapsed_ms, 3),
            }
        )
    torch.cuda.synchronize("cuda:0")
    return warmup_records


@contextmanager
def _capture_first_target_forward_ms(torch, model):
    """Capture GPU execution time of the initial target forward without syncs per call."""
    original_forward = model.forward
    captured: dict[str, Any] = {}

    def wrapped_forward(*args, **kwargs):
        if "start_event" in captured:
            return original_forward(*args, **kwargs)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        try:
            output = original_forward(*args, **kwargs)
        finally:
            end_event.record()
            captured["start_event"] = start_event
            captured["end_event"] = end_event
        return output

    model.forward = wrapped_forward
    try:
        yield captured
    finally:
        model.forward = original_forward
        if "start_event" in captured:
            torch.cuda.synchronize("cuda:0")
            captured["first_forward_ms"] = float(
                captured["start_event"].elapsed_time(captured["end_event"])
            )


def _call_method(
    torch,
    method: str,
    context: dict[str, Any],
    input_ids,
    *,
    max_new_tokens: int,
    warmup: bool = False,
    profiling: bool = False,
):
    from Benchmark.native_flashattn import run_native_method

    return run_native_method(
        torch, method, context, input_ids,
        max_new_tokens=max_new_tokens, profiling=profiling,
    )


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


def _load_method(torch, method: str, tokenizer, device, *, native_config=None) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM
    from Benchmark.native_flashattn import NativeInferenceConfig, build_domino_graph_runner

    native_config = native_config or NativeInferenceConfig()

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
        "native_config": native_config,
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
            domino_model_module=domino_module,
            domino_graph_runner=(
                build_domino_graph_runner(
                    draft, target, device,
                    source_path=REMOTE_ROOT / "externals/Domino/code/kernel/domino.py",
                ) if native_config.domino_cuda_graph else None
            ),
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
            confidence_threshold=native_config.dspark_confidence_threshold,
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
        from eagle.model import ea_model as eagle_model_module

        # Match the project's native AR profile; keep the native decoding loop.
        tree = resolve_eagle_tree_config(**native_config.eagle_tree())
        eagle_model = eagle_model_module.EaModel.from_pretrained(
            base_model_path=target_name,
            ea_model_path=draft_name,
            total_token=int(tree["total_token"]),
            depth=native_config.eagle_depth,
            top_k=native_config.eagle_top_k,
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
            eagle_model_module=eagle_model_module,
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
        phases = {
            name: (
                float(getattr(result, name))
                if getattr(result, name, None) is not None
                else None
            )
            for name in ("draft_latency_ms", "verification_latency_ms")
        }
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
        phases = {
            name: (
                float(getattr(result, name))
                if getattr(result, name, None) is not None
                else None
            )
            for name in ("prefill_ms", "decode_ms", "draft_latency_ms", "verification_latency_ms")
        }
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


@app.function(
    image=image,
    gpu=GPU,
    cpu=16,
    memory=65536,
    timeout=86400,
    max_containers=1,
    volumes={str(REMOTE_OUTPUT_ROOT): OUTPUT_VOLUME},
)
def run_flashattn_benchmark(
    *,
    mode: str = "smoke",
    datasets: str = "all",
    samples_per_dataset: int | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    methods: str = "all",
    warmup_tokens: int = 512,
    eagle_total_token: int = 17,
    eagle_depth: int = 16,
    eagle_top_k: int = 1,
    domino_cuda_graph: bool = True,
    dspark_confidence_threshold: float = 0.0,
    phase_timing_mode: str = "separate",
    strict_greedy_parity: bool = False,
    require_speedup: bool = False,
    repetitions: int = 1,
    seed: int = 42,
    sample_retries: int = 1,
    checkpoint_interval: int = 20,
    run_id: str = "",
    resume: bool = False,
    direct_target_audit: bool = False,
    verifier_audit: bool = False,
    preflight_only: bool = False,
    debug_cuda_launch_blocking: bool = False,
) -> dict[str, Any]:
    """Benchmark selected datasets/methods with paired batch-one requests."""
    import gc
    import sys

    started_perf = time.perf_counter()
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    sys.path.insert(0, str(REMOTE_SRC))
    from Benchmark.native_flashattn import (
        NativeInferenceConfig, collect_native_phase_profile, native_run_status, native_source_manifest,
        initialize_eagle_request_cache, prepare_measured_payload, validate_fa4_version,
    )

    native_config = NativeInferenceConfig(
        eagle_total_token=eagle_total_token, eagle_depth=eagle_depth, eagle_top_k=eagle_top_k,
        domino_cuda_graph=domino_cuda_graph, dspark_confidence_threshold=dspark_confidence_threshold,
        phase_timing_mode=phase_timing_mode, strict_greedy_parity=strict_greedy_parity,
        require_speedup=require_speedup,
    )
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
    from Benchmark.common.fa4_benchmark import (
        finalize_fa4_records,
        parse_selection,
        resolve_sample_limit,
    )
    from Benchmark.common.benchmark_data import validate_output_dir
    from Benchmark.common.flashattn_runtime import (
        AttentionDispatchTracker,
        first_token_mismatch,
        install_attention_dispatch_tracking,
        is_cuda_context_failure,
        resolve_flashattn_methods,
        validate_flashattn_runtime,
    )
    from Benchmark.common.io_util import JsonlWriter, validate_schema
    from Benchmark.common.quality_guard import repetition_metrics
    from Benchmark.common.rouge import add_rouge, aggregate_rouge

    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark process did not expose a CUDA GPU")
    if torch.cuda.get_device_capability(0)[0] < 10:
        raise RuntimeError("FA4 benchmark requires a Blackwell-class GPU (SM100+)")
    if mode not in {"smoke", "representative", "full"}:
        raise ValueError("mode must be smoke, representative, or full")
    if max_new_tokens < 1 or max_input_tokens < 1:
        raise ValueError("max-new-tokens and max-input-tokens must be positive")
    if warmup_tokens < 1 or repetitions < 1 or sample_retries < 0:
        raise ValueError("warmup-tokens/repetitions must be positive; sample-retries >= 0")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint-interval must be positive")
    if resume and not run_id:
        raise ValueError("resume requires an explicit run-id")
    selected_datasets = parse_selection(datasets, DATASETS, "dataset")
    selected_methods = resolve_flashattn_methods(methods)
    sample_limit = resolve_sample_limit(mode, samples_per_dataset, available=100)
    if not run_id:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + os.urandom(3).hex()
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run-id may contain only letters, numbers, hyphen and underscore")
    actual_versions = _version_snapshot()
    validate_fa4_version(actual_versions)
    if USE_MODAL:
        _validate_pins(actual_versions)
    try:
        import flash_attn.cute  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"FA4 CuTe runtime failed to import: {exc}") from exc
    install_attention_dispatch_tracking(ALL_ATTENTION_FUNCTIONS)
    runtime = _runtime_base(
        torch, actual_versions, methods=selected_methods
    )
    runtime["native_inference_config"] = native_config.manifest()
    baseline_config = validate_flashattn_runtime(
        runtime,
        methods=selected_methods,
        allow_installed_vllm=SERVER_MODE,
    )
    runtime["fa4_tree_mask_gpu_probe"] = _fa4_tree_mask_gpu_probe(torch)
    runtime["dataset_validation"] = validate_output_dir(
        REMOTE_ROOT / "datasets" / "eval_100", expected_count=100
    )
    runtime["native_source_sha256"] = native_source_manifest(REMOTE_ROOT, selected_methods)

    if preflight_only:
        return {
            "run_id": run_id,
            "summary": {
                "status": "preflight_passed",
                "run_id": run_id,
                "runtime_validation": baseline_config,
                "runtime": runtime,
                "versions": actual_versions,
                "gpu": runtime.get("gpu_name"),
                "attention_backend": ATTENTION,
                "batch_size": 1,
                "native_inference_config": native_config.manifest(),
                "methods": list(selected_methods),
                "datasets": list(selected_datasets),
            },
            "records": [],
            "jsonl": "",
            "artifact_files": {},
        }

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples, excluded_samples = _prepare_samples(
        tokenizer,
        datasets=selected_datasets,
        samples_per_dataset=sample_limit,
        max_input_tokens=max_input_tokens,
    )
    runtime["selected_samples"] = [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "input_tokens": row["input_tokens"],
            "source_input_tokens": row["source_input_tokens"],
            "was_truncated": row["was_truncated"],
            "source_sha256": row["source_sha256"],
        }
        for row in samples
    ]

    selected_sample_manifest = [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "source_sha256": row["source_sha256"],
            "input_tokens": row["input_tokens"],
        }
        for row in samples
    ]
    experiment_config = {
        "schema_version": 2,
        "native_inference_config": native_config.manifest(),
        "native_source_sha256": runtime["native_source_sha256"],
        "mode": mode,
        "datasets": list(selected_datasets),
        "methods": list(selected_methods),
        "sample_limit_per_dataset": sample_limit,
        "selected_samples_sha256": hashlib.sha256(
            json.dumps(
                selected_sample_manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "models": {method: MODEL_REPOS[method] for method in selected_methods},
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "warmup_tokens": warmup_tokens,
        "repetitions": repetitions,
        "seed": seed,
        "batch_size": 1,
        "dtype": "bfloat16",
        "attention_backend": ATTENTION,
        "direct_target_audit": direct_target_audit,
        "verifier_audit": verifier_audit,
        "package_versions": actual_versions,
    }
    experiment_signature = hashlib.sha256(
        json.dumps(
            experiment_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    records: list[dict[str, Any]] = []
    raw: dict[tuple[str, str, int], dict[str, Any]] = {}
    target_only_raw: dict[tuple[str, str, int], list[int]] = {}
    verifier_audit_raw: dict[tuple[str, str], dict[str, Any]] = {}
    method_configs: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    gpu_context_lost = False
    remote_run_dir = REMOTE_OUTPUT_ROOT / run_id
    remote_run_dir.mkdir(parents=True, exist_ok=True)
    partial_path = remote_run_dir / "results.partial.jsonl"
    warmup_path = remote_run_dir / "warmup.jsonl"
    events_path = remote_run_dir / "events.jsonl"
    state_path = remote_run_dir / "state.json"
    samples_path = remote_run_dir / "samples.jsonl"
    excluded_path = remote_run_dir / "excluded_samples.jsonl"

    def append_jsonl(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def persist_state() -> None:
        write_json(
            state_path,
            {
                "experiment_config": experiment_config,
                "experiment_signature": experiment_signature,
                "method_configs": method_configs,
                "failures": failures,
            },
        )

    if resume:
        if not state_path.is_file():
            raise ValueError(
                "cannot safely resume this run: state.json with an experiment signature is missing"
            )
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        if saved_state.get("experiment_signature") != experiment_signature:
            raise ValueError(
                "resume configuration does not match the original run; use its exact "
                "dataset/model/token/repetition/audit/native settings, package versions "
                "and source revision or choose a new run-id"
            )
        records = read_jsonl(partial_path)
        if not records:
            records = [
                row for row in read_jsonl(remote_run_dir / "results.jsonl")
                if row.get("scope") == "sample"
            ]
        method_configs.update(saved_state.get("method_configs", {}))
        for row in records:
            if row.get("status") == "success" and row.get("output_token_ids") is not None:
                key = (
                    str(row["method"]),
                    str(row["sample_id"]),
                    int(row.get("repeat_index", 0)),
                )
                raw[key] = {
                    "output_ids": [int(value) for value in row["output_token_ids"]],
                    "elapsed_ms": float(row["e2e_ms"]),
                    "acceptance_lengths": row.get("acceptance_lengths", []),
                    "draft_tokens_accepted": row.get("draft_tokens_accepted"),
                    "draft_tokens_proposed": row.get("draft_tokens_proposed"),
                    "verification_steps": row.get("verification_steps"),
                    "phases": {},
                }
    else:
        if any(path.exists() for path in (partial_path, remote_run_dir / "results.jsonl")):
            raise FileExistsError(
                f"run directory already contains results: {remote_run_dir}; pass resume=true or a new run-id"
            )
        samples_path.write_text("", encoding="utf-8")
        excluded_path.write_text("", encoding="utf-8")
        partial_path.write_text("", encoding="utf-8")
        warmup_path.write_text("", encoding="utf-8")
        events_path.write_text("", encoding="utf-8")
        for sample in samples:
            append_jsonl(
                samples_path,
                {
                    key: sample.get(key)
                    for key in (
                        "dataset", "sample_id", "source_index", "input_tokens",
                        "source_input_tokens", "was_truncated", "source_sha256", "dataset_sha256",
                    )
                },
            )
        for row in excluded_samples:
            append_jsonl(excluded_path, row)
        persist_state()
        OUTPUT_VOLUME.commit()

    append_jsonl(
        events_path,
        {
            "event": "run_started",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": run_id,
            "mode": mode,
            "datasets": selected_datasets,
            "methods": selected_methods,
            "sample_count": len(samples),
            "repetitions": repetitions,
            "resume": resume,
        },
    )
    OUTPUT_VOLUME.commit()

    for method in selected_methods:
        print(f"[FA4] loading {method}: {MODEL_REPOS[method]}", flush=True)
        context = None
        tracker = None
        dispatch_totals = _new_dispatch_totals()
        try:
            model_load_started = time.perf_counter()
            context = _load_method(torch, method, tokenizer, device, native_config=native_config)
            context["seed"] = seed
            method_model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
            append_jsonl(
                events_path,
                {
                    "event": "method_loaded",
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "method": method,
                    "model": MODEL_REPOS[method],
                    "model_load_ms": round(method_model_load_ms, 3),
                },
            )
            method_configs[method] = {
                "native_inference_config": native_config.manifest(),
                "domino_cuda_graph_active": context.get("domino_graph_runner") is not None,
                "eagle_tree": context.get("eagle_tree"),
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
                initialize_eagle_request_cache(
                    context["eagle_model"], max_prompt + max_new_tokens + 96,
                )

            tracker = AttentionDispatchTracker(
                context["target"], _dispatch_draft_model(method, context)
            )
            with tracker.recording():
                warmup_rows = _warm_method(
                    torch,
                    method,
                    context,
                    samples,
                    max_new_tokens=max_new_tokens,
                    warmup_tokens=warmup_tokens,
                )
            for warmup_row in warmup_rows:
                append_jsonl(warmup_path, warmup_row)
            OUTPUT_VOLUME.commit()
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
                sample_id = f"{sample['dataset']}:{sample['sample_id']}"
                for repeat_index in range(repetitions):
                    raw_key = (method, sample_id, repeat_index)
                    previous = next(
                        (
                            row for row in records
                            if row.get("method") == method
                            and row.get("sample_id") == sample_id
                            and int(row.get("repeat_index", 0)) == repeat_index
                        ),
                        None,
                    )
                    if previous is not None and previous.get("status") == "success":
                        continue
                    if previous is not None:
                        records.remove(previous)
                    sample_error = None
                    sample_error_traceback = None
                    for attempt in range(sample_retries + 1):
                        try:
                            input_ids = sample["input_ids"].to(device)
                            torch.cuda.reset_peak_memory_stats(device)
                            tracker.reset()
                            dflash_dispatch_before = None
                            if method == "dflash":
                                dflash_dispatch_before = dflash_fa4_attention_stats(
                                    context["dflash_model_module"]
                                )
                            with _capture_first_target_forward_ms(
                                torch, context["target"]
                            ) as first_forward:
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
                            payload["first_target_forward_ms"] = first_forward.get(
                                "first_forward_ms"
                            )
                            payload["attention_dispatch"] = attention_dispatch
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

                            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                            skipped_syncs = context.get("last_skipped_profiling_synchronizations", 0)
                            payload = prepare_measured_payload(
                                payload, first_forward_ms=payload["first_target_forward_ms"],
                                mode=native_config.phase_timing_mode,
                            )
                            del result

                            def profile_call():
                                # Run after the E2E timer and dispatch tracker have closed.
                                profile_result, profile_elapsed_ms = _time_call(
                                    torch, lambda: _call_method(
                                        torch, method, context, input_ids,
                                        max_new_tokens=max_new_tokens, profiling=True,
                                    ),
                                )
                                return _output_payload(
                                    method, profile_result, input_ids, profile_elapsed_ms, context,
                                )

                            phase_profile = collect_native_phase_profile(
                                native_config, method, payload, profile_call,
                            )
                            # A nested callback must not retain the model context between methods.
                            del profile_call

                            generated_ids = payload["output_ids"]
                            text = tokenizer.decode(
                                generated_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            ).strip()
                            num_output_tokens = len(generated_ids)
                            prefill = payload["ttft_ms"]
                            if prefill is None:
                                prefill = payload.get("first_target_forward_ms")
                            if prefill is not None:
                                prefill = max(0.0, float(prefill))
                            decode_ms = (
                                max(0.0, float(elapsed_ms) - prefill)
                                if prefill is not None else None
                            )
                            timing = {
                                "prefill_ms": round(prefill, 3) if prefill is not None else None,
                                "ttft_ms": round(prefill, 3) if prefill is not None else None,
                                "decode_ms": round(decode_ms, 3) if decode_ms is not None else None,
                                "e2e_ms": round(elapsed_ms, 3),
                                "server_reported_e2e_ms": None,
                                "peak_memory_gb": round(peak_gb, 4),
                                "model_load_ms": round(method_model_load_ms, 3),
                                "qps": round(1000.0 / elapsed_ms, 4) if elapsed_ms > 0 else None,
                                "draft_latency_ms": payload["phases"].get("draft_latency_ms"),
                                "verification_latency_ms": payload["phases"].get("verification_latency_ms"),
                                "verification_steps": payload["verification_steps"],
                                "draft_tokens_accepted": payload["draft_tokens_accepted"],
                                "draft_tokens_proposed": payload["draft_tokens_proposed"],
                                "draft_proposal_unit": "eagle_tree_node" if method == "eagle3" else "linear_draft_slot",
                            }
                            if num_output_tokens > 1 and decode_ms is not None and decode_ms > 0:
                                timing["tpot_ms"] = round(decode_ms / (num_output_tokens - 1), 4)
                                timing["decode_throughput_tok_s"] = round(
                                    (num_output_tokens - 1) / (decode_ms / 1000.0), 4
                                )
                            if method != "vanilla_hf":
                                from Benchmark.common.speculative_metrics import normalize_speculative_acceptance

                                timing.update(
                                    normalize_speculative_acceptance(
                                        verification_steps=payload["verification_steps"],
                                        draft_tokens_accepted=payload["draft_tokens_accepted"],
                                        draft_tokens_proposed=payload["draft_tokens_proposed"],
                                        fallback_avg_accept_length=(
                                            statistics.mean(payload["acceptance_lengths"])
                                            if payload["acceptance_lengths"] else None
                                        ),
                                    )
                                )
                            metadata = runtime_metadata()
                            config = {
                                "device": str(device),
                                "gpu_name": metadata.get("gpu_name"),
                                "dtype": "bfloat16",
                                "attention_backend": ATTENTION,
                                "seed": seed,
                                "temperature": 0.0,
                                "max_new_tokens": max_new_tokens,
                                "warmup_runs": len(samples),
                                "batch_size": 1,
                                "measurement_scope": "e2e_and_initial_target_forward",
                                "extra_metrics": {
                                    "backend": "transformers_native",
                                    "draft_model": context["draft_name"],
                                    "target_attention": method_configs[method]["target_attention"],
                                    "draft_attention": method_configs[method]["draft_attention"],
                                    "attention_dispatch": attention_dispatch,
                                    "target_attention_dispatch": attention_dispatch.get("target_attention_dispatch"),
                                    "target_attention_dispatch_calls": attention_dispatch.get("target_attention_dispatch_calls"),
                                    "target_fallback_attention_calls": attention_dispatch.get("target_fallback_attention_calls"),
                                    "draft_attention_dispatch": attention_dispatch.get("draft_attention_dispatch"),
                                    "draft_attention_dispatch_calls": attention_dispatch.get("draft_attention_dispatch_calls"),
                                    "draft_fallback_attention_calls": attention_dispatch.get("draft_fallback_attention_calls"),
                                    "unattributed_attention_calls": attention_dispatch.get("unattributed_attention_calls"),
                                    "dflash_block_size": DFLASH_BLOCK_SIZE if method == "dflash" else None,
                                    "draft_sdpa_fallback_calls": payload.get("dflash_sdpa_fallback_calls"),
                                    "eagle_tree": context.get("eagle_tree"),
                                    "native_inference_config": native_config.manifest(),
                                    "domino_cuda_graph_active": context.get("domino_graph_runner") is not None,
                                    "phase_profile": phase_profile,
                                    "skipped_profiling_synchronizations": skipped_syncs,
                                    "eagle_target_weight_audit": context.get("target_load_audit"),
                                    "greedy": True,
                                    "repeat_index": repeat_index,
                                    "initial_target_forward_measurement": (
                                        "native_method_ttft" if native_config.phase_timing_mode == "inline" and payload["ttft_ms"] is not None
                                        else "cuda_event_first_target_forward"
                                    ),
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
                            record.update(
                                {
                                    "repeat_index": repeat_index,
                                    "source_input_tokens": sample["source_input_tokens"],
                                    "input_was_truncated": sample["was_truncated"],
                                    "output_token_ids": generated_ids,
                                    "model_load_ms": round(method_model_load_ms, 3),
                                    "qps": timing["qps"],
                                    "tpot_ms": timing.get("tpot_ms"),
                                    "decode_throughput_tok_s": timing.get("decode_throughput_tok_s"),
                                    "status": "success" if generated_ids else "invalid_output",
                                    "repetition_flag": None,
                                    "quality_valid": None,
                                }
                            )
                            if payload["acceptance_lengths"]:
                                record["acceptance_lengths"] = [int(value) for value in payload["acceptance_lengths"]]
                            add_rouge(record, text, sample["reference"])
                            from Benchmark.common.metrics import add_semantic
                            add_semantic(record, text, sample["reference"])
                            quality = repetition_metrics(text)
                            record["repetition_flag"] = bool(quality["repetition_flag"])
                            record["quality_valid"] = bool(
                                text.strip() and num_output_tokens >= 4 and not quality["repetition_flag"]
                            )
                            record["extra_metrics"].update(
                                {
                                    "quality": quality,
                                    "quality_nonrepetitive": record["quality_valid"],
                                    "output_token_ids_match_reference_vanilla": None,
                                    "first_target_forward_ms": payload.get("first_target_forward_ms"),
                                }
                            )
                            records.append(record)
                            raw[raw_key] = payload
                            append_jsonl(partial_path, record)
                            append_jsonl(
                                events_path,
                                {
                                    "event": "sample_finished",
                                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "method": method,
                                    "dataset": sample["dataset"],
                                    "sample_id": sample_id,
                                    "repeat_index": repeat_index,
                                    "status": record["status"],
                                    "attempt": attempt + 1,
                                },
                            )
                            if len(records) % checkpoint_interval == 0:
                                OUTPUT_VOLUME.commit()
                            print(
                                f"[FA4] {method} {sample_id} r{repeat_index}: {num_output_tokens} tokens, "
                                f"{elapsed_ms:.1f} ms, {record['throughput_tok_s']} tok/s",
                                flush=True,
                            )
                            sample_error = None
                            break
                        except Exception as exc:
                            sample_error = exc
                            sample_error_traceback = traceback.format_exc()
                            if is_cuda_context_failure(exc):
                                gpu_context_lost = True
                                break
                            if attempt < sample_retries:
                                print(
                                    f"[FA4] retry {attempt + 1}/{sample_retries} {method} {sample_id} r{repeat_index}: "
                                    f"{type(exc).__name__}: {exc}",
                                    flush=True,
                                )
                                gc.collect()
                                torch.cuda.empty_cache()
                    if sample_error is not None:
                        error_text = f"{type(sample_error).__name__}: {sample_error}"
                        failure_record = build_status_record(
                            method=method,
                            dataset=sample["dataset"],
                            sample_id=sample_id,
                            status="runtime_error",
                            reason=error_text,
                            model=TARGET_MODEL,
                            config={
                                "device": str(device),
                                "gpu_name": runtime.get("gpu_name"),
                                "dtype": "bfloat16",
                                "attention_backend": ATTENTION,
                                "seed": seed,
                                "temperature": 0.0,
                                "max_new_tokens": max_new_tokens,
                                "batch_size": 1,
                            },
                        )
                        failure_record.update(
                            {
                                "repeat_index": repeat_index,
                                "source_input_tokens": sample["source_input_tokens"],
                                "input_was_truncated": sample["was_truncated"],
                                "error_traceback": sample_error_traceback,
                            }
                        )
                        records.append(failure_record)
                        append_jsonl(partial_path, failure_record)
                        OUTPUT_VOLUME.commit()
                        print(f"[FA4] sample failed {method} {sample_id}: {error_text}", flush=True)
                    if gpu_context_lost:
                        break
                if gpu_context_lost:
                    break
            if method != "vanilla_hf" and not gpu_context_lost:
                for sample in samples:
                    sample_id = f"{sample['dataset']}:{sample['sample_id']}"
                    if direct_target_audit:
                        target_only_raw[(method, sample_id, 0)] = _direct_target_greedy(
                            torch,
                            method,
                            context,
                            sample["input_ids"].to(device),
                            max_new_tokens=max_new_tokens,
                        )
                    vanilla_payload = raw.get(("vanilla_hf", sample_id, 0))
                    if verifier_audit and method in {"domino", "dspark"} and vanilla_payload is not None:
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
            persist_state()
            OUTPUT_VOLUME.commit()
        except Exception as exc:
            failures[method] = f"{type(exc).__name__}: {exc}"
            gpu_context_lost = is_cuda_context_failure(exc)
            print(f"[FA4] ERROR {method}: {failures[method]}", flush=True)
            print(traceback.format_exc(), flush=True)
            for sample in samples:
                sample_id = f"{sample['dataset']}:{sample['sample_id']}"
                for repeat_index in range(repetitions):
                    existing_success = any(
                        row.get("method") == method
                        and row.get("sample_id") == sample_id
                        and int(row.get("repeat_index", 0)) == repeat_index
                        and row.get("status") == "success"
                        for row in records
                    )
                    if existing_success:
                        continue
                    failure_record = build_status_record(
                        method=method,
                        dataset=sample["dataset"],
                        sample_id=sample_id,
                        status="runtime_error",
                        reason=failures[method],
                        model=TARGET_MODEL,
                        config={
                            "device": str(device),
                            "gpu_name": runtime.get("gpu_name"),
                            "dtype": "bfloat16",
                            "attention_backend": ATTENTION,
                            "seed": seed,
                            "temperature": 0.0,
                            "max_new_tokens": max_new_tokens,
                            "batch_size": 1,
                        },
                    )
                    failure_record.update(
                        {
                            "repeat_index": repeat_index,
                            "source_input_tokens": sample["source_input_tokens"],
                            "input_was_truncated": sample["was_truncated"],
                            "error_traceback": traceback.format_exc(),
                        }
                    )
                    records.append(failure_record)
                    append_jsonl(partial_path, failure_record)
            persist_state()
            OUTPUT_VOLUME.commit()
        finally:
            tracker = None
            if context is not None:
                context.clear()
                context = None
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
                    for repeat_index in range(repetitions):
                        if any(
                            row.get("method") == skipped_method
                            and row.get("sample_id") == sample_id
                            and int(row.get("repeat_index", 0)) == repeat_index
                            for row in records
                        ):
                            continue
                        failure_record = build_status_record(
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
                                "seed": seed,
                                "temperature": 0.0,
                                "max_new_tokens": max_new_tokens,
                                "batch_size": 1,
                            },
                        )
                        failure_record["repeat_index"] = repeat_index
                        records.append(failure_record)
                        append_jsonl(partial_path, failure_record)
                OUTPUT_VOLUME.commit()
            break

    # Keep optional verifier diagnostics on sample records before paired aggregation.
    for record in records:
        if record.get("status") != "success":
            continue
        method = str(record.get("method", ""))
        sample_id = str(record.get("sample_id", ""))
        record.setdefault("extra_metrics", {}).update(
            verifier_audit_raw.get((method, sample_id), {})
        )
        repeat_index = int(record.get("repeat_index", 0))
        direct_target = target_only_raw.get((method, sample_id, repeat_index))
        if direct_target is not None:
            reference = next(
                (
                    row.get("output_token_ids", [])
                    for row in records
                    if row.get("method") == "vanilla_hf"
                    and row.get("dataset") == record.get("dataset")
                    and row.get("sample_id") == sample_id
                    and int(row.get("repeat_index", 0)) == repeat_index
                    and row.get("status") == "success"
                ),
                [],
            )
            record["extra_metrics"]["direct_target_greedy_exact_match"] = (
                first_token_mismatch(reference, direct_target) is None
            )
            record["extra_metrics"]["direct_target_greedy_first_mismatch_token"] = (
                first_token_mismatch(reference, direct_target)
            )

    finalized = finalize_fa4_records(
        records,
        methods=selected_methods,
        datasets=selected_datasets,
        repetitions=repetitions,
        expected_samples=len(samples),
        reference_method="vanilla_hf",
    )
    records = finalized["records"]
    metric_bundle = finalized
    expected_cells = finalized["expected_records"]
    execution_complete = finalized["execution_complete"]
    execution_pass = finalized["execution_pass"]
    quality_pass = finalized["quality_pass"]
    exact_match_by_method = finalized["exact_match_by_method"]
    exact_match_all = finalized["exact_match_all"]
    speedup_by_method = finalized["speedup_by_method"]
    speedup_all_over_one = finalized["speedup_all_over_one"]
    failure_count = finalized["failure_count"]
    schema_violations = []
    for record in records:
        schema_errors = validate_schema(
            record, spec=record.get("method") != "vanilla_hf"
        )
        if schema_errors:
            schema_violations.append(
                {
                    "method": record.get("method"),
                    "dataset": record.get("dataset"),
                    "sample_id": record.get("sample_id"),
                    "errors": schema_errors,
                }
            )

    actual_runtime = dict(runtime)
    actual_runtime["methods"] = method_configs
    if len(method_configs) == len(selected_methods):
        try:
            runtime_validation = validate_flashattn_runtime(
                actual_runtime,
                methods=selected_methods,
                require_dispatch_proof=True,
                allow_installed_vllm=SERVER_MODE,
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

    run_status = native_run_status(
        native_config, runtime_pass=runtime_validation.get("passed"),
        execution_complete=execution_complete, failure_count=failure_count,
        quality_pass=quality_pass, exact_match_all=exact_match_all,
        speedup_all_over_one=speedup_all_over_one, schema_valid=not schema_violations,
    )

    try:
        summary_metadata = runtime_metadata()
    except Exception:
        summary_metadata = {}
    summary_runtime = {**summary_metadata, **runtime, "methods": method_configs}
    summary = {
        "record_type": "summary",
        "scope": "summary",
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_runtime_seconds": round(time.perf_counter() - started_perf, 3),
        "method": "native_transformers_fa4_all_baselines",
        "execution_backend": "server" if SERVER_MODE else "modal",
        "status": run_status,
        "backend": "transformers_native",
        "attention_backend": ATTENTION,
        "runtime_validation": runtime_validation,
        "runtime_environment": summary_runtime,
        "versions": actual_versions,
        "models": {method: MODEL_REPOS[method] for method in selected_methods},
        "gpu": runtime.get("gpu_name"),
        "batch_size": 1,
        "dtype": "bfloat16",
        "seed": seed,
        "temperature": 0.0,
        "mode": mode,
        "datasets": list(selected_datasets),
        "methods": list(selected_methods),
        "max_new_tokens": max_new_tokens,
        "max_input_tokens": max_input_tokens,
        "warmup_tokens": warmup_tokens,
        "native_inference_config": native_config.manifest(),
        "native_source_sha256": runtime["native_source_sha256"],
        "warmup_per_selected_sample": True,
        "repetitions": repetitions,
        "sample_retries": sample_retries,
        "dflash_block_size": DFLASH_BLOCK_SIZE if "dflash" in selected_methods else None,
        "sample_count": len(samples),
        "sample_count_by_dataset": {
            dataset: sum(row["dataset"] == dataset for row in samples)
            for dataset in selected_datasets
        },
        "expected_sample_method_repeats": expected_cells,
        "selected_samples": runtime["selected_samples"],
        "excluded_samples": excluded_samples,
        "data_sha256": {
            dataset: next(
                (row.get("dataset_sha256") for row in samples if row["dataset"] == dataset),
                None,
            )
            for dataset in selected_datasets
        },
        "method_metrics": metric_bundle["method_metrics"],
        "metrics_by_dataset": metric_bundle["metrics_by_dataset"],
        "parity": metric_bundle["parity"],
        "failures": failures,
        "failure_count": failure_count,
        "schema_valid": not schema_violations,
        "schema_violations": schema_violations,
        "execution_complete": execution_complete,
        "execution_pass": execution_pass,
        "quality_pass": quality_pass,
        "quality_all_nonrepetitive": quality_pass,
        "correctness_pass": exact_match_all,
        "greedy_exact_match_all_speculative_methods": exact_match_all,
        "greedy_exact_match_by_method": exact_match_by_method,
        "latency_speedup_above_one_all_speculative": speedup_all_over_one,
        "speedup_esr_by_method": speedup_by_method,
        "run_passed": (
            run_status == "success"
            and execution_pass
            and quality_pass
            and not schema_violations
            and runtime_validation.get("passed") is True
        ),
        "metric_definitions": {
            "e2e_ms": "CUDA-synchronized client wall time for the full generation call",
            "prefill_ms": "CUDA-event first target forward proxy; inline mode retains native TTFT when available",
            "draft_latency_ms": "native synchronized phase timer from a separate generation with matching output/acceptance; inline mode uses the timed generation",
            "verification_latency_ms": "same profiling scope as draft_latency_ms; null if unsupported or profiling output/acceptance differs",
            "tpot_ms": "max(e2e_ms - prefill_ms, 0) / (output_tokens - 1)",
            "throughput_tok_s": "output_tokens / e2e_ms",
            "decode_throughput_tok_s": "(output_tokens - 1) / decode_ms",
            "dsr": "mean paired Vanilla TPOT / mean method TPOT",
            "esr": "(mean Vanilla prefill + Vanilla TPOT * mean paired minimum output length) / (mean Vanilla prefill + method TPOT * mean paired minimum output length)",
            "token_lcs_overlap_with_vanilla": "token-ID LCS divided by Vanilla output token count, weighted by reference token count",
            "acceptance_rate": "accepted speculative draft tokens / proposed speculative draft tokens",
            "quality_valid": "non-empty output, at least 4 generated tokens, and no repetition collapse flag",
        },
        "measurement_limitations": [
            "Native HF does not expose one common server-side request timeline; queue wait, batch wait and server-reported E2E are null.",
            "Separate/off modes use the first target forward CUDA-event duration as a common prefill proxy, not complete time-to-first-token. Inline mode keeps native TTFT when exposed.",
            "Separate profiling phase times come from an additional generation; they are not additive components of measured E2E. Native Domino/DSpark phase times are unavailable and remain null.",
            "Exact token parity and speedup above one are diagnostic by default; optional gates do not alter measured metrics. Output validity is not semantic equivalence.",
            "Timing is specific to the checkpoint, prompt, GPU allocation and installed package versions.",
        ],
    }

    partial_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, default=str) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    result_path = remote_run_dir / "results.jsonl"
    result_path.write_text("", encoding="utf-8")
    writer = JsonlWriter(result_path)
    for record in records:
        writer.add(record)
    writer.finalize(summary)
    write_json(remote_run_dir / "run_report.json", summary)
    write_json(
        remote_run_dir / "progress.json",
        {
            "run_id": run_id,
            "status": run_status,
            "records_written": len(records),
            "expected_records": expected_cells,
            "execution_complete": execution_complete,
            "execution_pass": execution_pass,
            "quality_pass": quality_pass,
            "correctness_pass": exact_match_all,
            "runtime_validation_pass": runtime_validation.get("passed"),
        },
    )
    (remote_run_dir / "report_vi.md").write_text(
        _render_report({"summary": summary, "records": records}),
        encoding="utf-8",
    )

    csv_path = remote_run_dir / "metrics_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "dataset", "method", "samples", "successful_samples", "failed_samples",
            "mean_rouge1", "mean_rouge2", "mean_rougeL", "mean_bleu4",
            "mean_e2e_ms", "median_e2e_ms", "p90_e2e_ms", "mean_tpot_ms",
            "mean_throughput_tok_s", "dsr", "esr", "mean_acceptance_rate_percent",
            "greedy_exact_matches", "greedy_compared_samples",
            "token_lcs_overlap_with_vanilla",
        ]
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        csv_writer.writeheader()
        csv_rows = []
        for dataset in selected_datasets:
            for method in selected_methods:
                metric = metric_bundle["metrics_by_dataset"][dataset][method]
                csv_rows.append((dataset, method, metric))
        for method in selected_methods:
            csv_rows.append(("all", method, metric_bundle["method_metrics"][method]))
        for dataset, method, metric in csv_rows:
            speed = metric.get("speed_statistics", {})
            paired = metric.get("paired_speed_metrics", {})
            semantic = metric.get("semantic_metrics", {})
            e2e = speed.get("e2e_ms", {})
            tpot = speed.get("tpot_ms", {})
            throughput_stats = speed.get("throughput_tok_s", {})
            csv_writer.writerow(
                {
                    "dataset": dataset,
                    "method": method,
                    "samples": metric["samples"],
                    "successful_samples": metric["successful_samples"],
                    "failed_samples": metric["failed_samples"],
                    "mean_rouge1": metric.get("mean_rouge1"),
                    "mean_rouge2": metric.get("mean_rouge2"),
                    "mean_rougeL": metric.get("mean_rougeL"),
                    "mean_bleu4": semantic.get("bleu4"),
                    "mean_e2e_ms": e2e.get("mean"),
                    "median_e2e_ms": e2e.get("median"),
                    "p90_e2e_ms": e2e.get("p90"),
                    "mean_tpot_ms": tpot.get("mean"),
                    "mean_throughput_tok_s": throughput_stats.get("mean"),
                    "dsr": paired.get("dsr"),
                    "esr": paired.get("esr"),
                    "mean_acceptance_rate_percent": metric.get("mean_acceptance_rate_percent"),
                    "greedy_exact_matches": metric.get("greedy_exact_matches"),
                    "greedy_compared_samples": metric.get("greedy_compared_samples"),
                    "token_lcs_overlap_with_vanilla": metric.get("token_lcs_overlap_with_vanilla"),
                }
            )

    append_jsonl(
        events_path,
        {
            "event": "run_finished",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": run_id,
            "status": run_status,
            "execution_complete": execution_complete,
            "failure_count": failure_count,
            "runtime_validation_pass": runtime_validation.get("passed"),
        },
    )
    persist_state()
    OUTPUT_VOLUME.commit()
    return {
        "run_id": run_id,
        "summary": summary,
        "remote_run_dir": str(remote_run_dir),
        "artifact_files": {
            name: f"{run_id}/{name}"
            for name in (
                "results.jsonl", "run_report.json", "report_vi.md", "metrics_summary.csv",
                "warmup.jsonl", "events.jsonl", "samples.jsonl", "excluded_samples.jsonl",
                "progress.json", "state.json", "results.partial.jsonl",
            )
        },
    }

def _render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]

    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def render_row(method: str, metric: dict[str, Any]) -> str:
        speed = metric.get("speed_statistics", {})
        semantic = metric.get("semantic_metrics", {})
        paired = metric.get("paired_speed_metrics", {})
        e2e = speed.get("e2e_ms", {})
        prefill = speed.get("prefill_ms", {})
        tpot = speed.get("tpot_ms", {})
        throughput = speed.get("throughput_tok_s", {})
        return (
            f"| {method} | {metric.get('successful_samples', 0)}/{metric.get('samples', 0)} "
            f"| {fmt(metric.get('mean_rougeL'))} | {fmt(semantic.get('bleu4'))} "
            f"| {fmt(e2e.get('mean'))} | {fmt(e2e.get('p90'))} "
            f"| {fmt(prefill.get('mean'))} | {fmt(tpot.get('mean'))} "
            f"| {fmt(throughput.get('mean'))} | {fmt(paired.get('dsr'))} "
            f"| {fmt(paired.get('esr'))} | {fmt(metric.get('mean_acceptance_rate_percent'), 2)} "
            f"| {metric.get('greedy_exact_matches', 0)}/{metric.get('greedy_compared_samples', 0)} "
            f"| {fmt(metric.get('token_lcs_overlap_with_vanilla'), 4)} "
            f"| {metric.get('quality_valid_outputs', 0)}/{metric.get('successful_samples', 0)} |"
        )

    lines = [
        "# Benchmark batch-1 native FlashAttention-4",
        "",
        f"- Run ID: {summary.get('run_id')}; trạng thái: {summary.get('status')}",
        f"- Chế độ: {summary.get('mode')}; GPU: {summary.get('gpu')}; attention: {summary.get('attention_backend')}; batch size: 1",
        f"- Dataset: {', '.join(summary.get('datasets', []))}; số mẫu: {summary.get('sample_count')} ({summary.get('sample_count_by_dataset', {})})",
        f"- Methods: {', '.join(summary.get('methods', []))}; tối đa {summary.get('max_new_tokens')} token đầu ra; giới hạn input {summary.get('max_input_tokens')} token",
        f"- Runtime FA4/no-fallback: {summary.get('runtime_validation', {}).get('passed')}; greedy parity: {summary.get('correctness_pass')}; quality gate: {summary.get('quality_pass')}",
        f"- Cấu hình native: {summary.get('native_inference_config', {})}",
        f"- Thời gian: {fmt(summary.get('evaluation_runtime_seconds'))} giây; repeats: {summary.get('repetitions')}; seed: {summary.get('seed')}",
        "",
        "| Method | Thành công | ROUGE-L | BLEU-4 | E2E TB (ms) | E2E p90 (ms) | Prefill TB (ms) | TPOT TB (ms/token) | Tok/s TB | DSR | ESR | Accept (%) | Greedy exact | LCS overlap | Output hợp lệ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in summary.get("methods", []):
        lines.append(render_row(method, summary.get("method_metrics", {}).get(method, {})))

    for dataset in summary.get("datasets", []):
        lines.extend(
            [
                "",
                f"## Dataset: {dataset}",
                "",
                "| Method | Thành công | ROUGE-L | BLEU-4 | E2E TB (ms) | E2E p90 (ms) | Prefill TB (ms) | TPOT TB | Tok/s TB | DSR | ESR | Accept (%) | Greedy exact | LCS | Output hợp lệ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in summary.get("methods", []):
            metric = summary.get("metrics_by_dataset", {}).get(dataset, {}).get(method, {})
            lines.append(render_row(method, metric))

    lines.extend(["", "## Kiểm chứng runtime", ""])
    for method, dispatch in summary.get("runtime_validation", {}).get("methods", {}).items():
        lines.append(
            f"- {method}: target FA4={dispatch.get('target_attention_dispatch_calls')}, "
            f"target fallback={dispatch.get('target_fallback_attention_calls')}; "
            f"draft FA4={dispatch.get('draft_attention_dispatch_calls')}, "
            f"draft fallback={dispatch.get('draft_fallback_attention_calls')}."
        )
    lines.extend(
        [
            "",
            "## Công thức và phạm vi đo",
            "",
            "- DSR/ESR ghép cùng sample và repeat; ESR chuẩn hóa theo độ dài output ngắn hơn. Speedup được báo theo phép đo, không ép phải lớn hơn 1.",
            "- TPOT = (E2E - prefill) / (output_tokens - 1). Chế độ separate/off dùng CUDA-event của target forward đầu làm prefill proxy chung; inline giữ native TTFT nếu có. Proxy không bao gồm toàn bộ thời gian tới token đầu tiên.",
            "- Chế độ separate thu thập draft/verify time ở lượt profiling riêng, chỉ ghép khi output và acceptance ledger khớp lượt đo. Các pha này không cộng thành E2E của lượt đo; pha không được native implementation hỗ trợ để null.",
            "- Throughput = output_tokens / E2E; decode throughput = (output_tokens - 1) / decode time. Queue wait, batch wait, server startup và server E2E là null do chạy native Transformers không có request server.",
            "- Quality hợp lệ khi output không rỗng, có ít nhất 4 token và không có cờ repetition collapse. JSONL lưu ROUGE, ROUGE-Lsum, BLEU, length ratio, token IDs, acceptance counters và latency đầy đủ.",
            "- Exact greedy và speedup > 1 là chẩn đoán mặc định; chỉ chặn run khi bật strict_greedy_parity/require_speedup. Quality guard không chứng minh nội dung tương đương; cần đối chiếu ROUGE/BLEU và output.",
            "",
            "Runtime gate yêu cầu tất cả target/draft attention dispatch qua FA4, không fallback. Runner dùng Transformers native, không nạp vLLM.",
        ]
    )
    if summary.get("failures"):
        lines.extend(["", "## Lỗi method", ""])
        for method, error in summary["failures"].items():
            lines.append(f"- {method}: {error}")
    bad_rows = [row for row in result.get("records", []) if row.get("status") != "success"]
    if bad_rows:
        lines.extend(["", "## Mẫu lỗi", ""])
        for row in bad_rows[:30]:
            lines.append(
                f"- {row.get('method')} / {row.get('sample_id')} repeat "
                f"{row.get('repeat_index', 0)}: {row.get('reason')}"
            )
        if len(bad_rows) > 30:
            lines.append(f"- Còn {len(bad_rows) - 30} lỗi; xem results.jsonl.")
    return "\n".join(lines) + "\n"


def _download_volume_artifacts(
    volume,
    *,
    run_id: str,
    artifact_files: dict[str, str],
    local_run_dir: Path,
) -> None:
    """Download committed Modal Volume files from the local entrypoint."""
    local_run_dir.mkdir(parents=True, exist_ok=True)
    for filename, volume_path in artifact_files.items():
        local_path = local_run_dir / filename
        try:
            with local_path.open("wb") as handle:
                for chunk in volume.read_file(volume_path):
                    handle.write(chunk)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Modal Volume artifact is missing: {run_id}/{filename}"
            ) from exc


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    datasets: str = "all",
    samples_per_dataset: int | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    methods: str = "all",
    warmup_tokens: int = 512,
    eagle_total_token: int = 17,
    eagle_depth: int = 16,
    eagle_top_k: int = 1,
    domino_cuda_graph: bool = True,
    dspark_confidence_threshold: float = 0.0,
    phase_timing_mode: str = "separate",
    strict_greedy_parity: bool = False,
    require_speedup: bool = False,
    repetitions: int = 1,
    seed: int = 42,
    sample_retries: int = 1,
    checkpoint_interval: int = 20,
    run_id: str = "",
    resume: bool = False,
    direct_target_audit: bool = False,
    verifier_audit: bool = False,
    preflight_only: bool = False,
    download_only: bool = False,
    debug_cuda_launch_blocking: bool = False,
    output_dir: str = "outputs/modal_flashattn_benchmark",
) -> None:
    if resume and not run_id:
        raise SystemExit("--resume requires --run-id")
    if download_only and preflight_only:
        raise SystemExit("--download-only cannot be combined with --preflight-only")
    if download_only:
        if not run_id:
            raise SystemExit("--download-only requires --run-id")
        result = {
            "run_id": run_id,
            "artifact_files": {
                name: f"{run_id}/{name}" for name in ARTIFACT_FILENAMES
            },
        }
    else:
        result = run_flashattn_benchmark.remote(
            mode=mode,
            datasets=datasets,
            samples_per_dataset=samples_per_dataset,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            methods=methods,
            warmup_tokens=warmup_tokens,
            eagle_total_token=eagle_total_token,
            eagle_depth=eagle_depth,
            eagle_top_k=eagle_top_k,
            domino_cuda_graph=domino_cuda_graph,
            dspark_confidence_threshold=dspark_confidence_threshold,
            phase_timing_mode=phase_timing_mode,
            strict_greedy_parity=strict_greedy_parity,
            require_speedup=require_speedup,
            repetitions=repetitions,
            seed=seed,
            sample_retries=sample_retries,
            checkpoint_interval=checkpoint_interval,
            run_id=run_id,
            resume=resume,
            direct_target_audit=direct_target_audit,
            verifier_audit=verifier_audit,
            preflight_only=preflight_only,
            debug_cuda_launch_blocking=debug_cuda_launch_blocking,
        )
        if preflight_only:
            print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
            return

    local_run_dir = (PROJECT_ROOT / output_dir / result["run_id"]).resolve()
    _download_volume_artifacts(
        OUTPUT_VOLUME,
        run_id=result["run_id"],
        artifact_files=result["artifact_files"],
        local_run_dir=local_run_dir,
    )
    print((local_run_dir / "report_vi.md").read_text(encoding="utf-8"))
    print(f"Artifacts đã tải về: {local_run_dir}")
    if not download_only and result["summary"].get("run_passed") is not True:
        raise SystemExit(
            f"benchmark gates failed; inspect {local_run_dir / 'report_vi.md'}"
        )
