#!/usr/bin/env python3
"""Run a two-sample Qwen3-4B/DFlash smoke benchmark on Modal T4."""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum_Viet")
REMOTE_SRC = REMOTE_ROOT / "src"
REMOTE_DATA = REMOTE_ROOT / "datasets/eval_100/vietnews_100.jsonl"
TARGET_REPO = "Qwen/Qwen3-4B"
DRAFT_REPO = "z-lab/Qwen3-4B-DFlash-b16"
GPU_TYPE = os.environ.get("DFLASH_SMOKE_GPU", "T4")

app = modal.App("fast-infer-viet-dflash-t4-smoke")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "torch==2.13.0",
        extra_options=(
            "--index-url https://download.pytorch.org/whl/cu130 "
            "--extra-index-url https://pypi.org/simple"
        ),
    )
    .pip_install(
        "transformers==5.15.0",
        "accelerate==1.15.0",
        "huggingface_hub==1.31.0",
        "safetensors==0.8.0",
        "sentencepiece==0.2.2",
        "tqdm==4.70.1",
        "Jinja2==3.1.6",
    )
    .add_local_dir(
        str(ROOT / "src"), remote_path=str(REMOTE_SRC), copy=True
    )
    .add_local_dir(
        str(ROOT / "externals/dflash"),
        remote_path=str(REMOTE_ROOT / "externals/dflash"),
        copy=True,
    )
    .add_local_file(
        str(ROOT / "datasets/eval_100/vietnews_100.jsonl"),
        remote_path=str(REMOTE_DATA),
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_SRC}:{REMOTE_ROOT / 'externals/dflash'}",
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


@app.function(gpu=GPU_TYPE, cpu=8, memory=32768, timeout=3600, image=image)
def run_two_samples(max_new_tokens: int = 128, sample_count: int = 2) -> dict:
    """Benchmark native HF and the vendored upstream DFlash generator."""

    import hashlib
    import sys

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(REMOTE_SRC))
    sys.path.insert(0, str(REMOTE_ROOT / "externals/dflash"))
    from Benchmark.common.benchmark_data import render_prompt
    from Benchmark.common.prompt_format import format_chat_prompt
    from Benchmark.common.quality_guard import repetition_metrics
    from Benchmark.common.rouge import rouge_scores
    from Benchmark.dflash_compat import (
        install_dflash_cache_crop_compat,
        install_dflash_transformers_compat,
    )

    install_dflash_transformers_compat()
    import dflash.model as dflash_model

    install_dflash_cache_crop_compat(dflash_model)
    dflash_generate = dflash_model.dflash_generate
    DFlashDraftModel = dflash_model.DFlashDraftModel

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if not torch.cuda.is_available():
        raise RuntimeError("Modal function did not receive a CUDA GPU")

    gpu_name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    dtype = torch.float16 if capability[0] < 8 else torch.bfloat16

    api = HfApi()
    target_revision = api.model_info(TARGET_REPO).sha
    draft_revision = api.model_info(DRAFT_REPO).sha
    target_path = snapshot_download(TARGET_REPO, revision=target_revision)
    draft_path = snapshot_download(DRAFT_REPO, revision=draft_revision)

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_load_start = time.perf_counter()
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    draft = DFlashDraftModel.from_pretrained(
        draft_path,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - target_load_start) * 1000

    rows = []
    with open(REMOTE_DATA, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == sample_count:
                break
    if len(rows) != sample_count:
        raise RuntimeError(f"Requested {sample_count} samples, found {len(rows)}")

    stop_id = target.config.eos_token_id
    stop_token_ids = stop_id if isinstance(stop_id, list) else [stop_id]
    if stop_token_ids == [None] and tokenizer.eos_token_id is not None:
        stop_token_ids = [tokenizer.eos_token_id]

    def encode(row):
        prompt = format_chat_prompt(tokenizer, render_prompt(row))
        ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to("cuda:0")
        return ids

    def run_vanilla(input_ids, limit):
        return target.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=limit,
            use_cache=True,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    def run_dflash(input_ids, limit, block_size, return_stats=False):
        return dflash_generate(
            draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=limit,
            stop_token_ids=stop_token_ids,
            temperature=0.0,
            block_size=block_size,
            return_stats=return_stats,
        )

    # Warm each path once; exclude compilation and initial CUDA setup from timings.
    warm_input = encode(rows[0])
    warm_limit = min(max_new_tokens, 8)
    for invoke in (
        lambda: run_vanilla(warm_input, warm_limit),
        lambda: run_dflash(warm_input, warm_limit, 1),
        lambda: run_dflash(warm_input, warm_limit, int(draft.block_size)),
    ):
        invoke()
        torch.cuda.synchronize()
        del invoke
    del warm_input
    torch.cuda.empty_cache()

    def measure(invoke, input_ids):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = invoke()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024**3)
        output_ids = output[0, input_ids.shape[1]:]
        return output_ids.detach().cpu(), elapsed_ms, peak_allocated_gib

    sample_results = []
    for row in rows:
        input_ids = encode(row)
        input_len = int(input_ids.shape[1])

        vanilla_ids, vanilla_ms, vanilla_peak = measure(
            lambda: run_vanilla(input_ids, max_new_tokens), input_ids
        )
        dflash1_ids, dflash1_ms, dflash1_peak = measure(
            lambda: run_dflash(input_ids, max_new_tokens, 1), input_ids
        )
        block_size = int(draft.block_size)
        dflash_ids, dflash_ms, dflash_peak = measure(
            lambda: run_dflash(input_ids, max_new_tokens, block_size), input_ids
        )

        # Collect native acceptance counters separately so their per-phase CUDA
        # synchronizations cannot inflate the speed timing above.
        stats = run_dflash(
            input_ids,
            max_new_tokens,
            block_size,
            return_stats=True,
        )
        torch.cuda.synchronize()
        stats_ids = stats.output_ids[0, input_len:].detach().cpu()
        if not torch.equal(stats_ids, dflash_ids):
            raise RuntimeError("DFlash timed and instrumented greedy outputs differ")

        vanilla_text = tokenizer.decode(
            vanilla_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        dflash1_text = tokenizer.decode(
            dflash1_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        dflash_text = tokenizer.decode(
            dflash_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        reference = str(row["reference"])
        proposed = int(stats.draft_tokens_proposed)
        accepted = int(stats.draft_tokens_accepted)
        sample_results.append(
            {
                "sample_id": row["id"],
                "input_tokens": input_len,
                "reference": reference,
                "vanilla_hf": {
                    "output_tokens": int(vanilla_ids.numel()),
                    "e2e_ms": round(vanilla_ms, 3),
                    "tokens_per_second": round(vanilla_ids.numel() / (vanilla_ms / 1000), 3),
                    "peak_allocated_gib": round(vanilla_peak, 3),
                    "rouge": rouge_scores(vanilla_text, reference),
                    "repetition": repetition_metrics(vanilla_text),
                    "text": vanilla_text,
                },
                "dflash_block_size_1_control": {
                    "output_tokens": int(dflash1_ids.numel()),
                    "e2e_ms": round(dflash1_ms, 3),
                    "tokens_per_second": round(dflash1_ids.numel() / (dflash1_ms / 1000), 3),
                    "speedup_vs_vanilla": round(vanilla_ms / dflash1_ms, 3),
                    "peak_allocated_gib": round(dflash1_peak, 3),
                    "rouge": rouge_scores(dflash1_text, reference),
                    "text": dflash1_text,
                },
                "dflash_block_size_16": {
                    "output_tokens": int(dflash_ids.numel()),
                    "e2e_ms": round(dflash_ms, 3),
                    "tokens_per_second": round(dflash_ids.numel() / (dflash_ms / 1000), 3),
                    "speedup_vs_vanilla": round(vanilla_ms / dflash_ms, 3),
                    "speedup_vs_dflash_block_size_1": round(dflash1_ms / dflash_ms, 3),
                    "peak_allocated_gib": round(dflash_peak, 3),
                    "rouge": rouge_scores(dflash_text, reference),
                    "repetition": repetition_metrics(dflash_text),
                    "greedy_token_ids_match_vanilla": bool(torch.equal(dflash_ids, vanilla_ids)),
                    "draft_tokens_accepted": accepted,
                    "draft_tokens_proposed": proposed,
                    "acceptance_rate_percent": round(100 * accepted / proposed, 2) if proposed else None,
                    "mean_accepted_draft_tokens_per_step": round(
                        sum(max(int(length) - 1, 0) for length in stats.acceptance_lengths)
                        / max(len(stats.acceptance_lengths), 1),
                        3,
                    ),
                    "text": dflash_text,
                },
            }
        )
        del input_ids, vanilla_ids, dflash1_ids, dflash_ids, stats, stats_ids
        torch.cuda.empty_cache()

    def mean(path):
        values = []
        for sample in sample_results:
            value = sample
            for key in path:
                value = value[key]
            values.append(float(value))
        return round(statistics.mean(values), 4)

    source_path = REMOTE_ROOT / "externals/dflash/dflash/model.py"
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return {
        "status": "success",
        "mode": f"two_sample_modal_{gpu_name.lower().replace(' ', '_')}_dflash_smoke",
        "samples": len(sample_results),
        "dataset": "vietnews",
        "sample_selection": "first two rows in datasets/eval_100/vietnews_100.jsonl",
        "target_model": TARGET_REPO,
        "target_revision": target_revision,
        "draft_model": DRAFT_REPO,
        "draft_revision": draft_revision,
        "dflash_source_sha256": source_sha256,
        "configuration": {
            "gpu": gpu_name,
            "compute_capability": capability,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "dtype": str(dtype),
            "attention_backend": "sdpa",
            "batch_size": 1,
            "dflash_block_size": int(draft.block_size),
            "max_new_tokens": max_new_tokens,
            "sampling": "greedy",
            "model_load_ms": round(model_load_ms, 3),
            "timing_scope": "synchronized generation call; excludes model load and tokenization",
        },
        "mean_metrics": {
            "vanilla_e2e_ms": mean(("vanilla_hf", "e2e_ms")),
            "vanilla_tokens_per_second": mean(("vanilla_hf", "tokens_per_second")),
            "vanilla_rougeL": mean(("vanilla_hf", "rouge", "rougeL")),
            "dflash16_e2e_ms": mean(("dflash_block_size_16", "e2e_ms")),
            "dflash16_tokens_per_second": mean(("dflash_block_size_16", "tokens_per_second")),
            "dflash16_speedup_vs_vanilla": mean(("dflash_block_size_16", "speedup_vs_vanilla")),
            "dflash16_rougeL": mean(("dflash_block_size_16", "rouge", "rougeL")),
            "dflash16_acceptance_rate_percent": mean(("dflash_block_size_16", "acceptance_rate_percent")),
            "dflash16_peak_allocated_gib": mean(("dflash_block_size_16", "peak_allocated_gib")),
        },
        "sample_results": sample_results,
    }


@app.local_entrypoint()
def main(max_new_tokens: int = 128, sample_count: int = 2):
    result = run_two_samples.remote(
        max_new_tokens=max_new_tokens,
        sample_count=sample_count,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "outputs/modal_dflash_smoke" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# DFlash smoke benchmark trên Modal {result['configuration']['gpu']}",
        "",
        f"- Trạng thái: {result['status']}",
        f"- GPU: {result['configuration']['gpu']}; dtype: {result['configuration']['dtype']}; attention: SDPA",
        f"- Target/draft: {result['target_model']} + {result['draft_model']} (block size {result['configuration']['dflash_block_size']})",
        f"- Mẫu: {result['samples']} VietNews; batch size 1; max_new_tokens={max_new_tokens}",
        f"- Trung bình ROUGE-L: Vanilla {result['mean_metrics']['vanilla_rougeL']:.4f}, DFlash {result['mean_metrics']['dflash16_rougeL']:.4f}",
        f"- Speedup DFlash-16 / Vanilla: {result['mean_metrics']['dflash16_speedup_vs_vanilla']:.3f}x",
        f"- Acceptance: {result['mean_metrics']['dflash16_acceptance_rate_percent']:.2f}%; peak VRAM allocated {result['mean_metrics']['dflash16_peak_allocated_gib']:.2f} GiB",
        "",
    ]
    for sample in result["sample_results"]:
        lines.extend(
            [
                f"## {sample['sample_id']}",
                "",
                f"Reference: {sample['reference']}",
                "",
                f"Vanilla (ROUGE-L {sample['vanilla_hf']['rouge']['rougeL']:.4f}, {sample['vanilla_hf']['tokens_per_second']:.2f} tok/s): {sample['vanilla_hf']['text']}",
                "",
                f"DFlash-16 (ROUGE-L {sample['dflash_block_size_16']['rouge']['rougeL']:.4f}, {sample['dflash_block_size_16']['tokens_per_second']:.2f} tok/s): {sample['dflash_block_size_16']['text']}",
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nArtifacts: {output_dir}")
