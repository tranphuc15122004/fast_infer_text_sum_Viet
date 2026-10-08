#!/usr/bin/env python3
"""Smoke-test the Qwen3 DFlash checkpoint using its Hugging Face model-card API."""

from __future__ import annotations

import json
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

app = modal.App("fast-infer-viet-dflash-model-card-smoke")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "torch==2.9.1",
        extra_options=(
            "--index-url https://download.pytorch.org/whl/cu130 "
            "--extra-index-url https://pypi.org/simple"
        ),
    )
    .pip_install(
        "transformers==4.57.3",
        "accelerate==1.15.0",
        # The checkpoint's remote-code module imports this even for inference.
        "datasets==3.6.0",
        "huggingface_hub==0.36.0",
        "tokenizers==0.22.1",
        "safetensors==0.8.0",
        "sentencepiece==0.2.2",
        "numpy==2.3.5",
        "tqdm==4.70.1",
    )
    .add_local_dir(str(ROOT / "src"), remote_path=str(REMOTE_SRC), copy=True)
    .add_local_file(
        str(ROOT / "datasets/eval_100/vietnews_100.jsonl"),
        remote_path=str(REMOTE_DATA),
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_SRC),
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


@app.function(gpu="L4", cpu=8, memory=32768, timeout=3600, image=image)
def run_model_card_recipe(max_new_tokens: int = 512, sample_count: int = 2) -> dict:
    import torch
    import transformers
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from Benchmark.common.benchmark_data import render_prompt
    from Benchmark.common.prompt_format import format_chat_prompt
    from Benchmark.common.quality_guard import repetition_metrics
    from Benchmark.common.rouge import rouge_scores

    if not torch.cuda.is_available():
        raise RuntimeError("Modal function did not receive a CUDA GPU")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    gpu_name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))

    api = HfApi()
    target_revision = api.model_info(TARGET_REPO).sha
    draft_revision = api.model_info(DRAFT_REPO).sha
    target_path = snapshot_download(TARGET_REPO, revision=target_revision)
    draft_path = snapshot_download(DRAFT_REPO, revision=draft_revision)

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_start = time.perf_counter()
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        dtype="auto",
        device_map="cuda:0",
    ).eval()
    torch.cuda.synchronize()
    target_load_ms = (time.perf_counter() - target_start) * 1000

    rows = []
    with open(REMOTE_DATA, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == sample_count:
                break
    if len(rows) != sample_count:
        raise RuntimeError(f"Requested {sample_count} samples, found {len(rows)}")

    stop_id = tokenizer.eos_token_id
    stop_token_ids = [stop_id] if stop_id is not None else []

    def encode(row):
        prompt = format_chat_prompt(tokenizer, render_prompt(row))
        return tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)

    def run_vanilla(input_ids, limit):
        return target.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=limit,
            use_cache=True,
            eos_token_id=stop_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Warm the target path before collecting its per-sample latency.
    warm_input = encode(rows[0])
    run_vanilla(warm_input, min(max_new_tokens, 8))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    def measure(invoke, input_ids):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = invoke()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
        new_ids = output[0, input_ids.shape[1]:].detach().cpu()
        return new_ids, elapsed_ms, peak_gib

    vanilla = []
    for row in rows:
        input_ids = encode(row)
        output_ids, elapsed_ms, peak_gib = measure(
            lambda: run_vanilla(input_ids, max_new_tokens), input_ids
        )
        text = tokenizer.decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        vanilla.append(
            {
                "sample_id": row["id"],
                "input_ids": input_ids,
                "input_tokens": int(input_ids.shape[1]),
                "reference": row["reference"],
                "output_ids": output_ids,
                "output_tokens": int(output_ids.numel()),
                "e2e_ms": elapsed_ms,
                "tokens_per_second": output_ids.numel() / (elapsed_ms / 1000),
                "peak_allocated_gib": peak_gib,
                "text": text,
            }
        )

    draft_start = time.perf_counter()
    draft = AutoModel.from_pretrained(
        draft_path,
        trust_remote_code=True,
        dtype="auto",
        device_map="cuda:0",
    ).eval()
    torch.cuda.synchronize()
    draft_load_ms = (time.perf_counter() - draft_start) * 1000

    # The checkpoint's config supplies block_size=16; do not override it.
    block_size = int(draft.block_size)
    if block_size != 16:
        raise RuntimeError(f"Checkpoint block_size is {block_size}, expected 16")

    def run_spec(input_ids, limit):
        return draft.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=limit,
            temperature=0.0,
            stop_token_ids=stop_token_ids,
        )

    run_spec(vanilla[0]["input_ids"], min(max_new_tokens, 8))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    sample_results = []
    for baseline in vanilla:
        input_ids = baseline["input_ids"]
        output_ids, elapsed_ms, peak_gib = measure(
            lambda: run_spec(input_ids, max_new_tokens), input_ids
        )
        text = tokenizer.decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        ref = str(baseline["reference"])
        vanilla_text = baseline["text"]
        sample_results.append(
            {
                "sample_id": baseline["sample_id"],
                "input_tokens": baseline["input_tokens"],
                "reference": ref,
                "vanilla_hf": {
                    "output_tokens": baseline["output_tokens"],
                    "e2e_ms": round(baseline["e2e_ms"], 3),
                    "tokens_per_second": round(baseline["tokens_per_second"], 3),
                    "peak_allocated_gib": round(baseline["peak_allocated_gib"], 3),
                    "rouge": rouge_scores(vanilla_text, ref),
                    "repetition": repetition_metrics(vanilla_text),
                    "text": vanilla_text,
                },
                "dflash_model_card_spec_generate": {
                    "block_size": block_size,
                    "output_tokens": int(output_ids.numel()),
                    "e2e_ms": round(elapsed_ms, 3),
                    "tokens_per_second": round(output_ids.numel() / (elapsed_ms / 1000), 3),
                    "speedup_vs_vanilla": round(baseline["e2e_ms"] / elapsed_ms, 3),
                    "peak_allocated_gib": round(peak_gib, 3),
                    "rouge": rouge_scores(text, ref),
                    "repetition": repetition_metrics(text),
                    "greedy_token_ids_match_vanilla": bool(torch.equal(output_ids, baseline["output_ids"])),
                    "text": text,
                },
            }
        )
        del input_ids, output_ids
        torch.cuda.empty_cache()

    def mean(path):
        values = []
        for sample in sample_results:
            value = sample
            for key in path:
                value = value[key]
            values.append(float(value))
        return round(statistics.mean(values), 4)

    draft_repo_path = Path(draft_path)
    remote_code_hashes = {}
    for name in ("dflash.py", "modeling_dflash.py", "utils.py"):
        file_path = draft_repo_path / name
        if file_path.exists():
            import hashlib
            remote_code_hashes[name] = hashlib.sha256(file_path.read_bytes()).hexdigest()

    return {
        "status": "success",
        "mode": "two_sample_modal_l4_dflash_model_card_recipe",
        "samples": len(sample_results),
        "dataset": "vietnews",
        "sample_selection": "first two rows in datasets/eval_100/vietnews_100.jsonl",
        "target_model": TARGET_REPO,
        "target_revision": target_revision,
        "draft_model": DRAFT_REPO,
        "draft_revision": draft_revision,
        "remote_code_sha256": remote_code_hashes,
        "configuration": {
            "gpu": gpu_name,
            "compute_capability": capability,
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "accelerate": __import__("accelerate").__version__,
            "huggingface_hub": __import__("huggingface_hub").__version__,
            "target_dtype": str(target.dtype),
            "draft_dtype": str(next(draft.parameters()).dtype),
            "target_attention": getattr(target.config, "_attn_implementation", None),
            "draft_attention": getattr(draft.config, "_attn_implementation", None),
            "batch_size": 1,
            "dflash_block_size": block_size,
            "max_new_tokens": max_new_tokens,
            "sampling": "greedy (temperature=0)",
            "target_load_ms": round(target_load_ms, 3),
            "draft_load_ms": round(draft_load_ms, 3),
            "timing_scope": "synchronized generation call; excludes model load and tokenization",
        },
        "mean_metrics": {
            "vanilla_e2e_ms": mean(("vanilla_hf", "e2e_ms")),
            "vanilla_tokens_per_second": mean(("vanilla_hf", "tokens_per_second")),
            "vanilla_rougeL": mean(("vanilla_hf", "rouge", "rougeL")),
            "dflash_e2e_ms": mean(("dflash_model_card_spec_generate", "e2e_ms")),
            "dflash_tokens_per_second": mean(("dflash_model_card_spec_generate", "tokens_per_second")),
            "dflash_speedup_vs_vanilla": mean(("dflash_model_card_spec_generate", "speedup_vs_vanilla")),
            "dflash_rougeL": mean(("dflash_model_card_spec_generate", "rouge", "rougeL")),
            "dflash_peak_allocated_gib": mean(("dflash_model_card_spec_generate", "peak_allocated_gib")),
        },
        "sample_results": sample_results,
    }


@app.local_entrypoint()
def main(max_new_tokens: int = 512, sample_count: int = 2):
    result = run_model_card_recipe.remote(
        max_new_tokens=max_new_tokens,
        sample_count=sample_count,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "outputs/modal_dflash_modelcard_smoke" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics = result["mean_metrics"]
    lines = [
        "# DFlash model-card smoke benchmark trên Modal L4",
        "",
        f"- Trạng thái: {result['status']}",
        f"- Runtime: PyTorch {result['configuration']['torch']}, Transformers {result['configuration']['transformers']}, dtype draft {result['configuration']['draft_dtype']}",
        f"- Batch size 1; block size {result['configuration']['dflash_block_size']}; max_new_tokens={max_new_tokens}; SDPA/attention theo cấu hình model",
        f"- ROUGE-L trung bình: Vanilla {metrics['vanilla_rougeL']:.4f}; DFlash {metrics['dflash_rougeL']:.4f}",
        f"- Speedup trung bình theo mẫu: {metrics['dflash_speedup_vs_vanilla']:.3f}x",
        f"- Peak allocated VRAM DFlash: {metrics['dflash_peak_allocated_gib']:.2f} GiB",
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
                f"DFlash model-card API (ROUGE-L {sample['dflash_model_card_spec_generate']['rouge']['rougeL']:.4f}, {sample['dflash_model_card_spec_generate']['tokens_per_second']:.2f} tok/s): {sample['dflash_model_card_spec_generate']['text']}",
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nArtifacts: {output_dir}")
