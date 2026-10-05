#!/usr/bin/env python3
"""Run benchmark evaluation for trained DFlash draft model against Vietnamese test sets using vLLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from tqdm import tqdm

# Ensure benchmark modules can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Benchmark.common.benchmark_data import read_jsonl, render_prompt
from Benchmark.common.io_util import JsonlWriter
from Benchmark.common.rouge import add_rouge, aggregate_rouge
from Benchmark.common.prompt_format import format_chat_prompt


DATASET_FILES = {
    "vietnews": "vietnews_100.jsonl",
    "wikilingua": "wikilingua_100.jsonl",
    "vims": "vims_100.jsonl",
    "vlsp": "vlsp_100.jsonl",
}


def prepare_prompts(
    raw_samples: list[dict[str, Any]],
    ds_name: str,
    tokenizer: Any = None,
) -> list[str]:
    prompts = []
    for r in raw_samples:
        row = dict(r)
        if not row.get("dataset"):
            row["dataset"] = ds_name
        p = render_prompt(row)
        if tokenizer is not None:
            try:
                p = format_chat_prompt(tokenizer, p)
            except Exception:
                pass
        prompts.append(p)
    return prompts


def load_dataset_samples(data_dir: Path, dataset_name: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    file_name = DATASET_FILES.get(dataset_name, f"{dataset_name}_100.jsonl")
    data_path = data_dir / file_name
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    rows = read_jsonl(data_path)
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    return rows


def run_vllm_inference(
    llm: Any,
    prompts: list[str],
    references: list[str],
    sample_ids: list[str],
    *,
    sampling_params: Any,
    method_name: str,
    dataset_name: str,
    writer: JsonlWriter | None = None,
) -> dict[str, Any]:
    print(f"\n>>> Running vLLM inference: [{method_name}] on [{dataset_name}] ({len(prompts)} samples)...")
    
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    total_duration = time.perf_counter() - start_time
    
    total_prompt_tokens = 0
    total_output_tokens = 0
    records = []
    
    total_draft_accepted = 0
    total_draft_proposed = 0
    
    for i, out in enumerate(outputs):
        prompt_text = prompts[i]
        ref_text = references[i] if i < len(references) else ""
        sample_id = sample_ids[i] if i < len(sample_ids) else str(i)
        
        gen_text = out.outputs[0].text
        p_tokens = len(out.prompt_token_ids)
        o_tokens = len(out.outputs[0].token_ids)
        total_prompt_tokens += p_tokens
        total_output_tokens += o_tokens
        
        record = {
            "dataset": dataset_name,
            "method": method_name,
            "sample_id": sample_id,
            "prompt_tokens": p_tokens,
            "output_tokens": o_tokens,
            "generated_text": gen_text,
            "reference": ref_text,
        }
        
        # Extract speculative metrics if present
        metrics = getattr(out.outputs[0], "metrics", None)
        if metrics is not None:
            accepted = getattr(metrics, "num_accepted_tokens", getattr(metrics, "draft_tokens_accepted", None))
            proposed = getattr(metrics, "num_proposed_tokens", getattr(metrics, "draft_tokens_proposed", None))
            if accepted is not None:
                total_draft_accepted += int(accepted)
                record["draft_accepted_tokens"] = int(accepted)
            if proposed is not None:
                total_draft_proposed += int(proposed)
                record["draft_proposed_tokens"] = int(proposed)
        
        if ref_text:
            record = add_rouge(record, generated=gen_text, reference=ref_text)
            
        if writer is not None:
            writer.write(record)
        records.append(record)

    throughput = total_output_tokens / max(total_duration, 1e-6)
    
    # Calculate ROUGE aggregate
    rouge_summary = aggregate_rouge(records) if any("rouge1" in r for r in records) else {}
    
    acceptance_rate = (total_draft_accepted / max(total_draft_proposed, 1) * 100.0) if total_draft_proposed > 0 else None
    
    summary = {
        "dataset": dataset_name,
        "method": method_name,
        "num_samples": len(prompts),
        "total_output_tokens": total_output_tokens,
        "duration_s": round(total_duration, 2),
        "throughput_tok_s": round(throughput, 2),
        "acceptance_rate_pct": round(acceptance_rate, 2) if acceptance_rate is not None else None,
        "rouge1": round(rouge_summary.get("rouge1", 0.0) * 100, 2) if "rouge1" in rouge_summary else None,
        "rouge2": round(rouge_summary.get("rouge2", 0.0) * 100, 2) if "rouge2" in rouge_summary else None,
        "rougeL": round(rouge_summary.get("rougeL", 0.0) * 100, 2) if "rougeL" in rouge_summary else None,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DFlash with vLLM on VietBench")
    parser.add_argument("--model", required=True, help="Target model path (e.g. Qwen3-4B)")
    parser.add_argument("--draft-model", help="Draft model path (exported DFlash checkpoint)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "eval_100")
    parser.add_argument("--datasets", default="vietnews,wikilingua,vims,vlsp", help="Comma-separated datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "vllm_vietbench_eval")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per dataset (default: 100)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--skip-vanilla", action="store_true", help="Skip vanilla baseline run")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed in the current environment.") from exc

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    all_summaries: list[dict[str, Any]] = []

    # Load tokenizer for chat template formatting
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as exc:
        print(f"⚠️ Warning: Could not pre-load tokenizer ({exc}). Will obtain from vLLM engine.")

    # 1. Run Vanilla Baseline if not skipped
    if not args.skip_vanilla:
        print("\n================================================================================")
        print("🚀 STEP 1/2: Running Vanilla vLLM Baseline (No Speculative Decoding)")
        print("================================================================================")
        vanilla_llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
        )
        if tokenizer is None:
            try:
                tokenizer = vanilla_llm.get_tokenizer()
            except Exception:
                pass
        
        vanilla_writer = JsonlWriter(args.output_dir / "vanilla_vllm_records.jsonl")
        
        for ds_name in dataset_list:
            raw_samples = load_dataset_samples(args.data_dir, ds_name, args.max_samples)
            prompts = prepare_prompts(raw_samples, ds_name, tokenizer)
            references = [r.get("reference", r.get("summary", "")) for r in raw_samples]
            sample_ids = [str(r.get("id", idx)) for idx, r in enumerate(raw_samples)]
            
            res = run_vllm_inference(
                vanilla_llm,
                prompts,
                references,
                sample_ids,
                sampling_params=sampling_params,
                method_name="vanilla_vllm",
                dataset_name=ds_name,
                writer=vanilla_writer,
            )
            all_summaries.append(res)
            
        vanilla_writer.close()
        del vanilla_llm
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Run DFlash Speculative Decoding
    if args.draft_model:
        print("\n================================================================================")
        print("⚡ STEP 2/2: Running DFlash Speculative Decoding in vLLM")
        print("================================================================================")
        
        # Speculative config for DFlash
        speculative_config = {
            "method": "dflash",
            "model": str(args.draft_model),
            "num_speculative_tokens": 16,
        }
        
        dflash_llm = LLM(
            model=args.model,
            speculative_config=speculative_config,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
        )
        if tokenizer is None:
            try:
                tokenizer = dflash_llm.get_tokenizer()
            except Exception:
                pass
        
        dflash_writer = JsonlWriter(args.output_dir / "dflash_spec_records.jsonl")
        
        for ds_name in dataset_list:
            raw_samples = load_dataset_samples(args.data_dir, ds_name, args.max_samples)
            prompts = prepare_prompts(raw_samples, ds_name, tokenizer)
            references = [r.get("reference", r.get("summary", "")) for r in raw_samples]
            sample_ids = [str(r.get("id", idx)) for idx, r in enumerate(raw_samples)]
            
            res = run_vllm_inference(
                dflash_llm,
                prompts,
                references,
                sample_ids,
                sampling_params=sampling_params,
                method_name="dflash_spec",
                dataset_name=ds_name,
                writer=dflash_writer,
            )
            all_summaries.append(res)
            
        dflash_writer.close()
        del dflash_llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Generate Comparative Report
    print("\n================================================================================")
    print("📊 BÁO CÁO TỔNG KẾT VIETBENCH BENCHMARK")
    print("================================================================================")
    
    # Calculate speedup
    speedup_map = {}
    vanilla_rates = {s["dataset"]: s["throughput_tok_s"] for s in all_summaries if s["method"] == "vanilla_vllm"}
    for s in all_summaries:
        if s["method"] != "vanilla_vllm" and s["dataset"] in vanilla_rates:
            v_rate = vanilla_rates[s["dataset"]]
            s["speedup"] = round(s["throughput_tok_s"] / max(v_rate, 1e-6), 2)
        else:
            s["speedup"] = 1.0 if s["method"] == "vanilla_vllm" else None

    # Write Markdown Table
    md_lines = [
        "# Báo cáo Đánh giá VietBench (vLLM Speculative Decoding)",
        "",
        "| Dataset | Phương pháp | Mẫu | Throughput (tok/s) | Speedup | Accept Rate (%) | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    
    for s in all_summaries:
        sp_str = f"**{s.get('speedup')}x**" if s.get('speedup') else "-"
        acc_str = f"{s.get('acceptance_rate_pct')}%" if s.get('acceptance_rate_pct') is not None else "-"
        r1 = s.get("rouge1") or "-"
        r2 = s.get("rouge2") or "-"
        rL = s.get("rougeL") or "-"
        md_lines.append(
            f"| `{s['dataset']}` | `{s['method']}` | {s['num_samples']} | {s['throughput_tok_s']} | {sp_str} | {acc_str} | {r1} | {r2} | {rL} |"
        )
    
    report_text = "\n".join(md_lines)
    print(report_text)
    
    report_path = args.output_dir / "evaluation_summary.md"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    
    json_path = args.output_dir / "evaluation_summary.json"
    json_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    print(f"\n✅ Đã lưu toàn bộ báo cáo và kết quả chi tiết tại: {args.output_dir}")


if __name__ == "__main__":
    main()
