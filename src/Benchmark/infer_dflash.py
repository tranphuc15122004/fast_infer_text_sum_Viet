#!/usr/bin/env python3
"""DFlash representative benchmark adapter.

Runs the paired target + DFlash draft checkpoints on unified JSONL prompts and
records both DFlash and target-only timings.  ``block_size=1`` is the paired
autoregressive reference used for the speedup fields.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from Benchmark.common import io_util, metrics, rouge, verify
from Benchmark.common.data_loader import load_records
from Benchmark.common.input_utils import truncate_input_ids
from Benchmark.common.paths import ROOT
from Benchmark.common.reproducibility import seed_everything
from Benchmark.dflash_compat import install_dflash_transformers_compat


# DFlash is vendored rather than installed into the shared server Python.
# Register its package root before the lazy import in ``main`` so direct
# adapter runs and orchestrated runs behave identically.
DFLASH_ROOT = ROOT / "externals" / "dflash"
if str(DFLASH_ROOT) not in sys.path:
    sys.path.insert(0, str(DFLASH_ROOT))

_CANONICAL_LONGBENCH_DATASETS = {
    "gov_report",
    "qmsum",
    "multi_news",
    "lcc",
    "repobench-p",
    "vietnews",
    "wikilingua",
    "vims",
    "vlsp",
}


def round_optional(value: float | None, digits: int = 3) -> float | None:
    """Round an optional metric without fabricating an external reference."""

    return round(value, digits) if value is not None else None


def summarize_acceptance(
    acceptance_lengths: list[int], *, block_size: int
) -> dict[str, float | None]:
    """Normalize DFlash block acceptance telemetry for the shared schema."""

    if not acceptance_lengths:
        return {
            "avg_accept_length": None,
            "acceptance_rate": None,
            "rejected_draft_ratio": None,
        }
    average = sum(float(value) for value in acceptance_lengths) / len(acceptance_lengths)
    if block_size <= 1:
        acceptance_rate = None
        rejected_ratio = None
    else:
        proposed = len(acceptance_lengths) * (block_size - 1)
        accepted_draft = sum(max(0, int(value) - 1) for value in acceptance_lengths)
        acceptance_rate = round(accepted_draft / proposed, 4) if proposed else None
        rejected_ratio = (
            round(1.0 - acceptance_rate, 4)
            if acceptance_rate is not None
            else None
        )
    return {
        "avg_accept_length": round(average, 4),
        "acceptance_rate": acceptance_rate,
        "rejected_draft_ratio": rejected_ratio,
    }


def normalize_generation_token_ids(config, tokenizer) -> dict[str, tuple[int, int]]:
    """Repair stale BOS/EOS ids that are outside the loaded tokenizer vocab.

    Some local DFlash configs were copied from a Qwen checkpoint and contain
    ids such as 151643/151645 while the Llama tokenizer has a 128256-token
    vocabulary.  Transformers only warns during loading, then generation can
    silently use invalid stop ids.  Normalize before model construction and
    return the changes for an auditable runtime log.
    """

    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = int(getattr(config, "vocab_size", 0) or 0)

    def valid(value) -> bool:
        return isinstance(value, int) and 0 <= value < vocab_size

    changed: dict[str, tuple[int, int]] = {}
    for name in ("bos_token_id", "eos_token_id"):
        current = getattr(config, name, None)
        replacement = getattr(tokenizer, name, None)
        if replacement is None or not valid(int(replacement)):
            continue
        if current is None or not valid(current):
            old = current
            setattr(config, name, int(replacement))
            changed[name] = (old, int(replacement))
    return changed


def _dtype_and_attention() -> tuple[torch.dtype, str]:
    if not torch.cuda.is_available():
        raise SystemExit("DFlash Transformers adapter requires CUDA")
    capability = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    requested = os.environ.get("LONG_BENCH_DFLASH_ATTENTION", "sdpa").strip().lower()
    if requested in {"sdpa", "eager"}:
        return dtype, requested
    if requested not in {"flash_attention_2", "flash_attention_4"}:
        raise ValueError(
            "LONG_BENCH_DFLASH_ATTENTION must be sdpa, eager, "
            "flash_attention_2, or flash_attention_4"
        )
    return dtype, requested


def _chat_prompt(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _format_prompt(tokenizer, sample: dict) -> str:
    """Avoid applying a second chat template to rendered LongBench prompts."""

    raw = sample.get("raw") or {}
    if raw.get("dataset") in _CANONICAL_LONGBENCH_DATASETS:
        return sample["prompt"]
    return _chat_prompt(tokenizer, sample["prompt"])


def _run_generation(dflash_generate, draft, target, input_ids, *, max_new_tokens,
                    temperature, block_size):
    torch.cuda.synchronize()
    start = time.perf_counter()
    eos_id = target.config.eos_token_id
    stop_token_ids = eos_id if isinstance(eos_id, list) else [eos_id]
    result = dflash_generate(
        draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        block_size=block_size,
        return_stats=True,
    )
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def _timings(result, elapsed_s: float) -> tuple[float, float, float]:
    e2e_ms = elapsed_s * 1e3
    prefill_ms = float(result.time_to_first_token) * 1e3
    decode_ms = max(e2e_ms - prefill_ms, 0.0)
    return prefill_ms, decode_ms, e2e_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=0,
                        help="truncate each prompt to this many tokens before "
                             "generation (0 = no limit; use on T4 smoke runs)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("LONG_BENCH_SEED", "42")),
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="run only speculative decoding; attach an external Vanilla reference later",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = 1
        args.max_new_tokens = min(args.max_new_tokens, 32)

    dtype, attn_impl = _dtype_and_attention()
    install_dflash_transformers_compat()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel, dflash_generate

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_config = AutoConfig.from_pretrained(args.target_model)
    target_token_changes = normalize_generation_token_ids(target_config, tokenizer)
    draft_config = AutoConfig.from_pretrained(args.draft_model)
    draft_token_changes = normalize_generation_token_ids(draft_config, tokenizer)
    if target_token_changes or draft_token_changes:
        print(
            "[dflash] normalized generation token ids: "
            f"target={target_token_changes or 'none'} "
            f"draft={draft_token_changes or 'none'}",
            flush=True,
        )
    load_start = time.perf_counter()
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        config=target_config,
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        config=draft_config,
    ).to(device).eval()
    torch.cuda.synchronize(device)
    model_load_ms = round((time.perf_counter() - load_start) * 1000.0, 3)

    if args.data_file:
        prompts = load_records(Path(args.data_file), args.max_samples)
    else:
        prompts = [{"id": "prompt", "prompt": args.prompt, "reference": None}]

    block_size = args.block_size or int(draft.block_size)
    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []

    for sample in prompts:
        prompt = _format_prompt(tokenizer, sample)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        if args.max_input_tokens and args.max_input_tokens > 0 \
                and input_ids.shape[1] > args.max_input_tokens:
            input_ids = truncate_input_ids(input_ids, args.max_input_tokens).contiguous()
        input_len = int(input_ids.shape[1])

        if args.skip_reference:
            baseline, baseline_elapsed = None, None
        else:
            seed_everything(args.seed)
            baseline, baseline_elapsed = _run_generation(
                dflash_generate, draft, target, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                block_size=1,
            )
        seed_everything(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        result, elapsed = _run_generation(
            dflash_generate, draft, target, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            block_size=block_size,
        )

        output_ids = result.output_ids[0, input_len:]
        baseline_ids = (
            baseline.output_ids[0, input_len:] if baseline is not None else None
        )
        text = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        baseline_text = None
        if baseline_ids is not None:
            baseline_text = tokenizer.decode(
                baseline_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        n_tok = int(output_ids.shape[0])
        baseline_n_tok = int(baseline_ids.shape[0]) if baseline_ids is not None else None
        prefill_ms, decode_ms, e2e_ms = _timings(result, elapsed)
        if baseline is not None and baseline_elapsed is not None:
            base_prefill_ms, base_decode_ms, base_e2e_ms = _timings(
                baseline, baseline_elapsed
            )
        else:
            base_prefill_ms, base_decode_ms, base_e2e_ms = None, None, None

        record = {
            "method": "dflash",
            "dataset": "data-file" if args.data_file else "prompt",
            "task_type": sample.get("raw", {}).get("task_type"),
            "model": args.target_model,
            "draft_model": args.draft_model,
            "input_tokens": input_len,
            "retained_tokens": input_len,
            "output_tokens": n_tok,
            "baseline_output_tokens": baseline_n_tok,
            "batch_size": 1,
            "selector_latency_ms": None,
            "ttft_ms": round(prefill_ms, 3),
            "prefill_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "e2e_ms": round(e2e_ms, 3),
            "baseline_ttft_ms": round_optional(base_prefill_ms),
            "baseline_prefill_ms": round_optional(base_prefill_ms),
            "baseline_decode_ms": round_optional(base_decode_ms),
            "baseline_e2e_ms": round_optional(base_e2e_ms),
            "dense_prefill_ms": round_optional(base_prefill_ms),
            "dense_decode_ms": round_optional(base_decode_ms),
            "dense_e2e_ms": round_optional(base_e2e_ms),
            "tpot_ms": round(decode_ms / n_tok, 3) if n_tok else None,
            "throughput_tok_s": round(n_tok / (e2e_ms / 1e3), 2)
            if e2e_ms > 0 and n_tok else 0.0,
            "decode_throughput_tok_s": round(n_tok / (decode_ms / 1e3), 2)
            if decode_ms > 0 and n_tok else 0.0,
            "qps": None,
            "peak_memory_gb": round(
                torch.cuda.max_memory_allocated(device) / (1024**3), 6
            ),
            "model_load_ms": model_load_ms,
            "device": str(device),
            "measurement_scope": "full_e2e",
            "sample_id": sample["id"],
            "text": text,
            "reference_output": sample.get("reference"),
            "baseline_text": baseline_text,
            "block_size": block_size,
            "acceptance_lengths": list(result.acceptance_lengths),
            "draft_latency_ms": round_optional(
                getattr(result, "draft_latency_ms", None)
            ),
            "verification_latency_ms": round_optional(
                getattr(result, "verification_latency_ms", None)
            ),
            "draft_tokens_proposed": getattr(result, "draft_tokens_proposed", None),
            "draft_tokens_accepted": getattr(result, "draft_tokens_accepted", None),
            **summarize_acceptance(
                list(result.acceptance_lengths), block_size=block_size
            ),
            "speedup_scope": "paired_dflash_block_size_1" if baseline is not None else None,
            "speedup_valid": (
                baseline is not None
                and baseline_n_tok == n_tok
            ),
        }
        if record["task_type"] == "code_completion":
            metrics.add_code_completion(record, text, sample.get("reference"))
        else:
            rouge.add_rouge(record, text, sample.get("reference"))
            metrics.add_semantic(record, text, sample.get("reference"))
        writer.add(record)
        print(
            f"[sample {sample['id']}] dflash={e2e_ms:.1f}ms "
            + (
                f"baseline={base_e2e_ms:.1f}ms "
                if base_e2e_ms is not None
                else "reference=external "
            )
            + f"tokens={n_tok}",
            flush=True,
        )
        checks.append(verify.check_new_tokens(n_tok))
        checks.append(verify.check_output_text(text))

    if any(r.get("task_type") == "code_completion" for r in writer.records):
        quality = metrics.aggregate_code_completion(writer.records)
    else:
        quality = {
            **rouge.aggregate_rouge(writer.records),
            **metrics.aggregate_semantic(writer.records),
        }
    summary = {
        "type": "summary",
        "method": "dflash",
        "num_samples": len(prompts),
        "block_size": block_size,
        "speedup": metrics.aggregate_speedup(writer.records),
        **quality,
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("DFlash", checks)


if __name__ == "__main__":
    main()
