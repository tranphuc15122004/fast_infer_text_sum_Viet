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
from Benchmark.common.benchmark_runtime import runtime_metadata
from Benchmark.common.paired_reference import build_v2_record_fields
from Benchmark.dflash_timing_patch import apply_dflash_timing_patch
from Benchmark.common.input_utils import truncate_input_ids
from Benchmark.common.prompt_format import format_chat_prompt
from Benchmark.common.paths import ROOT
from Benchmark.common.reproducibility import seed_everything
from Benchmark.dflash_compat import (
    install_dflash_cache_crop_compat,
    install_dflash_transformers_compat,
)


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
    acceptance_lengths: list[int],
    *,
    block_size: int,
    draft_tokens_accepted: int | None = None,
    draft_tokens_proposed: int | None = None,
) -> dict[str, int | float | str | None]:
    """Normalize DFlash trace and candidate counters to the shared schema."""
    from Benchmark.common.speculative_metrics import normalize_speculative_acceptance

    steps = len(acceptance_lengths)
    accepted = draft_tokens_accepted
    proposed = draft_tokens_proposed
    if block_size <= 1:
        accepted = None
        proposed = None
    elif accepted is None and proposed is None and steps:
        proposed = steps * (block_size - 1)
        accepted = sum(max(0, int(value) - 1) for value in acceptance_lengths)

    trace_average = (
        sum(float(value) for value in acceptance_lengths) / steps
        if steps
        else None
    )
    normalized = normalize_speculative_acceptance(
        verification_steps=steps,
        draft_tokens_accepted=accepted,
        draft_tokens_proposed=proposed,
        fallback_avg_accept_length=trace_average,
    )
    return {**normalized, "draft_proposal_unit": "linear_draft_slot"}


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


def _format_prompt(tokenizer, sample: dict) -> str:
    """Use the same target-model chat framing as the other benchmark adapters."""

    return format_chat_prompt(tokenizer, sample["prompt"])


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


def _warmup_generation(
    dflash_generate,
    draft,
    target,
    input_ids,
    *,
    warmup_runs: int,
    max_new_tokens: int,
    temperature: float,
    block_size: int,
) -> None:
    """Warm DFlash's first-generation kernels outside measured timings."""

    warmup_new_tokens = max(1, min(int(max_new_tokens), 8))
    for _ in range(max(int(warmup_runs), 0)):
        _run_generation(
            dflash_generate,
            draft,
            target,
            input_ids,
            max_new_tokens=warmup_new_tokens,
            temperature=temperature,
            block_size=block_size,
        )


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
    parser.add_argument("--warmup-runs", type=int, default=3)
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
    dflash_timing_patch = apply_dflash_timing_patch(ROOT)
    install_dflash_transformers_compat()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import dflash.model as dflash_model

    install_dflash_cache_crop_compat(dflash_model)
    DFlashDraftModel = dflash_model.DFlashDraftModel
    dflash_generate = dflash_model.dflash_generate

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

    runtime = runtime_metadata()
    hardware = {
        "gpu_name": runtime.get("gpu_name"),
        "gpu_capability": runtime.get("gpu_capability"),
        "cuda_version": runtime.get("cuda_version"),
    }
    stop_token_ids = target_config.eos_token_id
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]
    block_size = args.block_size or int(draft.block_size)
    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []
    did_warmup = False

    for sample in prompts:
        prompt = _format_prompt(tokenizer, sample)
        encoded = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = encoded.input_ids.to(device)
        if args.max_input_tokens and args.max_input_tokens > 0 \
                and input_ids.shape[1] > args.max_input_tokens:
            input_ids = truncate_input_ids(input_ids, args.max_input_tokens).contiguous()
        input_len = int(input_ids.shape[1])

        if not did_warmup:
            warmup_new_tokens = max(1, min(int(args.max_new_tokens), 8))
            _warmup_generation(
                dflash_generate,
                draft,
                target,
                input_ids,
                warmup_runs=args.warmup_runs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                block_size=block_size,
            )
            print(
                "[dflash] warmup complete: "
                f"runs={max(args.warmup_runs, 0)} "
                f"max_new_tokens={warmup_new_tokens}",
                flush=True,
            )
            did_warmup = True

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
        torch.cuda.synchronize(device)
        request_start = time.perf_counter()
        # Recreate prompt IDs inside the timed request boundary. The preceding
        # encoding is used only for warmup and the same-runtime reference.
        prompt = _format_prompt(tokenizer, sample)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids
        if args.max_input_tokens and args.max_input_tokens > 0 \
                and input_ids.shape[1] > args.max_input_tokens:
            input_ids = truncate_input_ids(input_ids, args.max_input_tokens).contiguous()
        prompt_token_ids = input_ids[0].tolist()
        input_ids = input_ids.to(device)
        input_len = int(input_ids.shape[1])
        seed_everything(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        result, elapsed = _run_generation(
            dflash_generate, draft, target, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            block_size=block_size,
        )

        output_ids = result.output_ids[0, input_len:]
        text = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        torch.cuda.synchronize(device)
        request_wall_ms = (time.perf_counter() - request_start) * 1000.0
        baseline_ids = (
            baseline.output_ids[0, input_len:] if baseline is not None else None
        )
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
            "committed_tokens_per_step": list(
                getattr(result, "committed_tokens_per_step", result.acceptance_lengths)
            ),
            "draft_latency_ms": round_optional(
                getattr(result, "draft_latency_ms", None)
            ),
            "verification_latency_ms": round_optional(
                getattr(result, "verification_latency_ms", None)
            ),
            "draft_tokens_proposed": getattr(result, "draft_tokens_proposed", None),
            "draft_tokens_accepted": getattr(result, "draft_tokens_accepted", None),
            **summarize_acceptance(
                list(result.acceptance_lengths),
                block_size=block_size,
                draft_tokens_accepted=getattr(
                    result, "draft_tokens_accepted", None
                ),
                draft_tokens_proposed=getattr(
                    result, "draft_tokens_proposed", None
                ),
            ),
            "speedup_scope": (
                "paired_target_only"
                if baseline is not None
                else None
            ),
            "speedup_reference_method": (
                "dflash_block_size_1" if baseline is not None else None
            ),
            "paired_output_exact_match": (
                text == baseline_text if baseline_text is not None else None
            ),
            "paired_output_token_ids_match": (
                torch.equal(output_ids, baseline_ids)
                if baseline_ids is not None
                else None
            ),
            "paired_output_token_count_match": (
                baseline_n_tok == n_tok if baseline_n_tok is not None else None
            ),
            "paired_output_token_ratio": (
                round(n_tok / baseline_n_tok, 4)
                if baseline_n_tok is not None and baseline_n_tok > 0
                else None
            ),
        }
        visible_tokens = sum(
            1 for token in output_ids.tolist()
            if int(token) not in set(getattr(tokenizer, "all_special_ids", []) or [])
        )
        record.update(
            build_v2_record_fields(
                prompt_token_ids=prompt_token_ids,
                generation_config={
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                    "stop_token_ids": stop_token_ids,
                },
                hardware=hardware,
                request_wall_ms=request_wall_ms,
                native_elapsed_ms=e2e_ms,
                native_timing_scope="generation",
                timed_generated_tokens=n_tok,
                visible_output_tokens=visible_tokens,
                decode_active_ms=getattr(result, "strict_decode_active_ms", None),
                decode_token_count=getattr(result, "strict_decode_token_count", None),
                decode_phase_definition=getattr(result, "strict_decode_phase_definition", None),
                decode_phase_verified=getattr(result, "strict_decode_phase_verified", False) is True,
                timing_source=f"{dflash_timing_patch['patch_version']}:{dflash_timing_patch['patched_source_sha256']}",
                target_revision=str(args.target_model),
                tokenizer_revision=str(getattr(tokenizer, "name_or_path", args.target_model)),
                gpu_count=1,
                tp_size=1,
                batch_size=1,
                concurrency=1,
                cache_policy="no_cross_request_prefix_reuse",
            )
        )
        if baseline is not None and baseline_elapsed is not None:
            baseline_visible_tokens = sum(
                1 for token in baseline_ids.tolist()
                if int(token) not in set(getattr(tokenizer, "all_special_ids", []) or [])
            )
            record["native_reference"] = {
                **build_v2_record_fields(
                    prompt_token_ids=prompt_token_ids,
                    generation_config={
                        "temperature": args.temperature,
                        "max_new_tokens": args.max_new_tokens,
                        "seed": args.seed,
                        "stop_token_ids": stop_token_ids,
                    },
                    hardware=hardware,
                    native_elapsed_ms=base_e2e_ms,
                    native_timing_scope="generation",
                    timed_generated_tokens=baseline_n_tok,
                    visible_output_tokens=baseline_visible_tokens,
                    decode_active_ms=getattr(baseline, "strict_decode_active_ms", None),
                    decode_token_count=getattr(baseline, "strict_decode_token_count", None),
                    decode_phase_definition=getattr(baseline, "strict_decode_phase_definition", None),
                    decode_phase_verified=getattr(baseline, "strict_decode_phase_verified", False) is True,
                    timing_source=f"{dflash_timing_patch['patch_version']}:{dflash_timing_patch['patched_source_sha256']}",
                    target_revision=str(args.target_model),
                    tokenizer_revision=str(getattr(tokenizer, "name_or_path", args.target_model)),
                    gpu_count=1,
                    tp_size=1,
                    batch_size=1,
                    concurrency=1,
                    cache_policy="no_cross_request_prefix_reuse",
                ),
                "sample_id": sample["id"],
                "dataset": sample.get("raw", {}).get("dataset", Path(args.data_file).stem if args.data_file else "prompt"),
                "method": "dflash_block_size_1",
                "status": "success",
                "text": baseline_text,
                "output_tokens": baseline_n_tok,
            }
        record["dflash_timing_patch"] = dflash_timing_patch
        record["speedup_valid"] = metrics.has_valid_paired_speedup(record)
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
        "speedup_scope": (
            "paired_target_only" if not args.skip_reference else None
        ),
        "speedup_reference_method": (
            "dflash_block_size_1" if not args.skip_reference else None
        ),
        "speedup": metrics.aggregate_speedup(writer.records),
        **metrics.aggregate_paired_reference_fidelity(writer.records),
        **quality,
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("DFlash", checks)


if __name__ == "__main__":
    main()
