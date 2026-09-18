"""Generate deterministic target trajectories for DFlash summarization training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from .data import DEFAULT_SUMMARY_PROMPT_TEMPLATE, iter_summary_jsonl, render_summary_prompt
from .adaptive_inference import (
    AdaptiveInferenceSettings,
    BackoffResult,
    PreparedExample,
    adaptive_settings_from_args,
    add_adaptive_cli_args,
    length_bucket_batches,
    run_with_oom_backoff,
    select_cuda_batch_size,
    validate_adaptive_inference_settings,
)
from .distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
    merge_ranked_jsonl,
    ranked_path,
)


def _generated_ids(output: Any, prompt_length: int) -> list[int]:
    if isinstance(output, torch.Tensor):
        value = output
    elif hasattr(output, "sequences"):
        value = output.sequences
    else:
        raise TypeError("target.generate must return token sequences")
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError("target.generate must return exactly one sequence")
    return [int(token) for token in value[0, prompt_length:].detach().cpu().tolist()]


def _pad_prompt_batch(
    prompts: list[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not prompts:
        raise ValueError("cannot pad an empty prompt batch")
    lengths = [int(prompt.numel()) for prompt in prompts]
    width = max(lengths)
    input_ids = torch.full(
        (len(prompts), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((len(prompts), width), dtype=torch.long, device=device)
    for row, prompt in enumerate(prompts):
        length = lengths[row]
        input_ids[row, width - length :] = prompt.to(device=device, dtype=torch.long)
        attention_mask[row, width - length :] = 1
    return input_ids, attention_mask, lengths


def _generated_id_rows(output: Any, padded_prompt_length: int) -> list[list[int]]:
    if isinstance(output, torch.Tensor):
        value = output
    elif hasattr(output, "sequences"):
        value = output.sequences
    else:
        raise TypeError("target.generate must return token sequences")
    if value.ndim != 2:
        raise ValueError("target.generate must return a rank-2 sequence tensor")
    if value.shape[1] <= padded_prompt_length:
        return [[] for _ in range(value.shape[0])]
    return [
        [int(token) for token in row[padded_prompt_length:].detach().cpu().tolist()]
        for row in value
    ]


def _generate_batch(
    examples: list[PreparedExample],
    *,
    tokenizer: Any,
    target: Any,
    max_summary_tokens: int,
    device: torch.device,
) -> list[list[int]]:
    prompts = [example.payload["prompt"] for example in examples]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    input_ids, attention_mask, _lengths = _pad_prompt_batch(
        prompts,
        pad_token_id=int(pad_token_id),
        device=device,
    )
    with torch.inference_mode():
        output = target.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_summary_tokens,
            pad_token_id=int(pad_token_id),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    return _generated_id_rows(output, input_ids.shape[1])


def _trim_at_eos(token_ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return token_ids
    try:
        return token_ids[: token_ids.index(int(eos_token_id))]
    except ValueError:
        return token_ids


def generate_teacher_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tokenizer: Any,
    target: Any,
    target_model_path: str,
    max_length: int,
    max_source_tokens: int,
    max_summary_tokens: int,
    chat_template: str,
    prompt_template: str = DEFAULT_SUMMARY_PROMPT_TEMPLATE,
    device: str | torch.device,
    rank: int = 0,
    world_size: int = 1,
    allow_empty: bool = False,
    adaptive_settings: AdaptiveInferenceSettings | None = None,
) -> dict[str, int]:
    """Write target-generated summaries while retaining human references."""

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"teacher trajectory output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    written = 0
    rejected = 0
    processed_tokens = 0
    oom_retries = 0
    started = time.perf_counter()
    device_obj = torch.device(device)
    settings = adaptive_settings or AdaptiveInferenceSettings(
        enabled=False,
        min_batch_size=1,
        max_batch_size=1,
        bucket_window=1,
    )
    validate_adaptive_inference_settings(settings)
    current_batch_size = settings.min_batch_size
    selection = None
    observed_peak_reserved = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            window: list[PreparedExample] = []

            def flush_window(items: list[PreparedExample]) -> None:
                nonlocal written, processed_tokens, oom_retries
                nonlocal current_batch_size, selection, observed_peak_reserved
                if not items:
                    return
                if selection is None:
                    probe_items = sorted(items, key=lambda item: (-item.length, item.index))

                    def probe(batch_size: int) -> None:
                        selected = probe_items[:batch_size]
                        for _ in range(settings.probe_batches):
                            _generate_batch(
                                selected,
                                tokenizer=tokenizer,
                                target=target,
                                max_summary_tokens=max_summary_tokens,
                                device=device_obj,
                            )
                        return None

                    selection = select_cuda_batch_size(
                        probe,
                        settings=settings,
                        device=device_obj,
                        maximum=min(settings.max_batch_size, len(probe_items)),
                    )
                    current_batch_size = selection.batch_size
                    if device_obj.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(device_obj)
                results: dict[int, list[int]] = {}
                for batch in length_bucket_batches(
                    items,
                    batch_size=current_batch_size,
                    max_tokens=settings.max_tokens_per_batch,
                    window=max(settings.bucket_window, current_batch_size),
                ):
                    offset = 0
                    while offset < len(batch):
                        subset = batch[offset : offset + current_batch_size]

                        def work(size: int) -> list[list[int]]:
                            return _generate_batch(
                                subset[:size],
                                tokenizer=tokenizer,
                                target=target,
                                max_summary_tokens=max_summary_tokens,
                                device=device_obj,
                            )

                        result = run_with_oom_backoff(
                            work,
                            initial_batch_size=len(subset),
                            minimum_batch_size=min(settings.min_batch_size, len(subset)),
                        ) if settings.oom_backoff else BackoffResult(
                            work(len(subset)), len(subset), 0
                        )
                        for example, token_ids in zip(
                            subset[: result.batch_size], result.value, strict=True
                        ):
                            results[example.index] = token_ids
                            processed_tokens += example.length
                        oom_retries += result.oom_retries
                        if device_obj.type == "cuda" and torch.cuda.is_available():
                            observed_peak_reserved = max(
                                observed_peak_reserved,
                                int(torch.cuda.max_memory_reserved(device_obj)),
                            )
                        current_batch_size = min(current_batch_size, result.batch_size)
                        offset += result.batch_size
                for example in sorted(items, key=lambda item: item.index):
                    token_ids = _trim_at_eos(
                        results[example.index],
                        getattr(tokenizer, "eos_token_id", None),
                    )
                    record = example.payload["record"]
                    summary = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    if len(token_ids) < 2 or not summary:
                        rejected += 1
                        continue
                    payload = {
                        "id": record.id,
                        "document": record.document,
                        "summary": summary,
                        "reference_summary": record.summary,
                        **dict(record.metadata),
                        "teacher": {
                            "target_model_path": str(target_model_path),
                            "chat_template": chat_template,
                            "prompt_template": prompt_template,
                            "do_sample": False,
                            "max_length": max_length,
                            "max_source_tokens": max_source_tokens,
                            "max_summary_tokens": max_summary_tokens,
                        },
                    }
                    if world_size > 1:
                        payload["_source_index"] = example.index
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1

            for source_index, record in enumerate(iter_summary_jsonl(input_path)):
                if source_index % world_size != rank:
                    continue
                try:
                    prompt = render_summary_prompt(
                        record,
                        tokenizer,
                        max_length,
                        max_source_tokens=max_source_tokens,
                        max_summary_tokens=max_summary_tokens,
                        chat_template=chat_template,
                        prompt_template=prompt_template,
                    )
                    window.append(
                        PreparedExample(
                            source_index,
                            int(prompt.numel()) + max_summary_tokens,
                            {"record": record, "prompt": prompt},
                        )
                    )
                    if len(window) >= max(settings.bucket_window, settings.min_batch_size):
                        flush_window(window)
                        window = []
                except (TypeError, ValueError) as exc:
                    rejected += 1
                    continue
            flush_window(window)
        if written == 0 and not allow_empty:
            raise ValueError("target generation produced no usable summaries")
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    stats: dict[str, int] = {
        "written": written,
        "rejected": rejected,
    }
    if adaptive_settings is not None:
        elapsed = max(time.perf_counter() - started, 1e-9)
        stats.update(
            {
                "tokens": processed_tokens,
                "elapsed_ms": int(elapsed * 1000),
                "samples_per_sec_x1000": int(written / elapsed * 1000),
                "tokens_per_sec_x1000": int(processed_tokens / elapsed * 1000),
                "oom_retries": oom_retries,
                "adaptive_batch_size": int(current_batch_size),
                "adaptive_target_memory_bytes": int(
                    selection.target_memory_bytes if selection is not None else 0
                ),
                "adaptive_peak_reserved_bytes": int(
                    max(
                        observed_peak_reserved,
                        selection.peak_reserved_bytes if selection is not None else 0,
                    )
                ),
                "adaptive_probe_count": int(selection.probes if selection is not None else 0),
                "adaptive_hit_maximum": int(
                    selection.hit_maximum if selection is not None else False
                ),
            }
        )
    return stats


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen teacher trajectories for DFlash")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-source-tokens", type=int, required=True)
    parser.add_argument("--max-summary-tokens", type=int, required=True)
    parser.add_argument("--chat-template", default="qwen3")
    parser.add_argument("--prompt-template", default=DEFAULT_SUMMARY_PROMPT_TEMPLATE)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    add_adaptive_cli_args(parser)
    args = parser.parse_args(argv)
    if args.max_source_tokens < 0 or args.max_summary_tokens < 1:
        raise ValueError("source budget must be non-negative and summary budget positive")
    if args.max_source_tokens + args.max_summary_tokens > args.max_length:
        raise ValueError("source and summary token budgets exceed --max-length")
    context = initialize_distributed(args.device)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = (
            torch.device("cuda", context.local_rank)
            if context.is_distributed and torch.cuda.is_available()
            else torch.device(args.device)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=_dtype(args.torch_dtype),
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(device).eval()
        adaptive_settings = adaptive_settings_from_args(args, device)
        shard_output = ranked_path(args.output, context.rank, context.world_size)
        stats = generate_teacher_jsonl(
            args.input,
            shard_output,
            tokenizer=tokenizer,
            target=target,
            target_model_path=args.target_model_path,
            max_length=args.max_length,
            max_source_tokens=args.max_source_tokens,
            max_summary_tokens=args.max_summary_tokens,
            chat_template=args.chat_template,
            prompt_template=args.prompt_template,
            device=device,
            rank=context.rank,
            world_size=context.world_size,
            allow_empty=context.is_distributed,
            adaptive_settings=adaptive_settings,
        )
        totals = context.all_reduce_sum(
            torch.tensor(
                [
                    stats["written"],
                    stats["rejected"],
                    stats.get("tokens", 0),
                    stats.get("oom_retries", 0),
                ],
                dtype=torch.float64,
                device=device,
            )
        ).cpu()
        context.barrier()
        if context.is_main_process:
            merge_ranked_jsonl(
                [ranked_path(args.output, rank, context.world_size) for rank in range(context.world_size)],
                args.output,
            ) if context.is_distributed else None
            summary = {
                "written": int(totals[0]),
                "rejected": int(totals[1]),
                "tokens": int(totals[2]),
                "oom_retries": int(totals[3]),
            }
            if adaptive_settings.enabled:
                elapsed_ms = max(int(stats.get("elapsed_ms", 0)), 1)
                summary.update(
                    {
                        "elapsed_ms_rank0": elapsed_ms,
                        "tokens_per_sec_x1000_rank0": int(
                            totals[2].item() / (elapsed_ms / 1000.0) * 1000
                        ),
                        "adaptive_batch_size_rank0": int(
                            stats.get("adaptive_batch_size", 1)
                        ),
                        "adaptive_target_memory_bytes": int(
                            stats.get("adaptive_target_memory_bytes", 0)
                        ),
                        "adaptive_peak_reserved_bytes_rank0": int(
                            stats.get("adaptive_peak_reserved_bytes", 0)
                        ),
                    }
                )
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()


__all__ = ["generate_teacher_jsonl", "main"]
