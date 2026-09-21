"""Shared implementation for the Vanilla HF and Vanilla FA baselines."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

import torch

from common import io_util, metrics, rouge
from common.benchmark_runtime import (
    build_sample_record,
    measure_call,
    runtime_metadata,
)
from common.data_loader import load_records
from common.input_utils import truncate_input_ids
from common.quality_guard import is_degenerate_output
from common.reproducibility import seed_everything


def build_parser(default_backend: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default=os.environ.get("LONG_BENCH_MODEL")
        or os.environ.get("MODEL_TARGET"),
    )
    parser.add_argument("--data-file", default=os.environ.get("LONG_BENCH_DATA_FILE"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("LONG_BENCH_SEED", "42"))
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument(
        "--device", default=os.environ.get("LONG_BENCH_DEVICE", "cuda")
    )
    parser.add_argument("--dtype", default=os.environ.get("LONG_BENCH_DTYPE", "bfloat16"))
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LONG_BENCH_LOCAL_FILES_ONLY", "1") == "1",
    )
    parser.add_argument(
        "--attention-backend",
        choices=[default_backend],
        default=default_backend,
    )
    parser.add_argument("--run-id", default=os.environ.get("LONG_BENCH_RUN_ID"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _prompt_batch(tokenizer: Any, prompt: str, *, max_input_tokens: int) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt")
    return truncate_input_ids(encoded.input_ids, max_input_tokens)


def _generate(model: Any, input_ids: torch.Tensor, args: argparse.Namespace) -> Any:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": False,
        "pad_token_id": model.generation_config.pad_token_id,
    }
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    attention_mask = torch.ones_like(input_ids)
    return model.generate(input_ids, attention_mask=attention_mask, **kwargs)


def _warmup_args(args: argparse.Namespace, *, max_new_tokens: int = 8) -> argparse.Namespace:
    """Copy generation args with a short warmup budget.

    Warmup is for kernel/cache initialization, not for measuring long-form
    generation.  Reusing the benchmark's full output budget here can waste
    minutes before the first sample when the configured budget is 2048+.
    """
    values = vars(args).copy()
    values["max_new_tokens"] = min(int(args.max_new_tokens), int(max_new_tokens))
    return argparse.Namespace(**values)


def _next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scores = logits[:, -1, :]
    if temperature > 0:
        probabilities = torch.softmax(scores / temperature, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)
    return scores.argmax(dim=-1, keepdim=True)


def _is_eos(token: torch.Tensor, eos_token_id: int | list[int] | None) -> bool:
    if eos_token_id is None:
        return False
    eos_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    return int(token.reshape(-1)[0]) in {int(value) for value in eos_ids}


def _build_decode_attention_mask(
    input_ids: torch.Tensor, *, max_new_tokens: int
) -> torch.Tensor:
    """Allocate the complete 1D attention mask once for one request.

    The old decode loop appended one column with ``torch.cat`` at every step.
    Besides allocating repeatedly, that made long generations progressively
    more expensive.  A full mask is tiny compared with model activations and
    can be sliced as the KV cache grows.
    """
    total_tokens = int(input_ids.shape[1]) + max(int(max_new_tokens), 0)
    return torch.ones(
        (int(input_ids.shape[0]), total_tokens),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )


def _build_static_cache(
    model: Any,
    *,
    max_cache_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any | None, str]:
    """Build a preallocated Transformers cache when the runtime supports it.

    ``DynamicCache`` concatenates K/V tensors during every generated token.
    ``StaticCache`` writes into preallocated storage and is substantially more
    suitable for the long, single-request generations used by this benchmark.
    Older Transformers versions may not expose it or may reject a model
    configuration; those versions retain the public ``generate`` fallback.
    """
    try:
        from transformers.cache_utils import StaticCache
    except (ImportError, ModuleNotFoundError):
        return None, "dynamic"

    config = getattr(model, "config", None)
    if config is None:
        return None, "dynamic"

    try:
        cache = StaticCache(
            config=config,
            max_cache_len=max(int(max_cache_len), 1),
            device=device,
            dtype=dtype,
        )
    except RuntimeError as exc:
        # Never turn a real allocation failure into a slower second attempt.
        if "out of memory" in str(exc).lower():
            raise
        return None, "dynamic"
    except (AttributeError, TypeError, ValueError):
        return None, "dynamic"
    return cache, "static"


def _should_use_static_cache(attention_backend: str | None) -> bool:
    """Return whether the manual cache path is safe for this attention backend.

    Flash-Attention 2 combined with the Transformers ``StaticCache`` path has
    produced silently corrupted repeated-token outputs on some server
    combinations.  Keep the optimized static path for eager attention, while
    making FA use the public dynamic-cache semantics until its output parity is
    revalidated.  This only changes cache allocation; model weights, prompts,
    and decoding policy remain unchanged.
    """

    return attention_backend != "flash_attention_2"


def _timed_generate(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Greedy cached decoding with explicit prefill/decode timings.

    ``generate()`` exposes only one end-to-end wall time.  The manual loop
    records the prefill and incremental decode phases needed for ESR/DSR.  A
    compatibility fallback keeps the script usable with older Transformers
    cache APIs, while honestly leaving unavailable phase timings as ``null``.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    request_start = time.perf_counter()
    input_length = int(input_ids.shape[1])
    attention_mask = _build_decode_attention_mask(
        input_ids, max_new_tokens=args.max_new_tokens
    )
    try:
        model_dtype = next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        model_dtype = _dtype(getattr(args, "dtype", "float32"))
    if _should_use_static_cache(getattr(args, "attention_backend", None)):
        static_cache, cache_backend = _build_static_cache(
            model,
            max_cache_len=input_length + int(args.max_new_tokens),
            device=device,
            dtype=model_dtype,
        )
    else:
        static_cache, cache_backend = None, "dynamic_fa_safe"
    try:
        prefill_start = time.perf_counter()
        prefill_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask[:, :input_length],
            "use_cache": True,
            "return_dict": True,
        }
        if static_cache is not None:
            prefill_kwargs["past_key_values"] = static_cache
        prefill = model(**prefill_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        past = getattr(prefill, "past_key_values", None)
        if past is None:
            past = static_cache
        next_token = _next_token(prefill.logits, args.temperature)
        generated = [next_token]
        eos_id = tokenizer.eos_token_id

        decode_start = time.perf_counter()
        if not _is_eos(next_token, eos_id):
            for _ in range(max(args.max_new_tokens - 1, 0)):
                current_length = input_length + len(generated)
                step = model(
                    input_ids=next_token,
                    attention_mask=attention_mask[:, :current_length],
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = getattr(step, "past_key_values", None)
                if past is None:
                    past = static_cache
                next_token = _next_token(step.logits, args.temperature)
                generated.append(next_token)
                if _is_eos(next_token, eos_id):
                    break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        output_ids = torch.cat([input_ids, *generated], dim=1)
        e2e_ms = (time.perf_counter() - request_start) * 1000.0
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else None
        )
        output_tokens = len(generated)
        return output_ids, {
            "prefill_ms": round(prefill_ms, 3),
            "ttft_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "e2e_ms": round(e2e_ms, 3),
            "tpot_ms": round(decode_ms / max(output_tokens - 1, 1), 3),
            "peak_memory_gb": round(peak_memory_gb, 6)
            if peak_memory_gb is not None
            else None,
            "device": str(device),
            "kv_cache_backend": cache_backend,
            "attention_mask_strategy": "preallocated_slice",
        }
    except (AttributeError, IndexError, TypeError, ValueError):
        # Transformers 4.x and 5.x expose different cache classes/arguments.
        # Fall back to the stable public generate API rather than emitting a
        # partial record that looks like a valid phase measurement.
        output_ids, timing = measure_call(
            lambda: _generate(model, input_ids, args), device=device
        )
        timing.update(
            {
                "prefill_ms": None,
                "ttft_ms": None,
                "decode_ms": None,
                "tpot_ms": None,
                "kv_cache_backend": "generate",
                "attention_mask_strategy": "generate",
            }
        )
        return output_ids, timing


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    if not args.model:
        raise SystemExit("--model or LONG_BENCH_MODEL/MODEL_TARGET is required")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use orchestrator smoke preflight on this host")

    if args.attention_backend == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "vanilla_fa requires the installed flash-attn wheel; no fallback is allowed"
            ) from exc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=_dtype(args.dtype),
        attn_implementation=args.attention_backend,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    model.to(device).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def run(args: argparse.Namespace, *, method: str) -> int:
    if args.smoke:
        args.max_samples = 1
        args.max_new_tokens = min(args.max_new_tokens, 8)
        if args.max_input_tokens is None:
            args.max_input_tokens = 4096
    elif args.max_input_tokens is None:
        args.max_input_tokens = 0
    if not args.data_file:
        raise SystemExit("--data-file or LONG_BENCH_DATA_FILE is required")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    records = load_records(Path(args.data_file), args.max_samples)
    data_name = Path(args.data_file).stem

    load_start = time.perf_counter()
    model, tokenizer = _load_model(args, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_ms = round((time.perf_counter() - load_start) * 1000.0, 3)
    metadata = runtime_metadata()
    effective_attention_backend = getattr(
        getattr(model, "config", None), "_attn_implementation", None
    )
    config = {
        "device": str(device),
        "gpu_name": metadata.get("gpu_name"),
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup_runs,
        "batch_size": 1,
        "extra_metrics": {
            "requested_attention_backend": args.attention_backend,
            "effective_attention_backend": effective_attention_backend
            or "unknown",
        },
    }

    with torch.inference_mode():
        seed_everything(args.seed)
        warmup_ids = _prompt_batch(tokenizer, "Hello", max_input_tokens=0).to(device)
        warmup_args = _warmup_args(args)
        for _ in range(max(args.warmup_runs, 0)):
            _generate(model, warmup_ids, warmup_args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    writer = io_util.JsonlWriter(Path(args.output))
    successful = 0
    for sample in records:
        seed_everything(args.seed)
        input_ids = _prompt_batch(
            tokenizer,
            sample["prompt"],
            max_input_tokens=max(args.max_input_tokens, 0),
        ).to(device)
        input_tokens = int(input_ids.shape[1])
        with torch.inference_mode():
            output_ids, timing = _timed_generate(
                model, input_ids, tokenizer, args, device
            )
        new_ids = output_ids[0, input_tokens:]
        output_tokens = int(new_ids.shape[0])
        text = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        timing["model_load_ms"] = model_load_ms
        record = build_sample_record(
            method=method,
            dataset=sample.get("raw", {}).get("dataset", data_name),
            sample_id=sample["id"],
            model=str(args.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timing=timing,
            config={
                **config,
                "extra_metrics": {
                    **dict(config.get("extra_metrics", {}) or {}),
                    "kv_cache_backend": timing.get("kv_cache_backend"),
                    "attention_mask_strategy": timing.get(
                        "attention_mask_strategy"
                    ),
                },
            },
            text=text,
            reference_output=sample.get("reference"),
        )
        record["output_quality_guard"] = {
            "degenerate_repetition": is_degenerate_output(text),
            "action": "annotate_only",
        }
        rouge.add_rouge(record, text, sample.get("reference"))
        metrics.add_semantic(record, text, sample.get("reference"))
        if record["output_quality_guard"]["degenerate_repetition"]:
            print(
                f"[{method}][{sample['id']}] warning: output has a strong "
                "repeated-n-gram collapse signal; inference result is retained "
                "for audit but should not be used as a quality comparison.",
                flush=True,
            )
        record["run_id"] = args.run_id
        record["task_type"] = sample.get("raw", {}).get("task_type")
        writer.add(record)
        successful += 1
        print(
            f"[{method}][{record['dataset']}][{sample['id']}] "
            f"input={input_tokens} output={output_tokens} "
            f"prefill_ms={record['prefill_ms']} decode_ms={record['decode_ms']} "
            f"e2e_ms={record['e2e_ms']} tok_s={record['throughput_tok_s']} "
            f"decode_tok_s={record['decode_throughput_tok_s']} "
            f"cache={record['extra_metrics'].get('kv_cache_backend')} "
            f"attn={record['extra_metrics'].get('effective_attention_backend')}",
            flush=True,
        )

    summary = {
        "type": "summary",
        "method": method,
        "dataset": data_name,
        "run_id": args.run_id,
        "status": "success" if successful == len(records) else "failed",
        "num_samples": len(records),
        "successful_samples": successful,
        "model": args.model,
        "model_load_ms": model_load_ms,
        "attention_backend": args.attention_backend,
        "effective_attention_backend": effective_attention_backend or "unknown",
        "runtime": metadata,
        **rouge.aggregate_rouge(writer.records),
        **metrics.aggregate_semantic(writer.records),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    return 0 if successful == len(records) else 1
