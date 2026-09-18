"""Exact-target and ROUGE evaluation for a trained DFlash draft.

This module is deliberately separate from ``scripts/infer_dflash.py``: it
loads only the portable draft export produced by this package and has no
runtime dependency on ``MR_DFlash`` or the benchmark adapters.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

import torch

from .checkpoint import load_draft_initialization
from .data import (
    DEFAULT_SUMMARY_PROMPT_TEMPLATE,
    iter_summary_jsonl,
    render_summary_prompt,
)
from .distributed import (
    cleanup_distributed,
    initialize_distributed,
    merge_ranked_jsonl,
    ranked_path,
)
from .model import DFlashDraftModel


def _tokens(text: str) -> list[str]:
    return text.casefold().split()


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if overlap <= 0 or predicted <= 0 or reference <= 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2.0 * precision * recall / (precision + recall)


def _rouge_n(prediction: list[str], reference: list[str], n: int) -> float:
    if len(prediction) < n or len(reference) < n:
        return 0.0
    predicted_ngrams = Counter(tuple(prediction[i : i + n]) for i in range(len(prediction) - n + 1))
    reference_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
    overlap = sum((predicted_ngrams & reference_ngrams).values())
    return _f1(overlap, sum(predicted_ngrams.values()), sum(reference_ngrams.values()))


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """Return dependency-free whitespace ROUGE-1/2/L F1 scores."""

    predicted_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    return {
        "rouge1_f": _rouge_n(predicted_tokens, reference_tokens, 1),
        "rouge2_f": _rouge_n(predicted_tokens, reference_tokens, 2),
        "rougeL_f": _f1(
            _lcs_length(predicted_tokens, reference_tokens),
            len(predicted_tokens),
            len(reference_tokens),
        ),
    }


def summarize_generation_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    """Aggregate quality, exactness and paired end-to-end speed safely."""

    rows = list(records)
    if not rows:
        raise ValueError("generation evaluation produced no records")
    rouge_totals = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    exact = 0
    draft_seconds = 0.0
    target_seconds = 0.0
    acceptance_lengths: list[int] = []
    for row in rows:
        prediction = row.get("prediction")
        reference = row.get("reference_summary")
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise ValueError("generation record requires string prediction and reference_summary")
        for name, value in rouge_scores(prediction, reference).items():
            rouge_totals[name] += value
        exact += int(row.get("target_token_match") is True)
        draft_seconds += float(row.get("dflash_elapsed_s", 0.0))
        target_seconds += float(row.get("target_elapsed_s", 0.0))
        raw_acceptance = row.get("acceptance_lengths")
        if raw_acceptance is not None:
            if not isinstance(raw_acceptance, list) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in raw_acceptance
            ):
                raise ValueError("acceptance_lengths must be a list of non-negative integers")
            acceptance_lengths.extend(raw_acceptance)
    count = len(rows)
    result: dict[str, float | int] = {
        "num_samples": count,
        **{name: total / count for name, total in rouge_totals.items()},
        "target_exact_rate": exact / count,
        "dflash_e2e_s": draft_seconds,
        "target_e2e_s": target_seconds,
        "speedup": target_seconds / draft_seconds if draft_seconds > 0 else 0.0,
        "mean_acceptance_length": (
            sum(acceptance_lengths) / len(acceptance_lengths)
            if acceptance_lengths
            else None
        ),
    }
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_generated_ids(output: Any, prompt_length: int) -> list[int]:
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("generation must return one token-id sequence")
    return [int(value) for value in output[0, prompt_length:].detach().cpu().tolist()]


def _trim_eos(token_ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is not None and int(eos_token_id) in token_ids:
        return token_ids[: token_ids.index(int(eos_token_id))]
    return token_ids


def evaluate_draft_generation(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tokenizer: Any,
    target: Any,
    draft: Any,
    max_length: int,
    max_source_tokens: int,
    max_summary_tokens: int,
    chat_template: str,
    prompt_template: str,
    device: str | torch.device,
    max_samples: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    include_summary: bool = True,
    allow_empty: bool = False,
) -> dict[str, float | int | None]:
    """Evaluate DFlash against the frozen target and human references.

    ``target_token_match`` is only meaningful for deterministic generation;
    this function always uses greedy decoding for both paths.
    """

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"generation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    device_obj = torch.device(device)
    rows: list[dict[str, Any]] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    stop_token_ids = [int(eos_token_id)] if eos_token_id is not None else []
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for source_index, record in enumerate(
                iter_summary_jsonl(input_path, max_samples=max_samples)
            ):
                if source_index % world_size != rank:
                    continue
                prompt = render_summary_prompt(
                    record,
                    tokenizer,
                    max_length=max_length,
                    max_source_tokens=max_source_tokens,
                    max_summary_tokens=max_summary_tokens,
                    chat_template=chat_template,
                    prompt_template=prompt_template,
                ).unsqueeze(0).to(device_obj)
                prompt_length = int(prompt.shape[1])
                _sync(device_obj)
                started = time.perf_counter()
                draft_result = draft.spec_generate(
                    target,
                    prompt,
                    max_new_tokens=max_summary_tokens,
                    stop_token_ids=stop_token_ids,
                    temperature=0.0,
                    return_stats=True,
                )
                _sync(device_obj)
                draft_elapsed = time.perf_counter() - started
                _sync(device_obj)
                started = time.perf_counter()
                with torch.inference_mode():
                    target_output = target.generate(
                        prompt,
                        do_sample=False,
                        max_new_tokens=max_summary_tokens,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                _sync(device_obj)
                target_elapsed = time.perf_counter() - started
                if isinstance(draft_result, tuple):
                    if len(draft_result) != 2 or not isinstance(draft_result[1], Mapping):
                        raise ValueError("draft spec_generate returned malformed statistics")
                    draft_output, draft_stats = draft_result
                    acceptance_lengths = draft_stats.get("acceptance_lengths")
                else:
                    draft_output = draft_result
                    acceptance_lengths = None
                draft_ids = _as_generated_ids(draft_output, prompt_length)
                target_ids = _as_generated_ids(target_output, prompt_length)
                visible_ids = _trim_eos(draft_ids, eos_token_id)
                reference = record.metadata.get("reference_summary", record.summary)
                if not isinstance(reference, str):
                    raise ValueError(f"record {record.id!r} has non-string reference_summary")
                row = {
                    "id": record.id,
                    "prediction": tokenizer.decode(visible_ids, skip_special_tokens=True).strip(),
                    "reference_summary": reference,
                    "draft_tokens": len(draft_ids),
                    "target_tokens": len(target_ids),
                    "target_token_match": draft_ids == target_ids,
                    "dflash_elapsed_s": draft_elapsed,
                    "target_elapsed_s": target_elapsed,
                    "acceptance_lengths": acceptance_lengths,
                }
                row.update(rouge_scores(row["prediction"], reference))
                if world_size > 1:
                    row["_source_index"] = source_index
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                rows.append(row)
            if include_summary and rows:
                summary = {"type": "summary", **summarize_generation_records(rows)}
                handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
            elif include_summary and not allow_empty:
                raise ValueError("generation evaluation produced no records")
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    if not rows:
        if not allow_empty:
            raise ValueError("generation evaluation produced no records")
        return {"num_samples": 0}
    return summarize_generation_records(rows)


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return value


def _same_local_path(left: str, right: str) -> bool:
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(
        strict=False
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a portable DFlash draft on Vietnamese summaries")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-export", required=True)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--max-source-tokens", required=True, type=int)
    parser.add_argument("--max-summary-tokens", required=True, type=int)
    parser.add_argument("--chat-template", default="qwen3")
    parser.add_argument("--prompt-template", default=DEFAULT_SUMMARY_PROMPT_TEMPLATE)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.max_source_tokens < 0 or args.max_summary_tokens < 1:
        raise ValueError("source budget must be non-negative and summary budget positive")
    if args.max_source_tokens + args.max_summary_tokens > args.max_length:
        raise ValueError("source and summary token budgets exceed --max-length")
    export_dir = Path(args.draft_export)
    metadata_path = export_dir / "draft_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"portable draft metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stored_target_path = metadata.get("target_model_path")
    if not isinstance(stored_target_path, str) or not _same_local_path(
        stored_target_path, args.target_model_path
    ):
        raise ValueError("draft export target_model_path does not match --target-model-path")
    draft_payload = metadata.get("draft_config")
    if not isinstance(draft_payload, Mapping):
        raise ValueError("draft export metadata lacks draft_config")
    context = initialize_distributed(args.device)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

        device = (
            torch.device("cuda", context.local_rank)
            if context.is_distributed and torch.cuda.is_available()
            else torch.device(args.device)
        )
        dtype = _dtype(args.torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(device).eval()
        draft = DFlashDraftModel(Qwen3Config.from_dict(dict(draft_payload))).to(
            device=device, dtype=dtype
        )
        load_draft_initialization(export_dir, draft, metadata)
        result = evaluate_draft_generation(
            args.input,
            ranked_path(args.output, context.rank, context.world_size),
            tokenizer=tokenizer,
            target=target,
            draft=draft,
            max_length=args.max_length,
            max_source_tokens=args.max_source_tokens,
            max_summary_tokens=args.max_summary_tokens,
            chat_template=args.chat_template,
            prompt_template=args.prompt_template,
            device=device,
            max_samples=args.max_samples,
            rank=context.rank,
            world_size=context.world_size,
            include_summary=not context.is_distributed,
            allow_empty=context.is_distributed,
        )
        context.barrier()
        if context.is_main_process:
            if context.is_distributed:
                merge_ranked_jsonl(
                    [
                        ranked_path(args.output, rank, context.world_size)
                        for rank in range(context.world_size)
                    ],
                    args.output,
                )
                rows = [
                    json.loads(line)
                    for line in Path(args.output).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                result = summarize_generation_records(rows)
                with Path(args.output).open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"type": "summary", **result},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()


__all__ = [
    "evaluate_draft_generation",
    "main",
    "rouge_scores",
    "summarize_generation_records",
]
