"""Run a paired, offline evaluation of the synchronized vLLM baselines.

This entrypoint is for the B200/server runtime. It deliberately loads one
target + one draft at a time, then runs the same tokenized prompts and greedy
sampling settings for every selected method. Modal smoke tests remain in
``scripts/modal_vllm_pilot.py``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import math
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Benchmark.common.benchmark_data import read_jsonl, render_prompt
from Benchmark.common.io_util import JsonlWriter, validate_schema
from Benchmark.common.prompt_format import format_chat_prompt
from Benchmark.common.quality_guard import is_degenerate_output
from Benchmark.common.rouge import add_rouge, aggregate_rouge
from Benchmark.common.speculative_metrics import normalize_speculative_acceptance
from Benchmark.common.vllm_pilot import (
    install_domino_vllm_compat,
    paired_vllm_metrics,
    pilot_warmup_tokens,
    resolve_eagle3_aux_hidden_state_layers,
    select_pilot_methods,
    speculative_token_count,
)


METHODS = ("vanilla_vllm", "eagle3", "dflash", "domino", "dspark")
DRAFT_ARGUMENTS = {
    "eagle3": "eagle3_model",
    "dflash": "dflash_model",
    "domino": "domino_model",
    "dspark": "dspark_model",
}


def order_methods(method_names: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate a method list and always run vanilla first as the reference."""

    selected = select_pilot_methods(
        method_names, available_methods=METHODS, reference="vanilla_vllm"
    )
    return ("vanilla_vllm", *(method for method in selected if method != "vanilla_vllm"))


def mean_itl_ms(*, first_token_ts: float, last_token_ts: float, output_tokens: int) -> float | None:
    """Mean inter-token latency; vLLM emits one first token before ITL begins."""

    if output_tokens <= 1:
        return None
    duration_ms = max(0.0, float(last_token_ts) - float(first_token_ts)) * 1000.0
    return duration_ms / (output_tokens - 1)


def token_lcs_overlap(vanilla_ids: list[int], method_ids: list[int]) -> float | None:
    """Return ordered-token LCS overlap divided by the vanilla output length."""

    if not vanilla_ids:
        return None
    denominator = len(vanilla_ids)
    left_ids, right_ids = vanilla_ids, method_ids
    if len(right_ids) > len(left_ids):
        left_ids, right_ids = right_ids, left_ids
    row = [0] * (len(right_ids) + 1)
    for left in left_ids:
        previous = 0
        for index, right in enumerate(right_ids, start=1):
            saved = row[index]
            row[index] = previous + 1 if left == right else max(row[index], row[index - 1])
            previous = saved
    return row[-1] / denominator if denominator else None


def build_speculative_config(
    method: str,
    draft_path: str,
    draft_config: Any,
    *,
    target_num_hidden_layers: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build the vLLM speculative config and serializable checkpoint metadata."""

    if method == "vanilla_vllm":
        return None, {"vllm_method": None, "draft_model": None}
    if method not in DRAFT_ARGUMENTS:
        raise ValueError(f"unknown vLLM method: {method}")

    raw = draft_config.to_dict() if hasattr(draft_config, "to_dict") else dict(draft_config)
    speculative_tokens = speculative_token_count(method, draft_config)
    vllm_method = "dflash" if method == "domino" else method
    speculative = {
        "method": vllm_method,
        "model": str(draft_path),
        "num_speculative_tokens": speculative_tokens,
    }
    dflash_config = raw.get("dflash_config") or {}
    if method == "domino" and dflash_config.get("projector_type") != "domino":
        raise ValueError(
            "Domino checkpoint must declare dflash_config.projector_type='domino'"
        )

    raw_aux_layers = raw.get("eagle_aux_hidden_state_layer_ids")
    resolved_aux_layers = None
    if method == "eagle3":
        resolved_aux_layers = resolve_eagle3_aux_hidden_state_layers(
            draft_config, target_num_hidden_layers=target_num_hidden_layers
        )

    metadata = {
        "vllm_method": vllm_method,
        "draft_model": str(draft_path),
        "num_speculative_tokens": speculative_tokens,
        "draft_architectures": list(raw.get("architectures") or []),
        "draft_model_config": _jsonable(raw),
        "target_model_name_or_path": raw.get("target_model_name_or_path"),
        "target_layer_ids": _jsonable(raw.get("target_layer_ids")),
        "eagle_aux_hidden_state_layer_ids": _jsonable(raw_aux_layers),
        "resolved_eagle_aux_hidden_state_layer_ids": _jsonable(resolved_aux_layers),
        "target_num_hidden_layers": target_num_hidden_layers,
        "ttt_length": raw.get("ttt_length"),
        "projector_type": dflash_config.get("projector_type"),
        "dflash_config": _jsonable(dflash_config),
    }
    return speculative, metadata


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    slots = getattr(type(value), "__slots__", ())
    if slots:
        return {
            name: _jsonable(getattr(value, name))
            for name in slots
            if isinstance(name, str) and hasattr(value, name)
        }
    return str(value)

def _normalize_vllm_spec_metrics(metrics: Any, *, method: str) -> dict[str, Any]:
    raw = _jsonable(metrics)
    values = raw if isinstance(raw, dict) else {}

    def first_value(*names: str) -> tuple[Any, str | None]:
        for name in names:
            if name in values and values[name] is not None:
                return values[name], name
        return None, None

    steps, _ = first_value(
        "num_drafts", "num_verification_steps", "verification_steps", "num_spec_decode_steps"
    )
    accepted, _ = first_value(
        "num_accepted_tokens", "draft_tokens_accepted",
        "num_draft_tokens_accepted", "accepted_draft_tokens"
    )
    proposed, _ = first_value(
        "num_draft_tokens", "draft_tokens_proposed",
        "num_proposed_tokens", "proposed_draft_tokens"
    )
    acceptance_rate, _ = first_value("acceptance_rate", "draft_acceptance_rate")
    acceptance_percent, _ = first_value(
        "acceptance_rate_percent", "draft_acceptance_rate_percent"
    )
    if acceptance_rate is None and acceptance_percent is not None:
        try:
            acceptance_rate = float(acceptance_percent) / 100.0
        except (TypeError, ValueError, OverflowError):
            acceptance_rate = None
    average_length, _ = first_value(
        "avg_accept_length", "average_accept_length", "mean_accept_length"
    )
    normalized: dict[str, Any] = normalize_speculative_acceptance(
        verification_steps=steps,
        draft_tokens_accepted=accepted,
        draft_tokens_proposed=proposed,
        fallback_acceptance_rate=acceptance_rate,
        fallback_avg_accept_length=average_length,
    )
    normalized["draft_proposal_unit"] = (
        "draft_tree_nodes" if method == "eagle3" else "draft_tokens"
    ) if method != "vanilla_vllm" else None
    return normalized


def _append_event(
    path: Path, event: str, run_started_perf: float, **details: Any
) -> None:
    payload = _jsonable(details)
    if not isinstance(payload, dict):
        payload = {"details": payload}
    payload.update(
        {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round(max(0.0, time.perf_counter() - run_started_perf) * 1000.0, 3),
        }
    )
    _append_jsonl(path, payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_memory_snapshot(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    properties = torch.cuda.get_device_properties(0)
    snapshot = {
        "cuda_available": True,
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(getattr(properties, "total_memory", 0) or 0),
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "compute_capability": {
            "major": getattr(properties, "major", None),
            "minor": getattr(properties, "minor", None),
        },
    }
    return snapshot


def _nvidia_smi_snapshot() -> dict[str, Any]:
    """Capture device-level GPU telemetry, including vLLM worker allocations."""
    captured_at = datetime.now(timezone.utc).isoformat()
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
        if completed.returncode != 0:
            return {
                "available": False,
                "captured_at_utc": captured_at,
                "error": completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}",
            }
        devices = []
        for row in csv.reader(completed.stdout.splitlines()):
            fields = [field.strip() for field in row]
            if len(fields) != 9:
                continue
            index, name, total, used, free, gpu_util, memory_util, temperature, driver = fields
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "memory_total_mib": total,
                    "memory_used_mib": used,
                    "memory_free_mib": free,
                    "gpu_utilization_percent": gpu_util,
                    "memory_utilization_percent": memory_util,
                    "temperature_c": temperature,
                    "driver_version": driver,
                }
            )
        return {
            "available": bool(devices),
            "captured_at_utc": captured_at,
            "devices": devices,
            "raw": completed.stdout.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "captured_at_utc": captured_at,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _runtime_snapshot(torch: Any, vllm: Any, transformers: Any) -> dict[str, Any]:
    safe_runtime_env = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "OMP_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_LOGGING_LEVEL",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "cuda_runtime_version": getattr(getattr(torch, "version", None), "cuda", None),
        "environment_variables": {
            key: os.environ[key] for key in safe_runtime_env if key in os.environ
        },
        "gpu_process_memory": _gpu_memory_snapshot(torch),
        "gpu_system": _nvidia_smi_snapshot(),
    }


def _reference_text(row: dict[str, Any]) -> str | None:
    reference = row.get("reference")
    if reference is not None and str(reference).strip():
        return str(reference)
    answers = row.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0])
    if isinstance(answers, str) and answers.strip():
        return answers
    return None


def _prepare_samples(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_input_tokens: int,
    max_total_tokens: int,
    max_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", index))
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample id in data file: {sample_id}")
        seen_ids.add(sample_id)
        prompt = format_chat_prompt(tokenizer, render_prompt(row))
        token_ids = [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
        document_words = int(row.get("document_words", 0) or 0)
        if not token_ids:
            excluded.append(
                {
                    "sample_id": sample_id,
                    "reason": "empty_prompt",
                    "source_record": _jsonable(row),
                }
            )
            continue
        limit = min(max_input_tokens, max_total_tokens) if max_input_tokens > 0 else max_total_tokens
        if len(token_ids) > limit:
            excluded.append(
                {
                    "sample_id": sample_id,
                    "reason": "input_limit",
                    "input_tokens": len(token_ids),
                    "limit": limit,
                    "source_record": _jsonable(row),
                }
            )
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": str(row.get("dataset") or "unknown"),
                "prompt": prompt,
                "prompt_token_ids": token_ids,
                "reference": _reference_text(row),
                "document_words": document_words,
                "input_tokens": len(token_ids),
                "source_record": _jsonable(row),
            }
        )
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("no samples remain after applying the common input limits")
    return samples, excluded


def _local_checkpoint(path: str, label: str) -> Path:
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"{label} must be a local model directory with config.json: {path}")
    return checkpoint.resolve()


def _model_paths(args: argparse.Namespace, methods: tuple[str, ...]) -> dict[str, str]:
    target = str(_local_checkpoint(args.model, "target model"))
    paths = {"vanilla_vllm": target}
    for method, arg_name in DRAFT_ARGUMENTS.items():
        if method not in methods:
            continue
        value = getattr(args, arg_name)
        if not value:
            raise ValueError(f"--{arg_name.replace('_', '-')} is required when selecting {method}")
        paths[method] = str(_local_checkpoint(value, f"{method} draft model"))
    return paths


def _base_record(method: str, sample: dict[str, Any], model: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "method": method,
        "dataset": sample["dataset"],
        "model": model,
        "sample_id": sample["sample_id"],
        "input_tokens": sample["input_tokens"],
        "retained_tokens": sample["input_tokens"],
        "document_words": sample.get("document_words"),
        "output_tokens": 0,
        "batch_size": 1,
        "selector_latency_ms": None,
        "server_startup_ms": None,
        "queue_wait_ms": None,
        "batch_wait_ms": None,
        "ttft_ms": None,
        "prefill_ms": None,
        "draft_latency_ms": None,
        "verification_latency_ms": None,
        "tpot_ms": None,
        "e2e_ms": None,
        "server_reported_e2e_ms": None,
        "throughput_tok_s": None,
        "qps": None,
        "peak_memory_gb": None,
        "gpu_memory": None,
        "status": "failed",
        "text": "",
        "reference": sample["reference"],
        "prompt": sample["prompt"],
        "prompt_token_ids": sample["prompt_token_ids"],
        "output_token_ids": [],
        "repetition_flag": None,
        "decode_matches_text": None,
        "prompt_tokens_match": None,
        "finish_reason": None,
        "greedy_token_match": None,
        "token_lcs_overlap_with_vanilla": None,
        "quality_valid": None,
        "raw_request_output": None,
        "raw_request_metrics": None,
        "raw_speculative_metrics": None,
        "speculative_decoding_metrics": None,
        "error_traceback": None,
    }
    for key in (
        "avg_accept_length",
        "acceptance_rate",
        "acceptance_rate_percent",
        "accepted_draft_tokens_per_step",
        "draft_tokens_accepted",
        "draft_tokens_proposed",
        "draft_proposal_unit",
        "rejected_draft_ratio",
        "verification_steps",
    ):
        record[key] = None
    return record


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def _request_record(
    *,
    method: str,
    sample: dict[str, Any],
    model: str,
    request_output: Any,
    tokenizer: Any,
    client_wall_ms: float,
    gpu_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completion = request_output.outputs[0]
    token_ids = [int(token) for token in (completion.token_ids or [])]
    text = str(completion.text or "")
    request_metrics = getattr(request_output, "metrics", None)
    raw_metrics = _jsonable(request_metrics)
    raw_speculative = _jsonable(getattr(completion, "spec_decode_metrics", None))
    raw_output = _jsonable(request_output)

    def optional_metric_float(name: str) -> float | None:
        value = getattr(request_metrics, name, None) if request_metrics is not None else None
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    timing: dict[str, float | None] = {}
    scheduled_ts = optional_metric_float("scheduled_ts")
    first_ts = optional_metric_float("first_token_ts")
    last_ts = optional_metric_float("last_token_ts")
    queued_ts = optional_metric_float("queued_ts")
    if None not in (scheduled_ts, first_ts, last_ts, queued_ts):
        assert scheduled_ts is not None and first_ts is not None
        assert last_ts is not None and queued_ts is not None
        timing = {
            "queue_wait_ms": max(0.0, scheduled_ts - queued_ts) * 1000.0,
            "prefill_ms": max(0.0, first_ts - scheduled_ts) * 1000.0,
            "ttft_ms": (
                max(0.0, value) * 1000.0
                if (value := optional_metric_float("first_token_latency")) is not None
                else None
            ),
            "decode_ms": max(0.0, last_ts - first_ts) * 1000.0,
            "e2e_ms": max(0.0, last_ts - queued_ts) * 1000.0,
            "mean_itl_ms": mean_itl_ms(
                first_token_ts=first_ts, last_token_ts=last_ts, output_tokens=len(token_ids)
            ),
        }
    decoded_text = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    normalize = lambda value: " ".join(str(value).split())
    decode_matches = normalize(decoded_text) == normalize(text)
    actual_prompt_ids = [int(token) for token in (request_output.prompt_token_ids or [])]
    prompt_matches = actual_prompt_ids == sample["prompt_token_ids"]
    reported_tokens = getattr(request_metrics, "num_generation_tokens", None)
    generation_matches = reported_tokens is None or int(reported_tokens) == len(token_ids)
    finish_reason = str(completion.finish_reason or "")
    repetition = is_degenerate_output(text)
    prefill_ms = timing.get("prefill_ms")
    mean_tpot = timing.get("mean_itl_ms")
    e2e_ms = timing.get("e2e_ms")
    timing_valid = (
        request_metrics is not None
        and prefill_ms is not None
        and prefill_ms > 0
        and mean_tpot is not None
        and mean_tpot > 0
    )
    acceptance = _normalize_vllm_spec_metrics(raw_speculative, method=method)
    record = _base_record(method, sample, model)
    record.update(
        {
            "status": (
                "success"
                if token_ids
                and decode_matches
                and prompt_matches
                and generation_matches
                and finish_reason in {"stop", "length"}
                and timing_valid
                else "invalid"
            ),
            "text": text,
            "output_tokens": len(token_ids),
            "output_token_ids": token_ids,
            "repetition_flag": repetition,
            "quality_valid": bool(text.strip()) and len(token_ids) > 1 and not repetition,
            "decode_matches_text": decode_matches,
            "prompt_tokens_match": prompt_matches,
            "finish_reason": finish_reason,
            "raw_request_output": raw_output,
            "raw_request_metrics": raw_metrics,
            "raw_speculative_metrics": raw_speculative,
            "speculative_decoding_metrics": raw_speculative,
            "speculative_metrics_available": raw_speculative is not None,
            "raw_prefill_ms": prefill_ms,
            "prefill_ms": prefill_ms,
            "mean_itl_ms": mean_tpot,
            "decode_ms": timing.get("decode_ms"),
            "client_wall_ms": client_wall_ms,
            "server_reported_e2e_ms": e2e_ms,
            "queue_wait_ms": timing.get("queue_wait_ms"),
            "ttft_ms": timing.get("ttft_ms"),
            "tpot_ms": mean_tpot,
            "e2e_ms": e2e_ms,
            "throughput_tok_s": 1000.0 / mean_tpot if mean_tpot is not None and mean_tpot > 0 else None,
            "qps": 1000.0 / e2e_ms if e2e_ms is not None and e2e_ms > 0 else None,
            "gpu_memory": gpu_memory,
            "peak_memory_gb": (
                gpu_memory.get("peak_allocated_bytes", 0) / (1024**3)
                if gpu_memory and gpu_memory.get("cuda_available")
                else None
            ),
            **acceptance,
        }
    )
    add_rouge(record, text, sample["reference"])
    return record


def _failed_record(method: str, sample: dict[str, Any], model: str, error: str) -> dict[str, Any]:
    record = _base_record(method, sample, model)
    record.update({"status": "failed", "error": error})
    return record


def _annotate_parity(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    reference_rows = {
        row["sample_id"]: row
        for row in records
        if row["method"] == "vanilla_vllm" and row.get("output_token_ids")
    }
    summaries: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        method_rows = [row for row in records if row["method"] == method]
        compared = 0
        exact = 0
        overlap_numerator = 0.0
        overlap_denominator = 0
        per_sample = {}
        for row in method_rows:
            reference = reference_rows.get(row["sample_id"])
            output_ids = row.get("output_token_ids") or []
            if reference is None or not output_ids:
                continue
            reference_ids = reference.get("output_token_ids") or []
            matched = reference_ids == output_ids
            overlap = token_lcs_overlap(reference_ids, output_ids)
            row["greedy_token_match"] = matched
            row["token_lcs_overlap_with_vanilla"] = overlap
            compared += 1
            exact += int(matched)
            if overlap is not None:
                overlap_numerator += overlap * len(reference_ids)
                overlap_denominator += len(reference_ids)
            per_sample[row["sample_id"]] = {
                "exact_match": matched,
                "lcs_overlap": overlap,
                "reference_tokens": len(reference_ids),
                "method_tokens": len(output_ids),
            }
        summaries[method] = {
            "compared_samples": compared,
            "exact_matches": exact,
            "exact_match_rate": exact / compared if compared else None,
            "token_lcs_overlap": (
                overlap_numerator / overlap_denominator if overlap_denominator else None
            ),
            "per_sample": per_sample,
        }
    return summaries


def _method_metrics(records: list[dict[str, Any]], methods: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    output: dict[str, Any] = {}
    reference_rows = grouped.get("vanilla_vllm", [])
    for method in methods:
        rows = grouped.get(method, [])
        good = [row for row in rows if row.get("status") == "success"]
        numeric = [row for row in good if row.get("tpot_ms") is not None]
        tpot = sum(float(row["tpot_ms"]) for row in numeric) / len(numeric) if numeric else None
        prefills = [float(row["raw_prefill_ms"]) for row in good if row.get("raw_prefill_ms")]
        metrics = None
        if method == "vanilla_vllm" and len(good) == len(rows) and good:
            reference_itl = sum(float(row["tpot_ms"]) for row in good) / len(good)
            reference_prefill = sum(float(row["raw_prefill_ms"]) for row in good) / len(good)
            metrics = {
                "paired_samples": len(good),
                "reference_prefill_ms": reference_prefill,
                "reference_mean_itl_ms": reference_itl,
                "method_mean_itl_ms": reference_itl,
                "mean_min_output_tokens": sum(int(row["output_tokens"]) for row in good) / len(good),
                "dsr": 1.0,
                "esr": 1.0,
            }
        elif (
            reference_rows
            and len(good) == len(rows)
            and len(good) == len(reference_rows)
            and good
        ):
            try:
                metrics = paired_vllm_metrics(
                    [*reference_rows, *good], reference="vanilla_vllm"
                ).get(method)
            except ValueError:
                metrics = None
        quality_valid = sum(bool(row.get("quality_valid")) for row in rows)
        acceptance_values = [
            float(row["acceptance_rate"])
            for row in rows
            if row.get("acceptance_rate") is not None
        ]
        accept_length_values = [
            float(row["avg_accept_length"])
            for row in rows
            if row.get("avg_accept_length") is not None
        ]
        accepted_values = [
            int(row["draft_tokens_accepted"])
            for row in rows
            if row.get("draft_tokens_accepted") is not None
        ]
        proposed_values = [
            int(row["draft_tokens_proposed"])
            for row in rows
            if row.get("draft_tokens_proposed") is not None
        ]
        proposal_units = sorted({
            str(row["draft_proposal_unit"])
            for row in rows
            if row.get("draft_proposal_unit") is not None
        })
        rouge = aggregate_rouge(rows)
        output[method] = {
            "samples": len(rows),
            "successful_outputs": len(good),
            "quality_valid_outputs": quality_valid,
            "quality_valid_rate": quality_valid / len(rows) if rows else 0.0,
            "repetition_flags": sum(bool(row.get("repetition_flag")) for row in rows),
            "mean_rouge1": rouge.get("rouge1"),
            "mean_rouge2": rouge.get("rouge2"),
            "mean_rougeL": rouge.get("rougeL"),
            "mean_prefill_ms": sum(prefills) / len(prefills) if prefills else None,
            "mean_tpot_ms": tpot,
            "mean_throughput_tok_s": 1000.0 / tpot if tpot else None,
            "speculative_metrics_available_samples": sum(
                bool(row.get("speculative_metrics_available")) for row in rows
            ),
            "acceptance_rate_samples": len(acceptance_values),
            "mean_acceptance_rate": (
                sum(acceptance_values) / len(acceptance_values) if acceptance_values else None
            ),
            "mean_acceptance_rate_percent": (
                100.0 * sum(acceptance_values) / len(acceptance_values)
                if acceptance_values
                else None
            ),
            "mean_avg_accept_length": (
                sum(accept_length_values) / len(accept_length_values)
                if accept_length_values
                else None
            ),
            "total_draft_tokens_accepted_observed": (
                sum(accepted_values) if accepted_values else None
            ),
            "total_draft_tokens_proposed_observed": (
                sum(proposed_values) if proposed_values else None
            ),
            "draft_proposal_units": proposal_units,
            "paired_speed_metrics": metrics,
        }
    return output


def _shutdown_engine(engine: Any) -> None:
    if engine is None:
        return
    try:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    except Exception:
        try:
            engine.shutdown()
        except Exception:
            pass


def _evaluate(args: argparse.Namespace) -> int:
    run_started_utc = datetime.now(timezone.utc).isoformat()
    run_started_perf = time.perf_counter()
    methods = order_methods(args.methods)
    paths = _model_paths(args, methods)
    for required_file in (Path(args.data_file),):
        if not required_file.is_file():
            raise FileNotFoundError(f"evaluation data file not found: {required_file}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

    import torch
    import transformers
    import vllm
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM unified evaluation requires an available CUDA GPU")
    target_config = AutoConfig.from_pretrained(paths["vanilla_vllm"], local_files_only=True)
    target_layers = int(getattr(target_config, "num_hidden_layers", 0) or 0)
    tokenizer = AutoTokenizer.from_pretrained(paths["vanilla_vllm"], local_files_only=True)
    runtime_snapshot = _runtime_snapshot(torch, vllm, transformers)
    data_path = Path(args.data_file).resolve()
    data_sha256 = _sha256_file(data_path)
    tokenizer_metadata = {
        "class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "special_tokens_map": _jsonable(getattr(tokenizer, "special_tokens_map", {})),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    source_rows = read_jsonl(data_path)
    max_samples = args.max_samples
    if args.smoke and max_samples == 0:
        max_samples = 2
    samples, excluded = _prepare_samples(
        source_rows,
        tokenizer,
        max_input_tokens=args.max_input_tokens,
        max_total_tokens=args.max_model_len - args.max_new_tokens,
        max_samples=max_samples,
    )

    common_config = {
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "temperature": 0.0,
        "seed": args.seed,
        "batch_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "warmup_per_sample": True,
    }
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a single path component")
    output_dir = Path(args.output_dir).expanduser().resolve() / run_id
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"run output path is not a directory: {output_dir}")
        existing_names = {path.name for path in output_dir.iterdir()}
        if "console.log" not in existing_names or existing_names - {"console.log"}:
            raise FileExistsError(f"run output directory already contains artifacts: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "run_report.json"
    progress_path = output_dir / "progress.json"
    partial_records_path = output_dir / "results.partial.jsonl"
    events_path = output_dir / "events.jsonl"
    samples_path = output_dir / "samples.jsonl"
    excluded_path = output_dir / "excluded_samples.jsonl"
    warmup_path = output_dir / "warmup.jsonl"
    for artifact_path in (excluded_path, warmup_path, partial_records_path):
        artifact_path.touch(exist_ok=True)
    for sample in samples:
        _append_jsonl(
            samples_path,
            {
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "input_tokens": sample["input_tokens"],
                "document_words": sample["document_words"],
                "rendered_prompt": sample["prompt"],
                "prompt_token_ids": sample["prompt_token_ids"],
                "reference": sample["reference"],
                "source_record": sample["source_record"],
            },
        )
    for excluded_sample in excluded:
        _append_jsonl(excluded_path, excluded_sample)
    _write_json(progress_path, {"run_id": run_id, "status": "running", "records_written": 0})
    _append_event(events_path, "run_started", run_started_perf, run_id=run_id, methods=methods)
    _write_json(
        report_path,
        {
            "run_id": run_id,
            "status": "running",
            "started_at_utc": run_started_utc,
            "backend": "vllm",
            "vllm_version": getattr(vllm, "__version__", "unknown"),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "transformers_version": getattr(transformers, "__version__", "unknown"),
            "runtime_environment": runtime_snapshot,
            "gpu": torch.cuda.get_device_name(0),
            "target_model": paths["vanilla_vllm"],
            "target_model_config": _jsonable(target_config.to_dict()),
            "tokenizer": tokenizer_metadata,
            "configuration": common_config,
            "data_file": str(data_path),
            "data_sha256": data_sha256,
            "console_log": str(output_dir / "console.log"),
            "artifact_paths": {
                "samples": str(samples_path),
                "excluded_samples": str(excluded_path),
                "events": str(events_path),
                "warmup": str(warmup_path),
                "partial_results": str(partial_records_path),
            },
            "methods": list(methods),
            "samples": [
                {key: sample[key] for key in ("sample_id", "dataset", "input_tokens")}
                for sample in samples
            ],
            "excluded_samples": excluded,
        },
    )

    records: list[dict[str, Any]] = []
    model_configs: dict[str, dict[str, Any]] = {}
    for method in methods:
        model_path = paths[method]
        engine = None
        method_records: list[dict[str, Any]] = []
        speculative_config = None
        speculative_tokens = None
        _append_event(
            events_path,
            "method_started",
            run_started_perf,
            method=method,
            run_order=methods.index(method) + 1,
            target_model=paths["vanilla_vllm"],
            draft_model=None if method == "vanilla_vllm" else model_path,
        )
        try:
            draft_config = None
            method_metadata: dict[str, Any] = {"draft_model": None, "vllm_method": None}
            if method != "vanilla_vllm":
                draft_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
                speculative_config, method_metadata = build_speculative_config(
                    method,
                    model_path,
                    draft_config,
                    target_num_hidden_layers=target_layers,
                )
                speculative_tokens = int(method_metadata["num_speculative_tokens"])
            if method == "domino":
                install_domino_vllm_compat()

            model_configs[method] = {
                **method_metadata,
                "target_model": paths["vanilla_vllm"],
                "status": "loading",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_memory_before_load": _gpu_memory_snapshot(torch),
                "system_gpu_before_load": _nvidia_smi_snapshot()
            }
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            engine_kwargs = {
                "model": paths["vanilla_vllm"],
                "tokenizer": paths["vanilla_vllm"],
                "dtype": args.dtype,
                "trust_remote_code": False,
                "disable_log_stats": False,
                "enforce_eager": args.enforce_eager,
                "enable_prefix_caching": False,
                "max_model_len": args.max_model_len,
                "max_num_seqs": 1,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "seed": args.seed,
                "speculative_config": speculative_config,
            }
            if speculative_config is not None:
                engine_kwargs["per_request_spec_decode_metrics"] = "detailed"
            model_configs[method]["engine_kwargs"] = _jsonable(engine_kwargs)
            _append_event(
                events_path,
                "engine_load_started",
                run_started_perf,
                method=method,
                engine_kwargs=_jsonable(engine_kwargs),
                method_metadata=method_metadata,
                draft_model_config=(
                    _jsonable(draft_config.to_dict()) if draft_config is not None else None
                ),
            )
            engine_load_started_perf = time.perf_counter()
            engine = LLM(**engine_kwargs)
            torch.cuda.synchronize()
            load_ms = (time.perf_counter() - engine_load_started_perf) * 1000.0
            model_configs[method].update(
                {
                    "load_ms": load_ms,
                    "status": "running",
                    "gpu_memory_after_load": _gpu_memory_snapshot(torch),
                    "system_gpu_after_load": _nvidia_smi_snapshot()
                }
            )
            _append_event(
                events_path,
                "engine_loaded",
                run_started_perf,
                method=method,
                load_ms=load_ms,
                gpu_memory=model_configs[method]["gpu_memory_after_load"],
                system_gpu=model_configs[method]["system_gpu_after_load"],
                speculative_config=speculative_config,
            )
            torch.cuda.reset_peak_memory_stats()
            sampling = SamplingParams(
                temperature=0.0,
                seed=args.seed,
                max_tokens=args.max_new_tokens,
                n=1,
            )
            warmup_sampling = SamplingParams(
                temperature=0.0,
                seed=args.seed,
                max_tokens=min(
                    pilot_warmup_tokens(speculative_tokens), args.max_new_tokens
                ),
                n=1,
                ignore_eos=True,
            )
            model_configs[method]["sampling_params"] = _jsonable(sampling)
            model_configs[method]["warmup_sampling_params"] = _jsonable(warmup_sampling)
            _append_event(
                events_path,
                "generation_configured",
                run_started_perf,
                method=method,
                sampling_params=_jsonable(sampling),
                warmup_sampling_params=_jsonable(warmup_sampling),
            )
            _append_event(
                events_path,
                "warmup_started",
                run_started_perf,
                method=method,
                sample_count=len(samples),
                warmup_max_tokens=warmup_sampling.max_tokens,
            )
            for sample in samples:
                warmup_started = time.perf_counter()
                warmup_output = engine.generate(
                    [{"prompt_token_ids": sample["prompt_token_ids"]}],
                    warmup_sampling,
                    use_tqdm=False,
                )[0]
                torch.cuda.synchronize()
                warmup_wall_ms = (time.perf_counter() - warmup_started) * 1000.0
                warmup_memory = _gpu_memory_snapshot(torch)
                _append_jsonl(
                    warmup_path,
                    {
                        "method": method,
                        "sample_id": sample["sample_id"],
                        "input_tokens": sample["input_tokens"],
                        "max_tokens": warmup_sampling.max_tokens,
                        "client_wall_ms": warmup_wall_ms,
                        "output_tokens": len(warmup_output.outputs[0].token_ids or []),
                        "output_token_ids": [
                            int(token) for token in (warmup_output.outputs[0].token_ids or [])
                        ],
                        "text": str(warmup_output.outputs[0].text or ""),
                        "raw_request_output": _jsonable(warmup_output),
                        "gpu_memory": warmup_memory,
                    },
                )
                _append_event(
                    events_path,
                    "warmup_finished",
                    run_started_perf,
                    method=method,
                    sample_id=sample["sample_id"],
                    client_wall_ms=warmup_wall_ms,
                    output_tokens=len(warmup_output.outputs[0].token_ids or []),
                    gpu_memory=warmup_memory,
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for index, sample in enumerate(samples, start=1):
                request_started = time.perf_counter()
                request_started_at_utc = datetime.now(timezone.utc).isoformat()
                request_output = None
                _append_event(
                    events_path,
                    "request_started",
                    run_started_perf,
                    method=method,
                    sample_id=sample["sample_id"],
                    sample_index=index,
                    input_tokens=sample["input_tokens"],
                )
                _write_json(
                    progress_path,
                    {
                        "run_id": run_id,
                        "status": "running",
                        "stage": "request_running",
                        "current_method": method,
                        "current_method_index": methods.index(method) + 1,
                        "current_sample_id": sample["sample_id"],
                        "current_sample_index": index,
                        "records_written": len(records),
                        "method_status": {
                            name: config.get("status")
                            for name, config in model_configs.items()
                        },
                    },
                )
                try:
                    torch.cuda.reset_peak_memory_stats()
                    request_output = engine.generate(
                        [{"prompt_token_ids": sample["prompt_token_ids"]}],
                        sampling,
                        use_tqdm=False,
                    )[0]
                    torch.cuda.synchronize()
                    request_wall_ms = (time.perf_counter() - request_started) * 1000.0
                    request_gpu_memory = _gpu_memory_snapshot(torch)
                    record = _request_record(
                        method=method,
                        sample=sample,
                        model=paths["vanilla_vllm"],
                        request_output=request_output,
                        tokenizer=tokenizer,
                        client_wall_ms=request_wall_ms,
                        gpu_memory=request_gpu_memory,
                    )
                except Exception as exc:
                    error_traceback = traceback.format_exc()
                    traceback.print_exc()
                    request_wall_ms = (time.perf_counter() - request_started) * 1000.0
                    request_gpu_memory = _gpu_memory_snapshot(torch)
                    record = _failed_record(
                        method,
                        sample,
                        paths["vanilla_vllm"],
                        f"{type(exc).__name__}: {exc}",
                    )
                    record["client_wall_ms"] = request_wall_ms
                    record["gpu_memory"] = request_gpu_memory
                    record["error_traceback"] = error_traceback
                    if request_output is not None:
                        record["raw_request_output"] = _jsonable(request_output)
                        request_metrics = getattr(request_output, "metrics", None)
                        record["raw_request_metrics"] = _jsonable(request_metrics)
                        record["raw_speculative_metrics"] = _jsonable(
                            getattr(request_output.outputs[0], "spec_decode_metrics", None)
                        )
                        record["speculative_decoding_metrics"] = record["raw_speculative_metrics"]
                    _append_event(
                        events_path,
                        "request_failed",
                        run_started_perf,
                        method=method,
                        sample_id=sample["sample_id"],
                        error_type=type(exc).__name__,
                        error=str(exc),
                        error_traceback=error_traceback,
                        client_wall_ms=request_wall_ms,
                    )
                record["request_started_at_utc"] = request_started_at_utc
                record["request_ended_at_utc"] = datetime.now(timezone.utc).isoformat()
                record["sample_index"] = index
                record["method_run_order"] = methods.index(method) + 1
                if method != "vanilla_vllm":
                    record["speculative_method"] = method_metadata.get("vllm_method")
                    record["num_speculative_tokens"] = speculative_tokens
                problems = validate_schema(record, spec=method != "vanilla_vllm")
                if problems:
                    record["status"] = "invalid"
                    record["schema_errors"] = problems
                _append_event(
                    events_path,
                    "request_finished",
                    run_started_perf,
                    method=method,
                    sample_id=sample["sample_id"],
                    sample_index=index,
                    status=record.get("status"),
                    output_tokens=record.get("output_tokens"),
                    tpot_ms=record.get("tpot_ms"),
                    prefill_ms=record.get("prefill_ms"),
                    e2e_ms=record.get("e2e_ms"),
                    quality_valid=record.get("quality_valid"),
                    repetition_flag=record.get("repetition_flag"),
                    speculative_metrics_available=record.get("speculative_metrics_available"),
                    acceptance_rate=record.get("acceptance_rate"),
                    acceptance_rate_percent=record.get("acceptance_rate_percent"),
                    avg_accept_length=record.get("avg_accept_length"),
                    accepted_draft_tokens=record.get("draft_tokens_accepted"),
                    proposed_draft_tokens=record.get("draft_tokens_proposed"),
                    gpu_memory=record.get("gpu_memory"),
                )
                method_records.append(record)
                records.append(record)
                _append_jsonl(partial_records_path, record)
                _write_json(
                    progress_path,
                    {
                        "run_id": run_id,
                        "status": "running",
                        "current_method": method,
                        "current_method_index": methods.index(method) + 1,
                        "current_sample_id": sample["sample_id"],
                        "current_sample_index": index,
                        "records_written": len(records),
                        "method_status": {
                            name: config.get("status")
                            for name, config in model_configs.items()
                        },
                    },
                )
                print(
                    f"[{method} {index}/{len(samples)}] {sample['sample_id']} "
                    f"status={record['status']} output_tokens={record['output_tokens']} "
                    f"tpot_ms={record.get('tpot_ms')}",
                    flush=True,
                )
            model_configs[method]["status"] = (
                "success"
                if len(method_records) == len(samples)
                and all(row.get("status") == "success" for row in method_records)
                else "invalid"
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
            traceback.print_exc()
            model_configs.setdefault(method, {}).update(
                {
                    "status": "failed",
                    "error": error,
                    "error_traceback": error_traceback,
                }
            )
            _append_event(
                events_path,
                "method_failed",
                run_started_perf,
                method=method,
                error_type=type(exc).__name__,
                error=error,
                error_traceback=error_traceback,
            )
            existing_ids = {row["sample_id"] for row in method_records}
            for sample in samples:
                if sample["sample_id"] in existing_ids:
                    continue
                record = _failed_record(method, sample, paths["vanilla_vllm"], error)
                record["error"] = error
                record["error_traceback"] = error_traceback
                record["method_run_order"] = methods.index(method) + 1
                if method != "vanilla_vllm":
                    record["speculative_method"] = "dflash" if method == "domino" else method
                    record["num_speculative_tokens"] = speculative_tokens
                method_records.append(record)
                records.append(record)
                _append_jsonl(partial_records_path, record)
        finally:
            try:
                torch.cuda.synchronize()
                method_memory = _gpu_memory_snapshot(torch)
                system_method_memory = _nvidia_smi_snapshot()
                model_configs.setdefault(method, {})["gpu_memory_after_method"] = method_memory
                model_configs.setdefault(method, {})["system_gpu_after_method"] = system_method_memory
                _append_event(
                    events_path,
                    "method_finished",
                    run_started_perf,
                    method=method,
                    status=model_configs.get(method, {}).get("status"),
                    recorded_samples=len(method_records),
                    gpu_memory=method_memory,
                    system_gpu=system_method_memory,
                )
            except Exception as memory_exc:
                memory_traceback = traceback.format_exc()
                traceback.print_exc()
                _append_event(
                    events_path,
                    "gpu_memory_snapshot_failed",
                    run_started_perf,
                    method=method,
                    error=f"{type(memory_exc).__name__}: {memory_exc}",
                    error_traceback=memory_traceback,
                )
            _shutdown_engine(engine)
            del engine
            gc = __import__("gc")
            gc.collect()
            torch.cuda.empty_cache()
        _write_json(
            progress_path,
            {
                "run_id": run_id,
                "status": "running",
                "completed_methods": methods[: methods.index(method) + 1],
                "records_written": len(records),
                "method_status": {name: config.get("status") for name, config in model_configs.items()},
            },
        )

    parity = _annotate_parity(records)
    metric_summary = _method_metrics(records, methods)
    execution_complete = len(records) == len(methods) * len(samples)
    correctness_pass = all(
        parity[method]["compared_samples"] == len(samples)
        and parity[method]["exact_matches"] == len(samples)
        for method in methods
    )
    execution_pass = all(row.get("status") == "success" for row in records)
    quality_pass = all(bool(row.get("quality_valid")) for row in records)
    evaluation_finished_at_utc = datetime.now(timezone.utc).isoformat()
    evaluation_runtime_ms = round((time.perf_counter() - run_started_perf) * 1000.0, 3)
    _append_event(
        events_path,
        "metrics_aggregated",
        run_started_perf,
        execution_complete=execution_complete,
        execution_pass=execution_pass,
        quality_pass=quality_pass,
        correctness_pass=correctness_pass,
        method_metrics=metric_summary,
    )
    summary = {
        "record_type": "summary",
        "run_id": run_id,
        "started_at_utc": run_started_utc,
        "evaluation_finished_at_utc": evaluation_finished_at_utc,
        "evaluation_runtime_ms": evaluation_runtime_ms,
        "runtime_environment": runtime_snapshot,
        "data_sha256": data_sha256,
        "tokenizer": tokenizer_metadata,
        "target_model_config": _jsonable(target_config.to_dict()),
        "artifact_paths": {
            "console_log": str(output_dir / "console.log"),
            "samples": str(samples_path),
            "excluded_samples": str(excluded_path),
            "warmup": str(warmup_path),
            "events": str(events_path),
            "results": str(output_dir / "results.jsonl"),
            "partial_results": str(partial_records_path),
            "run_report": str(report_path),
            "markdown_report": str(output_dir / "report_vi.md"),
        },
        "status": "completed" if execution_complete else "partial",
        "backend": "vllm",
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "gpu": torch.cuda.get_device_name(0),
        "target_model": paths["vanilla_vllm"],
        "configuration": common_config,
        "methods": list(methods),
        "sample_count": len(samples),
        "sample_manifest": [
            {key: sample[key] for key in ("sample_id", "dataset", "input_tokens")}
            for sample in samples
        ],
        "excluded_sample_count": len(excluded),
        "excluded_samples": excluded,
        "data_file": str(Path(args.data_file).resolve()),
        "execution_complete": execution_complete,
        "correctness_pass": correctness_pass,
        "execution_pass": execution_pass,
        "quality_pass": quality_pass,
        "parity": parity,
        "method_metrics": metric_summary,
        "model_configs": model_configs,
        "metric_definitions": {
            "tpot_ms": "(last_token_ts - first_token_ts) / (output_tokens - 1)",
            "dsr": "mean TPOT of vanilla_vllm / mean TPOT of method",
            "esr": "(mean vanilla prefill + vanilla TPOT * paired min output length) / (mean vanilla prefill + method TPOT * paired min output length)",
            "token_lcs_overlap_with_vanilla": "LCS(output token IDs, vanilla token IDs) / vanilla output token count",
        },
    }
    output_dir = Path(args.output_dir).expanduser().resolve() / run_id
    writer = JsonlWriter(output_dir / "results.jsonl")
    for record in records:
        writer.add(record)
    writer.finalize(summary)
    _write_json(output_dir / "run_report.json", summary)
    _write_markdown_report(output_dir / "report_vi.md", summary, records)
    _append_event(
        events_path,
        "artifacts_written",
        run_started_perf,
        run_id=run_id,
        results=str(output_dir / "results.jsonl"),
        run_report=str(report_path),
        markdown_report=str(output_dir / "report_vi.md"),
    )
    _write_json(
        progress_path,
        {
            "run_id": run_id,
            "status": summary["status"],
            "records_written": len(records),
            "execution_pass": execution_pass,
            "quality_pass": quality_pass,
            "correctness_pass": correctness_pass,
        },
    )
    partial_records_path.unlink(missing_ok=True)
    print(f"Kết quả: {output_dir}", flush=True)
    print(json.dumps(metric_summary, ensure_ascii=False, indent=2), flush=True)
    if not execution_complete or not execution_pass or not quality_pass:
        return 1
    return 0


def _write_markdown_report(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = [
        f"# Báo cáo eval vLLM đồng bộ — {summary['run_id']}",
        "",
        f"- Trạng thái chạy: {summary['status']}",
        f"- GPU: {summary['gpu']}",
        f"- vLLM: {summary['vllm_version']}; PyTorch: {summary['torch_version']}",
        f"- Số mẫu chung: {summary['sample_count']}",
        f"- Parity greedy toàn bộ: {summary['correctness_pass']}",
        "",
        "| Method | Sinh thành công | Hợp lệ theo guard | Lặp | ROUGE-L | TPOT (ms/token) | DSR | ESR | Acceptance (%) | Accept length | Token khớp vanilla | LCS với vanilla |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in summary["method_metrics"].items():
        parity = summary["parity"][method]
        paired = metrics.get("paired_speed_metrics") or {}
        exact = f"{parity['exact_matches']}/{parity['compared_samples']}"
        overlap = parity.get("token_lcs_overlap")

        def fmt(value: Any) -> str:
            return "—" if value is None else f"{float(value):.3f}"

        overlap_text = "—" if overlap is None else f"{overlap * 100:.2f}%"
        lines.append(
            f"| {method} | {metrics['successful_outputs']}/{metrics['samples']} | "
            f"{metrics['quality_valid_outputs']}/{metrics['samples']} | {metrics['repetition_flags']} | "
            f"{fmt(metrics['mean_rougeL'])} | {fmt(metrics['mean_tpot_ms'])} | "
            f"{fmt(paired.get('dsr'))} | {fmt(paired.get('esr'))} | "
            f"{fmt(metrics.get('mean_acceptance_rate_percent'))} | "
            f"{fmt(metrics.get('mean_avg_accept_length'))} | {exact} | {overlap_text} |"
        )
    lines.extend(
        [
            "",
            "Acceptance và vLLM decode traces được giữ theo từng request trong results.jsonl "
            "dưới speculative_decoding_metrics/raw_speculative_metrics; các giá trị tổng hợp "
            "trong bảng chỉ dùng những request có counter hoặc rate hợp lệ.",
            "",
            "Kết quả là đo cho checkpoint, cấu hình, GPU và phiên bản vLLM ghi ở trên. "
            "correctness_pass=false nghĩa là có method không khớp greedy vanilla; không dùng "
            "speedup của method đó như kết quả đã qua kiểm chứng.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Eval vanilla, EAGLE3, DFlash, Domino, DSpark with one local vLLM runtime."
    )
    parser.add_argument("--model", required=True, help="Local target model directory")
    parser.add_argument("--data-file", required=True, help="Canonical JSONL input file")
    parser.add_argument("--output-dir", required=True, help="Root directory for unique run artifacts")
    parser.add_argument("--eagle3-model")
    parser.add_argument("--dflash-model")
    parser.add_argument("--domino-model")
    parser.add_argument("--dspark-model")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means every eligible input row")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--smoke", action="store_true", help="Limit to two common samples")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_samples < 0 or args.max_new_tokens <= 1 or args.max_input_tokens < 0:
        raise SystemExit("max-samples/input-tokens must be non-negative; max-new-tokens must exceed 1")
    if args.max_model_len <= args.max_new_tokens:
        raise SystemExit("max-model-len must exceed max-new-tokens")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise SystemExit("gpu-memory-utilization must be between 0 and 1")
    methods = order_methods(args.methods)
    paths = _model_paths(args, methods)
    if not Path(args.data_file).is_file():
        raise SystemExit(f"evaluation data file not found: {args.data_file}")
    if args.preflight_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        import torch
        import transformers
        import vllm
        from transformers import AutoConfig, AutoTokenizer

        target_config = AutoConfig.from_pretrained(
            paths["vanilla_vllm"], local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            paths["vanilla_vllm"], local_files_only=True
        )
        target_layers = int(getattr(target_config, "num_hidden_layers", 0) or 0)
        draft_checks = {}
        for method in methods:
            if method == "vanilla_vllm":
                continue
            draft_config = AutoConfig.from_pretrained(paths[method], local_files_only=True)
            _, metadata = build_speculative_config(
                method,
                paths[method],
                draft_config,
                target_num_hidden_layers=target_layers,
            )
            draft_checks[method] = metadata
        if "domino" in methods:
            install_domino_vllm_compat()

        cuda_available = bool(torch.cuda.is_available())
        status = "preflight_passed" if cuda_available else "preflight_failed"
        print(
            json.dumps(
                {
                    "status": status,
                    "python": sys.executable,
                    "torch": getattr(torch, "__version__", "unknown"),
                    "vllm": getattr(vllm, "__version__", "unknown"),
                    "transformers": getattr(transformers, "__version__", "unknown"),
                    "cuda_available": cuda_available,
                    "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
                    "target_model": paths["vanilla_vllm"],
                    "tokenizer_class": type(tokenizer).__name__,
                    "target_num_hidden_layers": target_layers,
                    "draft_checks": draft_checks,
                    "methods": methods,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if cuda_available else 1
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
