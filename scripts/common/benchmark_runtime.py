"""Shared measurement and JSON-safe record helpers for LongBench inference."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch


TIMING_FIELDS = (
    "model_load_ms",
    "server_startup_ms",
    "queue_wait_ms",
    "batch_wait_ms",
    "selector_latency_ms",
    "prefill_ms",
    "ttft_ms",
    "draft_latency_ms",
    "verification_latency_ms",
    "decode_ms",
    "server_reported_e2e_ms",
    "tpot_ms",
    "e2e_ms",
    "throughput_tok_s",
    "decode_throughput_tok_s",
    "qps",
    "peak_memory_gb",
)


def _cuda_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def _synchronize(device: torch.device) -> None:
    if _cuda_device(device):
        torch.cuda.synchronize(device)


def runtime_metadata() -> dict[str, object]:
    """Return reproducibility metadata without loading a model."""
    metadata: dict[str, object] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_count"] = torch.cuda.device_count()
        metadata["gpu_capability"] = ".".join(
            str(part) for part in torch.cuda.get_device_capability(0)
        )
        metadata["gpu_total_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 3
        )
    else:
        metadata["gpu_name"] = None
        metadata["gpu_count"] = 0
        metadata["gpu_capability"] = None
        metadata["gpu_total_memory_gb"] = None

    try:
        import transformers

        metadata["transformers_version"] = transformers.__version__
    except Exception:
        metadata["transformers_version"] = None
    try:
        import flash_attn

        metadata["flash_attn_version"] = getattr(flash_attn, "__version__", "installed")
    except Exception:
        metadata["flash_attn_version"] = None
    return metadata


def measure_call(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    reset_peak_memory: bool = True,
) -> tuple[Any, dict[str, object]]:
    """Measure a callable with CUDA synchronization and peak-memory capture."""
    device = torch.device(device)
    if _cuda_device(device) and reset_peak_memory:
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = time.perf_counter()
    value = fn()
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_memory_gb = None
    if _cuda_device(device):
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    return value, {
        "e2e_ms": round(elapsed_ms, 3),
        "device": str(device),
        "peak_memory_gb": round(peak_memory_gb, 6)
        if peak_memory_gb is not None
        else None,
    }


def _throughput(output_tokens: int | None, e2e_ms: object) -> float | None:
    if output_tokens is None or output_tokens <= 0:
        return None
    try:
        elapsed = float(e2e_ms)
    except (TypeError, ValueError):
        return None
    if elapsed <= 0:
        return None
    return round(output_tokens / (elapsed / 1000.0), 3)


def _common_record(
    *,
    method: str,
    dataset: str,
    sample_id: object,
    model: str | None,
    status: str,
    config: Mapping[str, object] | None,
) -> dict[str, object]:
    config = config or {}
    record: dict[str, object] = {
        "method": method,
        "dataset": dataset,
        "sample_id": sample_id,
        "model": model,
        "status": status,
        "scope": "sample",
        "input_tokens": None,
        "output_tokens": None,
        "retained_tokens": None,
        "model_load_ms": None,
        "server_startup_ms": None,
        "queue_wait_ms": None,
        "batch_wait_ms": None,
        "prefill_ms": None,
        "ttft_ms": None,
        "draft_latency_ms": None,
        "verification_latency_ms": None,
        "decode_ms": None,
        "server_reported_e2e_ms": None,
        "tpot_ms": None,
        "e2e_ms": None,
        "throughput_tok_s": None,
        "decode_throughput_tok_s": None,
        "qps": None,
        "peak_memory_gb": None,
        "selector_latency_ms": None,
        "batch_size": config.get("batch_size", 1),
        "device": config.get("device"),
        "gpu_name": config.get("gpu_name"),
        "dtype": config.get("dtype"),
        "attention_backend": config.get("attention_backend"),
        "seed": config.get("seed"),
        "temperature": config.get("temperature"),
        "max_new_tokens": config.get("max_new_tokens"),
        "warmup_runs": config.get("warmup_runs", 0),
        "measurement_scope": config.get("measurement_scope", "full_e2e"),
        "text": None,
        "reference_output": None,
        "extra_metrics": dict(config.get("extra_metrics", {}) or {}),
        "avg_accept_length": None,
        "acceptance_rate": None,
        "rejected_draft_ratio": None,
        "verification_steps": None,
    }
    return record


def build_status_record(
    *,
    method: str,
    dataset: str,
    sample_id: object,
    status: str,
    reason: str,
    model: str | None = None,
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a non-success record with all performance values explicitly null."""
    record = _common_record(
        method=method,
        dataset=dataset,
        sample_id=sample_id,
        model=model,
        status=status,
        config=config,
    )
    record["reason"] = reason
    return record


def build_sample_record(
    *,
    method: str,
    dataset: str,
    sample_id: object,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    timing: Mapping[str, object],
    config: Mapping[str, object] | None = None,
    text: str | None = None,
    reference_output: str | None = None,
) -> dict[str, object]:
    """Build a successful per-sample record and derive throughput."""
    record = _common_record(
        method=method,
        dataset=dataset,
        sample_id=sample_id,
        model=model,
        status="success",
        config=config,
    )
    record["input_tokens"] = int(input_tokens)
    record["output_tokens"] = int(output_tokens)
    record["text"] = text
    record["reference_output"] = reference_output
    for key in TIMING_FIELDS:
        if key in timing:
            record[key] = timing[key]
    for key in (
        "avg_accept_length",
        "acceptance_rate",
        "rejected_draft_ratio",
        "verification_steps",
    ):
        if key in timing:
            record[key] = timing[key]
    record["throughput_tok_s"] = _throughput(
        int(output_tokens), record.get("e2e_ms")
    )
    decode_ms = record.get("decode_ms")
    record["decode_throughput_tok_s"] = _throughput(output_tokens, decode_ms)
    if decode_ms is not None and output_tokens > 0:
        record["tpot_ms"] = round(float(decode_ms) / output_tokens, 3)
    return record


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    """Append one JSON-safe record, creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, default=str)
        handle.write("\n")
