#!/usr/bin/env python3
"""Run an official SGLang speculative backend over the common JSONL contract.

The speculative model implementation remains in SGLang/its upstream adapter;
this file only launches the official server, submits Vietnamese prompts, and
normalizes response timing metadata into the repository schema.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from common import io_util, metrics, rouge
from common.benchmark_runtime import build_sample_record, runtime_metadata
from common.data_loader import load_records
from common.input_utils import truncate_input_ids
from common.reproducibility import seed_everything


def _algorithm(method: str) -> str:
    return {
        "domino": os.environ.get("LONG_BENCH_DOMINO_ALGORITHM", "DFLASH"),
        "dflash": os.environ.get("LONG_BENCH_DFLASH_ALGORITHM", "DFLASH"),
        "dspark": os.environ.get("LONG_BENCH_DSPARK_ALGORITHM", "DSPARK"),
    }[method]


def resolve_batch_size(requested: str | int, *, total_memory_gb: float | None = None) -> int:
    """Resolve a conservative server concurrency without loading model weights."""

    if str(requested).lower() != "auto":
        value = int(requested)
        if value <= 0:
            raise ValueError("batch size must be positive")
        return value
    if total_memory_gb is None:
        try:
            import torch

            if torch.cuda.is_available():
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            total_memory_gb = None
    if total_memory_gb is None:
        return int(os.environ.get("LONG_BENCH_AUTO_BATCH_SIZE", "1"))
    if total_memory_gb >= 160:
        default = 8
    elif total_memory_gb >= 70:
        default = 4
    elif total_memory_gb >= 30:
        default = 2
    else:
        default = 1
    return max(1, int(os.environ.get("LONG_BENCH_AUTO_BATCH_SIZE", str(default))))


def build_server_args(
    *,
    method: str,
    model: str,
    draft_model: str,
    port: int,
    batch_size: int,
    tp_size: int,
    mem_fraction_static: float,
    attention_backend: str | None = None,
) -> list[str]:
    """Build only official SGLang CLI flags; no algorithm is reimplemented."""

    if method not in {"domino", "dflash", "dspark"}:
        raise ValueError(f"unsupported SGLang speculative method: {method}")
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--trust-remote-code",
        "--dtype",
        os.environ.get("LONG_BENCH_DTYPE", "bfloat16"),
    ]
    if attention_backend:
        command.extend(["--attention-backend", attention_backend])
    command.extend([
        "--tp-size",
        str(tp_size),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--max-running-requests",
        str(batch_size),
        "--cuda-graph-bs",
        *[str(index) for index in range(1, batch_size + 1)],
        "--cuda-graph-max-bs",
        str(batch_size),
        "--port",
        str(port),
        "--speculative-algorithm",
        _algorithm(method),
        "--speculative-draft-model-path",
        draft_model,
    ])
    return command


def extract_response_metrics(payload: dict[str, Any], *, request_elapsed_ms: float) -> dict[str, Any]:
    """Extract raw SGLang metadata without inventing unavailable timings."""

    meta = payload.get("meta_info") or {}

    def duration_ms(*keys: str) -> float | None:
        value = None
        selected_key = None
        for key in keys:
            if meta.get(key) is not None:
                value = meta[key]
                selected_key = key
                break
        if value is None:
            return None
        multiplier = 1.0 if str(selected_key).endswith("_ms") else 1000.0
        return round(float(value) * multiplier, 3)

    return {
        "input_tokens": int(meta["prompt_tokens"]) if meta.get("prompt_tokens") is not None else None,
        "output_tokens": int(meta["completion_tokens"]) if meta.get("completion_tokens") is not None else None,
        "queue_wait_ms": duration_ms("queue_wait_ms", "queue_wait_time", "queue_time"),
        "batch_wait_ms": duration_ms("batch_wait_ms", "batch_wait_time", "batch_time"),
        "prefill_ms": duration_ms("prompt_latency"),
        "ttft_ms": duration_ms("prompt_latency"),
        "decode_ms": duration_ms("completion_latency"),
        "draft_latency_ms": duration_ms("spec_draft_time", "draft_latency"),
        "verification_latency_ms": duration_ms("spec_verify_time", "verification_latency"),
        "server_reported_e2e_ms": duration_ms("e2e_latency", "request_time"),
        "e2e_ms": round(float(request_elapsed_ms), 3),
        "avg_accept_length": meta.get("spec_accept_length"),
        "acceptance_rate": meta.get("spec_acceptance_rate"),
        "verification_steps": meta.get("spec_verify_ct"),
        "rejected_draft_ratio": meta.get("spec_rejected_draft_ratio"),
    }


def _http_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if isinstance(value, list):
        if len(value) != 1:
            raise RuntimeError(f"unexpected SGLang response list length: {len(value)}")
        value = value[0]
    if not isinstance(value, dict):
        raise RuntimeError("SGLang response must be a JSON object")
    return value


def _wait_ready(base_url: str, process: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not become ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {process.returncode}")
        try:
            with urlrequest.urlopen(base_url + "/health", timeout=2) as response:
                if response.status < 400:
                    return
        except (OSError, urlerror.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(last_error)


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    """Stop the SGLang server and workers started in its process group."""

    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        else:
            process.kill()
        process.wait()


def _load_request_tokenizer(model: str, *, local_files_only: bool):
    """Load the target tokenizer only when an input-token cap is requested."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)


def _prepare_prompt(prompt: str, tokenizer: Any | None, max_input_tokens: int) -> str:
    """Apply the shared head+tail input cap before SGLang tokenizes the text."""

    if tokenizer is None or max_input_tokens <= 0:
        return prompt
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded.input_ids
    if input_ids.shape[1] <= max_input_tokens:
        return prompt
    trimmed = truncate_input_ids(input_ids, max_input_tokens)
    return tokenizer.decode(
        trimmed[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _request_one(
    base_url: str,
    sample: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    payload = _http_json(
        base_url + "/generate",
        {
            "text": _prepare_prompt(sample["prompt"], tokenizer, args.max_input_tokens),
            "sampling_params": {
                "temperature": args.temperature,
                "top_p": 1.0,
                "max_new_tokens": args.max_new_tokens,
            },
        },
        timeout=args.request_timeout,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    metrics = extract_response_metrics(payload, request_elapsed_ms=elapsed_ms)
    return sample, {"payload": payload, "metrics": metrics}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["domino", "dflash", "dspark"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", default=os.environ.get("LONG_BENCH_BATCH_SIZE", "auto"))
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)
    parser.add_argument(
        "--attention-backend",
        default=os.environ.get("LONG_BENCH_SGLANG_ATTENTION_BACKEND", "flashinfer"),
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LONG_BENCH_LOCAL_FILES_ONLY", "1") == "1",
    )
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--port", type=int, default=int(os.environ.get("LONG_BENCH_SGLANG_PORT", "30000")))
    parser.add_argument("--server-timeout", type=float, default=float(os.environ.get("LONG_BENCH_SERVER_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--request-timeout", type=float, default=float(os.environ.get("LONG_BENCH_REQUEST_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--run-id", default=os.environ.get("LONG_BENCH_RUN_ID"))
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.smoke:
        args.max_samples = 1
        args.max_new_tokens = min(args.max_new_tokens, 8)
    args.batch_size = resolve_batch_size(args.batch_size)
    if args.batch_size <= 0 or args.tp_size <= 0:
        raise SystemExit("batch size and tp size must be positive")
    if args.max_running_requests is None:
        args.max_running_requests = args.batch_size
    seed_everything(args.seed)
    records = load_records(Path(args.data_file), args.max_samples)
    request_tokenizer = None
    if args.max_input_tokens > 0:
        request_tokenizer = _load_request_tokenizer(
            args.model,
            local_files_only=args.local_files_only,
        )
    server_url = f"http://127.0.0.1:{args.port}"
    command = build_server_args(
        method=args.method,
        model=args.model,
        draft_model=args.draft_model,
        port=args.port,
        batch_size=args.max_running_requests,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        attention_backend=args.attention_backend,
    )
    print(f"[{args.method}] launching official SGLang: {' '.join(command)}", flush=True)
    server_start = time.perf_counter()
    process = subprocess.Popen(command, stdout=None, stderr=None, start_new_session=True)
    try:
        _wait_ready(server_url, process, args.server_timeout)
        server_startup_ms = round((time.perf_counter() - server_start) * 1000.0, 3)
        writer = io_util.JsonlWriter(Path(args.output))
        successful = 0
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            futures = [
                executor.submit(
                    _request_one,
                    server_url,
                    sample,
                    args,
                    request_tokenizer,
                )
                for sample in records
            ]
            for future in as_completed(futures):
                sample, result = future.result()
                payload = result["payload"]
                timing = result["metrics"]
                timing["server_startup_ms"] = server_startup_ms
                text = str(payload.get("text") or payload.get("output") or "")
                input_tokens = int(timing.get("input_tokens") or 0)
                output_tokens = int(timing.get("output_tokens") or 0)
                record = build_sample_record(
                    method=args.method,
                    dataset=sample.get("raw", {}).get("dataset", Path(args.data_file).stem),
                    sample_id=sample["id"],
                    model=args.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    timing=timing,
                    config={
                        "device": "cuda",
                        "dtype": os.environ.get("LONG_BENCH_DTYPE", "bfloat16"),
                        "attention_backend": args.attention_backend,
                        "seed": args.seed,
                        "temperature": args.temperature,
                        "max_new_tokens": args.max_new_tokens,
                        "batch_size": args.batch_size,
                        "measurement_scope": "full_e2e",
                        "extra_metrics": {
                            "server_startup_ms": server_startup_ms,
                            "tp_size": args.tp_size,
                            "max_running_requests": args.max_running_requests,
                            "verification_steps": timing.get("verification_steps"),
                            "server_reported_e2e_ms": timing.get("server_reported_e2e_ms"),
                        },
                    },
                    text=text,
                    reference_output=sample.get("reference"),
                )
                rouge.add_rouge(record, text, sample.get("reference"))
                metrics.add_semantic(record, text, sample.get("reference"))
                record["run_id"] = args.run_id
                record["raw_response_meta"] = payload.get("meta_info", {})
                writer.add(record)
                successful += 1
        records_written = list(writer.records)
        summary = {
            "type": "summary",
            "method": args.method,
            "dataset": Path(args.data_file).stem.replace("_100", ""),
            "run_id": args.run_id,
            "status": "success" if successful == len(records) else "failed",
            "num_samples": len(records),
            "successful_samples": successful,
            "model": args.model,
            "batch_size": args.batch_size,
            "tp_size": args.tp_size,
            "server_startup_ms": server_startup_ms,
            "runtime": runtime_metadata(),
            **rouge.aggregate_rouge(records_written),
            **metrics.aggregate_semantic(records_written),
        }
        writer.finalize(summary)
        return 0 if successful == len(records) else 1
    finally:
        _stop_process_group(process)


if __name__ == "__main__":
    raise SystemExit(main())
