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
import math
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from Benchmark.common import io_util, metrics, rouge
from Benchmark.common.benchmark_runtime import (
    build_sample_record,
    runtime_metadata,
)
from Benchmark.common.paired_reference import build_v2_record_fields, read_jsonl, token_ids_sha256
from Benchmark.common.data_loader import load_records
from Benchmark.common.input_utils import truncate_input_ids
from Benchmark.common.prompt_format import format_chat_prompt
from Benchmark.common.reproducibility import seed_everything


def _algorithm(method: str) -> str:
    return {
        "domino": os.environ.get("LONG_BENCH_DOMINO_ALGORITHM", "DFLASH"),
        "dflash": os.environ.get("LONG_BENCH_DFLASH_ALGORITHM", "DFLASH"),
        "dspark": os.environ.get("LONG_BENCH_DSPARK_ALGORITHM", "DSPARK"),
    }[method]


def resolve_stop_token_ids(tokenizer: Any) -> list[int]:
    """Return the target tokenizer's EOS ids for explicit SGLang stopping."""

    eos_ids = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_ids, int) and not isinstance(eos_ids, bool):
        values = [eos_ids]
    elif isinstance(eos_ids, (list, tuple)):
        values = list(eos_ids)
    else:
        values = []
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid tokenizer EOS token id: {value!r}")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("target tokenizer does not expose a valid EOS token id")
    return normalized


def _normalize_decoded_text(text: str) -> str:
    """Normalize only Unicode composition and whitespace for decode parity."""

    return " ".join(unicodedata.normalize("NFKC", text).split())


def response_text_token_integrity(
    payload: dict[str, Any], tokenizer: Any | None
) -> dict[str, Any]:
    """Compare SGLang's visible text with decoding its returned output IDs.

    This catches stream assembly bugs that duplicate text while token counts and
    server timing still look plausible. ``None`` means the check was not
    possible because the server omitted IDs or no tokenizer was available.
    """

    output_ids = payload.get("output_ids")
    if not isinstance(output_ids, (list, tuple)) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in output_ids
    ):
        return {
            "text_decode_matches_token_ids": None,
            "output_token_id_count": None,
        }

    result: dict[str, Any] = {
        "text_decode_matches_token_ids": None,
        "output_token_id_count": len(output_ids),
    }
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return result

    response_text = payload.get("text")
    if not isinstance(response_text, str):
        return result
    try:
        decoded_text = tokenizer.decode(
            list(output_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return result
    if not isinstance(decoded_text, str):
        return result

    result["text_decode_matches_token_ids"] = (
        _normalize_decoded_text(response_text)
        == _normalize_decoded_text(decoded_text)
    )
    return result


def build_sampling_params(
    *, temperature: float, max_new_tokens: int, stop_token_ids: list[int]
) -> dict[str, Any]:
    """Build deterministic generation parameters with target EOS stopping."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not stop_token_ids:
        raise ValueError("stop_token_ids must include the target EOS token")
    return {
        "temperature": float(temperature),
        "top_p": 1.0,
        "max_new_tokens": int(max_new_tokens),
        "stop_token_ids": list(dict.fromkeys(int(value) for value in stop_token_ids)),
        "ignore_eos": False,
    }


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
    random_seed: int = 42,
    disable_radix_cache: bool = False,
) -> list[str]:
    """Build only official SGLang CLI flags; no algorithm is reimplemented."""

    if method not in {"target_only", "domino", "dflash", "dspark"}:
        raise ValueError(f"unsupported SGLang speculative method: {method}")
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--enable-metrics",
        "--model-path",
        model,
        "--trust-remote-code",
        "--dtype",
        os.environ.get("LONG_BENCH_DTYPE", "bfloat16"),
        "--random-seed",
        str(random_seed),
    ]
    if attention_backend:
        command.extend(["--attention-backend", attention_backend])
    if disable_radix_cache:
        command.append("--disable-radix-cache")
    graph_batch_sizes = [str(index) for index in range(1, batch_size + 1)]
    command.extend([
        "--tp-size",
        str(tp_size),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--max-running-requests",
        str(batch_size),
        "--cuda-graph-bs-decode",
        *graph_batch_sizes,
        "--cuda-graph-max-bs-decode",
        str(batch_size),
        "--port",
        str(port),
    ])
    if method != "target_only":
        if not draft_model:
            raise ValueError(f"{method} requires a speculative draft model")
        command.extend([
            "--speculative-algorithm",
            _algorithm(method),
            "--speculative-draft-model-path",
            draft_model,
        ])
    return command


def extract_response_metrics(
    payload: dict[str, Any],
    *,
    request_elapsed_ms: float,
    stream_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract SGLang metadata and normalize acceptance counters."""

    from Benchmark.common.speculative_metrics import normalize_speculative_acceptance

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

    prefill_ms = duration_ms("prompt_latency")
    decode_ms = duration_ms("completion_latency")
    completion_tokens = meta.get("completion_tokens")
    strict_decode_ms = None
    strict_decode_tokens = None
    strict_decode_verified = False
    if completion_tokens is not None:
        try:
            strict_decode_tokens = max(int(completion_tokens) - 1, 0)
        except (TypeError, ValueError, OverflowError):
            strict_decode_tokens = None
        # decode_throughput is a derived rate, not timing evidence. The
        # strict phase is reportable only when SGLang returns its direct
        # completion_latency interval (first committed token -> final token).
        stream_evidence_ok = stream_timing is None
        if stream_timing is not None:
            stream_duration = stream_timing.get("decode_ms")
            try:
                stream_duration = float(stream_duration)
            except (TypeError, ValueError, OverflowError):
                stream_duration = 0.0
            stream_evidence_ok = (
                int(stream_timing.get("output_chunks") or 0) >= 2
                and stream_timing.get("decode_phase_verified") is True
                and math.isfinite(stream_duration)
                and stream_duration > 0.0
                and meta.get("completion_latency_source")
                == "sglang_api_server_monotonic_first_token_to_finished"
            )
        if (
            strict_decode_tokens is not None
            and strict_decode_tokens > 0
            and decode_ms is not None
            and math.isfinite(decode_ms)
            and decode_ms > 0
            and stream_evidence_ok
        ):
            strict_decode_ms = decode_ms
            strict_decode_verified = True
    acceptance_histogram = meta.get(
        "spec_accept_histogram",
        meta.get("spec_correct_drafts_histogram"),
    )
    if isinstance(acceptance_histogram, (list, tuple)):
        acceptance_histogram = [int(value) for value in acceptance_histogram]
    else:
        acceptance_histogram = None

    accepted = meta.get(
        "spec_num_correct_drafts", meta.get("spec_accepted_drafts")
    )
    proposed = meta.get(
        "spec_num_proposed_drafts", meta.get("spec_proposed_drafts")
    )
    runtime_acceptance_rate = meta.get(
        "spec_acceptance_rate", meta.get("spec_accept_rate")
    )
    acceptance = normalize_speculative_acceptance(
        verification_steps=meta.get("spec_verify_ct"),
        draft_tokens_accepted=accepted,
        draft_tokens_proposed=proposed,
        fallback_acceptance_rate=runtime_acceptance_rate,
        fallback_avg_accept_length=meta.get("spec_accept_length"),
    )

    return {
        "input_tokens": int(meta["prompt_tokens"])
        if meta.get("prompt_tokens") is not None
        else None,
        "output_tokens": int(meta["completion_tokens"])
        if meta.get("completion_tokens") is not None
        else None,
        "queue_wait_ms": duration_ms(
            "queue_wait_ms", "queue_wait_time", "queue_time"
        ),
        "batch_wait_ms": duration_ms(
            "batch_wait_ms", "batch_wait_time", "batch_time"
        ),
        "prefill_ms": prefill_ms,
        "ttft_ms": prefill_ms,
        "decode_ms": decode_ms,
        "draft_latency_ms": duration_ms("spec_draft_time", "draft_latency"),
        "verification_latency_ms": duration_ms(
            "spec_verify_time", "verification_latency"
        ),
        "server_reported_e2e_ms": duration_ms(
            "e2e_latency", "request_time"
        ),
        "e2e_ms": round(float(request_elapsed_ms), 3),
        "request_wall_ms": round(float(request_elapsed_ms), 3),
        "server_reported_completion_latency_ms": decode_ms,
        "strict_decode_active_ms": strict_decode_ms,
        "strict_decode_token_count": strict_decode_tokens,
        "strict_decode_phase_definition": (
            "after_first_token_committed_to_final_token"
        ),
        "strict_decode_phase_verified": strict_decode_verified,
        "measurement_scope": (
            "e2e_plus_decode"
            if strict_decode_verified and stream_timing is not None
            else "full_e2e"
            if prefill_ms is not None and decode_ms is not None
            else "e2e_only"
        ),
        "stream_decode_client_ms": (
            stream_timing.get("decode_ms") if stream_timing is not None else None
        ),
        "stream_output_chunks": (
            stream_timing.get("output_chunks") if stream_timing is not None else None
        ),
        "strict_decode_phase_source": (
            meta.get("completion_latency_source")
            if strict_decode_verified
            else None
        ),
        "timing_source": (
            "sglang_0.5.20_api_server_monotonic_stream_phase_timer"
            if strict_decode_verified
            else "sglang_0.5.20_api_server_request_time_stats"
        ),
        # acceptance length/rate/counters use the shared normalization.
        **acceptance,
        "draft_proposal_unit": "runtime_draft_candidate",
        # Histogram bins are indexed by accepted draft-token count; preserve
        # them instead of fabricating a per-verification trace.
        "acceptance_histogram": acceptance_histogram,
    }


def _parse_sglang_sse_response(
    response: Any, *, clock: Any = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect one SGLang SSE response and directly time its output events."""

    now = clock or time.perf_counter
    final_event: dict[str, Any] | None = None
    first_output_at: float | None = None
    last_output_at: float | None = None
    output_chunks = 0
    text_parts: list[str] = []
    output_ids: list[int] = []
    previous_ids: list[int] | None = None
    ids_mode: str | None = None

    for raw_line in response:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            break
        event = json.loads(data)
        if isinstance(event, list):
            if len(event) != 1:
                raise RuntimeError(
                    f"unexpected streamed SGLang response list length: {len(event)}"
                )
            event = event[0]
        if not isinstance(event, dict):
            raise RuntimeError("SGLang stream event must be a JSON object")

        event_time = float(now())
        event_text = event.get("text")
        event_ids = event.get("output_ids")
        if not isinstance(event_ids, (list, tuple)):
            event_ids = []
        normalized_ids = [int(token) for token in event_ids]
        has_output = bool(normalized_ids) or (
            isinstance(event_text, str) and bool(event_text)
        )
        if has_output:
            if first_output_at is None:
                first_output_at = event_time
            last_output_at = event_time
            output_chunks += 1
            if isinstance(event_text, str):
                text_parts.append(event_text)
            if normalized_ids:
                if previous_ids is None:
                    output_ids = list(normalized_ids)
                elif ids_mode is None:
                    if (
                        len(normalized_ids) > len(previous_ids)
                        and normalized_ids[: len(previous_ids)] == previous_ids
                    ):
                        ids_mode = "cumulative"
                        output_ids = list(normalized_ids)
                    else:
                        ids_mode = "delta"
                        output_ids.extend(normalized_ids)
                elif ids_mode == "cumulative":
                    if normalized_ids[: len(previous_ids)] != previous_ids:
                        raise RuntimeError(
                            "SGLang cumulative streamed output IDs changed prefix"
                        )
                    output_ids = list(normalized_ids)
                else:
                    output_ids.extend(normalized_ids)
                previous_ids = list(normalized_ids)

        meta = event.get("meta_info") or {}
        if not isinstance(meta, dict):
            raise RuntimeError("SGLang stream meta_info must be a JSON object")
        finish_reason = meta.get("finish_reason", event.get("finish_reason"))
        if finish_reason is not None:
            final_event = event
            break

    if final_event is None:
        raise RuntimeError("SGLang stream ended without a final finish_reason event")

    payload = dict(final_event)
    # SGLang 0.5.20 emits deltas when incremental streaming is enabled and
    # accumulated text snapshots when it is disabled. In both modes, the final
    # finish_reason event carries ReqState.get_text(): the complete response.
    # Use that final snapshot instead of concatenating event text, which would
    # duplicate or multiply the answer while leaving token/timing metadata intact.
    final_text = final_event.get("text")
    if isinstance(final_text, str):
        payload["text"] = final_text
    elif text_parts:
        payload["text"] = "".join(text_parts)
    if output_ids:
        payload["output_ids"] = output_ids

    decode_ms = None
    if (
        output_chunks >= 2
        and first_output_at is not None
        and last_output_at is not None
        and last_output_at > first_output_at
    ):
        decode_ms = (last_output_at - first_output_at) * 1000.0
    return payload, {
        "decode_ms": round(decode_ms, 6) if decode_ms is not None else None,
        "output_chunks": output_chunks,
        "decode_phase_verified": decode_ms is not None,
    }


def _http_json_stream(
    url: str, body: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(request, timeout=timeout) as response:
        return _parse_sglang_sse_response(response)


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


def _server_process_group_exists(pgid: int) -> bool:
    """Return whether a POSIX SGLang process group still has members."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    """Stop SGLang workers even if the server's group leader already exited."""
    if os.name == "nt":
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + 20.0
    while True:
        if process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    process.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
        if not _server_process_group_exists(process.pid):
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.1, remaining))

    if _server_process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if process.poll() is None:
                process.kill()
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _load_request_tokenizer(model: str, *, local_files_only: bool):
    """Load the target tokenizer only when an input-token cap is requested."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)


def _prepare_prompt(prompt: str, tokenizer: Any | None, max_input_tokens: int) -> str:
    """Chat-frame the prompt and apply the shared token cap before serving."""

    if tokenizer is None:
        return prompt
    prompt = format_chat_prompt(tokenizer, prompt)
    if max_input_tokens <= 0:
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


def _prompt_token_ids(prompt: str, tokenizer: Any, args: argparse.Namespace) -> tuple[str, list[int]]:
    prepared = _prepare_prompt(prompt, tokenizer, args.max_input_tokens)
    encoded = tokenizer(prepared, return_tensors="pt", add_special_tokens=False)
    return prepared, [int(value) for value in encoded.input_ids[0].tolist()]


def _generation_config(args: argparse.Namespace, stop_token_ids: list[int]) -> dict[str, Any]:
    return {
        "temperature": float(args.temperature),
        "max_new_tokens": int(args.max_new_tokens),
        "seed": int(args.seed),
        "stop_token_ids": list(stop_token_ids),
    }


def load_target_only_reference(
    path: Path,
    *,
    expected_sample_ids: list[str],
    expected_identity_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load a complete, fingerprint-matching SGLang target-only sidecar."""
    rows = read_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is None:
            raise ValueError(f"target-only sidecar row has no sample_id: {path}")
        sid = str(sample_id)
        if sid in indexed:
            raise ValueError(f"duplicate sample_id {sid} in target-only sidecar: {path}")
        indexed[sid] = row
    expected = [str(value) for value in expected_sample_ids]
    missing = sorted(set(expected) - set(indexed))
    extra = sorted(set(indexed) - set(expected))
    if missing or extra:
        raise ValueError(f"target-only sidecar sample coverage mismatch: missing={missing}, extra={extra}")
    identity_fields = (
        "prompt_token_sha256", "generation_config_sha256", "hardware_fingerprint",
        "runtime_config_sha256", "target_revision", "tokenizer_revision", "gpu_count", "tp_size",
        "batch_size", "concurrency", "cache_policy", "actual_input_tokens",
    )
    for sid in expected:
        row = indexed[sid]
        if row.get("contract_version") != 2:
            raise ValueError(f"target-only sidecar {sid} is not contract_version=2")
        if row.get("status", "success") != "success":
            raise ValueError(f"target-only sidecar {sid} status is {row.get('status')!r}")
        expected_identity = expected_identity_by_id[sid]
        for key in identity_fields:
            wanted = expected_identity.get(key)
            observed = row.get(key)
            if wanted is not None and observed != wanted:
                raise ValueError(
                    f"target-only sidecar fingerprint mismatch for sample {sid}: "
                    f"{key} expected {wanted!r}, observed {observed!r}"
                )
    return indexed


def _request_one(
    base_url: str,
    sample: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: Any | None,
    stop_token_ids: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    prepared_prompt = _prepare_prompt(sample["prompt"], tokenizer, args.max_input_tokens)
    local_ids: list[int] | None = None
    if tokenizer is not None:
        encoded = tokenizer(prepared_prompt, return_tensors="pt", add_special_tokens=False)
        local_ids = [int(value) for value in encoded.input_ids[0].tolist()]
    request_body = {
        # Ensure first_token_time is captured at the first output event rather
        # than when a non-streaming response is delivered at completion.
        "stream": True,
        "sampling_params": build_sampling_params(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
        ),
    }
    if getattr(args, "paper_speedup", False) and local_ids is not None:
        # SGLang's GenerateReqInput accepts token IDs directly. This preserves
        # the exact post-template/post-truncation tokens used by the v2 hash.
        request_body["input_ids"] = local_ids
    else:
        request_body["text"] = prepared_prompt
    payload, stream_timing = _http_json_stream(
        base_url + "/generate",
        request_body,
        timeout=args.request_timeout,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    request_metrics = extract_response_metrics(
        payload,
        request_elapsed_ms=elapsed_ms,
        stream_timing=stream_timing,
    )
    if local_ids is not None:
        request_metrics["prompt_token_sha256"] = token_ids_sha256(local_ids)
        request_metrics["client_prompt_tokens"] = len(local_ids)
        server_tokens = request_metrics.get("input_tokens")
        request_metrics["prompt_token_count_match"] = (
            int(server_tokens) == len(local_ids) if server_tokens is not None else None
        )
    request_metrics.update(response_text_token_integrity(payload, tokenizer))
    request_metrics["prompt_token_ids"] = local_ids
    return sample, {"payload": payload, "metrics": request_metrics}

def _run_server_phase(
    *,
    phase_method: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    tokenizer: Any,
    stop_token_ids: list[int],
) -> dict[str, dict[str, Any]]:
    """Run one SGLang mode over a fixed sample set and key results by ID."""
    command = build_server_args(
        method=phase_method,
        model=args.model,
        draft_model=args.draft_model if phase_method != "target_only" else None,
        port=args.port,
        batch_size=args.max_running_requests,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        attention_backend=args.attention_backend,
        random_seed=args.seed,
        disable_radix_cache=bool(getattr(args, "disable_radix_cache", False)),
    )
    print(
        f"[{phase_method}] launching official SGLang: {' '.join(command)}",
        flush=True,
    )
    server_url = f"http://127.0.0.1:{args.port}"
    server_start = time.perf_counter()
    process: subprocess.Popen[Any] | None = None
    termination_requested = False
    termination_signum: int | None = None
    on_main_thread = threading.current_thread() is threading.main_thread()
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum: int, frame: Any) -> None:
        # The runner first sends SIGTERM to this adapter's process group.  The
        # SGLang server has its own session, so stop it explicitly before the
        # adapter exits (including while request futures are being awaited).
        nonlocal termination_requested, termination_signum
        if process is None:
            # Popen may be interrupted after the OS has created the server but
            # before it returns the Popen handle. Defer exit until that handle
            # is available so the server's separate process group can be
            # stopped by the normal cleanup path.
            termination_requested = True
            termination_signum = signum
            return
        _stop_process_group(process)
        raise SystemExit(128 + signum)

    if on_main_thread:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    server_env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    patch_bootstrap = str(Path(__file__).resolve().parent / "sglang_patch_site")
    existing_pythonpath = server_env.get("PYTHONPATH", "")
    server_env["PYTHONPATH"] = os.pathsep.join(
        value for value in (patch_bootstrap, source_root, existing_pythonpath) if value
    )
    server_env["FAST_INFER_SGLANG_TIMING_PATCH"] = "1"
    try:
        process = subprocess.Popen(
            command,
            stdout=None,
            stderr=None,
            start_new_session=True,
            env=server_env,
        )
        try:
            if termination_requested:
                raise SystemExit(128 + int(termination_signum or signal.SIGTERM))
            _wait_ready(server_url, process, args.server_timeout)
            server_startup_ms = round((time.perf_counter() - server_start) * 1000.0, 3)
            results: dict[str, dict[str, Any]] = {}
            if getattr(args, "paper_speedup", False) and records:
                warmup_args = argparse.Namespace(**vars(args))
                warmup_args.max_new_tokens = min(int(args.max_new_tokens), 8)
                _request_one(server_url, records[0], warmup_args, tokenizer, stop_token_ids)
            with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
                futures = [
                    executor.submit(
                        _request_one,
                        server_url,
                        sample,
                        args,
                        tokenizer,
                        stop_token_ids,
                    )
                    for sample in records
                ]
                for future in as_completed(futures):
                    sample, result = future.result()
                    sample_id = str(sample["id"])
                    if sample_id in results:
                        raise ValueError(f"duplicate sample id in benchmark input: {sample_id}")
                    result["metrics"]["server_startup_ms"] = server_startup_ms
                    results[sample_id] = result
            return results
        finally:
            _stop_process_group(process)
    finally:
        if on_main_thread:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["target_only", "domino", "dflash", "dspark"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
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
    parser.add_argument("--target-only-reference-file", default=None)
    parser.add_argument("--paper-speedup", action="store_true")
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--port", type=int, default=int(os.environ.get("LONG_BENCH_SGLANG_PORT", "30000")))
    parser.add_argument("--server-timeout", type=float, default=float(os.environ.get("LONG_BENCH_SERVER_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--request-timeout", type=float, default=float(os.environ.get("LONG_BENCH_REQUEST_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--run-id", default=os.environ.get("LONG_BENCH_RUN_ID"))
    parser.add_argument("--smoke", action="store_true")
    return parser


def _sglang_version() -> str | None:
    try:
        return importlib_metadata.version("sglang")
    except importlib_metadata.PackageNotFoundError:
        return None


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
    if args.paper_speedup:
        installed_sglang = _sglang_version()
        if installed_sglang != "0.5.20":
            raise SystemExit(f"--paper-speedup requires pinned SGLang 0.5.20 for strict decode timing; found {installed_sglang!r}")
        if args.batch_size != 1 or args.max_running_requests != 1:
            raise SystemExit("--paper-speedup requires --batch-size 1 and --max-running-requests 1")
        if args.temperature != 0:
            raise SystemExit("--paper-speedup requires greedy temperature=0")
        if not args.disable_radix_cache:
            raise SystemExit("--paper-speedup requires --disable-radix-cache")
        if args.method in {"domino", "dspark"} and not args.target_only_reference_file:
            raise SystemExit("--paper-speedup Domino/DSpark require --target-only-reference-file")
    seed_everything(args.seed)
    records = load_records(Path(args.data_file), args.max_samples)
    request_tokenizer = _load_request_tokenizer(
        args.model,
        local_files_only=args.local_files_only,
    )
    stop_token_ids = resolve_stop_token_ids(request_tokenizer)
    generation_config = _generation_config(args, stop_token_ids)
    runtime = runtime_metadata()
    hardware = {
        "gpu_name": runtime.get("gpu_name"),
        "gpu_capability": runtime.get("gpu_capability"),
        "cuda_version": runtime.get("cuda_version"),
    }
    cache_policy = (
        "no_cross_request_prefix_reuse"
        if args.disable_radix_cache
        else "runtime_default_unverified"
    )
    runtime_config = {
        "engine": "sglang-0.5.20",
        "dtype": os.environ.get("LONG_BENCH_DTYPE", "bfloat16"),
        "attention_backend": args.attention_backend,
        "mem_fraction_static": args.mem_fraction_static,
        "cache_policy": cache_policy,
        "tp_size": args.tp_size,
        "batch_size": args.batch_size,
        "concurrency": args.max_running_requests,
    }
    expected_identity_by_id: dict[str, dict[str, Any]] = {}
    for sample in records:
        sample_id = str(sample["id"])
        _, token_ids = _prompt_token_ids(sample["prompt"], request_tokenizer, args)
        fields = build_v2_record_fields(
            prompt_token_ids=token_ids,
            generation_config=generation_config,
            hardware=hardware,
            target_revision=str(args.model),
            tokenizer_revision=str(getattr(request_tokenizer, "name_or_path", args.model)),
            gpu_count=args.tp_size,
            tp_size=args.tp_size,
            batch_size=args.batch_size,
            concurrency=args.max_running_requests,
            cache_policy=cache_policy,
            runtime_config=runtime_config,
        )
        expected_identity_by_id[sample_id] = {
            key: fields.get(key) for key in (
                "prompt_token_sha256", "generation_config_sha256", "hardware_fingerprint",
                "target_revision", "tokenizer_revision", "runtime_config_sha256", "gpu_count", "tp_size",
                "batch_size", "concurrency", "cache_policy", "actual_input_tokens",
            )
        }
    needs_target_only_reference = args.method in {"domino", "dspark"}
    reference_results: dict[str, dict[str, Any]] = {}
    if needs_target_only_reference and args.target_only_reference_file:
        reference_rows = load_target_only_reference(
            Path(args.target_only_reference_file),
            expected_sample_ids=[str(sample["id"]) for sample in records],
            expected_identity_by_id=expected_identity_by_id,
        )
        for sample_id, row in reference_rows.items():
            ref_wall = row.get("request_wall_ms")
            reference_results[sample_id] = {
                "record": row,
                "payload": {
                    "text": row.get("text"),
                    "meta_info": row.get("raw_response_meta", {}),
                },
                "metrics": {
                    key: row.get(key)
                    for key in (
                        "input_tokens", "output_tokens", "prefill_ms", "ttft_ms",
                        "decode_ms", "e2e_ms", "server_startup_ms",
                        "strict_decode_active_ms", "strict_decode_token_count",
                        "strict_decode_phase_definition", "strict_decode_phase_verified",
                    )
                } | {"request_wall_ms": ref_wall, "e2e_ms": ref_wall},
            }
    elif needs_target_only_reference:
        print(f"[{args.method}] collecting paired target-only reference timings", flush=True)
        reference_results = _run_server_phase(
            phase_method="target_only",
            records=records,
            args=args,
            tokenizer=request_tokenizer,
            stop_token_ids=stop_token_ids,
        )

    method_results = _run_server_phase(
        phase_method=args.method,
        records=records,
        args=args,
        tokenizer=request_tokenizer,
        stop_token_ids=stop_token_ids,
    )
    writer = io_util.JsonlWriter(Path(args.output))
    successful = 0
    for sample in records:
        sample_id = str(sample["id"])
        result = method_results[sample_id]
        payload = result["payload"]
        timing = result["metrics"]
        prompt_token_ids = timing.get("prompt_token_ids")
        server_startup_ms = timing.get("server_startup_ms")
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
                "measurement_scope": timing.get("measurement_scope", "e2e_only"),
                "stop_token_ids": stop_token_ids,
                "extra_metrics": {
                    "server_startup_ms": server_startup_ms,
                    "tp_size": args.tp_size,
                    "max_running_requests": args.max_running_requests,
                    "verification_steps": timing.get("verification_steps"),
                    "acceptance_histogram": timing.get("acceptance_histogram"),
                    "server_reported_e2e_ms": timing.get("server_reported_e2e_ms"),
                    "stream_decode_client_ms": timing.get("stream_decode_client_ms"),
                    "stream_output_chunks": timing.get("stream_output_chunks"),
                    "strict_decode_phase_source": timing.get("strict_decode_phase_source"),
                },
            },
            text=text,
            reference_output=sample.get("reference"),
        )
        record["acceptance_histogram"] = timing.get("acceptance_histogram")
        record["draft_tokens_accepted"] = timing.get("draft_tokens_accepted")
        record["draft_tokens_proposed"] = timing.get("draft_tokens_proposed")
        record["draft_proposal_unit"] = timing.get("draft_proposal_unit")
        record["accepted_draft_tokens_per_step"] = timing.get(
            "accepted_draft_tokens_per_step"
        )
        record["acceptance_rate_percent"] = timing.get("acceptance_rate_percent")
        local_prompt_tokens = timing.get("client_prompt_tokens")
        server_prompt_tokens = timing.get("input_tokens")
        # A tokenizer called with one string returns one flat list of token IDs.
        # Indexing [0] would select one integer token and len(int) raises, which
        # used to abort target-only VLSP sidecars after generation completed.
        text_token_count = len(
            request_tokenizer(text, add_special_tokens=False).input_ids
        )
        record.update(
            build_v2_record_fields(
                prompt_token_ids=prompt_token_ids or [],
                generation_config=generation_config,
                hardware=hardware,
                request_wall_ms=timing.get("request_wall_ms"),
                native_elapsed_ms=timing.get("request_wall_ms"),
                native_timing_scope="sglang_client_request_wall",
                timed_generated_tokens=output_tokens,
                visible_output_tokens=text_token_count,
                decode_active_ms=timing.get("strict_decode_active_ms"),
                decode_token_count=timing.get("strict_decode_token_count"),
                decode_phase_definition=timing.get("strict_decode_phase_definition"),
                decode_phase_verified=timing.get("strict_decode_phase_verified") is True,
                timing_source=timing.get(
                    "timing_source",
                    "sglang_0.5.20_api_server_request_time_stats",
                ),
                target_revision=str(args.model),
                tokenizer_revision=str(getattr(request_tokenizer, "name_or_path", args.model)),
                gpu_count=args.tp_size,
                tp_size=args.tp_size,
                batch_size=args.batch_size,
                concurrency=args.max_running_requests,
                cache_policy=cache_policy,
                runtime_config=runtime_config,
            )
        )
        record["prompt_token_count_match"] = timing.get("prompt_token_count_match")
        record["text_decode_matches_token_ids"] = timing.get(
            "text_decode_matches_token_ids"
        )
        record["output_token_id_count"] = timing.get("output_token_id_count")
        record["client_prompt_tokens"] = local_prompt_tokens
        record["server_prompt_tokens"] = server_prompt_tokens
        if args.method == "target_only":
            record["native_timing_scope"] = "sglang_client_request_wall"

        if needs_target_only_reference:
            reference = reference_results[sample_id]
            reference_timing = reference["metrics"]
            reference_payload = reference["payload"]
            baseline_text = str(
                reference_payload.get("text") or reference_payload.get("output") or ""
            )
            baseline_output_tokens = reference_timing.get("output_tokens")
            record["baseline_text"] = baseline_text
            record["baseline_raw_response_meta"] = reference_payload.get(
                "meta_info", {}
            )
            record["baseline_output_tokens"] = (
                int(baseline_output_tokens)
                if baseline_output_tokens is not None
                else None
            )
            record["baseline_prefill_ms"] = reference_timing.get("prefill_ms")
            record["baseline_ttft_ms"] = reference_timing.get("ttft_ms")
            record["baseline_decode_ms"] = reference_timing.get("decode_ms")
            record["baseline_e2e_ms"] = reference_timing.get("request_wall_ms", reference_timing.get("e2e_ms"))
            record["dense_prefill_ms"] = reference_timing.get("prefill_ms")
            record["dense_ttft_ms"] = reference_timing.get("ttft_ms")
            record["dense_decode_ms"] = reference_timing.get("decode_ms")
            record["dense_e2e_ms"] = reference_timing.get("request_wall_ms", reference_timing.get("e2e_ms"))
            record["baseline_server_startup_ms"] = reference_timing.get(
                "server_startup_ms"
            )
            record["native_reference_id"] = str(args.target_only_reference_file or "inline_target_only")
            record["native_reference"] = (
                reference.get("record") if isinstance(reference.get("record"), dict) else None
            )
            record["speedup_scope"] = "paired_target_only"
            record["speedup_reference_method"] = "sglang_target_only"
            record["paired_output_exact_match"] = text == baseline_text
            record["paired_output_token_count_match"] = (
                output_tokens == int(baseline_output_tokens)
                if baseline_output_tokens is not None
                else None
            )
            record["paired_output_token_ratio"] = (
                round(output_tokens / int(baseline_output_tokens), 4)
                if baseline_output_tokens is not None
                and int(baseline_output_tokens) > 0
                else None
            )
            record["speedup_valid"] = metrics.has_valid_paired_speedup(record)

        rouge.add_rouge(record, text, sample.get("reference"))
        metrics.add_semantic(record, text, sample.get("reference"))
        record["run_id"] = args.run_id
        record["raw_response_meta"] = payload.get("meta_info", {})
        writer.add(record)
        successful += 1
        print(
            f"[sample {sample_id}] {args.method}={record.get('e2e_ms')}ms "
            + (
                f"target_only={record.get('dense_e2e_ms')}ms "
                if needs_target_only_reference
                else ""
            )
            + f"tokens={output_tokens}"
            + (
                f" exact_match={record.get('paired_output_exact_match')}"
                if needs_target_only_reference
                else ""
            ),
            flush=True,
        )

    records_written = list(writer.records)
    method_startups = [
        result["metrics"].get("server_startup_ms")
        for result in method_results.values()
        if result["metrics"].get("server_startup_ms") is not None
    ]
    reference_startups = [
        result["metrics"].get("server_startup_ms")
        for result in reference_results.values()
        if result["metrics"].get("server_startup_ms") is not None
    ]
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
        "server_startup_ms": method_startups[0] if method_startups else None,
        "target_only_server_startup_ms": (
            reference_startups[0] if reference_startups else None
        ),
        "runtime": runtime,
        "speedup": metrics.aggregate_speedup(records_written),
        "speedup_scope": (
            "paired_target_only" if needs_target_only_reference else None
        ),
        "speedup_reference_method": (
            "sglang_target_only" if needs_target_only_reference else None
        ),
        **metrics.aggregate_paired_reference_fidelity(records_written),
        **rouge.aggregate_rouge(records_written),
        **metrics.aggregate_semantic(records_written),
    }
    writer.finalize(summary)
    return 0 if successful == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
