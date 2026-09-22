#!/usr/bin/env python3
"""Orchestrate the Vietnamese Qwen3-4B × 6-baseline experiment matrix.

This runner owns experiment selection, deterministic input subsets, preflight
statuses, child-process logs and a manifest.  Baseline implementations remain
in their individual scripts; the runner never silently substitutes one method
for another.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]

from Benchmark.common import io_util  # noqa: E402
from Benchmark.common.benchmark_data import (  # noqa: E402
    DATASETS,
    read_jsonl,
    select_rows,
    validate_output_dir,
)
from Benchmark.common.benchmark_runtime import (  # noqa: E402
    build_status_record,
    runtime_metadata,
)
from Benchmark.common.data_loader import normalize  # noqa: E402
from Benchmark.common.metric_audit import (  # noqa: E402
    BASELINE_MEASUREMENT_SCOPE,
    audit_output_file,
    format_audit_log,
)
from Benchmark.common.longbench_adapter import (  # noqa: E402
    BASELINES,
    DISABLED_MATRIX_BASELINES,
    SUPPORTED_BASELINES,
    baseline_config_from_env,
    build_adapter_command,
    convert_records_for_baseline,
    preflight_baseline,
)


EXTERNAL_REFERENCE_BASELINES = {"eagle3", "dflash", "domino", "dspark"}

# Baselines whose child process emits ONE aggregate record for the whole input
# instead of one record per sample. SSSD reports vLLM server totals; FAFO now
# emits per-sample records through its optional adapter sidecar. Data-parallel
# sharding remains aggregate-only for SSSD, while FAFO can be joined by sample.
AGGREGATE_ONLY_BASELINES = frozenset()


def _split(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.replace(",", " ").split()
    result: list[str] = []
    for item in value:
        result.extend(str(item).replace(",", " ").split())
    return result


def _filter_matrix_baselines(values: Sequence[str]) -> tuple[list[str], list[str]]:
    """Remove known non-comparable legacy baselines from the matrix.

    Older master env files still contain LongSpec and SSSD.  They are kept as
    standalone adapters, but must not abort or silently contaminate a current
    LongBench run: LongSpec is offline-incomplete and SSSD needs a native
    extension that is unavailable in the current server image.
    """

    disabled = set(DISABLED_MATRIX_BASELINES)
    selected: list[str] = []
    skipped: list[str] = []
    for value in values:
        if value in disabled:
            if value not in skipped:
                skipped.append(value)
            continue
        selected.append(value)
    return selected, skipped


def _reference_quality_error(path: Path) -> str | None:
    """Reject a dense reference with a known degenerate output."""

    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return f"reference is not readable JSONL: {exc}"
    bad_samples: list[Any] = []
    for row in rows:
        if row.get("type") == "summary":
            continue
        guard = row.get("output_quality_guard")
        if row.get("degenerate_repetition") or (
            isinstance(guard, Mapping) and guard.get("degenerate_repetition")
        ):
            bad_samples.append(row.get("sample_id"))
    if bad_samples:
        return f"degenerate output on samples {bad_samples[:5]}"
    return None


def _select_external_reference(
    run_dir: Path, dataset: str, baselines: Sequence[str]
) -> Path | None:
    """Select an already-computed batch-1 Vanilla record file.

    FlashAttention is preferred because it is the closest target-only
    reference for the speculative paths; Vanilla HF remains a deterministic
    fallback when FA was not requested or failed before writing output.
    """

    preferred = os.environ.get("LONG_BENCH_REFERENCE_BASELINE", "vanilla_fa")
    order = [preferred, "vanilla_fa", "vanilla_hf"]
    seen: set[str] = set()

    def usable(path: Path) -> bool:
        """Accept only a completed Vanilla output, never a status file."""

        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            return False
        return any(
            row.get("type") != "summary"
            and row.get("status", "success") == "success"
            and (row.get("e2e_ms") is not None or row.get("decode_ms") is not None)
            for row in rows
            if isinstance(row, dict)
        )

    for baseline in order:
        if baseline in seen or baseline not in baselines:
            continue
        seen.add(baseline)
        candidate = run_dir / baseline / f"{dataset}.jsonl"
        if (
            candidate.is_file()
            and candidate.stat().st_size > 0
            and usable(candidate)
            and _reference_quality_error(candidate) is None
        ):
            return candidate
    return None


def _attach_external_reference_metrics(
    path: Path,
    reference_path: Path,
    *,
    reference_baseline: str,
) -> int:
    """Join Vanilla timing onto speculative records without another infer.

    The join is by ``sample_id``.  Existing paired fields are preserved; this
    function adds explicit ``external_*`` fields and uses the common
    ``dense_*`` aliases so the collector can aggregate DSR/ESR consistently.
    """

    if not path.is_file() or not reference_path.is_file():
        return 0
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    references = {
        str(row.get("sample_id")): row
        for row in (
            json.loads(line)
            for line in reference_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("type") != "summary" and row.get("sample_id") is not None
    }

    attached = 0
    decode_pairs: list[tuple[float, float]] = []
    e2e_pairs: list[tuple[float, float]] = []
    for row in rows:
        if row.get("type") == "summary":
            continue
        reference = references.get(str(row.get("sample_id")))
        if reference is None:
            continue

        row["external_reference_baseline"] = reference_baseline
        row["speedup_scope"] = "external_reference"
        for source, target in (
            ("prefill_ms", "dense_prefill_ms"),
            ("ttft_ms", "dense_ttft_ms"),
            ("decode_ms", "dense_decode_ms"),
            ("e2e_ms", "dense_e2e_ms"),
        ):
            if row.get(target) is None and reference.get(source) is not None:
                row[target] = reference[source]
            if reference.get(source) is not None:
                row[f"external_reference_{source}"] = reference[source]

        # A throughput ratio is meaningful only when both methods generated
        # the same number of public output tokens.  FAFO can expose draft /
        # lookahead tokens and older smoke runs compared 32 FAFO tokens with
        # 8 Vanilla tokens, which produces a plausible but invalid speedup.
        method_tokens = row.get("output_tokens")
        reference_tokens = reference.get("output_tokens")
        try:
            method_tokens_int = int(method_tokens)
            reference_tokens_int = int(reference_tokens)
        except (TypeError, ValueError):
            same_output_budget = False
        else:
            same_output_budget = (
                method_tokens_int > 0
                and reference_tokens_int > 0
                and method_tokens_int == reference_tokens_int
            )
        row["speedup_valid"] = same_output_budget
        if not same_output_budget:
            row["speedup_invalid_reason"] = (
                "output_token_count_mismatch_or_missing"
            )

        spec_decode = row.get("decode_ms")
        ref_decode = reference.get("decode_ms")
        if (
            same_output_budget
            and spec_decode
            and ref_decode
            and float(spec_decode) > 0
        ):
            value = round(float(ref_decode) / float(spec_decode), 4)
            row["external_decode_speedup"] = value
            decode_pairs.append((float(ref_decode), float(spec_decode)))
        spec_e2e = row.get("e2e_ms")
        ref_e2e = reference.get("e2e_ms")
        if (
            same_output_budget
            and spec_e2e
            and ref_e2e
            and float(spec_e2e) > 0
        ):
            value = round(float(ref_e2e) / float(spec_e2e), 4)
            row["external_e2e_speedup"] = value
            e2e_pairs.append((float(ref_e2e), float(spec_e2e)))
        attached += 1

    for row in rows:
        if row.get("type") != "summary":
            continue
        row["speedup_scope"] = "external_reference"
        row["external_reference_baseline"] = reference_baseline
        if decode_pairs:
            row["external_decode_speedup"] = round(
                sum(ref for ref, _ in decode_pairs)
                / sum(spec for _, spec in decode_pairs),
                4,
            )
        if e2e_pairs:
            row["external_e2e_speedup"] = round(
                sum(ref for ref, _ in e2e_pairs)
                / sum(spec for _, spec in e2e_pairs),
                4,
            )

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return attached


def resolve_profile(
    *, mode: str, cuda_available: bool, allow_unsupported: bool = False
) -> dict[str, Any]:
    """Resolve sample limits and enforce the GPU policy for each profile."""

    if mode not in {"smoke", "representative", "full"}:
        raise SystemExit(f"invalid LongBench mode: {mode}")
    if mode in {"representative", "full"} and not cuda_available and not allow_unsupported:
        raise SystemExit(
            f"LongBench {mode} requires CUDA. Use smoke for CPU preflight or "
            "pass --allow-unsupported to record unavailable cells."
        )
    return {
        "mode": mode,
        "samples": {"smoke": 1, "representative": 20, "full": 100}[mode],
        "max_new_tokens": {"smoke": 8, "representative": 2048, "full": 2048}[mode],
        "cuda_available": bool(cuda_available),
        "allow_unsupported": bool(allow_unsupported),
    }


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number") from exc


def resolve_timeout_seconds(mode: str, cli_value: int | None = None) -> int:
    """Use a timeout that covers real long-form cells, not only smoke runs."""

    if cli_value is not None:
        if cli_value <= 0:
            raise SystemExit("--timeout-seconds must be positive")
        return cli_value
    if mode == "smoke":
        value = _env_int("LONG_BENCH_TIMEOUT_SECONDS", 900)
    elif mode == "representative":
        value = _env_int("LONG_BENCH_REPRESENTATIVE_TIMEOUT_SECONDS", 3600)
    else:
        value = _env_int("LONG_BENCH_FULL_TIMEOUT_SECONDS", 21600)
    if value <= 0:
        raise SystemExit("LongBench timeout must be positive")
    return value


def resolve_max_input_tokens(mode: str, cli_value: int | None) -> int:
    """Resolve the input cap, keeping smoke safe for quadratic attention.

    VietBench smoke is a wiring/runtime check, not a full-length quality run.
    An explicit CLI value still wins; ``0`` intentionally disables the cap for
    operators who want to override the smoke safety policy.
    """
    if cli_value is not None:
        if cli_value < 0:
            raise SystemExit("--max-input-tokens must be >= 0")
        return cli_value
    if mode == "smoke":
        value = _env_int("LONG_BENCH_SMOKE_MAX_INPUT_TOKENS", 4096)
    else:
        value = _env_int("LONG_BENCH_MAX_INPUT_TOKENS", 0)
    if value < 0:
        raise SystemExit("LONG_BENCH input-token cap must be >= 0")
    return value


def _resolve(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _effective_cuda_available() -> bool:
    """Respect both driver visibility and an explicit ``DEVICE=cpu`` policy."""
    requested = (os.environ.get("LONG_BENCH_DEVICE") or os.environ.get("FI_DEVICE") or "cuda").lower()
    if requested.startswith("cpu"):
        return False
    return _cuda_available()


def _source_manifest_hash(data_dir: Path) -> str | None:
    manifest = data_dir / "manifest.json"
    if not manifest.is_file():
        return None
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _dataset_profile_count(data_dir: Path) -> int:
    """Read the common per-dataset row count from a validated profile.

    The original profile has 200 rows per dataset, but derived profiles such
    as LongBench-100 must remain runnable without changing the source data.
    The manifest is the authority so a hand-truncated directory cannot be
    mistaken for a reproducible benchmark profile.
    """

    manifest_path = data_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = manifest["datasets"]
        values = {int(counts[dataset]["eval_records"]) for dataset in DATASETS}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Invalid Vietnamese benchmark manifest datasets: {manifest_path}"
        ) from exc
    if len(values) != 1:
        raise SystemExit(
            "LongBench profile must contain the same number of rows per dataset; "
            f"got {sorted(values)}"
        )
    count = values.pop()
    if count <= 0:
        raise SystemExit(f"LongBench profile row count must be positive, got {count}")
    return count


def _load_selected(
    data_dir: Path,
    dataset: str,
    count: int,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = data_dir / f"{dataset}_100.jsonl"
    if not path.is_file():
        raise SystemExit(f"LongBench dataset file not found: {path}")
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"LongBench dataset is empty: {path}")
    if count > len(rows):
        raise SystemExit(f"{dataset}: requested {count}, only {len(rows)} rows exist")
    if count == len(rows):
        selected = [dict(row, length_bin=row.get("length_bin")) for row in rows]
    elif count == 1:
        # A one-row smoke test should be cheap and deterministic; balanced
        # stratification is defined for the 5-bin representative/full counts.
        selected = [dict(rows[0])]
    else:
        try:
            selected = select_rows(rows, dataset=dataset, n=count, seed=seed)
        except ValueError as exc:
            raise SystemExit(
                f"{dataset}: profile count {count} cannot be selected from the "
                "canonical 5-bin layout; choose a positive multiple of 5"
            ) from exc
    normalized = [normalize(row, i) for i, row in enumerate(selected)]
    return selected, normalized


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_status_file(
    path: Path,
    *,
    baseline: str,
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    status: str,
    reason: str,
    model: str | None,
    config: Mapping[str, Any],
    run_id: str,
) -> int:
    writer = io_util.JsonlWriter(path)
    for sample in records:
        row = build_status_record(
            method=baseline,
            dataset=dataset,
            sample_id=sample["id"],
            status=status,
            reason=reason,
            model=model,
            config=config,
        )
        row.update(
            run_id=run_id,
            task_type=sample.get("raw", {}).get("task_type"),
        )
        writer.add(row)
    writer.finalize(
        {
            "type": "summary",
            "method": baseline,
            "dataset": dataset,
            "run_id": run_id,
            "status": status,
            "reason": reason,
            "preflight_only": status == "preflight_only",
            "num_samples": len(records),
            "successful_samples": 0,
        }
    )
    return len(records)


def _audit_cell_output(
    output_path: Path,
    *,
    baseline: str,
    dataset: str,
    run_dir: Path,
    expected_output_tokens: int | None,
    expected_samples: int | None = None,
) -> dict[str, Any]:
    """Audit one cell and emit a grep-friendly live log line."""

    if not output_path.is_file():
        return {}
    audit_path = run_dir / "logs" / f"{baseline}_{dataset}.metrics.json"
    try:
        summary = audit_output_file(
            output_path,
            baseline=baseline,
            dataset=dataset,
            audit_path=audit_path,
            expected_output_tokens=expected_output_tokens,
            expected_samples=expected_samples,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(
            f"[metrics-audit] {baseline}/{dataset} ERROR "
            f"could not audit {output_path}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return {"metric_audit_error": str(exc)}
    line = format_audit_log(
        baseline=baseline,
        dataset=dataset,
        summary=summary,
        audit_path=audit_path,
    )
    combined_log = run_dir / "logs" / "metrics_audit.log"
    combined_log.parent.mkdir(parents=True, exist_ok=True)
    with combined_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)
    return {
        "metric_audit_path": str(audit_path),
        "metric_audit_summary": summary,
        "metric_contract": summary.get("metric_contract"),
    }


def _safe_env(cuda_visible_devices: str | None = None) -> dict[str, str]:
    """Child environment with the shared Python path and selected GPU IDs.

    ``cuda_visible_devices`` (a physical/comma-separated id list such as ``"3"``
    or ``"0,1"``) overrides the process-wide selection and is how a
    data-parallel shard is pinned to exactly the GPU group it owns.
    """
    env = dict(os.environ)
    # Baseline output must reach the parent while inference is running.  This
    # applies to Python-based adapters and is harmless for other child tools.
    env["PYTHONUNBUFFERED"] = "1"
    source = str(ROOT / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    gpu_ids = env.get("LONG_BENCH_GPU_IDS") or env.get("FI_GPU_IDS")
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def _spawn_child(
    command: Sequence[str],
    *,
    output: Path,
    log_path: Path,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    """Start one child process while teeing its combined output to log/console.

    The log file and the reader thread are created before the child is waited
    on, so long inference runs stay observable.  Spawning and awaiting are
    split so the data-parallel path can start N batch-1 children (one per GPU
    group) and only then join them.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Create the file before spawning the child so operators can tail it as
    # soon as the cell is launched, even before its first output line.
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    try:
        popen_kwargs: dict[str, Any] = {
            "cwd": ROOT,
            "env": _safe_env(cuda_visible_devices),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "bufsize": 0,
        }
        if os.name != "nt":
            # Baseline adapters may launch wrappers, vLLM workers or other
            # subprocesses.  Put the whole child tree in its own process group
            # so a timeout cannot leave GPU work running after the cell ends.
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            list(command),
            **popen_kwargs,
        )
    except BaseException:
        log_handle.close()
        raise

    state: dict[str, str] = {"tail": ""}

    def _stream_output() -> None:
        assert proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                try:
                    chunk = os.read(proc.stdout.fileno(), 4096)
                except OSError:
                    break
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if not text:
                    continue
                state["tail"] = (state["tail"] + text)[-2000:]
                log_handle.write(text)
                log_handle.flush()
                print(f"[{log_path.stem}] {text}", end="", flush=True)
            remainder = decoder.decode(b"", final=True)
            if remainder:
                state["tail"] = (state["tail"] + remainder)[-2000:]
                log_handle.write(remainder)
                log_handle.flush()
                print(f"[{log_path.stem}] {remainder}", end="", flush=True)
        finally:
            proc.stdout.close()

    reader = threading.Thread(
        target=_stream_output,
        name=f"longbench-log-{log_path.stem}",
        daemon=True,
    )
    reader.start()
    return {
        "proc": proc,
        "reader": reader,
        "log_handle": log_handle,
        "log_path": log_path,
        "output": Path(output),
        "command": [str(part) for part in command],
        "start": time.perf_counter(),
        "state": state,
    }


def _kill_child_group(proc: subprocess.Popen[Any]) -> None:
    """Kill a child and every descendant started in its process group."""
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            # The leader may have exited between wait() and cleanup.  Fall
            # back to the direct child so this remains safe if group setup was
            # unavailable for an unusual platform/runtime.
            pass
    proc.kill()


def _cleanup_spawned_children(handles: Sequence[Mapping[str, Any]]) -> None:
    """Stop and reap children when launching a group fails part-way through."""
    for handle in handles:
        proc = handle["proc"]
        if proc.poll() is None:
            _kill_child_group(proc)
    for handle in handles:
        proc = handle["proc"]
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_child_group(proc)
            proc.wait()
        reader = handle["reader"]
        reader.join(timeout=1)
        if reader.is_alive() and proc.stdout is not None:
            proc.stdout.close()
            reader.join(timeout=1)
        log_handle = handle["log_handle"]
        log_handle.flush()
        log_handle.close()


def _await_child(
    handle: Mapping[str, Any], *, timeout_seconds: float
) -> dict[str, Any]:
    """Wait (bounded) for a spawned child and return its result record."""
    proc = handle["proc"]
    log_path = handle["log_path"]
    log_handle = handle["log_handle"]
    reader = handle["reader"]
    timed_out = False
    returncode: int | None = None
    try:
        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_child_group(proc)
            proc.wait()
            returncode = None
    finally:
        # A normal child closes stdout promptly.  On timeout, close the pipe
        # after killing the child so a descendant that inherited stdout cannot
        # keep the logging thread alive indefinitely.
        reader.join(timeout=5 if not timed_out else 1)
        if reader.is_alive() and proc.stdout is not None:
            proc.stdout.close()
            reader.join(timeout=1)
        log_handle.flush()
        log_handle.close()

    elapsed_ms = round((time.perf_counter() - handle["start"]) * 1000.0, 3)
    output = handle["output"]
    return {
        "status": "timeout" if timed_out else ("success" if returncode == 0 else "failed"),
        "returncode": returncode,
        "elapsed_ms": elapsed_ms,
        "output_exists": output.is_file(),
        "log": str(log_path),
        "log_tail": handle["state"]["tail"],
        "command": list(handle["command"]),
    }


def _run_child(
    command: Sequence[str],
    *,
    output: Path,
    log_path: Path,
    timeout_seconds: int,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    """Run one child to completion (sequential cells and the collector)."""
    handle = _spawn_child(
        command,
        output=output,
        log_path=log_path,
        cuda_visible_devices=cuda_visible_devices,
    )
    return _await_child(handle, timeout_seconds=timeout_seconds)


def _run_child_group(
    jobs: Sequence[Mapping[str, Any]],
    *,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Run independent batch-1 children concurrently on separate GPUs.

    Every job is spawned first so the GPU groups overlap, then joined against
    one shared deadline: ``timeout_seconds`` bounds the whole cell, exactly as
    it bounds a single sequential cell.  Each entry in ``jobs`` provides
    ``command``, ``output``, ``log_path`` and ``cuda_visible_devices``; the
    extra bookkeeping keys are copied onto the corresponding result.
    """
    handles: list[dict[str, Any]] = []
    try:
        for job in jobs:
            handles.append(
                _spawn_child(
                    job["command"],
                    output=job["output"],
                    log_path=job["log_path"],
                    cuda_visible_devices=job.get("cuda_visible_devices"),
                )
            )
    except BaseException:
        _cleanup_spawned_children(handles)
        raise
    deadline = time.perf_counter() + max(0.0, float(timeout_seconds))
    results: list[dict[str, Any]] = []
    for job, handle in zip(jobs, handles):
        remaining = max(0.0, deadline - time.perf_counter())
        result = _await_child(handle, timeout_seconds=remaining)
        for key in ("shard_index", "cuda_visible_devices", "sample_count", "output_path"):
            if key in job:
                result[key] = job[key]
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# VRAM budget planner: use the card fully without ever risking an OOM
# ---------------------------------------------------------------------------

def _vram_usage_by_gpu() -> dict[int, dict[str, float]] | None:
    """Live ``{gpu_index: {total_gb, free_gb, used_gb}}`` from nvidia-smi.

    Returns ``None`` when the inventory is unavailable.  The planner then keeps
    the requested concurrency instead of guessing memory it cannot observe, and
    the OOM retry stays as the only protection.
    """
    gpus = _nvidia_smi_gpus()
    if not gpus:
        return None
    usage: dict[int, dict[str, float]] = {}
    for gpu in gpus:
        total = gpu.get("total_memory_gb")
        free = gpu.get("free_memory_gb")
        if total is None or free is None:
            continue
        used = gpu.get("used_memory_gb")
        usage[int(gpu["index"])] = {
            "total_gb": float(total),
            "free_gb": float(free),
            "used_gb": float(used) if used is not None else float(total) - float(free),
        }
    return usage or None


def _group_capacity(
    group: Sequence[int],
    usage: Mapping[int, Mapping[str, float]] | None,
    *,
    usable_gb: float | None,
    child_gb: float,
) -> dict[str, Any]:
    """How many batch-1 children still fit in one GPU group.

    ``usable_gb`` is the schedulable ceiling (``--vram-budget-gb`` minus
    ``--vram-headroom-gb``), always capped by the card's *total* memory so a
    budget meant for a 180 GiB card cannot be spent on a small one.  A group
    spanning several cards is judged conservatively: the child must fit on the
    least empty card, and the budget is charged against the most occupied one.
    """
    if not usage or usable_gb is None:
        return {
            "free_gb": None,
            "other_used_gb": None,
            "usable_gb": None,
            "total_gb": None,
            "free_slots": None,
        }
    observed = [usage[index] for index in group if index in usage]
    if not observed:
        return {
            "free_gb": None,
            "other_used_gb": None,
            "usable_gb": None,
            "total_gb": None,
            "free_slots": None,
        }
    free_gb = min(entry["free_gb"] for entry in observed)
    other_used_gb = max(entry["used_gb"] for entry in observed)
    total_gb = min(entry["total_gb"] for entry in observed)
    effective_usable = min(float(usable_gb), total_gb)
    remaining = effective_usable - other_used_gb
    free_slots = int(remaining // child_gb) if child_gb > 0 else 0
    return {
        "free_gb": round(free_gb, 1),
        "other_used_gb": round(other_used_gb, 1),
        "usable_gb": round(effective_usable, 1),
        "total_gb": round(total_gb, 1),
        "free_slots": max(0, free_slots),
    }


def plan_shard_slots(
    gpu_groups: Sequence[Sequence[int]],
    *,
    processes_per_gpu: int,
    usable_gb: float | None,
    child_gb: float,
    sample_count: int,
    usage: Mapping[int, Mapping[str, float]] | None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Return one device group per concurrent batch-1 child, plus the plan.

    The returned slot list is what the cell actually launches: one entry per
    child, each entry being the GPU group that child owns.  ``processes_per_gpu``
    is the operator's upper bound and the observed free VRAM is the real one, so
    the effective concurrency is ``min(requested, what fits in the budget)``.
    """
    slots: list[list[int]] = []
    per_group: list[dict[str, Any]] = []
    for group in gpu_groups:
        capacity = _group_capacity(
            group, usage, usable_gb=usable_gb, child_gb=child_gb
        )
        if capacity["free_slots"] is None:
            allowed = int(processes_per_gpu)
        else:
            allowed = min(int(processes_per_gpu), int(capacity["free_slots"]))
        allowed = max(0, allowed)
        per_group.append(
            {
                "gpu_ids": [int(gpu) for gpu in group],
                **capacity,
                "planned_processes": allowed,
            }
        )
        slots.extend([[int(gpu) for gpu in group] for _ in range(allowed)])
    if len(slots) > sample_count:
        # Never create empty shards; drop the extra slots from the tail.
        slots = slots[:sample_count]
        for entry in per_group:
            entry["planned_processes"] = sum(
                1 for slot in slots if slot == entry["gpu_ids"]
            )
    plan = {
        "processes_per_gpu_requested": int(processes_per_gpu),
        "processes_per_gpu_planned": max(
            (entry["planned_processes"] for entry in per_group), default=0
        ),
        "usable_gb": None if usable_gb is None else round(float(usable_gb), 1),
        "child_reserve_gb": round(float(child_gb), 1),
        "nvidia_smi_available": bool(usage),
        "groups": per_group,
        "shard_slots": len(slots),
    }
    return slots, plan


def _wait_for_shard_slots(
    gpu_groups: Sequence[Sequence[int]],
    *,
    processes_per_gpu: int,
    usable_gb: float | None,
    child_gb: float,
    sample_count: int,
    wait_seconds: float,
    poll_seconds: float = 15.0,
) -> tuple[list[list[int]], dict[str, Any], float]:
    """Block until at least one shard slot fits, or the wait budget expires.

    Waiting (rather than launching into a known OOM) is what makes a long
    unattended sweep safe: a crowded card delays this cell instead of killing a
    child, and no process that someone else owns is ever touched.
    """
    if usable_gb is None:
        # Planning is disabled (budget 0): honour the requested concurrency and
        # do not shell out to nvidia-smi or wait for anything.  The OOM retry
        # remains the safety net.
        slots, plan = plan_shard_slots(
            gpu_groups,
            processes_per_gpu=processes_per_gpu,
            usable_gb=None,
            child_gb=child_gb,
            sample_count=sample_count,
            usage=None,
        )
        plan["waited_seconds"] = 0.0
        return slots, plan, 0.0

    started = time.perf_counter()
    deadline = started + max(0.0, float(wait_seconds))
    while True:
        usage = _vram_usage_by_gpu()
        slots, plan = plan_shard_slots(
            gpu_groups,
            processes_per_gpu=processes_per_gpu,
            usable_gb=usable_gb,
            child_gb=child_gb,
            sample_count=sample_count,
            usage=usage,
        )
        plan["waited_seconds"] = round(time.perf_counter() - started, 1)
        if slots or wait_seconds <= 0 or time.perf_counter() >= deadline:
            return slots, plan, plan["waited_seconds"]
        now = time.perf_counter()
        if now < deadline:
            print(
                "[vram] no slot free within budget "
                f"({plan['groups']}); retrying in "
                f"{min(poll_seconds, max(0.5, deadline - now)):.0f}s",
                flush=True,
            )
        time.sleep(min(poll_seconds, max(0.5, deadline - time.perf_counter())))


_OOM_MARKERS = (
    "out of memory",
    "outofmemoryerror",
    "cuda_error_out_of_memory",
)


def _log_shows_oom(path: Path, *, tail_bytes: int = 262144) -> bool:
    """Detect a CUDA/allocation OOM in a child log without reading all of it."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > tail_bytes:
                handle.seek(size - tail_bytes)
            blob = handle.read().decode("utf-8", errors="replace").lower()
    except OSError:
        return False
    return any(marker in blob for marker in _OOM_MARKERS)


def _replace_command_output(command: Sequence[str], new_output: Path) -> list[str] | None:
    """Point a built command's ``--output`` at ``new_output``.

    Every adapter appends ``--output <path>``; if a future adapter stops doing
    that the retry is skipped instead of silently appending to the previous,
    partial file.
    """
    parts = [str(part) for part in command]
    for position in range(len(parts) - 1, 0, -1):
        if parts[position] == "--output":
            parts[position + 1] = str(new_output)
            return parts
    return None


# ---------------------------------------------------------------------------
# Data-parallel execution: batch size 1 per GPU across many GPUs
# ---------------------------------------------------------------------------

def _record_weight(record: Mapping[str, Any]) -> int:
    """Rough per-sample cost used to balance shards across GPU groups."""
    raw = record.get("raw")
    if isinstance(raw, Mapping):
        try:
            tokens = int(raw.get("input_tokens") or 0)
        except (TypeError, ValueError):
            tokens = 0
        if tokens > 0:
            return tokens
    return len(str(record.get("prompt") or ""))


def shard_indices(
    records: Sequence[Mapping[str, Any]], shards: int
) -> list[list[int]]:
    """Split record indices into deterministic, length-balanced shards.

    LongBench rows span roughly 1k-14k tokens, so a contiguous split can pile
    every long document onto one GPU and leave the others idle.  A
    longest-first greedy assignment (ties broken by source index) balances the
    estimated work while staying fully reproducible; the canonical order is
    restored inside each shard so logs stay readable.
    """
    if shards <= 1 or len(records) <= 1:
        return [list(range(len(records)))]
    count = min(int(shards), len(records))
    groups: list[list[int]] = [[] for _ in range(count)]
    weights = [0] * count
    order = sorted(
        range(len(records)),
        key=lambda index: (-_record_weight(records[index]), index),
    )
    for index in order:
        # Empty groups are filled first, then the currently lightest group;
        # the final key keeps the assignment fully deterministic.
        target = min(
            range(count),
            key=lambda bucket: (0 if not groups[bucket] else 1, weights[bucket], bucket),
        )
        groups[target].append(index)
        weights[target] += _record_weight(records[index])
    return [sorted(group) for group in groups]


def _resolve_dp_gpu_groups(
    report: Mapping[str, Any],
    *,
    cuda_available: bool,
    gpus_per_shard: int,
) -> list[list[int]]:
    """Partition the selected GPUs into one device group per shard.

    ``--gpu-ids`` (physical indices) defines the pool; when it is unset every
    CUDA-visible device is used.  A group of size 1 means "one card per batch-1
    child", while larger groups support methods that need intra-process
    parallelism (for example tensor parallel).
    """
    if not cuda_available:
        return []
    ids = [int(value) for value in (report.get("requested_ids") or [])]
    if not ids:
        ids = list(range(int(report.get("visible_gpu_count") or 0)))
    if len(ids) < 2:
        return [ids] if ids else []
    if len(ids) % gpus_per_shard != 0:
        raise SystemExit(
            f"--dp-gpus-per-shard {gpus_per_shard} does not divide the "
            f"{len(ids)} selected GPU(s) {ids}; adjust --gpu-ids or the group "
            "size so every shard owns the same number of devices"
        )
    return [ids[start : start + gpus_per_shard] for start in range(0, len(ids), gpus_per_shard)]


def _merge_shard_outputs(
    outputs: Sequence[Path],
    *,
    output_path: Path,
    baseline: str,
    dataset: str,
    run_id: str,
    sample_order: Mapping[str, int],
    aggregate_only: bool,
    world_size: int,
    processes_per_gpu: int = 1,
) -> int:
    """Concatenate shard JSONL files into the canonical per-cell output file.

    Upstream per-shard summaries are preserved verbatim under
    ``shard_summaries`` instead of being arithmetically merged: their keys are
    baseline-specific and mixing them blindly could fabricate a metric.
    Canonical metrics are recomputed by ``collect_metrics.py`` from the merged
    sample rows.
    """
    rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for path in outputs:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "summary":
                shard_summaries.append(row)
            else:
                rows.append(row)

    def _sort_key(row: Mapping[str, Any]) -> tuple[int, int]:
        sample_id = row.get("sample_id")
        if sample_id is None:
            # Aggregate rows stay last; the sort is stable, so shard order and
            # within-shard order are preserved.
            return (1, 0)
        return (0, int(sample_order.get(str(sample_id), len(sample_order))))

    rows.sort(key=_sort_key)

    totals: dict[str, float] = {}
    for field in ("output_tokens", "prefill_ms", "ttft_ms", "decode_ms", "e2e_ms"):
        values = [
            float(row[field])
            for row in rows
            if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
        ]
        if values:
            totals[f"total_{field}"] = round(sum(values), 3)
    sample_rows = [
        row
        for row in rows
        if row.get("scope") != "aggregate" and row.get("sample_id") is not None
    ]

    summary: dict[str, Any] = {
        "type": "summary",
        "method": baseline,
        "dataset": dataset,
        "run_id": run_id,
        "data_parallel": True,
        "shard_count": int(world_size),
        "shards_merged": len(outputs),
        "num_records": len(rows),
        "num_samples": len(sample_rows),
        "num_aggregate_records": len(rows) - len(sample_rows),
        **totals,
        "aggregation_note": (
            "rows merged from data-parallel shards (batch size 1 per GPU group); "
            "canonical metrics are aggregated by collect_metrics.py from the "
            "sample rows, upstream per-shard summaries are preserved as-is "
            "under shard_summaries"
        ),
        "shard_summaries": shard_summaries,
    }
    if len(outputs) != int(world_size):
        summary["shards_missing"] = int(world_size) - len(outputs)
    if aggregate_only:
        summary["shard_aggregate_semantics"] = "per_shard"
    if int(processes_per_gpu) > 1:
        # Several batch-1 children shared each card, so latency/throughput were
        # measured under SM contention and must not be compared with a run that
        # had one process per card.
        summary["processes_per_gpu"] = int(processes_per_gpu)
        summary["measurement_note"] = (
            "shared_gpu: multiple batch-1 processes ran on the same card; "
            "per-sample latency/throughput are not comparable with "
            "one-process-per-card runs"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return len(rows)


def _run_data_parallel_cell(
    *,
    baseline: str,
    dataset: str,
    source_rows: Sequence[Mapping[str, Any]],
    normalized: Sequence[Mapping[str, Any]],
    run_dir: Path,
    output_path: Path,
    cfg: Mapping[str, Any],
    gpu_groups: Sequence[Sequence[int]],
    timeout_seconds: int,
    run_id: str,
    processes_per_gpu: int = 1,
    vram: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one matrix cell as concurrent batch-1 shards over the GPU pool.

    Each shard is an independent child pinned to its own device group, so
    ``processes_per_gpu > 1`` means several batch-1 children share a card.  The
    VRAM planner only launches what fits inside ``budget - headroom`` and waits
    when nothing fits: a child is never started into a known OOM, a running
    child is never killed, and a shard that still OOMs is retried alone.  The
    returned record mirrors ``_run_child``'s shape so the caller treats both
    paths identically.
    """
    vram_cfg = dict(vram or {})
    budget_gb = float(vram_cfg.get("budget_gb") or 0.0)
    headroom_gb = float(vram_cfg.get("headroom_gb") or 0.0)
    child_gb = float(vram_cfg.get("child_reserve_gb") or 0.0) or 1.0
    wait_seconds = float(vram_cfg.get("wait_seconds") or 0.0)
    oom_retries = int(vram_cfg.get("oom_retries") or 0)
    usable_gb = max(0.0, budget_gb - headroom_gb) if budget_gb > 0 else None

    start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Shard inputs live under inputs/shards (pristine, reproducible) while shard
    # outputs live next to the merged file under <baseline>/shards, so a child
    # never reads and writes the same path.
    input_shard_dir = run_dir / "inputs" / "shards"
    output_shard_dir = output_path.parent / "shards"
    output_shard_dir.mkdir(parents=True, exist_ok=True)

    def _result(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "status": "failed",
            "returncode": None,
            "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "output_exists": output_path.is_file(),
            "log": "",
            "log_tail": "",
            "command": [],
            "data_parallel": True,
            "shard_count": 0,
            "shards": [],
            "normalized_records": 0,
            "processes_per_gpu": 1,
            "vram_plan": None,
            "oom_retry_rounds": 0,
            "retried_shards": [],
        }
        base.update(overrides)
        return base

    slots, plan, waited = _wait_for_shard_slots(
        gpu_groups,
        processes_per_gpu=processes_per_gpu,
        usable_gb=usable_gb,
        child_gb=child_gb,
        sample_count=len(normalized),
        wait_seconds=wait_seconds,
    )
    plan["dataset"] = dataset
    if not slots:
        reason = (
            f"no batch-1 child fits in the {budget_gb:.0f} GiB VRAM budget "
            f"(usable {plan['usable_gb']} GiB, reserve {child_gb:.0f} GiB/child) "
            f"after waiting {waited:.0f}s; nothing was launched so no job was killed"
        )
        print(f"[{baseline}/{dataset}] vram_blocked: {reason}", file=sys.stderr, flush=True)
        return _result(status="vram_blocked", reason=reason, vram_plan=plan)

    sample_groups = shard_indices(normalized, len(slots))
    slots = slots[: len(sample_groups)]
    processes_per_card = max(
        (sum(1 for other in slots if other == slot) for slot in slots), default=1
    )
    plan["shard_slots"] = len(slots)
    plan["processes_per_gpu_actual"] = processes_per_card
    if processes_per_card > 1:
        print(
            f"[{baseline}/{dataset}] WARNING {processes_per_card} batch-1 "
            "processes share each card: throughput of the sweep improves but "
            "per-sample latency/throughput are no longer comparable with "
            "one-process-per-card runs (marked in the output and manifest)",
            file=sys.stderr,
            flush=True,
        )

    jobs: list[dict[str, Any]] = []
    for index, group in enumerate(sample_groups):
        device_group = slots[index]
        shard_source = [source_rows[position] for position in group]
        shard_normalized = [normalized[position] for position in group]
        source_path = input_shard_dir / f"{dataset}.dp{index}.jsonl"
        converted_path = input_shard_dir / f"{baseline}_{dataset}.dp{index}.jsonl"
        _write_jsonl(source_path, shard_source)
        converted_input = convert_records_for_baseline(
            baseline, shard_normalized, converted_path
        )
        shard_output = output_shard_dir / f"{dataset}.dp{index}.jsonl"
        shard_cfg = dict(cfg)
        if baseline in {"domino", "dspark"}:
            # SGLang owns a local HTTP server.  Concurrent DP shards must not
            # bind the same default port; physical GPU ids keep this mapping
            # deterministic and easy to diagnose in the manifest/logs.
            base_port = int(shard_cfg.get("port", 30000))
            shard_cfg["port"] = base_port + 100 + index
        command = build_adapter_command(
            baseline,
            data_file=source_path,
            converted_input=converted_input,
            output=shard_output,
            max_samples=len(shard_normalized),
            max_new_tokens=int(cfg.get("max_new_tokens") or 0),
            config=shard_cfg,
        )
        if command is None:
            return _result(
                status="unsupported_dataset",
                reason="adapter did not produce a command for this dataset",
                vram_plan=plan,
            )
        jobs.append(
            {
                "command": command,
                "output": shard_output,
                "log_path": run_dir / "logs" / f"{baseline}_{dataset}.dp{index}.log",
                "cuda_visible_devices": ",".join(str(gpu) for gpu in device_group),
                "shard_index": index,
                "sample_count": len(shard_normalized),
                "output_path": shard_output,
                "input_path": source_path,
                "gpu_ids": [int(gpu) for gpu in device_group],
                "source_records": shard_normalized,
                "concurrency": sum(1 for other in slots if other == device_group),
                "_final_status": "failed",
                "_last_result": {},
                "_retries": [],
            }
        )

    print(
        f"[{baseline}/{dataset}] launching {len(jobs)} batch-1 shard(s) "
        f"({processes_per_card}/card): "
        + ", ".join(
            f"shard {job['shard_index']} -> GPU {job['cuda_visible_devices']} "
            f"({job['sample_count']} sample(s))"
            for job in jobs
        ),
        flush=True,
    )

    def _finalize(job: dict[str, Any], result: dict[str, Any], path: Path) -> str:
        """Normalize a finished shard written to ``path`` and return its status."""
        status = str(result["status"])
        if status != "success":
            return status
        extra = (
            {"shared_gpu_concurrency": job["concurrency"]}
            if job["concurrency"] > 1
            else None
        )
        count = _normalize_child_output(
            path,
            baseline=baseline,
            dataset=dataset,
            source_records=job["source_records"],
            config=cfg,
            run_id=run_id,
            extra_fields=extra,
        )
        if count == 0:
            result["reason"] = "shard exited successfully but wrote no result records"
            return "failed"
        coverage_error = _sample_coverage_error(
            path, source_records=job["source_records"]
        )
        if coverage_error is not None:
            result["reason"] = coverage_error
            return "failed"
        result["normalized_records"] = count
        return "success"

    results = _run_child_group(jobs, timeout_seconds=timeout_seconds)
    for job, result in zip(jobs, results):
        job["_last_result"] = result
        job["_final_status"] = _finalize(job, result, job["output_path"])

    # Safety net for the long unattended runs: a shard that OOMed even inside
    # the budget is retried alone, after its siblings have exited, before the
    # cell is allowed to fail.  Nothing is killed and no cell is silently
    # dropped; every attempt is recorded.
    retry_rounds = 0
    retry_plans: list[dict[str, Any]] = []
    for attempt in range(1, oom_retries + 1):
        pending = [
            job
            for job in jobs
            if job["_final_status"] != "success" and _log_shows_oom(job["log_path"])
        ]
        if not pending:
            break
        retry_rounds = attempt
        retry_slots, retry_plan, retry_waited = _wait_for_shard_slots(
            gpu_groups,
            processes_per_gpu=1,
            usable_gb=usable_gb,
            child_gb=child_gb,
            sample_count=1,
            wait_seconds=wait_seconds,
        )
        print(
            f"[{baseline}/{dataset}] OOM retry {attempt}/{oom_retries}: "
            f"{len(pending)} shard(s), one at a time, "
            f"waited {retry_waited:.0f}s for VRAM",
            file=sys.stderr,
            flush=True,
        )
        retry_plans.append(retry_plan)
        if not retry_slots:
            reason = (
                "OOM retry blocked: no GPU slot satisfied the configured VRAM "
                f"budget after waiting {retry_waited:.0f}s"
            )
            print(
                f"[{baseline}/{dataset}] {reason}; leaving OOM shards failed",
                file=sys.stderr,
                flush=True,
            )
            for job in pending:
                job["_retries"].append(
                    {
                        "attempt": attempt,
                        "status": "vram_blocked",
                        "returncode": None,
                        "reason": reason,
                        "vram_plan": retry_plan,
                    }
                )
            break

        # The planner returns a concrete physical GPU group.  Use that group
        # for the retry instead of silently reusing the possibly-full shard
        # assignment from the failed concurrent round.
        retry_cuda_visible_devices = ",".join(
            str(gpu) for gpu in retry_slots[0]
        )
        for job in pending:
            retry_output = job["output_path"].with_name(
                f"{job['output_path'].stem}.r{attempt}.jsonl"
            )
            retry_log = job["log_path"].with_name(
                f"{job['log_path'].stem}.r{attempt}.log"
            )
            command = _replace_command_output(job["command"], retry_output)
            if command is None:
                print(
                    f"[{baseline}/{dataset}] shard {job['shard_index']}: cannot "
                    "retry (command has no --output), leaving it failed",
                    file=sys.stderr,
                    flush=True,
                )
                break
            result = _run_child(
                command,
                output=retry_output,
                log_path=retry_log,
                timeout_seconds=timeout_seconds,
                cuda_visible_devices=retry_cuda_visible_devices,
            )
            result["retry_attempt"] = attempt
            status = _finalize(job, result, retry_output)
            job["_retries"].append(
                {
                    "attempt": attempt,
                    "status": status,
                    "returncode": result.get("returncode"),
                    "log": str(retry_log),
                    "output": str(retry_output),
                    "cuda_visible_devices": retry_cuda_visible_devices,
                }
            )
            job["_last_result"] = result
            job["_final_status"] = status
            if status == "success":
                # Merge from the retry file; the partial first attempt stays on
                # disk for inspection but is never merged.
                job["output_path"] = retry_output
                job["log_path"] = retry_log
                job["output"] = retry_output

    entries: list[dict[str, Any]] = []
    for job in jobs:
        result = job["_last_result"]
        entry: dict[str, Any] = {
            "shard_index": job["shard_index"],
            "gpu_ids": job["gpu_ids"],
            "cuda_visible_devices": job["cuda_visible_devices"],
            "concurrency": job["concurrency"],
            "sample_count": job["sample_count"],
            "status": job["_final_status"],
            "returncode": result.get("returncode"),
            "elapsed_ms": result.get("elapsed_ms"),
            "log": result.get("log"),
            "input": str(job["input_path"]),
            "output": str(job["output_path"]),
            "output_exists": bool(result.get("output_exists")),
            "command": list(result.get("command") or job["command"]),
        }
        if result.get("reason"):
            entry["reason"] = result["reason"]
        if result.get("normalized_records") is not None:
            entry["normalized_records"] = result["normalized_records"]
        if job["_retries"]:
            entry["retries"] = job["_retries"]
        entries.append(entry)

    successful = [job for job in jobs if job["_final_status"] == "success"]
    merged_count = 0
    if successful:
        # Partial shard failures still contribute their records, matching the
        # existing behaviour where a failed child may have written partial
        # output; strict collection then flags the incomplete cell.
        merged_count = _merge_shard_outputs(
            [job["output_path"] for job in successful],
            output_path=output_path,
            baseline=baseline,
            dataset=dataset,
            run_id=run_id,
            sample_order={str(row["id"]): position for position, row in enumerate(normalized)},
            aggregate_only=baseline in AGGREGATE_ONLY_BASELINES,
            world_size=len(jobs),
            processes_per_gpu=processes_per_card,
        )
        if merged_count:
            _normalize_child_output(
                output_path,
                baseline=baseline,
                dataset=dataset,
                source_records=normalized,
                config=cfg,
                run_id=run_id,
                extra_fields=(
                    {"shared_gpu_concurrency": processes_per_card}
                    if processes_per_card > 1
                    else None
                ),
            )

    statuses = [job["_final_status"] for job in jobs]
    if statuses and all(value == "success" for value in statuses):
        status = "success"
    elif any(value == "timeout" for value in statuses):
        status = "timeout"
    else:
        status = "failed"
    first_failed = next((entry for entry in entries if entry["status"] != "success"), None)
    return _result(
        status=status,
        returncode=0 if status == "success" else 1,
        log=(first_failed or entries[0])["log"] if entries else "",
        log_tail="",
        shard_count=len(jobs),
        shards=entries,
        normalized_records=merged_count,
        processes_per_gpu=processes_per_card,
        vram_plan=plan,
        oom_retry_rounds=retry_rounds,
        oom_retry_plans=retry_plans,
        retried_shards=[
            entry["shard_index"] for entry in entries if entry.get("retries")
        ],
    )


def _run_collector(
    run_dir: Path,
    data_dir: Path,
    *,
    baselines: Sequence[str],
    datasets: Sequence[str],
    expected_samples: int,
    strict: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run the metric collector over a finished run directory.

    The collector runs as a child process with the same interpreter and env as
    the orchestrator so aggregation never shares process state with model
    runs, and its output lands in ``run_dir/metrics_summary.{json,csv,md}``
    with a log under ``run_dir/logs/``.  In ``strict`` mode the collector also
    validates that every (baseline, dataset) pair produced the expected number
    of successful samples.  Aggregation is best-effort reporting on top of the
    raw JSONL: a failure here never invalidates the cell outputs already
    written, but it is recorded in ``run_manifest.json`` under ``aggregate``.
    """
    out_path = run_dir / "metrics_summary.json"
    command = [
        sys.executable,
        str(ROOT / "src" / "Benchmark" / "collect_metrics.py"),
        "--outputs-dir",
        str(run_dir),
        "--data-dir",
        str(data_dir),
        "--out",
        str(out_path),
        "--csv",
        str(run_dir / "metrics_summary.csv"),
        "--md",
        str(run_dir / "metrics_summary.md"),
    ]
    if strict:
        command += [
            "--strict",
            "--expected-baselines",
            " ".join(baselines),
            "--expected-datasets",
            " ".join(datasets),
            "--expected-samples",
            str(expected_samples),
        ]
    result = _run_child(
        command,
        output=out_path,
        log_path=run_dir / "logs" / "collect_metrics.log",
        timeout_seconds=timeout_seconds,
    )
    result["strict"] = bool(strict)
    result["output_files"] = {
        "json": str(out_path),
        "csv": str(run_dir / "metrics_summary.csv"),
        "md": str(run_dir / "metrics_summary.md"),
    }
    return result


_TIMING_FIELDS = (
    "model_load_ms",
    "server_startup_ms",
    "queue_wait_ms",
    "batch_wait_ms",
    "selector_latency_ms",
    "draft_latency_ms",
    "verification_latency_ms",
    "server_reported_e2e_ms",
    "prefill_ms",
    "ttft_ms",
    "decode_ms",
    "tpot_ms",
    "e2e_ms",
    "throughput_tok_s",
    "decode_throughput_tok_s",
    "qps",
    "peak_memory_gb",
)


def _sample_coverage_error(
    path: Path, *, source_records: Sequence[Mapping[str, Any]]
) -> str | None:
    """Return a validation error when a successful shard has wrong sample ids."""
    expected = {
        str(record["id"])
        for record in source_records
        if record.get("id") is not None
    }
    observed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "summary":
            continue
        sample_id = row.get("sample_id")
        if sample_id is not None and row.get("scope") != "aggregate":
            observed.add(str(sample_id))
            continue
        sample_ids = row.get("sample_ids")
        if isinstance(sample_ids, (list, tuple, set)):
            observed.update(str(value) for value in sample_ids)

    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if not missing and not unexpected:
        return None
    details: list[str] = []
    if missing:
        details.append(f"missing={missing[:5]}")
    if unexpected:
        details.append(f"unexpected={unexpected[:5]}")
    return "shard sample coverage mismatch: " + "; ".join(details)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows while keeping malformed/empty files distinguishable."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _unresolved_sample_records(
    path: Path, *, source_records: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Return source rows absent or non-successful in a child JSONL output.

    A crashed adapter can leave a valid prefix of sample records, while a
    baseline may explicitly emit a failed row.  Both cases are retryable.  An
    aggregate record with ``sample_ids`` is treated as covering those samples
    only when its status is successful.
    """
    states: dict[str, bool] = {}
    for row in _jsonl_rows(path):
        if row.get("type") == "summary":
            continue
        status = str(row.get("status", "success"))
        sample_id = row.get("sample_id")
        if sample_id is not None and row.get("scope") != "aggregate":
            states[str(sample_id)] = status == "success"
            continue
        sample_ids = row.get("sample_ids")
        if status == "success" and isinstance(sample_ids, (list, tuple, set)):
            for value in sample_ids:
                states[str(value)] = True

    return [
        record
        for record in source_records
        if str(record.get("id")) not in states
        or not states[str(record.get("id"))]
    ]


def _safe_retry_paths(
    run_dir: Path,
    *,
    baseline: str,
    dataset: str,
    sample_id: object,
    attempt: int,
) -> dict[str, Path]:
    """Return isolated input/output/log paths for one sample retry."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._") or "sample"
    stem = f"{safe_id}.a{int(attempt)}"
    return {
        "input": run_dir / "attempts" / baseline / dataset / f"{stem}.input.jsonl",
        "output": run_dir / "attempts" / baseline / dataset / f"{stem}.output.jsonl",
        "log": run_dir / "logs" / "attempts" / baseline / dataset / f"{stem}.log",
    }


def _rewrite_safe_cell_output(
    output_path: Path,
    *,
    baseline: str,
    dataset: str,
    source_records: Sequence[Mapping[str, Any]],
    successful_rows: Sequence[Mapping[str, Any]],
    unresolved_reasons: Mapping[str, str],
    model: str | None,
    config: Mapping[str, Any],
    run_id: str,
    retry_count: int,
    retry_attempt_count: int = 0,
    retry_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Publish a complete per-sample file without inventing failed metrics."""
    success_by_id: dict[str, dict[str, Any]] = {}
    for original in successful_rows:
        row = dict(original)
        if row.get("type") == "summary":
            continue
        sample_id = row.get("sample_id")
        if sample_id is None or row.get("status", "success") != "success":
            continue
        success_by_id.setdefault(str(sample_id), row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    writer = io_util.JsonlWriter(output_path)
    unresolved_ids: list[str] = []
    for source in source_records:
        sample_id = str(source.get("id"))
        row = success_by_id.get(sample_id)
        if row is None:
            unresolved_ids.append(sample_id)
            row = build_status_record(
                method=baseline,
                dataset=dataset,
                sample_id=source.get("id"),
                status="failed",
                reason=unresolved_reasons.get(
                    sample_id, "sample did not produce a successful result"
                ),
                model=model,
                config=config,
            )
            row.update(
                retry_exhausted=True,
                retry_count=int(retry_count),
                task_type=source.get("task_type"),
            )
        writer.add(row)

    summary = {
        "type": "summary",
        "method": baseline,
        "dataset": dataset,
        "run_id": run_id,
        "status": "success" if not unresolved_ids else "failed",
        "safe_eval_complete": not unresolved_ids,
        "num_samples": len(source_records),
        "successful_samples": len(source_records) - len(unresolved_ids),
        "failed_samples": len(unresolved_ids),
        "unresolved_sample_count": len(unresolved_ids),
        "unresolved_sample_ids": unresolved_ids,
        "retried_sample_count": int(retry_count),
        "retry_attempt_count": int(retry_attempt_count),
        "retry_history": [dict(item) for item in (retry_history or [])],
        "retry_failed_sample_ids": unresolved_ids,
    }
    writer.finalize(summary)
    return {
        "safe_eval_complete": not unresolved_ids,
        "unresolved_sample_count": len(unresolved_ids),
        "unresolved_sample_ids": unresolved_ids,
        "successful_samples": len(source_records) - len(unresolved_ids),
        "retried_sample_count": int(retry_count),
        "retry_attempt_count": int(retry_attempt_count),
        "retry_history": [dict(item) for item in (retry_history or [])],
    }


def _retry_unresolved_samples(
    *,
    baseline: str,
    dataset: str,
    source_rows: Sequence[Mapping[str, Any]],
    normalized: Sequence[Mapping[str, Any]],
    output_path: Path,
    run_dir: Path,
    cfg: Mapping[str, Any],
    max_new_tokens: int,
    timeout_seconds: int,
    run_id: str,
    sample_retries: int,
    retry_backoff_seconds: float,
    retry_device_groups: Sequence[Sequence[int]] | None = None,
    retry_usable_gb: float | None = None,
    retry_child_gb: float = 1.0,
    retry_wait_seconds: float = 0.0,
    reference_path: Path | None = None,
    reference_baseline: str | None = None,
) -> dict[str, Any]:
    """Retry missing/failed samples one at a time and publish safe coverage.

    The initial child may have produced a valid prefix before crashing.  This
    function keeps those successes, reruns only unresolved IDs, and writes a
    null-timing ``failed`` row for anything still unresolved after the retry
    budget.  Thus a long run can continue while the final JSONL remains
    complete and auditable instead of silently dropping samples.
    """
    try:
        initial_rows = _jsonl_rows(output_path)
        unresolved = _unresolved_sample_records(
            output_path, source_records=normalized
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        initial_rows = []
        unresolved = list(normalized)
        initial_error = f"could not inspect initial output: {type(exc).__name__}: {exc}"
    else:
        initial_error = None

    if not unresolved:
        return {
            "safe_eval_complete": True,
            "unresolved_sample_count": 0,
            "unresolved_sample_ids": [],
            "successful_samples": len(normalized),
            "retried_sample_count": 0,
            "retry_attempt_count": 0,
            "retry_history": [],
        }

    source_by_id = {str(row.get("id")): row for row in source_rows}
    successful_rows = [
        row
        for row in initial_rows
        if row.get("type") != "summary"
        and row.get("sample_id") is not None
        and row.get("status", "success") == "success"
    ]
    unresolved_reasons: dict[str, str] = {}
    for row in initial_rows:
        if row.get("type") == "summary":
            continue
        sample_id = row.get("sample_id")
        if sample_id is not None and row.get("status", "success") != "success":
            unresolved_reasons[str(sample_id)] = str(
                row.get("reason") or "initial child emitted a non-success status"
            )
    if initial_error:
        unresolved_reasons.update(
            {str(row.get("id")): initial_error for row in unresolved}
        )

    retry_history: list[dict[str, Any]] = []
    retried_ids: set[str] = set()
    retry_attempt_count = 0
    groups = list(retry_device_groups or [])

    for position, record in enumerate(unresolved):
        sample_id = str(record.get("id"))
        if sample_retries > 0:
            retried_ids.add(sample_id)
        raw_record = source_by_id.get(sample_id, record)
        last_reason = unresolved_reasons.get(
            sample_id, "sample output is missing after the initial child run"
        )
        recovered = False
        for attempt in range(1, max(0, int(sample_retries)) + 1):
            retry_attempt_count += 1
            if attempt > 1 and retry_backoff_seconds > 0:
                time.sleep(float(retry_backoff_seconds) * (2 ** (attempt - 2)))
            paths = _safe_retry_paths(
                run_dir,
                baseline=baseline,
                dataset=dataset,
                sample_id=sample_id,
                attempt=attempt,
            )
            retry_cfg = dict(cfg)
            retry_cfg["skip_reference"] = reference_path is not None
            try:
                _write_jsonl(paths["input"], [raw_record])
                converted_input = convert_records_for_baseline(
                    baseline, [record], paths["input"].with_suffix(".converted.jsonl")
                )
                command = build_adapter_command(
                    baseline,
                    data_file=paths["input"],
                    converted_input=converted_input,
                    output=paths["output"],
                    max_samples=1,
                    max_new_tokens=max_new_tokens,
                    config=retry_cfg,
                )
                if command is None:
                    last_reason = "adapter did not produce a retry command"
                    retry_history.append(
                        {
                            "sample_id": sample_id,
                            "attempt": attempt,
                            "status": "unsupported_dataset",
                            "reason": last_reason,
                        }
                    )
                    break
                cuda_visible_devices = None
                if groups:
                    if retry_usable_gb is not None:
                        retry_slots, retry_plan, retry_waited = _wait_for_shard_slots(
                            groups,
                            processes_per_gpu=1,
                            usable_gb=retry_usable_gb,
                            child_gb=retry_child_gb,
                            sample_count=1,
                            wait_seconds=retry_wait_seconds,
                        )
                        if not retry_slots:
                            last_reason = (
                                "retry blocked: no GPU slot satisfied the "
                                f"VRAM budget after waiting {retry_waited:.0f}s"
                            )
                            retry_history.append(
                                {
                                    "sample_id": sample_id,
                                    "attempt": attempt,
                                    "status": "vram_blocked",
                                    "reason": last_reason,
                                    "vram_plan": retry_plan,
                                }
                            )
                            break
                        group = retry_slots[0]
                    else:
                        group = groups[position % len(groups)]
                    cuda_visible_devices = ",".join(str(gpu) for gpu in group)
                print(
                    f"[{baseline}/{dataset}] retry sample {sample_id} "
                    f"attempt {attempt}/{sample_retries}"
                    + (
                        f" on GPU {cuda_visible_devices}"
                        if cuda_visible_devices
                        else ""
                    ),
                    flush=True,
                )
                result = _run_child(
                    command,
                    output=paths["output"],
                    log_path=paths["log"],
                    timeout_seconds=timeout_seconds,
                    cuda_visible_devices=cuda_visible_devices,
                )
            except Exception as exc:
                result = {
                    "status": "failed",
                    "returncode": None,
                    "elapsed_ms": None,
                    "output_exists": paths["output"].is_file(),
                    "log": str(paths["log"]),
                    "log_tail": "",
                    "command": [],
                }
                last_reason = f"retry exception: {type(exc).__name__}: {exc}"
                retry_history.append(
                    {
                        "sample_id": sample_id,
                        "attempt": attempt,
                        "status": "failed",
                        "reason": last_reason,
                        "log": str(paths["log"]),
                    }
                )
                continue

            attempt_status = str(result.get("status") or "failed")
            attempt_reason = result.get("reason")
            if _log_shows_oom(paths["log"]):
                last_reason = "retry failed with CUDA OOM"
                attempt_reason = last_reason
            elif attempt_status == "timeout":
                last_reason = "retry timed out"
                attempt_reason = last_reason
            elif attempt_reason:
                last_reason = str(attempt_reason)
            elif attempt_status != "success":
                last_reason = (
                    f"retry child exited with status {attempt_status} "
                    f"(returncode={result.get('returncode')})"
                )

            attempt_rows: list[dict[str, Any]] = []
            if paths["output"].is_file():
                try:
                    _normalize_child_output(
                        paths["output"],
                        baseline=baseline,
                        dataset=dataset,
                        source_records=[record],
                        config=retry_cfg,
                        run_id=run_id,
                    )
                    if reference_path is not None and reference_baseline:
                        _attach_external_reference_metrics(
                            paths["output"],
                            reference_path,
                            reference_baseline=reference_baseline,
                        )
                    attempt_rows = _jsonl_rows(paths["output"])
                except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    last_reason = (
                        f"could not normalize retry output: {type(exc).__name__}: {exc}"
                    )

            recovered_rows = [
                row
                for row in attempt_rows
                if row.get("type") != "summary"
                and str(row.get("sample_id")) == sample_id
                and row.get("status", "success") == "success"
            ]
            retry_entry = {
                "sample_id": sample_id,
                "attempt": attempt,
                "status": "success" if recovered_rows else attempt_status,
                "returncode": result.get("returncode"),
                "elapsed_ms": result.get("elapsed_ms"),
                "output": str(paths["output"]),
                "log": str(paths["log"]),
            }
            if cuda_visible_devices:
                retry_entry["cuda_visible_devices"] = cuda_visible_devices
            if attempt_reason:
                retry_entry["reason"] = str(attempt_reason)
            retry_history.append(retry_entry)
            if recovered_rows:
                successful_rows.extend(recovered_rows)
                recovered = True
                break

        if not recovered:
            unresolved_reasons[sample_id] = last_reason

    result = _rewrite_safe_cell_output(
        output_path,
        baseline=baseline,
        dataset=dataset,
        source_records=normalized,
        successful_rows=successful_rows,
        unresolved_reasons=unresolved_reasons,
        model=str(cfg.get("model") or "") or None,
        config=cfg,
        run_id=run_id,
        retry_count=len(retried_ids),
        retry_attempt_count=retry_attempt_count,
        retry_history=retry_history,
    )
    result.update(
        retry_attempt_count=retry_attempt_count,
        retry_history=retry_history,
    )
    return result


def _normalize_child_output(
    path: Path,
    *,
    baseline: str,
    dataset: str,
    source_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    run_id: str,
    extra_fields: Mapping[str, Any] | None = None,
) -> int:
    """Normalize upstream JSONL fields in-place after a successful child run.

    ``extra_fields`` are stamped onto every non-summary row (for example the
    shared-GPU concurrency, so a packed run can be filtered out of a
    batch-1-only comparison later).
    """
    if not path.is_file():
        return 0
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return 0
    by_id = {str(row["id"]): row for row in source_records if row.get("id") is not None}
    normalized_rows: list[dict[str, Any]] = []
    observations = 0
    for original in rows:
        row = dict(original)
        if row.get("type") == "summary":
            row.update(method=baseline, dataset=dataset, run_id=run_id)
            normalized_rows.append(row)
            continue

        upstream_method = row.get("method")
        if upstream_method and upstream_method != baseline:
            row["upstream_method"] = upstream_method
        row["method"] = baseline
        row["dataset"] = dataset
        row["run_id"] = run_id
        row.setdefault("status", "success")
        row.setdefault("model", config.get("model"))
        row.setdefault("batch_size", 1)
        row.setdefault("device", config.get("device"))
        row.setdefault("dtype", config.get("dtype"))
        row.setdefault("seed", config.get("seed"))
        row.setdefault("temperature", config.get("temperature"))
        row.setdefault("max_new_tokens", config.get("max_new_tokens"))
        row.setdefault("warmup_runs", config.get("warmup_runs"))
        for key, value in (extra_fields or {}).items():
            row.setdefault(key, value)

        if row.get("sample_id") is None and row.get("question_id") is not None:
            row["sample_id"] = row["question_id"]
        if row.get("output_tokens") is None and row.get("new_tokens") is not None:
            row["output_tokens"] = row["new_tokens"]
        if row.get("text") is None and isinstance(row.get("answer"), str):
            row["text"] = row["answer"]
        if row.get("decode_ms") is None and row.get("eagle_time") is not None:
            row["decode_ms"] = round(float(row["eagle_time"]) * 1000.0, 3)
        if row.get("throughput_tok_s") is None and row.get("eagle_tok_s") is not None:
            row["throughput_tok_s"] = row["eagle_tok_s"]
        if row.get("dense_decode_ms") is None and row.get("naive_time") is not None:
            row["dense_decode_ms"] = round(float(row["naive_time"]) * 1000.0, 3)
        if row.get("eagle_time") is not None and not all(
            row.get(field) is not None
            for field in ("prefill_ms", "ttft_ms", "decode_ms", "e2e_ms")
        ):
            # EAGLE's upstream timer explicitly excludes prefill.  Keep that
            # fact visible instead of calling decode-only time "E2E".
            row.setdefault("measurement_scope", "decode_only")
        elif row.get("measurement_scope") is None:
            expected_scope = BASELINE_MEASUREMENT_SCOPE.get(baseline)
            if expected_scope is not None:
                row["measurement_scope"] = expected_scope

        sample_id = row.get("sample_id")
        source = by_id.get(str(sample_id)) if sample_id is not None else None
        if source:
            row.setdefault("reference_output", source.get("reference_output"))
            row.setdefault("task_type", source.get("task_type"))
        if row.get("task_type") is None and source_records:
            row["task_type"] = source_records[0].get("task_type")
        if row.get("reference_output") is None and source:
            row["reference_output"] = source.get("reference_output")

        aggregate = row.get("scope") == "aggregate" or row.get("sample_id") is None
        row["scope"] = "aggregate" if aggregate else "sample"
        if aggregate and not row.get("sample_ids"):
            row["sample_ids"] = [source["id"] for source in source_records]

        # Code-completion output must not carry summarization metrics from an
        # upstream helper.  The collector computes exact/edit scores from the
        # normalized text/reference pair.
        if row.get("task_type") == "code_completion":
            for key in list(row):
                if key.startswith(("rouge", "bleu")) or key == "length_ratio":
                    row.pop(key, None)
        for field in _TIMING_FIELDS:
            row.setdefault(field, None)
        normalized_rows.append(row)
        observations += 1

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized_rows),
        encoding="utf-8",
    )
    return observations


# ---------------------------------------------------------------------------
# GPU selection & inventory helpers
# ---------------------------------------------------------------------------

def _gpu_selection_raw() -> str | None:
    """Return the effective GPU-id request from the environment, if any."""
    for name in ("LONG_BENCH_GPU_IDS", "FI_GPU_IDS", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _parse_gpu_ids(value: str | None) -> list[int] | None:
    """Parse a comma/space separated GPU id list into validated integers."""
    if value is None or not str(value).strip():
        return None
    parts = [part for part in str(value).replace(",", " ").split() if part]
    try:
        ids = [int(part) for part in parts]
    except ValueError as exc:
        raise SystemExit(
            f"invalid GPU ids {value!r}: expected device indices such as "
            "'0', '2' or '0,1'"
        ) from exc
    if any(index < 0 for index in ids):
        raise SystemExit(f"invalid GPU ids {value!r}: indices must be >= 0")
    return ids


def _device_policy() -> str:
    return (
        os.environ.get("LONG_BENCH_DEVICE")
        or os.environ.get("FI_DEVICE")
        or "cuda"
    ).lower()


def _nvidia_smi_gpus() -> list[dict[str, Any]] | None:
    """Physical GPU inventory via nvidia-smi (immune to CUDA_VISIBLE_DEVICES).

    Returns None when nvidia-smi is absent or fails; the caller then falls
    back to what torch reports for the visible subset.
    """
    query = (
        "index,name,memory.total,memory.free,memory.used,"
        "utilization.gpu,compute_cap"
    )
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None

    def _gib(value: str) -> float | None:
        try:
            return round(float(value) / 1024.0, 1)  # MiB -> GiB
        except ValueError:
            return None

    gpus: list[dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        util_raw = fields[5]
        gpus.append(
            {
                "index": index,
                "name": fields[1],
                "total_memory_gb": _gib(fields[2]),
                "free_memory_gb": _gib(fields[3]),
                "used_memory_gb": _gib(fields[4]),
                "utilization_percent": int(util_raw) if util_raw.isdigit() else None,
                "compute_capability": fields[6] or None,
            }
        )
    return gpus or None


def _torch_visible_gpus() -> dict[str, Any]:
    """Describe the CUDA-visible device subset as seen by torch."""
    result: dict[str, Any] = {"torch_available": False}
    try:
        import torch
    except Exception:
        return result
    result["torch_available"] = True
    if not torch.cuda.is_available():
        result["cuda_available"] = False
        result["visible_gpu_count"] = 0
        result["visible_gpus"] = []
        return result
    result["cuda_available"] = True
    result["cuda_version"] = torch.version.cuda
    visible: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        visible.append(
            {
                "visible_index": index,
                "name": str(props.name),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / (1024**3), 1),
            }
        )
    result["visible_gpu_count"] = len(visible)
    result["visible_gpus"] = visible
    return result


def describe_gpu_assignment() -> dict[str, Any]:
    """Return a JSON-safe snapshot of host GPUs, requested ids and visibility."""
    requested_raw = _gpu_selection_raw()
    host = _nvidia_smi_gpus() or []
    report: dict[str, Any] = {
        "requested": requested_raw,
        "requested_ids": _parse_gpu_ids(requested_raw),
        "device_policy": _device_policy(),
        "host_gpu_count": len(host) or None,
        "host_gpus": host,
    }
    report.update(_torch_visible_gpus())
    return report


def _missing_requested_gpus(report: Mapping[str, Any]) -> list[int]:
    requested = report.get("requested_ids") or []
    host = report.get("host_gpus") or []
    if not requested or not host:
        return []
    present = {gpu["index"] for gpu in host}
    return [index for index in requested if index not in present]


def gpu_memory_guard_reason(
    report: Mapping[str, Any], *, min_free_gb: float
) -> str | None:
    """Return an actionable reason when the selected GPU is already crowded.

    ``nvidia-smi`` reports physical indices while torch uses the selected
    ``CUDA_VISIBLE_DEVICES`` subset.  The runner has already applied that
    mapping, so compare physical ids here before any model child is spawned.
    If no inventory is available, leave the decision to the child process
    rather than falsely claiming that the GPU is safe or unsafe.
    """
    if min_free_gb <= 0:
        return None
    host_gpus = list(report.get("host_gpus") or [])
    if not host_gpus:
        return None
    requested = list(report.get("requested_ids") or [])
    selected = requested or [int(host_gpus[0]["index"])]
    by_index = {int(gpu["index"]): gpu for gpu in host_gpus}
    failures: list[str] = []
    for index in selected:
        gpu = by_index.get(int(index))
        if gpu is None:
            failures.append(f"GPU {index}: not present on host")
            continue
        free = gpu.get("free_memory_gb")
        if free is None:
            failures.append(f"GPU {index}: free VRAM is unavailable")
        elif float(free) < min_free_gb:
            failures.append(f"GPU {index}: {float(free):.1f} GiB free")
    if not failures:
        return None
    return (
        f"selected GPU(s) do not meet the {min_free_gb:.1f} GiB free-VRAM "
        f"guard ({'; '.join(failures)}). Stop other GPU processes or select a "
        "different GPU; override with --min-free-gb 0 only if intentional."
    )


def print_gpu_summary(report: Mapping[str, Any], *, effective_cuda: bool) -> None:
    """Print a one-line GPU assignment banner for a normal run."""
    requested = report.get("requested_ids") or []
    host_count = report.get("host_gpu_count")
    parts: list[str] = []
    if host_count:
        parts.append(f"host has {host_count} GPU(s)")
    parts.append(
        "selected GPU " + (", ".join(map(str, requested)) if requested else "(auto)")
    )
    if effective_cuda:
        visible = report.get("visible_gpus") or []
        names = ", ".join(gpu["name"] for gpu in visible)
        parts.append(f"torch sees {report.get('visible_gpu_count', 0)} device(s) {names}")
    else:
        policy = report.get("device_policy") or "cuda"
        if policy.startswith("cpu"):
            parts.append(f"device policy {policy!r} -> CPU compute")
        else:
            parts.append("torch CUDA unavailable -> CPU compute")
    print("[gpu] " + " | ".join(parts))
    missing = _missing_requested_gpus(report)
    if missing:
        print(
            f"[gpu] WARNING requested GPU id(s) {missing} not found on the host; "
            "they will be invisible to CUDA",
            file=sys.stderr,
        )


def print_gpu_inventory(report: Mapping[str, Any]) -> None:
    """Print a human-readable GPU inventory and the effective mapping."""
    print("\nHost GPU inventory (physical indices, nvidia-smi):")
    host = report.get("host_gpus") or []
    if not host:
        print("  (no nvidia-smi data available)")
    for gpu in host:
        fields = [
            f"GPU {gpu['index']}: {gpu['name']}",
            f"{gpu.get('total_memory_gb')} GB total",
            f"{gpu.get('free_memory_gb')} GB free",
        ]
        util = gpu.get("utilization_percent")
        if util is not None:
            fields.append(f"{util}% util")
        if gpu.get("compute_capability"):
            fields.append(f"cap {gpu['compute_capability']}")
        print("  " + " | ".join(fields))

    requested = report.get("requested_ids") or []
    if requested:
        print(f"\nRequested GPU ids: {', '.join(map(str, requested))}")
    else:
        print("\nRequested GPU ids: (unset - torch default visibility)")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    print("\nTorch visibility:")
    if not report.get("torch_available"):
        print("  torch is not importable in this interpreter")
    elif not report.get("cuda_available"):
        print("  torch.cuda.is_available() = False (no visible CUDA device)")
    else:
        for gpu in report.get("visible_gpus") or []:
            print(
                "  "
                f"visible {gpu['visible_index']} -> {gpu['name']} "
                f"({gpu['total_memory_gb']} GB, cap {gpu['compute_capability']})"
            )
    missing = _missing_requested_gpus(report)
    if missing:
        print(
            f"\nWARNING: requested GPU id(s) {missing} are not present on the host "
            "and will be invisible to CUDA."
        )
    policy = report.get("device_policy") or "cuda"
    if policy.startswith("cpu"):
        print(f"\nNote: device policy is {policy!r} -> inference will run on CPU.")
    elif not report.get("cuda_available"):
        print("\nNote: no CUDA device is visible to torch -> inference will run on CPU.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "representative", "full"], default=os.environ.get("LONG_BENCH_MODE", "smoke"))
    parser.add_argument("--baselines", default=os.environ.get("LONG_BENCH_BASELINES", " ".join(BASELINES)))
    parser.add_argument("--datasets", default=os.environ.get("LONG_BENCH_DATASETS", " ".join(DATASETS)))
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("LONG_BENCH_DATA_DIR", "datasets/eval_100"))
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("LONG_BENCH_OUTPUT_DIR", "outputs/longbench_viet_100"))
    parser.add_argument("--model", default=os.environ.get("LONG_BENCH_MODEL") or os.environ.get("MODEL_TARGET"))
    parser.add_argument("--max-samples", "--samples-per-dataset", dest="max_samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="minimum free VRAM before launching children (default: 32; 0 disables)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-unsupported", action="store_true")
    parser.add_argument(
        "--gpu-ids",
        dest="gpu_ids",
        default=None,
        help="physical GPU id(s) to run on, e.g. '0', '2' or '0,1'; "
        "overrides LONG_BENCH_GPU_IDS / FI_GPU_IDS / CUDA_VISIBLE_DEVICES. "
        "With --data-parallel the list is the pool that gets partitioned into "
        "one device group per shard",
    )
    parser.add_argument(
        "--data-parallel",
        dest="data_parallel",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LONG_BENCH_DATA_PARALLEL", "0") == "1",
        help="run every (baseline, dataset) cell as N concurrent batch-1 child "
        "processes, one per GPU group, then merge the shards; default off "
        "(env: LONG_BENCH_DATA_PARALLEL=1)",
    )
    parser.add_argument(
        "--dp-gpus-per-shard",
        dest="dp_gpus_per_shard",
        type=int,
        default=None,
        help="GPUs owned by each data-parallel shard; 1 (the default) means one "
        "card per batch-1 child, 2+ is for methods that need intra-process "
        "parallelism (env: LONG_BENCH_DP_GPUS_PER_SHARD)",
    )
    parser.add_argument(
        "--dp-processes-per-gpu",
        dest="dp_processes_per_gpu",
        type=int,
        default=None,
        help="how many batch-1 processes may share one card; >1 uses spare "
        "VRAM to shorten a sweep but makes per-sample latency/throughput "
        "non-comparable (default: LONG_BENCH_DP_PROCESSES_PER_GPU or 1)",
    )
    parser.add_argument(
        "--vram-budget-gb",
        dest="vram_budget_gb",
        type=float,
        default=None,
        help="hard ceiling for VRAM usage per card; the planner never plans "
        "above it (default: LONG_BENCH_VRAM_BUDGET_GB or 170)",
    )
    parser.add_argument(
        "--vram-headroom-gb",
        dest="vram_headroom_gb",
        type=float,
        default=None,
        help="safety margin subtracted from the budget before packing children "
        "(default: LONG_BENCH_VRAM_HEADROOM_GB or 10)",
    )
    parser.add_argument(
        "--child-vram-gb",
        dest="child_vram_gb",
        type=float,
        default=None,
        help="conservative VRAM reserve per batch-1 child, used to size "
        "concurrency (default: LONG_BENCH_CHILD_VRAM_GB or 40; lower it only "
        "after checking measured peak_memory_gb)",
    )
    parser.add_argument(
        "--vram-wait-seconds",
        dest="vram_wait_seconds",
        type=int,
        default=None,
        help="how long a cell waits for free VRAM before it is recorded as "
        "vram_blocked instead of launching into an OOM (default: "
        "LONG_BENCH_VRAM_WAIT_SECONDS or 600)",
    )
    parser.add_argument(
        "--oom-retries",
        dest="oom_retries",
        type=int,
        default=None,
        help="how many times a shard that hit an OOM is retried alone, after "
        "its siblings exited (default: LONG_BENCH_OOM_RETRIES or 1; 0 disables)",
    )
    parser.add_argument(
        "--sample-retries",
        dest="sample_retries",
        type=int,
        default=None,
        help="retry each missing/failed sample independently after a cell or "
        "shard error (default: LONG_BENCH_SAMPLE_RETRIES or 2; 0 disables)",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        dest="retry_backoff_seconds",
        type=float,
        default=None,
        help="initial delay between sample retries; delay doubles per attempt "
        "(default: LONG_BENCH_RETRY_BACKOFF_SECONDS or 5)",
    )
    parser.add_argument(
        "--retry-failed-samples",
        dest="retry_failed_samples",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LONG_BENCH_RETRY_FAILED_SAMPLES", "1") == "1",
        help="recover missing/failed samples with isolated batch-1 attempts "
        "and emit null-timing status rows when exhausted (default: on)",
    )
    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="list host GPUs, the current selection and torch visibility, then exit",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LONG_BENCH_CONTINUE_ON_ERROR", "1") == "1",
        help="continue to the next cell when orchestration raises an exception "
        "(default: on)",
    )
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=os.environ.get("LONG_BENCH_STRICT", "1") == "1")
    parser.add_argument(
        "--collect",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LONG_BENCH_COLLECT", "1") == "1",
        help="run the metric collector over the finished run and write "
        "metrics_summary.{json,csv,md} into the run directory (default: on; "
        "always skipped for --preflight-only runs)",
    )
    return parser


def _execute_cell_once(
    *,
    baseline: str,
    dataset: str,
    source_rows: Sequence[Mapping[str, Any]],
    normalized: Sequence[Mapping[str, Any]],
    run_dir: Path,
    subset_path: Path,
    output_path: Path,
    cfg: Mapping[str, Any],
    dp_enabled: bool,
    dp_groups: Sequence[Sequence[int]],
    timeout_seconds: int,
    run_id: str,
    processes_per_gpu: int,
    vram: Mapping[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Build and launch one cell; callers decide whether to recover failures."""
    if dp_enabled:
        return _run_data_parallel_cell(
            baseline=baseline,
            dataset=dataset,
            source_rows=source_rows,
            normalized=normalized,
            run_dir=run_dir,
            output_path=output_path,
            cfg=cfg,
            gpu_groups=dp_groups,
            timeout_seconds=timeout_seconds,
            run_id=run_id,
            processes_per_gpu=processes_per_gpu,
            vram=vram,
        )

    converted = run_dir / "inputs" / f"{baseline}_{dataset}.jsonl"
    converted_input = convert_records_for_baseline(
        baseline, normalized, converted
    )
    command = build_adapter_command(
        baseline,
        data_file=subset_path,
        converted_input=converted_input,
        output=output_path,
        max_samples=len(normalized),
        max_new_tokens=max_new_tokens,
        config=cfg,
    )
    if command is None:
        return {
            "status": "unsupported_dataset",
            "reason": "adapter did not produce a command for this dataset",
            "returncode": None,
            "elapsed_ms": 0.0,
            "output_exists": output_path.is_file(),
            "log": "",
            "log_tail": "",
            "command": [],
            "data_parallel": False,
        }

    live_log = run_dir / "logs" / f"{baseline}_{dataset}.log"
    print(
        f"[{baseline}/{dataset}] launching {len(normalized)} sample(s)\n"
        f"[{baseline}/{dataset}] live log: {live_log}",
        flush=True,
    )
    return _run_child(
        command,
        output=output_path,
        log_path=live_log,
        timeout_seconds=timeout_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # GPU selection: --gpu-ids is authoritative; otherwise honour the existing
    # LONG_BENCH_GPU_IDS / FI_GPU_IDS / CUDA_VISIBLE_DEVICES chain. Applied
    # before torch is imported so enumeration and every child process observe
    # the same physical device(s). An explicit empty CUDA_VISIBLE_DEVICES
    # (CPU smoke) is left untouched.
    if args.gpu_ids is not None and args.gpu_ids.strip():
        _parse_gpu_ids(args.gpu_ids)  # fail fast on malformed values
        os.environ["LONG_BENCH_GPU_IDS"] = args.gpu_ids
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        selection = _gpu_selection_raw()
        if selection:
            os.environ["CUDA_VISIBLE_DEVICES"] = selection

    gpu_report = describe_gpu_assignment()
    cuda_available = _effective_cuda_available()

    # Data-parallel plan: partition the selected GPUs into one device group per
    # batch-1 shard.  Resolved before the early exit so --list-gpus can show the
    # exact groups a real run would use.
    dp_gpus_per_shard = (
        args.dp_gpus_per_shard
        if args.dp_gpus_per_shard is not None
        else _env_int("LONG_BENCH_DP_GPUS_PER_SHARD", 1)
    )
    if dp_gpus_per_shard < 1:
        raise SystemExit("--dp-gpus-per-shard must be >= 1")

    # VRAM budget: the planner packs batch-1 children up to
    # ``budget - headroom`` per card, and refuses to launch (waiting instead)
    # when nothing fits.  0 disables budget planning entirely.
    vram_budget_gb = (
        args.vram_budget_gb
        if args.vram_budget_gb is not None
        else _env_float("LONG_BENCH_VRAM_BUDGET_GB", 170.0)
    )
    vram_headroom_gb = (
        args.vram_headroom_gb
        if args.vram_headroom_gb is not None
        else _env_float("LONG_BENCH_VRAM_HEADROOM_GB", 10.0)
    )
    child_vram_gb = (
        args.child_vram_gb
        if args.child_vram_gb is not None
        else _env_float("LONG_BENCH_CHILD_VRAM_GB", 40.0)
    )
    vram_wait_seconds = (
        args.vram_wait_seconds
        if args.vram_wait_seconds is not None
        else _env_int("LONG_BENCH_VRAM_WAIT_SECONDS", 600)
    )
    oom_retries = (
        args.oom_retries
        if args.oom_retries is not None
        else _env_int("LONG_BENCH_OOM_RETRIES", 1)
    )
    sample_retries = (
        args.sample_retries
        if args.sample_retries is not None
        else _env_int("LONG_BENCH_SAMPLE_RETRIES", 2)
    )
    try:
        retry_backoff_seconds = (
            args.retry_backoff_seconds
            if args.retry_backoff_seconds is not None
            else float(os.environ.get("LONG_BENCH_RETRY_BACKOFF_SECONDS", "5"))
        )
    except ValueError as exc:
        raise SystemExit(
            "LONG_BENCH_RETRY_BACKOFF_SECONDS must be a number"
        ) from exc
    processes_per_gpu = (
        args.dp_processes_per_gpu
        if args.dp_processes_per_gpu is not None
        else _env_int("LONG_BENCH_DP_PROCESSES_PER_GPU", 1)
    )
    if vram_budget_gb < 0:
        raise SystemExit("--vram-budget-gb must be >= 0 (0 disables planning)")
    if vram_headroom_gb < 0:
        raise SystemExit("--vram-headroom-gb must be >= 0")
    if child_vram_gb <= 0:
        raise SystemExit("--child-vram-gb must be > 0")
    if vram_wait_seconds < 0:
        raise SystemExit("--vram-wait-seconds must be >= 0")
    if oom_retries < 0:
        raise SystemExit("--oom-retries must be >= 0")
    if sample_retries < 0:
        raise SystemExit("--sample-retries must be >= 0")
    if retry_backoff_seconds < 0:
        raise SystemExit("--retry-backoff-seconds must be >= 0")
    if processes_per_gpu < 1:
        raise SystemExit("--dp-processes-per-gpu must be >= 1")
    if vram_budget_gb > 0 and vram_headroom_gb >= vram_budget_gb:
        raise SystemExit(
            f"--vram-headroom-gb {vram_headroom_gb} must be smaller than "
            f"--vram-budget-gb {vram_budget_gb}, otherwise no child can be planned"
        )
    vram_cfg = {
        "budget_gb": vram_budget_gb,
        "headroom_gb": vram_headroom_gb,
        "usable_gb": (
            None if vram_budget_gb <= 0
            else round(vram_budget_gb - vram_headroom_gb, 1)
        ),
        "child_reserve_gb": child_vram_gb,
        "wait_seconds": vram_wait_seconds,
        "oom_retries": oom_retries,
        "sample_retries": sample_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
    }

    # Parallel execution needs either several GPU groups or several processes
    # sharing a card; both go through the same shard/merge machinery.
    parallel_requested = bool(args.data_parallel) or processes_per_gpu > 1
    dp_groups = (
        _resolve_dp_gpu_groups(
            gpu_report,
            cuda_available=cuda_available,
            gpus_per_shard=dp_gpus_per_shard,
        )
        if parallel_requested
        else []
    )

    if args.list_gpus:
        print_gpu_inventory(gpu_report)
        if parallel_requested:
            # Preview straight from the physical inventory so the plan is
            # visible even when torch itself cannot see CUDA (CPU dev boxes).
            preview_pool = dp_groups or _resolve_dp_gpu_groups(
                gpu_report, cuda_available=True, gpus_per_shard=dp_gpus_per_shard
            )
            preview_slots, preview_plan = plan_shard_slots(
                preview_pool,
                processes_per_gpu=processes_per_gpu,
                usable_gb=vram_cfg["usable_gb"],
                child_gb=child_vram_gb,
                sample_count=1_000_000,
                usage=_vram_usage_by_gpu(),
            )
            print(
                "\nParallel plan "
                f"(--dp-gpus-per-shard {dp_gpus_per_shard}, "
                f"--dp-processes-per-gpu {processes_per_gpu}, budget "
                f"{vram_budget_gb:.0f} GiB, usable {vram_cfg['usable_gb']} GiB, "
                f"reserve {child_vram_gb:.0f} GiB/child):"
            )
            for entry in preview_plan["groups"]:
                free = entry["free_gb"]
                print(
                    f"  GPU {', '.join(map(str, entry['gpu_ids']))}: "
                    f"{entry['planned_processes']} process(es)"
                    + (f", free {free} GiB" if free is not None else "")
                    + (
                        f", card total {entry['total_gb']} GiB"
                        if entry["total_gb"] is not None
                        else ""
                    )
                )
            print(
                f"  -> {len(preview_slots)} concurrent batch-1 child(ren) per cell"
            )
            if not cuda_available:
                print(
                    "\nNote: torch cannot see CUDA on this host, so a real run "
                    "would fall back to one CPU process per cell; the plan above "
                    "is what the GPUs would allow."
                )
        return 0

    print_gpu_summary(gpu_report, effective_cuda=cuda_available)
    dp_enabled = (
        bool(dp_groups)
        and (len(dp_groups) > 1 or processes_per_gpu > 1)
        and not args.preflight_only
    )
    if parallel_requested and not dp_enabled:
        if args.preflight_only:
            reason = "--preflight-only never launches inference children"
        elif not cuda_available:
            reason = "CUDA is not available to this process"
        elif not dp_groups:
            reason = "no GPU was selected"
        else:
            reason = "only one GPU group and --dp-processes-per-gpu 1"
        print(
            f"[parallel] requested but inactive ({reason}); falling back to one "
            "batch-1 process per cell",
            file=sys.stderr,
            flush=True,
        )
    if dp_enabled:
        _, startup_plan = plan_shard_slots(
            dp_groups,
            processes_per_gpu=processes_per_gpu,
            usable_gb=vram_cfg["usable_gb"],
            child_gb=child_vram_gb,
            sample_count=1_000_000,
            usage=_vram_usage_by_gpu(),
        )
        print(
            "[parallel] enabled: "
            + ", ".join(
                f"GPU {','.join(map(str, entry['gpu_ids']))} -> "
                f"{entry['planned_processes']}x batch-1"
                + (
                    f" (free {entry['free_gb']} GiB)"
                    if entry["free_gb"] is not None
                    else ""
                )
                for entry in startup_plan["groups"]
            )
            + f" | VRAM budget {vram_budget_gb:.0f} GiB (usable "
            f"{vram_cfg['usable_gb']} GiB, reserve {child_vram_gb:.0f} GiB/child, "
            f"wait {vram_wait_seconds}s, oom-retries {oom_retries})",
            flush=True,
        )
        if processes_per_gpu > 1:
            print(
                "[parallel] WARNING multiple batch-1 processes per card: the "
                "sweep finishes sooner but per-sample latency/throughput are no "
                "longer comparable with one-process-per-card runs",
                file=sys.stderr,
                flush=True,
            )
    profile = resolve_profile(
        mode=args.mode,
        cuda_available=cuda_available,
        allow_unsupported=args.allow_unsupported,
    )

    data_dir = _resolve(args.data_dir)
    output_root = _resolve(args.output_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"LongBench data directory not found: {data_dir}")
    dataset_profile_count = _dataset_profile_count(data_dir)
    try:
        # Validate the source before launching any model.  The manifest allows
        # both the canonical 200-row profile and derived profiles such as the
        # balanced 100-row/14k test set.
        validate_output_dir(data_dir, expected_count=dataset_profile_count)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid canonical LongBench data: {exc}") from exc

    requested_baselines = _split(args.baselines)
    unknown_baselines = sorted(set(requested_baselines) - set(SUPPORTED_BASELINES))
    if unknown_baselines:
        raise SystemExit(f"Unknown baseline(s): {', '.join(unknown_baselines)}")
    baselines, skipped_baselines = _filter_matrix_baselines(requested_baselines)
    if skipped_baselines:
        print(
            "[matrix] skipping disabled/non-comparable baseline(s): "
            + ", ".join(skipped_baselines)
        )
    datasets = _split(args.datasets)
    unknown_datasets = sorted(set(datasets) - set(DATASETS))
    if unknown_datasets:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown_datasets)}")
    if not baselines or not datasets:
        raise SystemExit("At least one baseline and dataset are required")

    if args.mode == "representative" and args.datasets == " ".join(DATASETS):
        configured = _split(os.environ.get("LONG_BENCH_REPRESENTATIVE_DATASETS", " ".join(DATASETS)))
        if configured:
            datasets = configured
    default_sample_count = {
        "smoke": _env_int("LONG_BENCH_SMOKE_SAMPLES", 1),
        "representative": _env_int("LONG_BENCH_REPRESENTATIVE_SAMPLES", 20),
        "full": _env_int("LONG_BENCH_FULL_SAMPLES", 100),
    }[args.mode]
    # A full run over a derived LongBench-100 profile should naturally use all
    # 100 rows.  Explicit --max-samples remains authoritative for smaller
    # smoke/representative subsets or for controlled ablations.
    sample_count = args.max_samples or min(default_sample_count, dataset_profile_count)
    max_new_tokens = args.max_new_tokens or {
        "smoke": _env_int("LONG_BENCH_SMOKE_MAX_NEW_TOKENS", 8),
        "representative": _env_int("LONG_BENCH_MAX_NEW_TOKENS", 2048),
        "full": _env_int("LONG_BENCH_MAX_NEW_TOKENS", 2048),
    }[args.mode]
    seed = args.seed if args.seed is not None else _env_int("LONG_BENCH_SEED", 42)
    temperature = args.temperature if args.temperature is not None else float(os.environ.get("LONG_BENCH_TEMPERATURE", "0"))
    warmup_runs = args.warmup_runs if args.warmup_runs is not None else _env_int("LONG_BENCH_WARMUP_RUNS", 3)
    max_input_tokens = resolve_max_input_tokens(args.mode, args.max_input_tokens)
    min_free_gb = (
        args.min_free_gb
        if args.min_free_gb is not None
        else _env_float("LONG_BENCH_MIN_FREE_GB", 32.0)
    )
    if min_free_gb < 0:
        raise SystemExit("--min-free-gb/LONG_BENCH_MIN_FREE_GB must be >= 0")
    gpu_guard_reason = gpu_memory_guard_reason(
        gpu_report, min_free_gb=min_free_gb
    ) if cuda_available else None
    if gpu_guard_reason and not args.preflight_only:
        print(f"[gpu] VRAM guard: {gpu_guard_reason}", file=sys.stderr)
        return 2
    timeout_seconds = resolve_timeout_seconds(args.mode, args.timeout_seconds)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "schema_version": "longbench-run-v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "preflight_only": bool(args.preflight_only),
        "data_dir": str(data_dir),
        "output_dir": str(run_dir),
        "source_manifest_sha256": _source_manifest_hash(data_dir),
        "model": args.model,
        "requested_baselines": requested_baselines,
        "baselines": baselines,
        "skipped_baselines": skipped_baselines,
        "datasets": datasets,
        "dataset_profile_count": dataset_profile_count,
        "sample_count": sample_count,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "warmup_runs": warmup_runs,
        "max_input_tokens": max_input_tokens,
        "min_free_gb": min_free_gb,
        "gpu_guard_reason": gpu_guard_reason,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "strict": bool(args.strict),
        "continue_on_error": bool(args.continue_on_error),
        "retry_failed_samples": bool(args.retry_failed_samples),
        "sample_retries": sample_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "allow_unsupported": bool(args.allow_unsupported),
        "gpu_ids": gpu_report.get("requested"),
        "gpu": gpu_report,
        "data_parallel": dp_enabled,
        "dp_requested": bool(parallel_requested),
        "dp_gpus_per_shard": dp_gpus_per_shard if dp_enabled else 1,
        "dp_processes_per_gpu": processes_per_gpu if dp_enabled else 1,
        "dp_world_size": len(dp_groups) if dp_enabled else 1,
        "dp_gpu_groups": [list(group) for group in dp_groups] if dp_enabled else [],
        "dp_aggregate_only_baselines": sorted(AGGREGATE_ONLY_BASELINES & set(baselines)),
        "vram": {
            "budget_gb": vram_budget_gb,
            "headroom_gb": vram_headroom_gb,
            "usable_gb": vram_cfg["usable_gb"],
            "child_reserve_gb": child_vram_gb,
            "wait_seconds": vram_wait_seconds,
            "oom_retries": oom_retries,
        },
        "runtime": runtime_metadata(),
        "cells": [],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    print(f"Run directory: {run_dir}", flush=True)
    print(
        f"Run manifest (live): {run_dir / 'run_manifest.json'}",
        flush=True,
    )

    def _append_cell(cell: Mapping[str, Any]) -> None:
        """Persist every completed cell so an interrupted sweep stays useful."""
        manifest["cells"].append(dict(cell))
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "run_manifest.json", manifest)

    failures = 0
    for dataset in datasets:
        source_rows, normalized = _load_selected(data_dir, dataset, sample_count, seed=seed)
        subset_path = run_dir / "inputs" / f"{dataset}.jsonl"
        _write_jsonl(subset_path, source_rows)
        for baseline in baselines:
            output_path = run_dir / baseline / f"{dataset}.jsonl"
            external_reference_path = None
            external_reference_baseline = None
            if baseline in EXTERNAL_REFERENCE_BASELINES:
                external_reference_path = _select_external_reference(
                    run_dir, dataset, baselines
                )
                if external_reference_path is not None:
                    external_reference_baseline = external_reference_path.parent.name
            cfg = baseline_config_from_env(baseline)
            cfg.update(
                model=args.model or cfg.get("model"),
                device=os.environ.get("LONG_BENCH_DEVICE", cfg.get("device", "cuda")),
                temperature=temperature,
                warmup_runs=warmup_runs,
                max_input_tokens=max_input_tokens,
                seed=seed,
                smoke=args.mode == "smoke",
                max_new_tokens=max_new_tokens,
            )
            cfg["skip_reference"] = external_reference_path is not None
            try:
                check = preflight_baseline(
                    baseline,
                    config=cfg,
                    cuda_available=cuda_available,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                check = {
                    "baseline": baseline,
                    "status": "preflight_error",
                    "reason": (
                        f"preflight exception: {type(exc).__name__}: {exc}"
                    ),
                    "requires_cuda": baseline in CUDA_BASELINES,
                    "cuda_available": bool(cuda_available),
                    "requirements": {},
                }
                print(
                    f"[{baseline}/{dataset}] preflight exception; continuing: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            cell: dict[str, Any] = {
                "baseline": baseline,
                "dataset": dataset,
                "sample_count": len(normalized),
                "preflight": check,
                "output": str(output_path),
            }
            if external_reference_path is not None:
                cell.update(
                    external_reference=str(external_reference_path),
                    external_reference_baseline=external_reference_baseline,
                    speedup_scope="external_reference",
                )
            if args.preflight_only:
                status = check["status"] if check["status"] != "ready" else "preflight_only"
                reason = check["reason"] or "preflight completed; inference was not requested"
                _write_status_file(
                    output_path,
                    baseline=baseline,
                    dataset=dataset,
                    records=normalized,
                    status=status,
                    reason=reason,
                    model=cfg.get("model"),
                    config=cfg,
                    run_id=run_id,
                )
                cell.update(status=status, reason=reason, returncode=0)
                cell.update(
                    _audit_cell_output(
                        output_path,
                        baseline=baseline,
                        dataset=dataset,
                        run_dir=run_dir,
                        expected_output_tokens=max_new_tokens,
                        expected_samples=len(normalized),
                    )
                )
                _append_cell(cell)
                print(f"[{baseline}/{dataset}] {status}: {reason}", flush=True)
                continue

            if check["status"] not in {"ready", "aggregate_only"}:
                if args.strict and not args.allow_unsupported and args.mode != "smoke":
                    failures += 1
                _write_status_file(
                    output_path,
                    baseline=baseline,
                    dataset=dataset,
                    records=normalized,
                    status=check["status"],
                    reason=check["reason"] or "baseline preflight did not pass",
                    model=cfg.get("model"),
                    config=cfg,
                    run_id=run_id,
                )
                cell.update(status=check["status"], reason=check["reason"])
                cell.update(
                    _audit_cell_output(
                        output_path,
                        baseline=baseline,
                        dataset=dataset,
                        run_dir=run_dir,
                        expected_output_tokens=max_new_tokens,
                        expected_samples=len(normalized),
                    )
                )
                _append_cell(cell)
                print(
                    f"[{baseline}/{dataset}] {check['status']}: {check['reason']}",
                    flush=True,
                )
                continue

            try:
                child = _execute_cell_once(
                    baseline=baseline,
                    dataset=dataset,
                    source_rows=source_rows,
                    normalized=normalized,
                    run_dir=run_dir,
                    subset_path=subset_path,
                    output_path=output_path,
                    cfg=cfg,
                    dp_enabled=dp_enabled,
                    dp_groups=dp_groups,
                    timeout_seconds=timeout_seconds,
                    run_id=run_id,
                    processes_per_gpu=processes_per_gpu,
                    vram=vram_cfg,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                child = {
                    "status": "failed",
                    "returncode": None,
                    "elapsed_ms": 0.0,
                    "output_exists": output_path.is_file(),
                    "log": "",
                    "log_tail": "",
                    "command": [],
                    "reason": (
                        f"orchestration exception: {type(exc).__name__}: {exc}"
                    ),
                    "data_parallel": dp_enabled,
                }
                print(
                    f"[{baseline}/{dataset}] orchestration exception; "
                    "continuing to next cell: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

            if child["status"] == "unsupported_dataset":
                reason = child.get("reason") or (
                    "adapter did not produce a command for this dataset"
                )
                _write_status_file(
                    output_path,
                    baseline=baseline,
                    dataset=dataset,
                    records=normalized,
                    status="unsupported_dataset",
                    reason=reason,
                    model=cfg.get("model"),
                    config=cfg,
                    run_id=run_id,
                )
                cell.update(status="unsupported_dataset", reason=reason)
                cell.update(
                    _audit_cell_output(
                        output_path,
                        baseline=baseline,
                        dataset=dataset,
                        run_dir=run_dir,
                        expected_output_tokens=max_new_tokens,
                        expected_samples=len(normalized),
                    )
                )
                _append_cell(cell)
                continue

            initial_status = str(child.get("status") or "failed")
            if dp_enabled:
                # Shards were already normalized before the merge; the merged
                # file has been normalized too.
                normalized_count = int(child.get("normalized_records") or 0)
            elif output_path.is_file():
                try:
                    normalized_count = _normalize_child_output(
                        output_path,
                        baseline=baseline,
                        dataset=dataset,
                        source_records=normalized,
                        config=cfg,
                        run_id=run_id,
                    )
                except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    normalized_count = 0
                    child["reason"] = (
                        f"could not normalize child output: {type(exc).__name__}: {exc}"
                    )
                    child["status"] = "failed"
                child["normalized_records"] = normalized_count
            else:
                normalized_count = 0

            if initial_status == "success" and normalized_count == 0:
                child["status"] = "failed"
                child["reason"] = child.get("reason") or (
                    "child exited successfully but wrote no result records"
                )

            # Reconcile every cell, even after a successful process exit:
            # adapters can omit individual rows or die after writing a prefix.
            # Retry attempts are isolated batch-1 children and can use the
            # same GPU groups as the data-parallel run, one sample at a time.
            try:
                safe_result = _retry_unresolved_samples(
                    baseline=baseline,
                    dataset=dataset,
                    source_rows=source_rows,
                    normalized=normalized,
                    output_path=output_path,
                    run_dir=run_dir,
                    cfg=cfg,
                    max_new_tokens=max_new_tokens,
                    timeout_seconds=timeout_seconds,
                    run_id=run_id,
                    sample_retries=(
                        sample_retries if args.retry_failed_samples else 0
                    ),
                    retry_backoff_seconds=retry_backoff_seconds,
                    retry_device_groups=dp_groups if dp_enabled else None,
                    retry_usable_gb=(
                        vram_cfg.get("usable_gb") if dp_enabled else None
                    ),
                    retry_child_gb=float(vram_cfg.get("child_reserve_gb") or 1.0),
                    retry_wait_seconds=float(vram_cfg.get("wait_seconds") or 0.0),
                    reference_path=external_reference_path,
                    reference_baseline=(
                        str(external_reference_baseline)
                        if external_reference_baseline
                        else None
                    ),
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                reason = (
                    f"safe recovery exception: {type(exc).__name__}: {exc}"
                )
                print(
                    f"[{baseline}/{dataset}] {reason}; writing failure rows "
                    "and continuing",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    fallback_rows = _jsonl_rows(output_path)
                except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
                    fallback_rows = []
                safe_result = _rewrite_safe_cell_output(
                    output_path,
                    baseline=baseline,
                    dataset=dataset,
                    source_records=normalized,
                    successful_rows=[
                        row
                        for row in fallback_rows
                        if row.get("type") != "summary"
                        and row.get("status", "success") == "success"
                    ],
                    unresolved_reasons={
                        str(row.get("id")): reason for row in normalized
                    },
                    model=str(cfg.get("model") or "") or None,
                    config=cfg,
                    run_id=run_id,
                    retry_count=0,
                )
                safe_result["retry_error"] = reason

            child.update(safe_result)
            if safe_result.get("safe_eval_complete"):
                if initial_status != "success":
                    child["initial_status"] = initial_status
                    child["retry_recovered"] = True
                child["status"] = "success"
                child["returncode"] = 0
            else:
                child["status"] = "failed"
                child["reason"] = (
                    f"{safe_result.get('unresolved_sample_count', 0)} sample(s) "
                    "remain unresolved after safe retries"
                )
            if external_reference_path is not None and output_path.is_file():
                attached = _attach_external_reference_metrics(
                    output_path,
                    external_reference_path,
                    reference_baseline=str(external_reference_baseline),
                )
                child["external_reference_records"] = attached
                if attached == 0:
                    child["external_reference_warning"] = (
                        "no sample_id matched the selected Vanilla reference"
                    )
            cell.update(child)
            if child["status"] != "success":
                failures += 1
                if not output_path.is_file():
                    _write_status_file(
                        output_path,
                        baseline=baseline,
                        dataset=dataset,
                        records=normalized,
                        status=child["status"],
                        reason=child.get("reason")
                        or f"child process failed; see {child['log']}",
                        model=cfg.get("model"),
                        config=cfg,
                        run_id=run_id,
                    )
            audit_result = _audit_cell_output(
                output_path,
                baseline=baseline,
                dataset=dataset,
                run_dir=run_dir,
                expected_output_tokens=max_new_tokens,
                expected_samples=len(normalized),
            )
            cell.update(audit_result)
            contract = audit_result.get("metric_contract") or {}
            if (
                child.get("status") == "success"
                and args.strict
                and not args.allow_unsupported
                and contract.get("status") != "complete"
            ):
                child["status"] = "metric_incomplete"
                child["reason"] = (
                    "metric contract failed: "
                    f"{contract.get('issue_counts', {})}"
                )
                cell.update(status="metric_incomplete", reason=child["reason"])
                failures += 1
            _append_cell(cell)
            print(
                f"[{baseline}/{dataset}] {child['status']} in "
                f"{child['elapsed_ms']} ms",
                flush=True,
            )

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failure_count"] = failures
    manifest["cell_count"] = len(manifest["cells"])

    # Aggregate metrics as part of the run so every finished run ships its own
    # metrics_summary.{json,csv,md}.  Preflight-only runs write status rows
    # instead of inference records, so aggregation is skipped for them.  Strict
    # completeness is only meaningful when every cell actually succeeded;
    # ``failures == 0`` is not enough because smoke and --allow-unsupported
    # runs may record blocked cells without counting them as failures.
    clean_cells = bool(manifest["cells"]) and all(
        cell.get("status") == "success" for cell in manifest["cells"]
    )
    if args.collect and not args.preflight_only:
        aggregate = _run_collector(
            run_dir,
            data_dir,
            baselines=baselines,
            datasets=datasets,
            expected_samples=sample_count,
            strict=bool(args.strict) and clean_cells,
            timeout_seconds=timeout_seconds,
        )
        if aggregate["status"] == "success":
            print(
                "[aggregate] metrics_summary.{json,csv,md} written to "
                f"{run_dir}",
                flush=True,
            )
        else:
            print(
                "[aggregate] collector failed "
                f"(exit {aggregate.get('returncode')}); see {aggregate['log']}",
                file=sys.stderr,
                flush=True,
            )
    elif args.preflight_only:
        aggregate = {
            "status": "skipped",
            "reason": "preflight-only run has no inference records to aggregate",
        }
    else:
        aggregate = {
            "status": "skipped",
            "reason": "metric collection disabled with --no-collect",
        }
    manifest["aggregate"] = aggregate
    _write_json(run_dir / "run_manifest.json", manifest)
    print(f"Run manifest: {run_dir / 'run_manifest.json'}", flush=True)
    strict_aggregate_failed = (
        aggregate.get("status") == "failed" and bool(aggregate.get("strict"))
    )
    return 1 if (failures or strict_aggregate_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
