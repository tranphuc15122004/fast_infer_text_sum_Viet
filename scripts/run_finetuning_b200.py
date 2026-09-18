#!/usr/bin/env python3
"""Resumable end-to-end DFlash fine-tuning launcher for a B200 server."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import shlex
import subprocess
import sys
import tempfile
from typing import Callable, Iterator, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = os.environ.get("FINETUNING_PYTHON", sys.executable)
_CURRENT_PROCESS: subprocess.Popen[str] | None = None


class LauncherError(RuntimeError):
    """A user-actionable configuration, stage, or artifact failure."""


@dataclass(frozen=True)
class RunPaths:
    output_root: Path
    teacher_train: Path
    teacher_eval: Path
    features_train: Path
    features_eval: Path
    checkpoints: Path
    run_config: Path
    run_manifest: Path
    state_dir: Path
    log_dir: Path
    lock_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LauncherError(f"config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise LauncherError(f"invalid YAML config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LauncherError(f"config root must be a mapping: {path}")
    return payload


def _mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise LauncherError(f"config section {key!r} must be a mapping")
    return value


def _safe_run_id(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise LauncherError("run_id must be a simple directory-safe name")
    return value


def resolve_paths(args: argparse.Namespace, run_id: str) -> RunPaths:
    output_root = Path(args.output_root).expanduser().resolve()
    return RunPaths(
        output_root=output_root,
        teacher_train=output_root / "teacher" / "train.jsonl",
        teacher_eval=output_root / "teacher" / "eval.jsonl",
        features_train=output_root / "features" / "train",
        features_eval=output_root / "features" / "eval",
        checkpoints=output_root / "checkpoints",
        run_config=output_root / "run_config.yaml",
        run_manifest=output_root / "run_manifest.json",
        state_dir=output_root / ".state",
        log_dir=output_root / "logs",
        lock_path=output_root / ".run.lock",
    )


def _config_value(
    mapping: dict[str, object], key: str, *, minimum: int = 1
) -> object:
    value = mapping.get(key)
    if value is None:
        raise LauncherError(f"resolved config requires {key!r}")
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise LauncherError(
            f"resolved config field {key!r} must be an integer >= {minimum}"
        )
    return value


def materialize_config(
    source_path: Path,
    destination: Path,
    paths: RunPaths,
    args: argparse.Namespace,
) -> dict[str, object]:
    payload = copy.deepcopy(_load_yaml(source_path))
    model = _mapping(payload, "model")
    data = _mapping(payload, "data")
    training = _mapping(payload, "training")

    target_model = args.target_model_path or model.get("target_model_path")
    if not isinstance(target_model, str) or not target_model:
        raise LauncherError(
            "target model is missing; pass --target-model-path or set model.target_model_path"
        )
    model["target_model_path"] = str(Path(target_model).expanduser().resolve())

    model_dtype = model.get("torch_dtype", "float32")
    feature_dtype = data.get("feature_dtype", model_dtype)
    if model_dtype != feature_dtype:
        raise LauncherError(
            "model.torch_dtype and data.feature_dtype must match because "
            "capture_features stores the requested model dtype"
        )

    _config_value(data, "max_length", minimum=1)
    _config_value(data, "max_source_tokens", minimum=0)
    _config_value(data, "max_summary_tokens", minimum=1)
    prompt_template = data.get("prompt_template")
    if not isinstance(prompt_template, str) or prompt_template.count("{document}") != 1:
        raise LauncherError(
            "data.prompt_template must contain {document} exactly once"
        )

    data["train_data_path"] = None
    data["eval_data_path"] = None
    data["hidden_states_path"] = str(paths.features_train)
    data["eval_hidden_states_path"] = str(paths.features_eval)

    payload["output_dir"] = str(paths.checkpoints)
    payload["device"] = "cuda"
    payload["offline"] = True
    if args.run_id is not None:
        payload["run_id"] = args.run_id
    run_id = _safe_run_id(payload.get("run_id", "dflash"))

    training["adaptive_batch_size"] = True
    if args.target_memory_fraction is not None:
        training["target_memory_fraction"] = args.target_memory_fraction
    if args.adaptive_min_batch_size is not None:
        training["adaptive_min_batch_size"] = args.adaptive_min_batch_size
    if args.adaptive_max_batch_size is not None:
        training["adaptive_max_batch_size"] = args.adaptive_max_batch_size
    if args.probe_batches is not None:
        training["adaptive_probe_batches"] = args.probe_batches
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size

    if not isinstance(model.get("num_draft_layers"), int) or model["num_draft_layers"] < 1:
        if not isinstance(model.get("target_layer_ids"), list) or not model["target_layer_ids"]:
            raise LauncherError(
                "model.num_draft_layers or model.target_layer_ids is required"
            )

    serialized = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    if destination.exists() and any(paths.state_dir.glob("*.json")):
        existing = _load_yaml(destination)
        if existing != payload:
            raise LauncherError(
                f"resolved config differs from existing resumable run: {destination}"
            )
    else:
        _atomic_write_text(destination, serialized)
    return payload


def _gpu_count_from_nvidia_smi() -> int:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise LauncherError("nvidia-smi is required when --nproc-per-node is omitted")
    result = subprocess.run(
        [executable, "-L"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise LauncherError(f"nvidia-smi -L failed: {result.stderr.strip()}")
    count = sum(bool(line.strip()) for line in result.stdout.splitlines())
    if count <= 0:
        raise LauncherError("nvidia-smi reported no visible GPUs")
    return count


def _python_check(
    python_bin: str,
    *,
    env: dict[str, str],
    require_cuda: bool,
    expected_gpu_count: int,
) -> None:
    code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        f"count=torch.cuda.device_count(); assert count >= {expected_gpu_count}, "
        "f'visible GPU count {count} is smaller than requested'; "
        "print(torch.__version__, count)"
        if require_cuda
        else "import torch; print(torch.__version__)"
    )
    result = subprocess.run(
        [python_bin, "-c", code],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise LauncherError(
            f"Python/CUDA preflight failed for {python_bin}:\n"
            f"{result.stdout}{result.stderr}"
        )


def _module_help(python_bin: str, module: str, env: dict[str, str]) -> None:
    result = subprocess.run(
        [python_bin, "-m", module, "--help"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise LauncherError(
            f"module preflight failed for {module}:\n{result.stdout}{result.stderr}"
        )


def validate_preflight(
    config: dict[str, object],
    paths: RunPaths,
    *,
    python_bin: str,
    nproc_per_node: int,
    train_input: Path,
    eval_input: Path,
    dry_run: bool,
) -> None:
    if not Path(python_bin).is_file() and shutil.which(python_bin) is None:
        raise LauncherError(f"Python interpreter not found: {python_bin}")
    for input_path in (train_input, eval_input):
        if not input_path.is_file():
            raise LauncherError(f"input JSONL not found: {input_path}")

    model = _mapping(config, "model")
    target_model = Path(str(model["target_model_path"]))
    if not target_model.is_dir() or not (target_model / "config.json").is_file():
        raise LauncherError(
            f"target model must be a local snapshot containing config.json: {target_model}"
        )

    if nproc_per_node <= 0:
        raise LauncherError("--nproc-per-node must be positive")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    _python_check(
        python_bin,
        env=env,
        require_cuda=not dry_run,
        expected_gpu_count=nproc_per_node,
    )
    for module in (
        "Finetuning.generate_targets",
        "Finetuning.capture_features",
        "Finetuning.run_train",
    ):
        _module_help(python_bin, module, env)


def _adaptive_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--adaptive-batch",
        "--target-memory-fraction",
        str(args.target_memory_fraction),
        "--adaptive-min-batch-size",
        str(args.adaptive_min_batch_size),
        "--adaptive-max-batch-size",
        str(args.adaptive_max_batch_size),
        "--max-tokens-per-batch",
        str(args.max_tokens_per_batch),
        "--bucket-window",
        str(args.bucket_window),
        "--probe-batches",
        str(args.probe_batches),
    ]
    return values


def _distributed_prefix(python_bin: str, nproc_per_node: int) -> list[str]:
    return [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
    ]


def build_commands(
    config: dict[str, object],
    paths: RunPaths,
    args: argparse.Namespace,
    *,
    python_bin: str,
    nproc_per_node: int,
) -> list[tuple[str, list[str], Path]]:
    model = _mapping(config, "model")
    data = _mapping(config, "data")
    model_path = str(model["target_model_path"])
    max_length = str(data["max_length"])
    max_source = str(data["max_source_tokens"])
    max_summary = str(data["max_summary_tokens"])
    chat_template = str(data.get("chat_template", "qwen3"))
    prompt_template = str(data["prompt_template"])
    dtype = str(model.get("torch_dtype", "float32"))
    capture_backend = str(args.capture_backend)
    capture_method = str(args.capture_method)
    prep_common = [
        "--target-model-path",
        model_path,
        "--max-length",
        max_length,
        "--max-source-tokens",
        max_source,
        "--max-summary-tokens",
        max_summary,
        "--chat-template",
        chat_template,
        "--prompt-template",
        prompt_template,
        "--torch-dtype",
        dtype,
        "--device",
        "cuda",
    ]
    adaptive = _adaptive_args(args)
    prefix = _distributed_prefix(python_bin, nproc_per_node)

    generate_base = [*prefix, "-m", "Finetuning.generate_targets", *prep_common, *adaptive]
    cache_base = [
        *prefix,
        "-m",
        "Finetuning.capture_features",
        *prep_common,
        *adaptive,
        "--capture-backend",
        capture_backend,
        "--capture-method",
        capture_method,
        "--sglang-tp-size",
        str(args.sglang_tp_size),
        "--sglang-attention-backend",
        str(args.sglang_attention_backend),
        "--sglang-mem-fraction-static",
        str(args.sglang_mem_fraction_static),
        "--sglang-max-running-requests",
        str(args.sglang_max_running_requests),
        "--sglang-max-total-tokens",
        str(args.sglang_max_total_tokens),
        "--parity-samples",
        str(args.parity_samples),
        "--parity-max-abs-error",
        str(args.parity_max_abs_error),
        "--parity-mean-abs-error",
        str(args.parity_mean_abs_error),
        "--parity-relative-l2-error",
        str(args.parity_relative_l2_error),
        "--parity-min-cosine-similarity",
        str(args.parity_min_cosine_similarity),
    ]
    if args.sglang_context_length is not None:
        cache_base.extend(["--sglang-context-length", str(args.sglang_context_length)])
    if args.sglang_disable_radix_cache:
        cache_base.append("--sglang-disable-radix-cache")

    target_layer_ids = model.get("target_layer_ids")
    if isinstance(target_layer_ids, list) and target_layer_ids:
        layer_args = ["--target-layer-ids", ",".join(str(item) for item in target_layer_ids)]
    else:
        layer_args = ["--num-draft-layers", str(model["num_draft_layers"])]

    return [
        (
            "generate_train",
            [
                *generate_base,
                "--input",
                str(args.train_input),
                "--output",
                str(paths.teacher_train),
            ],
            paths.teacher_train,
        ),
        (
            "generate_eval",
            [
                *generate_base,
                "--input",
                str(args.eval_input),
                "--output",
                str(paths.teacher_eval),
            ],
            paths.teacher_eval,
        ),
        (
            "cache_train",
            [
                *cache_base,
                "--input",
                str(paths.teacher_train),
                "--output",
                str(paths.features_train),
                *layer_args,
            ],
            paths.features_train,
        ),
        (
            "cache_eval",
            [
                *cache_base,
                "--input",
                str(paths.teacher_eval),
                "--output",
                str(paths.features_eval),
                *layer_args,
            ],
            paths.features_eval,
        ),
        (
            "train",
            [
                *prefix,
                "-m",
                "Finetuning.run_train",
                "--config",
                str(paths.run_config),
                "--device",
                "cuda",
                "--adaptive-batch-size",
            ],
            paths.checkpoints,
        ),
    ]


def _valid_jsonl(path: Path) -> None:
    if not path.is_file():
        raise LauncherError(f"teacher artifact is missing: {path}")
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LauncherError(f"invalid teacher JSONL {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise LauncherError(f"teacher row is not an object: {path}:{line_number}")
            count += 1
    if count == 0:
        raise LauncherError(f"teacher artifact is empty: {path}")


def _valid_features(path: Path) -> None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise LauncherError(f"feature manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LauncherError(f"invalid feature manifest: {manifest_path}") from exc
    generation = manifest.get("generation_dir")
    if not isinstance(generation, str) or not generation:
        raise LauncherError(f"feature manifest lacks generation_dir: {manifest_path}")
    generation_path = path / generation
    if not generation_path.is_dir():
        raise LauncherError(f"feature generation directory is missing: {generation_path}")
    if not any(generation_path.glob("feature_*.pt")):
        raise LauncherError(f"feature generation contains no feature records: {generation_path}")


def _valid_checkpoint(path: Path, run_id: str) -> None:
    if not path.is_dir():
        raise LauncherError(f"checkpoint root is missing: {path}")
    complete = [
        candidate
        for candidate in path.glob(f"{run_id}-step*")
        if candidate.is_dir() and (candidate / "COMPLETE").is_file()
    ]
    if not complete:
        raise LauncherError(f"no complete checkpoint found under {path}")


def _write_marker(path: Path, *, stage: str, command: Sequence[str], artifact: Path, status: str) -> None:
    payload = {
        "stage": stage,
        "status": status,
        "completed_at": _utc_now(),
        "command": list(command),
        "artifact": str(artifact),
    }
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _read_marker(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LauncherError(f"invalid stage marker: {path}") from exc
    if not isinstance(value, dict):
        raise LauncherError(f"stage marker must be an object: {path}")
    return value


def run_stage(
    name: str,
    command: Sequence[str],
    *,
    log_path: Path,
    marker_path: Path,
    artifact: Path,
    validator: Callable[[], None],
    environment: dict[str, str],
    dry_run: bool,
) -> None:
    marker = _read_marker(marker_path)
    if marker is not None:
        if marker.get("stage") != name:
            raise LauncherError(f"stage marker name mismatch: {marker_path}")
        validator()
        print(f"SKIP {name}: validated marker and artifact {artifact}")
        return

    try:
        validator()
    except LauncherError:
        pass
    else:
        _write_marker(
            marker_path,
            stage=name,
            command=command,
            artifact=artifact,
            status="recovered",
        )
        print(f"RECOVER {name}: artifact already complete at {artifact}")
        return

    command_text = shlex.join(list(command))
    if dry_run:
        print(f"RUN {name}: {command_text}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{_utc_now()}] START {name}\n$ {command_text}\n")
        log_handle.flush()
        global _CURRENT_PROCESS
        _CURRENT_PROCESS = subprocess.Popen(
            list(command),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert _CURRENT_PROCESS.stdout is not None
        for line in _CURRENT_PROCESS.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return_code = _CURRENT_PROCESS.wait()
        _CURRENT_PROCESS = None
        log_handle.write(f"[{_utc_now()}] END {name} returncode={return_code}\n")
    if return_code != 0:
        raise LauncherError(
            f"stage {name} failed with return code {return_code}; inspect {log_path}"
        )
    validator()
    _write_marker(
        marker_path,
        stage=name,
        command=command,
        artifact=artifact,
        status="completed",
    )
    print(f"DONE {name}: {artifact}")


def _signal_handler(signum: int, _frame: object) -> None:
    if _CURRENT_PROCESS is not None:
        try:
            os.killpg(_CURRENT_PROCESS.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


@contextmanager
def _signal_guard() -> Iterator[None]:
    previous_term = signal.signal(signal.SIGTERM, _signal_handler)
    previous_int = signal.signal(signal.SIGINT, _signal_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


@contextmanager
def _job_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LauncherError(f"another fine-tuning job holds the lock: {path}") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _write_run_manifest(
    paths: RunPaths,
    *,
    config: dict[str, object],
    args: argparse.Namespace,
    nproc_per_node: int,
) -> None:
    serialized = yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode()
    payload = {
        "created_at": _utc_now(),
        "config_sha256": hashlib.sha256(serialized).hexdigest(),
        "train_input": str(args.train_input),
        "eval_input": str(args.eval_input),
        "target_model_path": str(_mapping(config, "model")["target_model_path"]),
        "nproc_per_node": nproc_per_node,
        "target_memory_fraction": args.target_memory_fraction,
        "adaptive_min_batch_size": args.adaptive_min_batch_size,
        "adaptive_max_batch_size": args.adaptive_max_batch_size,
        "max_tokens_per_batch": args.max_tokens_per_batch,
        "bucket_window": args.bucket_window,
        "probe_batches": args.probe_batches,
        "capture_backend": args.capture_backend,
        "capture_method": args.capture_method,
        "sglang_tp_size": args.sglang_tp_size,
        "sglang_attention_backend": args.sglang_attention_backend,
        "sglang_mem_fraction_static": args.sglang_mem_fraction_static,
        "sglang_max_running_requests": args.sglang_max_running_requests,
        "sglang_max_total_tokens": args.sglang_max_total_tokens,
        "sglang_context_length": args.sglang_context_length,
        "sglang_disable_radix_cache": args.sglang_disable_radix_cache,
        "parity_samples": args.parity_samples,
        "parity_max_abs_error": args.parity_max_abs_error,
        "parity_mean_abs_error": args.parity_mean_abs_error,
        "parity_relative_l2_error": args.parity_relative_l2_error,
        "parity_min_cosine_similarity": args.parity_min_cosine_similarity,
    }
    if paths.run_manifest.exists():
        try:
            previous = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LauncherError(f"invalid run manifest: {paths.run_manifest}") from exc
        comparable = {key: previous.get(key) for key in payload if key != "created_at"}
        if comparable != {key: payload[key] for key in comparable}:
            raise LauncherError(
                "run arguments/config differ from existing run_manifest.json; "
                "use a new --output-root"
            )
        return
    _atomic_write_text(paths.run_manifest, json.dumps(payload, indent=2) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable DFlash regeneration, caching, and training on B200"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--eval-input", type=Path, required=True)
    parser.add_argument("--target-model-path")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--python", dest="python_bin", default=DEFAULT_PYTHON)
    parser.add_argument("--target-memory-fraction", type=float)
    parser.add_argument("--adaptive-min-batch-size", type=int)
    parser.add_argument("--adaptive-max-batch-size", type=int)
    parser.add_argument("--max-tokens-per-batch", type=int, default=0)
    parser.add_argument("--bucket-window", type=int, default=512)
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument(
        "--capture-backend",
        choices=("hf", "sglang"),
        default=os.environ.get("FINETUNE_CAPTURE_BACKEND", "hf"),
    )
    parser.add_argument(
        "--capture-method",
        choices=("eagle3", "dflash", "dspark"),
        default=os.environ.get("FINETUNE_CAPTURE_METHOD", "dflash"),
    )
    parser.add_argument(
        "--sglang-tp-size",
        type=int,
        default=int(os.environ.get("FINETUNE_CAPTURE_SGLANG_TP_SIZE", "1")),
    )
    parser.add_argument(
        "--sglang-attention-backend",
        default=os.environ.get("FINETUNE_CAPTURE_SGLANG_ATTENTION_BACKEND", "flashinfer"),
    )
    parser.add_argument(
        "--sglang-mem-fraction-static",
        type=float,
        default=float(os.environ.get("FINETUNE_CAPTURE_SGLANG_MEM_FRACTION_STATIC", "0.40")),
    )
    parser.add_argument(
        "--sglang-max-running-requests",
        type=int,
        default=int(os.environ.get("FINETUNE_CAPTURE_SGLANG_MAX_RUNNING_REQUESTS", "8")),
    )
    parser.add_argument(
        "--sglang-max-total-tokens",
        type=int,
        default=int(os.environ.get("FINETUNE_CAPTURE_SGLANG_MAX_TOTAL_TOKENS", "0")),
    )
    parser.add_argument(
        "--sglang-context-length",
        type=int,
        default=(
            int(os.environ["FINETUNE_CAPTURE_SGLANG_CONTEXT_LENGTH"])
            if os.environ.get("FINETUNE_CAPTURE_SGLANG_CONTEXT_LENGTH")
            else None
        ),
    )
    parser.add_argument(
        "--sglang-disable-radix-cache",
        action="store_true",
        default=os.environ.get("FINETUNE_CAPTURE_SGLANG_DISABLE_RADIX_CACHE", "0") == "1",
    )
    parser.add_argument(
        "--parity-samples",
        type=int,
        default=int(os.environ.get("FINETUNE_CAPTURE_PARITY_SAMPLES", "2")),
    )
    parser.add_argument(
        "--parity-max-abs-error",
        type=float,
        default=float(os.environ.get("FINETUNE_CAPTURE_PARITY_MAX_ABS_ERROR", "0.05")),
    )
    parser.add_argument(
        "--parity-mean-abs-error",
        type=float,
        default=float(os.environ.get("FINETUNE_CAPTURE_PARITY_MEAN_ABS_ERROR", "0.01")),
    )
    parser.add_argument(
        "--parity-relative-l2-error",
        type=float,
        default=float(os.environ.get("FINETUNE_CAPTURE_PARITY_RELATIVE_L2_ERROR", "0.05")),
    )
    parser.add_argument(
        "--parity-min-cosine-similarity",
        type=float,
        default=float(os.environ.get("FINETUNE_CAPTURE_PARITY_MIN_COSINE_SIMILARITY", "0.999")),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.config = str(Path(args.config).expanduser().resolve())
    args.train_input = Path(args.train_input).expanduser().resolve()
    args.eval_input = Path(args.eval_input).expanduser().resolve()
    if args.target_model_path:
        args.target_model_path = str(Path(args.target_model_path).expanduser().resolve())
    if args.nproc_per_node is None:
        args.nproc_per_node = _gpu_count_from_nvidia_smi()
    if args.target_memory_fraction is None:
        args.target_memory_fraction = 0.90
    if args.adaptive_min_batch_size is None:
        args.adaptive_min_batch_size = 1
    if args.adaptive_max_batch_size is None:
        args.adaptive_max_batch_size = 256
    if not 0.0 < args.target_memory_fraction < 1.0:
        raise LauncherError("--target-memory-fraction must be between 0 and 1")
    if args.adaptive_min_batch_size <= 0:
        raise LauncherError("--adaptive-min-batch-size must be positive")
    if args.adaptive_max_batch_size < args.adaptive_min_batch_size:
        raise LauncherError(
            "--adaptive-max-batch-size must be >= --adaptive-min-batch-size"
        )
    if args.max_tokens_per_batch < 0:
        raise LauncherError("--max-tokens-per-batch must be non-negative")
    if args.bucket_window < args.adaptive_min_batch_size:
        raise LauncherError("--bucket-window must be >= --adaptive-min-batch-size")
    if args.probe_batches <= 0:
        raise LauncherError("--probe-batches must be positive")
    if args.sglang_tp_size != 1:
        raise LauncherError(
            "--sglang-tp-size must be 1 for the current data-parallel cache launcher"
        )
    if args.sglang_mem_fraction_static <= 0 or args.sglang_mem_fraction_static >= 1:
        raise LauncherError("--sglang-mem-fraction-static must be in (0, 1)")
    if args.sglang_max_running_requests <= 0 or args.sglang_max_total_tokens < 0:
        raise LauncherError("SGLang request/token limits are invalid")
    if args.parity_samples < 0:
        raise LauncherError("--parity-samples must be non-negative")

    source_config = _load_yaml(Path(args.config))
    source_run_id = _safe_run_id(args.run_id or source_config.get("run_id", "dflash"))
    paths = resolve_paths(args, source_run_id)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    with _signal_guard(), _job_lock(paths.lock_path):
        config = materialize_config(Path(args.config), paths.run_config, paths, args)
        _write_run_manifest(
            paths,
            config=config,
            args=args,
            nproc_per_node=args.nproc_per_node,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        validate_preflight(
            config,
            paths,
            python_bin=args.python_bin,
            nproc_per_node=args.nproc_per_node,
            train_input=args.train_input,
            eval_input=args.eval_input,
            dry_run=args.dry_run,
        )
        run_id = _safe_run_id(config.get("run_id", source_run_id))
        commands = build_commands(
            config,
            paths,
            args,
            python_bin=args.python_bin,
            nproc_per_node=args.nproc_per_node,
        )
        validators: dict[str, Callable[[], None]] = {
            "generate_train": lambda: _valid_jsonl(paths.teacher_train),
            "generate_eval": lambda: _valid_jsonl(paths.teacher_eval),
            "cache_train": lambda: _valid_features(paths.features_train),
            "cache_eval": lambda: _valid_features(paths.features_eval),
            "train": lambda: _valid_checkpoint(paths.checkpoints, run_id),
        }
        for name, command, artifact in commands:
            run_stage(
                name,
                command,
                log_path=paths.log_dir / f"{name}.log",
                marker_path=paths.state_dir / f"{name}.json",
                artifact=artifact,
                validator=validators[name],
                environment=env,
                dry_run=args.dry_run,
            )
        print(f"ALL DONE: {paths.output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LauncherError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
