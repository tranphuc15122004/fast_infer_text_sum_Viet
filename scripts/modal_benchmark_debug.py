#!/usr/bin/env python3
"""GPU Modal debug runner for the Vietnamese LongBench smoke path.

The local B200-simulation venv is fingerprinted but not uploaded.  Modal builds
an equivalent Python 3.12 image and downloads public Qwen3 checkpoints into the
temporary job filesystem, so this runner creates no persistent Volume.

Example::

    FAST_INFER_VENV=/home/tuantb/fast_infer_text_sum/.venv \
    MODAL_GPU='H100!' \
    modal run scripts/modal_benchmark_debug.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/repo")
REMOTE_RUN_ROOT = Path("/tmp/fast-infer-modal-debug")
GPU = os.environ.get("MODAL_GPU", "H100!")
LOCAL_VENV = Path(
    os.environ.get("FAST_INFER_VENV", "/home/tuantb/fast_infer_text_sum/.venv")
)
REMOTE_DATA = REMOTE_ROOT / "datasets" / "eval_100"

MODEL = os.environ.get("MODAL_QWEN3_MODEL", "Qwen/Qwen3-4B")
EAGLE_MODEL = os.environ.get(
    "MODAL_QWEN3_EAGLE_MODEL", "AngelSlim/Qwen3-4B_eagle3"
)
DFLASH_MODEL = os.environ.get(
    "MODAL_QWEN3_DFLASH_MODEL", "z-lab/Qwen3-4B-DFlash-b16"
)
DOMINO_MODEL = os.environ.get(
    "MODAL_QWEN3_DOMINO_MODEL", "Huang2020/Qwen3-4B-Domino-b16"
)
DSPARK_MODEL = os.environ.get(
    "MODAL_QWEN3_DSPARK_MODEL", "deepseek-ai/dspark_qwen3_4b_block7"
)

BASELINES = (
    "vanilla_hf",
    "vanilla_fa",
    "eagle3",
    "dflash",
    "domino",
    "dspark",
)
PACKAGE_NAMES = (
    "torch",
    "transformers",
    "tokenizers",
    "sglang",
    "sglang-kernel",
    "flashinfer-python",
    "flashinfer-cubin",
    "triton",
    "flash-attn",
    "flash-attn-4",
    "apache-tvm-ffi",
    "quack-kernels",
    "torch_c_dlpack_ext",
    "nvidia-cutlass-dsl",
    "modal",
)
MODAL_REQUIRED_VERSIONS = {
    # SGLang 0.5.19 pins Torch 2.13.0; the local venv's 2.11.0 cannot be retained.
    "torch": "2.13.0",
    "transformers": "5.12.1",
    "tokenizers": "0.22.2",
    "sglang": "0.5.19",
    "sglang-kernel": "0.4.6.post1",
    "flashinfer-python": "0.6.18",
    "flash-attn-4": "4.0.0b19",
}


def parse_baselines(value: str, allowed: tuple[str, ...] = BASELINES) -> list[str]:
    """Parse a comma/space-separated baseline list without duplicate runs."""

    selected = list(dict.fromkeys(value.replace(",", " ").split()))
    if not selected:
        raise ValueError("at least one baseline must be selected")
    unknown = [name for name in selected if name not in allowed]
    if unknown:
        raise ValueError(f"unsupported baseline(s): {', '.join(unknown)}")
    return selected


def validate_smoke_result(
    returncode: int | None,
    manifest: dict[str, Any] | None,
    expected_baselines: list[str],
) -> dict[str, Any]:
    """Require one successful, complete, audited VietNews record per baseline."""

    overall_issues: list[str] = []
    per_baseline: dict[str, dict[str, Any]] = {}
    if returncode != 0:
        overall_issues.append(f"benchmark process exit code: {returncode}")
    if manifest is None:
        overall_issues.append("run manifest is missing or unreadable")
        manifest = {}
    if manifest.get("failure_count", 0) != 0:
        overall_issues.append(f"manifest failure_count={manifest.get('failure_count')}")

    cells_by_baseline: dict[str, list[dict[str, Any]]] = {}
    for cell in manifest.get("cells", []):
        if isinstance(cell, dict):
            cells_by_baseline.setdefault(str(cell.get("baseline", "")), []).append(cell)

    for baseline in expected_baselines:
        issues: list[str] = []
        cells = cells_by_baseline.get(baseline, [])
        if len(cells) != 1:
            issues.append(f"expected one VietNews cell, found {len(cells)}")
        for cell in cells:
            if cell.get("dataset") != "vietnews":
                issues.append(f"unexpected dataset: {cell.get('dataset')}")
            if cell.get("status") != "success":
                issues.append(f"cell status={cell.get('status')}")
            if int(cell.get("sample_count", 0) or 0) != 1:
                issues.append(f"sample coverage={cell.get('sample_count')}, expected 1")
            preflight = cell.get("preflight") or {}
            if preflight.get("status") != "ready":
                issues.append(f"preflight status={preflight.get('status')}")
            contract = cell.get("metric_contract") or {}
            if contract.get("status") != "complete":
                issues.append(f"metric contract status={contract.get('status')}")
            if (
                int(contract.get("observed_samples", 0) or 0) != 1
                or int(contract.get("expected_samples", 0) or 0) != 1
            ):
                issues.append(
                    "metric contract coverage mismatch: "
                    f"observed={contract.get('observed_samples')}, "
                    f"expected={contract.get('expected_samples')}"
                )
            audit = cell.get("metric_audit_summary") or {}
            if int(audit.get("num_records", 0) or 0) != 1:
                issues.append(f"metric audit records={audit.get('num_records')}, expected 1")
            if audit.get("status_counts") != {"success": 1}:
                issues.append(f"metric audit statuses={audit.get('status_counts')}")
        per_baseline[baseline] = {
            "status": "passed" if not issues else "failed",
            "issues": issues,
        }
        overall_issues.extend(f"{baseline}: {issue}" for issue in issues)

    unexpected = sorted(set(cells_by_baseline) - set(expected_baselines))
    if unexpected:
        overall_issues.append(f"unexpected baseline cells: {', '.join(unexpected)}")
    return {
        "status": "passed" if not overall_issues else "failed",
        "baselines": per_baseline,
        "issues": overall_issues,
    }


def capture_environment_fingerprint(venv_path: Path) -> dict[str, Any]:
    """Capture relevant pins and Torch/CUDA facts from the source venv."""

    python = Path(venv_path) / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"venv interpreter does not exist: {python}")
    code = """
import importlib.metadata as metadata
import json, platform, sys, torch
names = __PACKAGE_NAMES__
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
try:
    cuda_available = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda_available else None
except Exception as exc:
    cuda_available, gpu = False, None
    torch_error = f"{type(exc).__name__}: {exc}"
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": packages,
    "torch_cuda": torch.version.cuda,
    "cuda_available": cuda_available,
    "gpu": gpu,
    "torch_error": locals().get("torch_error"),
}, ensure_ascii=False))
""".replace("__PACKAGE_NAMES__", repr(list(PACKAGE_NAMES)))
    details = subprocess.run(
        [str(python), "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if details.returncode != 0:
        raise RuntimeError(f"could not inspect source venv: {details.stdout}")
    try:
        fingerprint = json.loads(details.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid source venv fingerprint: {details.stdout}") from exc
    frozen = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--disable-pip-version-check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    fingerprint["pip_freeze"] = frozen.stdout if frozen.returncode == 0 else None
    fingerprint["pip_freeze_returncode"] = frozen.returncode
    fingerprint["venv_path"] = str(venv_path)
    return fingerprint


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _persist_local_outputs(result: dict[str, Any]) -> Path:
    """Write returned Modal logs, benchmark artifacts and the report locally."""

    run_id = str(result["run_id"])
    output_dir = PROJECT_ROOT / "outputs" / "modal_debug" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, content in (result.get("logs") or {}).items():
        path = output_dir / "logs" / f"{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
    for relative_name, content in (result.get("artifacts") or {}).items():
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe artifact path from Modal: {relative_name}")
        path = output_dir / "benchmark" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")

    environment = {
        "source_venv": result.get("source_environment"),
        "modal": result.get("modal_environment"),
        "package_differences": result.get("package_differences"),
        "checkpoints": result.get("checkpoints"),
    }
    _write_json(output_dir / "environment.json", environment)

    summary = {
        key: value
        for key, value in result.items()
        if key not in {"logs", "artifacts", "source_environment"}
    }
    _write_json(output_dir / "run_report.json", summary)

    runtime = result.get("modal_environment") or {}
    packages = runtime.get("packages") or {}
    lines = [
        f"# Báo cáo debug Modal — {run_id}",
        "",
        f"- Kết quả: **{result.get('status', 'unknown')}**",
        f"- GPU yêu cầu/thực tế: {GPU} / {runtime.get('gpu')}",
        f"- Compute capability: {runtime.get('compute_capability')}",
        f"- CUDA Torch / driver API: {runtime.get('torch_cuda')} / "
        f"{runtime.get('cuda_driver_api')}",
        f"- Venv nguồn: {(result.get('source_environment') or {}).get('venv_path')}",
        "",
        "## Phiên bản Modal đã chạy",
        "",
    ]
    lines.extend(f"- {name}=={version}" for name, version in packages.items())
    lines.extend(["", "## Checkpoint", ""])
    for role, item in (result.get("checkpoints") or {}).items():
        lines.append(
            f"- {role}: {item.get('repo_id')} @ {item.get('checkpoint_id')} "
            f"({len(item.get('weight_files') or [])} file trọng số)"
        )
    lines.extend(["", "## Các lượt smoke", ""])
    for stage, item in (result.get("stages") or {}).items():
        lines.append(f"- {stage}: **{item.get('status')}**")
        for issue in item.get("issues", []):
            lines.append(f"  - {issue}")
        if item.get("command"):
            lines.append(f"  - Lệnh: {' '.join(item['command'])}")
    if result.get("errors"):
        lines.extend(["", "## Lỗi", ""])
        lines.extend(f"- {error}" for error in result["errors"])
    lines.extend(
        [
            "",
            "## Giới hạn diễn giải",
            "",
            "Đây là smoke 1 mẫu, đầu ra tối đa 8 token: chỉ xác nhận đường chạy, "
            "coverage và metric audit; không dùng ROUGE hoặc timing để kết luận "
            "chất lượng/tốc độ.",
            "Modal H100/H200 dùng Hopper, không xác nhận nhánh kernel SM100/B200. "
            "Cần chạy smoke cuối trên B200.",
            "Không tạo Modal Volume và không thay đổi môi trường server.",
            "",
            "Full log ở logs/; JSONL, manifest và audit ở benchmark/.",
        ]
    )
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir

app = modal.App("fast-infer-text-sum-viet-debug")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.12.1",
        "tokenizers==0.22.2",
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "einops==0.8.2",
        "huggingface_hub==1.21.0",
        "numpy==2.2.6",
        "protobuf==6.33.6",
        "psutil==7.2.2",
        "rouge_score==0.1.2",
        "safetensors==0.8.0",
        "sentencepiece==0.2.1",
        "tqdm==4.68.3",
        "regex==2026.6.28",
        "packaging==26.2",
        "Jinja2==3.1.6",
        "filelock==3.29.4",
        "sympy==1.14.0",
        "networkx==3.6.1",
        "PyYAML==6.0.3",
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(
        str(PROJECT_ROOT / "datasets" / "eval_100"),
        remote_path=str(REMOTE_DATA),
        copy=True,
    )
)

# Keep the large Torch/Transformers layer cacheable independently from the
# SGLang resolver.  uv is substantially faster here and gives Modal a clear
# package-install layer instead of making one pip resolver transaction with
# the whole ML stack.
image = image.uv_pip_install(
    # SGLang 0.5.19's cu13 extra pins FlashInfer 0.6.18.  The configured
    # mirror does not publish a separate 0.6.18 cubin package, so let the
    # extra select the compatible CUDA artifacts.
    "flashinfer-python[cu13]==0.6.18",
    "sglang==0.5.19",
    # sglang 0.5.19 declares this exact kernel build; 0.4.7 is used by a
    # newer server manifest and is incompatible with this SGLang release.
    "sglang-kernel==0.4.6.post1",
)
# NVIDIA's pip CUDA 13 wheel keeps libcudart under ``lib`` while nvcc/JIT
# build scripts conventionally link ``$CUDA_HOME/lib64``. Normalize that
# layout once in the image so SGLang/FlashInfer extensions can link.
image = image.run_commands(
    "ln -s /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib "
    "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib64 && "
    "ln -s /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13 "
    "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so"
)

if os.environ.get("MODAL_INSTALL_FA4", "1").strip().lower() in {"1", "true", "yes"}:
    image = image.pip_install(
        "apache-tvm-ffi==0.1.11",
        "nvidia-cutlass-dsl==4.6.2",
        "nvidia-cutlass-dsl-libs-base==4.6.2",
        "nvidia-cutlass-dsl-libs-cu13==4.6.2",
        "quack-kernels==0.6.4",
        "torch_c_dlpack_ext==0.1.5",
        "typing_extensions==4.16.0",
        "flash-attn-4==4.0.0b19",
        extra_options="--no-deps",
    )

for _name in ("EAGLE", "dflash", "Domino", "SpecForge"):
    image = image.add_local_dir(
        str(PROJECT_ROOT / "externals" / _name),
        remote_path=str(REMOTE_ROOT / "externals" / _name),
        copy=True,
    )


def _child_env(
    *,
    baselines: str,
    model_paths: dict[str, str],
    hf_home: Path,
    run_root: Path,
    output_dir: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = [
        str(REMOTE_ROOT / "src"),
        str(REMOTE_ROOT / "externals" / "EAGLE"),
        str(REMOTE_ROOT / "externals" / "dflash"),
        str(REMOTE_ROOT / "externals" / "Domino"),
        str(REMOTE_ROOT / "externals" / "SpecForge"),
    ]
    hf_hub_cache = hf_home / "hub"
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "FAST_INFER_PYTHON": sys.executable,
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_hub_cache),
            "TRANSFORMERS_CACHE": str(hf_hub_cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRITON_CACHE_DIR": str(run_root / "triton"),
            "TORCH_EXTENSIONS_DIR": str(run_root / "torch_extensions"),
            "FLASHINFER_WORKSPACE_BASE": str(run_root / "flashinfer"),
            "SGLANG_ENABLE_JIT_DEEPGEMM": "false",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "CUDA_HOME": os.environ.get(
                "CUDA_HOME", "/usr/local/lib/python3.12/site-packages/nvidia/cu13"
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(
                (
                    os.environ.get("LD_LIBRARY_PATH", ""),
                    "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib",
                )
            ).strip(os.pathsep),
            "MODEL_TARGET": model_paths["target"],
            "LONG_BENCH_MODEL": model_paths["target"],
            "MODEL_EAGLE_DRAFT": model_paths.get("eagle3", EAGLE_MODEL),
            "LONG_BENCH_EAGLE_MODEL": model_paths.get("eagle3", EAGLE_MODEL),
            "MODEL_DFLASH_DRAFT": model_paths.get("dflash", DFLASH_MODEL),
            "LONG_BENCH_DFLASH_MODEL": model_paths.get("dflash", DFLASH_MODEL),
            "MODEL_DOMINO_DRAFT": model_paths.get("domino", DOMINO_MODEL),
            "LONG_BENCH_DOMINO_MODEL": model_paths.get("domino", DOMINO_MODEL),
            "MODEL_DSPARK_DRAFT": model_paths.get("dspark", DSPARK_MODEL),
            "LONG_BENCH_DSPARK_MODEL": model_paths.get("dspark", DSPARK_MODEL),
            "LONG_BENCH_DATA_DIR": str(REMOTE_DATA),
            "LONG_BENCH_OUTPUT_DIR": str(output_dir),
            "LONG_BENCH_DEVICE": "cuda",
            "LONG_BENCH_GPU_IDS": "0",
            "LONG_BENCH_LOCAL_FILES_ONLY": "1",
            "LONG_BENCH_MODE": "smoke",
            "LONG_BENCH_BASELINES": baselines,
            "LONG_BENCH_DATASETS": "vietnews",
            "LONG_BENCH_MAX_NEW_TOKENS": "8",
            "LONG_BENCH_SMOKE_MAX_NEW_TOKENS": "8",
            "LONG_BENCH_MAX_INPUT_TOKENS": "4096",
            "LONG_BENCH_SMOKE_MAX_INPUT_TOKENS": "4096",
            "LONG_BENCH_WARMUP_RUNS": "1",
            "LONG_BENCH_SEED": "42",
            "LONG_BENCH_TEMPERATURE": "0",
            "LONG_BENCH_STRICT": "1",
            "LONG_BENCH_COLLECT": "1",
            "LONG_BENCH_RETRY_FAILED_SAMPLES": "0",
            "LONG_BENCH_SAMPLE_RETRIES": "0",
            "LONG_BENCH_CONTINUE_ON_ERROR": "1",
            "LONG_BENCH_TIMEOUT_SECONDS": "900",
            "LONG_BENCH_ATTENTION_BACKEND": os.environ.get(
                "MODAL_VANILLA_ATTENTION_BACKEND", "flash_attention_4"
            ),
            "LONG_BENCH_SGLANG_ATTENTION_BACKEND": os.environ.get(
                "MODAL_SGLANG_ATTENTION_BACKEND", "triton"
            ),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run one bounded command and persist its complete combined output."""

    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
        output = completed.stdout
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = _decode_output(exc.stdout)
        timed_out = True
        returncode = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "log_path": str(log_path),
        "output_tail": output[-20000:],
    }


def _download_checkpoints(
    *,
    cache_dir: Path,
    models: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Download only requested public snapshots and verify configs/weights."""

    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HOME": str(cache_dir.parent),
            "HF_HUB_CACHE": str(cache_dir),
            "TRANSFORMERS_CACHE": str(cache_dir),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_DATASETS_OFFLINE": "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    local_paths: dict[str, str] = {}
    checkpoints: dict[str, Any] = {}
    for role, repo_id in models.items():
        if not repo_id:
            raise ValueError(f"checkpoint is not configured for role {role}")
        print(f"[checkpoint] downloading/checking {role}: {repo_id}", flush=True)
        snapshot = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN") or None,
            local_files_only=False,
        )
        snapshot_path = Path(snapshot)
        config_ok = (snapshot_path / "config.json").is_file()
        weight_files = sorted(
            {
                path
                for pattern in ("*.safetensors", "*.bin", "*.pt")
                for path in snapshot_path.glob(pattern)
            }
        )
        if not config_ok or not weight_files:
            raise RuntimeError(
                f"incomplete checkpoint {repo_id}: path={snapshot_path}, "
                f"config={config_ok}, weights={len(weight_files)}"
            )
        local_paths[role] = str(snapshot_path)
        checkpoints[role] = {
            "repo_id": repo_id,
            "checkpoint_id": snapshot_path.name,
            "snapshot": str(snapshot_path),
            "config": config_ok,
            "weight_files": [path.name for path in weight_files],
            "weight_bytes": sum(path.stat().st_size for path in weight_files),
        }
        print(
            f"[checkpoint] ready {role}: {snapshot_path.name} "
            f"({len(weight_files)} weight files)",
            flush=True,
        )
    return local_paths, checkpoints


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=9000,
    max_containers=1,
)
def debug(
    *,
    action: str,
    baselines: str,
    run_id: str,
    source_environment: dict[str, Any],
    model: str = MODEL,
    eagle_model: str = EAGLE_MODEL,
    dflash_model: str = DFLASH_MODEL,
    domino_model: str = DOMINO_MODEL,
    dspark_model: str = DSPARK_MODEL,
) -> dict[str, Any]:
    """Run focused DSpark first, then the six-baseline regression on one H100."""

    if Path(run_id).name != run_id or not run_id:
        raise ValueError(f"unsafe run id: {run_id!r}")
    if action not in {"smoke", "debug"}:
        raise ValueError(f"unsupported action: {action}")
    selected_baselines = ["dspark"] if action == "smoke" else parse_baselines(baselines)
    run_root = REMOTE_RUN_ROOT / run_id
    hf_home = run_root / "hf"
    hf_cache = hf_home / "hub"
    benchmark_root = run_root / "benchmark"
    log_root = run_root / "logs"
    for directory in (hf_cache, benchmark_root, log_root):
        directory.mkdir(parents=True, exist_ok=True)

    model_ids = {
        "target": model,
        "eagle3": eagle_model,
        "dflash": dflash_model,
        "domino": domino_model,
        "dspark": dspark_model,
    }
    checkpoints: dict[str, Any] = {}
    local_paths: dict[str, str] = {}
    stages: dict[str, Any] = {}
    errors: list[str] = []
    logs: dict[str, str] = {}
    modal_environment: dict[str, Any] = {}
    package_differences: dict[str, Any] = {}
    overall_status = "failed"

    try:
        needed_roles = {"target"}
        if action == "smoke":
            needed_roles.add("dspark")
        else:
            for baseline in selected_baselines:
                role = {
                    "eagle3": "eagle3",
                    "dflash": "dflash",
                    "domino": "domino",
                    "dspark": "dspark",
                }.get(baseline)
                if role:
                    needed_roles.add(role)

        requested_models = {role: model_ids[role] for role in sorted(needed_roles)}
        local_paths, checkpoints = _download_checkpoints(
            cache_dir=hf_cache,
            models=requested_models,
        )
        print(f"[preflight] verified checkpoints: {sorted(checkpoints)}", flush=True)

        probe_code = """
import importlib.metadata as metadata
import importlib.util
import json
import platform
import subprocess
import sys
import torch

names = __PACKAGE_NAMES__
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
cuda_available = False
gpu = None
capability = None
torch_error = None
try:
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        gpu = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
except Exception as exc:
    torch_error = f"{type(exc).__name__}: {exc}"
algorithms = {}
try:
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    for name in ("DFLASH", "DSPARK"):
        try:
            algorithms[name] = str(SpeculativeAlgorithm.from_string(name))
        except Exception as exc:
            algorithms[name] = f"ERROR: {type(exc).__name__}: {exc}"
except Exception as exc:
    algorithms["import_error"] = f"{type(exc).__name__}: {exc}"
try:
    fa4_importable = importlib.util.find_spec("flash_attn.cute") is not None
except Exception:
    fa4_importable = False
smi = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, check=False)
smi_text = smi.stdout if smi.returncode == 0 else f"nvidia-smi exit {smi.returncode}: {smi.stdout}"
driver_api = None
for line in smi_text.splitlines():
    if "CUDA Version:" in line:
        driver_api = line.split("CUDA Version:", 1)[1].split("|", 1)[0].strip()
        break
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "packages": packages,
    "torch_cuda": torch.version.cuda,
    "cuda_available": cuda_available,
    "gpu": gpu,
    "compute_capability": capability,
    "torch_error": torch_error,
    "cuda_driver_api": driver_api,
    "nvidia_smi": smi_text,
    "algorithms": algorithms,
    "flash_attention_4_importable": fa4_importable,
}, ensure_ascii=False))
""".replace("__PACKAGE_NAMES__", repr(list(PACKAGE_NAMES)))
        probe_env = _child_env(
            baselines="dspark",
            model_paths=local_paths,
            hf_home=hf_home,
            run_root=run_root,
            output_dir=benchmark_root / "preflight",
        )
        probe = _run(
            [sys.executable, "-c", probe_code],
            env=probe_env,
            cwd=REMOTE_ROOT,
            log_path=log_root / "preflight.log",
            timeout_seconds=300,
        )
        logs["preflight"] = (log_root / "preflight.log").read_text(encoding="utf-8")
        if probe["returncode"] != 0:
            raise RuntimeError(f"runtime probe failed: {probe['output_tail']}")
        modal_environment = json.loads(probe["output_tail"].splitlines()[-1])
        pip_check = _run(
            [sys.executable, "-m", "pip", "check"],
            env=probe_env,
            cwd=REMOTE_ROOT,
            log_path=log_root / "pip_check.log",
            timeout_seconds=120,
        )
        pip_freeze = _run(
            [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
            env=probe_env,
            cwd=REMOTE_ROOT,
            log_path=log_root / "pip_freeze.log",
            timeout_seconds=120,
        )
        logs["pip_check"] = (log_root / "pip_check.log").read_text(encoding="utf-8")
        logs["pip_freeze"] = (log_root / "pip_freeze.log").read_text(encoding="utf-8")
        modal_environment["pip_check_returncode"] = pip_check["returncode"]
        modal_environment["pip_check"] = logs["pip_check"]
        modal_environment["pip_freeze_returncode"] = pip_freeze["returncode"]
        modal_environment["pip_freeze"] = logs["pip_freeze"]

        source_packages = source_environment.get("packages") or {}
        modal_packages = modal_environment.get("packages") or {}
        package_differences = {
            name: {"source_venv": source_packages.get(name), "modal": modal_packages.get(name)}
            for name in PACKAGE_NAMES
            if source_packages.get(name) != modal_packages.get(name)
        }
        preflight_issues: list[str] = []
        if modal_environment.get("pip_check_returncode") != 0:
            preflight_issues.append(
                f"pip check failed: {modal_environment.get('pip_check')}"
            )
        if modal_environment.get("pip_freeze_returncode") != 0:
            preflight_issues.append("could not capture the Modal package fingerprint")
        if not modal_environment.get("cuda_available"):
            preflight_issues.append(
                f"torch.cuda.is_available is false: {modal_environment.get('torch_error')}"
            )
        gpu_name = str(modal_environment.get("gpu") or "")
        if "H100" not in gpu_name and "H200" not in gpu_name:
            preflight_issues.append(f"expected Modal H100/H200, got {gpu_name or 'no GPU'}")
        if modal_environment.get("torch_cuda") != "13.0":
            preflight_issues.append(
                f"expected Torch CUDA 13.0, got {modal_environment.get('torch_cuda')}"
            )
        if modal_environment.get("cuda_driver_api") != "13.0":
            preflight_issues.append(
                f"expected driver API 13.0, got {modal_environment.get('cuda_driver_api')}"
            )
        if "ERROR" in str(modal_environment.get("algorithms", {}).get("DSPARK", "")):
            preflight_issues.append(
                f"SGLang DSPARK parse failed: {modal_environment.get('algorithms')}"
            )
        required_versions = MODAL_REQUIRED_VERSIONS
        for name, expected in required_versions.items():
            actual = modal_packages.get(name)
            if actual != expected:
                preflight_issues.append(f"package mismatch {name}: expected {expected}, got {actual}")
        if not modal_environment.get("flash_attention_4_importable"):
            preflight_issues.append("flash_attn.cute is not importable")
        preflight_issues.extend(
            f"checkpoint {role} missing config or weights"
            for role, item in checkpoints.items()
            if not item.get("config") or not item.get("weight_files")
        )
        stages["preflight"] = {
            "status": "passed" if not preflight_issues else "failed",
            "issues": preflight_issues,
            "checkpoint_roles": sorted(checkpoints),
            "algorithms": modal_environment.get("algorithms"),
        }
        if preflight_issues:
            errors.extend(preflight_issues)
        else:
            def run_benchmarks(
                stage_name: str,
                stage_baselines: list[str],
                timeout_seconds: int,
            ) -> dict[str, Any]:
                stage_output = benchmark_root / stage_name
                longbench_run_id = f"{stage_name}-{run_id}"
                env = _child_env(
                    baselines=",".join(stage_baselines),
                    model_paths=local_paths,
                    hf_home=hf_home,
                    run_root=run_root,
                    output_dir=stage_output,
                )
                command = [
                    sys.executable,
                    str(REMOTE_ROOT / "src" / "Benchmark" / "run_longbench_200.py"),
                    "--mode", "smoke",
                    "--baselines", ",".join(stage_baselines),
                    "--datasets", "vietnews",
                    "--data-dir", str(REMOTE_DATA),
                    "--output-dir", str(stage_output),
                    "--max-samples", "1",
                    "--max-new-tokens", "8",
                    "--max-input-tokens", "4096",
                    "--warmup-runs", "1",
                    "--sample-retries", "0",
                    "--no-retry-failed-samples",
                    "--continue-on-error",
                    "--strict",
                    "--collect",
                    "--run-id", longbench_run_id,
                ]
                log_path = log_root / f"{stage_name}.log"
                command_result = _run(
                    command,
                    env=env,
                    cwd=REMOTE_ROOT,
                    log_path=log_path,
                    timeout_seconds=timeout_seconds,
                )
                logs[stage_name] = log_path.read_text(encoding="utf-8")
                manifest_path = stage_output / longbench_run_id / "run_manifest.json"
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = None
                validation = validate_smoke_result(
                    command_result["returncode"],
                    manifest,
                    stage_baselines,
                )
                return {
                    "status": validation["status"],
                    "issues": validation["issues"],
                    "baselines": validation["baselines"],
                    "command": command,
                    "returncode": command_result["returncode"],
                    "timed_out": command_result["timed_out"],
                    "elapsed_seconds": command_result["elapsed_seconds"],
                    "output_tail": command_result["output_tail"],
                    "manifest_path": str(manifest_path),
                }

            if action == "smoke":
                focus = run_benchmarks("dspark_focus", ["dspark"], 1800)
                stages["dspark_focus"] = focus
                if focus["status"] == "passed":
                    remaining_models = {
                        role: model_ids[role]
                        for role in ("eagle3", "dflash", "domino")
                    }
                    extra_paths, extra_checkpoints = _download_checkpoints(
                        cache_dir=hf_cache,
                        models=remaining_models,
                    )
                    local_paths.update(extra_paths)
                    checkpoints.update(extra_checkpoints)
                    matrix = run_benchmarks("regression", list(BASELINES), 6600)
                    stages["regression"] = matrix
                    overall_status = matrix["status"]
                    errors.extend(matrix["issues"])
                else:
                    overall_status = "failed"
                    errors.extend(focus["issues"])
            else:
                focused = run_benchmarks("targeted", selected_baselines, 3600)
                stages["targeted"] = focused
                overall_status = focused["status"]
                errors.extend(focused["issues"])
    except Exception as exc:
        overall_status = "failed"
        errors.append(f"{type(exc).__name__}: {exc}")
        print(f"[modal-debug-error] {type(exc).__name__}: {exc}", flush=True)

    artifacts: dict[str, str] = {}
    for path in sorted(benchmark_root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md", ".log"}:
            relative_name = path.relative_to(benchmark_root).as_posix()
            try:
                artifacts[relative_name] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
    for path in log_root.glob("*.log"):
        logs.setdefault(path.stem, path.read_text(encoding="utf-8"))

    return {
        "run_id": run_id,
        "action": action,
        "status": overall_status,
        "gpu_requested": GPU,
        "modal_environment": modal_environment,
        "source_environment": source_environment,
        "package_differences": package_differences,
        "checkpoints": checkpoints,
        "stages": stages,
        "errors": errors,
        "logs": logs,
        "artifacts": artifacts,
        "persistent_volume_used": False,
        "server_environment_changed": False,
    }


@app.local_entrypoint()
def main(
    action: str = "smoke",
    baseline: str = "dspark",
    baselines: str = "",
    run_id: str = "",
    model: str = MODEL,
    eagle_model: str = EAGLE_MODEL,
    dflash_model: str = DFLASH_MODEL,
    domino_model: str = DOMINO_MODEL,
    dspark_model: str = DSPARK_MODEL,
) -> None:
    if not (LOCAL_VENV / "bin" / "python").is_file():
        raise SystemExit(f"FAST_INFER_VENV interpreter does not exist: {LOCAL_VENV}")
    if action not in {"smoke", "debug"}:
        raise SystemExit("action must be smoke or debug")
    selected = parse_baselines(baselines or baseline) if action == "debug" else ["dspark"]
    source_environment = capture_environment_fingerprint(LOCAL_VENV)
    selected_run_id = run_id or (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + uuid.uuid4().hex[:8]
    )
    try:
        result = debug.remote(
            action=action,
            baselines=",".join(selected),
            run_id=selected_run_id,
            source_environment=source_environment,
            model=model,
            eagle_model=eagle_model,
            dflash_model=dflash_model,
            domino_model=domino_model,
            dspark_model=dspark_model,
        )
    except Exception as exc:
        failure = {
            "run_id": selected_run_id,
            "action": action,
            "status": "failed",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "source_environment": source_environment,
            "persistent_volume_used": False,
            "server_environment_changed": False,
        }
        output_dir = PROJECT_ROOT / "outputs" / "modal_debug" / selected_run_id
        _write_json(output_dir / "run_report.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        print(f"Artifacts: {output_dir}", flush=True)
        raise SystemExit(1) from exc

    output_dir = _persist_local_outputs(result)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"logs", "artifacts", "source_environment"}
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {output_dir}", flush=True)
    if result.get("status") != "passed":
        raise SystemExit(1)
