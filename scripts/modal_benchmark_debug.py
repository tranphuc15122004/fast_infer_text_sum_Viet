#!/usr/bin/env python3
"""GPU Modal debug runner for the Vietnamese LongBench smoke path.

The local B200-simulation venv is too large to upload as a raw directory.  The
Modal Volume therefore carries the equivalent Python 3.12 environment and
HF/model cache; this runner executes the repository code with that interpreter
and records the source venv path in the returned provenance.

Example::

    FAST_INFER_VENV=/home/tuantb/fast_infer_text_sum/.venv \
    MODAL_GPU=B200 \
    modal run scripts/modal_benchmark_debug.py --baseline eagle3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/repo")
VOLUME_ROOT = Path("/mnt/fast-infer")
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "fast-infer-text-sum-cache")
GPU = os.environ.get("MODAL_GPU", "B200")
LOCAL_VENV = Path(
    os.environ.get("FAST_INFER_VENV", "/home/tuantb/fast_infer_text_sum/.venv")
)
REMOTE_VENV = VOLUME_ROOT / "venv"
REMOTE_PYTHON = REMOTE_VENV / "bin" / "python"
REMOTE_DATA = REMOTE_ROOT / "datasets" / "eval_100"
REMOTE_OUTPUT = VOLUME_ROOT / "outputs" / "longbench_viet_modal_debug"

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

app = modal.App("fast-infer-text-sum-viet-debug")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
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

if os.environ.get("MODAL_INSTALL_FA4", "0").strip().lower() in {"1", "true", "yes"}:
    image = image.pip_install(
        "apache-tvm-ffi==0.1.9",
        "nvidia-cutlass-dsl==4.5.2",
        "nvidia-cutlass-dsl-libs-base==4.5.2",
        "nvidia-cutlass-dsl-libs-cu13==4.5.2",
        "quack-kernels==0.5.0",
        "torch_c_dlpack_ext==0.1.5",
        "typing_extensions==4.15.0",
        "flash-attn-4==4.0.0b15",
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
    baseline: str,
    model: str,
    eagle_model: str,
    dflash_model: str,
    domino_model: str,
    dspark_model: str,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(REMOTE_ROOT / "src"),
                    str(REMOTE_ROOT / "externals" / "EAGLE"),
                    str(REMOTE_ROOT / "externals" / "dflash"),
                    str(REMOTE_ROOT / "externals" / "Domino"),
                    str(REMOTE_ROOT / "externals" / "SpecForge"),
                )
            ),
            "FAST_INFER_PYTHON": str(REMOTE_PYTHON),
            "HF_HOME": str(VOLUME_ROOT / "hf"),
            "HF_HUB_CACHE": str(VOLUME_ROOT / "hf" / "hub"),
            "TRANSFORMERS_CACHE": str(VOLUME_ROOT / "hf" / "hub"),
            "TRITON_CACHE_DIR": str(VOLUME_ROOT / "triton"),
            "TORCH_EXTENSIONS_DIR": str(VOLUME_ROOT / "torch_extensions"),
            "FLASHINFER_WORKSPACE_BASE": str(VOLUME_ROOT / "flashinfer"),
            # Qwen3-4B is dense; importing SGLang's optional JIT DeepGEMM
            # path would require a system CUDA toolkit that the Modal slim
            # image does not expose, even though CUDA runtime is available.
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
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "LONG_BENCH_MODEL": model,
            "MODEL_TARGET": model,
            "LONG_BENCH_EAGLE_MODEL": eagle_model,
            "MODEL_EAGLE_DRAFT": eagle_model,
            "LONG_BENCH_DFLASH_MODEL": dflash_model,
            "MODEL_DFLASH_DRAFT": dflash_model,
            "LONG_BENCH_DOMINO_MODEL": domino_model,
            "MODEL_DOMINO_DRAFT": domino_model,
            "LONG_BENCH_DSPARK_MODEL": dspark_model,
            "MODEL_DSPARK_DRAFT": dspark_model,
            "LONG_BENCH_DATA_DIR": str(REMOTE_DATA),
            "LONG_BENCH_OUTPUT_DIR": str(REMOTE_OUTPUT),
            "LONG_BENCH_DEVICE": "cuda",
            "LONG_BENCH_GPU_IDS": "0",
            "LONG_BENCH_LOCAL_FILES_ONLY": "1",
            "LONG_BENCH_MODE": "smoke",
            "LONG_BENCH_BASELINES": baseline,
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
            # FlashInfer 0.6.18 falls back to nvcc JIT for SM100 when its
            # matching cubin is absent.  The slim Modal image has CUDA runtime
            # but no /usr/local/cuda/bin/nvcc, so use SGLang Triton attention
            # for a compiler-independent correctness smoke.
            "LONG_BENCH_SGLANG_ATTENTION_BACKEND": os.environ.get(
                "MODAL_SGLANG_ATTENTION_BACKEND", "triton"
            ),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _run(command: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(REMOTE_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output_tail": completed.stdout[-20000:],
    }


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=7200,
)
def setup_qwen3(
    *,
    model: str = MODEL,
    eagle_model: str = EAGLE_MODEL,
    dflash_model: str = DFLASH_MODEL,
    domino_model: str = DOMINO_MODEL,
    dspark_model: str = DSPARK_MODEL,
) -> dict[str, Any]:
    """Download the complete Qwen3 smoke matrix and verify the runtime.

    Model files are kept in the Modal Volume's Hugging Face cache.  The
    benchmark functions later switch HF into offline mode, so a successful
    setup is a reproducible prerequisite rather than an implicit network
    download during an inference run.
    """

    hf_home = VOLUME_ROOT / "hf"
    hf_cache = hf_home / "hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_cache),
            "TRANSFORMERS_CACHE": str(hf_cache),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_DATASETS_OFFLINE": "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )

    from huggingface_hub import snapshot_download

    models = {
        "target": model,
        "eagle3": eagle_model,
        "dflash": dflash_model,
        "domino": domino_model,
        "dspark": dspark_model,
    }
    downloaded: dict[str, Any] = {}
    for role, repo_id in models.items():
        if not repo_id:
            downloaded[role] = {"repo_id": repo_id, "status": "not_configured"}
            continue
        print(f"[setup] downloading {role}: {repo_id}", flush=True)
        snapshot = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(hf_cache),
            token=os.environ.get("HF_TOKEN") or None,
            local_files_only=False,
        )
        snapshot_path = Path(snapshot)
        config_ok = (snapshot_path / "config.json").is_file()
        weight_count = sum(
            1
            for pattern in ("*.safetensors", "*.bin", "*.pt")
            for _ in snapshot_path.glob(pattern)
        )
        if not config_ok or weight_count == 0:
            raise RuntimeError(
                f"Downloaded {repo_id} but snapshot is incomplete: "
                f"path={snapshot_path}, config={config_ok}, weights={weight_count}"
            )
        downloaded[role] = {
            "repo_id": repo_id,
            "snapshot": str(snapshot_path),
            "config": config_ok,
            "weight_files": weight_count,
        }
        print(f"[setup] ready {role}: {snapshot_path}", flush=True)

    runtime: dict[str, Any] = {}
    try:
        import torch
        import transformers
        import sglang
        import flashinfer

        runtime.update(
            {
                "python": sys.executable,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "sglang": getattr(sglang, "__version__", "unknown"),
                "flashinfer": getattr(flashinfer, "__version__", "unknown"),
                "cuda": bool(torch.cuda.is_available()),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        runtime["algorithms"] = {}
        for algorithm in ("DFLASH", "DSPARK"):
            try:
                runtime["algorithms"][algorithm] = str(
                    SpeculativeAlgorithm.from_string(algorithm)
                )
            except Exception as exc:  # keep setup report actionable
                runtime["algorithms"][algorithm] = f"ERROR: {exc}"
    except Exception as exc:
        runtime["error"] = f"{type(exc).__name__}: {exc}"

    try:
        volume.commit()
    except Exception as exc:
        runtime["volume_commit_warning"] = str(exc)
    return {"models": downloaded, "runtime": runtime, "volume": VOLUME_NAME}


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(VOLUME_ROOT): volume},
    timeout=3600,
)
def debug(
    *,
    baseline: str,
    model: str = MODEL,
    eagle_model: str = EAGLE_MODEL,
    dflash_model: str = DFLASH_MODEL,
    domino_model: str = DOMINO_MODEL,
    dspark_model: str = DSPARK_MODEL,
) -> dict[str, Any]:
    # The image is the reproducible Modal equivalent of the developer venv.
    # Never select the old optional Volume venv: it may belong to a previous
    # Llama run and does not contain the Qwen3 SGLang/FlashInfer layer.
    runtime_python = sys.executable

    env = _child_env(
        baseline=baseline,
        model=model,
        eagle_model=eagle_model,
        dflash_model=dflash_model,
        domino_model=domino_model,
        dspark_model=dspark_model,
    )
    env["FAST_INFER_PYTHON"] = runtime_python
    probe = _run(
        [
            runtime_python,
            "-c",
            (
                "import json,sys,torch,transformers; "
                "print(json.dumps({'python':sys.executable,'torch':torch.__version__,"
                "'torch_cuda':torch.version.cuda,'transformers':transformers.__version__,"
                "'cuda':bool(torch.cuda.is_available()),'gpu':"
                "(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}))"
            ),
        ],
        env=env,
    )
    if probe["returncode"] != 0:
        raise RuntimeError(json.dumps(probe, ensure_ascii=False))
    result = _run(
        [
            runtime_python,
            str(REMOTE_ROOT / "src" / "Benchmark" / "run_longbench_200.py"),
            "--mode",
            "smoke",
            "--baselines",
            baseline,
            "--datasets",
            "vietnews",
            "--data-dir",
            str(REMOTE_DATA),
            "--output-dir",
            str(REMOTE_OUTPUT),
            "--max-samples",
            "1",
            "--max-new-tokens",
            "8",
            "--max-input-tokens",
            "4096",
            "--warmup-runs",
            "1",
            "--sample-retries",
            "0",
            "--no-retry-failed-samples",
            "--continue-on-error",
            "--strict",
            "--collect",
            "--run-id",
            f"modal-debug-{baseline}-{time.time_ns()}",
        ],
        env=env,
    )
    result["baseline"] = baseline
    result["gpu_contract"] = json.loads(probe["output_tail"].splitlines()[-1])
    result["source_venv"] = str(LOCAL_VENV)
    result["remote_python"] = runtime_python
    result["volume"] = VOLUME_NAME
    try:
        volume.commit()
    except Exception as exc:
        result["volume_commit_warning"] = str(exc)
    return result


@app.local_entrypoint()
def main(
    action: str = "debug",
    baseline: str = "eagle3",
    model: str = MODEL,
    eagle_model: str = EAGLE_MODEL,
    dflash_model: str = DFLASH_MODEL,
    domino_model: str = DOMINO_MODEL,
    dspark_model: str = DSPARK_MODEL,
) -> None:
    if not LOCAL_VENV.is_dir():
        raise SystemExit(f"FAST_INFER_VENV does not exist: {LOCAL_VENV}")
    if action not in {"setup", "debug"}:
        raise SystemExit(f"unsupported action: {action}")
    if action == "setup":
        result = setup_qwen3.remote(
            model=model,
            eagle_model=eagle_model,
            dflash_model=dflash_model,
            domino_model=domino_model,
            dspark_model=dspark_model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if baseline not in {"vanilla_hf", "vanilla_fa", "eagle3", "dflash", "domino", "dspark"}:
        raise SystemExit(f"unsupported baseline: {baseline}")
    result = debug.remote(
        baseline=baseline,
        model=model,
        eagle_model=eagle_model,
        dflash_model=dflash_model,
        domino_model=domino_model,
        dspark_model=dspark_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("returncode") != 0:
        raise SystemExit(int(result["returncode"]))
