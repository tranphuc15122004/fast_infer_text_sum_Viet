"""GPU Modal smoke test for the complete offline DFlash preparation/training path.

The smoke test intentionally creates a tiny local Qwen3 snapshot inside the
Modal container. It exercises the real project CLIs without downloading a
gated model or reading `datasets/`:

    teacher regeneration -> hidden-state caching -> adaptive DFlash training

Run from the repository root with::

    FAST_INFER_VENV=/home/tuantb/fast_infer_text_sum/.venv \
      modal run scripts/modal_finetuning_smoke.py

Use `MODAL_GPU=B200` when the smoke must run on B200. The default is L4,
which is sufficient to validate CUDA, adaptive batching, and the end-to-end
file/manifest/checkpoint contracts at lower cost.

Set `MODAL_CAPTURE_BACKEND=sglang` to validate the production hidden-state
cache path through SpecForge's in-process `OfflineSGLangCapture` and its HF
parity gate.  This mode installs the version-pinned SGLang dependency in the
Modal image and mounts the vendored SpecForge source.

Set `MODAL_SGLANG_ATTENTION_BACKEND=flashinfer` when the Modal image has a
CUDA-devel toolchain. The default `triton` backend keeps the smoke independent
of a system `nvcc`; production B200 should use the master-configured
`flashinfer` backend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/workspace/repo"
GPU = os.environ.get("MODAL_GPU", "L4")
CAPTURE_BACKEND = os.environ.get("MODAL_CAPTURE_BACKEND", "hf").lower()
if CAPTURE_BACKEND not in {"hf", "sglang"}:
    raise ValueError("MODAL_CAPTURE_BACKEND must be 'hf' or 'sglang'")
TORCH_VERSION = "2.13.0" if CAPTURE_BACKEND == "sglang" else "2.11.0"
SGLANG_ATTENTION_BACKEND = os.environ.get("MODAL_SGLANG_ATTENTION_BACKEND", "triton")
VENV_FINETUNING_DEPS = [
    "accelerate==1.14.0",
    "safetensors==0.8.0",
    "numpy==2.2.6",
    "PyYAML==6.0.3",
    "huggingface_hub==1.21.0",
    "fsspec==2026.4.0",
    "tqdm==4.68.3",
    "regex==2026.6.28",
    "packaging==26.2",
    "Jinja2==3.1.6",
    "filelock==3.29.4",
    "sympy==1.14.0",
    "networkx==3.6.1",
]


app = modal.App("finetuning-dflash-e2e-smoke")

# The host venv cannot be copied into another container because its Python
# executable is a host-specific symlink. Matching package versions is the
# portable Modal contract.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        "transformers==5.12.1",
        "tokenizers==0.22.2",
        *VENV_FINETUNING_DEPS,
        *(["sglang==0.5.18", "yunchang"] if CAPTURE_BACKEND == "sglang" else []),
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src"),
        remote_path=f"{REMOTE_ROOT}/src",
        copy=True,
    )
    .add_local_dir(
        str(PROJECT_ROOT / "externals" / "SpecForge"),
        remote_path=f"{REMOTE_ROOT}/externals/SpecForge",
        copy=True,
    )
)


def _run(command: list[str], *, cwd: str, env: dict[str, str]) -> dict[str, Any]:
    """Run one project CLI and retain enough output to diagnose a failure."""

    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=900,
    )
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-12000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _last_json_line(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise RuntimeError(f"command did not emit a JSON summary:\n{output[-4000:]}")


def _write_tiny_qwen_snapshot(model_dir: Path) -> None:
    """Create an offline Qwen3-compatible model and tokenizer for the smoke."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    base_tokens = [
        "[PAD]",
        "[UNK]",
        "SUMMARIZE",
        "doc",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "gold",
        "summary",
    ]
    vocab_tokens = base_tokens + [f"t{index}" for index in range(128 - len(base_tokens))]
    vocab = {token: index for index, token in enumerate(vocab_tokens)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    # Keep role wrappers out of this tiny tokenizer. Production code still
    # goes through apply_chat_template and span alignment.
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir)

    config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention", "full_attention"],
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(model_dir, safe_serialization=True)


def _write_fixture(path: Path) -> None:
    records = [
        {"id": "modal-001", "document": "doc one two", "summary": "gold summary"},
        {"id": "modal-002", "document": "doc three four", "summary": "gold summary"},
        {"id": "modal-003", "document": "doc five six", "summary": "gold summary"},
        {"id": "modal-004", "document": "doc one three", "summary": "gold summary"},
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=16384,
    timeout=1800,
)
def smoke(
    capture_backend: str = "hf",
    attention_backend: str = "triton",
) -> dict[str, Any]:
    import accelerate
    import fsspec
    import huggingface_hub
    import numpy
    import safetensors
    import torch
    import tokenizers
    import transformers
    import yaml

    if capture_backend not in {"hf", "sglang"}:
        raise ValueError("capture_backend must be 'hf' or 'sglang'")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal function did not expose CUDA")
    runtime_versions = {
        "torch": torch.__version__.split("+", 1)[0],
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "accelerate": accelerate.__version__,
        "safetensors": safetensors.__version__,
        "numpy": numpy.__version__,
        "yaml": yaml.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "fsspec": fsspec.__version__,
    }
    if capture_backend == "hf":
        expected_versions = {
            "torch": "2.11.0",
            "torch_cuda": "13.0",
            "transformers": "5.12.1",
            "tokenizers": "0.22.2",
            "accelerate": "1.14.0",
            "safetensors": "0.8.0",
            "numpy": "2.2.6",
            "yaml": "6.0.3",
            "huggingface_hub": "1.21.0",
            "fsspec": "2026.4.0",
        }
        if runtime_versions != expected_versions:
            raise RuntimeError(
                "Modal runtime differs from the requested venv package contract: "
                f"expected={expected_versions}, actual={runtime_versions}"
            )
    device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device)
    work = Path(tempfile.mkdtemp(prefix="finetuning-modal-smoke-"))
    model_dir = work / "tiny-qwen3"
    input_path = work / "input.jsonl"
    regenerated_path = work / "teacher.jsonl"
    feature_dir = work / "features"
    train_dir = work / "train"
    config_path = work / "train.yaml"
    _write_tiny_qwen_snapshot(model_dir)
    _write_fixture(input_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{REMOTE_ROOT}/src{os.pathsep}{REMOTE_ROOT}/externals/SpecForge"
    )
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["FAST_INFER_CACHE_ROOT"] = str(work / "runtime-cache")

    generation_command = [
        sys.executable,
        "-m",
        "Finetuning.generate_targets",
        "--input",
        str(input_path),
        "--output",
        str(regenerated_path),
        "--target-model-path",
        str(model_dir),
        "--max-length",
        "16",
        "--max-source-tokens",
        "8",
        "--max-summary-tokens",
        "4",
        "--prompt-template",
        "SUMMARIZE {document}",
        "--torch-dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--adaptive-max-batch-size",
        "8",
        "--probe-batches",
        "1",
        "--bucket-window",
        "4",
    ]
    generation_result = _run(generation_command, cwd=REMOTE_ROOT, env=env)
    generation_summary = _last_json_line(generation_result["stdout_tail"])
    generated_records = [
        json.loads(line)
        for line in regenerated_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(generated_records) != 4:
        raise RuntimeError(f"regeneration wrote {len(generated_records)} records, expected 4")

    capture_command = [
        sys.executable,
        "-m",
        "Finetuning.capture_features",
        "--input",
        str(regenerated_path),
        "--output",
        str(feature_dir),
        "--target-model-path",
        str(model_dir),
        "--target-layer-ids",
        "0",
        "--max-length",
        "16",
        "--max-source-tokens",
        "8",
        "--max-summary-tokens",
        "4",
        "--prompt-template",
        "SUMMARIZE {document}",
        "--torch-dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--adaptive-max-batch-size",
        "8",
        "--probe-batches",
        "1",
        "--bucket-window",
        "4",
        "--capture-backend",
        capture_backend,
        "--capture-method",
        "dflash",
        "--parity-samples",
        "2" if capture_backend == "sglang" else "0",
    ]
    if capture_backend == "sglang":
        capture_command.extend(
            [
                "--sglang-attention-backend",
                attention_backend,
                "--sglang-mem-fraction-static",
                "0.40",
                "--sglang-max-running-requests",
                "2",
                "--sglang-max-total-tokens",
                "64",
                "--sglang-context-length",
                "64",
                "--sglang-disable-radix-cache",
            ]
        )
    capture_result = _run(capture_command, cwd=REMOTE_ROOT, env=env)
    capture_summary = _last_json_line(capture_result["stdout_tail"])
    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"capture did not publish {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("capture_backend") != capture_backend:
        raise RuntimeError(
            f"capture manifest backend {manifest.get('capture_backend')!r} "
            f"does not match requested {capture_backend!r}"
        )
    capture_stats = capture_summary.get("stats", {})
    if capture_backend == "sglang":
        parity = capture_stats.get("parity")
        if not isinstance(parity, dict) or parity.get("passed") is not True:
            raise RuntimeError(f"SGLang capture parity gate did not pass: {parity!r}")
    feature_generation = feature_dir / manifest.get("generation_dir", "")
    feature_files = list(feature_generation.glob("feature_*.pt"))
    if len(feature_files) != 4:
        raise RuntimeError(f"capture wrote {len(feature_files)} feature records, expected 4")

    config_path.write_text(
        f"""model:
  target_model_path: {model_dir}
  torch_dtype: bfloat16
  mask_token_id: 0
  num_draft_layers: 1
  draft_intermediate_size: 64
  block_size: 4
  target_layer_ids: [0]
data:
  hidden_states_path: {feature_dir}
  eval_hidden_states_path: {feature_dir}
  max_length: 16
  max_source_tokens: 8
  max_summary_tokens: 4
  chat_template: qwen3
  prompt_template: "SUMMARIZE {{document}}"
  feature_dtype: bfloat16
  num_workers: 0
  pin_memory: true
training:
  max_steps: 1
  batch_size: 1
  accumulation_steps: 1
  learning_rate: 0.001
  warmup_ratio: 0.0
  num_anchors: 2
  objective_chunk_blocks: 2
  attention_backend: eager
  loss_type: dflash
  save_interval: 1
  eval_interval: 1
  log_interval: 1
  adaptive_batch_size: true
  target_memory_fraction: 0.90
  adaptive_min_batch_size: 1
  adaptive_max_batch_size: 8
  adaptive_probe_batches: 1
output_dir: {train_dir}
run_id: modal-smoke
device: cuda
offline: true
""",
        encoding="utf-8",
    )
    train_command = [
        sys.executable,
        "-m",
        "Finetuning.run_train",
        "--config",
        str(config_path),
        "--smoke",
    ]
    train_result = _run(train_command, cwd=REMOTE_ROOT, env=env)
    checkpoint = train_dir / "modal-smoke-step1"
    complete_marker = checkpoint / "COMPLETE"
    if not complete_marker.is_file():
        raise RuntimeError(f"training did not publish a complete checkpoint: {checkpoint}")
    extra = json.loads((checkpoint / "extra.json").read_text(encoding="utf-8"))
    adaptive = extra.get("adaptive_batch")
    if not isinstance(adaptive, dict) or int(adaptive.get("batch_size", 0)) < 1:
        raise RuntimeError("training checkpoint is missing adaptive batch metadata")

    return {
        "status": "passed",
        "gpu": {
            "name": gpu_properties.name,
            "total_memory_gib": round(gpu_properties.total_memory / 2**30, 2),
            "device_index": device,
        },
        "runtime_versions": runtime_versions,
        "workspace": str(work),
        "generation": generation_summary,
        "capture": capture_summary,
        "manifest": manifest,
        "training": {
            "checkpoint": str(checkpoint),
            "adaptive_batch": adaptive,
            "stdout_tail": train_result["stdout_tail"],
        },
        "checks": {
            "generated_records": len(generated_records),
            "feature_records": len(feature_files),
            "complete_checkpoint": complete_marker.is_file(),
        },
    }


@app.local_entrypoint()
def main() -> None:
    # Pass this explicitly: the remote function imports this module in a
    # clean container, where host-only MODAL_CAPTURE_BACKEND is not present.
    result = smoke.remote(CAPTURE_BACKEND, SGLANG_ATTENTION_BACKEND)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
