from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_finetuning_b200.py"


def _fixture(tmp_path: Path) -> dict[str, Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    train_input = tmp_path / "train.jsonl"
    eval_input = tmp_path / "eval.jsonl"
    record = {"id": "1", "document": "doc", "summary": "sum"}
    for path in (train_input, eval_input):
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "target_model_path": None,
                    "torch_dtype": "bfloat16",
                    "mask_token_id": 0,
                    "num_draft_layers": 1,
                    "block_size": 4,
                },
                "data": {
                    "hidden_states_path": None,
                    "eval_hidden_states_path": None,
                    "max_length": 16,
                    "max_source_tokens": 8,
                    "max_summary_tokens": 4,
                    "chat_template": "qwen3",
                    "prompt_template": "SUMMARIZE {document}",
                    "feature_dtype": "bfloat16",
                },
                "training": {
                    "max_steps": 1,
                    "batch_size": 1,
                    "adaptive_batch_size": False,
                    "adaptive_max_batch_size": 8,
                },
                "output_dir": "unused",
                "run_id": "test-run",
                "device": "auto",
                "offline": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "train_input": train_input,
        "eval_input": eval_input,
        "config": config,
    }


def test_dry_run_prints_all_distributed_stages_and_resolved_paths(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--config",
            str(fixture["config"]),
            "--train-input",
            str(fixture["train_input"]),
            "--eval-input",
            str(fixture["eval_input"]),
            "--target-model-path",
            str(fixture["model"]),
            "--output-root",
            str(output_root),
            "--nproc-per-node",
            "8",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "generate_train" in output
    assert "generate_eval" in output
    assert "cache_train" in output
    assert "cache_eval" in output
    assert "train" in output
    assert "--nproc_per_node 8" in output
    assert str(output_root / "teacher" / "train.jsonl") in output
    assert str(output_root / "features" / "eval") in output
    assert "--adaptive-batch" in output

    resolved = yaml.safe_load(
        (output_root / "run_config.yaml").read_text(encoding="utf-8")
    )
    assert resolved["model"]["target_model_path"] == str(fixture["model"])
    assert resolved["data"]["hidden_states_path"] == str(
        output_root / "features" / "train"
    )
    assert resolved["data"]["eval_hidden_states_path"] == str(
        output_root / "features" / "eval"
    )
    assert resolved["training"]["adaptive_batch_size"] is True


def test_dry_run_skips_valid_completed_teacher_stage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "run"
    teacher_dir = output_root / "teacher"
    state_dir = output_root / ".state"
    teacher_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    teacher_path = teacher_dir / "train.jsonl"
    teacher_path.write_text(
        json.dumps({"id": "1", "document": "doc", "summary": "sum"}) + "\n",
        encoding="utf-8",
    )
    (state_dir / "generate_train.json").write_text(
        json.dumps({"stage": "generate_train", "artifact": str(teacher_path)}) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--config",
            str(fixture["config"]),
            "--train-input",
            str(fixture["train_input"]),
            "--eval-input",
            str(fixture["eval_input"]),
            "--target-model-path",
            str(fixture["model"]),
            "--output-root",
            str(output_root),
            "--nproc-per-node",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "SKIP generate_train" in result.stdout


def test_dry_run_managed_server_generation_uses_cpu_dispatcher(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "managed-run"
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--config",
            str(fixture["config"]),
            "--train-input",
            str(fixture["train_input"]),
            "--eval-input",
            str(fixture["eval_input"]),
            "--target-model-path",
            str(fixture["model"]),
            "--output-root",
            str(output_root),
            "--nproc-per-node",
            "2",
            "--generation-backend",
            "sglang",
            "--generation-launch-servers",
            "--generation-server-gpu-group",
            "0",
            "--generation-server-gpu-group",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    generate_lines = [
        line for line in result.stdout.splitlines() if "RUN generate_" in line
    ]
    assert generate_lines
    assert all("torch.distributed.run" not in line for line in generate_lines)
    assert all("--generation-server-url" in line for line in generate_lines)
    assert "--generation-server-url http://127.0.0.1:30000/v1" in generate_lines[0]
