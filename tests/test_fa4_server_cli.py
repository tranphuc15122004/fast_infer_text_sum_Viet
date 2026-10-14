from __future__ import annotations

import pytest
import os
import subprocess
import sys
import types
from pathlib import Path

from Benchmark import fa4_server
from Benchmark.fa4_server import build_parser, resolve_server_models, runner_kwargs


def test_server_cli_exposes_native_profiles_and_keeps_parity_diagnostic():
    args = build_parser().parse_args([])
    kwargs = runner_kwargs(args)
    assert kwargs["eagle_total_token"] == 17
    assert kwargs["eagle_depth"] == 16
    assert kwargs["eagle_top_k"] == 1
    assert kwargs["domino_cuda_graph"] is True
    assert kwargs["phase_timing_mode"] == "separate"
    assert kwargs["strict_greedy_parity"] is False
    assert kwargs["warmup_tokens"] == 512
    assert kwargs["require_speedup"] is False


def test_server_cli_propagates_explicit_diagnostic_overrides():
    args = build_parser().parse_args([
        "--eagle-total-token", "18", "--eagle-depth", "4", "--eagle-top-k", "2",
        "--no-domino-cuda-graph", "--phase-timing-mode", "inline",
        "--strict-greedy-parity", "--require-speedup",
    ])
    kwargs = runner_kwargs(args)
    assert kwargs["eagle_total_token"] == 18
    assert kwargs["eagle_depth"] == 4
    assert kwargs["eagle_top_k"] == 2
    assert kwargs["domino_cuda_graph"] is False
    assert kwargs["phase_timing_mode"] == "inline"
    assert kwargs["strict_greedy_parity"] is True
    assert kwargs["require_speedup"] is True


def test_server_model_resolution_uses_master_config_paths():
    env = {
        "MODEL_TARGET": "/models/Qwen3-4B",
        "MODEL_EAGLE_DRAFT": "/models/Qwen3-4B-eagle3",
        "MODEL_DFLASH_DRAFT": "/models/Qwen3-4B-dflash16",
        "MODEL_DOMINO_DRAFT": "/models/Qwen3-4B-domino16",
        "MODEL_DSPARK_DRAFT": "/models/Qwen3-4B-dspark7",
    }

    assert resolve_server_models(env) == {
        "vanilla_hf": "/models/Qwen3-4B",
        "eagle3": "/models/Qwen3-4B-eagle3",
        "dflash": "/models/Qwen3-4B-dflash16",
        "domino": "/models/Qwen3-4B-domino16",
        "dspark": "/models/Qwen3-4B-dspark7",
    }


def test_server_model_resolution_fails_when_master_path_is_missing():
    with pytest.raises(ValueError, match="MODEL_DFLASH_DRAFT"):
        resolve_server_models({
            "MODEL_TARGET": "/models/Qwen3-4B",
            "MODEL_EAGLE_DRAFT": "/models/eagle",
            "MODEL_DOMINO_DRAFT": "/models/domino",
            "MODEL_DSPARK_DRAFT": "/models/dspark",
        })


def test_server_model_resolution_only_requires_selected_vanilla_dflash_pair():
    env = {
        "MODEL_TARGET": "/models/Qwen3-4B",
        "MODEL_DFLASH_DRAFT": "/outputs/finetuned-dflash",
    }

    assert resolve_server_models(env, "vanilla_hf,dflash") == {
        "vanilla_hf": "/models/Qwen3-4B",
        "dflash": "/outputs/finetuned-dflash",
    }


def test_server_cli_accepts_vietbench_data_directory():
    args = build_parser().parse_args([
        "--data-dir", "datasets/eval_100",
        "--datasets", "vietnews,vlsp",
        "--methods", "vanilla_hf,dflash",
    ])

    kwargs = runner_kwargs(args)
    assert kwargs["data_dir"] == "datasets/eval_100"
    assert kwargs["datasets"] == "vietnews,vlsp"
    assert kwargs["methods"] == "vanilla_hf,dflash"


def test_server_cli_defaults_to_committed_vietbench_directory():
    args = build_parser().parse_args([])

    assert runner_kwargs(args)["data_dir"] == "datasets/eval_100"


def test_native_fa4_runner_resolves_vietbench_jsonl_names(tmp_path):
    from Benchmark.native_fa4_benchmark import resolve_dataset_file

    dataset = tmp_path / "vietnews_100.jsonl"
    dataset.touch()

    assert resolve_dataset_file(tmp_path, "vietnews") == dataset


def test_server_main_routes_selected_models_and_data_to_native_runner(monkeypatch, tmp_path):
    captured = {}

    def run_flashattn_benchmark(**kwargs):
        captured.update(kwargs)
        return {"summary": {"status": "preflight_passed"}}

    native_runner = types.ModuleType("Benchmark.native_fa4_benchmark")
    native_runner.run_flashattn_benchmark = run_flashattn_benchmark
    monkeypatch.setitem(sys.modules, "Benchmark.native_fa4_benchmark", native_runner)
    monkeypatch.setenv("MODEL_TARGET", "/models/Qwen3-4B")
    monkeypatch.setenv("MODEL_DFLASH_DRAFT", "/outputs/finetuned-dflash")
    monkeypatch.setenv("FI_OFFLINE", "1")

    data_dir = tmp_path / "eval_100"
    output_dir = tmp_path / "fa4-results"
    result = fa4_server.main([
        "--mode", "smoke",
        "--datasets", "vietnews",
        "--methods", "vanilla_hf,dflash",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--preflight-only",
    ])

    assert result == 0
    assert captured["data_dir"] == str(data_dir.resolve())
    assert captured["output_dir"] == str(output_dir.resolve())
    assert captured["methods"] == "vanilla_hf,dflash"
    assert os.environ["MODEL_DFLASH_DRAFT"] == "/outputs/finetuned-dflash"


def test_server_cli_exposes_smoke_and_representative_modes():
    args = build_parser().parse_args(
        ["--mode", "representative", "--datasets", "vietnews", "--methods", "all"]
    )

    assert args.mode == "representative"
    assert args.datasets == "vietnews"
    assert args.methods == "all"
    assert args.output_dir == "outputs/fa4_native_benchmark"


def test_server_mode_import_does_not_load_modal_sdk():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["FA4_EXECUTION_BACKEND"] = "server"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), str(root / "scripts"), env.get("PYTHONPATH", ""))
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, modal_flashattn_pilot as runner; "
            "assert runner.SERVER_MODE and not runner.USE_MODAL; "
            "assert 'modal' not in sys.modules",
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_shell_launcher_uses_server_cli_and_shared_master_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    master = tmp_path / "master.env"
    master.write_text(
        "\n".join(
            (
                f'FI_PYTHON="{sys.executable}"',
                'FI_DEVICE="cuda"',
                'FI_GPU_IDS="0"',
                'FI_TARGET_GPU="B200"',
                'FI_OFFLINE="1"',
                f'FI_HF_HOME="{tmp_path / "hf"}"',
                'MODEL_TARGET="/models/target"',
                'MODEL_EAGLE_DRAFT="/models/eagle"',
                'MODEL_DFLASH_DRAFT="/models/dflash"',
                'MODEL_DOMINO_DRAFT="/models/domino"',
                'MODEL_DSPARK_DRAFT="/models/dspark"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["FAST_INFER_MASTER_CONFIG"] = str(master)
    env["FAST_INFER_CACHE_ROOT"] = str(tmp_path / "cache")

    result = subprocess.run(
        ["bash", "scripts/run_fa4_benchmark.sh", "--help"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "trực tiếp trên" in result.stdout
    assert "--mode {smoke,representative,full}" in result.stdout
