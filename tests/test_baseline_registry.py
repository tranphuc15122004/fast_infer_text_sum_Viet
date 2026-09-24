from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_qwen3_viet_baseline_registry_has_no_duplicate_methods() -> None:
    from Benchmark.common.longbench_adapter import BASELINES

    assert BASELINES == (
        "vanilla_hf",
        "vanilla_fa",
        "eagle3",
        "dflash",
        "domino",
        "dspark",
    )


def test_baseline_config_uses_qwen3_target_and_viet_defaults(monkeypatch) -> None:
    from Benchmark.common.longbench_adapter import baseline_config_from_env

    monkeypatch.setenv("MODEL_TARGET", "/models/Qwen3-4B")
    monkeypatch.setenv("LONG_BENCH_EAGLE_MODEL", "/models/eagle3-qwen3-4b")
    config = baseline_config_from_env("eagle3")
    assert config["model"] == "/models/Qwen3-4B"
    assert config["eagle_model"] == "/models/eagle3-qwen3-4b"


def test_sglang_config_keeps_auto_concurrency_resolvable(monkeypatch) -> None:
    from Benchmark.common.longbench_adapter import baseline_config_from_env, build_adapter_command

    monkeypatch.setenv("MODEL_TARGET", "/models/Qwen3-4B")
    monkeypatch.setenv("MODEL_DOMINO_DRAFT", "/models/Qwen3-4B-Domino")
    monkeypatch.setenv("LONG_BENCH_BATCH_SIZE", "auto")
    monkeypatch.delenv("LONG_BENCH_MAX_RUNNING_REQUESTS", raising=False)
    config = baseline_config_from_env("domino")
    assert config["batch_size"] == "auto"
    assert config["max_running_requests"] == "auto"
    command = build_adapter_command(
        "domino",
        config=config,
        data_file=ROOT / "datasets/eval_100/vietnews_100.jsonl",
        output=ROOT / "outputs/test.jsonl",
        max_samples=1,
        max_new_tokens=8,
    )
    assert command is not None
    assert "auto" in command
    assert "--max-running-requests" not in command


def test_dflash_command_propagates_warmup_runs() -> None:
    from Benchmark.common.longbench_adapter import build_adapter_command

    command = build_adapter_command(
        "dflash",
        config={
            "model": "/models/Qwen3-4B",
            "dflash_model": "/models/Qwen3-4B-DFlash",
            "warmup_runs": 1,
        },
        data_file=ROOT / "datasets/eval_100/vietnews_100.jsonl",
        output=ROOT / "outputs/test.jsonl",
        max_samples=1,
        max_new_tokens=8,
    )

    assert command is not None
    warmup_flag = command.index("--warmup-runs")
    assert command[warmup_flag + 1] == "1"


def test_reference_selection_ignores_vanilla_status_records(tmp_path) -> None:
    from Benchmark.run_longbench_200 import _select_external_reference

    for baseline, row in (
        ("vanilla_fa", {"status": "unsupported_cpu", "sample_id": "x"}),
        (
            "vanilla_hf",
            {"status": "success", "sample_id": "x", "e2e_ms": 12.0},
        ),
    ):
        path = tmp_path / baseline / "vietnews.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    selected = _select_external_reference(
        tmp_path,
        "vietnews",
        ["vanilla_hf", "vanilla_fa", "eagle3"],
    )
    assert selected == tmp_path / "vanilla_hf" / "vietnews.jsonl"
