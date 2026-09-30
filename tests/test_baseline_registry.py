from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
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


def test_longbench_ar16_profile_ignores_stale_legacy_eagle_tree_values() -> None:
    config_script = ROOT / "scripts/common/config.sh"
    env = {
        **os.environ,
        "EAGLE_TOTAL_TOKENS": "60",
        "EAGLE_DEPTH": "5",
        "EAGLE_TOP_K": "10",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source '{config_script}'; fast_infer__load_longbench; "
            "printf '%s %s %s' \"$LONG_BENCH_EAGLE_TOTAL_TOKEN\" "
            "\"$LONG_BENCH_EAGLE_DEPTH\" \"$LONG_BENCH_EAGLE_TOP_K\"",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout == "17 16 1"

def test_eagle3_launcher_defaults_to_linear_16_token_draft_block() -> None:
    from Benchmark.common.longbench_adapter import (
        baseline_config_from_env,
        build_adapter_command,
    )

    config = baseline_config_from_env("eagle3", env={})
    command = build_adapter_command(
        "eagle3",
        config=config,
        data_file=ROOT / "datasets/eval_100/vietnews_100.jsonl",
        output=ROOT / "outputs/eagle3.jsonl",
        max_samples=1,
        max_new_tokens=32,
    )
    assert command is not None
    actual = {
        flag: command[command.index(flag) + 1]
        for flag in ("--total-token", "--depth", "--top-k")
    }
    # EaModel subtracts one from total_token to get proposal nodes. One branch
    # with depth 16 gives a single autoregressive draft path of 16 tokens.
    assert actual == {
        "--total-token": "17",
        "--depth": "16",
        "--top-k": "1",
    }


def test_eagle3_smoke_requires_stock_target_logits_parity() -> None:
    from Benchmark.common.longbench_adapter import (
        baseline_config_from_env,
        build_adapter_command,
    )

    config = baseline_config_from_env("eagle3", env={})
    config["smoke"] = True
    config["eagle_check_target_parity"] = False
    command = build_adapter_command(
        "eagle3",
        config=config,
        data_file=ROOT / "datasets/eval_100/vietnews_100.jsonl",
        output=ROOT / "outputs/eagle3-smoke.jsonl",
        max_samples=1,
        max_new_tokens=32,
    )

    assert command is not None
    assert "--check-target-parity" in command


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


def test_eagle3_keeps_internal_naive_pair_even_with_external_reference() -> None:
    from Benchmark.common.longbench_adapter import build_adapter_command

    command = build_adapter_command(
        "eagle3",
        config={
            "model": "/models/Qwen3-4B",
            "eagle_model": "/models/Qwen3-4B-Eagle3",
            "skip_reference": True,
            "temperature": 0,
        },
        data_file=ROOT / "datasets/eval_100/vietnews_100.jsonl",
        output=ROOT / "outputs/test.jsonl",
        max_samples=1,
        max_new_tokens=128,
    )

    assert command is not None
    assert "--skip-naive" not in command


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


def test_target_only_is_an_internal_sglang_adapter_command():
    from Benchmark.common.longbench_adapter import build_adapter_command

    command = build_adapter_command(
        "target_only",
        config={
            "model": "/models/Qwen3-4B",
            "batch_size": 1,
            "max_running_requests": 1,
            "paper_speedup": True,
            "disable_radix_cache": True,
        },
        data_file=ROOT / "datasets/eval_100/vietnews_100.jsonl",
        output=ROOT / "outputs/target_only.jsonl",
        max_samples=1,
        max_new_tokens=8,
    )

    assert command is not None
    assert "--method" in command
    assert command[command.index("--method") + 1] == "target_only"
    assert "--paper-speedup" in command
    assert "--disable-radix-cache" in command


def test_speculative_paper_command_uses_shared_target_only_sidecar():
    from Benchmark.common.longbench_adapter import build_adapter_command

    command = build_adapter_command(
        "domino",
        config={
            "model": "/models/Qwen3-4B",
            "domino_model": "/models/Qwen3-4B-Domino",
            "batch_size": 1,
            "max_running_requests": 1,
            "paper_speedup": True,
            "target_only_reference_file": "/run/references/target_only.jsonl",
        },
        data_file=ROOT / "datasets/eval_100/vietnews_100.jsonl",
        output=ROOT / "outputs/domino.jsonl",
        max_samples=1,
        max_new_tokens=8,
    )

    assert command is not None
    assert command[command.index("--target-only-reference-file") + 1] == "/run/references/target_only.jsonl"
    assert command.count("--disable-radix-cache") == 1


def test_paper_matrix_pins_one_reference_first():
    import pytest

    from Benchmark.common.longbench_adapter import BASELINES
    from Benchmark.run_longbench_200 import _pin_paper_baselines

    pinned = _pin_paper_baselines(list(reversed(BASELINES)), "vanilla_hf")

    assert pinned[0] == "vanilla_hf"
    assert set(pinned) == set(BASELINES)
    with pytest.raises(SystemExit, match="exactly these six"):
        _pin_paper_baselines(["vanilla_fa", "dflash"], "vanilla_fa")


def test_full_paper_run_requires_passing_smoke_gate(tmp_path):
    import json
    import pytest

    from Benchmark.common.longbench_adapter import BASELINES
    from Benchmark.run_longbench_200 import _validate_paper_smoke_audit

    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    (smoke_dir / "run_manifest.json").write_text(json.dumps({
        "run_id": "smoke-1", "baselines": list(BASELINES),
        "datasets": ["vietnews", "wikilingua", "vims", "vlsp"],
        "model": "/models/Qwen3-4B",
    }), encoding="utf-8")
    (smoke_dir / "audit_v2.json").write_text(json.dumps({
        "common_reference": "vanilla_fa",
        "paper_gate": {"status": "smoke_pass"},
    }), encoding="utf-8")

    result = _validate_paper_smoke_audit(
        smoke_dir, common_reference="vanilla_fa", model="/models/Qwen3-4B"
    )
    assert result["run_id"] == "smoke-1"
    with pytest.raises(SystemExit, match="same common reference"):
        _validate_paper_smoke_audit(
            smoke_dir, common_reference="vanilla_hf", model="/models/Qwen3-4B"
        )


def test_shared_sglang_reference_helper_normalizes_one_v2_sidecar(tmp_path, monkeypatch):
    import json

    from Benchmark import run_longbench_200 as runner

    source = [{"id": "sample-1", "prompt": "hello", "reference_output": "gold"}]
    output = tmp_path / "references" / "sglang_target_only" / "vietnews.jsonl"
    captured = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)
        path = kwargs["output_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "sample_id": "sample-1", "status": "success", "contract_version": 2,
            "prompt_token_sha256": "prompt", "generation_config_sha256": "generation",
            "hardware_fingerprint": "gpu", "prompt_token_count_match": True,
            "request_wall_ms": 12.0, "native_elapsed_ms": 12.0, "text": "answer",
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {"status": "success", "returncode": 0}

    monkeypatch.setattr(runner, "_execute_cell_once", fake_execute)
    result = runner._create_shared_sglang_reference(
        dataset="vietnews", source_rows=source, normalized=source, run_dir=tmp_path,
        subset_path=tmp_path / "inputs.jsonl", run_id="r1", model="/models/Qwen3-4B",
        temperature=0.0, warmup_runs=1, max_input_tokens=0, seed=42,
        max_new_tokens=8, timeout_seconds=60, vram={},
    )

    assert result["status"] == "success"
    assert result["success_samples"] == 1
    assert captured["baseline"] == "target_only"
    assert captured["cfg"]["batch_size"] == 1
    assert captured["cfg"]["disable_radix_cache"] is True
    saved = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert saved["sample_order"] == 0
    assert saved["dataset"] == "vietnews"


def test_paper_preflight_rejects_unpinned_sglang(monkeypatch):
    from Benchmark.common import longbench_adapter as adapter

    monkeypatch.setattr(adapter, "_module_importable", lambda name: (True, None))
    monkeypatch.setattr(adapter, "_sglang_algorithm_supported", lambda algorithm: (True, None))
    monkeypatch.setattr(adapter.importlib_metadata, "version", lambda name: "0.5.19")

    result = adapter.preflight_baseline(
        "domino",
        config={
            "model": "org/target",
            "domino_model": "org/draft",
            "paper_speedup": True,
        },
        cuda_available=True,
    )

    assert result["status"] == "missing_dependency"
    assert "requires SGLang 0.5.20" in result["reason"]
    assert result["requirements"]["sglang_version"]["installed"] == "0.5.19"


def test_full_paper_smoke_gate_rejects_changed_dataset_hash(tmp_path):
    import json
    import pytest

    from Benchmark.common.longbench_adapter import BASELINES
    from Benchmark.run_longbench_200 import _validate_paper_smoke_audit

    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    (smoke_dir / "run_manifest.json").write_text(json.dumps({
        "run_id": "smoke-2", "baselines": list(BASELINES),
        "datasets": ["vietnews", "wikilingua", "vims", "vlsp"],
        "model": "/models/Qwen3-4B", "target_revision": "rev-a",
        "tokenizer_revision": "tok-a", "dataset_sha256": {"vietnews": "hash-a"},
    }), encoding="utf-8")
    (smoke_dir / "audit_v2.json").write_text(json.dumps({
        "common_reference": "vanilla_fa",
        "paper_gate": {"status": "smoke_pass"},
    }), encoding="utf-8")

    with pytest.raises(SystemExit, match="dataset content differs"):
        _validate_paper_smoke_audit(
            smoke_dir, common_reference="vanilla_fa", model="/models/Qwen3-4B",
            target_revision="rev-a", tokenizer_revision="tok-a",
            dataset_sha256={"vietnews": "hash-b"},
        )
