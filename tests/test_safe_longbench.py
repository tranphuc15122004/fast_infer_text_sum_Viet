from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_unresolved_samples_include_missing_and_failed_rows(tmp_path) -> None:
    from Benchmark.run_longbench_200 import _unresolved_sample_records

    output = tmp_path / "cell.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "a", "status": "success"}),
                json.dumps({"sample_id": "b", "status": "failed", "reason": "OOM"}),
                json.dumps({"type": "summary", "status": "failed"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    unresolved = _unresolved_sample_records(output, source_records=source)

    assert [row["id"] for row in unresolved] == ["b", "c"]


def test_child_env_preserves_scheduler_cuda_visibility(monkeypatch) -> None:
    from Benchmark.run_longbench_200 import _safe_env

    # A cluster scheduler may expose a physical GPU through a UUID or a
    # remapped index. Empty config aliases must not turn that mapping into an
    # empty CUDA_VISIBLE_DEVICES value for the child.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-by-scheduler")
    monkeypatch.setenv("LONG_BENCH_GPU_IDS", "")
    monkeypatch.setenv("FI_GPU_IDS", "")

    child_env = _safe_env()

    assert child_env["CUDA_VISIBLE_DEVICES"] == "GPU-allocated-by-scheduler"


def test_child_env_explicit_gpu_pin_overrides_inherited_visibility(monkeypatch) -> None:
    from Benchmark.run_longbench_200 import _safe_env

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-by-scheduler")

    child_env = _safe_env(cuda_visible_devices="0")

    assert child_env["CUDA_VISIBLE_DEVICES"] == "0"


def test_rewrite_safe_output_preserves_success_and_fills_failed_samples(tmp_path) -> None:
    from Benchmark.run_longbench_200 import _rewrite_safe_cell_output

    output = tmp_path / "cell.jsonl"
    source = [
        {"id": "a", "reference": "ra", "task_type": "summarization"},
        {"id": "b", "reference": "rb", "task_type": "summarization"},
    ]
    successful = [
        {
            "sample_id": "a",
            "status": "success",
            "text": "summary a",
            "e2e_ms": 10.0,
        }
    ]

    result = _rewrite_safe_cell_output(
        output,
        baseline="vanilla_hf",
        dataset="vietnews",
        source_records=source,
        successful_rows=successful,
        unresolved_reasons={"b": "retry exhausted after OOM"},
        model="/models/Qwen3-4B",
        config={"device": "cuda", "max_new_tokens": 8},
        run_id="run-1",
        retry_count=2,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["unresolved_sample_count"] == 1
    assert [row.get("sample_id") for row in rows[:-1]] == ["a", "b"]
    assert rows[0]["status"] == "success"
    assert rows[1]["status"] == "failed"
    assert rows[1]["reason"] == "retry exhausted after OOM"
    assert rows[-1]["type"] == "summary"
    assert rows[-1]["safe_eval_complete"] is False


def test_retry_attempt_path_is_unique_per_sample_and_attempt(tmp_path) -> None:
    from Benchmark.run_longbench_200 import _safe_retry_paths

    first = _safe_retry_paths(
        tmp_path, baseline="dflash", dataset="vims", sample_id="row/1", attempt=1
    )
    second = _safe_retry_paths(
        tmp_path, baseline="dflash", dataset="vims", sample_id="row/1", attempt=2
    )

    assert first["output"] != second["output"]
    assert first["log"] != second["log"]
    assert "row_1" in first["output"].name
    assert first["output"].parent == tmp_path / "attempts" / "dflash" / "vims"


def test_retry_unresolved_samples_runs_each_sample_and_recovers(monkeypatch, tmp_path) -> None:
    import Benchmark.run_longbench_200 as runner

    output = tmp_path / "vanilla_hf" / "vietnews.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"sample_id": "a", "status": "success", "text": "done"})
        + "\n",
        encoding="utf-8",
    )
    raw = [
        {"id": "a", "document": "A", "reference": "RA"},
        {"id": "b", "document": "B", "reference": "RB"},
    ]
    normalized = [
        {"id": "a", "prompt": "A", "reference": "RA", "raw": raw[0]},
        {"id": "b", "prompt": "B", "reference": "RB", "raw": raw[1]},
    ]
    commands: list[list[str]] = []

    monkeypatch.setattr(
        runner,
        "build_adapter_command",
        lambda *args, **kwargs: ["fake-child"],
    )

    def fake_run_child(command, *, output, log_path, timeout_seconds, cuda_visible_devices=None):
        commands.append(list(command))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "sample_id": "b",
                    "status": "success",
                    "text": "recovered",
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "e2e_ms": 3.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "success",
            "returncode": 0,
            "elapsed_ms": 3.0,
            "output_exists": True,
            "log": str(log_path),
            "log_tail": "",
            "command": list(command),
        }

    monkeypatch.setattr(runner, "_run_child", fake_run_child)

    result = runner._retry_unresolved_samples(
        baseline="vanilla_hf",
        dataset="vietnews",
        source_rows=raw,
        normalized=normalized,
        output_path=output,
        run_dir=tmp_path,
        cfg={"model": "/models/Qwen3-4B", "device": "cuda"},
        max_new_tokens=8,
        timeout_seconds=10,
        run_id="run-1",
        sample_retries=2,
        retry_backoff_seconds=0,
    )

    assert result["safe_eval_complete"] is True
    assert result["retried_sample_count"] == 1
    assert len(commands) == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row.get("sample_id") for row in rows[:-1]] == ["a", "b"]
    assert rows[1]["text"] == "recovered"


def test_main_continues_to_next_cell_after_orchestration_exception(monkeypatch, tmp_path) -> None:
    import Benchmark.run_longbench_200 as runner

    monkeypatch.setattr(runner, "_effective_cuda_available", lambda: False)
    monkeypatch.setattr(
        runner,
        "preflight_baseline",
        lambda *args, **kwargs: {
            "status": "ready",
            "reason": None,
            "requirements": {},
        },
    )
    monkeypatch.setattr(
        runner,
        "_run_collector",
        lambda *args, **kwargs: {"status": "skipped", "reason": "test"},
    )
    calls: list[str] = []

    def fake_execute(**kwargs):
        dataset = kwargs["dataset"]
        calls.append(dataset)
        if dataset == "vietnews":
            raise RuntimeError("synthetic cell failure")
        output = kwargs["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(
                json.dumps(
                    {
                        "sample_id": row["id"],
                        "status": "success",
                        "text": "ok",
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "e2e_ms": 1.0,
                    }
                )
                for row in kwargs["normalized"]
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "success",
            "returncode": 0,
            "elapsed_ms": 1.0,
            "output_exists": True,
            "log": "",
            "log_tail": "",
            "command": ["fake"],
        }

    monkeypatch.setattr(runner, "_execute_cell_once", fake_execute)

    exit_code = runner.main(
        [
            "--mode",
            "smoke",
            "--baselines",
            "vanilla_hf",
            "--datasets",
            "vietnews wikilingua",
            "--data-dir",
            str(ROOT / "datasets" / "eval_100"),
            "--output-dir",
            str(tmp_path),
            "--allow-unsupported",
            "--no-collect",
            "--no-retry-failed-samples",
        ]
    )

    assert exit_code == 1
    assert calls == ["vietnews", "wikilingua"]
    run_dir = next(tmp_path.iterdir())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert [cell["dataset"] for cell in manifest["cells"]] == ["vietnews", "wikilingua"]
    assert manifest["cells"][0]["status"] == "failed"
    assert manifest["cells"][1]["status"] == "success"


def test_retry_unresolved_sample_retries_after_oom_log(monkeypatch, tmp_path) -> None:
    import Benchmark.run_longbench_200 as runner

    output = tmp_path / "cell.jsonl"
    raw = [{"id": "oom-1", "document": "D", "reference": "R"}]
    normalized = [
        {"id": "oom-1", "prompt": "D", "reference": "R", "raw": raw[0]}
    ]
    attempts = 0

    monkeypatch.setattr(
        runner,
        "build_adapter_command",
        lambda *args, **kwargs: ["fake-child"],
    )

    def fake_run_child(command, *, output, log_path, timeout_seconds, cuda_visible_devices=None):
        nonlocal attempts
        attempts += 1
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if attempts == 1:
            log_path.write_text("RuntimeError: CUDA out of memory", encoding="utf-8")
            return {
                "status": "failed",
                "returncode": 1,
                "elapsed_ms": 1.0,
                "output_exists": False,
                "log": str(log_path),
                "log_tail": "CUDA out of memory",
                "command": list(command),
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "sample_id": "oom-1",
                    "status": "success",
                    "text": "recovered",
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "e2e_ms": 2.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "success",
            "returncode": 0,
            "elapsed_ms": 2.0,
            "output_exists": True,
            "log": str(log_path),
            "log_tail": "",
            "command": list(command),
        }

    monkeypatch.setattr(runner, "_run_child", fake_run_child)

    result = runner._retry_unresolved_samples(
        baseline="vanilla_hf",
        dataset="vietnews",
        source_rows=raw,
        normalized=normalized,
        output_path=output,
        run_dir=tmp_path,
        cfg={"model": "/models/Qwen3-4B", "device": "cuda"},
        max_new_tokens=8,
        timeout_seconds=10,
        run_id="run-oom",
        sample_retries=2,
        retry_backoff_seconds=0,
    )

    assert result["safe_eval_complete"] is True
    assert attempts == 2
    assert result["retry_history"][0]["reason"] == "retry failed with CUDA OOM"


def test_metric_collector_ignores_retry_attempt_artifacts(tmp_path) -> None:
    from Benchmark.collect_metrics import load_run_records

    main = tmp_path / "vanilla_hf" / "vietnews.jsonl"
    main.parent.mkdir(parents=True, exist_ok=True)
    main.write_text(
        json.dumps({"sample_id": "a", "status": "success", "e2e_ms": 1.0}) + "\n",
        encoding="utf-8",
    )
    attempt = tmp_path / "attempts" / "vanilla_hf" / "vietnews" / "a.a1.output.jsonl"
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(
        json.dumps({"sample_id": "a", "status": "success", "e2e_ms": 999.0}) + "\n",
        encoding="utf-8",
    )

    loaded = load_run_records(tmp_path)

    assert len(loaded["records"]) == 1
    assert loaded["records"][0]["e2e_ms"] == 1.0
