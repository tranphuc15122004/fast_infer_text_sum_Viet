from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _launcher_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_finetuning_b200.py"
    spec = importlib.util.spec_from_file_location("run_finetuning_b200", path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)
    return launcher


def test_generation_command_args_selects_server_pool_without_torchrun() -> None:
    launcher = _launcher_module()

    args = SimpleNamespace(
        generation_backend="sglang",
        generation_server_urls=["http://127.0.0.1:30000/v1", "http://127.0.0.1:30001/v1"],
        generation_model="qwen3",
        generation_concurrency_per_server=16,
        generation_timeout_seconds=90,
        generation_retries=2,
        generation_backoff_seconds=0.25,
        generation_window_size=0,
    )

    command = launcher.generation_command_args(args)

    assert command[:2] == ["--generation-backend", "sglang"]
    assert command.count("--generation-server-url") == 2
    assert "--generation-model" in command
    assert "--generation-concurrency-per-server" in command


def test_generation_server_urls_are_derived_for_managed_gpu_servers() -> None:
    launcher = _launcher_module()
    args = SimpleNamespace(
        generation_launch_servers=True,
        generation_backend="sglang",
        generation_server_urls=[],
        generation_server_gpu_groups=["2", "3"],
        generation_server_host="127.0.0.1",
        generation_server_base_port=30100,
    )

    assert launcher.resolve_generation_server_urls(args, nproc_per_node=2) == [
        "http://127.0.0.1:30100/v1",
        "http://127.0.0.1:30101/v1",
    ]
