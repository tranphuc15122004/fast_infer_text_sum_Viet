from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _launcher_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "launch_finetuning_target_servers.py"
    spec = importlib.util.spec_from_file_location("launch_finetuning_target_servers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_server_specs_create_one_openai_endpoint_per_gpu_group() -> None:
    launcher = _launcher_module()
    args = SimpleNamespace(
        backend="sglang",
        model_path="/models/qwen3",
        gpu_groups=["0", "1"],
        host="127.0.0.1",
        base_port=30000,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction=0.88,
        context_length=None,
        served_model_name="qwen3",
        max_num_seqs=0,
    )

    specs = launcher.build_server_specs(args)

    assert [spec.env["CUDA_VISIBLE_DEVICES"] for spec in specs] == ["0", "1"]
    assert [spec.port for spec in specs] == [30000, 30001]
    assert all("--tp-size" in spec.command for spec in specs)
    assert all("--mem-fraction-static" in spec.command for spec in specs)

