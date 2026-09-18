from __future__ import annotations

import os
from pathlib import Path

import pytest

from Finetuning.config import RunConfig
from Finetuning.run_train import run_training


@pytest.mark.skipif(
    os.environ.get("FINETUNING_RUN_REAL_QWEN3") != "1",
    reason="set FINETUNING_RUN_REAL_QWEN3=1",
)
def test_qwen3_4b_one_step_local_snapshot() -> None:
    model_path = os.environ.get("FINETUNING_QWEN3_4B_PATH")
    assert model_path and Path(model_path).is_dir()
    config = RunConfig.from_local_qwen3(
        target_model_path=model_path,
        output_dir=Path("outputs/finetuning-qwen3-smoke"),
        max_steps=1,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
    )
    assert run_training(config).is_dir()
