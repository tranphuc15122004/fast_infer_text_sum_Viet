from __future__ import annotations

import json
import math

import pytest
import torch
from torch.utils.data import Dataset

try:
    from Finetuning.config import RunConfig, load_run_config
    from Finetuning.run_train import evaluate_checkpoint, run_training
except ModuleNotFoundError as exc:  # Red phase: assembly is not ported yet.
    _IMPORT_ERROR = exc


def _require_api() -> None:
    if "_IMPORT_ERROR" in globals():
        pytest.fail(f"Finetuning integration API is not implemented: {_IMPORT_ERROR}")


def test_synthetic_end_to_end_training(tmp_path) -> None:
    _require_api()
    config = RunConfig.from_synthetic(
        output_dir=tmp_path / "run",
        max_steps=4,
        batch_size=1,
        eval_interval=1,
        attention_backend="eager",
    )
    final_checkpoint = run_training(config)
    assert final_checkpoint.is_dir()
    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    steps = [row for row in records if row.get("type") == "step"]
    assert len(steps) == 4
    losses = [row["loss"] for row in steps]
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]
    restored = evaluate_checkpoint(final_checkpoint, config)
    assert math.isfinite(restored["loss"])


def test_cli_rejects_missing_feature_source(tmp_path) -> None:
    _require_api()
    path = tmp_path / "invalid.yaml"
    path.write_text("model: {}\ndata: {}\ntraining: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hidden_states_path"):
        load_run_config(path)


def test_real_config_requires_a_pre_captured_feature_store(tmp_path) -> None:
    _require_api()
    path = tmp_path / "raw-trajectory.yaml"
    path.write_text(
        """
model:
  target_model_path: /models/qwen3
data:
  train_data_path: /data/teacher.jsonl
  hidden_states_path: null
  max_length: 2048
  max_source_tokens: 1536
  max_summary_tokens: 384
training: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="capture"):
        load_run_config(path)


def test_real_config_accepts_explicit_budgets_and_loader_options(tmp_path) -> None:
    _require_api()
    path = tmp_path / "real.yaml"
    path.write_text(
        """
model:
  target_model_path: /models/qwen3
data:
  hidden_states_path: /cache/features
  max_length: 2048
  max_source_tokens: 1536
  max_summary_tokens: 384
  num_workers: 2
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 3
training:
  shuffle: true
""",
        encoding="utf-8",
    )

    config = load_run_config(path)

    assert config.data.max_source_tokens == 1536
    assert config.data.max_summary_tokens == 384
    assert config.data.num_workers == 2
    assert config.data.prefetch_factor == 3
    assert config.training.shuffle is True


def test_feature_loader_remains_lazy_dataloader() -> None:
    _require_api()
    from Finetuning.run_train import _loader
    from torch.utils.data import DataLoader

    class TinyFeatures(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            del index
            return {
                "input_ids": torch.tensor([1, 2, 3]),
                "loss_mask": torch.tensor([0.0, 1.0, 0.0]),
                "hidden_states": torch.zeros(3, 4),
            }

    dataset = TinyFeatures()
    loader = _loader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )

    assert isinstance(loader, DataLoader)
    assert loader.dataset is dataset


def test_train_rejects_feature_cache_with_a_different_prompt_contract() -> None:
    _require_api()
    from Finetuning.config import DataConfig, ModelConfig, TrainingConfig
    from Finetuning.features import FeatureManifest
    from Finetuning.run_train import _validate_feature_manifest

    config = RunConfig(
        model=ModelConfig(target_model_path="/models/qwen3"),
        data=DataConfig(
            hidden_states_path="/cache/features",
            max_length=32,
            max_source_tokens=16,
            max_summary_tokens=8,
        ),
        training=TrainingConfig(),
    ).validate()
    layers = [1, 3]
    manifest = FeatureManifest(
        model_id="/models/qwen3",
        tokenizer_id="/models/qwen3",
        layer_ids=layers,
        hidden_size=4,
        max_length=32,
        hidden_states_dtype="torch.float32",
        prompt_contract={
            "chat_template": "qwen3",
            "max_source_tokens": 16,
            "max_summary_tokens": 8,
            "prompt_template": config.data.prompt_template,
        },
    )

    _validate_feature_manifest(manifest, config, layers)
    manifest.prompt_contract["max_summary_tokens"] = 12
    with pytest.raises(ValueError, match="prompt_contract"):
        _validate_feature_manifest(manifest, config, layers)
