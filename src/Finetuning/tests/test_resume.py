from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from Finetuning.capture_features import FeatureManifest, capture_dataset
from Finetuning.generate_targets import (
    _load_completed_records,
    generate_teacher_jsonl,
)


class MockTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": [20 + i for i, _ in enumerate(text.split())]}

    def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: Any) -> list[int]:
        ids = [1]
        for message in messages:
            ids.extend(self(message["content"])["input_ids"])
        return ids + [2]

    def decode(self, ids: list[int], **_kwargs: Any) -> str:
        return " ".join(str(v) for v in ids if v != self.eos_token_id)


class MockTarget:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, input_ids: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        self.call_count += input_ids.shape[0]
        suffix = torch.tensor([[7, 8, 99]] * input_ids.shape[0], dtype=torch.long)
        return torch.cat([input_ids, suffix], dim=1)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_completed_records_handles_valid_and_partial_lines(tmp_path: Path) -> None:
    part_file = tmp_path / "train.jsonl.part"

    # Initially empty
    completed_ids, completed_indices, count = _load_completed_records(part_file)
    assert count == 0
    assert completed_ids == set()
    assert completed_indices == set()

    # Write 2 full valid rows and 1 truncated line simulating mid-crash write
    with part_file.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "doc-1", "_source_index": 0, "summary": "s1"}) + "\n")
        handle.write(json.dumps({"id": "doc-2", "_source_index": 1, "summary": "s2"}) + "\n")
        handle.write('{"id": "doc-3", "docu')  # corrupted partial write

    completed_ids, completed_indices, count = _load_completed_records(part_file)
    assert count == 2
    assert completed_ids == {"doc-1", "doc-2"}
    assert completed_indices == {0, 1}


def test_generate_targets_resumes_from_part_file(tmp_path: Path) -> None:
    source_file = tmp_path / "source.jsonl"
    output_file = tmp_path / "teacher.jsonl"
    part_file = tmp_path / "teacher.jsonl.part"

    examples = [
        {"id": f"doc-{i}", "document": f"văn bản dài {i}", "summary": f"gold {i}"}
        for i in range(5)
    ]
    _write_jsonl(source_file, examples)

    # Pre-populate part_file with first 2 completed records
    pre_completed = [
        {
            "id": "doc-0",
            "_source_index": 0,
            "document": "văn bản dài 0",
            "summary": "7 8",
            "reference_summary": "gold 0",
            "teacher": {"target_model_path": "/mock"},
        },
        {
            "id": "doc-1",
            "_source_index": 1,
            "document": "văn bản dài 1",
            "summary": "7 8",
            "reference_summary": "gold 1",
            "teacher": {"target_model_path": "/mock"},
        },
    ]
    _write_jsonl(part_file, pre_completed)

    target = MockTarget()
    tokenizer = MockTokenizer()

    stats = generate_teacher_jsonl(
        source_file,
        output_file,
        tokenizer=tokenizer,
        target=target,
        target_model_path="/mock",
        max_length=32,
        max_source_tokens=16,
        max_summary_tokens=8,
        chat_template="qwen3",
        device="cpu",
    )

    assert output_file.is_file()
    assert not part_file.exists()  # Finalized and replaced into final output

    with output_file.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    assert len(rows) == 5
    assert stats["written"] == 5
    # The target generator was only invoked for the remaining 3 records (doc-2, doc-3, doc-4)
    assert target.call_count == 3


def test_capture_features_resumes_from_staging(tmp_path: Path) -> None:
    output_dir = tmp_path / "features"
    staging_dir = tmp_path / f".staging_{output_dir.name}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create feature_00000000.pt and feature_00000001.pt in staging
    mock_record_0 = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "loss_mask": torch.tensor([0, 1, 1], dtype=torch.long),
        "hidden_states": torch.randn(3, 8, dtype=torch.float32),
    }
    mock_record_1 = {
        "input_ids": torch.tensor([4, 5], dtype=torch.long),
        "loss_mask": torch.tensor([1, 1], dtype=torch.long),
        "hidden_states": torch.randn(2, 8, dtype=torch.float32),
    }
    torch.save(mock_record_0, staging_dir / "feature_00000000.pt")
    torch.save(mock_record_1, staging_dir / "feature_00000001.pt")

    examples = [
        {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]},
        {"input_ids": [4, 5], "loss_mask": [1, 1]},
        {"input_ids": [6, 7, 8, 9], "loss_mask": [0, 0, 1, 1]},
    ]

    mock_config = SimpleNamespace(num_hidden_layers=2, hidden_size=4, text_config=None)
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    # Hidden states returning 2 layers: embedding + 2 hidden layers
    mock_model.return_value = MagicMock(
        hidden_states=(
            None,
            torch.randn(1, 4, 4, dtype=torch.float32),
            torch.randn(1, 4, 4, dtype=torch.float32),
        )
    )

    with patch("Finetuning.capture_features._load_local_target", return_value=mock_model):
        with patch("Finetuning.capture_features.Path.is_dir", return_value=True):
            manifest = capture_dataset(
                target_model_path=tmp_path / "model",
                prepared_examples=examples,
                output_dir=output_dir,
                target_layer_ids=[0, 1],
                max_length=128,
                device="cpu",
                dtype=torch.float32,
            )

    assert manifest is not None
    assert (output_dir / "manifest.json").is_file()
    assert not staging_dir.exists()  # Staging cleaned up after publishing

    # Check published features
    gen_dir = output_dir / manifest.generation_dir
    feature_files = sorted(gen_dir.glob("feature_*.pt"))
    assert len(feature_files) == 3
    assert feature_files[0].name == "feature_00000000.pt"
    assert feature_files[1].name == "feature_00000001.pt"
    assert feature_files[2].name == "feature_00000002.pt"
