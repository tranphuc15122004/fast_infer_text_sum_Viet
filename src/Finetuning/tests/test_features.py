from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

try:
    from Finetuning.capture_features import capture_dataset
    from Finetuning.features import (
        FeatureManifest,
        OfflineFeatureDataset,
        collate_features,
        validate_feature_record,
    )
except ModuleNotFoundError as exc:  # Red phase: the feature contract is absent.
    _FEATURE_IMPORT_ERROR = exc


def _require_feature_api() -> None:
    if "_FEATURE_IMPORT_ERROR" in globals():
        pytest.fail(f"offline feature API is not implemented: {_FEATURE_IMPORT_ERROR}")


def tiny_manifest() -> FeatureManifest:
    _require_feature_api()
    return FeatureManifest(
        model_id="tiny-qwen3",
        revision="rev-1",
        tokenizer_id="tiny-tokenizer",
        layer_ids=[1, 3],
        hidden_size=4,
        max_length=8,
        hidden_states_dtype="torch.float32",
    )


def test_feature_manifest_round_trip_preserves_contract_fields() -> None:
    manifest = tiny_manifest()

    restored = FeatureManifest.from_dict(manifest.to_dict())

    assert restored.model_id == "tiny-qwen3"
    assert restored.revision == "rev-1"
    assert restored.tokenizer_id == "tiny-tokenizer"
    assert restored.layer_ids == [1, 3]
    assert restored.max_length == 8
    assert restored.feature_width == 8
    assert restored.hidden_states_dtype == "torch.float32"


def test_invalid_feature_sequence_lengths_are_rejected() -> None:
    manifest = tiny_manifest()
    record = {
        "input_ids": torch.ones(7, dtype=torch.long),
        "loss_mask": torch.ones(8),
        "hidden_states": torch.ones(8, 8),
    }

    with pytest.raises(ValueError, match="sequence lengths"):
        validate_feature_record(record, manifest)


def test_invalid_feature_width_and_dtypes_are_rejected() -> None:
    manifest = tiny_manifest()
    base = {
        "input_ids": torch.ones(4, dtype=torch.long),
        "loss_mask": torch.tensor([0, 1, 1, 0], dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="feature width"):
        validate_feature_record(
            {**base, "hidden_states": torch.ones(4, 7)}, manifest
        )
    with pytest.raises(ValueError, match="dtype"):
        validate_feature_record(
            {
                **base,
                "input_ids": base["input_ids"].to(torch.int32),
                "hidden_states": torch.ones(4, 8),
            },
            manifest,
        )


def test_manifest_and_records_reject_non_integer_input_ids() -> None:
    manifest = tiny_manifest()
    with pytest.raises(ValueError, match="input_ids.*integer"):
        FeatureManifest(
            model_id="tiny-qwen3",
            layer_ids=[1, 3],
            hidden_size=4,
            max_length=8,
            input_ids_dtype="torch.float32",
        )
    with pytest.raises(ValueError, match="input_ids.*integer"):
        validate_feature_record(
            {
                "input_ids": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "loss_mask": torch.tensor([0, 1, 1, 0], dtype=torch.float32),
                "hidden_states": torch.ones(4, 8),
            },
            manifest,
        )


def test_fractional_loss_mask_is_rejected() -> None:
    manifest = tiny_manifest()

    with pytest.raises(ValueError, match="binary"):
        validate_feature_record(
            {
                "input_ids": torch.ones(4, dtype=torch.long),
                "loss_mask": torch.tensor([0.0, 0.5, 1.0, 0.0]),
                "hidden_states": torch.ones(4, 8),
            },
            manifest,
        )


def test_collator_canonicalizes_integral_ids_and_binary_masks() -> None:
    _require_feature_api()

    batch = collate_features(
        [
            {
                "input_ids": torch.tensor([1.0, 2.0]),
                "loss_mask": torch.tensor([0, 1], dtype=torch.int64),
                "hidden_states": torch.ones(2, 4),
            }
        ]
    )

    assert batch["input_ids"].dtype == torch.long
    assert batch["loss_mask"].dtype == torch.float32


def _write_feature(path, *, length: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_ids": torch.arange(1, length + 1, dtype=torch.long),
        "loss_mask": torch.tensor([0, 1, 1, 1, 0], dtype=torch.float32)[:length],
        "hidden_states": torch.ones(length, 8),
    }
    if path.name.endswith(".gz"):
        with gzip.open(path, "wb") as handle:
            torch.save(payload, handle)
    else:
        torch.save(payload, path)


def test_offline_dataset_recurses_reads_gzip_and_truncates_consistently(tmp_path) -> None:
    _require_feature_api()
    manifest = FeatureManifest(
        model_id="tiny-qwen3", layer_ids=[1, 3], hidden_size=4, max_length=3
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    _write_feature(tmp_path / "nested" / "plain.ckpt")
    _write_feature(tmp_path / "deeper" / "compressed.ckpt.gz")

    dataset = OfflineFeatureDataset(tmp_path)

    assert len(dataset) == 2
    for record in dataset:
        assert record["input_ids"].shape == (3,)
        assert record["loss_mask"].shape == (3,)
        assert record["hidden_states"].shape == (3, 8)
        assert record["loss_mask"].tolist() == [0.0, 1.0, 1.0]


def test_feature_validation_rejects_missing_adjacent_supervision() -> None:
    manifest = tiny_manifest()
    record = {
        "input_ids": torch.ones(4, dtype=torch.long),
        "loss_mask": torch.tensor([1, 0, 1, 0], dtype=torch.float32),
        "hidden_states": torch.ones(4, 8),
    }

    with pytest.raises(ValueError, match="two consecutive supervised tokens"):
        validate_feature_record(record, manifest)


def test_collator_right_pads_without_mixing_feature_width() -> None:
    _require_feature_api()

    batch = collate_features(
        [
            {
                "input_ids": torch.tensor([1, 2], dtype=torch.long),
                "loss_mask": torch.tensor([0, 1], dtype=torch.float32),
                "hidden_states": torch.ones(2, 4),
            },
            {
                "input_ids": torch.tensor([3], dtype=torch.long),
                "loss_mask": torch.tensor([1], dtype=torch.float32),
                "hidden_states": torch.ones(1, 4) * 2,
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 2)
    assert batch["loss_mask"].shape == (2, 2)
    assert batch["hidden_states"].shape == (2, 2, 4)
    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["loss_mask"].tolist() == [[0, 1], [1, 0]]
    assert torch.equal(batch["hidden_states"][1, 1], torch.zeros(4))


def test_offline_dataset_reads_cpu_tensor_records(tmp_path) -> None:
    _require_feature_api()
    manifest = tiny_manifest()
    tmp_path.joinpath("manifest.json").write_text(
        __import__("json").dumps(manifest.to_dict()), encoding="utf-8"
    )
    torch.save(
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "loss_mask": torch.tensor([0, 1, 1, 0], dtype=torch.float32),
            "hidden_states": torch.ones(4, 8),
        },
        tmp_path / "feature_000000.pt",
    )

    dataset = OfflineFeatureDataset(tmp_path)
    record = dataset[0]

    assert len(dataset) == 1
    assert all(value.device.type == "cpu" for value in record.values())


def test_offline_dataset_defers_tensor_deserialization_until_getitem(
    tmp_path, monkeypatch
) -> None:
    _require_feature_api()
    manifest = tiny_manifest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    _write_feature(tmp_path / "feature_000000.pt")
    _write_feature(tmp_path / "feature_000001.pt")
    import Finetuning.features as feature_module

    calls = 0
    original_load = feature_module._load_record

    def counted_load(path):
        nonlocal calls
        calls += 1
        return original_load(path)

    monkeypatch.setattr(feature_module, "_load_record", counted_load)
    dataset = OfflineFeatureDataset(tmp_path)

    assert calls == 0
    _ = dataset[0]
    assert calls == 1


def test_offline_dataset_requires_manifest_before_loading(tmp_path) -> None:
    _require_feature_api()

    with pytest.raises(FileNotFoundError, match="manifest"):
        OfflineFeatureDataset(tmp_path)


def test_offline_dataset_rejects_internal_and_record_symlinks(tmp_path) -> None:
    _require_feature_api()
    manifest = tiny_manifest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({**manifest.to_dict(), "generation_dir": "active"}),
        encoding="utf-8",
    )
    outside = tmp_path.parent / "outside_features"
    outside.mkdir()
    (outside / "feature_00000000.pt").write_bytes(b"not a tensor")
    (tmp_path / "active").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        OfflineFeatureDataset(tmp_path)

    outside_root = tmp_path / "outside_root"
    outside_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        OfflineFeatureDataset(alias / "features")

    (tmp_path / "active").unlink()
    (tmp_path / "active").mkdir()
    (tmp_path / "active" / "feature_00000000.pt").symlink_to(
        outside / "feature_00000000.pt"
    )
    with pytest.raises(ValueError, match="symlink"):
        OfflineFeatureDataset(tmp_path)


class FakeTargetModel(nn.Module):
    def __init__(self, *, bad_width: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=3,
            hidden_size=4,
            _commit_hash="local-rev",
        )
        self.bad_width = bad_width
        self.seen_no_grad = False
        self.seen_output_hidden_states = False

    def forward(self, input_ids, *, output_hidden_states, use_cache=False):
        del use_cache
        self.seen_no_grad = not torch.is_grad_enabled()
        self.seen_output_hidden_states = output_hidden_states
        batch, sequence = input_ids.shape
        width = 5 if self.bad_width else 4
        states = tuple(
            torch.full((batch, sequence, width), float(layer))
            for layer in range(self.config.num_hidden_layers + 1)
        )
        return SimpleNamespace(hidden_states=states)


def test_capture_dataset_is_local_eval_no_grad_and_manifest_first(
    tmp_path, monkeypatch
) -> None:
    _require_feature_api()
    model = FakeTargetModel()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(snapshot)
            assert kwargs["local_files_only"] is True
            return model

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    output_dir = tmp_path / "features"
    examples = [
        {
            "input_ids": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "loss_mask": torch.tensor([0, 1, 1, 0], dtype=torch.float32),
        }
    ]

    manifest = capture_dataset(
        str(snapshot), examples, output_dir, [0, 2], 4, "cpu", torch.float32
    )

    assert manifest.feature_width == 8
    assert model.training is False
    assert model.seen_no_grad is True
    assert model.seen_output_hidden_states is True
    assert (output_dir / "manifest.json").is_file()
    stored = torch.load(
        output_dir / manifest.generation_dir / "feature_00000000.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(stored["hidden_states"][:, :4], torch.ones(4, 4))
    assert torch.equal(stored["hidden_states"][:, 4:], torch.full((4, 4), 3.0))
    assert stored["input_ids"].dtype == torch.long
    assert len(OfflineFeatureDataset(output_dir, manifest=manifest)) == 1


def test_capture_dataset_batches_and_trims_variable_length_features(tmp_path, monkeypatch) -> None:
    from Finetuning.adaptive_inference import AdaptiveInferenceSettings

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    class BatchedFakeTarget(FakeTargetModel):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, input_ids, *, attention_mask, output_hidden_states, use_cache=False):
            del attention_mask, use_cache
            self.batch_sizes.append(int(input_ids.shape[0]))
            self.seen_no_grad = not torch.is_grad_enabled()
            self.seen_output_hidden_states = output_hidden_states
            batch, sequence = input_ids.shape
            states = tuple(
                torch.full((batch, sequence, self.config.hidden_size), float(layer))
                for layer in range(self.config.num_hidden_layers + 1)
            )
            return SimpleNamespace(hidden_states=states)

    model = BatchedFakeTarget()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            return model

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    output_dir = tmp_path / "features"
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "loss_mask": torch.tensor([0, 1, 1], dtype=torch.float32),
        },
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "loss_mask": torch.tensor([0, 1, 1, 1, 1], dtype=torch.float32),
        },
    ]

    manifest = capture_dataset(
        str(snapshot),
        examples,
        output_dir,
        [0, 2],
        8,
        "cpu",
        torch.float32,
        adaptive_settings=AdaptiveInferenceSettings(
            enabled=True,
            min_batch_size=2,
            max_batch_size=2,
            bucket_window=8,
        ),
    )

    assert model.batch_sizes == [2]
    dataset = OfflineFeatureDataset(output_dir, manifest=manifest)
    records = [dataset[index] for index in range(len(dataset))]
    assert [record["input_ids"].shape[0] for record in records] == [3, 5]
    assert all(record["hidden_states"].shape[1] == 8 for record in records)


def test_capture_dataset_republishes_without_stale_features(tmp_path, monkeypatch) -> None:
    _require_feature_api()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = FakeTargetModel()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            return model

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    example = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "loss_mask": torch.tensor([0, 1, 1], dtype=torch.float32),
    }
    output_dir = tmp_path / "features"

    capture_dataset(str(snapshot), [example, example], output_dir, [0], 3, "cpu", torch.float32)
    capture_dataset(str(snapshot), [example], output_dir, [0], 3, "cpu", torch.float32)

    dataset = OfflineFeatureDataset(output_dir)
    assert len(dataset) == 1
    assert len(list(dataset.record_root.glob("feature_*.pt"))) == 1


def test_capture_dataset_rejects_fractional_loss_mask(tmp_path, monkeypatch) -> None:
    _require_feature_api()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = FakeTargetModel()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            return model

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    with pytest.raises(ValueError, match="binary"):
        capture_dataset(
            str(snapshot),
            [
                {
                    "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                    "loss_mask": torch.tensor([0.0, 0.5, 1.0]),
                }
            ],
            tmp_path / "features",
            [0],
            3,
            "cpu",
            torch.float32,
        )


def test_capture_dataset_fails_on_missing_snapshot_and_width_mismatch(
    tmp_path, monkeypatch
) -> None:
    _require_feature_api()
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "loss_mask": torch.tensor([0, 1, 1], dtype=torch.float32),
        }
    ]
    with pytest.raises(FileNotFoundError, match="local target model snapshot"):
        capture_dataset(
            str(tmp_path / "missing"), examples, tmp_path / "out", [0], 3, "cpu", torch.float32
        )

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = FakeTargetModel(bad_width=True)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            return model

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    with pytest.raises(ValueError, match="feature width"):
        capture_dataset(
            str(snapshot), examples, tmp_path / "out", [0], 3, "cpu", torch.float32
        )


def test_capture_cli_writes_prompt_provenance_from_a_lazy_jsonl_source(
    tmp_path, monkeypatch
) -> None:
    from Finetuning.capture_features import main

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = tmp_path / "teacher.jsonl"
    source.write_text(
        json.dumps(
            {"id": "vi-1", "document": "một hai", "summary": "tóm tắt đủ"}
        )
        + "\n",
        encoding="utf-8",
    )
    model = FakeTargetModel()

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = 0

        def __call__(self, text, **_kwargs):
            return {"input_ids": [20 + index for index, _ in enumerate(text.split())]}

        def apply_chat_template(self, messages, **_kwargs):
            ids = [1]
            for message in messages:
                ids.extend(self(message["content"])["input_ids"])
            return ids + [2, 3]

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: Tokenizer()),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: model),
    )
    output = tmp_path / "features"

    main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--target-model-path",
            str(snapshot),
            "--target-layer-ids",
            "0,2",
            "--max-length",
            "32",
            "--max-source-tokens",
            "16",
            "--max-summary-tokens",
            "8",
            "--device",
            "cpu",
            "--torch-dtype",
            "float32",
        ]
    )

    manifest = OfflineFeatureDataset(output).manifest
    assert manifest.tokenizer_id == str(snapshot)
    assert manifest.prompt_contract == {
        "chat_template": "qwen3",
        "max_source_tokens": 16,
        "max_summary_tokens": 8,
        "prompt_template": (
            "Hãy tóm tắt văn bản sau bằng tiếng Việt. "
            "Chỉ trả lời bằng bản tóm tắt:\n\n{document}"
        ),
    }


def test_capture_cli_can_resolve_target_layers_from_draft_layer_count(
    tmp_path, monkeypatch
) -> None:
    from Finetuning.capture_features import main

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = tmp_path / "teacher.jsonl"
    source.write_text(
        json.dumps(
            {"id": "vi-1", "document": "một hai", "summary": "tóm tắt đủ"}
        )
        + "\n",
        encoding="utf-8",
    )
    model = FakeTargetModel()

    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, **_kwargs):
            return {"input_ids": [20 + index for index, _ in enumerate(text.split())]}

        def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
            ids = [1]
            for message in messages:
                if message["role"] == "user":
                    ids.extend([4, *self(message["content"])["input_ids"], 5])
                else:
                    ids.extend([6, 7, *self(message["content"])["input_ids"]])
            return ids + ([6, 7] if add_generation_prompt else [99])

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: Tokenizer()),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: model),
    )
    output = tmp_path / "features"

    main(
        [
            "--input", str(source), "--output", str(output),
            "--target-model-path", str(snapshot),
            "--num-draft-layers", "2",
            "--max-length", "32", "--max-source-tokens", "16",
            "--max-summary-tokens", "8", "--device", "cpu",
            "--torch-dtype", "float32",
        ]
    )

    assert OfflineFeatureDataset(output).manifest.layer_ids == [1, 0]
