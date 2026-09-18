from __future__ import annotations

import json

import torch

from Finetuning.capture_features import merge_feature_shards
from Finetuning.features import FeatureManifest, OfflineFeatureDataset
from Finetuning.distributed import merge_ranked_jsonl


def test_merge_ranked_jsonl_restores_source_order_and_removes_internal_index(tmp_path) -> None:
    shard_zero = tmp_path / "teacher.jsonl.rank00000"
    shard_one = tmp_path / "teacher.jsonl.rank00001"
    shard_zero.write_text(
        json.dumps({"_source_index": 0, "id": "a"})
        + "\n"
        + json.dumps({"_source_index": 2, "id": "c"})
        + "\n",
        encoding="utf-8",
    )
    shard_one.write_text(
        json.dumps({"_source_index": 1, "id": "b"}) + "\n",
        encoding="utf-8",
    )

    destination = tmp_path / "teacher.jsonl"
    merge_ranked_jsonl([shard_zero, shard_one], destination)

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["a", "b", "c"]
    assert all("_source_index" not in row for row in rows)


def test_merge_feature_shards_publishes_one_readable_manifest(tmp_path) -> None:
    shards = []
    for rank, source_index in enumerate((0, 1)):
        shard = tmp_path / f"features.rank{rank:05d}"
        generation = shard / ".generations" / f"generation-{rank}"
        generation.mkdir(parents=True)
        manifest = FeatureManifest(
            model_id="/models/qwen3",
            tokenizer_id="/models/qwen3",
            layer_ids=[0],
            hidden_size=2,
            max_length=4,
            hidden_states_dtype="torch.float32",
            generation_dir=generation.relative_to(shard).as_posix(),
        )
        (shard / "manifest.json").write_text(
            json.dumps(manifest.to_dict()), encoding="utf-8"
        )
        torch.save(
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "loss_mask": torch.tensor([0.0, 1.0, 1.0, 0.0]),
                "hidden_states": torch.zeros((4, 2)),
            },
            generation / f"feature_{source_index:08d}.pt",
        )
        shards.append(shard)

    destination = tmp_path / "features"
    merged = merge_feature_shards(shards, destination)

    assert merged.generation_dir is not None
    dataset = OfflineFeatureDataset(destination)
    assert len(dataset) == 2
