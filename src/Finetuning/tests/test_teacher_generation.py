from __future__ import annotations

import json

import torch


def test_teacher_generation_writes_target_trajectory_and_keeps_reference(tmp_path) -> None:
    from Finetuning.generate_targets import generate_teacher_jsonl

    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, **_kwargs):
            return {"input_ids": [20 + index for index, _ in enumerate(text.split())]}

        def apply_chat_template(self, messages, **_kwargs):
            ids = [1]
            for message in messages:
                ids.extend(self(message["content"])["input_ids"])
            return ids + [2]

        def decode(self, ids, **_kwargs):
            return " ".join(str(value) for value in ids if value != self.eos_token_id)

    class Target:
        def generate(self, input_ids, **_kwargs):
            return torch.cat(
                [input_ids, torch.tensor([[7, 8, 99]], dtype=torch.long)], dim=1
            )

    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "vi-1", "document": "văn bản dài", "summary": "gold"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "teacher.jsonl"

    stats = generate_teacher_jsonl(
        source,
        output,
        tokenizer=Tokenizer(),
        target=Target(),
        target_model_path="/models/qwen3",
        max_length=32,
        max_source_tokens=16,
        max_summary_tokens=8,
        chat_template="qwen3",
        device="cpu",
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert stats == {"written": 1, "rejected": 0}
    assert row["summary"] == "7 8"
    assert row["reference_summary"] == "gold"
    assert row["teacher"]["target_model_path"] == "/models/qwen3"
    assert row["teacher"]["do_sample"] is False


def test_teacher_generation_batches_padded_prompts_and_preserves_rows(tmp_path) -> None:
    from Finetuning.adaptive_inference import AdaptiveInferenceSettings
    from Finetuning.generate_targets import generate_teacher_jsonl

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = 0

        def __call__(self, text, **_kwargs):
            return {"input_ids": [20 + index for index, _ in enumerate(text.split())]}

        def apply_chat_template(self, messages, **_kwargs):
            ids = [1]
            for message in messages:
                ids.extend(self(message["content"])["input_ids"])
            return ids + [2]

        def decode(self, ids, **_kwargs):
            return " ".join(str(value) for value in ids if value != self.eos_token_id)

    class Target:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def generate(self, input_ids, **_kwargs):
            self.batch_sizes.append(int(input_ids.shape[0]))
            batch = input_ids.shape[0]
            continuation = torch.tensor(
                [[7 + row, 8 + row, 99] for row in range(batch)],
                dtype=torch.long,
            )
            return torch.cat([input_ids, continuation], dim=1)

    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "vi-1", "document": "ngắn", "summary": "gold-1"})
        + "\n"
        + json.dumps(
            {"id": "vi-2", "document": "văn bản dài hơn", "summary": "gold-2"}
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "teacher.jsonl"
    target = Target()

    generate_teacher_jsonl(
        source,
        output,
        tokenizer=Tokenizer(),
        target=target,
        target_model_path="/models/qwen3",
        max_length=32,
        max_source_tokens=16,
        max_summary_tokens=8,
        chat_template="qwen3",
        device="cpu",
        adaptive_settings=AdaptiveInferenceSettings(
            enabled=True,
            min_batch_size=2,
            max_batch_size=2,
            bucket_window=8,
        ),
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert target.batch_sizes == [2]
    assert [row["id"] for row in rows] == ["vi-1", "vi-2"]
    assert [row["summary"] for row in rows] == ["7 8", "8 9"]
