from __future__ import annotations

import json

import torch


def test_summary_quality_reports_rouge_and_exact_target_match() -> None:
    from Finetuning.generation_evaluation import summarize_generation_records

    result = summarize_generation_records(
        [
            {
                "prediction": "Việt Nam tăng trưởng mạnh",
                "reference_summary": "Việt Nam tăng trưởng mạnh",
                "target_token_match": True,
                "dflash_elapsed_s": 0.2,
                "target_elapsed_s": 0.4,
            },
            {
                "prediction": "ngắn",
                "reference_summary": "bản tóm tắt dài hơn",
                "target_token_match": False,
                "dflash_elapsed_s": 0.3,
                "target_elapsed_s": 0.6,
            },
        ]
    )

    assert result["num_samples"] == 2
    assert result["target_exact_rate"] == 0.5
    assert result["rouge1_f"] < 1.0
    assert result["rouge1_f"] > 0.0
    assert result["speedup"] == 2.0


def test_generation_evaluation_keeps_human_reference_from_teacher_jsonl(tmp_path) -> None:
    from Finetuning.generation_evaluation import evaluate_draft_generation

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
        def generate(self, input_ids, **_kwargs):
            return torch.cat(
                [input_ids, torch.tensor([[7, 8, 99]], dtype=torch.long)], dim=1
            )

    class Draft:
        def spec_generate(self, _target, input_ids, **_kwargs):
            output = torch.cat(
                [input_ids, torch.tensor([[7, 8, 99]], dtype=torch.long)], dim=1
            )
            return output, {"acceptance_lengths": [2]}

    source = tmp_path / "teacher.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "vi-1",
                "document": "văn bản dài",
                "summary": "teacher trajectory",
                "reference_summary": "gold summary",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.jsonl"

    result = evaluate_draft_generation(
        source,
        output,
        tokenizer=Tokenizer(),
        target=Target(),
        draft=Draft(),
        max_length=64,
        max_source_tokens=32,
        max_summary_tokens=8,
        chat_template="qwen3",
        prompt_template="{document}",
        device="cpu",
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["prediction"] == "7 8"
    assert rows[0]["reference_summary"] == "gold summary"
    assert rows[0]["target_token_match"] is True
    assert rows[0]["acceptance_lengths"] == [2]
    assert rows[-1]["type"] == "summary"
    assert result["target_exact_rate"] == 1.0
    assert result["mean_acceptance_length"] == 2.0
