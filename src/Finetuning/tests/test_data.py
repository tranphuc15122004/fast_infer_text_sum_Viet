from __future__ import annotations

from pathlib import Path

import pytest
import torch

try:
    from Finetuning.data import (
        SummaryRecord,
        build_summary_loss_mask,
        load_summary_jsonl,
        render_summary_prompt,
        render_summary_example,
    )
except ModuleNotFoundError as exc:  # Red phase: the adapter is not ported yet.
    _DATA_IMPORT_ERROR = exc


def _require_data_api() -> None:
    if "_DATA_IMPORT_ERROR" in globals():
        pytest.fail(f"summary data API is not implemented: {_DATA_IMPORT_ERROR}")


class FakeQwenTokenizer:
    """Small local tokenizer with Qwen-style user/assistant delimiters."""

    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    def _encode_text(self, text: str) -> list[int]:
        ids = []
        for token in text.split():
            if token not in self._vocabulary:
                self._vocabulary[token] = 200 + len(self._vocabulary)
            ids.append(self._vocabulary[token])
        return ids

    def __call__(self, text: str, *, add_special_tokens: bool = False, **_kwargs):
        del add_special_tokens
        return {"input_ids": self._encode_text(text)}

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        **_kwargs,
    ):
        assert tokenize is True
        assert return_dict is False
        ids = [100]
        for message in conversation:
            role = message["role"]
            if role == "user":
                ids.extend([1, *self._encode_text(message["content"]), 98])
            elif role == "assistant":
                ids.extend([2, 3, *self._encode_text(message["content"])])
            else:
                raise AssertionError(f"unexpected role: {role}")
        if add_generation_prompt:
            ids.extend([2, 3])
        elif conversation and conversation[-1]["role"] == "assistant":
            ids.append(self.eos_token_id)
        return ids


class MismatchQwenTokenizer(FakeQwenTokenizer):
    """Tokenizer whose standalone summary encoding differs from chat encoding."""

    all_special_ids = [97, 99]

    def __call__(self, text: str, *, add_special_tokens: bool = False, **_kwargs):
        del text, add_special_tokens
        return {"input_ids": [700, 701]}

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        **_kwargs,
    ):
        ids = super().apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
        )
        if not add_generation_prompt and conversation[-1]["role"] == "assistant":
            ids.insert(-1, 97)  # end-of-turn control token before EOS
        return ids

def test_load_summary_jsonl_preserves_unicode_and_metadata() -> None:
    _require_data_api()
    fixture = Path(__file__).parent / "fixtures" / "synthetic_summary.jsonl"

    records = load_summary_jsonl(fixture)

    assert len(records) >= 2
    assert all(isinstance(record, SummaryRecord) for record in records)
    assert any("Việt" in record.document or "Việt" in record.summary for record in records)
    assert records[0].metadata["split"] == "synthetic"


def test_summary_loss_mask_only_supervises_assistant_span() -> None:
    _require_data_api()

    mask = build_summary_loss_mask(
        torch.arange(8), assistant_start=5, assistant_end=8
    )

    assert mask.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]


def test_render_summary_example_is_qwen_compatible_and_causally_truncated() -> None:
    _require_data_api()
    tokenizer = FakeQwenTokenizer()
    record = SummaryRecord(
        id="unicode",
        document="Tài liệu rất dài",
        summary="Tóm tắt ngắn gọn",
    )

    example = render_summary_example(
        record,
        tokenizer,
        max_length=12,
        prompt_template="{document}",
    )

    assert set(("input_ids", "loss_mask")).issubset(example)
    assert example["input_ids"].ndim == 1
    assert example["input_ids"].shape == example["loss_mask"].shape
    assert int(example["loss_mask"].sum()) >= 2
    assert example["loss_mask"][-1].item() == 0
    assert example["input_ids"].dtype == torch.long


def test_render_summary_example_rejects_short_supervision_after_truncation() -> None:
    _require_data_api()
    record = SummaryRecord(id="short", document="Tài liệu", summary="Một")

    with pytest.raises(ValueError, match="two consecutive supervised tokens"):
        render_summary_example(record, FakeQwenTokenizer(), max_length=32)


def test_render_reserves_summary_budget_before_truncating_document() -> None:
    _require_data_api()
    tokenizer = FakeQwenTokenizer()
    record = SummaryRecord(
        id="long-source",
        document="một hai ba bốn năm sáu bảy tám",
        summary="tóm tắt đủ dài",
    )

    example = render_summary_example(
        record,
        tokenizer,
        max_length=12,
        max_source_tokens=2,
        max_summary_tokens=3,
        prompt_template="{document}",
    )

    assert int(example["loss_mask"].sum()) == 3
    assert example["loss_mask"][-1].item() == 0
    assert example["input_ids"].shape[0] <= 12


def test_generation_prompt_reserves_the_teacher_response_budget() -> None:
    _require_data_api()
    prompt = render_summary_prompt(
        SummaryRecord(
            id="prompt",
            document="một hai ba bốn năm sáu bảy tám",
            summary="reference không dùng để tạo prompt",
        ),
        FakeQwenTokenizer(),
        max_length=12,
        max_source_tokens=2,
        max_summary_tokens=3,
        prompt_template="{document}",
    )

    assert prompt.ndim == 1
    assert prompt.shape[0] <= 9


def test_render_fallback_does_not_supervise_control_token_suffix() -> None:
    _require_data_api()
    record = SummaryRecord(id="mismatch", document="Tài liệu", summary="Một bản")

    example = render_summary_example(record, MismatchQwenTokenizer(), max_length=32)

    control_positions = (example["input_ids"] == 97) | (example["input_ids"] == 99)
    assert example["loss_mask"][control_positions].sum().item() == 0
    assert example["loss_mask"].sum().item() == 2


def test_prepare_summary_iterator_streams_rendered_examples(tmp_path) -> None:
    from Finetuning.prepare_data import iter_summary_examples

    source = tmp_path / "teacher.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"id":"one","document":"một hai","summary":"tóm tắt đủ"}',
                '{"id":"two","document":"ba bốn","summary":"tóm tắt đủ"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples = iter_summary_examples(
        source,
        FakeQwenTokenizer(),
        max_length=32,
        max_samples=1,
    )

    assert not isinstance(examples, list)
    first = next(examples)
    assert first["id"] == "one"
    assert first["input_ids"].ndim == 1
    with pytest.raises(StopIteration):
        next(examples)


def test_default_prompt_explicitly_requests_a_vietnamese_summary() -> None:
    _require_data_api()

    class RecordingTokenizer(FakeQwenTokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.messages = []

        def apply_chat_template(self, conversation, **kwargs):
            self.messages.append(conversation)
            return super().apply_chat_template(conversation, **kwargs)

    tokenizer = RecordingTokenizer()
    render_summary_prompt(
        SummaryRecord(id="directive", document="nội dung", summary="tham chiếu"),
        tokenizer,
        max_length=64,
        max_source_tokens=32,
        max_summary_tokens=8,
    )

    assert tokenizer.messages[0][0]["content"].startswith("Hãy tóm tắt")
    assert tokenizer.messages[0][0]["content"].endswith("nội dung")
