from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class DummyTokenizer:
    chat_template = "test-template"

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "Hãy tóm tắt."}]
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        return "<|im_start|>user\nHãy tóm tắt.<|im_end|>\n<|im_start|>assistant\n"


def test_format_chat_prompt_uses_tokenizer_template() -> None:
    from Benchmark.common.prompt_format import format_chat_prompt

    result = format_chat_prompt(DummyTokenizer(), "Hãy tóm tắt.")

    assert result == "<|im_start|>user\nHãy tóm tắt.<|im_end|>\n<|im_start|>assistant\n"


def test_format_chat_prompt_does_not_wrap_missing_or_preformatted_templates() -> None:
    from Benchmark.common.prompt_format import format_chat_prompt

    class NoTemplate:
        chat_template = None

    prompt = "Hãy tóm tắt."
    assert format_chat_prompt(NoTemplate(), prompt) == prompt
    assert format_chat_prompt(DummyTokenizer(), "<|im_start|>user\nx") == "<|im_start|>user\nx"
