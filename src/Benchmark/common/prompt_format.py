"""Shared model-facing prompt formatting for chat-tuned target checkpoints."""

from __future__ import annotations


_CHAT_TEMPLATE_MARKERS = (
    "<|im_start|>",
    "<|start_header_id|>",
    "[INST]",
)


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Render a plain task prompt with the target tokenizer's chat template.

    Dataset builders keep prompts as plain text so their content is portable.
    Chat-tuned checkpoints such as Qwen3 require their tokenizer's message
    framing at inference time. Already-rendered prompts pass through unchanged.
    """

    prompt = str(prompt)
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    if any(marker in prompt for marker in _CHAT_TEMPLATE_MARKERS):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if not isinstance(formatted, str):
        raise TypeError(
            "tokenizer.apply_chat_template(tokenize=False) must return a string"
        )
    return formatted
