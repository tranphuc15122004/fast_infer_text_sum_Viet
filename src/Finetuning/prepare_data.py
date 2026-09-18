"""Prepare local document/summary JSONL for offline feature capture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import iter_summary_jsonl, render_summary_example


def iter_summary_examples(
    path: str | Path,
    tokenizer: Any,
    *,
    max_length: int,
    chat_template: str = "qwen3",
    max_samples: int | None = None,
    max_source_tokens: int | None = None,
    max_summary_tokens: int | None = None,
    prompt_template: str | None = None,
):
    """Render JSONL records lazily for capture without materializing the corpus.

    Malformed or too-short target trajectories are intentionally skipped.  The
    capture command rejects an entirely unusable input through
    ``capture_dataset``; this lets one bad document avoid invalidating a long
    offline batch while retaining deterministic input order for the rest.
    """

    for record in iter_summary_jsonl(path, max_samples=max_samples):
        try:
            kwargs = (
                {"prompt_template": prompt_template}
                if prompt_template is not None
                else {}
            )
            rendered = render_summary_example(
                record,
                tokenizer,
                max_length=max_length,
                chat_template=chat_template,
                max_source_tokens=max_source_tokens,
                max_summary_tokens=max_summary_tokens,
                **kwargs,
            )
        except ValueError:
            continue
        yield {**rendered, "id": record.id}


def prepare_summary_examples(
    path: str | Path,
    tokenizer: Any,
    *,
    max_length: int,
    chat_template: str = "qwen3",
    max_samples: int | None = None,
    max_source_tokens: int | None = None,
    max_summary_tokens: int | None = None,
    prompt_template: str | None = None,
) -> list[dict[str, Any]]:
    examples = list(
        iter_summary_examples(
            path,
            tokenizer,
            max_length=max_length,
            chat_template=chat_template,
            max_samples=max_samples,
            max_source_tokens=max_source_tokens,
            max_summary_tokens=max_summary_tokens,
            prompt_template=prompt_template,
        )
    )
    if not examples:
        raise ValueError("no trainable summary examples after rendering")
    return examples


__all__ = ["iter_summary_examples", "prepare_summary_examples"]
