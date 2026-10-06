"""Conservative output-integrity checks for benchmark records.

The guard is intentionally annotation-only: it identifies an unmistakably
degenerate repeated n-gram, but callers decide whether that should invalidate a
run.  It must never alter model inputs or generated tokens.
"""

from __future__ import annotations

from collections import Counter
import re


def is_degenerate_output(
    text: str | None,
    *,
    min_words: int = 32,
    ngram_size: int = 3,
    min_repeats: int = 8,
) -> bool:
    """Return ``True`` only for strong repeated-n-gram collapse signals."""

    words = re.findall(r"\S+", text or "")
    if len(words) < min_words or ngram_size <= 0:
        return False
    ngrams = Counter(
        tuple(words[index : index + ngram_size])
        for index in range(len(words) - ngram_size + 1)
    )
    return bool(ngrams and max(ngrams.values()) >= min_repeats)


def repetition_metrics(text: str | None, *, ngram_size: int = 3) -> dict[str, float | int | bool]:
    """Measure word and trigram repetition and flag long collapsed outputs."""

    words = [word.casefold() for word in re.findall(r"\S+", text or "")]
    ngrams = Counter(
        tuple(words[index : index + ngram_size])
        for index in range(max(0, len(words) - ngram_size + 1))
    ) if ngram_size > 0 else Counter()
    ngram_count = sum(ngrams.values())
    repeated_trigram_ratio = (
        1.0 - len(ngrams) / ngram_count if ngram_count else 0.0
    )
    degenerate = len(words) >= 32 and (
        is_degenerate_output(text, ngram_size=ngram_size)
        or repeated_trigram_ratio >= 0.25
    )
    return {
        "word_count": len(words),
        "unique_word_ratio": len(set(words)) / max(1, len(words)),
        "repeated_trigram_ratio": repeated_trigram_ratio,
        "repetition_flag": degenerate,
    }
