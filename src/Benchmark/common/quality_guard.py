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
