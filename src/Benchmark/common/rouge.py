"""Dependency-free ROUGE-1/2/L metrics for the benchmark harness.

Reference implementation (thuật toán + interface) lấy từ
``PoTR_article_summary/external/HeterSumGraph/tools/utils.py``:

    from rouge import Rouge
    def rouge_all(hyps, refer):
        score = rouge.get_scores(hyps, refer)[0]   # {"rouge-1"/"rouge-2"/"rouge-l": {p, r, f}}
        return score

Repo này dùng một venv chung; ROUGE được triển khai pure-Python nên không cần
thêm package ``rouge`` vào dependency manifest. Module này không phụ thuộc gì,
có cùng interface
``rouge_all(hyp, ref)`` và bổ sung key phẳng theo schema §13
(``rouge1/rouge2/rougeL``) để ghi trực tiếp vào record output.

Định nghĩa ROUGE (chuẩn Lin 2004):
  * ROUGE-N: overlap n-gram / (hyp n-gram cho precision, ref n-gram cho recall)
  * ROUGE-L: longest common subsequence (LCS) giữa token sequences
  * F1      = 2 * P * R / (P + R), trả 0.0 khi không có overlap/empty input

Tokenization giữ nguyên hành vi của package ``rouge`` (lowercase + split
whitespace) để kết quả so sánh được với PoTR; thêm NFKC normalize cho tiếng
Việt (dấu Unicode).
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence


def _normalise(text: str) -> str:
    """NFKC normalize + collapse whitespace (giữ dấu tiếng Việt)."""
    return " ".join(unicodedata.normalize("NFKC", text or "").split())


def tokenize(text: str) -> List[str]:
    """Token đơn giản: lowercase + split whitespace (khớp package ``rouge``)."""
    return _normalise(text).lower().split()


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Counter các n-gram (instance-level, trùng lặp được tính nhiều lần)."""
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(
        tuple(tokens[i : i + n])
        for i in range(len(tokens) - n + 1)
    )


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def rouge_n(hyp: str, ref: str, n: int = 2) -> Dict[str, float]:
    """ROUGE-N precision/recall/F1 giữa hypothesis và reference."""
    h_tokens = tokenize(hyp)
    r_tokens = tokenize(ref)

    if not h_tokens or not r_tokens:
        return {"p": 0.0, "r": 0.0, "f": 0.0}

    h_ngrams = _ngrams(h_tokens, n)
    r_ngrams = _ngrams(r_tokens, n)

    # A one-token hypothesis has no ROUGE-2 (or higher) n-grams. Treat this
    # as zero overlap instead of dividing by an empty n-gram collection.
    hypothesis_count = sum(h_ngrams.values())
    reference_count = sum(r_ngrams.values())
    if hypothesis_count == 0 or reference_count == 0:
        return {"p": 0.0, "r": 0.0, "f": 0.0}

    overlap = sum((h_ngrams & r_ngrams).values())
    precision = overlap / hypothesis_count
    recall = overlap / reference_count

    return {"p": precision, "r": recall, "f": _f1(precision, recall)}


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Độ dài Longest Common Subsequence (DP O(n*m))."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def rouge_l(hyp: str, ref: str) -> Dict[str, float]:
    """ROUGE-L precision/recall/F1 dựa trên LCS."""
    h_tokens = tokenize(hyp)
    r_tokens = tokenize(ref)

    if not h_tokens or not r_tokens:
        return {"p": 0.0, "r": 0.0, "f": 0.0}

    lcs = _lcs_length(h_tokens, r_tokens)
    precision = lcs / len(h_tokens)
    recall = lcs / len(r_tokens)

    return {"p": precision, "r": recall, "f": _f1(precision, recall)}


def rouge_all(hyp: str, ref: str) -> Dict[str, Dict[str, float]]:
    """Interface tương đương ``rouge_all`` của PoTR/HeterSumGraph.

    Trả về::

        {
            "rouge-1": {"p": .., "r": .., "f": ..},
            "rouge-2": {"p": .., "r": .., "f": ..},
            "rouge-l": {"p": .., "r": .., "f": ..},
        }
    """
    return {
        "rouge-1": rouge_n(hyp, ref, 1),
        "rouge-2": rouge_n(hyp, ref, 2),
        "rouge-l": rouge_l(hyp, ref),
    }


def rouge_scores(hyp: str, ref: str) -> Dict[str, float]:
    """Flat F1 theo schema §13: ``{"rouge1": f, "rouge2": f, "rougeL": f}``."""
    all_scores = rouge_all(hyp, ref)
    return {
        "rouge1": round(all_scores["rouge-1"]["f"], 4),
        "rouge2": round(all_scores["rouge-2"]["f"], 4),
        "rougeL": round(all_scores["rouge-l"]["f"], 4),
    }


def add_rouge(
    record: dict,
    hyp: str,
    ref: Optional[str],
    *,
    prefix: str = "",
) -> Optional[Dict[str, float]]:
    """Ghi ``rouge1/rouge2/rougeL`` vào record nếu có reference.

    ``prefix`` cho phép lưu nhiều text khác nhau trong cùng record (vd
    GemFilter lưu cả ``base_text`` và ``gemfilter_text`` → prefix "base_").
    Trả về dict scores đã ghi, hoặc None nếu không có reference.
    """
    if not ref or not str(ref).strip():
        return None

    scores = rouge_scores(hyp, ref)
    for key, value in scores.items():
        record[prefix + key] = value
    return scores


def aggregate_rouge(
    records: Sequence[Mapping],
    *,
    prefix: str = "",
) -> Dict[str, float]:
    """Trung bình ROUGE-1/2/L trên các record đã có giá trị.

    Dùng cho record ``summary`` cuối file JSONL. Key trả về giữ prefix
    (vd ``base_rouge1``).
    """
    out: Dict[str, float] = {}
    for base_key in ("rouge1", "rouge2", "rougeL"):
        key = prefix + base_key
        values = [float(r[key]) for r in records if r.get(key) is not None]
        if values:
            out[key] = round(sum(values) / len(values), 4)
    return out


def format_scores(scores: Dict[str, Dict[str, float]]) -> str:
    """In đẹp kết quả ``rouge_all`` (cho log/verify)."""
    parts = []
    for name in ("rouge-1", "rouge-2", "rouge-l"):
        s = scores.get(name, {})
        parts.append(f"{name}: P={s.get('p', 0.0):.3f} R={s.get('r', 0.0):.3f} F={s.get('f', 0.0):.3f}")
    return " | ".join(parts)
