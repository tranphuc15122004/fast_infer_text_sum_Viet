"""JSONL output + unified result schema (baseline_repo_guide.md §13)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from Benchmark.common.paths import ROOT

# Base schema: every run record must carry at least these keys.
BASE_SCHEMA_KEYS = [
    "method",
    "dataset",
    "model",
    "input_tokens",
    "retained_tokens",
    "output_tokens",
    "batch_size",
    "selector_latency_ms",
    "server_startup_ms",
    "queue_wait_ms",
    "batch_wait_ms",
    "ttft_ms",
    "draft_latency_ms",
    "verification_latency_ms",
    "tpot_ms",
    "e2e_ms",
    "server_reported_e2e_ms",
    "throughput_tok_s",
    "qps",
    "peak_memory_gb",
]

# Speculative methods add these keys.
SPEC_SCHEMA_KEYS = [
    "avg_accept_length",
    "acceptance_rate",
    "draft_latency_ms",
    "verification_latency_ms",
    "rejected_draft_ratio",
    "verification_steps",
]


def _json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON values (tensors, numpy, Path, torch.Size)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # torch tensors / numpy scalars / torch.Size
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    return str(obj)


class JsonlWriter:
    """Append JSON lines to a file; finalize() writes the summary record."""

    def __init__(self, path: Path):
        # Resolve relative paths against the repo root so scripts work no
        # matter which directory the wrapper cd's into.
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def add(self, record: dict) -> None:
        self.records.append(record)
        safe = _json_safe(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")

    def finalize(self, summary: dict) -> dict:
        safe = _json_safe(summary)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        return summary


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def throughput(tokens: list[int], times_s: list[float]) -> float:
    """Inverse of mean per-generation TPOT (matches DFlash/EAGLE scripts)."""
    tpot = [
        t / n for n, t in zip(tokens, times_s) if n > 0 and t > 0
    ]
    return 1.0 / mean(tpot) if tpot else 0.0


def missing_keys(record: dict, keys: list[str]) -> list[str]:
    return [k for k in keys if k not in record]


def validate_schema(record: dict, spec: bool = False) -> list[str]:
    """Return a list of missing base schema keys (empty == valid)."""
    problems = missing_keys(record, BASE_SCHEMA_KEYS)
    if spec:
        problems += missing_keys(record, SPEC_SCHEMA_KEYS)
    return problems


def print_table(rows: list[tuple[str, Any]]) -> None:
    width = max(len(str(k)) for k, _ in rows)
    print("=" * 56)
    for k, v in rows:
        print(f"{k:<{width}} : {v}")
    print("=" * 56)
