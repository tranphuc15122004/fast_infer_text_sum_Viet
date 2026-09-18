"""Small, dependency-free correctness checks used by every smoke test.

Each baseline script collects a list of ``(bool, str)`` checks and calls
``report_checks(name, checks)`` which prints PASS/FAIL and returns the exit
code (1 if any check failed).
"""

from __future__ import annotations

import sys
from typing import Callable


def check_output_text(text: str, min_len: int = 1) -> tuple[bool, str]:
    ok = bool(text and len(text.strip()) >= min_len)
    return ok, (
        f"output text non-empty (len={len(text.strip())})"
        if ok
        else "output text empty or too short"
    )


def check_new_tokens(n: int) -> tuple[bool, str]:
    ok = n > 0
    return ok, f"generated tokens = {n} (> 0)" if ok else "generated 0 tokens"


def check_finite_logits(tensor) -> tuple[bool, str]:
    try:
        import torch

        ok = bool(torch.isfinite(tensor).all())
        return ok, "logits finite" if ok else "logits contain NaN/Inf"
    except Exception as e:  # pragma: no cover - torch unavailable
        return False, f"could not check logits: {e}"


def check_deterministic(fn: Callable[[], object], seed: int = 0) -> tuple[bool, str]:
    """Run fn twice with the same seed; pass if outputs are identical."""
    import torch

    torch.manual_seed(seed)
    a = fn()
    torch.manual_seed(seed)
    b = fn()
    ok = a == b
    return ok, "deterministic across two runs" if ok else "NON-deterministic output"


def check_retention(source_text: str, output_text: str, keywords: list[str]) -> tuple[bool, str]:
    """Pass if every keyword survives from source into output (approx. baselines)."""
    missing = [k for k in keywords if k not in output_text]
    ok = not missing
    return ok, (
        f"retained {len(keywords) - len(missing)}/{len(keywords)} keywords"
        if ok
        else f"missing keywords: {missing}"
    )


def report_checks(name: str, checks: list[tuple[bool, str]]) -> int:
    """Print checks and return process exit code."""
    print(f"\n[{name}] verification checks:")
    failed = 0
    for ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        failed += 0 if ok else 1
    print(f"[{name}] {'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    return 0 if failed == 0 else 1


def finish(name: str, checks: list[tuple[bool, str]]) -> None:
    code = report_checks(name, checks)
    sys.exit(code)
