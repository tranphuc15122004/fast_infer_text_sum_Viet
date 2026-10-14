#!/usr/bin/env python3
"""Thin entrypoint for the src/Benchmark native FA4 VietBench runner."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from Benchmark.fa4_server import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
