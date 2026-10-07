#!/usr/bin/env python3
"""Entrypoint mỏng cho báo cáo chẩn đoán FA4 từ artifact offline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Benchmark.fa4_diagnostics import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
