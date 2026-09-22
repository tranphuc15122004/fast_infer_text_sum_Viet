#!/usr/bin/env python3
"""Compatibility launcher for the Benchmark data package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from Benchmark.data.analyze_distribution import main


if __name__ == "__main__":
    raise SystemExit(main())
