#!/usr/bin/env python3
"""Compatibility launcher for the Benchmark package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Benchmark.infer_dspark import main


if __name__ == "__main__":
    raise SystemExit(main())
