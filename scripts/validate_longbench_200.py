#!/usr/bin/env python3
"""Validate a LongBench canonical output directory and its manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.benchmark_data import validate_output_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()
    summary = validate_output_dir(args.data_dir, expected_count=args.expected_count)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"VALID: {summary['total']} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
