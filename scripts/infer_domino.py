#!/usr/bin/env python3
"""Thin baseline entry point for the official SGLang Domino/DFLASH path."""

from __future__ import annotations

import sys

from infer_sglang_spec import main as _main


def main() -> int:
    sys.argv[1:1] = ["--method", "domino"]
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
