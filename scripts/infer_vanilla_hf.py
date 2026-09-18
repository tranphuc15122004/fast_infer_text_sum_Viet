#!/usr/bin/env python3
"""Vanilla Hugging Face inference using PyTorch eager attention."""

from __future__ import annotations

from common.vanilla_inference import build_parser as _build_parser, run


def build_parser():
    return _build_parser("eager", __doc__)


def main() -> int:
    return run(build_parser().parse_args(), method="vanilla_hf")


if __name__ == "__main__":
    raise SystemExit(main())
