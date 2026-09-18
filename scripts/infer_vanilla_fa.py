#!/usr/bin/env python3
"""Vanilla Hugging Face inference using flash-attention 2."""

from __future__ import annotations

from common.vanilla_inference import build_parser as _build_parser, run


def build_parser():
    return _build_parser("flash_attention_2", __doc__)


def main() -> int:
    return run(build_parser().parse_args(), method="vanilla_fa")


if __name__ == "__main__":
    raise SystemExit(main())
