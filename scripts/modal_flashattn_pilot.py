#!/usr/bin/env python3
"""Compatibility entrypoint for the native FA4 benchmark module."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import Benchmark.native_fa4_benchmark as _runner  # noqa: E402

# A compatibility caller may have imported the implementation in non-server
# mode earlier in this process. Re-evaluate its backend only after the server
# launcher sets FA4_EXECUTION_BACKEND.
if os.environ.get("FA4_EXECUTION_BACKEND") == "server" and not _runner.SERVER_MODE:
    _runner = importlib.reload(_runner)

SERVER_MODE = _runner.SERVER_MODE
USE_MODAL = _runner.USE_MODAL
app = _runner.app
main = _runner.main
run_flashattn_benchmark = _runner.run_flashattn_benchmark
_call_method = _runner._call_method

__all__ = [
    "SERVER_MODE",
    "USE_MODAL",
    "app",
    "main",
    "run_flashattn_benchmark",
    "_call_method",
]
