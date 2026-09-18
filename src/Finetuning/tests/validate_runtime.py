"""Bounded synthetic runtime validator for the Finetuning pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import os

from Finetuning.config import load_run_config
from Finetuning.run_train import run_training
from Finetuning.tests.validate_static import validate_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    config = load_run_config(args.config)
    if args.device is not None:
        config.device = args.device
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    checkpoint = run_training(config)
    validate_run(config.resolved_output_dir)
    if not checkpoint.is_dir():
        raise ValueError("final checkpoint directory is missing")
    if os.environ.get("RANK", "0") == "0":
        print("runtime validation: PASS")


if __name__ == "__main__":
    main()
