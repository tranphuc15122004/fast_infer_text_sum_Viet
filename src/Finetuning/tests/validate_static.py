"""Static artifact checks for a completed Finetuning run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import math


def validate_run(output_dir: str | Path) -> None:
    root = Path(output_dir)
    required = ("metrics.jsonl", "train.log")
    missing = [name for name in required if not (root / name).is_file()]
    checkpoints = list(root.glob("*-step*/COMPLETE"))
    if missing or not checkpoints:
        raise ValueError(f"missing training artifacts: files={missing}, checkpoints={checkpoints}")
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    steps = [record for record in records if record.get("type") == "step"]
    if not steps:
        raise ValueError("metrics.jsonl contains no optimizer step records")
    for record in steps:
        for key in ("step", "loss", "grad_norm", "lr", "step_time_s", "tokens_per_s", "mfu"):
            if key not in record:
                raise ValueError(f"step record missing metric {key!r}")
            if key != "mfu" and not math.isfinite(float(record[key])):
                raise ValueError(f"non-finite metric {key!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    validate_run(args.output_dir)
    print("static validation: PASS")


if __name__ == "__main__":
    main()
