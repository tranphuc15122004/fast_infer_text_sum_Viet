#!/usr/bin/env python3
"""Audit metric coverage in an existing LongBench run without inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.metric_audit import audit_output_file, format_audit_log


def audit_run(run_dir: Path, expected_output_tokens: int | None = None) -> int:
    """Audit canonical baseline/dataset JSONL files below ``run_dir``."""

    run_dir = Path(run_dir)
    count = 0
    combined_log = run_dir / "logs" / "metrics_audit.log"
    combined_log.parent.mkdir(parents=True, exist_ok=True)
    for output_path in sorted(run_dir.rglob("*.jsonl")):
        if output_path.parent.name in {"inputs", "logs", "shards", "attempts"}:
            continue
        if output_path.parent == run_dir:
            continue
        relative_parts = output_path.relative_to(run_dir).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if "attempts" in relative_parts[:-1]:
            continue
        baseline = output_path.parent.name
        dataset = output_path.stem
        audit_path = run_dir / "logs" / f"{baseline}_{dataset}.metrics.json"
        summary = audit_output_file(
            output_path,
            baseline=baseline,
            dataset=dataset,
            audit_path=audit_path,
            expected_output_tokens=expected_output_tokens,
        )
        line = format_audit_log(
            baseline=baseline,
            dataset=dataset,
            summary=summary,
            audit_path=audit_path,
        )
        with combined_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-output-tokens", type=int, default=None)
    args = parser.parse_args()
    count = audit_run(args.run_dir, args.expected_output_tokens)
    print(f"Audited {count} canonical JSONL file(s) under {args.run_dir}")


if __name__ == "__main__":
    main()
