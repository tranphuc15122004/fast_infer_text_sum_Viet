#!/usr/bin/env python3
"""Run the shared native FA4 benchmark in the current server process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Benchmark.fa4_server import build_parser, resolve_server_models, runner_kwargs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        models = resolve_server_models(os.environ)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    os.environ["FA4_EXECUTION_BACKEND"] = "server"
    os.environ["FA4_OUTPUT_ROOT"] = str(
        (ROOT / args.output_dir).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir).resolve()
    )
    for method, env_name in (
        ("vanilla_hf", "MODEL_TARGET"),
        ("eagle3", "MODEL_EAGLE_DRAFT"),
        ("dflash", "MODEL_DFLASH_DRAFT"),
        ("domino", "MODEL_DOMINO_DRAFT"),
        ("dspark", "MODEL_DSPARK_DRAFT"),
    ):
        os.environ[env_name] = models[method]

    if os.environ.get("FI_OFFLINE", "1") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault(
        "HF_HOME",
        os.environ.get("FI_HF_HOME", str(Path.home() / ".cache/huggingface")),
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    sys.path.insert(0, str(ROOT / "scripts"))
    from modal_flashattn_pilot import run_flashattn_benchmark  # noqa: PLC0415

    result = run_flashattn_benchmark(**runner_kwargs(args))
    if args.preflight_only:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0

    run_dir = Path(result["remote_run_dir"])
    report_path = run_dir / "report_vi.md"
    if report_path.is_file():
        print(report_path.read_text(encoding="utf-8"))
    print(f"Kết quả trên server: {run_dir}")
    if result["summary"].get("run_passed") is not True:
        print(f"Benchmark gate chưa đạt; xem {report_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
