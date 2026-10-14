"""CLI and environment setup for native FA4 benchmark runs on B200."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any


SERVER_MODEL_ENV = {
    "vanilla_hf": "MODEL_TARGET",
    "eagle3": "MODEL_EAGLE_DRAFT",
    "dflash": "MODEL_DFLASH_DRAFT",
    "domino": "MODEL_DOMINO_DRAFT",
    "dspark": "MODEL_DSPARK_DRAFT",
}
SUPPORTED_METHODS = tuple(SERVER_MODEL_ENV)


def resolve_server_models(
    environ: Mapping[str, str], methods: str = "all"
) -> dict[str, str]:
    """Resolve configured model paths for only the selected methods."""
    requested = [part for part in methods.replace(",", " ").split() if part]
    if requested == ["all"]:
        selected = SUPPORTED_METHODS
    else:
        unknown = sorted(set(requested) - set(SUPPORTED_METHODS))
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}")
        if len(requested) != len(set(requested)):
            raise ValueError("Duplicate method in --methods")
        if "vanilla_hf" not in requested or "dflash" not in requested:
            raise ValueError("FA4 comparisons require both vanilla_hf and dflash")
        selected = tuple(method for method in SUPPORTED_METHODS if method in requested)

    missing = [
        SERVER_MODEL_ENV[method]
        for method in selected
        if not str(environ.get(SERVER_MODEL_ENV[method], "")).strip()
    ]
    if missing:
        raise ValueError(
            "Missing model paths from the _Viet master config: " + ", ".join(missing)
        )
    return {
        method: str(environ[SERVER_MODEL_ENV[method]]).strip()
        for method in selected
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Chạy benchmark tiếng Việt batch-1 bằng Transformers native + "
            "FlashAttention-4 (FA4) trực tiếp trên GPU server."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "representative", "full"),
        default=os.environ.get("LONG_BENCH_MODE", "smoke"),
    )
    parser.add_argument("--smoke", dest="mode", action="store_const", const="smoke")
    parser.add_argument(
        "--representative", dest="mode", action="store_const", const="representative"
    )
    parser.add_argument("--full", dest="mode", action="store_const", const="full")
    parser.add_argument(
        "--datasets", default=os.environ.get("LONG_BENCH_DATASETS", "all")
    )
    parser.add_argument("--samples-per-dataset", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--methods", default=os.environ.get("FA4_METHODS", "all"))
    parser.add_argument("--warmup-tokens", type=int, default=512)
    parser.add_argument("--eagle-total-token", type=int, default=17)
    parser.add_argument("--eagle-depth", type=int, default=16)
    parser.add_argument("--eagle-top-k", type=int, default=1)
    parser.add_argument(
        "--domino-cuda-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dspark-confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--phase-timing-mode",
        choices=("separate", "inline", "off"),
        default="separate",
        help="separate đo pha ở lượt riêng ngoài E2E; inline để đối chiếu phiên bản cũ",
    )
    parser.add_argument("--strict-greedy-parity", action="store_true")
    parser.add_argument("--require-speedup", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-retries", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--direct-target-audit", action="store_true")
    parser.add_argument("--verifier-audit", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--debug-cuda-launch-blocking", action="store_true")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("FA4_DATA_DIR", "datasets/eval_100"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("FA4_OUTPUT_ROOT", "outputs/fa4_native_benchmark"),
    )
    return parser


def runner_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "datasets": args.datasets,
        "samples_per_dataset": args.samples_per_dataset,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "methods": args.methods,
        "warmup_tokens": args.warmup_tokens,
        "eagle_total_token": args.eagle_total_token,
        "eagle_depth": args.eagle_depth,
        "eagle_top_k": args.eagle_top_k,
        "domino_cuda_graph": args.domino_cuda_graph,
        "dspark_confidence_threshold": args.dspark_confidence_threshold,
        "phase_timing_mode": args.phase_timing_mode,
        "strict_greedy_parity": args.strict_greedy_parity,
        "require_speedup": args.require_speedup,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "sample_retries": args.sample_retries,
        "checkpoint_interval": args.checkpoint_interval,
        "run_id": args.run_id,
        "resume": args.resume,
        "direct_target_audit": args.direct_target_audit,
        "verifier_audit": args.verifier_audit,
        "preflight_only": args.preflight_only,
        "debug_cuda_launch_blocking": args.debug_cuda_launch_blocking,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.methods = ",".join(args.methods.replace(",", " ").split())
    try:
        models = resolve_server_models(os.environ, args.methods)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "src"))
    env_keys = {
        "FA4_EXECUTION_BACKEND",
        "FA4_DATA_DIR",
        "FA4_OUTPUT_ROOT",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "PYTHONUNBUFFERED",
        *SERVER_MODEL_ENV.values(),
    }
    previous_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["FA4_EXECUTION_BACKEND"] = "server"
        os.environ["FA4_DATA_DIR"] = str(data_dir.resolve())
        os.environ["FA4_OUTPUT_ROOT"] = str(output_dir.resolve())
        for method, model_path in models.items():
            os.environ[SERVER_MODEL_ENV[method]] = model_path
        if os.environ.get("FI_OFFLINE", "1") == "1":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PYTHONUNBUFFERED", "1")

        native_runner = importlib.import_module("Benchmark.native_fa4_benchmark")
        if not getattr(native_runner, "SERVER_MODE", True):
            native_runner = importlib.reload(native_runner)
        result = native_runner.run_flashattn_benchmark(**runner_kwargs(args))
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if args.preflight_only:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0
    run_dir = Path(result["remote_run_dir"])
    report_path = run_dir / "report_vi.md"
    if report_path.is_file():
        print(report_path.read_text(encoding="utf-8"))
    print(f"Kết quả: {run_dir}")
    if result["summary"].get("run_passed") is not True:
        print(f"Benchmark gate chưa đạt; xem {report_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
