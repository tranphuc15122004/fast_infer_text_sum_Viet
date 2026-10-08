"""CLI and environment helpers for native FA4 runs on the B200 server."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any


SERVER_MODEL_ENV = {
    "vanilla_hf": "MODEL_TARGET",
    "eagle3": "MODEL_EAGLE_DRAFT",
    "dflash": "MODEL_DFLASH_DRAFT",
    "domino": "MODEL_DOMINO_DRAFT",
    "dspark": "MODEL_DSPARK_DRAFT",
}


def resolve_server_models(environ: Mapping[str, str]) -> dict[str, str]:
    """Resolve every checkpoint from the canonical master config environment."""

    missing = [key for key in SERVER_MODEL_ENV.values() if not str(environ.get(key, "")).strip()]
    if missing:
        raise ValueError(
            "Missing server model paths from the _Viet master config: "
            + ", ".join(missing)
        )
    return {
        method: str(environ[key]).strip()
        for method, key in SERVER_MODEL_ENV.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Chạy benchmark batch-1 native Transformers + FlashAttention-4 "
            "trực tiếp trên GPU B200 của server."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "representative", "full"), default="smoke")
    parser.add_argument(
        "--datasets", default="all", help="all hoặc danh sách phân tách bằng dấu phẩy"
    )
    parser.add_argument("--samples-per-dataset", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument(
        "--methods", default="all", help="all hoặc danh sách method phân tách bằng dấu phẩy"
    )
    parser.add_argument("--warmup-tokens", type=int, default=512)
    parser.add_argument("--eagle-total-token", type=int, default=17)
    parser.add_argument("--eagle-depth", type=int, default=16)
    parser.add_argument("--eagle-top-k", type=int, default=1)
    parser.add_argument("--domino-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dspark-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--phase-timing-mode", choices=("separate", "inline", "off"), default="separate",
                        help="separate đo pha ở lượt riêng ngoài E2E; inline để đối chiếu phiên bản cũ")
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
    parser.add_argument("--output-dir", default="outputs/fa4_native_benchmark")
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
    }
