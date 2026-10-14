#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_master
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh"

export FA4_EXECUTION_BACKEND=server
export FA4_DATA_DIR="${FA4_DATA_DIR:-$ROOT/datasets/eval_100}"
export FA4_OUTPUT_ROOT="${FA4_OUTPUT_ROOT:-$ROOT/outputs/fa4_native_benchmark}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${FI_GPU_IDS:-0}}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

if [[ "${SMOKE:-0}" == "1" ]]; then export LONG_BENCH_MODE=smoke; fi
if [[ "${FULL:-0}" == "1" ]]; then export LONG_BENCH_MODE=full; fi

cd "$ROOT"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/run_fa4_server.py" "$@"
