#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--config" ]]; then
  export FAST_INFER_MASTER_CONFIG="${2:?--config requires a path}"
  shift 2
elif [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi
source "$ROOT/scripts/common/config.sh" || exit 1
fast_infer_load_config longbench
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${LONG_BENCH_MODEL:?LONG_BENCH_MODEL is required}"
DATA_FILE="${LONG_BENCH_DATA_FILE:-${DATA_INPUT:-}}"
OUTPUT_FILE="${LONG_BENCH_OUTPUT_FILE:-$ROOT/outputs/longbench_viet_100/vanilla_fa.jsonl}"
: "${DATA_FILE:?LONG_BENCH_DATA_FILE or DATA_INPUT is required}"

ARGS=(--model "$LONG_BENCH_MODEL" --data-file "$DATA_FILE"
  --max-samples "${RUN_SAMPLES:-${LONG_BENCH_REPRESENTATIVE_SAMPLES:-20}}"
  --max-new-tokens "${RUN_MAX_NEW_TOKENS:-$LONG_BENCH_MAX_NEW_TOKENS}"
  --temperature "${RUN_TEMPERATURE:-$LONG_BENCH_TEMPERATURE}"
  --seed "$LONG_BENCH_SEED" --warmup-runs "$LONG_BENCH_WARMUP_RUNS"
  --max-input-tokens "$LONG_BENCH_MAX_INPUT_TOKENS"
  --device "$LONG_BENCH_DEVICE" --dtype "$LONG_BENCH_DTYPE"
  --output "$OUTPUT_FILE")
[[ "${SMOKE:-0}" == "1" || "${LONG_BENCH_MODE:-}" == "smoke" ]] && ARGS+=(--smoke)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" -m Benchmark.infer_vanilla_fa "${ARGS[@]}" "$@"
