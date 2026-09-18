#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config dflash || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

# Preserve the GSM8K entry point only when the master explicitly selects it.
# The unified master also contains the GSM8K defaults, so checking whether
# BACKEND/DATASET happen to be set would incorrectly hijack representative
# JSONL runs.
case "${DFLASH_MODE:-representative}" in
  representative) ;;
  gsm8k) exec bash "$ROOT/scripts/run_dflash_gsm8k.sh" "$@" ;;
  *)
    echo "DFLASH_MODE must be representative or gsm8k: ${DFLASH_MODE}" >&2
    exit 2
    ;;
esac

: "${TARGET_MODEL:?TARGET_MODEL is required}"
: "${DRAFT_MODEL:?DRAFT_MODEL is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

ARGS=(
  --target-model "$TARGET_MODEL"
  --draft-model "$DRAFT_MODEL"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --temperature "${TEMPERATURE:-0}"
  --output "$OUTPUT_FILE"
)
if [[ -n "${DATA_FILE:-}" ]]; then
  ARGS+=(--data-file "$DATA_FILE" --max-samples "${MAX_SAMPLES:-5}")
else
  ARGS+=(--prompt "${PROMPT:-The capital of France is}")
fi
[[ -n "${BLOCK_SIZE:-}" ]] && ARGS+=(--block-size "$BLOCK_SIZE")
[[ -n "${MAX_INPUT_TOKENS:-}" ]] && ARGS+=(--max-input-tokens "$MAX_INPUT_TOKENS")
[[ "${SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:$ROOT/externals/dflash${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_dflash.py" "${ARGS[@]}" "$@"
