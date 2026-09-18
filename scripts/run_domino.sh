#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then export FAST_INFER_MASTER_CONFIG="$1"; shift; fi
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config domino
source "$ROOT/scripts/common/runtime.sh"
: "${MODEL:?MODEL is required}"
: "${DRAFT_MODEL:?DRAFT_MODEL is required}"
: "${DATA_FILE:?DATA_FILE is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_domino.py" \
  --model "$MODEL" --draft-model "$DRAFT_MODEL" \
  --data-file "$DATA_FILE" --output "$OUTPUT_FILE" \
  --max-samples "${MAX_SAMPLES:-1}" --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
  --batch-size "${BATCH_SIZE:-auto}" --tp-size "${TP_SIZE:-1}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.9}" "$@"
