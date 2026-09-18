#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config eagle3 || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${BASE_MODEL:?BASE_MODEL is required}"
: "${EAGLE_MODEL:?EAGLE_MODEL is required}"
: "${BENCH_NAME:?BENCH_NAME is required}"
: "${QUESTION_BEGIN:?QUESTION_BEGIN is required}"
: "${QUESTION_END:?QUESTION_END is required}"
: "${MAX_NEW_TOKENS:?MAX_NEW_TOKENS is required}"
: "${NUM_CHOICES:?NUM_CHOICES is required}"
: "${TEMPERATURE:?TEMPERATURE is required}"
: "${TOTAL_TOKEN:?TOTAL_TOKEN is required}"
: "${DEPTH:?DEPTH is required}"
: "${TOP_K:?TOP_K is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

if [[ ! -d "$BASE_MODEL" || ! -f "$BASE_MODEL/config.json" ]]; then
  echo "Qwen3 base model or config.json not found: $BASE_MODEL" >&2
  exit 1
fi

if [[ ! -d "$EAGLE_MODEL" || ! -f "$EAGLE_MODEL/config.json" ]]; then
  echo "EAGLE3 checkpoint or config.json not found: $EAGLE_MODEL" >&2
  exit 1
fi

if [[ ! -f "$EAGLE_MODEL/model.safetensors" && ! -f "$EAGLE_MODEL/pytorch_model.bin" ]]; then
  echo "EAGLE3 checkpoint has no model.safetensors or pytorch_model.bin: $EAGLE_MODEL" >&2
  exit 1
fi

QUESTION_FILE="$ROOT/externals/EAGLE/eagle/data/$BENCH_NAME/question.jsonl"
# Plug-and-play: set DATA_FILE in the env to point at your own jsonl.
# Records must use the EAGLE chat format: {"id": N, "turns": ["user prompt"]}.
if [[ -n "${DATA_FILE:-}" ]]; then
  QUESTION_FILE="$DATA_FILE"
fi
if [[ ! -f "$QUESTION_FILE" ]]; then
  echo "Question file not found: $QUESTION_FILE" >&2
  exit 1
fi

cd "$ROOT"

# Check dimensions before loading several GB of weights onto the GPU.
"$FAST_INFER_PYTHON" - "$BASE_MODEL/config.json" "$EAGLE_MODEL/config.json" <<'PY'
import json
import sys

base_path, eagle_path = sys.argv[1:]
with open(base_path) as f:
    base = json.load(f)
with open(eagle_path) as f:
    eagle = json.load(f)

checks = {
    "hidden_size": (base.get("hidden_size"), eagle.get("hidden_size")),
    "num_attention_heads": (base.get("num_attention_heads"), eagle.get("num_attention_heads")),
    "num_key_value_heads": (base.get("num_key_value_heads"), eagle.get("num_key_value_heads")),
    "head_dim": (base.get("head_dim"), eagle.get("head_dim")),
    "vocab_size": (base.get("vocab_size"), eagle.get("vocab_size")),
}

print("EAGLE3 compatibility check:")
for key, (left, right) in checks.items():
    print(f"  {key}: base={left}, eagle3={right}")

mismatches = [key for key, (left, right) in checks.items() if left != right]
if mismatches:
    raise SystemExit("Incompatible base/EAGLE3 config fields: " + ", ".join(mismatches))
PY

export PYTHONPATH="$ROOT/externals/EAGLE${PYTHONPATH:+:$PYTHONPATH}"

NAIVE_ARGS=()
if [[ "${SKIP_NAIVE:-0}" == "1" ]]; then
  NAIVE_ARGS+=(--skip-naive)
fi

SMOKE_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--smoke)
fi

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/eagle3_infer_qwen3.py" \
  --base-model "$BASE_MODEL" \
  --eagle-model "$EAGLE_MODEL" \
  --question-file "$QUESTION_FILE" \
  --question-begin "$QUESTION_BEGIN" \
  --question-end "$QUESTION_END" \
  --num-choices "$NUM_CHOICES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-0}" \
  --total-token "$TOTAL_TOKEN" \
  --depth "$DEPTH" \
  --top-k "$TOP_K" \
  --temperature "$TEMPERATURE" \
  "${NAIVE_ARGS[@]}" \
  "${SMOKE_ARGS[@]}" \
  --output "$OUTPUT_FILE" \
  "$@"
