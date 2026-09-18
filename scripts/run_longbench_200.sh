#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--config" ]]; then
  [[ $# -ge 2 ]] || { echo "--config requires a master env path" >&2; exit 2; }
  export FAST_INFER_MASTER_CONFIG="$2"
  shift 2
elif [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

source "$ROOT/scripts/common/config.sh"
fast_infer_load_config longbench
source "$ROOT/scripts/common/runtime.sh"

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/run_longbench_200.py" "$@"
