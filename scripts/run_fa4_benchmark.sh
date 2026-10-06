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
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

cd "$ROOT"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/run_fa4_server.py" "$@"
