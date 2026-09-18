#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${FINETUNING_PYTHON:-python3}"

exec "$PYTHON_BIN" "$ROOT/scripts/run_finetuning_b200.py" "$@"
