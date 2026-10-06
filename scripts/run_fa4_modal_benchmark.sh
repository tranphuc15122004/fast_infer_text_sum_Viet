#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec modal run "$ROOT/scripts/modal_flashattn_pilot.py" "$@"
