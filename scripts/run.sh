#!/usr/bin/env bash
# Dispatcher: maps a baseline name to its launcher. Every launcher loads the
# same external master through config/master.path.
#
# Usage:
#   bash scripts/run.sh <baseline> [extra args...]
#
# Each baseline has a wrapper `scripts/run_<baseline>.sh` that invokes the
# shared Python 3.12 interpreter after loading the master config.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${1:?usage: run.sh <baseline> [args...]}"
shift || true

case "$BASELINE" in
  longbench_200) WRAPPER="scripts/run_longbench_200.sh" ;;
  vanilla_hf)   WRAPPER="scripts/run_vanilla_hf.sh" ;;
  vanilla_fa)   WRAPPER="scripts/run_vanilla_fa.sh" ;;
  eagle3)      WRAPPER="scripts/run_eagle3.sh" ;;
  dflash)      WRAPPER="scripts/run_dflash.sh" ;;
  domino)      WRAPPER="scripts/run_domino.sh" ;;
  dspark)      WRAPPER="scripts/run_dspark.sh" ;;
  *)
    echo "Unknown baseline: $BASELINE" >&2
    echo "Available: longbench_200 vanilla_hf vanilla_fa eagle3 dflash domino dspark" >&2
    exit 1
    ;;
esac

exec "$ROOT/$WRAPPER" "$@"
