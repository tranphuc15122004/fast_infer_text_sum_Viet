#!/usr/bin/env bash
# Shared Python runtime for every benchmark launcher.
#
# This file is meant to be sourced after ROOT has been defined. It can also
# derive ROOT from its own location when used by a standalone helper.

if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

fast_infer_resolve_python() {
  local candidate=""

  if [[ -n "${FAST_INFER_PYTHON:-}" ]]; then
    candidate="$FAST_INFER_PYTHON"
  elif [[ -n "${FAST_INFER_VENV:-}" ]]; then
    candidate="$FAST_INFER_VENV/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidate="$VIRTUAL_ENV/bin/python"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    candidate="$ROOT/.venv/bin/python"
  else
    # A production server may intentionally have no project venv. In that
    # case use the Python 3.12 command provided by the image/PATH.
    candidate="${FAST_INFER_SYSTEM_PYTHON:-python3}"
  fi

  # Production B200 images expose the shared interpreter as `python3` on PATH,
  # while local simulation commonly supplies an absolute .venv path. Resolve
  # command names once so every child process receives an executable path.
  if [[ "$candidate" != */* ]]; then
    candidate="$(command -v "$candidate" 2>/dev/null || true)"
  fi

  if [[ ! -x "$candidate" ]]; then
    echo "Shared Python interpreter not found or not executable: $candidate" >&2
    echo "Create it with: bash $ROOT/scripts/setup_venv.sh --offline" >&2
    return 1
  fi

  printf '%s\n' "$candidate"
}

fast_infer_require_python312() {
  local selected
  selected="$(fast_infer_resolve_python)" || return 1

  if ! "$selected" -c 'import sys; sys.exit(1) if sys.version_info[:2] != (3, 12) else None'; then
    echo "Shared runtime must use Python 3.12: $selected" >&2
    return 1
  fi

  export FAST_INFER_PYTHON="$selected"
}

fast_infer_prepare_cache_defaults() {
  # CUDA extensions such as FlashInfer/Triton write compile metadata during
  # import.  A mounted server home may be read-only (and local sandboxes
  # commonly make it so), therefore keep task caches in an explicit writable
  # location unless the operator already configured one.
  local cache_root="${FAST_INFER_CACHE_ROOT:-/tmp/fast_infer_cache}"
  export FAST_INFER_CACHE_ROOT="$cache_root"
  export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$cache_root/flashinfer}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$cache_root/triton}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$cache_root/torch_extensions}"
  mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
}

fast_infer_require_python312
fast_infer_prepare_cache_defaults
