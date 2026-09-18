"""Path / model-resolution helpers shared by every baseline runner.

The whole codebase uses the pattern:

    * the root project dir is ``ROOT`` (fast_infer_text_sum/)
    * vendored baselines live under ``ROOT/externals/<repo>``
    * models are expected in the local HuggingFace cache first; only when a
      snapshot is missing do scripts fall back to ``snapshot_download``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNALS = ROOT / "externals"
OUTPUTS = ROOT / "outputs"

# HF cache root (HF_HOME override respected).
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
HF_HUB = HF_HOME / "hub"
DATASET_CACHE_ROOT = HF_HOME / "datasets" / "fast_infer_text_sum"


def repo_dir(name: str) -> Path:
    """Return the vendored repo dir for ``name`` under externals/."""
    return EXTERNALS / name


def dataset_cache_dir(name: str) -> Path:
    """Return the shared cache directory for a benchmark dataset group."""
    if not name or Path(name).name != name:
        raise ValueError(f"dataset cache name must be a single directory: {name!r}")
    return DATASET_CACHE_ROOT / name


def hf_token() -> str | None:
    """Return the HF token from the environment (or None)."""
    return os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    ) or None


def snapshot_dir(repo_id: str, revision: str | None = None) -> Path | None:
    """Return the local snapshot dir for a HF repo, or None if not cached.

    Matches the ``models--owner--name/snapshots/<revision>/`` layout used by
    huggingface_hub. If the repo is not cached, returns None (the caller can
    then decide to download).
    """
    safe = repo_id.replace("/", "--")
    snap_root = HF_HUB / f"models--{safe}" / "snapshots"
    if not snap_root.is_dir():
        return None
    if revision:
        cand = snap_root / revision
        if cand.is_dir():
            return cand
        return None
    # No explicit revision: return the only snapshot, else the newest by mtime.
    snaps = sorted(
        (p for p in snap_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snaps[0] if snaps else None


def model_config(model_dir: Path | str) -> dict:
    """Read config.json of a local model dir."""
    with (Path(model_dir) / "config.json").open() as f:
        return json.load(f)


def require_file(path: Path, what: str = "") -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def gpu_name() -> str:
    """Return a short GPU name for records (best effort, no torch import)."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return os.environ.get("GPU_NAME", "unknown")
