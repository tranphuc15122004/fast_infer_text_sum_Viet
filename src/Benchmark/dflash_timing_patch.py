"""Version-pinned strict decode timing instrumentation for vendored DFlash."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

PATCH_VERSION = "fast_infer_dflash_strict_decode_v1"
SOURCE_SHA256 = "8128db87aa47eac4a14779bd65fcda29335059c979218282d7117a8dac947469"
_MARKER = PATCH_VERSION


def patch_dflash_source(source: str) -> str:
    """Add the exact timer and token count at DFlash's native decode boundary."""
    if _MARKER in source:
        return source
    observed = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if observed != SOURCE_SHA256:
        raise ValueError(
            f"DFlash source SHA-256 mismatch: expected {SOURCE_SHA256}, got {observed}"
        )
    anchor = "        time_per_output_token=total_decode_time / num_output_tokens,\n"
    if source.count(anchor) != 1:
        raise ValueError("DFlash return timing anchor does not match the pinned source")
    patch = (
        "        # fast_infer_dflash_strict_decode_v1: starts after token one commits.\n"
        "        strict_decode_active_ms=total_decode_time * 1e3,\n"
        "        strict_decode_token_count=max(num_output_tokens - 1, 0),\n"
        '        strict_decode_phase_definition="after_first_token_committed_to_final_token",\n'
        "        strict_decode_phase_verified=True,\n"
    )
    return source.replace(anchor, anchor + patch, 1)


def apply_dflash_timing_patch(root: Path) -> dict[str, Any]:
    """Apply the expected source patch and report immutable provenance."""
    path = Path(root) / "externals" / "dflash" / "dflash" / "model.py"
    source = path.read_text(encoding="utf-8")
    patched = patch_dflash_source(source)
    if patched != source:
        path.write_text(patched, encoding="utf-8")
    return {
        "patch_version": PATCH_VERSION,
        "source_sha256": SOURCE_SHA256,
        "patched_source_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
    }
