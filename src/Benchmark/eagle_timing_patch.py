"""Version-pinned, idempotent EAGLE strict-decode instrumentation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


PATCH_VERSION = "eagle-strict-decode-v1"
EXPECTED_UTILS_SHA256 = "68dadf1977dd76624f87d0dc5ce9f840783ccaeb7554861f1206daa664367fe9"
EXPECTED_MODEL_SHA256 = "3ea963565ea8b5b29033ae2f27a160eee69ffce00d5670eb21307d50b00e407b"
_UTILS_MARKER = "fast_infer_strict_decode_v1"
_MODEL_MARKER = "fast_infer_strict_decode_v1"


def patch_eagle_sources(utils_source: str, model_source: str) -> tuple[str, str]:
    """Return patched EAGLE source text; repeated application is byte-stable."""
    utils_patched = _UTILS_MARKER in utils_source
    model_patched = _MODEL_MARKER in model_source
    if utils_patched and model_patched:
        return utils_source, model_source
    if utils_patched != model_patched:
        raise ValueError("EAGLE strict-decode patch is only partially applied")

    utils_anchor = "    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)\n"
    if utils_anchor not in utils_source:
        raise ValueError("EAGLE initialize_tree first-token anchor does not match the pinned source")
    utils_insert = utils_anchor + (
        "    # fast_infer_strict_decode_v1: first token is now committed.\n"
        "    if input_ids.is_cuda:\n"
        "        torch.cuda.synchronize(input_ids.device)\n"
        "    model._fast_infer_first_token_commit_time = time.perf_counter()\n"
    )
    utils_source = utils_source.replace(utils_anchor, utils_insert, 1)

    naive_marker = "    def naivegenerate("
    if naive_marker not in model_source:
        raise ValueError("EAGLE naivegenerate anchor does not match the pinned source")
    before_naive, naive_source = model_source.split(naive_marker, 1)
    loop_anchor = "        for idx in range(max_length):\n"
    if loop_anchor not in naive_source:
        raise ValueError("EAGLE naive decode-loop anchor does not match the pinned source")
    naive_source = naive_source.replace(
        loop_anchor,
        "        self._fast_infer_first_token_commit_time = None\n" + loop_anchor,
        1,
    )
    token_forward_anchor = (
        "            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)\n"
    )
    if token_forward_anchor not in naive_source:
        raise ValueError("EAGLE naive first-token forward anchor does not match the pinned source")
    token_forward_insert = (
        "            if self._fast_infer_first_token_commit_time is None:\n"
        "                if input_ids.is_cuda:\n"
        "                    torch.cuda.synchronize(input_ids.device)\n"
        "                self._fast_infer_first_token_commit_time = time.perf_counter()\n"
        + token_forward_anchor
    )
    naive_source = naive_source.replace(token_forward_anchor, token_forward_insert, 1)
    model_source = before_naive + naive_marker + naive_source

    phase_anchor = "        phase_timings = {\n"
    phase_payload = (
        "        # fast_infer_strict_decode_v1: phase begins after token one commits.\n"
        "        strict_decode_start = getattr(self, \"_fast_infer_first_token_commit_time\", None)\n"
        "        strict_decode_ms = (max(0.0, (time.perf_counter() - strict_decode_start) * 1000.0)\n"
        "                            if strict_decode_start is not None else None)\n"
        "        strict_decode_tokens = max(int(input_ids.shape[1] - input_len) - 1, 0)\n"
        + phase_anchor
    )
    if model_source.count(phase_anchor) < 2:
        raise ValueError("EAGLE speculative/naive timing dictionaries do not match the pinned source")
    model_source = model_source.replace(phase_anchor, phase_payload, 2)
    spec_decode_anchor = '            "decode_ms": decode_ms,\n'
    naive_decode_anchor = '            "decode_ms": max(0.0, e2e_ms - prefill_ms),\n'
    if spec_decode_anchor not in model_source or naive_decode_anchor not in model_source:
        raise ValueError("EAGLE phase timing field anchors do not match the pinned source")
    strict_fields = (
        '            "strict_decode_active_ms": strict_decode_ms,\n'
        '            "strict_decode_token_count": strict_decode_tokens,\n'
        '            "strict_decode_phase_definition": "after_first_token_committed_to_final_token",\n'
        '            "strict_decode_phase_verified": strict_decode_ms is not None,\n'
    )
    model_source = model_source.replace(spec_decode_anchor, spec_decode_anchor + strict_fields, 1)
    model_source = model_source.replace(naive_decode_anchor, naive_decode_anchor + strict_fields, 1)
    return utils_source, model_source


def apply_eagle_timing_patch(root: Path) -> dict[str, Any]:
    """Apply the expected source patch and return source provenance metadata."""
    eagle = Path(root) / "externals" / "EAGLE" / "eagle" / "model"
    utils_path = eagle / "utils.py"
    model_path = eagle / "ea_model.py"
    utils_source = utils_path.read_text(encoding="utf-8")
    model_source = model_path.read_text(encoding="utf-8")
    if _UTILS_MARKER in utils_source and _MODEL_MARKER in model_source:
        return {
            "patch_version": PATCH_VERSION,
            "source_utils_sha256": EXPECTED_UTILS_SHA256,
            "source_model_sha256": EXPECTED_MODEL_SHA256,
            "patched_utils_sha256": hashlib.sha256(utils_source.encode()).hexdigest(),
            "patched_model_sha256": hashlib.sha256(model_source.encode()).hexdigest(),
        }
    actual_utils = hashlib.sha256(utils_source.encode()).hexdigest()
    actual_model = hashlib.sha256(model_source.encode()).hexdigest()
    if actual_utils != EXPECTED_UTILS_SHA256 or actual_model != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "EAGLE vendored source changed from the reviewed timing-patch version; "
            f"utils={actual_utils}, ea_model={actual_model}"
        )
    patched_utils, patched_model = patch_eagle_sources(utils_source, model_source)
    utils_path.write_text(patched_utils, encoding="utf-8")
    model_path.write_text(patched_model, encoding="utf-8")
    return {
        "patch_version": PATCH_VERSION,
        "source_utils_sha256": actual_utils,
        "source_model_sha256": actual_model,
        "patched_utils_sha256": hashlib.sha256(patched_utils.encode()).hexdigest(),
        "patched_model_sha256": hashlib.sha256(patched_model.encode()).hexdigest(),
    }
