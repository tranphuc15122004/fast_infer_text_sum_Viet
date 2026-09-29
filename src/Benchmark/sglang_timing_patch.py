"""Expose a direct first-token-to-finish timer in pinned SGLang responses.

Stock SGLang 0.5.20 emits ``decode_throughput`` in its final ``/generate``
response, but not the underlying duration.  The benchmark runs the endpoint in
streaming mode and patches only the response metadata conversion to expose the
same monotonic-clock interval that SGLang uses to compute that throughput.
"""

from __future__ import annotations

import math
from typing import Any


_SUPPORTED_SGLANG_VERSION = "0.5.20"
_PATCH_MARKER = "_fast_infer_direct_completion_latency_patch"


def patch_api_stats_class(stats_class: type[Any]) -> None:
    """Add direct ``completion_latency`` metadata to an SGLang stats class."""

    original = stats_class.convert_to_output_meta_info
    if getattr(original, _PATCH_MARKER, False):
        return

    def convert_to_output_meta_info(
        self: Any, scheduler_time_stats: Any = None, completion_tokens: int = 0
    ) -> dict[str, Any]:
        meta = original(
            self,
            scheduler_time_stats=scheduler_time_stats,
            completion_tokens=completion_tokens,
        )
        try:
            token_count = int(completion_tokens)
            first = float(self.first_token_time)
            finished = float(self.finished_time)
            duration = finished - first
        except (AttributeError, TypeError, ValueError, OverflowError):
            return meta
        if (
            token_count > 1
            and first > 0.0
            and finished > first
            and math.isfinite(duration)
        ):
            meta["completion_latency"] = duration
            meta["completion_latency_source"] = (
                "sglang_api_server_monotonic_first_token_to_finished"
            )
        return meta

    setattr(convert_to_output_meta_info, _PATCH_MARKER, True)
    stats_class.convert_to_output_meta_info = convert_to_output_meta_info


def install() -> None:
    """Install the version-pinned metadata patch before launching SGLang."""

    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("sglang")
    except PackageNotFoundError as exc:
        raise RuntimeError("SGLang timing patch requires sglang==0.5.20") from exc
    # Local CUDA wheel tags (for example ``+cu130``) do not change the
    # upstream Python source version this patch is anchored to.
    source_version = installed.split("+", 1)[0]
    if source_version != _SUPPORTED_SGLANG_VERSION:
        raise RuntimeError(
            "SGLang timing patch was validated for sglang=="
            f"{_SUPPORTED_SGLANG_VERSION}, found {installed}"
        )

    from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

    patch_api_stats_class(APIServerReqTimeStats)
    print(
        "[benchmark timing] installed direct SGLang completion timer "
        f"for sglang=={installed}",
        flush=True,
    )
