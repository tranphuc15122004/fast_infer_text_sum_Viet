"""Compatibility helpers for the vendored DFlash decoder."""


def install_dflash_transformers_compat() -> bool:
    """Restore the removed cache hook expected by the vendored DFlash code.

    DFlash only uses ``activate_past_recording`` as an initialization hook
    before calling the public ``DynamicCache.update`` and ``crop`` methods.
    Transformers 5 removed that hook and performs cache bookkeeping directly,
    so a no-op method is the correct compatibility behavior.
    """

    from transformers.cache_utils import DynamicCache

    if hasattr(DynamicCache, "activate_past_recording"):
        return False

    def activate_past_recording(self) -> None:
        return None

    DynamicCache.activate_past_recording = activate_past_recording
    return True
