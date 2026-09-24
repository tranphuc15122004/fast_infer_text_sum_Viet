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


def crop_dflash_cache_to_length(cache, length: int) -> None:
    """Discard only an unused DFlash KV-cache tail.

    Transformers 5 interprets ``DynamicCache.crop`` arguments as a number of
    tokens to remove when negative; ``crop(0)`` instead means crop to length
    zero. DFlash calls its helper when no speculative tokens were rejected, so
    forwarding zero would erase the entire target cache and corrupt decoding.
    """

    excess = int(cache.get_seq_length()) - int(length)
    if excess > 0:
        cache.crop(-excess)


def install_dflash_cache_crop_compat(dflash_model) -> bool:
    """Patch the vendored decoder to avoid destructive zero-length cache crops."""

    current = getattr(dflash_model, "_crop_to", None)
    if getattr(current, "_fast_infer_safe_cache_crop", False):
        return False
    crop_dflash_cache_to_length._fast_infer_safe_cache_crop = True
    dflash_model._crop_to = crop_dflash_cache_to_length
    return True
