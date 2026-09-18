"""Compatibility helpers for model configuration schemas.

Transformers 5 stores Llama RoPE values in ``rope_parameters`` and mirrors
them through ``rope_scaling``.  Older vendored inference implementations read
``config.rope_theta`` directly.  These helpers make that one-way compatibility
adjustment on an in-memory config without changing the model checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any


def _find_numeric(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Real) and not isinstance(value, bool):
            return float(value)
    for value in mapping.values():
        if isinstance(value, Mapping):
            found = _find_numeric(value, keys)
            if found is not None:
                return found
    return None


def get_rope_theta(config: Any, default: float = 10000.0) -> float:
    """Return the Llama RoPE base from old or new config layouts."""

    value = getattr(config, "rope_theta", None)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)

    for attr in ("rope_parameters", "rope_scaling"):
        value = getattr(config, attr, None)
        if isinstance(value, Mapping):
            found = _find_numeric(value, ("rope_theta", "base"))
            if found is not None:
                return found

    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
        if isinstance(raw, Mapping):
            found = _find_numeric(raw, ("rope_theta", "base"))
            if found is not None:
                return found

    return float(default)


def ensure_rope_theta(config: Any, default: float = 10000.0) -> Any:
    """Populate the legacy ``config.rope_theta`` attribute in-place."""

    value = getattr(config, "rope_theta", None)
    if not isinstance(value, Real) or isinstance(value, bool):
        setattr(config, "rope_theta", get_rope_theta(config, default=default))
    return config


__all__ = ["ensure_rope_theta", "get_rope_theta"]
