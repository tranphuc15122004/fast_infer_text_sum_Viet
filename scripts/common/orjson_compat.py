"""Minimal stdlib-``json`` implementation of the ``orjson`` API subset that the
vendored SSSD/SGLang fork uses on its offline throughput/benchmark path.

``externals/SSSD/python/sglang`` imports ``orjson`` at module scope (e.g.
``sglang/srt/utils/common.py``) and only ever calls ``orjson.loads(...)`` or
``orjson.dumps(..., option=...)`` on the code paths exercised by
``bench_offline_throughput``.  On servers where the real ``orjson`` C
extension is not installed, ``infer_sssd.py`` installs this module as
``orjson.py`` on ``PYTHONPATH`` so the same upstream code runs against the
Python standard library only -- no extra third-party dependency.

Behavioural notes vs. the real package:

* ``dumps`` returns ``bytes`` (UTF-8, ``ensure_ascii=False``) like orjson.
* ``loads`` accepts ``str``/``bytes``/``bytearray``/``memoryview``.
* The option flags referenced anywhere in the SSSD tree are defined; only the
  subset actually used (``OPT_NON_STR_KEYS``, ``OPT_SERIALIZE_NUMPY``,
  indentation, sorted keys, trailing newline) changes the emitted output.
* NaN / Infinity raise like orjson's default instead of producing the
  non-standard ``NaN``/``Infinity`` literals ``json.dumps`` emits by default.
"""

from __future__ import annotations

import json
from json import JSONDecodeError  # noqa: F401  # same exception family as orjson

# orjson public option flags (values match the real package so ``|``-combined
# option expressions keep working when this module shadows it).
OPT_INDENT_2 = 2
OPT_APPEND_NEWLINE = 4
OPT_NAIVE_UTC = 8
OPT_OMIT_MICROSECONDS = 16
OPT_NON_STR_KEYS = 32
OPT_SERIALIZE_NUMPY = 64
OPT_SORT_KEYS = 128
OPT_STRICT_INTEGER = 256
OPT_UTC_Z = 512
OPT_PASSTHROUGH_DATACLASS = 1024
OPT_PASSTHROUGH_DATETIME = 2048
OPT_PASSTHROUGH_SUBCLASS = 4096
OPT_PASSTHROUGH_TUPLE = 8192
OPT_SERIALIZE_DATACLASS = 16384
OPT_SERIALIZE_DATETIME = 32768
OPT_SERIALIZE_NUMPY_DTYPE = 65536
OPT_SERIALIZE_UUID = 131072

__version__ = "0.0.0-stdlib-json"


def _numpy_default(obj: object):
    """Convert numpy scalars/arrays to JSON-native values (OPT_SERIALIZE_NUMPY)."""
    module = type(obj).__module__ or ""
    if module.startswith("numpy"):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def dumps(obj: object, default=None, option: int | None = None) -> bytes:
    """Serialize ``obj`` to UTF-8 JSON bytes, mirroring ``orjson.dumps``.

    Only the option flags implemented below are honoured; the rest are accepted
    for import compatibility but have no effect.
    """
    if option is None:
        option = 0
    kwargs: dict = {}
    if option & OPT_INDENT_2:
        kwargs["indent"] = 2
    if option & OPT_SORT_KEYS:
        kwargs["sort_keys"] = True
    if default is None and (option & OPT_SERIALIZE_NUMPY):
        default = _numpy_default
    # json.dumps already stringifies non-str dict keys, which is what
    # orjson.OPT_NON_STR_KEYS asks for, so no extra handling is needed.
    text = json.dumps(
        obj,
        default=default,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        **kwargs,
    )
    if option & OPT_APPEND_NEWLINE:
        text += "\n"
    return text.encode("utf-8")


def loads(value) -> object:
    """Parse JSON from ``str`` or a bytes-like buffer, mirroring orjson.loads."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8")
    return json.loads(value)
