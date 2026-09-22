"""Compatibility helpers for the vendored EAGLE model.

The Qwen3 EAGLE model is generated from a newer Transformers source tree than
the offline server wheel.  Its generated typing-only model code imports
``LossKwargs`` from ``transformers.utils``.  Some supported Transformers
versions do not export that name, even though the class is only used to build
the ``**kwargs`` type annotation for the causal-LM forward method.
"""

from typing import TypedDict


def install_eagle_transformers_compat() -> bool:
    """Install missing Transformers symbols required by the vendored EAGLE.

    Returns ``True`` when this helper added a compatibility symbol and
    ``False`` when the installed Transformers package already provides it.
    The fallback is deliberately a ``TypedDict``: EAGLE only consumes this
    class while constructing a generated typing annotation, not as a runtime
    loss implementation.
    """

    import transformers.utils as transformers_utils

    if hasattr(transformers_utils, "LossKwargs"):
        return False

    class LossKwargs(TypedDict, total=False):
        labels: object

    transformers_utils.LossKwargs = LossKwargs
    return True
