"""Compatibility helpers for the vendored EAGLE model.

The Qwen3 EAGLE model is generated from a newer Transformers source tree than
the offline server wheel.  Its generated typing-only model code imports
``LossKwargs`` from ``transformers.utils``.  Some supported Transformers
versions do not export that name, even though the class is only used to build
the ``**kwargs`` type annotation for the causal-LM forward method.
"""

from typing import TypedDict


def _default_rope_parameters(config, device=None, seq_len=None):
    """Compute the original, unscaled RoPE parameters.

    Transformers 5.8+ removed the ``"default"`` entry from
    ``ROPE_INIT_FUNCTIONS`` and moved this implementation into the generated
    model class.  The vendored EAGLE Qwen3 class still calls the registry, so
    keep the equivalent calculation available for that older generated code.
    """

    import torch

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        base = rope_parameters.get("rope_theta")
    else:
        base = None
    base = base or getattr(config, "rope_theta", 10000.0)
    dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
            / dim
        )
    )
    return inv_freq, 1.0


def install_eagle_transformers_compat() -> bool:
    """Install missing Transformers symbols required by the vendored EAGLE.

    Returns ``True`` when this helper added a compatibility symbol and
    ``False`` when the installed Transformers package already provides it.
    The fallback is deliberately a ``TypedDict``: EAGLE only consumes this
    class while constructing a generated typing annotation, not as a runtime
    loss implementation.
    """

    import transformers.utils as transformers_utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    installed = False
    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            labels: object

        transformers_utils.LossKwargs = LossKwargs
        installed = True

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _default_rope_parameters
        installed = True

    return installed
