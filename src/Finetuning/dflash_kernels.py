"""Kernel factories used by the self-contained Qwen3 DFlash backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
)


@dataclass(frozen=True)
class DFlashKernels:
    """Construction boundary for native or optional DFlash kernels."""

    make_rms_norm: Callable[[int, float], nn.Module]
    make_mlp: Callable[[Qwen3Config], nn.Module]


def _make_qwen3_rms_norm(hidden_size: int, eps: float) -> nn.Module:
    return Qwen3RMSNorm(hidden_size, eps=eps)


def _make_qwen3_mlp(config: Qwen3Config) -> nn.Module:
    return Qwen3MLP(config)


DEFAULT_DFLASH_KERNELS = DFlashKernels(
    make_rms_norm=_make_qwen3_rms_norm,
    make_mlp=_make_qwen3_mlp,
)


def load_liger_dflash_kernels() -> DFlashKernels:
    """Resolve Liger lazily while keeping the native import path offline-safe."""

    try:
        from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP
    except ModuleNotFoundError as exc:
        if exc.name in {"liger_kernel", "liger_kernel.transformers"}:
            raise ImportError(
                "Liger DFlash kernels require the optional `liger_kernel` package."
            ) from exc
        raise

    def make_rms_norm(hidden_size: int, eps: float) -> nn.Module:
        return LigerRMSNorm(hidden_size, eps=eps)

    def make_mlp(config: Qwen3Config) -> nn.Module:
        return LigerSwiGLUMLP(config)

    return DFlashKernels(
        make_rms_norm=make_rms_norm,
        make_mlp=make_mlp,
    )
