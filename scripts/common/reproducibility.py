"""Shared RNG seeding for comparable baseline generations."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> int:
    """Reset Python, NumPy and Torch RNGs to the same benchmark seed.

    This intentionally does not enable all deterministic CUDA algorithms:
    doing so would change the performance path being benchmarked.  With the
    benchmark's default ``temperature=0`` this fixes sampling RNG state while
    preserving each baseline's intended kernel/backend.
    """

    value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    try:
        import numpy as np

        np.random.seed(value)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    except ImportError:
        pass
    return value
