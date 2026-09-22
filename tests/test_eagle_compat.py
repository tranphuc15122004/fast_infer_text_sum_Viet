from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_eagle_compat_installs_missing_loss_kwargs(monkeypatch):
    import transformers.utils as transformers_utils

    monkeypatch.delattr(transformers_utils, "LossKwargs", raising=False)

    from Benchmark.eagle_compat import install_eagle_transformers_compat

    installed = install_eagle_transformers_compat()

    assert installed is True
    assert hasattr(transformers_utils, "LossKwargs")
    assert "labels" in getattr(transformers_utils.LossKwargs, "__annotations__", {})


def test_eagle_compat_does_not_replace_existing_loss_kwargs(monkeypatch):
    import transformers.utils as transformers_utils

    class ExistingLossKwargs:
        pass

    monkeypatch.setattr(
        transformers_utils,
        "LossKwargs",
        ExistingLossKwargs,
        raising=False,
    )

    from Benchmark.eagle_compat import install_eagle_transformers_compat

    installed = install_eagle_transformers_compat()

    assert installed is False
    assert transformers_utils.LossKwargs is ExistingLossKwargs
