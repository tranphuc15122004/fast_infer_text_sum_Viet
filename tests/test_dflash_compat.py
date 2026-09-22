from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_dflash_compat_adds_removed_cache_recording_hook(monkeypatch):
    from transformers.cache_utils import DynamicCache

    monkeypatch.delattr(DynamicCache, "activate_past_recording", raising=False)

    from Benchmark.dflash_compat import install_dflash_transformers_compat

    installed = install_dflash_transformers_compat()

    assert installed is True
    cache = object.__new__(DynamicCache)
    assert cache.activate_past_recording() is None


def test_dflash_compat_does_not_replace_existing_hook(monkeypatch):
    from transformers.cache_utils import DynamicCache

    def existing_hook(self):
        return "existing"

    monkeypatch.setattr(
        DynamicCache,
        "activate_past_recording",
        existing_hook,
        raising=False,
    )

    from Benchmark.dflash_compat import install_dflash_transformers_compat

    installed = install_dflash_transformers_compat()

    assert installed is False
    cache = object.__new__(DynamicCache)
    assert cache.activate_past_recording() == "existing"
