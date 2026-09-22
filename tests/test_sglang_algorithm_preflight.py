from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_sglang_algorithm_preflight_uses_installed_registry(monkeypatch):
    import importlib

    from Benchmark.common import longbench_adapter

    class FakeAlgorithms:
        @classmethod
        def from_string(cls, name):
            if name != "DFLASH":
                raise ValueError(f"unknown algorithm: {name}")
            return name

    original_import = importlib.import_module

    def fake_import(name):
        if name == "sglang.srt.speculative.spec_info":
            return SimpleNamespace(SpeculativeAlgorithm=FakeAlgorithms)
        return original_import(name)

    monkeypatch.setattr(longbench_adapter.importlib, "import_module", fake_import)

    assert longbench_adapter._sglang_algorithm_supported("DFLASH") == (True, None)
    supported, reason = longbench_adapter._sglang_algorithm_supported("DSPARK")
    assert supported is False
    assert "DSPARK" in str(reason)
