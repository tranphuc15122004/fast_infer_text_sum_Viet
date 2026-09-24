from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class FakeCache:
    def __init__(self, seq_length: int) -> None:
        self.seq_length = seq_length
        self.crop_calls: list[int] = []

    def get_seq_length(self) -> int:
        return self.seq_length

    def crop(self, amount: int) -> None:
        self.crop_calls.append(amount)
        if amount < 0:
            self.seq_length += amount
        else:
            self.seq_length = min(self.seq_length, amount)


def test_dflash_cache_crop_is_noop_when_cache_is_already_at_requested_length() -> None:
    from Benchmark.dflash_compat import crop_dflash_cache_to_length

    cache = FakeCache(238)
    crop_dflash_cache_to_length(cache, 238)

    assert cache.crop_calls == []
    assert cache.get_seq_length() == 238


def test_dflash_cache_crop_removes_only_rejected_tail_tokens() -> None:
    from Benchmark.dflash_compat import crop_dflash_cache_to_length

    cache = FakeCache(240)
    crop_dflash_cache_to_length(cache, 238)

    assert cache.crop_calls == [-2]
    assert cache.get_seq_length() == 238


def test_dflash_cache_crop_compat_patches_vendored_decoder_module() -> None:
    from Benchmark.dflash_compat import install_dflash_cache_crop_compat

    module = SimpleNamespace(_crop_to=lambda cache, length: cache.crop(0))
    installed = install_dflash_cache_crop_compat(module)
    cache = FakeCache(238)

    assert installed is True
    module._crop_to(cache, 238)
    assert cache.crop_calls == []
