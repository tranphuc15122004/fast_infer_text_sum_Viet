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


def test_dflash_normalizes_qwen_stop_ids_for_llama_vocab() -> None:
    from Benchmark.infer_dflash import normalize_generation_token_ids

    class Config:
        vocab_size = 128256
        bos_token_id = 151643
        eos_token_id = 151645

    class Tokenizer:
        bos_token_id = 128000
        eos_token_id = 128001

        def __len__(self):
            return 128256

    changed = normalize_generation_token_ids(Config, Tokenizer())

    assert changed == {
        "bos_token_id": (151643, 128000),
        "eos_token_id": (151645, 128001),
    }
    assert Config.bos_token_id == 128000
    assert Config.eos_token_id == 128001


def test_dflash_warmup_calls_generation_before_measurement(monkeypatch) -> None:
    from Benchmark import infer_dflash

    calls = []

    def record_generation(*args, **kwargs):
        calls.append(kwargs)
        return None, 0.0

    monkeypatch.setattr(infer_dflash, "_run_generation", record_generation)
    infer_dflash._warmup_generation(
        dflash_generate=object(),
        draft=object(),
        target=object(),
        input_ids=object(),
        warmup_runs=2,
        max_new_tokens=8,
        temperature=0.0,
        block_size=16,
    )

    assert calls == [
        {"max_new_tokens": 8, "temperature": 0.0, "block_size": 16},
        {"max_new_tokens": 8, "temperature": 0.0, "block_size": 16},
    ]
