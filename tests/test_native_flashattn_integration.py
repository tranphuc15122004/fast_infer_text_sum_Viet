from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from Benchmark.fa4_server import build_parser, runner_kwargs
from Benchmark.native_flashattn import NativeInferenceConfig


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setenv("FA4_EXECUTION_BACKEND", "server")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(set_seed=lambda seed: None))
    return importlib.import_module("modal_flashattn_pilot")


def test_actual_runner_accepts_every_server_cli_argument(runner):
    kwargs = runner_kwargs(build_parser().parse_args([]))
    params = inspect.signature(runner.run_flashattn_benchmark).parameters
    assert set(kwargs) <= set(params)


def test_actual_domino_call_passes_the_native_graph_and_keeps_greedy(runner):
    calls = []
    graph = object()
    draft = SimpleNamespace(block_size=16, spec_generate=lambda *a, **k: calls.append(k))
    context = {"target": object(), "draft": draft, "stop_token_ids": [99],
               "domino_graph_runner": graph, "native_config": NativeInferenceConfig(),
               "domino_model_module": SimpleNamespace(cuda_time=lambda *a: 1.0)}
    runner._call_method(None, "domino", context, "input", max_new_tokens=64)
    assert calls[0]["graph_runner"] is graph
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["use_bias"] is True
    assert calls[0]["block_size"] == 16


def test_actual_dflash_call_can_profile_separately_without_changing_native_global_timer(runner):
    calls = []
    syncs = []
    original_timer = lambda: syncs.append(True) or 1.0
    module = SimpleNamespace(_cuda_time=original_timer)
    def generate(*a, **k):
        module._cuda_time()
        calls.append(k)
        return "native-result"
    context = {"target": object(), "draft": object(), "stop_token_ids": [99],
               "dflash_generate": generate, "dflash_model_module": module,
               "native_config": NativeInferenceConfig()}
    assert runner._call_method(None, "dflash", context, "input", max_new_tokens=64) == "native-result"
    assert syncs == []
    assert context["last_skipped_profiling_synchronizations"] == 1
    runner._call_method(None, "dflash", context, "input", max_new_tokens=64, profiling=True)
    assert syncs == [True]
    assert module._cuda_time is original_timer
    assert calls[0] == calls[1]
    assert calls[0]["block_size"] == 16
    assert calls[0]["return_stats"] is True


def test_vendored_dflash_loop_keeps_tokens_and_acceptance_when_profiling_barriers_are_removed(monkeypatch):
    """CPU algorithm regression only; this does not validate the FA4 GPU kernel."""
    import time
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from Benchmark.dflash_compat import (
        install_dflash_cache_crop_compat, install_dflash_transformers_compat,
    )
    from Benchmark.native_flashattn import run_native_method

    path = Path(__file__).resolve().parents[1] / "externals/dflash/dflash/model.py"
    if not path.is_file():
        pytest.skip("vendored DFlash unavailable")
    install_dflash_transformers_compat()
    spec = importlib.util.spec_from_file_location("_test_native_dflash_cpu", path)
    native = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, native)
    spec.loader.exec_module(native)
    install_dflash_cache_crop_compat(native)
    profiling_calls = []
    def cpu_profiling_timer():
        profiling_calls.append(True)
        return time.perf_counter()
    monkeypatch.setattr(native, "_cuda_time", cpu_profiling_timer)

    torch.manual_seed(0)
    shared = dict(vocab_size=32, hidden_size=32, intermediate_size=64,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                  max_position_embeddings=128, eos_token_id=None, pad_token_id=0)
    target_config = Qwen3Config(num_hidden_layers=4, **shared)
    target_config._attn_implementation = "eager"
    draft_config = Qwen3Config(num_hidden_layers=1, **shared)
    draft_config.num_target_layers = 4
    draft_config.dflash_config = {"block_size": 16, "mask_token_id": 31, "target_layer_ids": [1]}
    draft_config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(target_config).eval()
    draft = native.DFlashDraftModel(draft_config).eval()
    context = {"target": target, "draft": draft, "stop_token_ids": [],
               "dflash_generate": native.dflash_generate, "dflash_model_module": native,
               "native_config": NativeInferenceConfig()}
    input_ids = torch.tensor([[1, 5, 7]])
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        direct = native.dflash_generate(draft, target, input_ids, 17, None,
                                       block_size=16, return_stats=True)
        assert profiling_calls
        profiling_calls.clear()
        measured = run_native_method(torch, "dflash", context, input_ids, max_new_tokens=17)
        assert not profiling_calls
        assert context["last_skipped_profiling_synchronizations"] > 0
        profile = run_native_method(torch, "dflash", context, input_ids, max_new_tokens=17, profiling=True)
        assert profiling_calls
        assert measured.output_ids.tolist() == direct.output_ids.tolist() == profile.output_ids.tolist()
        for key in ("acceptance_lengths", "draft_tokens_accepted", "draft_tokens_proposed", "committed_tokens_per_step"):
            assert getattr(measured, key) == getattr(direct, key) == getattr(profile, key)
    finally:
        torch.set_num_threads(threads)
