import pytest
from types import SimpleNamespace

import Benchmark.common.flashattn_runtime as flashattn_runtime

from Benchmark.common.flashattn_runtime import (
    first_token_mismatch,
    resolve_flashattn_methods,
    select_median_samples,
    select_smoke_datasets,
    validate_flashattn_runtime,
)


METHODS = ("vanilla_hf", "eagle3", "dflash", "domino", "dspark")


def test_first_token_mismatch_reports_first_difference_or_length_boundary():
    assert first_token_mismatch([10, 11, 12], [10, 12, 12]) == 1
    assert first_token_mismatch([10, 11], [10, 11, 12]) == 2
    assert first_token_mismatch([10, 11], [10, 11]) is None


def _runtime():
    return {
        "installed_distributions": ["torch", "transformers", "flash-attn-4"],
        "imported_modules": ["torch", "transformers", "flash_attn"],
        "batch_size": 1,
        "methods": {
            name: {
                "target_attention": "flash_attention_4",
                "draft_attention": "flash_attention_4" if name != "vanilla_hf" else None,
                "target_attention_dispatch": "flash_attention_4",
                "target_attention_dispatch_calls": 4,
                "target_fallback_attention_calls": 0,
                "draft_attention_dispatch": (
                    "flash_attention_4" if name != "vanilla_hf" else None
                ),
                "draft_attention_dispatch_calls": 3 if name != "vanilla_hf" else 0,
                "draft_fallback_attention_calls": 0,
            }
            for name in METHODS
        },
    }


def test_runtime_requires_all_methods_to_use_fa4_and_batch_one():
    result = validate_flashattn_runtime(_runtime(), methods=METHODS)

    assert result["passed"] is True
    assert result["batch_size"] == 1
    assert result["attention_backend"] == "flash_attention_4"


def test_comparison_methods_can_select_vanilla_and_dflash_only():
    assert resolve_flashattn_methods("vanilla_hf,dflash") == (
        "vanilla_hf",
        "dflash",
    )


def test_comparison_method_selection_rejects_unknown_or_duplicate_names():
    with pytest.raises(ValueError, match="unknown"):
        resolve_flashattn_methods("vanilla_hf,other")
    with pytest.raises(ValueError, match="unique"):
        resolve_flashattn_methods("vanilla_hf,dflash,dflash")


def test_fa4_gate_rejects_dflash_config_without_real_dispatch_proof():
    runtime = _runtime()
    runtime["methods"]["dflash"]["draft_attention_dispatch"] = "sdpa"
    runtime["methods"]["dflash"]["draft_attention_dispatch_calls"] = 0

    with pytest.raises(ValueError, match="DFlash FA4 dispatch was not observed"):
        validate_flashattn_runtime(
            runtime, methods=METHODS, require_dispatch_proof=True
        )


def test_fa4_gate_accepts_verified_dflash_dispatch():
    runtime = _runtime()
    runtime["methods"]["dflash"]["draft_attention_dispatch"] = "flash_attention_4"
    runtime["methods"]["dflash"]["draft_attention_dispatch_calls"] = 12
    runtime["methods"]["dflash"]["draft_sdpa_fallback_calls"] = 0

    result = validate_flashattn_runtime(
        runtime, methods=METHODS, require_dispatch_proof=True
    )

    assert result["methods"]["dflash"]["draft_attention_dispatch_calls"] == 12


def test_attention_dispatch_tracker_attributes_registry_calls_by_model_role():
    tracker_type = getattr(flashattn_runtime, "AttentionDispatchTracker", None)
    install_tracker = getattr(
        flashattn_runtime, "install_attention_dispatch_tracking", None
    )
    assert tracker_type is not None and callable(install_tracker), (
        "runtime must expose dispatch tracking for every FA4 model path"
    )

    target_attention = SimpleNamespace()
    draft_attention = SimpleNamespace()
    target = SimpleNamespace(modules=lambda: [target_attention])
    draft = SimpleNamespace(modules=lambda: [draft_attention])
    tracker = tracker_type(target, draft)
    registry = {
        "flash_attention_4": lambda module, *args, **kwargs: ("fa4", module),
        "sdpa": lambda module, *args, **kwargs: ("sdpa", module),
    }
    install_tracker(registry)

    with tracker.recording():
        assert registry["flash_attention_4"](target_attention) == (
            "fa4",
            target_attention,
        )
        assert registry["sdpa"](draft_attention) == ("sdpa", draft_attention)

    stats = tracker.snapshot()
    assert stats["target_attention_dispatch"] == "flash_attention_4"
    assert stats["target_attention_dispatch_calls"] == 1
    assert stats["draft_attention_dispatch"] is None
    assert stats["draft_attention_dispatch_calls"] == 0
    assert stats["draft_fallback_attention_calls"] == 1
    assert stats["draft_fallback_attention_backends"] == {"sdpa": 1}


def test_cuda_context_failure_detection_covers_asynchronous_kernel_errors():
    detector = getattr(flashattn_runtime, "is_cuda_context_failure", None)
    assert callable(detector), "runtime must detect poisoned CUDA contexts after kernel errors"
    assert detector(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert detector(RuntimeError("unspecified launch failure"))
    assert not detector(RuntimeError("model checkpoint not found"))


def test_fa4_gate_requires_runtime_dispatch_for_every_target_and_draft():
    runtime = _runtime()
    for method in METHODS:
        config = runtime["methods"][method]
        config.update(
            {
                "target_attention_dispatch": "flash_attention_4",
                "target_attention_dispatch_calls": 3,
                "target_fallback_attention_calls": 0,
                "draft_attention_dispatch": (
                    "flash_attention_4" if method != "vanilla_hf" else None
                ),
                "draft_attention_dispatch_calls": 2 if method != "vanilla_hf" else 0,
                "draft_fallback_attention_calls": 0,
            }
        )

    runtime["methods"]["dflash"].update(
        {
            "draft_attention_dispatch": "flash_attention_4",
            "draft_attention_dispatch_calls": 2,
            "draft_sdpa_fallback_calls": 0,
        }
    )
    runtime["methods"]["domino"]["draft_attention_dispatch_calls"] = 0

    with pytest.raises(ValueError, match="domino draft FA4 dispatch was not observed"):
        validate_flashattn_runtime(
            runtime, methods=METHODS, require_dispatch_proof=True
        )




def test_runtime_rejects_vllm_even_if_not_imported():
    runtime = _runtime()
    runtime["installed_distributions"].append("vllm")

    with pytest.raises(ValueError, match="vLLM must not be installed"):
        validate_flashattn_runtime(runtime, methods=METHODS)


def test_runtime_rejects_vllm_plugin_distribution_and_import():
    runtime = _runtime()
    runtime["installed_distributions"].append("fast-infer-viet-vllm-plugin")
    with pytest.raises(ValueError, match="including adapter plugins"):
        validate_flashattn_runtime(runtime, methods=METHODS)

    runtime = _runtime()
    runtime["imported_modules"].append("Benchmark.common.vllm_pilot_plugin")
    with pytest.raises(ValueError, match="vLLM modules or plugins"):
        validate_flashattn_runtime(runtime, methods=METHODS)


def test_runtime_rejects_a_baseline_that_fell_back_to_eager():
    runtime = _runtime()
    runtime["methods"]["dspark"]["target_attention"] = "eager"

    with pytest.raises(ValueError, match="dspark target attention"):
        validate_flashattn_runtime(runtime, methods=METHODS)


def test_runtime_rejects_batch_size_above_one():
    runtime = _runtime()
    runtime["batch_size"] = 2

    with pytest.raises(ValueError, match="batch_size must be 1"):
        validate_flashattn_runtime(runtime, methods=METHODS)


def test_runtime_requires_flash_attention_4_package():
    runtime = _runtime()
    runtime["installed_distributions"].remove("flash-attn-4")

    with pytest.raises(ValueError, match="flash-attn-4 must be installed"):
        validate_flashattn_runtime(runtime, methods=METHODS)


def test_speculative_baseline_requires_a_fa4_draft_path():
    runtime = _runtime()
    runtime["methods"]["dspark"]["draft_attention"] = None

    with pytest.raises(ValueError, match="dspark draft attention is missing"):
        validate_flashattn_runtime(runtime, methods=METHODS)


def test_smoke_datasets_are_evenly_spread_for_two_samples():
    assert select_smoke_datasets(("vietnews", "wikilingua", "vims", "vlsp"), 2) == (
        "vietnews",
        "vlsp",
    )


def test_median_sample_selection_is_deterministic_and_respects_token_cap():
    candidates = [
        {"dataset": "vietnews", "sample_id": "vn:1", "input_tokens": 4},
        {"dataset": "vietnews", "sample_id": "vn:2", "input_tokens": 8},
        {"dataset": "vietnews", "sample_id": "vn:3", "input_tokens": 12},
        {"dataset": "vietnews", "sample_id": "vn:4", "input_tokens": 20},
    ]

    selected = select_median_samples(
        candidates, datasets=("vietnews",), max_input_tokens=15
    )

    assert [row["sample_id"] for row in selected] == ["vn:2"]
