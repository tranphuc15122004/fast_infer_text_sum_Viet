#!/usr/bin/env python3
"""Compare vendored author inference with the synchronized vLLM pilot trace.

Run from the repository root with:
    modal run scripts/modal_author_reference_compare.py::run_reference

The input is generated from one existing pilot sample, preserving its exact
prompt token IDs and the vLLM vanilla/EAGLE3/Domino outputs for parity checks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "outputs/modal_vllm_smoke/20260930T173501Z-cfee7aba"
INPUT_PATH = ROOT / "outputs/modal_vllm_smoke_input/author_reference_sample.json"
DOMINO_CODE = ROOT / "externals/Domino/code"
EAGLE_CODE = ROOT / "externals/EAGLE/eagle"

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12")
    .entrypoint([])
    # The author implementations in these vendored repositories target the
    # Transformers 4.x API. Keep the installed Torch/CUDA stack from the H100
    # vLLM image, but pin the Python model API to a compatible 4.x release.
    .pip_install("transformers==4.57.3", "accelerate>=1.1.0")
    .add_local_dir(str(DOMINO_CODE), remote_path="/workspace/Domino/code", copy=True)
    .add_local_dir(str(EAGLE_CODE), remote_path="/workspace/EAGLE/eagle", copy=True)
    .add_local_file(str(INPUT_PATH), remote_path="/workspace/author_reference_sample.json", copy=True)
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

app = modal.App("fast-infer-viet-author-reference")


def _lcs(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _parity(actual: list[int], expected: list[int]) -> dict:
    common = 0
    for left, right in zip(actual, expected):
        if left != right:
            break
        common += 1
    return {
        "exact": actual == expected,
        "actual_tokens": len(actual),
        "expected_tokens": len(expected),
        "common_prefix_tokens": common,
        "first_mismatch_index": common if common < min(len(actual), len(expected)) else None,
        "lcs_tokens": _lcs(actual, expected),
        "lcs_fraction_of_expected": _lcs(actual, expected) / max(1, len(expected)),
    }


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=3600)
def run_reference(max_new_tokens: int = 512) -> dict:
    import gc
    import sys
    import traceback

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["HF_HOME"] = "/tmp/hf-author-reference"
    os.environ["HF_HUB_CACHE"] = "/tmp/hf-author-reference/hub"

    with open("/workspace/author_reference_sample.json", encoding="utf-8") as handle:
        case = json.load(handle)

    target_path = snapshot_download(
        repo_id=case["target_repo"], revision=case["target_revision"]
    )
    domino_path = snapshot_download(
        repo_id=case["domino_repo"], revision=case["domino_revision"]
    )
    eagle_path = snapshot_download(
        repo_id=case["eagle_repo"], revision=case["eagle_revision"]
    )
    tokenizer = AutoTokenizer.from_pretrained(target_path, use_fast=True)
    input_ids = torch.tensor([case["prompt_token_ids"]], dtype=torch.long, device="cuda")
    prompt_text = tokenizer.decode(
        case["prompt_token_ids"], skip_special_tokens=False
    )
    reencoded = tokenizer.encode(prompt_text, add_special_tokens=False)
    tokenizer_check = {
        "exact_prompt_ids_match": reencoded == case["prompt_token_ids"],
        "prompt_tokens": len(case["prompt_token_ids"]),
        "reencoded_tokens": len(reencoded),
    }

    try:
        from transformers.utils import is_flash_attn_2_available

        attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    except Exception:
        attn_impl = "sdpa"

    result = {
        "sample_id": case["sample_id"],
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "target_repo": case["target_repo"],
        "target_revision": case["target_revision"],
        "attention_implementation": attn_impl,
        "dtype": "bfloat16",
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "prompt_tokens": len(case["prompt_token_ids"]),
        "tokenizer_check": tokenizer_check,
        "vllm_outputs": case["vllm_outputs"],
    }

    # First establish a direct Transformers greedy reference with the same
    # target checkpoint, prompt IDs, dtype, and stop token as the pilot.
    try:
        target = AutoModelForCausalLM.from_pretrained(
            target_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to("cuda").eval()
        target.config.use_cache = True
        target.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=min(64, max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hf_vanilla_full = target.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        torch.cuda.synchronize()
        hf_vanilla_wall_ms = (time.perf_counter() - t0) * 1000.0
        hf_vanilla_ids = [int(x) for x in hf_vanilla_full[0, input_ids.shape[1]:].tolist()]
        result["transformers_vanilla"] = {
            "status": "success",
            "output_token_ids": hf_vanilla_ids,
            "wall_ms": hf_vanilla_wall_ms,
            "parity_vs_vllm_vanilla": _parity(
                hf_vanilla_ids, case["vllm_outputs"]["vanilla_vllm"]["token_ids"]
            ),
        }
    except Exception as exc:
        result["transformers_vanilla"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        target = None

    # Invoke the vendored Domino author implementation directly, without the
    # vLLM adapter. Graph capture is disabled here; it changes execution speed,
    # not the draft/verify algorithm under test.
    try:
        sys.path.insert(0, "/workspace/Domino/code")
        from dflash import DFlashDraftModel

        draft_config = AutoConfig.from_pretrained(domino_path)
        draft = DFlashDraftModel.from_pretrained(
            domino_path,
            config=draft_config,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        ).to("cuda").eval()
        if target is None:
            target = AutoModelForCausalLM.from_pretrained(
                target_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            ).to("cuda").eval()
        target.config.use_cache = True
        block_size = int(draft.block_size)

        draft.spec_generate(
            input_ids=input_ids,
            target=target,
            max_new_tokens=min(64, max_new_tokens),
            temperature=0.0,
            stop_token_ids=[tokenizer.eos_token_id],
            block_size=block_size,
            use_bias=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        domino_out = draft.spec_generate(
            input_ids=input_ids,
            target=target,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            stop_token_ids=[tokenizer.eos_token_id],
            block_size=block_size,
            use_bias=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        domino_wall_ms = (time.perf_counter() - t0) * 1000.0
        domino_ids = [
            int(x)
            for x in domino_out.output_ids[0, domino_out.num_input_tokens:].tolist()
        ]
        result["domino_author"] = {
            "status": "success",
            "repo": case["domino_repo"],
            "revision": case["domino_revision"],
            "block_size": block_size,
            "output_token_ids": domino_ids,
            "text": tokenizer.decode(domino_ids, skip_special_tokens=True),
            "wall_ms": domino_wall_ms,
            "source_prefill_ms": float(domino_out.time_to_first_token) * 1000.0,
            "source_decode_ms_per_token": float(domino_out.time_per_output_token) * 1000.0,
            "acceptance_lengths": [int(x) for x in domino_out.acceptance_lengths],
            "mean_acceptance_length": (
                sum(domino_out.acceptance_lengths) / max(1, len(domino_out.acceptance_lengths))
            ),
            "parity_vs_vllm_vanilla": _parity(
                domino_ids, case["vllm_outputs"]["vanilla_vllm"]["token_ids"]
            ),
            "parity_vs_vllm_domino": _parity(
                domino_ids, case["vllm_outputs"]["domino"]["token_ids"]
            ),
            "parity_vs_transformers_vanilla": (
                _parity(domino_ids, result["transformers_vanilla"]["output_token_ids"])
                if result["transformers_vanilla"].get("status") == "success"
                else None
            ),
        }
    except Exception as exc:
        result["domino_author"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if "draft" in locals():
            draft = None
        target = None
        gc.collect()
        torch.cuda.empty_cache()

    # The vendored EAGLE entrypoint uses this model path and `eagenerate`.
    # AngelSlim's checkpoint is the same artifact served by vLLM, but it is an
    # unofficial EAGLE3 Qwen3 checkpoint; report compatibility failures clearly.
    try:
        sys.path.insert(0, "/workspace/EAGLE")

        # This vendored Qwen3 file mixes the newer `inputs_embeds` masking call
        # with the Transformers 4.53 API (`input_embeds`, explicit
        # `cache_position`). Adapt only this call boundary for batch-1 prompts;
        # the attention mask implementation itself remains Transformers 4.53.
        import eagle.model.modeling_qwen3_kv as eagle_qwen3
        from transformers.masking_utils import (
            create_causal_mask as hf_create_causal_mask,
            create_sliding_window_causal_mask as hf_create_sliding_window_causal_mask,
        )

        def adapt_mask_call(mask_fn):
            def wrapped(*args, **kwargs):
                if "inputs_embeds" in kwargs:
                    kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
                if "cache_position" not in kwargs:
                    position_ids = kwargs.get("position_ids")
                    if position_ids is None:
                        raise ValueError("reference mask call omitted position_ids")
                    kwargs["cache_position"] = position_ids[0]
                return mask_fn(*args, **kwargs)
            return wrapped

        eagle_qwen3.create_causal_mask = adapt_mask_call(hf_create_causal_mask)
        eagle_qwen3.create_sliding_window_causal_mask = adapt_mask_call(
            hf_create_sliding_window_causal_mask
        )
        from eagle.model.ea_model import EaModel

        eagle = EaModel.from_pretrained(
            base_model_path=target_path,
            ea_model_path=eagle_path,
            total_token=16,
            depth=8,
            top_k=4,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
            use_eagle3=True,
        ).eval()
        eagle_inputs = input_ids.to(next(eagle.parameters()).device)
        eagle.eagenerate(
            eagle_inputs,
            temperature=0.0,
            max_new_tokens=min(64, max_new_tokens),
            log=True,
            return_phase_timings=True,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eagle_out = eagle.eagenerate(
            eagle_inputs,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            log=True,
            return_phase_timings=True,
        )
        torch.cuda.synchronize()
        eagle_wall_ms = (time.perf_counter() - t0) * 1000.0
        eagle_full_ids, new_tokens, accepted, source_e2e_s, accept_lengths, phase = eagle_out
        eagle_ids = [int(x) for x in eagle_full_ids[0, len(case["prompt_token_ids"]):].tolist()]
        result["eagle3_author"] = {
            "status": "success",
            "repo": case["eagle_repo"],
            "revision": case["eagle_revision"],
            "total_token": 16,
            "depth": 8,
            "top_k": 4,
            "output_token_ids": eagle_ids,
            "text": tokenizer.decode(eagle_ids, skip_special_tokens=True),
            "wall_ms": eagle_wall_ms,
            "source_e2e_ms": float(source_e2e_s) * 1000.0,
            "new_tokens_counter": int(new_tokens),
            "accepted_counter": int(accepted),
            "phase_timings": phase,
            "acceptance_lengths": [int(x) for x in accept_lengths],
            "parity_vs_vllm_vanilla": _parity(
                eagle_ids, case["vllm_outputs"]["vanilla_vllm"]["token_ids"]
            ),
            "parity_vs_vllm_eagle3": _parity(
                eagle_ids, case["vllm_outputs"]["eagle3"]["token_ids"]
            ),
            "parity_vs_transformers_vanilla": (
                _parity(eagle_ids, result["transformers_vanilla"]["output_token_ids"])
                if result["transformers_vanilla"].get("status") == "success"
                else None
            ),
        }
    except Exception as exc:
        result["eagle3_author"] = {
            "status": "failed_or_unsupported",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    return json.dumps(result, ensure_ascii=False)


# EAGLE's vendored Qwen3 implementation imports internals that are not present
# in Transformers 4.57. Run it separately in the untouched vLLM image, which
# matches the pilot's Torch 2.13 / Transformers 5.17 runtime.
eagle_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12")
    .entrypoint([])
    # The vendored EAGLE Qwen3 source imports LossKwargs and the legacy
    # "default" RoPE implementation, both compatible with Transformers 4.53.x.
    # Install the package without dependency resolution so the vLLM image's
    # pinned Torch/CUDA stack remains unchanged.
    .run_commands("python -m pip install transformers==4.53.3 tokenizers==0.21.4 huggingface-hub==0.36.0 --no-deps --target=/opt/transformers453")
    .add_local_dir(str(EAGLE_CODE), remote_path="/workspace/EAGLE/eagle", copy=True)
    .add_local_file(str(INPUT_PATH), remote_path="/workspace/author_reference_sample.json", copy=True)
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


@app.function(image=eagle_image, gpu="H100", cpu=8, memory=32768, timeout=3600)
def run_eagle_reference(max_new_tokens: int = 512) -> str:
    import glob
    import importlib
    import importlib.util
    import json
    import os
    import sys
    import time
    import traceback
    from pathlib import Path

    # Match the vLLM pilot's package-path setup. The Modal worker prepends its
    # own site-packages, while vLLM's Python/CUDA packages can live elsewhere.
    sites = sorted(
        {
            path
            for pattern in (
                "/usr/local/lib/python*/site-packages",
                "/usr/local/lib/python*/dist-packages",
                "/usr/lib/python*/site-packages",
                "/usr/lib/python*/dist-packages",
                "/opt/venv/lib/python*/site-packages",
                "/opt/conda/lib/python*/site-packages",
                "/root/.venv/lib/python*/site-packages",
                "/venv/lib/python*/site-packages",
            )
            for path in glob.glob(pattern)
        }
    )
    sites = [
        path for path in sites
        if (Path(path) / "torch" / "__init__.py").is_file()
        or (Path(path) / "transformers" / "__init__.py").is_file()
    ]
    # Modal may preload the image's original Transformers before this worker
    # starts. Prefer the pinned 4.53.3 install and evict any preloaded modules.
    pinned_sites = ["/opt/transformers453"]
    sites = pinned_sites + [path for path in sites if path not in pinned_sites]
    sys.path[:0] = [path for path in sites if path not in sys.path]
    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            sys.modules.pop(name, None)
    import typing_extensions

    site_module = next(
        (Path(path) / "typing_extensions.py" for path in sites
         if (Path(path) / "typing_extensions.py").is_file()),
        None,
    )
    if not hasattr(typing_extensions, "Sentinel") and site_module is not None:
        spec = importlib.util.spec_from_file_location("typing_extensions", site_module)
        replacement = importlib.util.module_from_spec(spec)
        sys.modules["typing_extensions"] = replacement
        spec.loader.exec_module(replacement)
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    for name in list(sys.modules):
        if name == "google" or name.startswith("google."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The vendored Qwen3 model uses LossKwargs only as a TypedDict mixin for
    # call-time annotations. The type moved/vanished across Transformers
    # releases, so add a no-op compatibility alias without changing inference.
    from typing import TypedDict

    if not hasattr(transformers.utils, "LossKwargs"):
        transformers.utils.LossKwargs = TypedDict("LossKwargs", {})

    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    os.environ["HF_HOME"] = "/tmp/hf-eagle-reference"
    os.environ["HF_HUB_CACHE"] = "/tmp/hf-eagle-reference/hub"
    case = json.loads(Path("/workspace/author_reference_sample.json").read_text())
    target_path = snapshot_download(
        repo_id=case["target_repo"], revision=case["target_revision"]
    )
    eagle_path = snapshot_download(
        repo_id=case["eagle_repo"], revision=case["eagle_revision"]
    )
    tokenizer = AutoTokenizer.from_pretrained(target_path, use_fast=False)
    input_ids = torch.tensor([case["prompt_token_ids"]], dtype=torch.long, device="cuda")
    result = {
        "sample_id": case["sample_id"],
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "transformers_origin": transformers.__file__,
        "target_repo": case["target_repo"],
        "target_revision": case["target_revision"],
        "eagle_repo": case["eagle_repo"],
        "eagle_revision": case["eagle_revision"],
        "prompt_tokens": len(case["prompt_token_ids"]),
        "settings": {"temperature": 0.0, "total_token": 16, "depth": 8, "top_k": 4},
    }

    try:
        target = AutoModelForCausalLM.from_pretrained(
            target_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        target.config.use_cache = True
        baseline = target.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        baseline_ids = [int(x) for x in baseline[0, input_ids.shape[1]:].tolist()]
        del baseline
        del target
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        sys.path.insert(0, "/workspace/EAGLE")
        from eagle.model.ea_model import EaModel

        eagle = EaModel.from_pretrained(
            base_model_path=target_path,
            ea_model_path=eagle_path,
            total_token=16,
            depth=8,
            top_k=4,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_eagle3=True,
        ).eval()
        eagle_inputs = input_ids.to(next(eagle.parameters()).device)
        eagle.eagenerate(
            eagle_inputs,
            temperature=0.0,
            max_new_tokens=min(64, max_new_tokens),
            log=True,
            return_phase_timings=True,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generated = eagle.eagenerate(
            eagle_inputs,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            log=True,
            return_phase_timings=True,
        )
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        full_ids, new_tokens, accepted, source_e2e_s, accept_lengths, phase = generated
        output_ids = [int(x) for x in full_ids[0, input_ids.shape[1]:].tolist()]
        result.update({
            "status": "success",
            "output_token_ids": output_ids,
            "text": tokenizer.decode(output_ids, skip_special_tokens=True),
            "wall_ms": wall_ms,
            "source_e2e_ms": float(source_e2e_s) * 1000.0,
            "new_tokens_counter": int(new_tokens),
            "accepted_counter": int(accepted),
            "acceptance_lengths": [int(x) for x in accept_lengths],
            "phase_timings": phase,
            "parity_vs_transformers_vanilla": _parity(output_ids, baseline_ids),
            "parity_vs_vllm_vanilla": _parity(
                output_ids, case["vllm_outputs"]["vanilla_vllm"]["token_ids"]
            ),
            "parity_vs_vllm_eagle3": _parity(
                output_ids, case["vllm_outputs"]["eagle3"]["token_ids"]
            ),
        })
    except Exception as exc:
        result.update({
            "status": "failed_or_unsupported",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    return json.dumps(result, ensure_ascii=False)


@app.local_entrypoint()
def save_eagle_reference(max_new_tokens: int = 512) -> None:
    """Run the EAGLE source probe and save its returned JSON locally."""
    output = ROOT / "outputs/modal_reference_compare/20260930T173501Z-cfee7aba/eagle_author_transformers453_final.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_eagle_reference.remote(max_new_tokens=max_new_tokens)
    output.write_text(payload + "\n", encoding="utf-8")
    record = json.loads(payload)
    print(json.dumps({
        "saved": str(output),
        "status": record.get("status"),
        "error": record.get("error"),
        "gpu": record.get("gpu"),
        "transformers_version": record.get("transformers_version"),
        "output_tokens": len(record.get("output_token_ids", [])),
        "parity_vs_vllm_vanilla": record.get("parity_vs_vllm_vanilla"),
        "parity_vs_vllm_eagle3": record.get("parity_vs_vllm_eagle3"),
        "acceptance_lengths": record.get("acceptance_lengths"),
    }, ensure_ascii=False), flush=True)
