#!/usr/bin/env python3
"""Paired two-prompt GPU pilot for the common vLLM 0.30 backend.

Run with ``modal run scripts/modal_vllm_pilot.py --max-new-tokens 512``.
Each method loads the same Qwen3-4B target on one Modal GPU, uses greedy
decoding, disables prefix caching, and leaves vLLM CUDA Graphs enabled by
default. Results are returned and written under ignored ``outputs/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import uuid

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum_Viet")
REMOTE_SRC = REMOTE_ROOT / "src"
REMOTE_DATA = REMOTE_ROOT / "outputs" / "modal_vllm_smoke_input" / "vietnews_2.jsonl"
GPU = os.environ.get("MODAL_GPU", "H100")

EAGLE3_MODEL_REPO = os.environ.get(
    "MODAL_EAGLE3_MODEL_REPO", "AngelSlim/Qwen3-4B_eagle3"
)
MODEL_REPOS = {
    "vanilla_vllm": "Qwen/Qwen3-4B",
    # This Eagle3 checkpoint uses an architecture supported by vLLM 0.30 and
    # preserves the K=16 setup used by the synchronized comparison.
    "eagle3": EAGLE3_MODEL_REPO,
    "dflash": "z-lab/Qwen3-4B-DFlash-b16",
    "domino": "Huang2020/Qwen3-4B-Domino-b16",
    "dspark": "deepseek-ai/dspark_qwen3_4b_block7",
}
METHODS = tuple(MODEL_REPOS)
MAX_INPUT_TOKENS = 8192
MAX_MODEL_LEN = 12288

app = modal.App("fast-infer-viet-vllm-pilot")
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12")
    .entrypoint([])
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path=str(REMOTE_SRC), copy=True)
    .add_local_file(str(PROJECT_ROOT / "outputs/modal_vllm_smoke_input/vietnews_2.jsonl"), remote_path=str(REMOTE_DATA), copy=True)
    .env(
        {
            "PYTHONPATH": str(REMOTE_SRC),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "MODAL_EAGLE3_MODEL_REPO": EAGLE3_MODEL_REPO,
        }
    )
)


def _candidate_site_packages() -> list[str]:
    import glob

    patterns = (
        "/usr/local/lib/python*/site-packages",
        "/usr/local/lib/python*/dist-packages",
        "/usr/lib/python*/site-packages",
        "/usr/lib/python*/dist-packages",
        "/opt/venv/lib/python*/site-packages",
        "/opt/conda/lib/python*/site-packages",
        "/root/.venv/lib/python*/site-packages",
        "/venv/lib/python*/site-packages",
    )
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    return [
        path
        for path in paths
        if (Path(path) / "torch" / "__init__.py").is_file()
        or (Path(path) / "vllm" / "__init__.py").is_file()
    ]


def _prepare_vllm_runtime() -> dict:
    import importlib.util
    import sys
    import typing_extensions

    sites = _candidate_site_packages()
    sys.path[:0] = [path for path in sites if path not in sys.path]
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [*sites, *[part for part in inherited_pythonpath.split(os.pathsep) if part]]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_parts))
    site_module = next(
        (Path(path) / "typing_extensions.py" for path in sites
         if (Path(path) / "typing_extensions.py").is_file()),
        None,
    )
    if not hasattr(typing_extensions, "Sentinel") and site_module is not None:
        spec = importlib.util.spec_from_file_location("typing_extensions", site_module)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load image typing_extensions from {site_module}")
        replacement = importlib.util.module_from_spec(spec)
        sys.modules["typing_extensions"] = replacement
        spec.loader.exec_module(replacement)
        typing_extensions = replacement
    # Modal preloads protobuf 6.31.1, while the vLLM image includes matching
    # protobuf 6.33.6 and generated code. Clear the preloaded modules before
    # vLLM imports and force Python protobuf to avoid Modal's mismatched upb .so.
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    for name in list(sys.modules):
        if name == "google" or name.startswith("google."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return {
        "site_packages": sites,
        "subprocess_pythonpath": os.environ.get("PYTHONPATH"),
        "typing_extensions_path": getattr(typing_extensions, "__file__", None),
        "typing_extensions_has_sentinel": hasattr(typing_extensions, "Sentinel"),
    }


@app.function(image=image, cpu=2, timeout=300)
def probe_runtime() -> dict:
    import importlib
    import importlib.util
    import importlib.metadata
    import sys
    import traceback

    runtime = _prepare_vllm_runtime()
    packages = {}
    for name in ("torch", "vllm", "transformers", "huggingface_hub"):
        try:
            module = importlib.import_module(name)
            packages[name] = {
                "origin": getattr(module, "__file__", None),
                "version": getattr(module, "__version__", None),
            }
        except Exception as exc:
            packages[name] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import google
        import google.protobuf
        from google.protobuf import message as protobuf_message

        packages["protobuf_diagnostics"] = {
            "google_paths": list(getattr(google, "__path__", [])),
            "protobuf_path": getattr(google.protobuf, "__file__", None),
            "message_path": getattr(protobuf_message, "__file__", None),
            "protobuf_version": getattr(google.protobuf, "__version__", None),
            "protobuf_implementation": os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
            "distribution_version": importlib.metadata.version("protobuf"),
            "modal_paths": [path for path in sys.path if "modal" in path.lower()],
            "protobuf_modules": {
                name: getattr(module, "__file__", None)
                for name, module in sys.modules.items()
                if name.startswith("google.") and name.count(".") <= 2
            },
        }
    except Exception as exc:
        packages["protobuf_diagnostics"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        from vllm import LLM, SamplingParams

        plugins = importlib.metadata.entry_points(group="vllm.general_plugins")
        packages["vllm_llm_api"] = {
            "available": True,
            "domino_plugin_discovered": any(
                plugin.name == "fast_infer_viet_domino" for plugin in plugins
            ),
        }
    except Exception as exc:
        packages["vllm_llm_api"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    return {
        "python": sys.executable,
        "version": sys.version,
        **runtime,
        "packages": packages,
    }


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)



@app.function(image=image, gpu=GPU, cpu=8, memory=32768, timeout=3600, max_containers=1)
def run_pilot(
    max_new_tokens: int = 512,
    method_names: str = "vanilla_vllm,eagle3,dflash,domino,dspark",
    enforce_eager: bool = False,
) -> dict:
    """Run selected methods sequentially on the same GPU and vLLM release."""

    import gc
    import re
    import sys
    from dataclasses import asdict, is_dataclass

    # DSpark is supported only by vLLM's V2 runner, and Domino's compatibility
    # patch hooks the V2 DFlashSpeculator. Set this before importing vLLM so the
    # worker and the driver agree on the runner implementation.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

    runtime = _prepare_vllm_runtime()
    print(f"[runtime] python={sys.executable}; runtime={runtime}", flush=True)
    import torch
    import transformers
    import vllm
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    from Benchmark.common.benchmark_data import read_jsonl, render_prompt
    from Benchmark.common.prompt_format import format_chat_prompt
    from Benchmark.common.quality_guard import is_degenerate_output
    from Benchmark.common.rouge import add_rouge
    from Benchmark.common.vllm_pilot import (
        install_domino_vllm_compat,
        paired_vllm_metrics,
        pilot_warmup_tokens,
        resolve_eagle3_aux_hidden_state_layers,
        select_pilot_methods,
        speculative_token_count,
    )

    selected_methods = select_pilot_methods(
        method_names, available_methods=METHODS, reference="vanilla_vllm"
    )

    if max_new_tokens < 64:
        raise ValueError("max_new_tokens must be at least 64 for the quality pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal container did not expose a CUDA GPU")

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    run_root = Path("/tmp/fast-infer-vllm-pilot") / run_id
    cache_dir = run_root / "hf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "90"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    api = HfApi()
    checkpoints = {}
    paths = {}
    for role in selected_methods:
        repo_id = MODEL_REPOS[role]
        info = api.model_info(repo_id, revision="main")
        revision = info.sha
        snapshot = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir / "hub"),
            local_files_only=False,
        )
        snapshot_path = Path(snapshot)
        if not (snapshot_path / "config.json").is_file():
            raise RuntimeError(f"checkpoint config missing: {repo_id}@{revision}")
        if not any(snapshot_path.glob("*.safetensors")):
            raise RuntimeError(f"safetensors weights missing: {repo_id}@{revision}")
        paths[role] = str(snapshot_path)
        checkpoints[role] = {"repo_id": repo_id, "revision": revision}

    tokenizer = AutoTokenizer.from_pretrained(paths["vanilla_vllm"], local_files_only=True)
    target_config = AutoConfig.from_pretrained(
        paths["vanilla_vllm"], local_files_only=True
    )
    target_num_hidden_layers = int(
        getattr(target_config, "num_hidden_layers", 0) or 0
    )
    samples = read_jsonl(REMOTE_DATA)
    candidates = []
    for row in samples:
        prompt = format_chat_prompt(tokenizer, render_prompt(row))
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if 0 < len(token_ids) <= MAX_INPUT_TOKENS:
            candidates.append((len(token_ids), row, prompt, token_ids))
    candidates.sort(key=lambda item: (item[0], str(item[1].get("id", ""))))
    if len(candidates) < 2:
        raise RuntimeError(f"only {len(candidates)} VietNews examples fit the input limit")
    selected = []
    for quantile in (0.50, 0.90):
        index = round(quantile * (len(candidates) - 1))
        candidate = candidates[index]
        if all(candidate[1].get("id") != old[1].get("id") for old in selected):
            selected.append(candidate)
    if len(selected) < 2:
        selected.append(candidates[-1])
    prompt_rows = [
        {
            "sample_id": str(row.get("id")),
            "prompt": prompt,
            "prompt_token_ids": [int(token) for token in token_ids],
            "reference": str(row.get("reference") or row.get("answers", [""])[0]),
            "document_words": int(row.get("document_words", 0) or 0),
            "input_tokens": len(token_ids),
            "length_quantile": quantile,
        }
        for (quantile, (input_tokens, row, prompt, token_ids)) in zip(
            (0.50, 0.90), selected
        )
    ]

    records = []
    model_configs = {}
    device_name = torch.cuda.get_device_name(0)
    for method in selected_methods:
        engine = None
        method_records = []
        start_load = time.perf_counter()
        try:
            spec_config = None
            speculative_tokens = None
            if method != "vanilla_vllm":
                draft_config = AutoConfig.from_pretrained(
                    paths[method], local_files_only=True, trust_remote_code=False
                )
                k = speculative_token_count(method, draft_config)
                speculative_tokens = k
                spec_method = "dflash" if method == "domino" else method
                spec_config = {
                    "method": spec_method,
                    "model": paths[method],
                    "num_speculative_tokens": k,
                }
                raw_target_layer_ids = getattr(draft_config, "target_layer_ids", None)
                raw_eagle_aux_layer_ids = getattr(
                    draft_config, "eagle_aux_hidden_state_layer_ids", None
                )
                eagle_aux_layer_ids = (
                    resolve_eagle3_aux_hidden_state_layers(
                        draft_config,
                        target_num_hidden_layers=target_num_hidden_layers,
                    )
                    if method == "eagle3"
                    else None
                )
                draft_dflash_config = getattr(draft_config, "dflash_config", {}) or {}
                model_configs[method] = {
                    "vllm_method": spec_method,
                    "num_speculative_tokens": k,
                    "draft_architectures": list(
                        getattr(draft_config, "architectures", []) or []
                    ),
                    "target_model_name_or_path": getattr(
                        draft_config, "target_model_name_or_path", None
                    ),
                    "target_layer_ids": _jsonable(raw_target_layer_ids),
                    "eagle_aux_hidden_state_layer_ids": _jsonable(
                        raw_eagle_aux_layer_ids
                    ),
                    "resolved_eagle_aux_hidden_state_layer_ids": (
                        _jsonable(eagle_aux_layer_ids) if method == "eagle3" else None
                    ),
                    "target_num_hidden_layers": target_num_hidden_layers,
                    "ttt_length": getattr(draft_config, "ttt_length", None),
                    "projector_type": (
                        draft_dflash_config.get("projector_type")
                        if isinstance(draft_dflash_config, dict)
                        else None
                    ),
                    "dflash_config": _jsonable(draft_dflash_config),
                }

            if method == "domino":
                install_domino_vllm_compat()
            engine_kwargs = {
                "model": paths["vanilla_vllm"],
                "tokenizer": paths["vanilla_vllm"],
                "dtype": "bfloat16",
                "trust_remote_code": False,
                "disable_log_stats": False,
                "enforce_eager": enforce_eager,
                "enable_prefix_caching": False,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.88,
                "seed": 42,
                "speculative_config": spec_config,
            }
            if spec_config is not None:
                engine_kwargs["per_request_spec_decode_metrics"] = "detailed"
            engine = LLM(**engine_kwargs)
            load_ms = (time.perf_counter() - start_load) * 1000.0
            sampling = SamplingParams(
                temperature=0.0,
                seed=42,
                max_tokens=max_new_tokens,
                n=1,
            )
            warmup_sampling = SamplingParams(
                temperature=0.0,
                seed=42,
                max_tokens=pilot_warmup_tokens(speculative_tokens),
                n=1,
                ignore_eos=True,
            )
            # Exercise each measured prompt before timing so prompt-shape and
            # speculative Triton JIT costs do not land on the first sample.
            for sample in prompt_rows:
                engine.generate(
                    [sample["prompt"]], warmup_sampling, use_tqdm=False
                )
            torch.cuda.reset_peak_memory_stats()

            for sample in prompt_rows:
                request_started = time.perf_counter()
                request_output = engine.generate(
                    [sample["prompt"]], sampling, use_tqdm=False
                )[0]
                client_wall_ms = (time.perf_counter() - request_started) * 1000.0
                completion = request_output.outputs[0]
                token_ids = [int(token) for token in completion.token_ids]
                text = str(completion.text or "")
                request_metrics = request_output.metrics
                raw_metrics = _jsonable(
                    asdict(request_metrics)
                    if request_metrics is not None and is_dataclass(request_metrics)
                    else request_metrics
                )
                if request_metrics is None:
                    timing = {}
                else:
                    scheduled_ts = float(request_metrics.scheduled_ts)
                    first_ts = float(request_metrics.first_token_ts)
                    last_ts = float(request_metrics.last_token_ts)
                    queued_ts = float(request_metrics.queued_ts)
                    timing = {
                        "queue_ms": max(0.0, scheduled_ts - queued_ts) * 1000.0,
                        "prefill_ms": max(0.0, first_ts - scheduled_ts) * 1000.0,
                        "ttft_ms": max(0.0, float(request_metrics.first_token_latency)) * 1000.0,
                        "decode_ms": max(0.0, last_ts - first_ts) * 1000.0,
                        # arrival_time is wall-clock time while these other
                        # timestamps are monotonic. Use queued_ts as the
                        # comparable start point to avoid a bogus zero duration.
                        "e2e_ms": max(0.0, last_ts - queued_ts) * 1000.0,
                        "mean_itl_ms": (
                            max(0.0, last_ts - first_ts) * 1000.0 / (len(token_ids) - 1)
                            if len(token_ids) > 1
                            else None
                        ),
                    }
                decoded_ids = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                normalize = lambda value: " ".join(str(value).split())
                decode_matches = normalize(decoded_ids) == normalize(text)
                finish_reason = str(completion.finish_reason or "")
                repeated = is_degenerate_output(text)
                nonempty = bool(text.strip()) and bool(token_ids)
                metrics_complete = bool(
                    request_metrics is not None
                    and timing.get("prefill_ms", 0.0) > 0
                    and timing.get("mean_itl_ms") is not None
                    and timing.get("mean_itl_ms", 0.0) > 0
                )
                input_token_ids = [
                    int(token) for token in (request_output.prompt_token_ids or [])
                ]
                prompt_tokens_match = input_token_ids == sample["prompt_token_ids"]
                reported_generation_tokens = (
                    getattr(request_metrics, "num_generation_tokens", None)
                    if request_metrics is not None
                    else None
                )
                generation_tokens_match = bool(
                    request_metrics is not None
                    and (
                        reported_generation_tokens is None
                        or int(reported_generation_tokens) == len(token_ids)
                    )
                )
                valid = (
                    nonempty
                    and decode_matches
                    and prompt_tokens_match
                    and generation_tokens_match
                    and not repeated
                    and finish_reason in {"stop", "length"}
                    and metrics_complete
                )
                record = {
                    "method": method,
                    "dataset": "vietnews",
                    "sample_id": sample["sample_id"],
                    "status": "success" if valid else "invalid",
                    "text": text,
                    "reference": sample["reference"],
                    "prompt_token_ids": sample["prompt_token_ids"],
                    "vllm_prompt_token_ids": input_token_ids,
                    "output_token_ids": token_ids,
                    "input_tokens": len(input_token_ids),
                    "output_tokens": len(token_ids),
                    "finish_reason": finish_reason,
                    "text_decode_matches_token_ids": decode_matches,
                    "prompt_tokens_match_tokenizer": prompt_tokens_match,
                    "generation_tokens_match_token_ids": generation_tokens_match,
                    "nonempty_output": nonempty,
                    "repetition_flag": repeated,
                    "unique_word_ratio": (
                        len(set(re.findall(r"\S+", text)))
                        / max(1, len(re.findall(r"\S+", text)))
                    ),
                    "task_type": "summarization",
                    "temperature": 0.0,
                    "seed": 42,
                    "max_new_tokens": max_new_tokens,
                    "max_input_tokens": MAX_INPUT_TOKENS,
                    "client_wall_ms": client_wall_ms,
                    **timing,
                    "raw_request_metrics": raw_metrics,
                    "speculative_decoding_metrics": _jsonable(
                        completion.spec_decode_metrics
                    ),
                }
                add_rouge(record, text, sample["reference"])
                method_records.append(record)
                records.append(record)
            model_configs.setdefault(method, {})["load_ms"] = load_ms
            # vLLM owns CUDA in its EngineCore subprocess, so the parent
            # process allocator reports 0 and is not a valid per-method peak.
            model_configs[method]["peak_memory_gb"] = None
            model_configs[method]["status"] = (
                "success" if all(row["status"] == "success" for row in method_records) else "invalid"
            )
        except Exception as exc:
            model_configs.setdefault(method, {})["status"] = "failed"
            model_configs[method]["error"] = f"{type(exc).__name__}: {exc}"
            for sample in prompt_rows:
                if any(
                    row["method"] == method and row["sample_id"] == sample["sample_id"]
                    for row in method_records
                ):
                    continue
                record = {
                    "method": method,
                    "dataset": "vietnews",
                    "sample_id": sample["sample_id"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "text": "",
                    "reference": sample["reference"],
                    "input_tokens": sample["input_tokens"],
                    "output_tokens": 0,
                    "prompt_token_ids": sample["prompt_token_ids"],
                    "output_token_ids": [],
                    "repetition_flag": None,
                }
                records.append(record)
                method_records.append(record)
        finally:
            if engine is not None:
                try:
                    engine.llm_engine.engine_core.shutdown(timeout=30)
                except Exception:
                    pass
                del engine
                gc.collect()
                torch.cuda.empty_cache()

    try:
        metric_rows = paired_vllm_metrics(records, reference="vanilla_vllm")
        metric_error = None
    except Exception as exc:
        metric_rows = {}
        metric_error = f"{type(exc).__name__}: {exc}"
    expected = len(selected_methods) * len(prompt_rows)
    successful = sum(row.get("status") == "success" for row in records)
    return {
        "run_id": run_id,
        "status": "passed" if successful == expected and not metric_error else "failed",
        "validation": {
            "expected_records": expected,
            "observed_records": len(records),
            "successful_records": successful,
            "invalid_or_failed_records": expected - successful,
            "metric_error": metric_error,
        },
        "gpu": device_name,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "runtime_contract": {
            "backend": "vllm",
            "model_runner": "v2",
            "use_v2_model_runner": True,
            "methods": list(selected_methods),
            "target_repo": MODEL_REPOS["vanilla_vllm"],
            "dtype": "bfloat16",
            "temperature": 0.0,
            "seed": 42,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": False,
            "max_num_seqs": 1,
            "max_new_tokens": max_new_tokens,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "warmup_requests_per_method": len(prompt_rows),
        },
        "methods": list(selected_methods),
        "checkpoints": checkpoints,
        "model_configs": model_configs,
        "samples": [
            {key: value for key, value in sample.items() if key != "prompt"}
            for sample in prompt_rows
        ],
        "paired_metrics": metric_rows,
        "records": records,
    }


def _write_local_artifacts(result: dict, run_id: str) -> Path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from Benchmark.common.io_util import JsonlWriter
    from Benchmark.common.rouge import aggregate_rouge
    from Benchmark.common.vllm_pilot import paired_vllm_metrics

    output_dir = PROJECT_ROOT / "outputs" / "modal_vllm_smoke" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    records = result.get("records", [])
    methods = tuple(result.get("methods") or METHODS)
    metric_error = None
    try:
        metric_rows = paired_vllm_metrics(records, reference="vanilla_vllm")
    except Exception as exc:
        metric_rows = {}
        metric_error = f"{type(exc).__name__}: {exc}"
    result["paired_metrics"] = metric_rows
    validation = result.setdefault("validation", {})
    validation["metric_error"] = metric_error
    reference_by_id = {
        row.get("sample_id"): row
        for row in records
        if row.get("method") == "vanilla_vllm" and row.get("status") == "success"
    }
    def _lcs_length(left: list[int], right: list[int]) -> int:
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

    reference_token_total = sum(
        len(row.get("output_token_ids") or []) for row in reference_by_id.values()
    )
    token_parity = {}
    token_lcs_overlap = {}
    for method in methods:
        method_rows = [
            row for row in records
            if row.get("method") == method and row.get("status") == "success"
        ]
        matches = sum(
            row.get("output_token_ids") == reference_by_id.get(row.get("sample_id"), {}).get("output_token_ids")
            for row in method_rows
        )
        lcs_tokens = sum(
            _lcs_length(
                reference_by_id[row["sample_id"]].get("output_token_ids") or [],
                row.get("output_token_ids") or [],
            )
            for row in method_rows
            if row.get("sample_id") in reference_by_id
        )
        token_parity[method] = {"matched_samples": matches, "total_samples": len(method_rows)}
        token_lcs_overlap[method] = {
            "lcs_tokens": lcs_tokens,
            "reference_tokens": reference_token_total,
            "overlap_percent": (
                100.0 * lcs_tokens / reference_token_total if reference_token_total else None
            ),
        }
    validation["greedy_token_parity_with_vanilla"] = token_parity
    validation["token_lcs_overlap_with_vanilla"] = token_lcs_overlap

    writer = JsonlWriter(output_dir / "results.jsonl")
    for record in records:
        writer.add(record)
    writer.finalize(
        {
            "type": "summary",
            "run_id": run_id,
            "status": result.get("status"),
            "record_count": len(records),
            "validation": result.get("validation"),
            "paired_metrics": metric_rows,
        }
    )

    report_json = {key: value for key, value in result.items() if key != "records"}
    (output_dir / "run_report.json").write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Báo cáo pilot vLLM chung — {run_id}",
        "",
        f"- Trạng thái: **{result.get('status', 'failed')}**",
        f"- GPU: {result.get('gpu', 'unknown')}",
        f"- vLLM: {result.get('vllm_version', 'unknown')}",
        f"- Coverage: {result.get('validation', {}).get('successful_records', 0)}/"
        f"{result.get('validation', {}).get('expected_records', 0)} output hợp lệ",
        "- Hai mẫu VietNews được chọn theo phân vị 50% và 90% độ dài prompt sau tokenization.",
        "- Mỗi method dùng cùng target, dtype BF16, prompt, greedy decode, giới hạn token,"
        " một GPU; `enforce_eager`="
        f"`{result.get('runtime_contract', {}).get('enforce_eager')}`, không prefix cache;"
        f" mỗi engine warm-up trên {result.get('runtime_contract', {}).get('warmup_requests_per_method', 1)} prompt không tính điểm.",
        "- TPOT lấy từ khoảng cách giữa các token sinh của RequestOutput vLLM; prefill lấy"
        " từ mốc schedule đến token đầu tiên. E2E dùng queued_ts đến last_token_ts vì"
        " arrival_time là wall-clock còn các mốc còn lại là monotonic. Startup/warm-up bỏ ra.",
        "- DSR = TPOT_vanilla_vllm / TPOT_method. ESR = (P_common + TPOT_vanilla ×"
        " mean(L_min)) / (P_common + TPOT_method × mean(L_min)).",
        "- Token overlap tính LCS trên token IDs, chia cho tổng output token của vanilla; "
        "parity là số mẫu khớp toàn bộ chuỗi token.",
        "",
        "| Baseline | Trạng thái | Output hợp lệ | Lặp | LCS trùng vanilla | Chuỗi khớp | ROUGE-L | Prefill (ms) | TPOT (ms/token) | E2E (ms) | DSR | ESR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        rows = [row for row in records if row.get("method") == method]
        config = result.get("model_configs", {}).get(method, {})
        valid = sum(row.get("status") == "success" for row in rows)
        repeats = sum(bool(row.get("repetition_flag")) for row in rows)
        rouge = aggregate_rouge(rows).get("rougeL")
        tpot_values = [float(row["mean_itl_ms"]) for row in rows if row.get("mean_itl_ms")]
        tpot = sum(tpot_values) / len(tpot_values) if tpot_values else None
        prefill_values = [float(row["prefill_ms"]) for row in rows if row.get("prefill_ms")]
        prefill = sum(prefill_values) / len(prefill_values) if prefill_values else None
        e2e_values = [float(row["e2e_ms"]) for row in rows if row.get("e2e_ms")]
        e2e = sum(e2e_values) / len(e2e_values) if e2e_values else None
        metrics = metric_rows.get(method, {})
        parity = token_parity.get(method, {})
        parity_text = f"{parity.get('matched_samples', 0)}/{parity.get('total_samples', 0)}"
        overlap = token_lcs_overlap.get(method, {})
        overlap_text = (
            "—" if overlap.get("overlap_percent") is None
            else f"{overlap['overlap_percent']:.2f}%"
        )
        def fmt(value):
            return "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {method} | {config.get('status', 'failed')} | {valid}/{len(rows)} | "
            f"{repeats} | {overlap_text} | {parity_text} | {fmt(rouge)} | {fmt(prefill)} | "
            f"{fmt(tpot)} | {fmt(e2e)} | "
            f"{fmt(metrics.get('dsr'))} | {fmt(metrics.get('esr'))} |"
        )
    if metric_error:
        lines.extend(["", f"Metric chưa tính được: `{metric_error}`"])
    lines.extend(["", "## Output đã sinh", ""])
    for sample in result.get("samples", []):
        lines.extend([f"### {sample.get('sample_id')}", ""])
        for method in methods:
            row = next(
                (
                    item
                    for item in records
                    if item.get("method") == method
                    and item.get("sample_id") == sample.get("sample_id")
                ),
                None,
            )
            lines.extend([f"**{method}** — {row.get('status') if row else 'missing'}", ""])
            if row and row.get("text"):
                lines.extend([row["text"].strip(), ""])
            elif row and row.get("error"):
                lines.extend([f"Lỗi: `{row['error']}`", ""])
    lines.extend(
        [
            "## Giới hạn",
            "",
            "Hai mẫu là pilot kiểm thử, không đủ để kết luận hiệu năng toàn benchmark."
            " Kết quả dùng snapshot public trên Modal, không khẳng định hash trùng với"
            " model/cache local của lần chạy B200. Domino đi qua adapter dựa trên DFlash"
            " trong vLLM 0.30; cần giữ acceptance trace và kiểm thử parity rộng hơn trước"
            " khi dùng số liệu Domino cho báo cáo chính. Peak VRAM theo từng method không"
            " đo được ở process cha vì vLLM chạy EngineCore trong process con; trường này"
            " được để null, không diễn giải thành 0 GiB.",
            "",
            "Metric raw, token IDs, request timestamps, speculative acceptance và revision"
            " của checkpoint được lưu trong `results.jsonl` và `run_report.json`.",
        ]
    )
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


@app.local_entrypoint()
def main(
    max_new_tokens: int = 512,
    run_id: str = "",
    probe_only: bool = False,
    method_names: str = "vanilla_vllm,eagle3,dflash,domino,dspark",
    enforce_eager: bool = False,
) -> None:
    selected_run_id = run_id or (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    )
    if probe_only:
        print(json.dumps(probe_runtime.remote(), indent=2))
        return
    try:
        result = run_pilot.remote(
            max_new_tokens=max_new_tokens,
            method_names=method_names,
            enforce_eager=enforce_eager,
        )
    except Exception as exc:
        result = {
            "run_id": selected_run_id,
            "status": "failed",
            "validation": {"metric_error": f"{type(exc).__name__}: {exc}"},
            "records": [],
            "model_configs": {},
            "samples": [],
        }
    actual_run_id = str(result.get("run_id") or selected_run_id)
    output_dir = _write_local_artifacts(result, actual_run_id)
    print(f"Modal vLLM pilot: {result.get('status')}")
    print(f"Artifacts: {output_dir}")
    if result.get("status") != "passed":
        raise SystemExit(1)
