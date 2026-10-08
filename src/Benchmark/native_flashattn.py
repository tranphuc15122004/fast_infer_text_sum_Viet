"""Native decoder profiles and measurement adapters; decoding loops stay vendored."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
from typing import Any, Mapping

FA4_VERSION = "4.0.0b32"
PHASE_TIMING_KEYS = ("prefill_ms", "decode_ms", "e2e_ms", "draft_latency_ms", "verification_latency_ms")


def native_source_manifest(root, methods):
    """Fingerprint vendored decoders/adapters so a resume cannot mix revisions."""
    paths = {"src/Benchmark/native_flashattn.py", "src/Benchmark/common/flashattn4_tree_attention.py",
             "src/Benchmark/common/flashattn_runtime.py", "src/Benchmark/common/vanilla_inference.py"}
    method_paths = {
        "dflash": ("externals/dflash/dflash/model.py", "src/Benchmark/dflash_compat.py",
                   "src/Benchmark/dflash_fa4_attention.py"),
        "eagle3": ("externals/EAGLE/eagle/model/ea_model.py", "externals/EAGLE/eagle/model/cnets.py",
                   "externals/EAGLE/eagle/model/modeling_qwen3_kv.py", "externals/EAGLE/eagle/model/utils.py",
                   "src/Benchmark/eagle_compat.py", "src/Benchmark/eagle_fa4_attention.py",
                   "src/Benchmark/eagle3_infer_qwen3.py"),
        "domino": ("externals/Domino/code/dflash.py", "externals/Domino/code/kernel/domino.py"),
        "dspark": ("externals/DeepSpec/deepspec/eval/dspark/evaluator.py",
                   "externals/DeepSpec/deepspec/modeling/dspark/qwen3/__init__.py",
                   "externals/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py",
                   "externals/DeepSpec/deepspec/modeling/dspark/qwen3/config.py"),
    }
    for method in methods:
        paths.update(method_paths.get(method, ()))
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in sorted(paths)}


@dataclass(frozen=True)
class NativeInferenceConfig:
    # Project native AR profile: one path with 16 proposals.
    eagle_total_token: int = 17
    eagle_depth: int = 16
    eagle_top_k: int = 1
    domino_cuda_graph: bool = True
    dspark_confidence_threshold: float = 0.0
    phase_timing_mode: str = "separate"
    strict_greedy_parity: bool = False
    require_speedup: bool = False

    def __post_init__(self):
        if self.eagle_total_token < 1 or self.eagle_depth < 0 or self.eagle_top_k < 1:
            raise ValueError("invalid EAGLE total-token/depth/top-k")
        capacity = 1 + self.eagle_top_k + self.eagle_depth * self.eagle_top_k ** 2
        if self.eagle_total_token > capacity:
            raise ValueError(f"EAGLE total-token exceeds tree capacity {capacity}")
        if not 0 <= self.dspark_confidence_threshold <= 1:
            raise ValueError("DSpark confidence threshold must be in [0, 1]")
        if self.phase_timing_mode not in {"separate", "inline", "off"}:
            raise ValueError("phase-timing-mode must be separate, inline or off")

    def eagle_tree(self):
        return {"total_token": self.eagle_total_token, "depth": self.eagle_depth, "top_k": self.eagle_top_k}

    def manifest(self):
        return asdict(self)


def validate_fa4_version(versions: Mapping[str, Any]) -> None:
    actual = versions.get("flash-attn-4")
    if actual != FA4_VERSION:
        raise ValueError(
            f"Native B200 profile requires flash-attn-4=={FA4_VERSION}; found {actual!r}. "
            "Install the pinned cu13 package from the offline wheelhouse."
        )


@dataclass
class TimingScopeStats:
    skipped_synchronizations: int = 0


class _CudaProfilingProxy:
    def __init__(self, original, stats):
        self._original, self._stats = original, stats

    def __getattr__(self, name):
        return getattr(self._original, name)

    def synchronize(self, *args, **kwargs):
        self._stats.skipped_synchronizations += 1


class _TorchProfilingProxy:
    def __init__(self, original, stats):
        self._original = original
        self.cuda = _CudaProfilingProxy(original.cuda, stats)

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextmanager
def native_timing_scope(method: str, context: Mapping[str, Any], *, synchronized: bool):
    """Omit module-local profiling barriers and always restore their functions.

    Global torch.cuda, tensor .item(), FA4 and graph capture are unchanged.
    Outer E2E still synchronizes before/after generation. This scope requires
    the sequential, single-threaded benchmark worker.
    """
    stats = TimingScopeStats()
    module = attribute = original = None
    if not synchronized:
        if method == "eagle3":
            module, attribute = context["eagle_model_module"], "torch"
            original = module.torch
            replacement = _TorchProfilingProxy(original, stats)
        elif method in {"dflash", "domino"}:
            module = context[f"{method}_model_module"]
            attribute = "_cuda_time" if method == "dflash" else "cuda_time"
            original = getattr(module, attribute)

            def replacement(*args, **kwargs):
                stats.skipped_synchronizations += 1
                return time.perf_counter()
        if module is not None:
            setattr(module, attribute, replacement)
    try:
        yield stats
    finally:
        if module is not None:
            setattr(module, attribute, original)


def run_native_method(torch, method, context, input_ids, *, max_new_tokens, profiling=False):
    """Dispatch directly to the native entrypoints without copying their loops."""
    from transformers import set_seed

    set_seed(int(context.get("seed", 42)))
    options = context.get("native_config", NativeInferenceConfig())
    synchronized = profiling or options.phase_timing_mode == "inline"
    with native_timing_scope(method, context, synchronized=synchronized) as stats:
        result = _dispatch_native_method(torch, method, context, input_ids, max_new_tokens)
    context["last_skipped_profiling_synchronizations"] = stats.skipped_synchronizations
    return result


def _dispatch_native_method(torch, method, context, input_ids, max_new_tokens):
    target = context["target"]
    stop_ids = context["stop_token_ids"] or None
    if method == "vanilla_hf":
        return target.generate(input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            eos_token_id=stop_ids, pad_token_id=context["tokenizer"].pad_token_id)
    if method == "dflash":
        return context["dflash_generate"](context["draft"], target=target, input_ids=input_ids,
            max_new_tokens=max_new_tokens, stop_token_ids=stop_ids,
            temperature=0.0, block_size=16, return_stats=True)
    if method == "domino":
        return context["draft"].spec_generate(input_ids, target=target,
            max_new_tokens=max_new_tokens, temperature=0.0, stop_token_ids=stop_ids,
            block_size=context["draft"].block_size, use_bias=True, return_dict=True,
            graph_runner=context.get("domino_graph_runner"))
    if method == "dspark":
        context["evaluator"].args.max_new_tokens = int(max_new_tokens)
        return context["evaluator"].generate_one_sample(input_ids=input_ids, stop_token_ids=stop_ids)
    if method == "eagle3":
        from Benchmark.eagle3_infer_qwen3 import timed_generate
        return timed_generate(context["eagle_model"], input_ids, temperature=0.0,
            max_new_tokens=max_new_tokens, total_token=int(context["eagle_tree"]["total_token"]),
            spec=True, is_llama3=False, include_phase_timings=True, stop_token_ids=stop_ids)
    raise ValueError(f"unknown native method {method!r}")


def prepare_measured_payload(payload, *, first_forward_ms, mode):
    measured = {**payload, "phases": dict(payload.get("phases", {}))}
    if mode != "inline":
        measured["ttft_ms"] = first_forward_ms
        for key in PHASE_TIMING_KEYS:
            measured["phases"][key] = None
    return measured


def attach_profile_phases(measured, profile):
    keys = ("output_ids", "acceptance_lengths", "draft_tokens_accepted",
            "draft_tokens_proposed", "verification_steps")
    matched = all(measured.get(key) == profile.get(key) for key in keys)
    metadata = {"source": "separate_generation", "output_and_acceptance_match": matched,
                "profiling_e2e_ms": profile.get("elapsed_ms"), "profiling_ttft_ms": profile.get("ttft_ms")}
    if matched:
        for key in ("draft_latency_ms", "verification_latency_ms"):
            measured["phases"][key] = profile.get("phases", {}).get(key)
    return metadata


def collect_native_phase_profile(options, method, measured, profile_call):
    """Profile after measurement; unsupported native phase timers remain null."""
    if options.phase_timing_mode != "separate":
        return {"source": "inline_generation" if options.phase_timing_mode == "inline" else "disabled"}
    if method not in {"dflash", "eagle3"}:
        return {"source": "unavailable_native_phase_timers"}
    return attach_profile_phases(measured, profile_call())


def initialize_eagle_request_cache(model, max_length):
    """Scope cache aliases to this call so the benchmark loop can release EAGLE."""
    from eagle.model.kv_cache import initialize_past_key_values

    cache, storage, length = initialize_past_key_values(model.base_model, max_length=max_length)
    model.past_key_values = cache
    model.past_key_values_data = storage
    model.current_length_data = length


def build_domino_graph_runner(draft, target, device, *, factory=None, source_path=None):
    """Use the same correction graph and dimensions as Domino's HF benchmark."""
    if factory is None:
        source_path = source_path or Path(__file__).resolve().parents[2] / "externals/Domino/code/kernel/domino.py"
        name = "_fast_infer_native_domino_graph"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, source_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load native Domino graph: {source_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(name, None)
                raise
        factory = sys.modules[name].DraftCorrectionGraphRunner
    shift_label = bool(getattr(draft.config, "dflash_config", {}).get("shift_label", False))
    prefix_len = int(getattr(draft, "pure_draft_prefix_len", 0))
    proposal_size = int(draft.block_size) if shift_label else int(draft.block_size) - 1
    if not 0 <= prefix_len < proposal_size:
        raise ValueError("Domino CUDA Graph requires a non-empty correction suffix")
    return factory(draft_model=draft, target_model=target, batch_size=1,
        steps=proposal_size - prefix_len, hidden_dim=int(target.lm_head.weight.shape[1]),
        gru_hidden_dim=int(draft.prefix_gru.hidden_size), vocab_size=int(target.lm_head.weight.shape[0]),
        prefix_token_count=1 + prefix_len, device=device)


def native_run_status(options, *, runtime_pass, execution_complete, failure_count,
                      quality_pass, exact_match_all, speedup_all_over_one, schema_valid):
    if not runtime_pass or not execution_complete or failure_count:
        return "runtime_failure"
    if not quality_pass:
        return "quality_failure"
    if not schema_valid:
        return "schema_failure"
    if options.strict_greedy_parity and not exact_match_all:
        return "greedy_parity_failure"
    if options.require_speedup and not speedup_all_over_one:
        return "speedup_not_above_one"
    return "success"
