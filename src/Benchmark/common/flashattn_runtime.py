"""Runtime guards for the native Transformers FlashAttention pilot."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from functools import wraps
from typing import Any


EXPECTED_ATTENTION_BACKEND = "flash_attention_4"
SUPPORTED_METHODS = ("vanilla_hf", "eagle3", "dflash", "domino", "dspark")
_ACTIVE_DISPATCH_TRACKER: ContextVar["AttentionDispatchTracker | None"] = ContextVar(
    "fast_infer_attention_dispatch_tracker", default=None
)
_TRACKED_REGISTRY_BACKENDS = (
    EXPECTED_ATTENTION_BACKEND,
    "flash_attention_2",
    "sdpa",
    "eager",
    "flex_attention",
)
_CUDA_CONTEXT_FAILURE_MARKERS = (
    "illegal memory access",
    "unspecified launch failure",
    "device-side assert",
)


def is_cuda_context_failure(error: BaseException) -> bool:
    """Identify asynchronous CUDA errors that make later GPU work unreliable."""

    message = str(error).lower()
    return any(marker in message for marker in _CUDA_CONTEXT_FAILURE_MARKERS)


class AttentionDispatchTracker:
    """Attribute Transformers attention-registry calls to target or draft models."""

    def __init__(self, target_model: Any, draft_model: Any | None = None) -> None:
        self._module_roles: dict[int, str] = {}
        for module in target_model.modules():
            self._module_roles[id(module)] = "target"
        if draft_model is not None:
            for module in draft_model.modules():
                self._module_roles.setdefault(id(module), "draft")
        self.reset()

    def reset(self) -> None:
        self._calls = {
            "target": {backend: 0 for backend in _TRACKED_REGISTRY_BACKENDS},
            "draft": {backend: 0 for backend in _TRACKED_REGISTRY_BACKENDS},
        }
        self._unattributed_calls = {backend: 0 for backend in _TRACKED_REGISTRY_BACKENDS}

    @contextmanager
    def recording(self):
        """Record attention calls made inside this context only."""

        token = _ACTIVE_DISPATCH_TRACKER.set(self)
        try:
            yield self
        finally:
            _ACTIVE_DISPATCH_TRACKER.reset(token)

    def record(
        self,
        backend: str,
        *,
        module: Any | None = None,
        role: str | None = None,
    ) -> None:
        if backend not in _TRACKED_REGISTRY_BACKENDS:
            return
        attributed_role = role or self._module_roles.get(id(module))
        if attributed_role not in self._calls:
            self._unattributed_calls[backend] += 1
            return
        self._calls[attributed_role][backend] += 1

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for role in ("target", "draft"):
            calls = self._calls[role]
            fa4_calls = int(calls[EXPECTED_ATTENTION_BACKEND])
            fallback_backends = {
                backend: int(count)
                for backend, count in calls.items()
                if backend != EXPECTED_ATTENTION_BACKEND and count
            }
            result.update(
                {
                    f"{role}_attention_dispatch": (
                        EXPECTED_ATTENTION_BACKEND if fa4_calls else None
                    ),
                    f"{role}_attention_dispatch_calls": fa4_calls,
                    f"{role}_fallback_attention_calls": sum(
                        fallback_backends.values()
                    ),
                    f"{role}_fallback_attention_backends": fallback_backends,
                }
            )
        result["unattributed_attention_calls"] = {
            backend: int(count)
            for backend, count in self._unattributed_calls.items()
            if count
        }
        return result


def record_attention_dispatch(
    backend: str,
    *,
    module: Any | None = None,
    role: str | None = None,
) -> None:
    """Record custom attention adapters that bypass Transformers' registry."""

    tracker = _ACTIVE_DISPATCH_TRACKER.get()
    if tracker is not None:
        tracker.record(backend, module=module, role=role)


def install_attention_dispatch_tracking(registry: Any) -> None:
    """Wrap FA4 and fallback registry entries without changing their behavior."""

    installed_backends = set()
    for backend in _TRACKED_REGISTRY_BACKENDS:
        try:
            original = registry[backend]
        except (KeyError, TypeError):
            continue
        installed_backends.add(backend)
        marker = f"_fast_infer_dispatch_tracker_{backend}"
        if getattr(original, marker, False):
            continue

        def make_wrapper(attention_backend: str, attention_fn: Any):
            @wraps(attention_fn)
            def tracked(module, *args, **kwargs):
                record_attention_dispatch(attention_backend, module=module)
                return attention_fn(module, *args, **kwargs)

            setattr(tracked, f"_fast_infer_dispatch_tracker_{attention_backend}", True)
            return tracked

        registry[backend] = make_wrapper(backend, original)
    if EXPECTED_ATTENTION_BACKEND not in installed_backends:
        raise RuntimeError(
            "Transformers attention registry does not expose "
            f"{EXPECTED_ATTENTION_BACKEND!r}; cannot verify real FA4 dispatch"
        )


def resolve_flashattn_methods(value: str) -> tuple[str, ...]:
    """Resolve a native FA4 method subset, including the Vanilla/DFlash pair."""

    specification = str(value).strip()
    if specification.lower() == "all":
        methods = SUPPORTED_METHODS
    else:
        methods = tuple(part.strip() for part in specification.split(",") if part.strip())
    if not methods:
        raise ValueError("at least one FlashAttention method must be selected")
    if len(set(methods)) != len(methods):
        raise ValueError("FlashAttention method names must be unique")
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"unknown FlashAttention methods: {unknown}")
    if "vanilla_hf" not in methods or "dflash" not in methods:
        raise ValueError("comparison requires both vanilla_hf and dflash")
    return methods


def first_token_mismatch(reference: Iterable[int], candidate: Iterable[int]) -> int | None:
    """Return the first zero-based token offset where two generated sequences differ."""

    reference_tokens = tuple(int(value) for value in reference)
    candidate_tokens = tuple(int(value) for value in candidate)
    for index, (expected, actual) in enumerate(zip(reference_tokens, candidate_tokens)):
        if expected != actual:
            return index
    if len(reference_tokens) != len(candidate_tokens):
        return min(len(reference_tokens), len(candidate_tokens))
    return None


def _normalized_package_name(name: Any) -> str:
    return str(name).strip().lower().replace("_", "-").split("==", 1)[0]


def validate_flashattn_runtime(
    runtime: Mapping[str, Any],
    *,
    methods: Iterable[str],
    require_dispatch_proof: bool = False,
    allow_installed_vllm: bool = False,
) -> dict[str, Any]:
    """Fail closed unless the run is batch-1, does not import vLLM, and uses FA4.

    ``require_dispatch_proof`` additionally checks that each selected method's
    target path, and every speculative draft path, actually entered FA4 during
    inference without an observed alternate attention backend. Some shared
    server environments install vLLM and related distributions for unrelated
    jobs; ``allow_installed_vllm`` permits them to be present while still
    rejecting any vLLM import.
    """

    installed = {
        _normalized_package_name(value)
        for value in runtime.get("installed_distributions", [])
    }
    imported = {
        str(value).strip().lower()
        for value in runtime.get("imported_modules", [])
    }
    vllm_related_distributions = sorted(
        value for value in installed if "vllm" in value
    )
    vllm_installed = bool(vllm_related_distributions)
    if vllm_installed and not allow_installed_vllm:
        raise ValueError(
            "vLLM must not be installed in the native FA4 runtime, including adapter "
            f"plugins; found {vllm_related_distributions}"
        )
    vllm_imported = any("vllm" in value for value in imported)
    if vllm_imported:
        raise ValueError(
            "vLLM modules or plugins must not be imported in the native FA4 runtime"
        )
    if "flash-attn-4" not in installed:
        raise ValueError("flash-attn-4 must be installed in the native FA4 runtime")
    if not any(value == "flash_attn" or value.startswith("flash_attn.") for value in imported):
        raise ValueError("flash_attn.cute must be imported in the native FA4 runtime")
    if int(runtime.get("batch_size", 0)) != 1:
        raise ValueError("batch_size must be 1 for the synchronized pilot")

    configs = runtime.get("methods", {})
    if not isinstance(configs, Mapping):
        raise ValueError("runtime methods must be a mapping")
    checked = {}
    for method in methods:
        config = configs.get(method)
        if not isinstance(config, Mapping):
            raise ValueError(f"{method} attention configuration is missing")
        target_attention = config.get("target_attention")
        if target_attention != EXPECTED_ATTENTION_BACKEND:
            raise ValueError(
                f"{method} target attention resolved to {target_attention!r}; "
                f"expected {EXPECTED_ATTENTION_BACKEND!r}"
            )
        draft_attention = config.get("draft_attention")
        if str(method) != "vanilla_hf" and draft_attention is None:
            raise ValueError(f"{method} draft attention is missing")
        if draft_attention is not None and draft_attention != EXPECTED_ATTENTION_BACKEND:
            raise ValueError(
                f"{method} draft attention resolved to {draft_attention!r}; "
                f"expected {EXPECTED_ATTENTION_BACKEND!r}"
            )
        checked_method = {
            "target_attention": target_attention,
            "draft_attention": draft_attention,
        }
        if require_dispatch_proof:
            roles = ("target",) if str(method) == "vanilla_hf" else ("target", "draft")
            for role in roles:
                dispatch = config.get(f"{role}_attention_dispatch")
                dispatch_calls = int(
                    config.get(f"{role}_attention_dispatch_calls", 0) or 0
                )
                fallback_calls_value = config.get(
                    f"{role}_fallback_attention_calls"
                )
                if fallback_calls_value is None and str(method) == "dflash" and role == "draft":
                    fallback_calls_value = config.get("draft_sdpa_fallback_calls", 0)
                fallback_calls = int(fallback_calls_value or 0)
                if (
                    dispatch != EXPECTED_ATTENTION_BACKEND
                    or dispatch_calls <= 0
                    or fallback_calls != 0
                ):
                    if str(method) == "dflash":
                        detail = "DFlash FA4 dispatch was not observed"
                    else:
                        detail = f"{method} {role} FA4 dispatch was not observed"
                    raise ValueError(
                        f"{detail} (or alternate attention backend was used): "
                        f"dispatch={dispatch!r}, fa4_calls={dispatch_calls}, "
                        f"fallback_calls={fallback_calls}, "
                        f"fallback_backends={config.get(f'{role}_fallback_attention_backends', {})}"
                    )
                checked_method.update(
                    {
                        f"{role}_attention_dispatch": dispatch,
                        f"{role}_attention_dispatch_calls": dispatch_calls,
                        f"{role}_fallback_attention_calls": fallback_calls,
                        f"{role}_fallback_attention_backends": config.get(
                            f"{role}_fallback_attention_backends", {}
                        ),
                    }
                )
        checked[str(method)] = checked_method

    return {
        "passed": True,
        "batch_size": 1,
        "attention_backend": EXPECTED_ATTENTION_BACKEND,
        "vllm_installed": vllm_installed,
        "vllm_related_distributions": vllm_related_distributions,
        "vllm_imported": vllm_imported,
        "methods": checked,
    }


def select_smoke_datasets(
    dataset_names: Iterable[str], sample_count: int
) -> tuple[str, ...]:
    """Select evenly spaced datasets for a small, deterministic smoke run."""

    names = tuple(str(name) for name in dataset_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("smoke datasets must be non-empty and unique")
    count = int(sample_count)
    if count < 1 or count > len(names):
        raise ValueError(f"sample_count must be between 1 and {len(names)}")
    if count == len(names):
        return names
    if count == 1:
        return (names[(len(names) - 1) // 2],)
    indices = [round(index * (len(names) - 1) / (count - 1)) for index in range(count)]
    return tuple(names[index] for index in indices)


def select_median_samples(
    candidates: list[dict[str, Any]],
    *,
    datasets: Iterable[str],
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    """Select the lower-median prompt by tokenized input length per dataset."""

    selected = []
    for dataset in datasets:
        rows = []
        for row in candidates:
            try:
                input_tokens = int(row.get("input_tokens", 0))
            except (TypeError, ValueError):
                continue
            if (
                str(row.get("dataset", "")) == str(dataset)
                and 0 < input_tokens <= int(max_input_tokens)
            ):
                rows.append(row)
        rows.sort(key=lambda row: (int(row["input_tokens"]), str(row.get("sample_id", ""))))
        if not rows:
            raise ValueError(
                f"no {dataset} smoke prompt fits max_input_tokens={max_input_tokens}"
            )
        selected.append(dict(rows[(len(rows) - 1) // 2]))
    return selected
