"""Length-aware adaptive batching for CUDA inference preparation phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar

import torch


@dataclass(frozen=True)
class AdaptiveInferenceSettings:
    """Controls bounded preparation batching and CUDA memory backoff."""

    enabled: bool = True
    target_memory_fraction: float = 0.90
    min_batch_size: int = 1
    max_batch_size: int = 256
    max_tokens_per_batch: int = 0
    bucket_window: int = 512
    probe_batches: int = 2
    oom_backoff: bool = True


@dataclass(frozen=True)
class PreparedExample:
    """An item that can be sorted by estimated padded model-work length."""

    index: int
    length: int
    payload: Any = None


@dataclass(frozen=True)
class AdaptiveBatch:
    examples: tuple[PreparedExample, ...]
    padded_tokens: int


@dataclass(frozen=True)
class AdaptiveBatchSelection:
    batch_size: int
    target_memory_bytes: int
    peak_reserved_bytes: int
    probes: int
    hit_maximum: bool


T = TypeVar("T")


@dataclass(frozen=True)
class BackoffResult(Generic[T]):
    value: T
    batch_size: int
    oom_retries: int


def validate_adaptive_inference_settings(
    settings: AdaptiveInferenceSettings,
) -> None:
    if not 0.0 < settings.target_memory_fraction < 1.0:
        raise ValueError("target_memory_fraction must be between 0 and 1")
    if settings.min_batch_size <= 0:
        raise ValueError("min_batch_size must be positive")
    if settings.max_batch_size < settings.min_batch_size:
        raise ValueError("max_batch_size must be >= min_batch_size")
    if settings.max_tokens_per_batch < 0:
        raise ValueError("max_tokens_per_batch must be non-negative")
    if settings.bucket_window < settings.min_batch_size:
        raise ValueError("bucket_window must be >= min_batch_size")
    if settings.probe_batches <= 0:
        raise ValueError("probe_batches must be positive")


def add_adaptive_cli_args(parser: Any) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adaptive-batch",
        dest="adaptive_batch",
        action="store_true",
        help="enable length-aware CUDA adaptive batching",
    )
    group.add_argument(
        "--no-adaptive-batch",
        dest="adaptive_batch",
        action="store_false",
        help="disable adaptive batching and process one example at a time",
    )
    parser.set_defaults(adaptive_batch=None)
    parser.add_argument("--target-memory-fraction", type=float, default=0.90)
    parser.add_argument("--adaptive-min-batch-size", type=int, default=1)
    parser.add_argument("--adaptive-max-batch-size", type=int, default=256)
    parser.add_argument("--max-tokens-per-batch", type=int, default=0)
    parser.add_argument("--bucket-window", type=int, default=512)
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument(
        "--no-oom-backoff",
        action="store_true",
        help="fail instead of halving a batch after CUDA OOM",
    )


def adaptive_settings_from_args(args: Any, device: torch.device) -> AdaptiveInferenceSettings:
    enabled = args.adaptive_batch if args.adaptive_batch is not None else device.type == "cuda"
    settings = AdaptiveInferenceSettings(
        enabled=bool(enabled),
        target_memory_fraction=float(args.target_memory_fraction),
        min_batch_size=int(args.adaptive_min_batch_size),
        max_batch_size=int(args.adaptive_max_batch_size),
        max_tokens_per_batch=int(args.max_tokens_per_batch),
        bucket_window=int(args.bucket_window),
        probe_batches=int(args.probe_batches),
        oom_backoff=not bool(args.no_oom_backoff),
    )
    validate_adaptive_inference_settings(settings)
    if not settings.enabled:
        return AdaptiveInferenceSettings(
            enabled=False,
            min_batch_size=1,
            max_batch_size=1,
            bucket_window=1,
            target_memory_fraction=settings.target_memory_fraction,
            max_tokens_per_batch=settings.max_tokens_per_batch,
            probe_batches=settings.probe_batches,
            oom_backoff=settings.oom_backoff,
        )
    return settings


def _is_cuda_oom(error: BaseException) -> bool:
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    if isinstance(error, cuda_oom):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _padded_tokens(examples: list[PreparedExample]) -> int:
    if not examples:
        return 0
    return max(item.length for item in examples) * len(examples)


def _window_batches(
    window: list[PreparedExample],
    *,
    batch_size: int,
    max_tokens: int,
) -> Iterator[AdaptiveBatch]:
    ordered = sorted(window, key=lambda item: (item.length, item.index))
    current: list[PreparedExample] = []
    for item in ordered:
        candidate = [*current, item]
        exceeds_size = len(candidate) > batch_size
        exceeds_tokens = max_tokens > 0 and _padded_tokens(candidate) > max_tokens
        if current and (exceeds_size or exceeds_tokens):
            yield AdaptiveBatch(tuple(current), _padded_tokens(current))
            current = [item]
            continue
        if not current and max_tokens > 0 and _padded_tokens(candidate) > max_tokens:
            raise ValueError(
                "one prepared example exceeds max_tokens_per_batch: "
                f"index={item.index}, estimated_tokens={item.length}, limit={max_tokens}"
            )
        current = candidate
    if current:
        yield AdaptiveBatch(tuple(current), _padded_tokens(current))


def length_bucket_batches(
    examples: Iterable[PreparedExample],
    *,
    batch_size: int,
    max_tokens: int = 0,
    window: int = 512,
) -> Iterator[list[PreparedExample]]:
    """Yield deterministic length buckets without materializing the corpus."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if window < batch_size:
        raise ValueError("window must be >= batch_size")
    buffer: list[PreparedExample] = []
    for example in examples:
        if example.index < 0 or example.length <= 0:
            raise ValueError("prepared examples require non-negative index and positive length")
        buffer.append(example)
        if len(buffer) >= window:
            for batch in _window_batches(
                buffer,
                batch_size=batch_size,
                max_tokens=max_tokens,
            ):
                yield list(batch.examples)
            buffer = []
    if buffer:
        for batch in _window_batches(
            buffer,
            batch_size=batch_size,
            max_tokens=max_tokens,
        ):
            yield list(batch.examples)


def run_with_oom_backoff(
    work: Callable[[int], T],
    *,
    initial_batch_size: int,
    minimum_batch_size: int = 1,
) -> BackoffResult[T]:
    """Run work, halving its batch size only when CUDA reports an OOM."""

    if initial_batch_size <= 0 or minimum_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if initial_batch_size < minimum_batch_size:
        raise ValueError("initial_batch_size must be >= minimum_batch_size")
    current = initial_batch_size
    retries = 0
    while True:
        try:
            value = work(current)
            return BackoffResult(value, current, retries)
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            if current <= minimum_batch_size:
                raise RuntimeError(
                    "adaptive inference could not fit the minimum batch size "
                    f"({minimum_batch_size})"
                ) from error
            current = max(minimum_batch_size, current // 2)
            retries += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def select_cuda_batch_size(
    probe: Callable[[int], int | None],
    *,
    settings: AdaptiveInferenceSettings,
    device: torch.device,
    maximum: int | None = None,
) -> AdaptiveBatchSelection:
    """Select a batch whose real probe peak stays below the VRAM target."""

    validate_adaptive_inference_settings(settings)
    if not settings.enabled:
        return AdaptiveBatchSelection(
            settings.min_batch_size,
            0,
            0,
            0,
            False,
        )
    if device.type != "cuda" or not torch.cuda.is_available():
        return AdaptiveBatchSelection(
            settings.min_batch_size,
            0,
            0,
            0,
            False,
        )
    maximum = min(
        settings.max_batch_size,
        settings.max_batch_size if maximum is None else maximum,
    )
    if maximum < settings.min_batch_size:
        raise ValueError("available examples are fewer than min_batch_size")
    target_bytes = int(
        torch.cuda.get_device_properties(device).total_memory
        * settings.target_memory_fraction
    )
    observed: dict[int, int] = {}

    def fits(batch_size: int) -> bool:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            measured = probe(batch_size)
            peak = int(
                measured
                if measured is not None
                else torch.cuda.max_memory_reserved(device)
            )
            observed[batch_size] = peak
            return peak <= target_bytes
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            observed[batch_size] = int(torch.cuda.max_memory_reserved(device))
            torch.cuda.empty_cache()
            return False

    if not fits(settings.min_batch_size):
        raise RuntimeError(
            "adaptive inference could not fit the minimum batch size "
            f"({settings.min_batch_size})"
        )
    best = settings.min_batch_size
    candidate = best * 2
    failure: int | None = None
    while candidate <= maximum:
        if fits(candidate):
            best = candidate
            candidate *= 2
        else:
            failure = candidate
            break
    high = maximum if failure is None else min(maximum, failure - 1)
    low = best + 1
    while low <= high:
        candidate = (low + high) // 2
        if fits(candidate):
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return AdaptiveBatchSelection(
        batch_size=best,
        target_memory_bytes=target_bytes,
        peak_reserved_bytes=observed.get(best, 0),
        probes=len(observed),
        hit_maximum=best == maximum,
    )


__all__ = [
    "AdaptiveBatch",
    "AdaptiveBatchSelection",
    "AdaptiveInferenceSettings",
    "BackoffResult",
    "PreparedExample",
    "adaptive_settings_from_args",
    "add_adaptive_cli_args",
    "length_bucket_batches",
    "run_with_oom_backoff",
    "select_cuda_batch_size",
    "validate_adaptive_inference_settings",
]
