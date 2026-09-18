"""Lazy SpecForge OfflineSGLangCapture integration and parity utilities.

The public SGLang HTTP API is a text-generation interface.  Hidden-state
capture needs the version-pinned in-process ModelRunner used by SpecForge, so
all SGLang/SpecForge imports stay behind explicit backend selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist


class OfflineSGLangCaptureError(RuntimeError):
    """An actionable failure while loading or using offline SGLang capture."""


@dataclass(frozen=True)
class NormalizedCaptureRows:
    """Validated per-sample hidden rows returned by an offline backend."""

    aux_rows: tuple[torch.Tensor, ...]
    last_rows: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ParityThresholds:
    """Numerical thresholds for HF-vs-SGLang hidden-state parity."""

    max_abs_error: float = 5e-2
    mean_abs_error: float = 1e-2
    relative_l2_error: float = 5e-2
    min_cosine_similarity: float = 0.999

    def validate(self) -> None:
        if self.max_abs_error < 0 or self.mean_abs_error < 0:
            raise ValueError("absolute parity thresholds must be non-negative")
        if self.relative_l2_error < 0:
            raise ValueError("relative_l2_error must be non-negative")
        if not -1.0 <= self.min_cosine_similarity <= 1.0:
            raise ValueError("min_cosine_similarity must be within [-1, 1]")


@dataclass(frozen=True)
class ParityReport:
    """Serializable aggregate parity result."""

    passed: bool
    sample_count: int
    token_count: int
    max_abs_error: float
    mean_abs_error: float
    relative_l2_error: float
    min_cosine_similarity: float
    thresholds: ParityThresholds

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["thresholds"] = asdict(self.thresholds)
        return payload


def _as_tensor_rows(value: Iterable[torch.Tensor], name: str) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 3:
            raise ValueError(f"{name} tensor must have shape [batch, seq, width]")
        return tuple(value[index] for index in range(value.shape[0]))
    rows = tuple(value)
    if not all(isinstance(row, torch.Tensor) for row in rows):
        raise TypeError(f"{name} must contain tensors")
    return rows


def normalize_capture_rows(
    aux_rows: Iterable[torch.Tensor],
    last_rows: Iterable[torch.Tensor],
    *,
    expected_lengths: Sequence[int],
    expected_aux_width: int,
    expected_last_width: int,
    dtype: torch.dtype,
) -> NormalizedCaptureRows:
    """Validate, detach and CPU-materialize rows from an offline backend."""

    aux = _as_tensor_rows(aux_rows, "aux_hidden_states")
    last = _as_tensor_rows(last_rows, "last_hidden_states")
    if len(aux) != len(expected_lengths) or len(last) != len(expected_lengths):
        raise ValueError(
            "capture returned a different number of rows: "
            f"aux={len(aux)}, last={len(last)}, expected={len(expected_lengths)}"
        )
    normalized_aux: list[torch.Tensor] = []
    normalized_last: list[torch.Tensor] = []
    for index, expected_length in enumerate(expected_lengths):
        for name, row, expected_width in (
            ("aux_hidden_states", aux[index], expected_aux_width),
            ("last_hidden_states", last[index], expected_last_width),
        ):
            if row.ndim != 2:
                raise ValueError(f"{name} row {index} must have shape [seq, width]")
            if int(row.shape[0]) != int(expected_length):
                raise ValueError(
                    f"{name} row {index} length {row.shape[0]} does not match "
                    f"expected length {expected_length}"
                )
            if int(row.shape[1]) != int(expected_width):
                raise ValueError(
                    f"{name} row {index} width {row.shape[1]} does not match "
                    f"expected width {expected_width}"
                )
        normalized_aux.append(aux[index].detach().to(device="cpu", dtype=dtype))
        normalized_last.append(last[index].detach().to(device="cpu", dtype=dtype))
    return NormalizedCaptureRows(tuple(normalized_aux), tuple(normalized_last))


def _cosine_similarity(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    if reference_norm == 0 and candidate_norm == 0:
        return 1.0
    if reference_norm == 0 or candidate_norm == 0:
        return 0.0
    return float(torch.dot(reference, candidate) / (reference_norm * candidate_norm))


def compare_hidden_rows(
    reference_rows: Sequence[torch.Tensor],
    candidate_rows: Sequence[torch.Tensor],
    *,
    thresholds: ParityThresholds | None = None,
) -> ParityReport:
    """Compare equal-shaped per-sample hidden rows without hiding shape errors."""

    selected = thresholds or ParityThresholds()
    selected.validate()
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(
            f"parity row count mismatch: reference={len(reference_rows)}, "
            f"candidate={len(candidate_rows)}"
        )
    if not reference_rows:
        raise ValueError("parity requires at least one row")
    max_abs = 0.0
    total_abs = 0.0
    total_elements = 0
    total_reference_sq = 0.0
    total_delta_sq = 0.0
    min_cosine = 1.0
    token_count = 0
    for index, (reference, candidate) in enumerate(zip(reference_rows, candidate_rows, strict=True)):
        if tuple(reference.shape) != tuple(candidate.shape):
            raise ValueError(
                f"parity shape mismatch at row {index}: "
                f"reference={tuple(reference.shape)}, candidate={tuple(candidate.shape)}"
            )
        ref = reference.detach().float().cpu()
        cand = candidate.detach().float().cpu()
        delta = cand - ref
        abs_delta = delta.abs()
        max_abs = max(max_abs, float(abs_delta.max().item()))
        total_abs += float(abs_delta.sum().item())
        total_elements += int(abs_delta.numel())
        total_reference_sq += float(ref.square().sum().item())
        total_delta_sq += float(delta.square().sum().item())
        min_cosine = min(min_cosine, _cosine_similarity(ref, cand))
        token_count += int(ref.shape[0]) if ref.ndim >= 1 else 0
    mean_abs = total_abs / max(total_elements, 1)
    relative_l2 = (total_delta_sq**0.5) / max(total_reference_sq**0.5, 1e-12)
    passed = (
        max_abs <= selected.max_abs_error
        and mean_abs <= selected.mean_abs_error
        and relative_l2 <= selected.relative_l2_error
        and min_cosine >= selected.min_cosine_similarity
    )
    return ParityReport(
        passed=passed,
        sample_count=len(reference_rows),
        token_count=token_count,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        relative_l2_error=relative_l2,
        min_cosine_similarity=min_cosine,
        thresholds=selected,
    )


def _specforge_root() -> Path:
    return Path(__file__).resolve().parents[2] / "externals" / "SpecForge"


def _ensure_specforge_import_path() -> None:
    root = _specforge_root()
    if not (root / "specforge").is_dir():
        raise OfflineSGLangCaptureError(f"vendored SpecForge not found: {root}")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)


def sglang_capture_available() -> tuple[bool, str | None]:
    """Return availability without importing SGLang during HF-only runs."""

    try:
        _prepare_runtime_cache_env()
        _ensure_specforge_import_path()
        importlib.import_module("sglang")
        importlib.import_module("specforge.offline_capture")
    except Exception as exc:  # dependency import errors are user-facing preflight data
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _prepare_runtime_cache_env() -> None:
    """Keep SGLang/FlashInfer/Triton compilation artifacts writable.

    The production runtime exports these variables from ``runtime.sh``.  The
    Python entry point is also used directly by Modal and by torchrun, so it
    must provide the same safe defaults before importing SGLang.  In
    particular, FlashInfer's default cache under a read-only home directory
    otherwise fails during ModelRunner construction.
    """

    root = Path(
        os.environ.get("FAST_INFER_CACHE_ROOT", "/tmp/fast_infer_offline_capture")
    )
    paths = {
        "FLASHINFER_WORKSPACE_BASE": root / "flashinfer",
        "TRITON_CACHE_DIR": root / "triton",
        "TORCH_EXTENSIONS_DIR": root / "torch_extensions",
    }
    for variable, path in paths.items():
        value = os.environ.setdefault(variable, str(path))
        Path(value).mkdir(parents=True, exist_ok=True)

    # Modal's CUDA Python wheels place nvcc under site-packages rather than
    # /usr/local/cuda.  FlashInfer's JIT compiler only consults CUDA_HOME, so
    # discover that layout without overwriting an administrator-provided
    # CUDA_HOME.
    nvcc_candidates: list[Path] = []
    found_nvcc = shutil.which("nvcc")
    if found_nvcc:
        nvcc_candidates.append(Path(found_nvcc))
    package_roots = [Path(entry) for entry in sys.path if entry]
    package_roots.extend(
        Path("/usr/local/lib").glob("python*/site-packages")
    )
    package_roots.extend(
        Path("/usr/lib").glob("python*/site-packages")
    )
    for entry in package_roots:
        for toolkit_name in ("cuda_nvcc", "cu13", "cu12"):
            nvcc_candidates.append(
                entry / "nvidia" / toolkit_name / "bin" / "nvcc"
            )
    nvcc_candidates.extend(
        [
            Path("/usr/local/cuda/bin/nvcc"),
            Path("/usr/local/cuda-13.0/bin/nvcc"),
            Path("/usr/local/cuda-12.8/bin/nvcc"),
        ]
    )
    nvcc_path = next((candidate for candidate in nvcc_candidates if candidate.exists()), None)
    if nvcc_path is not None:
        cuda_home = nvcc_path.parent.parent
        os.environ.setdefault("CUDA_HOME", str(cuda_home))
        os.environ.setdefault("CUDA_PATH", str(cuda_home))
        os.environ["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"


def initialize_specforge_distributed(*, tp_size: int) -> None:
    """Initialize SpecForge's device meshes before its ModelRunner is built.

    The existing Finetuning distributed helper intentionally remains untouched
    for HF mode.  SpecForge must create its TP/DP meshes first because its
    OfflineSGLangCapture reads the TP group from ``specforge.distributed``.
    """

    if tp_size <= 0:
        raise ValueError("SGLang tensor parallel size must be positive")
    _prepare_runtime_cache_env()
    _ensure_specforge_import_path()
    specforge_dist = importlib.import_module("specforge.distributed")
    if dist.is_initialized():
        if specforge_dist.get_tp_group() is None:
            raise OfflineSGLangCaptureError(
                "torch.distributed is already initialized without SpecForge device meshes; "
                "select the SGLang backend before initializing the Finetuning process group"
            )
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size % tp_size != 0:
        raise OfflineSGLangCaptureError(
            f"WORLD_SIZE={world_size} must be divisible by SGLang TP size={tp_size}"
        )
    specforge_dist.init_distributed(tp_size=tp_size)


def destroy_specforge_distributed() -> None:
    """Best-effort cleanup for the process groups owned by SpecForge."""

    try:
        _ensure_specforge_import_path()
        specforge_dist = importlib.import_module("specforge.distributed")
        if dist.is_initialized():
            specforge_dist.destroy_distributed()
    except Exception:
        # The caller is already handling the original capture exception; never
        # mask it with teardown errors from a version-specific SGLang build.
        return


class OfflineSGLangCaptureAdapter:
    """Thin adapter around the official SpecForge in-process capture class."""

    def __init__(
        self,
        capture: Any,
        *,
        layer_ids: Sequence[int],
        capture_method: str,
        dtype: torch.dtype,
    ) -> None:
        self.capture = capture
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.capture_method = str(capture_method)
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        layer_ids: Sequence[int],
        capture_method: str,
        dtype: torch.dtype,
        attention_backend: str = "flashinfer",
        mem_fraction_static: float = 0.40,
        max_running_requests: int = 8,
        max_total_tokens: int = 16384,
        context_length: int | None = None,
        disable_radix_cache: bool = False,
        trust_remote_code: bool = False,
    ) -> "OfflineSGLangCaptureAdapter":
        available, reason = sglang_capture_available()
        if not available:
            raise OfflineSGLangCaptureError(
                "OfflineSGLangCapture requires the vendored SpecForge backend and "
                f"a compatible SGLang install: {reason}"
            )
        try:
            module = importlib.import_module("specforge.offline_capture")
            capture = module.OfflineSGLangCapture.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                attention_backend=attention_backend,
                mem_fraction_static=float(mem_fraction_static),
                max_running_requests=int(max_running_requests),
                max_total_tokens=int(max_total_tokens),
                context_length=context_length,
                disable_radix_cache=bool(disable_radix_cache),
            )
            capture.set_capture_layers(
                [int(layer_id) for layer_id in layer_ids],
                capture_method=capture_method,
            )
        except Exception as exc:
            raise OfflineSGLangCaptureError(
                f"failed to initialize SpecForge OfflineSGLangCapture: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return cls(
            capture,
            layer_ids=layer_ids,
            capture_method=capture_method,
            dtype=dtype,
        )

    def capture_rows(
        self,
        input_ids: Sequence[torch.Tensor],
        *,
        expected_width: int,
    ) -> NormalizedCaptureRows:
        rows = [
            row.detach().to(device="cpu", dtype=torch.long).tolist()
            for row in input_ids
        ]
        if not rows:
            return NormalizedCaptureRows((), ())
        aux_rows, last_rows = self.capture.capture_rows(rows)
        hidden_size = int(last_rows[0].shape[-1]) if last_rows else 0
        return normalize_capture_rows(
            aux_rows,
            last_rows,
            expected_lengths=[len(row) for row in rows],
            expected_aux_width=expected_width,
            expected_last_width=hidden_size,
            dtype=self.dtype,
        )


__all__ = [
    "NormalizedCaptureRows",
    "OfflineSGLangCaptureAdapter",
    "OfflineSGLangCaptureError",
    "ParityReport",
    "ParityThresholds",
    "compare_hidden_rows",
    "destroy_specforge_distributed",
    "initialize_specforge_distributed",
    "normalize_capture_rows",
    "sglang_capture_available",
]
