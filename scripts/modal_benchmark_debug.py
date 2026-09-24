#!/usr/bin/env python3
"""GPU Modal debug runner for the Vietnamese LongBench smoke path.

Modal builds a Python 3.12 image from the exact versions in requirements.txt.
Internal file:// wheel references are resolved by wheel filename and fetched
from public package indexes; no local venv or private server path is uploaded.
Public Qwen3 checkpoints are downloaded into the temporary job filesystem, so
this runner creates no persistent Volume.

Example::

    MODAL_GPU=B200 modal run scripts/modal_benchmark_debug.py --action smoke
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from typing import Any
from urllib.parse import unquote, urlsplit

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/repo")
REMOTE_RUN_ROOT = Path("/tmp/fast-infer-modal-debug")
GPU = os.environ.get("MODAL_GPU", "B200")
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
if not REQUIREMENTS_FILE.is_file():
    REQUIREMENTS_FILE = REMOTE_ROOT / "requirements.txt"
REMOTE_DATA = REMOTE_ROOT / "datasets" / "eval_100"

MODEL = os.environ.get("MODAL_QWEN3_MODEL", "Qwen/Qwen3-4B")
EAGLE_MODEL = os.environ.get(
    "MODAL_QWEN3_EAGLE_MODEL", "AngelSlim/Qwen3-4B_eagle3"
)
DFLASH_MODEL = os.environ.get(
    "MODAL_QWEN3_DFLASH_MODEL", "z-lab/Qwen3-4B-DFlash-b16"
)
DOMINO_MODEL = os.environ.get(
    "MODAL_QWEN3_DOMINO_MODEL", "Huang2020/Qwen3-4B-Domino-b16"
)
DSPARK_MODEL = os.environ.get(
    "MODAL_QWEN3_DSPARK_MODEL", "deepseek-ai/dspark_qwen3_4b_block7"
)

BASELINES = (
    "vanilla_hf",
    "vanilla_fa",
    "eagle3",
    "dflash",
    "domino",
    "dspark",
)
PACKAGE_NAMES = (
    "torch",
    "transformers",
    "tokenizers",
    "accelerate",
    "datasets",
    "einops",
    "huggingface_hub",
    "numpy",
    "protobuf",
    "psutil",
    "rouge_score",
    "safetensors",
    "sentencepiece",
    "tqdm",
    "regex",
    "packaging",
    "Jinja2",
    "filelock",
    "sympy",
    "networkx",
    "PyYAML",
    "sglang",
    "sglang-kernel",
    "flashinfer-python",
    "flashinfer-cubin",
    "triton",
    "flash-attn",
    "flash-attn-4",
    "apache-tvm-ffi",
    "quack-kernels",
    "torch_c_dlpack_ext",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu13",
    "typing_extensions",
    "ninja",
    "modal",
)


def parse_requirements_pin_map(path: Path) -> dict[str, str]:
    """Read exact versions, resolving local wheel references by wheel filename.

    The server's requirements may point at private ``file://`` wheelhouse paths.
    Modal cannot access those paths; the package/version pins remain usable from
    PyPI or the package owner's public index.
    """

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name, parse_wheel_filename

    versions: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except Exception as exc:
            raise ValueError(
                f"invalid requirement at {path}:{line_number}: {line}"
            ) from exc
        package_name = canonicalize_name(requirement.name)
        if requirement.url:
            wheel_name = Path(unquote(urlsplit(requirement.url).path)).name
            try:
                artifact_name, artifact_version, _, _ = parse_wheel_filename(wheel_name)
            except Exception as exc:
                raise ValueError(
                    f"cannot extract version from direct wheel at {path}:{line_number}: {line}"
                ) from exc
            if canonicalize_name(artifact_name) != package_name:
                raise ValueError(
                    f"wheel name mismatch at {path}:{line_number}: "
                    f"{artifact_name} != {requirement.name}"
                )
            version = str(artifact_version)
        else:
            exact = [item.version for item in requirement.specifier if item.operator == "=="]
            if len(exact) != 1:
                continue
            version = exact[0]
        prior = versions.get(package_name)
        if prior is not None and prior != version:
            raise ValueError(
                f"conflicting pins for {package_name}: {prior} and {version}"
            )
        versions[package_name] = version
    return versions


MODAL_REQUIREMENT_PINS = parse_requirements_pin_map(REQUIREMENTS_FILE)


def _requirement_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    from packaging.utils import canonicalize_name

    result: dict[str, str] = {}
    for name in package_names:
        key = canonicalize_name(name)
        if key not in MODAL_REQUIREMENT_PINS:
            raise ValueError(f"{name} is not exactly pinned in {REQUIREMENTS_FILE}")
        result[name] = MODAL_REQUIREMENT_PINS[key]
    return result


# These are the compatible Hugging Face/data/runtime pins copied from the
# current requirements.txt.  The server-only direct wheel URLs are reduced to
# their version pins and fetched from public package indexes on Modal.
MODAL_COMPATIBLE_PACKAGE_NAMES = (
    "transformers",
    "tokenizers",
    "accelerate",
    "datasets",
    "einops",
    "huggingface_hub",
    "numpy",
    "protobuf",
    "psutil",
    "safetensors",
    "sentencepiece",
    "tqdm",
    "regex",
    "packaging",
    "Jinja2",
    "filelock",
    "sympy",
    "networkx",
    "PyYAML",
    "ninja",
    "triton",
)
MODAL_COMPATIBLE_SOURCE_VERSIONS = _requirement_versions(
    MODAL_COMPATIBLE_PACKAGE_NAMES
)

# Runtime stack pins that are required for SGLang, speculative kernels and FA4.
# They are read from requirements.txt rather than copied from another venv.
MODAL_REQUIRED_OVERRIDE_NAMES = (
    "torch",
    "sglang",
    "sglang-kernel",
    "flashinfer-python",
    "flashinfer-cubin",
    "apache-tvm-ffi",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu13",
    "quack-kernels",
    "typing_extensions",
    "torch_c_dlpack_ext",
    "flash-attn-4",
)
MODAL_REQUIRED_OVERRIDES = _requirement_versions(MODAL_REQUIRED_OVERRIDE_NAMES)
# FA4 b19 calls quack.activation.sub_packed_f32x2, removed by the pinned
# Quack runtime. Upstream FA4 b26 fixes this; b32 also includes PyTorch 2.13
# extension-build support. Its metadata requires TVM FFI >=0.1.12.
MODAL_REQUIRED_OVERRIDES.update(
    {"flash-attn-4": "4.0.0b32", "apache-tvm-ffi": "0.1.12"}
)
# The server freeze has mixed CUDA compiler/header releases (13.2/13.3) on top
# of a CUDA 13.0 runtime. FlashInfer JIT on SM100 requires matching 13.0 headers
# and nvcc, so select the components published by cuda-toolkit 13.0.3.0.
MODAL_CUDA_TOOLCHAIN_OVERRIDES = {
    "nvidia-cuda-cccl": "13.0.85",
    "nvidia-cuda-crt": "13.0.88",
    "nvidia-cuda-nvcc": "13.0.88",
    "nvidia-nvvm": "13.0.88",
}
MODAL_REQUIRED_OVERRIDES.update(MODAL_CUDA_TOOLCHAIN_OVERRIDES)
MODAL_RUNTIME_ADDITIONS = {"rouge_score": "0.1.2", "absl-py": "2.5.0"}
MODAL_REQUIRED_VERSIONS = {
    **MODAL_COMPATIBLE_SOURCE_VERSIONS,
    **MODAL_REQUIRED_OVERRIDES,
    **MODAL_RUNTIME_ADDITIONS,
}
MODAL_REQUIRED_OVERRIDE_REASONS = {
    "torch": "Pinned in requirements.txt as the CUDA 13 runtime used for B200.",
    "sglang": "Pinned in requirements.txt for Domino and DSpark speculative serving.",
    "sglang-kernel": "Must match the SGLang kernel version pinned in requirements.txt.",
    "flashinfer-python": "Pinned in requirements.txt; installed from the public package index.",
    "flashinfer-cubin": "Pinned in requirements.txt; installed from the official FlashInfer index.",
    "apache-tvm-ffi": "FA4 4.0.0b32 requires TVM FFI >=0.1.12,<0.2.",
    "nvidia-cutlass-dsl": "Pinned FA4/SGLang kernel dependency from requirements.txt.",
    "nvidia-cutlass-dsl-libs-base": "Pinned FA4 dependency from requirements.txt.",
    "nvidia-cutlass-dsl-libs-cu13": "Pinned CUDA 13 FA4 dependency from requirements.txt.",
    "quack-kernels": "Pinned FA4 dependency from requirements.txt.",
    "typing_extensions": "Pinned FA4 dependency from requirements.txt.",
    "torch_c_dlpack_ext": "Pinned FA4 dependency from requirements.txt.",
    "flash-attn-4": "FA4 4.0.0b32 fixes the removed Quack packed-subtraction API and supports Torch 2.13.",
    "nvidia-cuda-cccl": "Override mixed server headers with CUDA 13.0.85 to match B200 runtime.",
    "nvidia-cuda-crt": "Override mixed server headers with CUDA 13.0.88 to match B200 runtime.",
    "nvidia-cuda-nvcc": "Override nvcc 13.2 with CUDA 13.0.88; FlashInfer JIT rejected mixed compiler/headers.",
    "nvidia-nvvm": "Override NVVM 13.2 with CUDA 13.0.88 to match nvcc/runtime.",
}

MODAL_SPECIAL_INSTALL_PACKAGE_NAMES = {
    "torch",
    "sglang",
    "flashinfer-python",
    "flashinfer-cubin",
    "flash-attn-4",
    # vLLM is present in the server freeze but is not used by these six baselines.
    "vllm",
}
MODAL_IMAGE_PACKAGE_NAMES = tuple(
    name
    for name in MODAL_REQUIREMENT_PINS
    if name not in MODAL_SPECIAL_INSTALL_PACKAGE_NAMES
)
MODAL_PROBE_PACKAGE_NAMES = tuple(
    dict.fromkeys(
        (
            *PACKAGE_NAMES,
            *MODAL_IMAGE_PACKAGE_NAMES,
            "sglang",
            "flashinfer-python",
            "flashinfer-cubin",
            "flash-attn-4",
            *MODAL_RUNTIME_ADDITIONS,
            "vllm",
        )
    )
)
MODAL_INSTALL_PIN_NAMES = tuple(
    name
    for name in MODAL_REQUIREMENT_PINS
    if name != "vllm"
    and name not in MODAL_CUDA_TOOLCHAIN_OVERRIDES
    and name not in {"flash-attn-4", "apache-tvm-ffi"}
)
MODAL_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
MODAL_TORCH_SPEC = f"torch=={MODAL_REQUIRED_OVERRIDES['torch']}+cu130"
MODAL_PYPI_INDEX = "https://pypi.org/simple"


def ensure_cuda_runtime_link(cuda_home: Path) -> dict[str, str]:
    """Expose pip-installed libcudart under CUDA_HOME/lib64 for extension linkers."""

    runtime_root = cuda_home.parent
    candidates = [
        path
        for path in runtime_root.rglob("libcudart.so*")
        if path.is_file() and "stubs" not in path.parts
    ]
    candidates.sort(
        key=lambda path: (
            path.name != "libcudart.so",
            path.parent != cuda_home / "lib",
            len(path.name),
            str(path),
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"no pip-installed libcudart.so found under {runtime_root}"
        )
    target = candidates[0]
    lib64 = cuda_home / "lib64"
    lib64.mkdir(parents=True, exist_ok=True)
    link = lib64 / "libcudart.so"
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return {"link": str(link), "target": str(target)}
        link.unlink()
    link.symlink_to(target)
    return {"link": str(link), "target": str(target)}


def parse_baselines(value: str, allowed: tuple[str, ...] = BASELINES) -> list[str]:
    """Parse a comma/space-separated baseline list without duplicate runs."""

    selected = list(dict.fromkeys(value.replace(",", " ").split()))
    if not selected:
        raise ValueError("at least one baseline must be selected")
    unknown = [name for name in selected if name not in allowed]
    if unknown:
        raise ValueError(f"unsupported baseline(s): {', '.join(unknown)}")
    return selected


def validate_smoke_result(
    returncode: int | None,
    manifest: dict[str, Any] | None,
    expected_baselines: list[str],
) -> dict[str, Any]:
    """Require one successful, complete, audited VietNews record per baseline."""

    overall_issues: list[str] = []
    per_baseline: dict[str, dict[str, Any]] = {}
    if returncode != 0:
        overall_issues.append(f"benchmark process exit code: {returncode}")
    if manifest is None:
        overall_issues.append("run manifest is missing or unreadable")
        manifest = {}
    if manifest.get("failure_count", 0) != 0:
        overall_issues.append(f"manifest failure_count={manifest.get('failure_count')}")

    cells_by_baseline: dict[str, list[dict[str, Any]]] = {}
    for cell in manifest.get("cells", []):
        if isinstance(cell, dict):
            cells_by_baseline.setdefault(str(cell.get("baseline", "")), []).append(cell)

    for baseline in expected_baselines:
        issues: list[str] = []
        cells = cells_by_baseline.get(baseline, [])
        if len(cells) != 1:
            issues.append(f"expected one VietNews cell, found {len(cells)}")
        for cell in cells:
            if cell.get("dataset") != "vietnews":
                issues.append(f"unexpected dataset: {cell.get('dataset')}")
            if cell.get("status") != "success":
                issues.append(f"cell status={cell.get('status')}")
            if int(cell.get("sample_count", 0) or 0) != 1:
                issues.append(f"sample coverage={cell.get('sample_count')}, expected 1")
            preflight = cell.get("preflight") or {}
            if preflight.get("status") != "ready":
                issues.append(f"preflight status={preflight.get('status')}")
            contract = cell.get("metric_contract") or {}
            if contract.get("status") != "complete":
                issues.append(f"metric contract status={contract.get('status')}")
            if (
                int(contract.get("observed_samples", 0) or 0) != 1
                or int(contract.get("expected_samples", 0) or 0) != 1
            ):
                issues.append(
                    "metric contract coverage mismatch: "
                    f"observed={contract.get('observed_samples')}, "
                    f"expected={contract.get('expected_samples')}"
                )
            audit = cell.get("metric_audit_summary") or {}
            if int(audit.get("num_records", 0) or 0) != 1:
                issues.append(f"metric audit records={audit.get('num_records')}, expected 1")
            if audit.get("status_counts") != {"success": 1}:
                issues.append(f"metric audit statuses={audit.get('status_counts')}")
        per_baseline[baseline] = {
            "status": "passed" if not issues else "failed",
            "issues": issues,
        }
        overall_issues.extend(f"{baseline}: {issue}" for issue in issues)

    unexpected = sorted(set(cells_by_baseline) - set(expected_baselines))
    if unexpected:
        overall_issues.append(f"unexpected baseline cells: {', '.join(unexpected)}")
    return {
        "status": "passed" if not overall_issues else "failed",
        "baselines": per_baseline,
        "issues": overall_issues,
    }


def capture_requirements_fingerprint(
    requirements_path: Path = REQUIREMENTS_FILE,
) -> dict[str, Any]:
    """Fingerprint the declared server package pins without reading its venv."""

    from packaging.utils import canonicalize_name

    pins = parse_requirements_pin_map(requirements_path)
    direct_reference_names: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            from packaging.requirements import Requirement

            requirement = Requirement(line)
            if requirement.url:
                direct_reference_names.append(canonicalize_name(requirement.name))
    return {
        "kind": "requirements.txt",
        "requirements_path": str(requirements_path),
        "requirements_sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
        "packages": {
            name: pins.get(canonicalize_name(name)) for name in PACKAGE_NAMES
        },
        "all_exact_pins": pins,
        "internal_wheel_references_resolved_by_version": sorted(direct_reference_names),
    }


def classify_pip_check_output(
    output: str,
    modal_packages: dict[str, Any],
) -> dict[str, list[str]]:
    """Separate real dependency gaps from conflicts in the frozen pin set.

    Some packages' upstream metadata disagrees with other explicit versions in
    the recorded server freeze (for example SGLang/CUTLASS and outlines/core).
    Such a line is accepted only if both the requiring package and installed
    dependency exactly match their requirements.txt pins. Missing/unpinned
    dependencies and unrelated conflicts remain fatal.
    """

    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    package_versions = {
        canonicalize_name(name): value for name, value in modal_packages.items()
    }

    def matches_pin(actual: Any, expected: str | None) -> bool:
        if actual is None or expected is None:
            return False
        try:
            return SpecifierSet(f"=={expected}").contains(
                Version(str(actual)), prereleases=True
            )
        except Exception:
            return str(actual) == expected

    accepted: list[str] = []
    errors: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        allowed = False
        if " has requirement " in line and ", but you have " in line:
            try:
                requiring, detail = line.split(" has requirement ", 1)
                requiring_name, requiring_version = requiring.rsplit(" ", 1)
                _, installed = detail.rsplit(", but you have ", 1)
                installed_name, installed_version = installed.rstrip(".").rsplit(" ", 1)
                requiring_key = canonicalize_name(requiring_name)
                installed_key = canonicalize_name(installed_name)
                requiring_pin = MODAL_REQUIRED_OVERRIDES.get(
                    requiring_key, MODAL_REQUIREMENT_PINS.get(requiring_key)
                )
                installed_pin = MODAL_REQUIRED_OVERRIDES.get(
                    installed_key, MODAL_REQUIREMENT_PINS.get(installed_key)
                )
                allowed = (
                    matches_pin(package_versions.get(requiring_key), requiring_pin)
                    and package_versions.get(requiring_key) == requiring_version
                    and matches_pin(package_versions.get(installed_key), installed_pin)
                    and package_versions.get(installed_key) == installed_version
                )
            except (ValueError, TypeError):
                allowed = False
        (accepted if allowed else errors).append(line)
    return {"requirements_frozen_conflicts": accepted, "errors": errors}


def modal_package_policy_issues(
    source_packages: dict[str, Any],
    modal_packages: dict[str, Any],
) -> list[str]:
    """Require requirements.txt application pins and B200 runtime stack pins."""

    issues: list[str] = []
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    def matches_pin(actual: Any, expected: str) -> bool:
        if actual is None:
            return False
        try:
            return SpecifierSet(f"=={expected}").contains(
                Version(str(actual)), prereleases=True
            )
        except Exception:
            return str(actual) == expected

    for name, expected in MODAL_COMPATIBLE_SOURCE_VERSIONS.items():
        source_actual = source_packages.get(name)
        modal_actual = modal_packages.get(name)
        if not matches_pin(source_actual, expected):
            issues.append(
                f"requirements compatible pin mismatch {name}: "
                f"expected {expected}, got {source_actual}"
            )
        if not matches_pin(modal_actual, expected):
            issues.append(
                f"clone-compatible package mismatch {name}: "
                f"requirements {expected}, Modal {modal_actual}"
            )
    for name, expected in MODAL_REQUIRED_OVERRIDES.items():
        actual = modal_packages.get(name)
        if not matches_pin(actual, expected):
            issues.append(
                f"requirements B200 runtime pin mismatch {name}: "
                f"expected {expected}, got {actual}"
            )
    for name, expected in MODAL_RUNTIME_ADDITIONS.items():
        actual = modal_packages.get(name)
        if actual != expected:
            issues.append(
                f"required benchmark addition mismatch {name}: "
                f"expected {expected}, got {actual}"
            )
    return issues


def modal_requirement_pin_issues(modal_packages: dict[str, Any]) -> list[str]:
    """Ensure every installed requirements pin is present at its exact version."""

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    issues: list[str] = []
    for name in MODAL_INSTALL_PIN_NAMES:
        expected = MODAL_REQUIREMENT_PINS[name]
        actual = modal_packages.get(name)
        try:
            matches = actual is not None and SpecifierSet(f"=={expected}").contains(
                Version(str(actual)), prereleases=True
            )
        except Exception:
            matches = str(actual) == expected
        if not matches:
            issues.append(
                f"requirements pin mismatch {name}: expected {expected}, got {actual}"
            )
    return issues


def modal_gpu_preflight_issues(
    modal_environment: dict[str, Any],
    *,
    requested_gpu: str = GPU,
) -> list[str]:
    """Require an actual B200 with CUDA and Blackwell compute capability."""

    issues: list[str] = []
    if not modal_environment.get("cuda_available"):
        issues.append(
            "torch.cuda.is_available is false: "
            f"{modal_environment.get('torch_error')}"
        )
    requested_type = requested_gpu.split(":", 1)[0].rstrip("!")
    if requested_type != "B200":
        issues.append(f"expected a B200 GPU request, got {requested_gpu}")
    gpu_name = str(modal_environment.get("gpu") or "")
    if "B200" not in gpu_name:
        issues.append(f"expected Modal B200, got {gpu_name or 'no GPU'}")
    capability = modal_environment.get("compute_capability")
    if (
        not isinstance(capability, (list, tuple))
        or not capability
        or int(capability[0]) < 10
    ):
        issues.append(
            "expected Blackwell compute capability (SM100 or newer), "
            f"got {capability}"
        )
    return issues


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _persist_local_outputs(result: dict[str, Any]) -> Path:
    """Write returned Modal logs, benchmark artifacts and the report locally."""

    run_id = str(result["run_id"])
    output_dir = PROJECT_ROOT / "outputs" / "modal_debug" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, content in (result.get("logs") or {}).items():
        path = output_dir / "logs" / f"{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
    for relative_name, content in (result.get("artifacts") or {}).items():
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe artifact path from Modal: {relative_name}")
        path = output_dir / "benchmark" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")

    environment = {
        "source_requirements": result.get("source_environment"),
        "modal": result.get("modal_environment"),
        "package_differences": result.get("package_differences"),
        "package_policy": result.get("package_policy"),
        "checkpoints": result.get("checkpoints"),
    }
    _write_json(output_dir / "environment.json", environment)

    summary = {
        key: value
        for key, value in result.items()
        if key not in {"logs", "artifacts", "source_environment"}
    }
    _write_json(output_dir / "run_report.json", summary)

    runtime = result.get("modal_environment") or {}
    packages = runtime.get("packages") or {}
    lines = [
        f"# Báo cáo debug Modal — {run_id}",
        "",
        f"- Kết quả: **{result.get('status', 'unknown')}**",
        f"- Chế độ benchmark: {result.get('benchmark_mode', 'smoke')}",
        f"- Giới hạn output: {result.get('max_new_tokens', 8)} token",
        f"- GPU yêu cầu/thực tế: {GPU} / {runtime.get('gpu')}",
        f"- Compute capability: {runtime.get('compute_capability')}",
        f"- CUDA Torch / driver API: {runtime.get('torch_cuda')} / "
        f"{runtime.get('cuda_driver_api')}",
        f"- requirements.txt: {(result.get('source_environment') or {}).get('requirements_path')}",
        f"- SHA-256: {(result.get('source_environment') or {}).get('requirements_sha256')}",
        "",
        "## Pin môi trường Modal",
        "",
        "Các pin ứng dụng được giữ theo requirements.txt; nhóm CUDA/SGLang/FlashInfer/FA4 được cài theo pin runtime B200:",
    ]
    lines.extend(
        f"- {name}=={version}: {MODAL_REQUIRED_OVERRIDE_REASONS[name]}"
        for name, version in MODAL_REQUIRED_OVERRIDES.items()
    )
    lines.extend(["", "## Phiên bản Modal đã chạy", ""])
    lines.extend(f"- {name}=={version}" for name, version in packages.items())
    lines.extend(["", "## Checkpoint", ""])
    for role, item in (result.get("checkpoints") or {}).items():
        lines.append(
            f"- {role}: {item.get('repo_id')} @ {item.get('checkpoint_id')} "
            f"({len(item.get('weight_files') or [])} file trọng số)"
        )
    lines.extend(["", "## Các lượt smoke", ""])
    for stage, item in (result.get("stages") or {}).items():
        lines.append(f"- {stage}: **{item.get('status')}**")
        for issue in item.get("issues", []):
            lines.append(f"  - {issue}")
        if item.get("command"):
            lines.append(f"  - Lệnh: {' '.join(item['command'])}")
    if result.get("errors"):
        lines.extend(["", "## Lỗi", ""])
        lines.extend(f"- {error}" for error in result["errors"])
    lines.extend(
        [
            "",
            "## Giới hạn diễn giải",
            "",
            f"Đây là lượt {result.get('benchmark_mode', 'smoke')} 1 mẫu, "
            f"đầu ra tối đa {result.get('max_new_tokens', 8)} token: "
            "chỉ xác nhận đường chạy, coverage và metric audit; không dùng ROUGE hoặc timing để kết luận "
            "chất lượng/tốc độ.",
            "Modal B200 xác nhận nhánh kernel Blackwell/SM100; smoke này không "
            "thay thế lần chạy production với master config và cache offline trên server B200.",
            "Không tạo Modal Volume và không thay đổi môi trường server.",
            "",
            "Full log ở logs/; JSONL, manifest và audit ở benchmark/.",
        ]
    )
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir

def _requirement_pin(name: str) -> str:
    from packaging.utils import canonicalize_name

    version = MODAL_REQUIREMENT_PINS[canonicalize_name(name)]
    return f"{name}=={version}"


app = modal.App("fast-infer-text-sum-viet-debug")

# Install the cu130 PyTorch wheel explicitly for B200. The requirement uses the
# public version (2.13.0); PyTorch's CUDA index supplies its CUDA 13 build.
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    MODAL_TORCH_SPEC,
    extra_options=(
        f"--index-url {MODAL_TORCH_INDEX} --extra-index-url {MODAL_PYPI_INDEX}"
    ),
)
# SGLang's published metadata hard-pins an older CUTLASS than the server
# requirements freeze. Install it without dependency resolution, then install
# the rest of the frozen SGLang runtime set from exact package/version pins.
image = image.pip_install(
    _requirement_pin("sglang"), extra_options="--no-deps"
)
# Install the full frozen package set without asking pip to reconcile it. This
# mirrors the recorded server freeze, whose deliberately overlaid package pins
# include known upstream Requires-Dist conflicts. Internal wheel URLs were
# converted to exact name/version pins; only unrelated vLLM is omitted.
image = image.pip_install(
    *[_requirement_pin(name) for name in MODAL_IMAGE_PACKAGE_NAMES],
    "rouge-score==0.1.2",
    extra_options="--no-deps",
)
# Correct only CUDA compiler/header components whose requirements.txt versions
# produced an incompatible FlashInfer JIT toolchain (nvcc 13.2, headers 13.3).
# cuda-toolkit 13.0.3.0 publishes this matched CUDA 13.0 component set.
image = image.pip_install(
    f"cuda-toolkit[cccl,crt,nvcc,nvvm]=={MODAL_REQUIREMENT_PINS['cuda-toolkit']}"
)
# rouge-score has one runtime dependency absent from the requirements freeze.
image = image.pip_install("absl-py==2.5.0", extra_options="--no-deps")
# FlashInfer Python is on PyPI; its matching cubins are served from the official
# FlashInfer index. Keep both versions equal to the internal wheel filenames.
image = image.pip_install(
    _requirement_pin("flashinfer-python"),
    extra_options="--no-deps --extra-index-url https://flashinfer.ai/whl",
)
image = image.pip_install(
    _requirement_pin("flashinfer-cubin"),
    extra_options="--no-deps --index-url https://flashinfer.ai/whl",
)
# FA4's frozen b19 uses a Quack API removed by the requirements.txt Quack pin.
# Install the upstream API fix plus its required TVM FFI version without letting
# its metadata alter unrelated packages in the recorded server freeze.
image = image.pip_install(
    f"apache-tvm-ffi=={MODAL_REQUIRED_OVERRIDES['apache-tvm-ffi']}",
    extra_options="--no-deps",
)
image = image.pip_install(
    f"flash-attn-4=={MODAL_REQUIRED_OVERRIDES['flash-attn-4']}",
    extra_options="--no-deps",
)
# SGLang is installed after its complete dependency pin set so its published
# metadata cannot replace the selected runtime versions.
image = image.pip_install(
    _requirement_pin("sglang"), extra_options="--no-deps"
)
image = image.add_local_file(
    str(REQUIREMENTS_FILE),
    remote_path=str(REMOTE_ROOT / "requirements.txt"),
    copy=True,
)
image = image.add_local_dir(
    str(PROJECT_ROOT / "src"), remote_path=str(REMOTE_ROOT / "src"), copy=True
).add_local_dir(
    str(PROJECT_ROOT / "datasets" / "eval_100"),
    remote_path=str(REMOTE_DATA),
    copy=True,
)

for _name in ("EAGLE", "dflash", "Domino", "SpecForge"):
    image = image.add_local_dir(
        str(PROJECT_ROOT / "externals" / _name),
        remote_path=str(REMOTE_ROOT / "externals" / _name),
        copy=True,
    )


def _child_env(
    *,
    baselines: str,
    model_paths: dict[str, str],
    hf_home: Path,
    run_root: Path,
    output_dir: Path,
    max_new_tokens: int = 8,
    benchmark_mode: str = "smoke",
) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = [
        str(REMOTE_ROOT / "src"),
        str(REMOTE_ROOT / "externals" / "EAGLE"),
        str(REMOTE_ROOT / "externals" / "dflash"),
        str(REMOTE_ROOT / "externals" / "Domino"),
        str(REMOTE_ROOT / "externals" / "SpecForge"),
    ]
    hf_hub_cache = hf_home / "hub"
    cuda_home = Path(
        os.environ.get("CUDA_HOME", "/usr/local/lib/python3.12/site-packages/nvidia/cu13")
    )
    cuda_library_paths = [str(cuda_home / "lib64"), str(cuda_home / "lib")]
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "FAST_INFER_PYTHON": sys.executable,
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_hub_cache),
            "TRANSFORMERS_CACHE": str(hf_hub_cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRITON_CACHE_DIR": str(run_root / "triton"),
            "TORCH_EXTENSIONS_DIR": str(run_root / "torch_extensions"),
            "FLASHINFER_WORKSPACE_BASE": str(run_root / "flashinfer"),
            "SGLANG_ENABLE_JIT_DEEPGEMM": "false",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "CUDA_HOME": os.environ.get(
                "CUDA_HOME", "/usr/local/lib/python3.12/site-packages/nvidia/cu13"
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(
                filter(
                    None,
                    [os.environ.get("LD_LIBRARY_PATH", ""), *cuda_library_paths],
                )
            ),
            "LIBRARY_PATH": os.pathsep.join(
                filter(
                    None,
                    [os.environ.get("LIBRARY_PATH", ""), *cuda_library_paths],
                )
            ),
            "MODEL_TARGET": model_paths["target"],
            "LONG_BENCH_MODEL": model_paths["target"],
            "MODEL_EAGLE_DRAFT": model_paths.get("eagle3", EAGLE_MODEL),
            "LONG_BENCH_EAGLE_MODEL": model_paths.get("eagle3", EAGLE_MODEL),
            "MODEL_DFLASH_DRAFT": model_paths.get("dflash", DFLASH_MODEL),
            "LONG_BENCH_DFLASH_MODEL": model_paths.get("dflash", DFLASH_MODEL),
            "MODEL_DOMINO_DRAFT": model_paths.get("domino", DOMINO_MODEL),
            "LONG_BENCH_DOMINO_MODEL": model_paths.get("domino", DOMINO_MODEL),
            "MODEL_DSPARK_DRAFT": model_paths.get("dspark", DSPARK_MODEL),
            "LONG_BENCH_DSPARK_MODEL": model_paths.get("dspark", DSPARK_MODEL),
            "LONG_BENCH_DATA_DIR": str(REMOTE_DATA),
            "LONG_BENCH_OUTPUT_DIR": str(output_dir),
            "LONG_BENCH_DEVICE": "cuda",
            "LONG_BENCH_GPU_IDS": "0",
            "LONG_BENCH_LOCAL_FILES_ONLY": "1",
            "LONG_BENCH_MODE": benchmark_mode,
            "LONG_BENCH_BASELINES": baselines,
            "LONG_BENCH_DATASETS": "vietnews",
            "LONG_BENCH_MAX_NEW_TOKENS": str(max_new_tokens),
            "LONG_BENCH_SMOKE_MAX_NEW_TOKENS": str(max_new_tokens),
            "LONG_BENCH_MAX_INPUT_TOKENS": "4096",
            "LONG_BENCH_SMOKE_MAX_INPUT_TOKENS": "4096",
            "LONG_BENCH_WARMUP_RUNS": "1",
            "LONG_BENCH_BATCH_SIZE": "1",
            "LONG_BENCH_MAX_RUNNING_REQUESTS": "1",
            "LONG_BENCH_MEM_FRACTION_STATIC": os.environ.get(
                "MODAL_SGLANG_MEM_FRACTION_STATIC", "0.75"
            ),
            "LONG_BENCH_SEED": "42",
            "LONG_BENCH_TEMPERATURE": "0",
            "LONG_BENCH_STRICT": "1",
            "LONG_BENCH_COLLECT": "1",
            "LONG_BENCH_RETRY_FAILED_SAMPLES": "0",
            "LONG_BENCH_SAMPLE_RETRIES": "0",
            "LONG_BENCH_CONTINUE_ON_ERROR": "1",
            "LONG_BENCH_TIMEOUT_SECONDS": "900",
            "LONG_BENCH_ATTENTION_BACKEND": os.environ.get(
                "MODAL_VANILLA_ATTENTION_BACKEND", "flash_attention_4"
            ),
            "LONG_BENCH_SGLANG_ATTENTION_BACKEND": os.environ.get(
                "MODAL_SGLANG_ATTENTION_BACKEND", "flashinfer"
            ),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run one bounded command and persist its complete combined output."""

    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
        output = completed.stdout
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = _decode_output(exc.stdout)
        timed_out = True
        returncode = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "log_path": str(log_path),
        "output_tail": output[-20000:],
    }


def _download_checkpoints(
    *,
    cache_dir: Path,
    models: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Download only requested public snapshots and verify configs/weights."""

    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HOME": str(cache_dir.parent),
            "HF_HUB_CACHE": str(cache_dir),
            "TRANSFORMERS_CACHE": str(cache_dir),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_DATASETS_OFFLINE": "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # Xet is enabled in huggingface_hub 1.x. High-performance mode
            # uses the Modal host's available network/CPU for large public
            # model files; a longer read timeout tolerates slow anonymous HF
            # transfers while still surfacing stalled connections.
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        }
    )

    def cached_bytes() -> int:
        total = 0
        for root, _, filenames in os.walk(cache_dir):
            for filename in filenames:
                path = Path(root) / filename
                try:
                    if not path.is_symlink():
                        total += path.stat().st_size
                except OSError:
                    pass
        return total

    local_paths: dict[str, str] = {}
    checkpoints: dict[str, Any] = {}
    for role, repo_id in models.items():
        if not repo_id:
            raise ValueError(f"checkpoint is not configured for role {role}")
        print(f"[checkpoint] downloading/checking {role}: {repo_id}", flush=True)
        progress_stop = threading.Event()
        started = time.monotonic()
        initial_bytes = cached_bytes()

        def report_progress() -> None:
            last_bytes = initial_bytes
            while not progress_stop.wait(30):
                current_bytes = cached_bytes()
                print(
                    f"[checkpoint-progress] role={role} "
                    f"elapsed={time.monotonic() - started:.0f}s "
                    f"cache_bytes={current_bytes} "
                    f"delta_bytes={current_bytes - initial_bytes} "
                    f"recent_delta_bytes={current_bytes - last_bytes}",
                    flush=True,
                )
                last_bytes = current_bytes

        progress_thread = threading.Thread(
            target=report_progress,
            name=f"checkpoint-progress-{role}",
            daemon=True,
        )
        progress_thread.start()
        try:
            snapshot = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                token=os.environ.get("HF_TOKEN") or None,
                local_files_only=False,
            )
        finally:
            progress_stop.set()
            progress_thread.join(timeout=5)
        snapshot_path = Path(snapshot)
        config_ok = (snapshot_path / "config.json").is_file()
        weight_files = sorted(
            {
                path
                for pattern in ("*.safetensors", "*.bin", "*.pt")
                for path in snapshot_path.glob(pattern)
            }
        )
        if not config_ok or not weight_files:
            raise RuntimeError(
                f"incomplete checkpoint {repo_id}: path={snapshot_path}, "
                f"config={config_ok}, weights={len(weight_files)}"
            )
        local_paths[role] = str(snapshot_path)
        checkpoints[role] = {
            "repo_id": repo_id,
            "checkpoint_id": snapshot_path.name,
            "snapshot": str(snapshot_path),
            "config": config_ok,
            "weight_files": [path.name for path in weight_files],
            "weight_bytes": sum(path.stat().st_size for path in weight_files),
        }
        print(
            f"[checkpoint] ready {role}: {snapshot_path.name} "
            f"({len(weight_files)} weight files)",
            flush=True,
        )
    return local_paths, checkpoints


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=9000,
    max_containers=1,
)
def debug(
    *,
    action: str,
    baselines: str,
    run_id: str,
    source_environment: dict[str, Any],
    model: str = MODEL,
    eagle_model: str = EAGLE_MODEL,
    dflash_model: str = DFLASH_MODEL,
    domino_model: str = DOMINO_MODEL,
    dspark_model: str = DSPARK_MODEL,
    max_new_tokens: int = 8,
    benchmark_mode: str = "smoke",
) -> dict[str, Any]:
    """Run focused DSpark first, then the six-baseline regression on one B200."""

    if Path(run_id).name != run_id or not run_id:
        raise ValueError(f"unsafe run id: {run_id!r}")
    if action not in {"smoke", "debug"}:
        raise ValueError(f"unsupported action: {action}")
    selected_baselines = ["dspark"] if action == "smoke" else parse_baselines(baselines)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if benchmark_mode not in {"smoke", "representative"}:
        raise ValueError("benchmark_mode must be smoke or representative")
    if action == "smoke" and benchmark_mode != "smoke":
        raise ValueError("action=smoke requires benchmark_mode=smoke")
    run_root = REMOTE_RUN_ROOT / run_id
    hf_home = run_root / "hf"
    hf_cache = hf_home / "hub"
    benchmark_root = run_root / "benchmark"
    log_root = run_root / "logs"
    for directory in (hf_cache, benchmark_root, log_root):
        directory.mkdir(parents=True, exist_ok=True)

    model_ids = {
        "target": model,
        "eagle3": eagle_model,
        "dflash": dflash_model,
        "domino": domino_model,
        "dspark": dspark_model,
    }
    checkpoints: dict[str, Any] = {}
    local_paths: dict[str, str] = {}
    stages: dict[str, Any] = {}
    errors: list[str] = []
    logs: dict[str, str] = {}
    modal_environment: dict[str, Any] = {}
    package_differences: dict[str, Any] = {}
    overall_status = "failed"

    try:
        needed_roles = {"target"}
        if action == "smoke":
            needed_roles.add("dspark")
        else:
            for baseline in selected_baselines:
                role = {
                    "eagle3": "eagle3",
                    "dflash": "dflash",
                    "domino": "domino",
                    "dspark": "dspark",
                }.get(baseline)
                if role:
                    needed_roles.add(role)

        cuda_runtime_link = ensure_cuda_runtime_link(
            Path(os.environ.get("CUDA_HOME", "/usr/local/lib/python3.12/site-packages/nvidia/cu13"))
        )
        print(
            "[preflight] CUDA runtime linker alias: "
            f"{cuda_runtime_link['link']} -> {cuda_runtime_link['target']}",
            flush=True,
        )
        requested_models = {role: model_ids[role] for role in sorted(needed_roles)}
        local_paths, checkpoints = _download_checkpoints(
            cache_dir=hf_cache,
            models=requested_models,
        )
        print(f"[preflight] verified checkpoints: {sorted(checkpoints)}", flush=True)

        probe_code = """
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import torch

names = __PACKAGE_NAMES__
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
cuda_available = False
gpu = None
capability = None
torch_error = None
try:
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        gpu = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
except Exception as exc:
    torch_error = f"{type(exc).__name__}: {exc}"
algorithms = {}
try:
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    for name in ("DFLASH", "DSPARK"):
        try:
            algorithms[name] = str(SpeculativeAlgorithm.from_string(name))
        except Exception as exc:
            algorithms[name] = f"ERROR: {type(exc).__name__}: {exc}"
except Exception as exc:
    algorithms["import_error"] = f"{type(exc).__name__}: {exc}"
fa4_importable = False
fa4_import_error = None
try:
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    fa4_importable = callable(flash_attn_func) and callable(flash_attn_varlen_func)
except Exception as exc:
    fa4_import_error = f"{type(exc).__name__}: {exc}"
flashinfer_importable = False
flashinfer_import_error = None
try:
    import flashinfer
    flashinfer_importable = True
except Exception as exc:
    flashinfer_import_error = f"{type(exc).__name__}: {exc}"
smi = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, check=False)
smi_text = smi.stdout if smi.returncode == 0 else f"nvidia-smi exit {smi.returncode}: {smi.stdout}"
driver_api = None
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
nvcc_path = Path(cuda_home) / "bin" / "nvcc"
nvcc = subprocess.run([str(nvcc_path), "--version"], text=True,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
nvcc_text = nvcc.stdout if nvcc.returncode == 0 else f"nvcc exit {nvcc.returncode}: {nvcc.stdout}"
nvcc_release = None
for line in nvcc_text.splitlines():
    if "release " in line:
        nvcc_release = line.split("release ", 1)[1].split(",", 1)[0].strip()
        break
for line in smi_text.splitlines():
    if "CUDA Version:" in line:
        driver_api = line.split("CUDA Version:", 1)[1].split("|", 1)[0].strip()
        break
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "packages": packages,
    "torch_cuda": torch.version.cuda,
    "cuda_available": cuda_available,
    "gpu": gpu,
    "compute_capability": capability,
    "torch_error": torch_error,
    "cuda_driver_api": driver_api,
    "nvcc_release": nvcc_release,
    "nvcc_path": str(nvcc_path),
    "nvcc_version_output": nvcc_text,
    "nvidia_smi": smi_text,
    "algorithms": algorithms,
    "flash_attention_4_importable": fa4_importable,
    "flash_attention_4_import_error": fa4_import_error,
    "flashinfer_importable": flashinfer_importable,
    "flashinfer_import_error": flashinfer_import_error,
}, ensure_ascii=False))
""".replace("__PACKAGE_NAMES__", repr(list(MODAL_PROBE_PACKAGE_NAMES)))
        probe_env = _child_env(
            baselines="dspark",
            model_paths=local_paths,
            hf_home=hf_home,
            run_root=run_root,
            output_dir=benchmark_root / "preflight",
        )
        probe = _run(
            [sys.executable, "-c", probe_code],
            env=probe_env,
            cwd=REMOTE_ROOT,
            log_path=log_root / "preflight.log",
            timeout_seconds=300,
        )
        logs["preflight"] = (log_root / "preflight.log").read_text(encoding="utf-8")
        if probe["returncode"] != 0:
            raise RuntimeError(f"runtime probe failed: {probe['output_tail']}")
        modal_environment = json.loads(probe["output_tail"].splitlines()[-1])
        modal_environment["cuda_runtime_link"] = cuda_runtime_link
        pip_check = _run(
            [sys.executable, "-m", "pip", "check"],
            env=probe_env,
            cwd=REMOTE_ROOT,
            log_path=log_root / "pip_check.log",
            timeout_seconds=120,
        )
        pip_freeze = _run(
            [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
            env=probe_env,
            cwd=REMOTE_ROOT,
            log_path=log_root / "pip_freeze.log",
            timeout_seconds=120,
        )
        logs["pip_check"] = (log_root / "pip_check.log").read_text(encoding="utf-8")
        logs["pip_freeze"] = (log_root / "pip_freeze.log").read_text(encoding="utf-8")
        modal_environment["pip_check_returncode"] = pip_check["returncode"]
        modal_environment["pip_check"] = logs["pip_check"]
        modal_environment["pip_check_policy"] = classify_pip_check_output(
            logs["pip_check"], modal_environment.get("packages") or {}
        )
        modal_environment["pip_freeze_returncode"] = pip_freeze["returncode"]
        modal_environment["pip_freeze"] = logs["pip_freeze"]

        source_packages = source_environment.get("packages") or {}
        modal_packages = modal_environment.get("packages") or {}
        package_differences = {
            name: {"requirements": source_packages.get(name), "modal": modal_packages.get(name)}
            for name in PACKAGE_NAMES
            if source_packages.get(name) != modal_packages.get(name)
        }
        preflight_issues: list[str] = []
        pip_check_policy = modal_environment.get("pip_check_policy") or {}
        if pip_check_policy.get("errors"):
            preflight_issues.append(
                "pip check found non-requirements dependency errors: "
                + " | ".join(pip_check_policy["errors"])
            )
        if modal_environment.get("pip_freeze_returncode") != 0:
            preflight_issues.append("could not capture the Modal package fingerprint")
        preflight_issues.extend(
            modal_gpu_preflight_issues(modal_environment, requested_gpu=GPU)
        )
        if modal_environment.get("torch_cuda") != "13.0":
            preflight_issues.append(
                f"expected Torch CUDA 13.0, got {modal_environment.get('torch_cuda')}"
            )
        driver_api = str(modal_environment.get("cuda_driver_api") or "")
        try:
            driver_version = tuple(int(part) for part in driver_api.split(".")[:2])
        except ValueError:
            driver_version = ()
        if driver_version < (13, 0):
            preflight_issues.append(
                f"expected driver API >=13.0, got {driver_api or 'unknown'}"
            )
        if modal_environment.get("nvcc_release") != "13.0":
            preflight_issues.append(
                "expected matched CUDA compiler release 13.0, got "
                f"{modal_environment.get('nvcc_release') or 'unknown'} "
                f"at {modal_environment.get('nvcc_path')}"
            )
        if not Path(cuda_runtime_link["link"]).is_file():
            preflight_issues.append(
                f"CUDA runtime linker alias is missing: {cuda_runtime_link['link']}"
            )
        algorithms = modal_environment.get("algorithms", {})
        for algorithm in ("DFLASH", "DSPARK"):
            if "ERROR" in str(algorithms.get(algorithm, "")):
                preflight_issues.append(
                    f"SGLang {algorithm} parse failed: {algorithms}"
                )
        if not modal_environment.get("flashinfer_importable"):
            preflight_issues.append(
                "FlashInfer import failed: "
                f"{modal_environment.get('flashinfer_import_error')}"
            )
        if modal_packages.get("flashinfer-python") != modal_packages.get("flashinfer-cubin"):
            preflight_issues.append(
                "FlashInfer Python/cubin versions differ: "
                f"{modal_packages.get('flashinfer-python')} vs "
                f"{modal_packages.get('flashinfer-cubin')}"
            )
        preflight_issues.extend(
            modal_package_policy_issues(source_packages, modal_packages)
        )
        preflight_issues.extend(modal_requirement_pin_issues(modal_packages))
        if not modal_environment.get("flash_attention_4_importable"):
            preflight_issues.append(
                "FlashAttention-4 symbol import failed: "
                f"{modal_environment.get('flash_attention_4_import_error')}"
            )
        preflight_issues.extend(
            f"checkpoint {role} missing config or weights"
            for role, item in checkpoints.items()
            if not item.get("config") or not item.get("weight_files")
        )
        stages["preflight"] = {
            "status": "passed" if not preflight_issues else "failed",
            "issues": preflight_issues,
            "checkpoint_roles": sorted(checkpoints),
            "algorithms": modal_environment.get("algorithms"),
        }
        if preflight_issues:
            errors.extend(preflight_issues)
        else:
            def run_benchmarks(
                stage_name: str,
                stage_baselines: list[str],
                timeout_seconds: int,
            ) -> dict[str, Any]:
                stage_output = benchmark_root / stage_name
                longbench_run_id = f"{stage_name}-{run_id}"
                env = _child_env(
                    baselines=",".join(stage_baselines),
                    model_paths=local_paths,
                    hf_home=hf_home,
                    run_root=run_root,
                    output_dir=stage_output,
                    max_new_tokens=max_new_tokens,
                    benchmark_mode=benchmark_mode,
                )
                command = [
                    sys.executable,
                    str(REMOTE_ROOT / "src" / "Benchmark" / "run_longbench_200.py"),
                    "--mode", benchmark_mode,
                    "--baselines", ",".join(stage_baselines),
                    "--datasets", "vietnews",
                    "--data-dir", str(REMOTE_DATA),
                    "--output-dir", str(stage_output),
                    "--max-samples", "1",
                    "--max-new-tokens", str(max_new_tokens),
                    "--max-input-tokens", "4096",
                    "--warmup-runs", "1",
                    "--sample-retries", "0",
                    "--no-retry-failed-samples",
                    "--continue-on-error",
                    "--strict",
                    "--collect",
                    "--run-id", longbench_run_id,
                ]
                log_path = log_root / f"{stage_name}.log"
                command_result = _run(
                    command,
                    env=env,
                    cwd=REMOTE_ROOT,
                    log_path=log_path,
                    timeout_seconds=timeout_seconds,
                )
                logs[stage_name] = log_path.read_text(encoding="utf-8")
                manifest_path = stage_output / longbench_run_id / "run_manifest.json"
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = None
                validation = validate_smoke_result(
                    command_result["returncode"],
                    manifest,
                    stage_baselines,
                )
                return {
                    "status": validation["status"],
                    "issues": validation["issues"],
                    "baselines": validation["baselines"],
                    "command": command,
                    "returncode": command_result["returncode"],
                    "timed_out": command_result["timed_out"],
                    "elapsed_seconds": command_result["elapsed_seconds"],
                    "output_tail": command_result["output_tail"],
                    "manifest_path": str(manifest_path),
                }

            if action == "smoke":
                focus = run_benchmarks("dspark_focus", ["dspark"], 1800)
                stages["dspark_focus"] = focus
                if focus["status"] == "passed":
                    remaining_models = {
                        role: model_ids[role]
                        for role in ("eagle3", "dflash", "domino")
                    }
                    extra_paths, extra_checkpoints = _download_checkpoints(
                        cache_dir=hf_cache,
                        models=remaining_models,
                    )
                    local_paths.update(extra_paths)
                    checkpoints.update(extra_checkpoints)
                    matrix = run_benchmarks("regression", list(BASELINES), 6600)
                    stages["regression"] = matrix
                    overall_status = matrix["status"]
                    errors.extend(matrix["issues"])
                else:
                    overall_status = "failed"
                    errors.extend(focus["issues"])
            else:
                focused = run_benchmarks("targeted", selected_baselines, 3600)
                stages["targeted"] = focused
                overall_status = focused["status"]
                errors.extend(focused["issues"])
    except Exception as exc:
        overall_status = "failed"
        errors.append(f"{type(exc).__name__}: {exc}")
        print(f"[modal-debug-error] {type(exc).__name__}: {exc}", flush=True)

    artifacts: dict[str, str] = {}
    for path in sorted(benchmark_root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md", ".log"}:
            relative_name = path.relative_to(benchmark_root).as_posix()
            try:
                artifacts[relative_name] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
    for path in log_root.glob("*.log"):
        logs.setdefault(path.stem, path.read_text(encoding="utf-8"))

    return {
        "run_id": run_id,
        "action": action,
        "benchmark_mode": benchmark_mode,
        "max_new_tokens": max_new_tokens,
        "status": overall_status,
        "gpu_requested": GPU,
        "modal_environment": modal_environment,
        "source_environment": source_environment,
        "package_differences": package_differences,
        "package_policy": {
            "clone_compatible_versions": MODAL_COMPATIBLE_SOURCE_VERSIONS,
            "required_overrides": MODAL_REQUIRED_OVERRIDES,
            "override_reasons": MODAL_REQUIRED_OVERRIDE_REASONS,
        },
        "checkpoints": checkpoints,
        "stages": stages,
        "errors": errors,
        "logs": logs,
        "artifacts": artifacts,
        "persistent_volume_used": False,
        "server_environment_changed": False,
    }


@app.local_entrypoint()
def main(
    action: str = "smoke",
    baseline: str = "dspark",
    baselines: str = "",
    run_id: str = "",
    model: str = MODEL,
    eagle_model: str = EAGLE_MODEL,
    dflash_model: str = DFLASH_MODEL,
    domino_model: str = DOMINO_MODEL,
    dspark_model: str = DSPARK_MODEL,
    max_new_tokens: int = 8,
    benchmark_mode: str = "smoke",
) -> None:
    if action not in {"smoke", "debug"}:
        raise SystemExit("action must be smoke or debug")
    selected = parse_baselines(baselines or baseline) if action == "debug" else ["dspark"]
    source_environment = capture_requirements_fingerprint()
    selected_run_id = run_id or (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + uuid.uuid4().hex[:8]
    )
    try:
        result = debug.remote(
            action=action,
            baselines=",".join(selected),
            run_id=selected_run_id,
            source_environment=source_environment,
            model=model,
            eagle_model=eagle_model,
            dflash_model=dflash_model,
            domino_model=domino_model,
            dspark_model=dspark_model,
            max_new_tokens=max_new_tokens,
            benchmark_mode=benchmark_mode,
        )
    except Exception as exc:
        failure = {
            "run_id": selected_run_id,
            "action": action,
            "status": "failed",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "source_environment": source_environment,
            "persistent_volume_used": False,
            "server_environment_changed": False,
        }
        output_dir = PROJECT_ROOT / "outputs" / "modal_debug" / selected_run_id
        _write_json(output_dir / "run_report.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        print(f"Artifacts: {output_dir}", flush=True)
        raise SystemExit(1) from exc

    output_dir = _persist_local_outputs(result)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"logs", "artifacts", "source_environment"}
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {output_dir}", flush=True)
    if result.get("status") != "passed":
        raise SystemExit(1)
