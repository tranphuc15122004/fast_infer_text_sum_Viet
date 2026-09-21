#!/usr/bin/env python3
"""Kiểm tra offline Python và dependency của benchmark tiếng Việt.

Checker này chỉ import module và đọc metadata version. Nó không gọi
Hugging Face, không tải model/dataset và không gọi network API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

CORE_MODULES = (
    "torch",
    "transformers",
    "numpy",
    "yaml",
    "safetensors",
    "tqdm",
    "accelerate",
    "tokenizers",
    "huggingface_hub",
)

# Profile server bao phủ các baseline tiếng Việt hiện có cùng pipeline
# Finetuning/SpecForge. Package CUDA được kiểm tra ở profile này; profile
# minimal vẫn chạy được trên máy dev CPU.
PROFILE_MODULES = {
    "minimal": CORE_MODULES,
    "server": CORE_MODULES
    + (
        "triton",
        "flashinfer",
        "flash_attn",
        "sglang",
        "sgl_kernel",
        "vllm",
        "dflash",
        "eagle.model.ea_model",
        "deepspec",
        "specforge",
        "Finetuning.run_train",
        "Finetuning.offline_sglang_capture",
    ),
}

OPTIONAL_MODULES = {"flash_attn", "vllm"}
DIST_NAMES = {
    "torch": "torch",
    "transformers": "transformers",
    "numpy": "numpy",
    "yaml": "PyYAML",
    "safetensors": "safetensors",
    "tqdm": "tqdm",
    "accelerate": "accelerate",
    "tokenizers": "tokenizers",
    "huggingface_hub": "huggingface-hub",
    "triton": "triton",
    "flashinfer": "flashinfer-python",
    "flash_attn": "flash-attn",
    "sglang": "sglang",
    "sgl_kernel": "sglang-kernel",
    "vllm": "vllm",
    "dflash": "dflash",
    "deepspec": "deepspec",
    "specforge": "specforge",
}


def _add_local_paths() -> None:
    """Make vendored source importable without installing it into Python."""

    for relative in (
        "src",
        "externals/dflash",
        "externals/EAGLE",
        "externals/DeepSpec",
        "externals/SpecForge",
    ):
        path = ROOT / relative
        if path.is_dir():
            sys.path.insert(0, str(path))


def _local_pythonpath() -> list[str]:
    return [
        str(ROOT / relative)
        for relative in (
            "src",
            "externals/dflash",
            "externals/EAGLE",
            "externals/DeepSpec",
            "externals/SpecForge",
        )
        if (ROOT / relative).is_dir()
    ]


def _probe_module(module_name: str) -> tuple[bool, dict[str, object] | str]:
    """Import một module trong process con.

    CUDA extension có thể làm interpreter chết bằng SIGSEGV khi wheel cu130
    bị probe nhầm trên driver local cũ. Tách từng import giúp báo lỗi rõ ràng
    và vẫn kiểm tra được các module còn lại.
    """

    probe_code = """
import importlib
import importlib.metadata
import json
import sys

module_name = sys.argv[1]
distribution = sys.argv[2]
module = importlib.import_module(module_name)
version = getattr(module, "__version__", None)
if not version and distribution:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
if not version:
    version = "local"
payload = {"version": str(version)}
if module_name == "torch":
    payload["cuda_available"] = bool(module.cuda.is_available())
print(json.dumps(payload))
"""
    env = dict(os.environ)
    paths = _local_pythonpath()
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    distribution = DIST_NAMES.get(module_name)
    if distribution is None:
        distribution = DIST_NAMES.get(module_name.split(".", 1)[0], "")
    result = subprocess.run(
        [sys.executable, "-c", probe_code, module_name, distribution],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        if result.returncode < 0:
            detail = f"terminated by signal {-result.returncode}"
        else:
            detail = f"exited with code {result.returncode}"
        diagnostic = (result.stderr or result.stdout).strip()
        if diagnostic:
            detail += f": {diagnostic.splitlines()[-1]}"
        return False, detail
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return False, "probe returned no metadata"
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return False, f"invalid probe metadata: {exc}"
    if not isinstance(payload, dict):
        return False, "probe metadata is not an object"
    return True, payload


def _module_names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(
        token
        for token in value.replace(",", " ").split()
        if token.strip()
    )
    if not names:
        raise ValueError("--modules không được rỗng")
    return names


def _prepare_runtime_cache() -> None:
    cache_root = Path(
        os.environ.get("FAST_INFER_CACHE_ROOT", "/tmp/fast_infer_cache")
    )
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(cache_root / "flashinfer"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions")
    )
    for name in (
        "FLASHINFER_WORKSPACE_BASE",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    ):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_MODULES),
        default="server",
        help="profile dependency cần kiểm tra (mặc định: server)",
    )
    parser.add_argument(
        "--modules",
        help="ghi đè danh sách module, phân cách bằng dấu phẩy hoặc khoảng trắng",
    )
    parser.add_argument(
        "--allow-python-version-mismatch",
        action="store_true",
        help="cho phép Python khác 3.12; chỉ dùng debug trên máy local",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _add_local_paths()
    _prepare_runtime_cache()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    try:
        modules = _module_names(args.modules) or PROFILE_MODULES[args.profile]
    except ValueError as exc:
        print(f"Shared environment preflight: FAIL: {exc}", file=sys.stderr)
        return 1

    failures: list[str] = []
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(
        "mode: offline import-only "
        f"(profile={args.profile}; no model loading)"
    )
    if sys.version_info[:2] != (3, 12):
        message = "Python 3.12 is required on the B200 server"
        if args.allow_python_version_mismatch:
            print(f"WARN {message}")
        else:
            print(f"FAIL {message}")
            failures.append(message)

    for module_name in modules:
        available, detail = _probe_module(module_name)
        if not available:
            level = "WARN OPTIONAL" if module_name in OPTIONAL_MODULES else "FAIL"
            print(f"{level} {module_name}: {detail}")
            if module_name not in OPTIONAL_MODULES:
                failures.append(f"{module_name}: {detail}")
            continue
        version = detail["version"]
        print(f"PASS {module_name} {version}")
        if module_name == "torch" and "cuda_available" in detail:
            print(f"CUDA available: {detail['cuda_available']}")

    if failures:
        print("\nShared environment preflight: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nShared environment preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
