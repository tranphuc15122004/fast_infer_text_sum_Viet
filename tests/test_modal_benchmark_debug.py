from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_parse_baselines_accepts_comma_separated_values_and_deduplicates() -> None:
    from modal_benchmark_debug import parse_baselines

    assert parse_baselines(" dspark, vanilla_hf dspark ") == [
        "dspark",
        "vanilla_hf",
    ]


def test_validate_smoke_result_requires_successful_audited_coverage() -> None:
    from modal_benchmark_debug import validate_smoke_result

    manifest = {
        "failure_count": 0,
        "cells": [
            {
                "baseline": "dspark",
                "dataset": "vietnews",
                "status": "success",
                "sample_count": 1,
                "preflight": {"status": "ready", "requirements": {}},
                "metric_audit_summary": {
                    "num_records": 1,
                    "status_counts": {"success": 1},
                    "issue_counts": {},
                },
                "metric_contract": {
                    "status": "complete",
                    "observed_samples": 1,
                    "expected_samples": 1,
                    "issue_counts": {},
                },
            }
        ],
    }

    assert validate_smoke_result(0, manifest, ["dspark"]) == {
        "status": "passed",
        "baselines": {"dspark": {"status": "passed", "issues": []}},
        "issues": [],
    }

    manifest["cells"][0]["metric_contract"]["observed_samples"] = 0
    result = validate_smoke_result(0, manifest, ["dspark"])
    assert result["status"] == "failed"
    assert "coverage mismatch" in " ".join(result["baselines"]["dspark"]["issues"])


def test_validate_smoke_result_rejects_missing_dependency_even_on_zero_exit() -> None:
    from modal_benchmark_debug import validate_smoke_result

    manifest = {
        "failure_count": 0,
        "cells": [
            {
                "baseline": "dspark",
                "dataset": "vietnews",
                "status": "missing_dependency",
                "preflight": {
                    "status": "missing_dependency",
                    "reason": "missing dependency: sglang",
                    "requirements": {"sglang": False},
                },
                "metric_contract": {"status": "metric_incomplete"},
            }
        ],
    }

    result = validate_smoke_result(0, manifest, ["dspark"])

    assert result["status"] == "failed"
    assert any("missing_dependency" in issue for issue in result["issues"])


def test_run_persists_complete_output_log(tmp_path: Path) -> None:
    from modal_benchmark_debug import _run

    log_path = tmp_path / "baseline.log"
    result = _run(
        [sys.executable, "-c", "print('first-line'); print('last-line')"],
        env={},
        cwd=tmp_path,
        log_path=log_path,
        timeout_seconds=10,
    )

    assert result["returncode"] == 0
    assert result["log_path"] == str(log_path)
    assert log_path.read_text(encoding="utf-8") == "first-line\nlast-line\n"
    assert "first-line" in result["output_tail"]



def test_modal_runtime_pins_separate_clone_versions_from_required_overrides() -> None:
    from modal_benchmark_debug import (
        MODAL_COMPATIBLE_SOURCE_VERSIONS,
        MODAL_REQUIRED_OVERRIDES,
        MODAL_REQUIRED_VERSIONS,
        MODAL_TORCH_SPEC,
        MODAL_IMAGE_PACKAGE_NAMES,
        MODAL_RUNTIME_ADDITIONS,
        MODAL_PROBE_PACKAGE_NAMES,
        MODAL_CUDA_TOOLCHAIN_OVERRIDES,
        MODAL_INSTALL_PIN_NAMES,
        modal_requirement_pin_issues,
    )

    assert MODAL_REQUIRED_VERSIONS["torch"] == "2.13.0"
    assert MODAL_TORCH_SPEC == "torch==2.13.0+cu130"
    assert "triton" in MODAL_IMAGE_PACKAGE_NAMES
    assert "cuda-tile" in MODAL_IMAGE_PACKAGE_NAMES
    assert "vllm" in MODAL_PROBE_PACKAGE_NAMES
    assert "cuda-tile" in MODAL_PROBE_PACKAGE_NAMES
    assert not {
        "torch", "sglang", "flashinfer-python", "flashinfer-cubin", "flash-attn-4", "vllm"
    } & set(MODAL_IMAGE_PACKAGE_NAMES)
    assert MODAL_RUNTIME_ADDITIONS == {"rouge_score": "0.1.2", "absl-py": "2.5.0"}
    assert MODAL_REQUIRED_VERSIONS["sglang"] == "0.5.20"
    assert MODAL_REQUIRED_VERSIONS["sglang-kernel"] == "0.4.7"
    assert MODAL_REQUIRED_VERSIONS["flashinfer-python"] == "0.6.18.post1"
    assert MODAL_REQUIRED_VERSIONS["flashinfer-cubin"] == "0.6.18.post1"
    assert MODAL_REQUIRED_VERSIONS["flash-attn-4"] == "4.0.0b32"
    assert MODAL_COMPATIBLE_SOURCE_VERSIONS["transformers"] == "5.12.1"
    assert MODAL_COMPATIBLE_SOURCE_VERSIONS["accelerate"] == "1.15.0"
    assert MODAL_REQUIRED_OVERRIDES["torch"] == "2.13.0"
    assert MODAL_CUDA_TOOLCHAIN_OVERRIDES == {
        "nvidia-cuda-cccl": "13.0.85",
        "nvidia-cuda-crt": "13.0.88",
        "nvidia-cuda-nvcc": "13.0.88",
        "nvidia-nvvm": "13.0.88",
    }
    assert not set(MODAL_CUDA_TOOLCHAIN_OVERRIDES) & set(MODAL_INSTALL_PIN_NAMES)



def test_requirements_pin_parser_reads_versions_from_direct_wheel_urls(tmp_path: Path) -> None:
    from modal_benchmark_debug import parse_requirements_pin_map

    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "torch==2.13.0\n"
        "sglang @ file:///wheelhouse/sglang-0.5.20-cp312-cp312-manylinux.whl#sha256=abc\n"
        "flashinfer-cubin @ file:///wheelhouse/flashinfer_cubin-0.6.18.post1-py3-none-any.whl#sha256=def\n",
        encoding="utf-8",
    )

    assert parse_requirements_pin_map(requirements) == {
        "torch": "2.13.0",
        "sglang": "0.5.20",
        "flashinfer-cubin": "0.6.18.post1",
    }


def test_modal_source_fingerprint_uses_requirements_not_external_venv() -> None:
    from modal_benchmark_debug import capture_requirements_fingerprint

    fingerprint = capture_requirements_fingerprint()

    assert fingerprint["kind"] == "requirements.txt"
    assert fingerprint["packages"]["torch"] == "2.13.0"
    assert fingerprint["packages"]["sglang"] == "0.5.20"
    assert fingerprint["packages"]["flashinfer-python"] == "0.6.18.post1"
    assert fingerprint["requirements_sha256"]
    assert fingerprint["all_exact_pins"]["flashinfer-cubin"] == "0.6.18.post1"
    assert fingerprint["all_exact_pins"]["vllm"] == "0.30.0"

def test_modal_package_policy_preserves_clone_pins_and_allows_required_overrides() -> None:
    from modal_benchmark_debug import (
        MODAL_COMPATIBLE_SOURCE_VERSIONS,
        MODAL_REQUIRED_OVERRIDES,
        MODAL_RUNTIME_ADDITIONS,
        modal_package_policy_issues,
    )

    source_packages = {
        **MODAL_COMPATIBLE_SOURCE_VERSIONS,
        "torch": "2.11.0",
        "flashinfer-python": "0.6.12",
    }
    modal_packages = {
        **MODAL_COMPATIBLE_SOURCE_VERSIONS,
        **MODAL_REQUIRED_OVERRIDES,
        **MODAL_RUNTIME_ADDITIONS,
    }

    assert modal_package_policy_issues(source_packages, modal_packages) == []

    modal_packages["accelerate"] = "1.14.0"
    issues = modal_package_policy_issues(source_packages, modal_packages)
    assert any("accelerate" in issue for issue in issues)


def test_modal_requirement_pin_policy_covers_the_install_set() -> None:
    from modal_benchmark_debug import (
        MODAL_IMAGE_PACKAGE_NAMES,
        MODAL_INSTALL_PIN_NAMES,
        MODAL_REQUIREMENT_PINS,
        MODAL_TORCH_SPEC,
        modal_requirement_pin_issues,
    )

    installed = {name: MODAL_REQUIREMENT_PINS[name] for name in MODAL_INSTALL_PIN_NAMES}
    installed["torch"] = "2.13.0+cu130"

    assert MODAL_TORCH_SPEC == "torch==2.13.0+cu130"
    assert "vllm" not in MODAL_IMAGE_PACKAGE_NAMES
    assert modal_requirement_pin_issues(installed) == []

    installed["sglang-kernel"] = "0.4.6"
    assert any("sglang-kernel" in issue for issue in modal_requirement_pin_issues(installed))


def test_pip_check_policy_allows_only_conflicts_between_matching_freeze_pins() -> None:
    from modal_benchmark_debug import classify_pip_check_output

    packages = {
        "sglang": "0.5.20",
        "nvidia-cutlass-dsl": "4.7.1",
        "outlines": "0.1.11",
        "outlines_core": "0.2.14",
        "accelerate": "1.15.0",
        "cuda-toolkit": "13.0.3.0",
        "nvidia-cuda-nvcc": "13.0.88",
        "vllm": "0.30.0",
        "flashinfer-python": "0.6.18.post1",
    }
    output = (
        "sglang 0.5.20 has requirement nvidia-cutlass-dsl[cu13]==4.6.2, "
        "but you have nvidia-cutlass-dsl 4.7.1.\n"
        "outlines 0.1.11 has requirement outlines_core==0.1.26, "
        "but you have outlines-core 0.2.14.\n"
        "rouge-score 0.1.2 requires absl-py, which is not installed.\n"
        "other-package 1.0 has requirement accelerate==1.0, but you have accelerate 1.15.0.\n"
        "vllm 0.30.0 has requirement flashinfer-python==0.6.18, "
        "but you have flashinfer-python 0.6.18.post1.\n"
    )

    result = classify_pip_check_output(output, packages)

    assert result["requirements_frozen_conflicts"] == [
        "sglang 0.5.20 has requirement nvidia-cutlass-dsl[cu13]==4.6.2, but you have nvidia-cutlass-dsl 4.7.1.",
        "outlines 0.1.11 has requirement outlines_core==0.1.26, but you have outlines-core 0.2.14.",
        "vllm 0.30.0 has requirement flashinfer-python==0.6.18, but you have flashinfer-python 0.6.18.post1.",
    ]
    assert len(result["errors"]) == 2


def test_modal_gpu_preflight_accepts_only_blackwell_b200() -> None:
    from modal_benchmark_debug import modal_gpu_preflight_issues

    assert modal_gpu_preflight_issues(
        {"cuda_available": True, "gpu": "NVIDIA B200", "compute_capability": [10, 0]},
        requested_gpu="B200",
    ) == []

    issues = modal_gpu_preflight_issues(
        {"cuda_available": True, "gpu": "NVIDIA H200", "compute_capability": [9, 0]},
        requested_gpu="B200",
    )
    assert any("B200" in issue for issue in issues)
    assert any("Blackwell" in issue for issue in issues)



def test_ensure_cuda_runtime_link_exposes_versioned_libcudart(tmp_path: Path) -> None:
    from modal_benchmark_debug import ensure_cuda_runtime_link

    cuda_home = tmp_path / "nvidia" / "cu13"
    runtime_lib = cuda_home / "lib"
    runtime_lib.mkdir(parents=True)
    runtime_file = runtime_lib / "libcudart.so.13.0.96"
    runtime_file.write_text("stub", encoding="utf-8")

    result = ensure_cuda_runtime_link(cuda_home)

    link = Path(result["link"])
    assert link == cuda_home / "lib64" / "libcudart.so"
    assert link.is_symlink()
    assert link.resolve() == runtime_file.resolve()


def test_child_env_propagates_requested_smoke_generation_budget(tmp_path: Path) -> None:
    from modal_benchmark_debug import _child_env

    env = _child_env(
        baselines="dflash",
        model_paths={"target": str(tmp_path / "target")},
        hf_home=tmp_path / "hf",
        run_root=tmp_path / "run",
        output_dir=tmp_path / "out",
        max_new_tokens=128,
        benchmark_mode="representative",
    )

    assert env["LONG_BENCH_MODE"] == "representative"
    assert env["LONG_BENCH_MAX_NEW_TOKENS"] == "128"
    assert env["LONG_BENCH_SMOKE_MAX_NEW_TOKENS"] == "128"
