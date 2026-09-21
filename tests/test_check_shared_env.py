from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_shared_env.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_checker_can_probe_a_specific_import_without_loading_models():
    result = _run(
        "--modules",
        "json",
        "--allow-python-version-mismatch",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS json" in result.stdout
    assert "from_pretrained" not in result.stdout


def test_checker_reports_all_failed_imports_instead_of_stopping_at_first():
    result = _run(
        "--modules",
        "json,fast_infer_module_that_does_not_exist",
        "--allow-python-version-mismatch",
    )

    assert result.returncode == 1
    assert "PASS json" in result.stdout
    assert "FAIL fast_infer_module_that_does_not_exist" in result.stdout
    assert "Shared environment preflight: FAIL" in result.stdout


def test_checker_contains_native_import_crashes_to_one_module(tmp_path):
    crashing_module = tmp_path / "crashing_native_module.py"
    crashing_module.write_text(
        "import os\nimport signal\nos.kill(os.getpid(), signal.SIGSEGV)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--modules",
            "json,crashing_native_module",
            "--allow-python-version-mismatch",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "PASS json" in result.stdout
    assert "FAIL crashing_native_module" in result.stdout
    assert "terminated" in result.stdout
