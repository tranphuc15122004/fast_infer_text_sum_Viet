from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _fake_python(path: Path) -> None:
    path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def test_production_runtime_does_not_use_ambient_virtualenv(tmp_path: Path) -> None:
    fake_bin = tmp_path / "system-bin"
    fake_bin.mkdir()
    system_python = fake_bin / "python3"
    _fake_python(system_python)

    virtualenv = tmp_path / "phucvenv"
    virtualenv_python = virtualenv / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    _fake_python(virtualenv_python)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "VIRTUAL_ENV": str(virtualenv),
            "FAST_INFER_PYTHON": "",
            "FAST_INFER_VENV": "",
            "FAST_INFER_SYSTEM_PYTHON": "python3",
            "FAST_INFER_CACHE_ROOT": str(tmp_path / "cache"),
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'ROOT="$1"; source "$ROOT/scripts/common/runtime.sh"; printf "%s\\n" "$FAST_INFER_PYTHON"',
            "runtime-test",
            str(ROOT),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(system_python)
