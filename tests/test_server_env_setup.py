from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup_server_env.py"


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    (repo / "datasets" / "eval_100").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "fast_infer_master_Viet.env").write_text(
        "MODEL_TARGET=/models/operator-target\n"
        "LONG_BENCH_DATA_DIR=datasets/eval_100\n",
        encoding="utf-8",
    )
    return repo


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    shared = repo.parent / "shared-data"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-dir",
            str(repo),
            "--shared-data-dir",
            str(shared),
            "--master-example",
            str(repo / "fast_infer_master_Viet.env"),
            *args,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_setup_creates_viet_master_pointer_and_preserves_operator_config(tmp_path):
    repo = _make_repo(tmp_path)

    result = _run(repo, "--setup")

    assert result.returncode == 0, result.stdout + result.stderr
    master = tmp_path / "shared-data" / "fast_infer_master_Viet.env"
    assert master.is_file()
    pointer = repo / "config" / "master.path"
    assert pointer.read_text(encoding="utf-8").splitlines()[-1] == str(master)
    assert "MODEL_TARGET=/models/operator-target" in master.read_text(
        encoding="utf-8"
    )


def test_check_is_read_only_when_viet_master_is_missing(tmp_path):
    repo = _make_repo(tmp_path)

    result = _run(repo, "--check", "--skip-dependencies", "--skip-data-validation")

    assert result.returncode != 0
    assert not (tmp_path / "shared-data").exists()
    assert not (repo / "config" / "master.path").exists()


def test_setup_refreshes_only_its_managed_defaults(tmp_path):
    repo = _make_repo(tmp_path)
    first = _run(repo, "--setup")
    assert first.returncode == 0, first.stdout + first.stderr

    master = tmp_path / "shared-data" / "fast_infer_master_Viet.env"
    current = master.read_text(encoding="utf-8")
    master.write_text(
        current.replace("datasets/eval_100", "datasets/old_eval")
        + "\nMODEL_TARGET=/models/changed-by-operator\n",
        encoding="utf-8",
    )

    second = _run(repo, "--setup")

    assert second.returncode == 0, second.stdout + second.stderr
    refreshed = master.read_text(encoding="utf-8")
    assert "MODEL_TARGET=/models/changed-by-operator" in refreshed
    assert "LONG_BENCH_DATA_DIR=" + str(repo / "datasets" / "eval_100") in refreshed
    managed = refreshed.split("# BEGIN setup_server_env.py managed defaults", 1)[1]
    assert "datasets/old_eval" not in managed


def test_install_honors_fi_offline_without_reaching_a_package_index(tmp_path):
    repo = _make_repo(tmp_path)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    fake_python = tmp_path / "python312"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-c\" ]; then echo 3.12; exit 0; fi\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ]; then exit 0; fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = dict(os.environ)
    env["FI_OFFLINE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-dir",
            str(repo),
            "--shared-data-dir",
            str(tmp_path / "shared-data"),
            "--master-example",
            str(repo / "fast_infer_master_Viet.env"),
            "--setup",
            "--install",
            "--python",
            str(fake_python),
            "--requirements",
            str(requirements),
            "--wheelhouse",
            str(wheelhouse),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--no-index" in result.stdout
