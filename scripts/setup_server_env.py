#!/usr/bin/env python3
"""Khởi tạo và kiểm tra runtime server cho benchmark tóm tắt tiếng Việt.

Mặc định script chỉ bootstrap master config/pointer và chạy preflight import.
Việc cài package phải bật rõ ``--install-dependencies``; khi server offline,
truyền ``--offline --wheelhouse`` để pip không truy cập public index.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


DEFAULT_SHARED_DATA_DIR = Path(
    "/workspace/storage-shared/nlp/dungdx4/phuc_projects/data"
)
MASTER_FILENAME = "fast_infer_master_Viet.env"
DATASETS = ("vietnews", "wikilingua", "vims", "vlsp")
MANAGED_BEGIN = "# BEGIN setup_server_env.py managed defaults"
MANAGED_END = "# END setup_server_env.py managed defaults"


class SetupError(RuntimeError):
    """Lỗi setup có thông báo đủ rõ cho operator xử lý."""


def _resolve(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _shell_value(value: str | Path) -> str:
    return shlex.quote(str(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--setup",
        "--init",
        dest="setup",
        action="store_true",
        help="tạo/cập nhật master pointer và master config, không check",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="chỉ kiểm tra, không tạo hoặc sửa file repository",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="setup rồi chạy toàn bộ preflight (mặc định)",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="root repository (mặc định: vị trí script)",
    )
    parser.add_argument(
        "--shared-data-dir",
        default=str(DEFAULT_SHARED_DATA_DIR),
        help=f"thư mục chứa master config (mặc định: {DEFAULT_SHARED_DATA_DIR})",
    )
    parser.add_argument(
        "--master-config",
        default=None,
        help="master config override; mặc định <shared-data-dir>/fast_infer_master_Viet.env",
    )
    parser.add_argument(
        "--master-example",
        default=None,
        help="file mẫu để tạo master mới (mặc định: <repo>/fast_infer_master_Viet.env)",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="interpreter cần kiểm tra/cài package (mặc định: python3)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="thư mục eval_100; mặc định <repo>/datasets/eval_100",
    )
    parser.add_argument(
        "--profile",
        choices=("minimal", "server"),
        default="server",
        help="dependency profile cho preflight (mặc định: server)",
    )
    parser.add_argument(
        "--modules",
        default=None,
        help="ghi đè module list truyền cho checker",
    )
    parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="bỏ qua import preflight; chỉ debug filesystem",
    )
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="bỏ qua kiểm tra 4 file eval_100",
    )
    parser.add_argument(
        "--install-dependencies",
        "--install",
        dest="install_dependencies",
        action="store_true",
        help="cài requirements bằng interpreter đã chọn",
    )
    parser.add_argument(
        "--requirements",
        default=None,
        help="manifest requirements (mặc định: <repo>/requirements.txt)",
    )
    parser.add_argument(
        "--wheelhouse",
        default=None,
        help="thư mục wheel offline; mặc định dùng B200_WHEELHOUSE nếu có",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="pip chỉ dùng wheelhouse, tuyệt đối không truy cập index",
    )
    return parser


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    repo = _resolve(args.repo_dir)
    shared = _resolve(args.shared_data_dir)
    configured_master = (
        args.master_config
        or os.environ.get("FAST_INFER_MASTER_CONFIG")
        or str(shared / MASTER_FILENAME)
    )
    return {
        "repo": repo,
        "shared": shared,
        "master": _resolve(configured_master),
        "pointer": repo / "config" / "master.path",
        "example": _resolve(args.master_example)
        if args.master_example
        else repo / "fast_infer_master_Viet.env",
        "data": _resolve(args.data_dir)
        if args.data_dir
        else repo / "datasets" / "eval_100",
        "requirements": _resolve(args.requirements)
        if args.requirements
        else repo / "requirements.txt",
        "checker": repo / "scripts" / "check_shared_env.py",
    }


def _generated_master_block(paths: dict[str, Path]) -> str:
    return (
        f"\n\n{MANAGED_BEGIN}\n"
        "FI_PYTHON=python3\n"
        "FI_OFFLINE=1\n"
        "FI_DEVICE=cuda\n"
        f"DATA_ROOT={_shell_value(paths['shared'])}\n"
        f"LONG_BENCH_DATA_DIR={_shell_value(paths['data'])}\n"
        f"LONG_BENCH_OUTPUT_DIR={_shell_value(paths['repo'] / 'outputs' / 'longbench_viet_100')}\n"
        "LONG_BENCH_LOCAL_FILES_ONLY=1\n"
        f"{MANAGED_END}\n"
    )


def _refresh_managed_master(text: str, block: str) -> str | None:
    begin = text.find(MANAGED_BEGIN)
    end_marker = text.find(MANAGED_END, begin if begin >= 0 else 0)
    if begin < 0 or end_marker < 0:
        return None
    start = text.rfind("\n", 0, begin) + 1
    end = text.find("\n", end_marker)
    end = len(text) if end < 0 else end + 1
    return text[:start].rstrip("\n") + block + text[end:]


def _init_master(paths: dict[str, Path]) -> None:
    master = paths["master"]
    master.parent.mkdir(parents=True, exist_ok=True)
    block = _generated_master_block(paths)
    if master.exists():
        current = master.read_text(encoding="utf-8")
        refreshed = _refresh_managed_master(current, block)
        if refreshed is not None and refreshed != current:
            master.write_text(refreshed, encoding="utf-8")
            print(f"UPDATE managed master defaults: {master}")
        else:
            print(f"KEEP master config: {master}")
        return
    example = paths["example"]
    if not example.is_file():
        raise SetupError(f"không tìm thấy master example: {example}")
    master.write_text(
        example.read_text(encoding="utf-8") + block,
        encoding="utf-8",
    )
    print(f"CREATE master config: {master}")


def _pointer_target(pointer: Path) -> Path | None:
    if not pointer.is_file():
        return None
    for line in pointer.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            return _resolve(value)
    return None


def _init_pointer(paths: dict[str, Path]) -> None:
    pointer = paths["pointer"]
    pointer.parent.mkdir(parents=True, exist_ok=True)
    expected = str(paths["master"]) + "\n"
    current = pointer.read_text(encoding="utf-8") if pointer.exists() else ""
    if _pointer_target(pointer) == paths["master"] and current.endswith(expected):
        print(f"KEEP master pointer: {pointer}")
        return
    pointer.write_text(
        "# Đường dẫn master config Việt do setup_server_env.py quản lý.\n"
        + expected,
        encoding="utf-8",
    )
    print(f"UPDATE master pointer: {pointer} -> {paths['master']}")


def initialize(paths: dict[str, Path]) -> None:
    paths["shared"].mkdir(parents=True, exist_ok=True)
    _init_master(paths)
    _init_pointer(paths)
    print(f"KEEP repository eval data: {paths['data']}")


def _python_executable(value: str) -> str:
    candidate = shutil.which(value) if "/" not in value else value
    if not candidate or not os.access(candidate, os.X_OK):
        raise SetupError(f"không tìm thấy executable Python: {value}")
    return candidate


def _check_python(value: str) -> str:
    executable = _python_executable(value)
    probe = subprocess.run(
        [
            executable,
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}'); "
            "sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    version = probe.stdout.strip() or "unknown"
    if probe.returncode != 0:
        raise SetupError(
            f"Python server phải là 3.12, nhưng {executable} trả về {version}"
        )
    print(f"PASS Python: {executable} ({version})")
    return executable


def _check_master(paths: dict[str, Path]) -> None:
    master = paths["master"]
    if not master.is_file():
        raise SetupError(f"master config chưa tồn tại: {master}; chạy --setup trước")
    if _pointer_target(paths["pointer"]) != master:
        raise SetupError(
            f"master pointer không trỏ tới {master}: {paths['pointer']}; chạy --setup"
        )
    syntax = subprocess.run(
        ["bash", "-n", str(master)], text=True, capture_output=True, check=False
    )
    if syntax.returncode != 0:
        raise SetupError(f"master config sai Bash syntax: {syntax.stderr.strip()}")
    print(f"PASS master config: {master}")


def _check_data(paths: dict[str, Path]) -> None:
    data_dir = paths["data"]
    if not data_dir.is_dir():
        raise SetupError(f"thư mục eval_100 chưa tồn tại: {data_dir}")
    for dataset in DATASETS:
        path = data_dir / f"{dataset}_100.jsonl"
        if not path.is_file():
            raise SetupError(f"thiếu dataset file: {path}")
        rows = []
        try:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record không phải object")
                    rows.append(row)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise SetupError(f"dataset không hợp lệ {path}: {exc}") from exc
        if len(rows) != 100:
            raise SetupError(f"{path}: cần 100 records, nhận {len(rows)}")
        for index, row in enumerate(rows, start=1):
            if not str(row.get("id") or "").strip():
                raise SetupError(f"{path}:{index}: thiếu id")
            if not str(row.get("document") or "").strip():
                raise SetupError(f"{path}:{index}: thiếu document")
            if not str(row.get("reference") or "").strip():
                raise SetupError(f"{path}:{index}: thiếu reference")
        print(f"PASS dataset: {path} (100 records)")


def _install_dependencies(paths: dict[str, Path], args: argparse.Namespace, python: str) -> None:
    requirements = paths["requirements"]
    if not requirements.is_file():
        raise SetupError(f"không tìm thấy requirements manifest: {requirements}")
    wheelhouse_value = args.wheelhouse or os.environ.get("B200_WHEELHOUSE")
    offline = args.offline or any(
        os.environ.get(name) == "1"
        for name in ("B200_OFFLINE", "FI_OFFLINE")
    )
    command = [python, "-m", "pip", "install", "--prefer-binary", "-r", str(requirements)]
    if offline:
        if not wheelhouse_value:
            raise SetupError("offline install cần --wheelhouse hoặc B200_WHEELHOUSE")
        wheelhouse = _resolve(wheelhouse_value)
        if not wheelhouse.is_dir():
            raise SetupError(f"wheelhouse không tồn tại: {wheelhouse}")
        command.extend(["--no-index", "--find-links", str(wheelhouse)])
    elif wheelhouse_value:
        wheelhouse = _resolve(wheelhouse_value)
        if not wheelhouse.is_dir():
            raise SetupError(f"wheelhouse không tồn tại: {wheelhouse}")
        command.extend(["--find-links", str(wheelhouse)])
    print("RUN:", " ".join(shlex.quote(part) for part in command))
    env = dict(os.environ)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    subprocess.run(command, env=env, check=True)
    print("PASS dependency installation")


def _check_dependencies(paths: dict[str, Path], python: str, args: argparse.Namespace) -> None:
    if args.skip_dependencies:
        print("SKIP dependency preflight: --skip-dependencies")
        return
    checker = paths["checker"]
    if not checker.is_file():
        raise SetupError(f"không tìm thấy dependency checker: {checker}")
    command = [python, str(checker), "--profile", args.profile]
    if args.modules:
        command.extend(["--modules", args.modules])
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    result = subprocess.run(command, env=env, text=True, check=False)
    if result.returncode != 0:
        raise SetupError("dependency preflight thất bại; kiểm tra package/CUDA stack")
    print("PASS shared dependency preflight")


def check(paths: dict[str, Path], args: argparse.Namespace) -> None:
    python = _check_python(args.python)
    _check_master(paths)
    if args.skip_data_validation:
        print("SKIP eval_100 validation: --skip-data-validation")
    else:
        _check_data(paths)
    _check_dependencies(paths, python, args)
    print("\nVietnamese server environment preflight: PASS")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _paths(args)
    try:
        if args.check:
            check(paths, args)
        elif args.setup:
            initialize(paths)
            if args.install_dependencies:
                python = _check_python(args.python)
                _install_dependencies(paths, args, python)
            print("\nVietnamese server environment setup: PASS")
        else:
            initialize(paths)
            if args.install_dependencies:
                python = _check_python(args.python)
                _install_dependencies(paths, args, python)
            check(paths, args)
    except (OSError, SetupError, subprocess.CalledProcessError) as exc:
        print(f"Vietnamese server environment: FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
