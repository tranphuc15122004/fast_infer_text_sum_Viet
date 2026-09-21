#!/usr/bin/env python3
"""Launch one OpenAI-compatible SGLang/vLLM target server per GPU group."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class ServerSpec:
    backend: str
    port: int
    gpu_group: str
    command: tuple[str, ...]
    env: dict[str, str]


def _gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([item for item in visible.split(",") if item.strip()])
    result = subprocess.run(
        ["nvidia-smi", "-L"], text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi -L failed: {result.stderr.strip()}")
    return sum(bool(line.strip()) for line in result.stdout.splitlines())


def _default_gpu_groups() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    count = _gpu_count()
    if count <= 0:
        raise RuntimeError("no visible GPU found; pass --gpu-group explicitly")
    return [str(index) for index in range(count)]


def build_server_specs(args: argparse.Namespace) -> list[ServerSpec]:
    """Build deterministic child commands for all endpoint processes."""

    if args.backend not in {"sglang", "vllm"}:
        raise ValueError("backend must be sglang or vllm")
    if args.tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if not 0 < args.mem_fraction < 1:
        raise ValueError("mem_fraction must be in (0, 1)")
    groups = [str(group).strip() for group in args.gpu_groups if str(group).strip()]
    if not groups:
        raise ValueError("at least one --gpu-group is required")
    python = str(getattr(args, "python", sys.executable))

    specs: list[ServerSpec] = []
    for offset, group in enumerate(groups):
        port = int(args.base_port) + offset
        if args.backend == "sglang":
            command = [
                python,
                "-m",
                "sglang.launch_server",
                "--model-path",
                str(args.model_path),
                "--host",
                str(args.host),
                "--port",
                str(port),
                "--tp-size",
                str(args.tp_size),
                "--dtype",
                str(args.dtype),
                "--mem-fraction-static",
                str(args.mem_fraction),
            ]
            if args.context_length is not None:
                command.extend(["--context-length", str(args.context_length)])
            if args.max_num_seqs:
                command.extend(["--max-running-requests", str(args.max_num_seqs)])
        else:
            command = [
                python,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(args.model_path),
                "--host",
                str(args.host),
                "--port",
                str(port),
                "--tensor-parallel-size",
                str(args.tp_size),
                "--dtype",
                str(args.dtype),
                "--gpu-memory-utilization",
                str(args.mem_fraction),
            ]
            if args.context_length is not None:
                command.extend(["--max-model-len", str(args.context_length)])
            if args.max_num_seqs:
                command.extend(["--max-num-seqs", str(args.max_num_seqs)])
        if args.served_model_name:
            command.extend(["--served-model-name", str(args.served_model_name)])
        specs.append(
            ServerSpec(
                backend=args.backend,
                port=port,
                gpu_group=group,
                command=tuple(command),
                env={"CUDA_VISIBLE_DEVICES": group},
            )
        )
    return specs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sglang", "vllm"), default="sglang")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu-group", dest="gpu_groups", action="append", default=[])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mem-fraction", type=float, default=0.88)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--served-model-name")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/target_servers/logs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _terminate(processes: Sequence[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.gpu_groups:
        args.gpu_groups = _default_gpu_groups()
    specs = build_server_specs(args)
    for spec in specs:
        command = shlex.join(list(spec.command))
        print(f"CUDA_VISIBLE_DEVICES={spec.gpu_group} {command}")
    if args.dry_run:
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[str]] = []
    logs: list[object] = []
    try:
        for index, spec in enumerate(specs):
            log = (args.log_dir / f"{args.backend}_{spec.port}_{index}.log").open(
                "a", encoding="utf-8"
            )
            logs.append(log)
            environment = os.environ.copy()
            environment.update(spec.env)
            process = subprocess.Popen(
                list(spec.command),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
        print(f"started {len(processes)} {args.backend} endpoint(s); press Ctrl-C to stop")
        return_code = 0
        for process in processes:
            if process.wait() != 0:
                return_code = 1
        return return_code
    except KeyboardInterrupt:
        return 130
    finally:
        _terminate(processes)
        for log in logs:
            log.close()  # type: ignore[union-attr]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ServerSpec", "build_server_specs", "main"]
