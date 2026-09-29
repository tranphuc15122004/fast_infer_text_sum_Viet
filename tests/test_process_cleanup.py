from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")


def test_runner_timeout_gives_adapter_a_sigterm_cleanup_window(tmp_path: Path) -> None:
    from Benchmark.run_longbench_200 import _kill_child_group

    ready = tmp_path / "adapter-ready"
    terminated = tmp_path / "adapter-terminated"
    child_code = """
import signal
import sys
import time
from pathlib import Path
ready, terminated = map(Path, sys.argv[1:])
def on_term(signum, frame):
    terminated.write_text('term', encoding='utf-8')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, on_term)
ready.write_text('ready', encoding='utf-8')
while True:
    time.sleep(0.05)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(ready), str(terminated)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "adapter child did not finish installing its SIGTERM handler"

        _kill_child_group(child)

        assert terminated.read_text(encoding="utf-8") == "term"
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def test_sglang_cleanup_stops_workers_after_server_leader_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import Benchmark.infer_sglang_spec as adapter

    ready = tmp_path / "worker-ready"
    terminated = tmp_path / "worker-terminated"
    child_code = """
import signal
import sys
import time
from pathlib import Path
terminated, ready = map(Path, sys.argv[1:])
def on_term(signum, frame):
    terminated.write_text('term', encoding='utf-8')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, on_term)
ready.write_text('ready', encoding='utf-8')
while True:
    time.sleep(0.05)
"""
    leader_code = """
import subprocess
import sys
import time
from pathlib import Path
terminated, ready = sys.argv[1:3]
child_code = sys.argv[3]
child = subprocess.Popen(
    [sys.executable, '-c', child_code, terminated, ready],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
deadline = time.monotonic() + 5
while not Path(ready).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
if not Path(ready).exists():
    raise RuntimeError('worker did not become ready')
print(child.pid, flush=True)
"""
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            leader_code,
            str(terminated),
            str(ready),
            child_code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    worker_pid = None
    try:
        assert leader.stdout is not None
        worker_pid_text = leader.stdout.readline().strip()
        assert worker_pid_text, "server launcher did not report its worker PID"
        worker_pid = int(worker_pid_text)
        assert leader.wait(timeout=5) == 0

        original_group_exists = getattr(adapter, "_server_process_group_exists", None)

        def group_exists(pgid: int) -> bool:
            if terminated.exists():
                return False
            if original_group_exists is not None:
                return original_group_exists(pgid)
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True

        monkeypatch.setattr(
            adapter, "_server_process_group_exists", group_exists, raising=False
        )
        adapter._stop_process_group(leader)

        assert terminated.read_text(encoding="utf-8") == "term"
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait()
        if worker_pid is not None:
            try:
                os.kill(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_sigterm_during_sglang_spawn_waits_for_process_handle(
    tmp_path: Path,
) -> None:
    server_pid_file = tmp_path / "server.pid"
    server_ready = tmp_path / "server.ready"
    server_stopped = tmp_path / "server.stopped"
    allow_spawn_to_return = tmp_path / "allow-return"
    server_code = """
import signal
import sys
import time
from pathlib import Path
ready, stopped = map(Path, sys.argv[1:])
def on_term(signum, frame):
    stopped.write_text('term', encoding='utf-8')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, on_term)
ready.write_text('ready', encoding='utf-8')
while True:
    time.sleep(0.05)
"""
    adapter_code = """
import argparse
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, sys.argv[1])
from Benchmark import infer_sglang_spec as adapter
real_popen = subprocess.Popen
pid_file, ready, stopped, allow_return = map(Path, sys.argv[2:6])
server_code = sys.argv[6]
def fake_popen(*args, **kwargs):
    process = real_popen(
        [sys.executable, '-c', server_code, str(ready), str(stopped)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not ready.exists():
        process.kill()
        process.wait()
        raise RuntimeError('fake SGLang server failed to become ready')
    pid_file.write_text(str(process.pid), encoding='utf-8')
    while not allow_return.exists():
        time.sleep(0.01)
    return process
adapter.subprocess.Popen = fake_popen
adapter.build_server_args = lambda **kwargs: ['fake-sglang-server']
adapter._wait_ready = lambda *args, **kwargs: None
args = argparse.Namespace(
    model='fake-target', draft_model='fake-draft', port=31000,
    max_running_requests=1, tp_size=1, mem_fraction_static=0.5,
    attention_backend='triton', seed=42, disable_radix_cache=False,
    server_timeout=10, batch_size=1, paper_speedup=False,
)
adapter._run_server_phase(
    phase_method='domino', records=[], args=args,
    tokenizer=None, stop_token_ids=[],
)
"""
    adapter_process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            adapter_code,
            str(ROOT / "src"),
            str(server_pid_file),
            str(server_ready),
            str(server_stopped),
            str(allow_spawn_to_return),
            server_code,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    server_pid = None
    try:
        deadline = time.monotonic() + 10
        while (
            not server_pid_file.exists()
            and adapter_process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert server_pid_file.exists(), "adapter never launched the fake SGLang process"
        server_pid = int(server_pid_file.read_text(encoding="utf-8"))

        # Signal while Popen is still in progress and the adapter has no
        # process handle available to its SIGTERM callback yet.
        os.kill(adapter_process.pid, signal.SIGTERM)
        allow_spawn_to_return.write_text("continue", encoding="utf-8")

        deadline = time.monotonic() + 5
        while not server_stopped.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server_stopped.exists(), "SGLang survived SIGTERM during process spawn"
        assert adapter_process.wait(timeout=5) == 128 + signal.SIGTERM
    finally:
        if adapter_process.poll() is None:
            try:
                os.killpg(adapter_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            adapter_process.wait()
        if server_pid is not None and not server_stopped.exists():
            try:
                os.killpg(server_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
