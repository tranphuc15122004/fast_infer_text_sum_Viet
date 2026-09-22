# Move Benchmark Runtime Into `src/Benchmark` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Đưa toàn bộ implementation phục vụ benchmark Việt vào package `src/Benchmark`, còn `scripts/` chỉ giữ launcher shell và các entrypoint Python mỏng tương thích.

**Architecture:** `src/Benchmark` là package Python chính, gồm runner LongBench, các baseline inference, adapter, schema/metric, preflight và data pipeline. Các wrapper trong `scripts/` chỉ resolve runtime/config, đặt `PYTHONPATH` và gọi `python -m Benchmark.<module>`; không còn import trực tiếp từ `scripts/common`.

**Tech Stack:** Python 3.12, package `src` layout, Bash launcher, pytest, các dependency/runtime hiện có của benchmark.

**Spec:** Yêu cầu refactor của người dùng và contract trong `AGENTS.md`, `docs/vietbench_evaluation.md`.

## Global Constraints

- Giữ nguyên CLI, output schema, master-config resolution và tên baseline hiện tại.
- Server production dùng `python3` Python 3.12, không tải dependency qua internet.
- `scripts/` chỉ chứa launcher; compatibility wrapper Python được phép tồn tại nhưng phải chỉ delegate sang `Benchmark`.
- Không đưa `outputs/`, `checkpoints/`, `datasets/raw/` hoặc secret vào source package.
- Mọi import benchmark nội bộ dùng namespace `Benchmark` hoặc `Benchmark.common`.

### Task 1: Lock the new package boundary with a failing test

**Files:**
- Create: `tests/test_benchmark_src_layout.py`

- [x] **Step 1: Write the failing test**

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_benchmark_runtime_is_importable_from_src_package() -> None:
    from Benchmark.common.longbench_adapter import BASELINES
    from Benchmark.run_longbench_200 import _parser

    assert BASELINES == ("vanilla_hf", "vanilla_fa", "eagle3", "dflash", "domino", "dspark")
    assert _parser().parse_args(["--mode", "smoke"]).mode == "smoke"


def test_shell_launcher_delegates_to_benchmark_package() -> None:
    launcher = (ROOT / "scripts" / "run_longbench_200.sh").read_text()
    assert "-m Benchmark.run_longbench_200" in launcher
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_benchmark_src_layout.py`

Expected: FAIL because the `Benchmark` package and new module launcher do not exist yet.

### Task 2: Move benchmark implementation into `src/Benchmark`

**Files:**
- Create: `src/Benchmark/__init__.py`
- Move: `scripts/common/*.py` -> `src/Benchmark/common/*.py`
- Move: benchmark Python modules from `scripts/` -> `src/Benchmark/`
- Move: `scripts/data/*.py` -> `src/Benchmark/data/*.py`
- Modify: moved module imports and repository-root calculations

- [x] **Step 1: Move the implementation trees**

Move the existing implementation files without changing behavior, then add package initializers for `Benchmark`, `Benchmark.common`, and `Benchmark.data`.

- [x] **Step 2: Update imports and `ROOT`/data path calculations**

Replace `common.*` imports with `Benchmark.common.*`, remove script-directory path injection, and adjust moved modules so `ROOT` remains the repository root under the `src/` layout.

- [x] **Step 3: Run import/compile checks**

Run: `PYTHONPATH=src python3 -m compileall -q src/Benchmark`

Expected: exit 0 with no syntax errors.

### Task 3: Reduce `scripts/` to launchers and preserve CLI compatibility

**Files:**
- Modify: `scripts/run_*.sh`, `scripts/run_eagle3_qwen3.sh`, `scripts/common/runtime.sh`
- Replace: Python files moved from `scripts/` with thin `python -m Benchmark...` wrappers where existing documented paths must remain valid
- Delete: `scripts/common/*.py` implementation files and `scripts/data/*.py` implementation files after wrappers are in place

- [x] **Step 1: Update shell launchers**

Export `PYTHONPATH="$ROOT/src"` and invoke `Benchmark` modules directly while preserving all existing arguments and environment setup.

- [x] **Step 2: Add compatibility entrypoints**

Keep commands such as `python3 scripts/run_longbench_200.py`, `python3 scripts/check_shared_env.py`, and `python3 scripts/audit_benchmark_metrics.py` working through minimal delegates.

- [x] **Step 3: Verify scripts contain no benchmark implementation imports**

Run: `rg -n "from common|import common|sys\.path.*scripts/common" scripts`

Expected: no matches except documentation comments that explicitly describe compatibility behavior.

### Task 4: Update tests and documentation to the new ownership boundary

**Files:**
- Modify: `tests/test_*.py` that currently prepend `scripts/` or import `common`
- Modify: `docs/baselines/*.md`, `docs/vietbench_*.md`, `README.md`, `AGENTS.md`
- Modify: source docstrings referring to `scripts/infer_*.py` as implementation

- [x] **Step 1: Update test imports**

Tests must prepend `src/` and import `Benchmark.*`; assertions and runtime behavior stay unchanged.

- [x] **Step 2: Update ownership/documentation paths**

Describe `src/Benchmark` as the implementation location and `scripts/run_*.sh` as launcher location; preserve user-facing commands.

- [x] **Step 3: Run the focused test suite**

Run: `PYTHONPATH=src pytest -q tests/test_benchmark_src_layout.py tests/test_baseline_registry.py tests/test_longbench_metric_contract.py tests/test_safe_longbench.py tests/test_sglang_adapter.py tests/test_vietbench_contract.py`

Expected: all focused tests pass.

### Task 5: Verify the complete refactor

**Files:**
- No new implementation files.

- [x] **Step 1: Run static checks**

Run: `bash -n scripts/*.sh scripts/data/*.sh 2>/dev/null || true; PYTHONPATH=src python3 -m compileall -q src/Benchmark`

- [x] **Step 2: Run CPU preflight**

Run: `DEVICE=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 scripts/run_longbench_200.py --mode smoke --baselines 'vanilla_hf vanilla_fa eagle3 dflash domino dspark' --datasets 'vietnews' --data-dir datasets/eval_100 --output-dir /tmp/fast_infer_viet_layout_preflight --preflight-only --allow-unsupported --no-collect --no-retry-failed-samples`

- [x] **Step 3: Run the full test suite**

Run: `PYTHONPATH=src pytest -q tests`

Expected: no regression in existing benchmark or Finetuning tests attributable to the package move.
