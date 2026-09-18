# B200 End-to-End Fine-tuning Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Provide one resumable server-side launcher that runs Vietnamese DFlash regeneration, hidden-state caching, and multi-GPU training safely on B200.

**Architecture:** A Python orchestrator owns configuration materialization, validation, stage markers, file locking, logs, and subprocess execution. A thin Bash wrapper resolves the repository root and invokes the configured server Python. Each preparation/training stage runs with torch.distributed.run and shares the same filesystem. Completed artifacts are validated and resumed without rerunning; incomplete or inconsistent stages fail loudly.

**Tech Stack:** Python 3.12 standard library, PyYAML already required by src/Finetuning, Bash, torch.distributed.run, NCCL, existing Finetuning.generate_targets, Finetuning.capture_features, and Finetuning.run_train.

**Spec:** src/Finetuning/README.md and the server conventions in AGENTS.md.

## Global Constraints

- The server does not read datasets/; train/eval JSONL paths are explicit arguments.
- All model paths must be local snapshots because offline: true.
- training.batch_size remains per GPU; adaptive batch probing selects a safe fixed local batch before DDP training.
- Train and cache artifacts must be on a shared filesystem visible to all ranks.
- No stage may silently overwrite an existing completed artifact.
- Every stage writes a log and a completion marker only after its output contract is valid.
- A rerun after interruption must reuse valid completed stages and rerun only the missing stage.
- The launcher must not require yq, Modal, internet access, or an extra virtual environment.
- The B200 job command must exit nonzero on any failed subprocess or invalid artifact.

---

### Task 1: Define the launcher contract with a dry-run regression test

**Files:**
- Create: tests/test_finetuning_b200_launcher.py
- Create later: scripts/run_finetuning_b200.py

**Interfaces:**
- The launcher exposes main(argv: list[str] | None = None) -> int.
- --dry-run materializes a resolved config and prints all stage commands without invoking GPU subprocesses.
- The command accepts --config, --train-input, --eval-input, --output-root, --target-model-path, and --nproc-per-node.

- [ ] Step 1: Write the failing test. Create a valid tiny source config and local model marker, invoke the launcher with --dry-run --nproc-per-node 8, and assert output contains train/eval generate, train/eval cache, and train commands, all with eight ranks and resolved feature paths.
- [ ] Step 2: Run the test and verify it fails because the launcher does not exist:
  PYTHONPATH=src /home/tuantb/fast_infer_text_sum/.venv/bin/python -m pytest -q tests/test_finetuning_b200_launcher.py

---

### Task 2: Implement configuration materialization and preflight validation

**Files:**
- Create: scripts/run_finetuning_b200.py
- Modify: tests/test_finetuning_b200_launcher.py

**Interfaces:**
- resolve_paths(args) returns output_root, train/eval teacher paths, train/eval feature paths, run config, state, and log paths.
- materialize_config(source, destination, paths, args) updates target model, feature paths, output directory, device, run ID, and adaptive training settings without changing the source config.
- validate_preflight(config, paths, python_bin, nproc_per_node) checks local model snapshot, input JSONL files, PyTorch/CUDA, required modules, and positive GPU count.
- The materialized config sets data.train_data_path and data.eval_data_path to null because run_train consumes only captured features.

- [ ] Step 1: Add failing assertions for the resolved feature paths, target model path, and training.adaptive_batch_size: true.
- [ ] Step 2: Run the focused test and verify RED.
- [ ] Step 3: Implement dataclass-based paths and YAML materialization. Use yaml.safe_load, validate mapping sections, write the resolved YAML through a temporary sibling and os.replace, and reject a source config whose model dtype and feature dtype differ because the current capture CLI stores the requested model dtype.
- [ ] Step 4: Run the focused test and verify GREEN.

---

### Task 3: Implement resumable stage execution, logs, lock, and artifact checks

**Files:**
- Modify: scripts/run_finetuning_b200.py
- Modify: tests/test_finetuning_b200_launcher.py

**Interfaces:**
- run_stage(name, command, log_path, marker_path, validator, dry_run) skips only when marker and validator both pass; recovers a valid artifact when a process finished before marker publication; streams stdout/stderr to a stage log; writes a JSON marker only after validation; and raises on nonzero exit or invalid output.
- Teacher validator requires a nonempty valid JSONL file.
- Feature validator requires manifest.json, a valid generation_dir, and at least one feature_*.pt.
- Training validator requires a complete run_id-stepN checkpoint containing COMPLETE.
- A nonblocking fcntl.flock lock at output_root/.run.lock prevents duplicate jobs.

- [ ] Step 1: Add a failing resume test with a valid teacher artifact and marker; assert the second dry-run says SKIP rather than RUN.
- [ ] Step 2: Run focused tests and verify RED.
- [ ] Step 3: Implement markers containing stage name, UTC timestamp, command, and artifact path. Logs live under output_root/logs/<stage>.log. Use start_new_session=True and terminate the current process group on SIGTERM/SIGINT.
- [ ] Step 4: Run focused tests and verify GREEN.

---

### Task 4: Implement the end-to-end distributed command graph

**Files:**
- Modify: scripts/run_finetuning_b200.py
- Create: scripts/run_finetuning_b200.sh
- Modify: tests/test_finetuning_b200_launcher.py

**Interfaces:**
- Stage order is exactly generate_train, generate_eval, cache_train, cache_eval, train.
- Every stage uses the same --standalone --nproc_per_node=N launcher.
- Generation and caching receive adaptive target fraction, min/max batch, token cap, bucket window, probe count, and OOM backoff controls.
- Cache layer selection is derived from model.target_layer_ids or model.num_draft_layers, matching the training config.
- Training uses the materialized config and --device cuda.
- The environment exports PYTHONPATH=<repo>/src, offline Transformers/HF flags, unbuffered output, and NCCL async error handling.
- The Bash wrapper invokes FINETUNING_PYTHON or python3 and does not require Modal.

- [ ] Step 1: Add command assertions for both inputs, both teacher paths, both feature paths, --nproc_per_node 8, --adaptive-batch, and Finetuning.run_train --config.
- [ ] Step 2: Run focused tests and verify RED.
- [ ] Step 3: Implement commands using sys.executable -m torch.distributed.run rather than a separate torchrun binary, preserving the selected server interpreter. Do not use scripts/run_dflash.sh because it is an inference launcher.
- [ ] Step 4: Run focused tests and verify GREEN.

---

### Task 5: Document the long-running B200 operation

**Files:**
- Modify: src/Finetuning/README.md
- Modify: scripts/run_finetuning_b200.sh

**Interfaces:**
- Document one command for a fresh run and one command for resume.
- Document output layout, logs, marker files, shared filesystem requirement, and how to inspect adaptive VRAM metadata.
- Document that --nproc-per-node defaults to nvidia-smi -L count and must match visible GPUs.
- Document that the Modal fixture smoke is not production-data validation.

- [ ] Step 1: Add this server command:
  bash scripts/run_finetuning_b200.sh --config src/Finetuning/configs/qwen3_4b.yaml --train-input /server/data/train.jsonl --eval-input /server/data/eval.jsonl --target-model-path /server/models/Qwen3-4B --output-root /server/work/dflash-qwen3-4b
- [ ] Step 2: Add the resume command with the same output-root.
- [ ] Step 3: Verify documentation flags against bash scripts/run_finetuning_b200.sh --help.

---

### Task 6: Verify the implementation

**Files:**
- No new files.

- [ ] Step 1: Run bash -n scripts/run_finetuning_b200.sh and python -m py_compile scripts/run_finetuning_b200.py.
- [ ] Step 2: Run the focused launcher tests.
- [ ] Step 3: Run PYTHONPATH=src /home/tuantb/fast_infer_text_sum/.venv/bin/python -m pytest -q src/Finetuning/tests.
- [ ] Step 4: Run a dry-run with eight fake ranks and confirm no GPU subprocess is launched and all five stage commands are printed.
- [ ] Step 5: Keep the Modal B200 smoke as the remote CUDA validation; the launcher is validated locally by dry-run and on-server by the documented command.

