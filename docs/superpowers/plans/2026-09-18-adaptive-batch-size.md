# Adaptive Batch Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically choose the largest safe per-GPU batch size for DFlash before training while targeting a configurable fraction of each GPU's VRAM.

**Architecture:** A pure search helper handles candidate generation and fit selection. A CUDA preflight probe runs real DFlash forward/backward passes on offline feature batches, restores model/RNG state after every candidate, and synchronizes the minimum selected batch across DDP ranks. The selected batch is stored in checkpoint metadata and reused on resume.

**Tech Stack:** PyTorch CUDA memory statistics, existing DDP context, offline feature DataLoader, atomic checkpoint metadata.

**Spec:** Approved adaptive preflight design in the conversation; existing multi-GPU design in `docs/superpowers/plans/2026-09-18-finetuning-multigpu.md`.

## Global Constraints

- Adaptive tuning runs only before a fresh CUDA training run; CPU/synthetic and disabled paths remain unchanged.
- The batch size is per GPU; DDP synchronizes the minimum successful value across ranks.
- The tuner targets reserved memory below `target_memory_fraction` and never catches non-OOM runtime errors.
- The selected batch remains fixed for the entire run and is restored from checkpoint metadata on resume.
- `accumulation_steps` continues to control effective global batch size without changing instantaneous VRAM use.

### Task 1: Search and probe contract

**Files:**
- Create: `src/Finetuning/adaptive_batch.py`
- Test: `src/Finetuning/tests/test_adaptive_batch.py`

- [x] Test candidate generation, largest-fitting selection, and validation failures first.
- [x] Implement exponential plus binary-search candidate selection.
- [x] Implement CUDA probe state restoration, peak reserved-memory measurement, OOM-only failure handling, and DDP minimum reduction.

### Task 2: Configuration and runtime integration

**Files:**
- Modify: `src/Finetuning/config.py`
- Modify: `src/Finetuning/run_train.py`
- Modify: `src/Finetuning/trainer.py`
- Modify: `src/Finetuning/configs/qwen3_4b.yaml`
- Modify: `src/Finetuning/configs/qwen3_8b.yaml`
- Test: `src/Finetuning/tests/test_integration.py`

- [x] Add adaptive controls and validate their ranges.
- [x] Tune after the real draft/target modules exist but before the final DataLoader and Trainer are built.
- [x] Reuse the stored local batch size when resuming and reject incompatible effective batch metadata.
- [x] Record selected batch, target fraction, observed peak, and global batch in checkpoint metadata.

### Task 3: Documentation and verification

**Files:**
- Modify: `src/Finetuning/README.md`
- Modify: `docs/superpowers/plans/2026-09-18-adaptive-batch-size.md`

- [x] Document B200 defaults, probe behavior, global batch calculation, and fallback knobs.
- [x] Run full unit tests, CPU DDP smoke, static compilation, and a CUDA probe only when CUDA is available.
