# DFlash Multi-GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add single-node `torchrun` support to the Qwen3 DFlash pipeline while preserving the existing single-GPU commands.

**Architecture:** Use one process per GPU. DDP wraps the trainable DFlash model, while the frozen target embedding/lm_head remain replicated. Distributed samplers shard offline features and rank zero owns metrics/checkpoint publication. Teacher generation and feature capture use deterministic rank shards and merge atomic outputs on rank zero.

**Tech Stack:** PyTorch Distributed/DDP, `torchrun`, `DistributedSampler`, existing offline feature manifest and atomic checkpoint layers.

**Spec:** `src/Finetuning/plans/2026-09-07-dflash-specforge-training.md` plus the approved DDP single-node design in the conversation.

## Global Constraints

- Existing single-GPU commands must continue to work unchanged.
- Training uses `sdpa` by default; no Flex Attention changes are included.
- Only the DFlash draft is trainable; target embedding and lm_head remain frozen.
- Rank zero is the only process that writes shared checkpoints, metrics, and merged artifacts.
- Distributed tests must run on CPU with the `gloo` backend; B200 execution is a follow-up validation.

### Task 1: Distributed runtime contract

**Files:**
- Create: `src/Finetuning/distributed.py`
- Test: `src/Finetuning/tests/test_distributed.py`

- [x] Add environment-aware rank/world-size discovery, process-group initialization, device selection, barrier, scalar reduction, and cleanup.
- [x] Add tests for single-process fallback and rank-shard index arithmetic without requiring CUDA.

### Task 2: DDP training and distributed data loading

**Files:**
- Modify: `src/Finetuning/strategy.py`
- Modify: `src/Finetuning/run_train.py`
- Modify: `src/Finetuning/trainer.py`
- Modify: `src/Finetuning/evaluation.py`
- Test: `src/Finetuning/tests/test_distributed_training.py`

- [x] Write a two-process CPU test that verifies DDP gradients synchronize and only rank zero publishes a checkpoint.
- [x] Add `DistributedSampler` support and call `set_epoch` for deterministic reshuffling.
- [x] Wrap only the DFlash model forward path in DDP while keeping checkpoint metadata/state keys draft-local.
- [x] Correct additive loss scaling across ranks using the global denominator.
- [x] Reduce validation metrics across ranks and synchronize checkpoint publication.
- [x] Preserve the single-process Trainer API.

### Task 3: Distributed teacher generation and hidden-state capture

**Files:**
- Modify: `src/Finetuning/generate_targets.py`
- Modify: `src/Finetuning/capture_features.py`
- Create or modify: `src/Finetuning/tests/test_distributed_preparation.py`

- [x] Shard input records by source index under `torchrun`.
- [x] Write rank-local temporary JSONL/feature stores.
- [x] Merge records and feature files on rank zero in source order with duplicate detection.
- [x] Preserve atomic publication and the existing manifest contract.
- [x] Keep non-distributed behavior unchanged.

### Task 4: CLI, documentation, and verification

**Files:**
- Modify: `src/Finetuning/README.md`
- Modify: `src/Finetuning/__init__.py`
- Modify: `src/Finetuning/tests/validate_runtime.py`

- [x] Document `torchrun` commands, effective global batch size, shared filesystem requirements, and rank-zero artifacts.
- [x] Add distributed smoke validation and run the complete unit suite plus two-process CPU validation.
