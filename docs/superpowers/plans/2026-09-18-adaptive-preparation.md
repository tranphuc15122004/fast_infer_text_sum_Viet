# Adaptive Regeneration and Feature-Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add length-aware adaptive batching to regeneration and hidden-state caching so each B200 uses a safe high-VRAM batch without changing artifacts.

**Experiment directory:** `docs/superpowers/`

**Hypothesis:** Length-bucketed token batching with CUDA preflight and OOM backoff increases preparation throughput while preserving target outputs and feature contracts.

**Validation scope:** Unit tests, CPU fake-model tests, existing full Finetuning tests, and two-process DDP preparation smoke. CUDA VRAM validation is run only on a CUDA host.

**Evaluation design:** Preparation evaluation is throughput/memory oriented rather than model quality oriented. Each phase reports samples, tokens, elapsed time, samples/sec, tokens/sec, peak reserved VRAM, and OOM retries; output parity and feature-shape contracts are mandatory.

**Architecture:** A shared preparation module plans deterministic length buckets and executes bounded adaptive batches. Generation and feature capture provide phase-specific callbacks for real forward work, while preserving the existing rank-local atomic writers and rank-0 merge.

**Tech Stack:** PyTorch, Transformers, existing `DistributedContext`, offline JSONL/feature-store formats, pytest.

## Shared Scaffold

### Existing infra (do not replace)

- Target sharding and collectives: `src/Finetuning/distributed.py`
- Teacher generation: `src/Finetuning/generate_targets.py`
- Feature capture and atomic publication: `src/Finetuning/capture_features.py`
- Prompt/token preparation: `src/Finetuning/data.py`, `src/Finetuning/prepare_data.py`
- Feature schema and validation: `src/Finetuning/features.py`
- Existing preparation tests: `src/Finetuning/tests/test_teacher_generation.py`, `test_distributed_preparation.py`

### New shared controls

- Create `src/Finetuning/adaptive_inference.py` with:
  - `AdaptiveInferenceSettings` dataclass;
  - `PreparedExample` and `AdaptiveBatch` typed records;
  - `length_bucket_batches(...)` deterministic bounded planner;
  - `adaptive_cuda_batch(...)` exponential preflight plus binary refinement;
  - `run_with_oom_backoff(...)` that retries only CUDA OOM and returns metrics.
- Add CLI controls to both preparation commands and YAML-compatible defaults:
  `--adaptive-batch`, `--target-memory-fraction`, `--adaptive-max-batch-size`,
  `--max-tokens-per-batch`, `--bucket-window`, `--probe-batches`, and
  `--no-adaptive-batch`.

## Subtask 1: Shared adaptive inference planner

**Role:** Provide deterministic length bucketing, token-budget planning, CUDA memory probing, and OOM-only backoff used by both preparation phases.

**Implementation:** Add the shared module and configuration validation. A batch planner sorts only within a bounded window by `(length, source_index)`, groups adjacent examples, and respects both sample and token limits. The CUDA probe executes a caller-supplied forward callback on representative batches, resets peak stats, targets 90% of total device memory, and restores the allocator state between candidates. Backoff halves the batch and retries the same work; unrelated exceptions propagate.

**Unit Tests:** `src/Finetuning/tests/test_adaptive_inference.py` covers deterministic buckets, no cross-window reordering, token budget, invalid settings, successful probe search, OOM retry, and propagation of non-OOM errors. **Status: complete; 5 tests pass.**

### Step 1: Write failing tests

Add tests for the exact public interfaces:

```python
def test_length_bucket_batches_respects_window_and_token_budget():
    examples = [Example(index, length) for index, length in enumerate([9, 2, 8, 3])]
    batches = list(length_bucket_batches(examples, batch_size=3, max_tokens=10, window=4))
    assert [[item.index for item in batch] for batch in batches] == [[1, 3], [2], [0]]

def test_oom_backoff_retries_with_half_batch():
    attempts = []
    def work(batch_size):
        attempts.append(batch_size)
        if batch_size > 2:
            raise RuntimeError("CUDA out of memory")
        return batch_size
    assert run_with_oom_backoff(work, initial_batch_size=8, minimum_batch_size=1) == 2
    assert attempts == [8, 4, 2]
```

### Step 2: Run tests and verify RED

Run:

```bash
PYTHONPATH=src pytest -q src/Finetuning/tests/test_adaptive_inference.py
```

Expected: collection failure because `Finetuning.adaptive_inference` does not exist.

### Step 3: Implement shared module

Implement the typed records, planner, settings validation, CUDA probe, and OOM backoff in `src/Finetuning/adaptive_inference.py`. Keep CPU tests independent of CUDA by injecting the work callback and memory provider.

### Step 4: Run tests and verify GREEN

Run the same command. Expected: all shared planner tests pass.

## Subtask 2: Batched adaptive regeneration

**Role:** Increase target trajectory generation throughput while preserving exact greedy outputs and record identity.

**Implementation:** Modify `generate_targets.py` to buffer a bounded rank-local window, use the tokenizer to create left-padded batches with attention masks, run `target.generate` in inference mode, extract continuation tokens using each row's true prompt length, and write records in source-index order. Use the shared planner and backoff. Add phase metrics to rank-0 JSON output and preserve the existing atomic rank-local merge.

**Unit Tests:** Extend `test_teacher_generation.py` with a batched fake tokenizer/target that returns one continuation per row, verifies prompt-length trimming, preserves metadata/source order, and retries an injected CUDA OOM. Extend `test_distributed_preparation.py` with batch planner integration using a fake model. **Status: complete; generation regression tests pass.**

### Step 1: Write failing tests

Add a batched generation fixture with two different prompt lengths and assert decoded summaries match the single-example contract, while the fake target records batch sizes.

### Step 2: Run tests and verify RED

Run:

```bash
PYTHONPATH=src pytest -q src/Finetuning/tests/test_teacher_generation.py src/Finetuning/tests/test_distributed_preparation.py
```

Expected: failure because the generation API still invokes one example at a time.

### Step 3: Implement batched generation

Add bounded buffering, left padding, attention masks, row-wise extraction, adaptive retry, and metrics. Do not alter greedy decoding flags or JSON schema.

### Step 4: Run tests and verify GREEN

Run the same command. Expected: existing and new generation tests pass.

## Subtask 3: Batched adaptive feature caching

**Role:** Increase hidden-state capture throughput while preserving per-record feature shapes, dtype, manifest, and atomic publication.

**Implementation:** Modify `capture_features.py` to batch prepared examples from a bounded window, pad input IDs and attention masks, run the frozen target with `output_hidden_states=True` and `use_cache=False`, trim selected layers back to each row's true length, and atomically write each record. Use the same adaptive planner/backoff and report phase metrics.

**Unit Tests:** Extend `test_features.py` with a batched fake model returning `[batch, padded_seq, hidden]` hidden states and assert each output record is trimmed correctly. Test that the manifest and feature dtype stay unchanged. **Status: complete; feature regression tests pass.**

### Step 1: Write failing tests

Add a two-row batch capture test with lengths 3 and 5 and assert output shapes `(3, feature_width)` and `(5, feature_width)` plus exact CPU dtype.

### Step 2: Run tests and verify RED

Run:

```bash
PYTHONPATH=src pytest -q src/Finetuning/tests/test_features.py
```

Expected: failure because capture currently calls the single-example path only.

### Step 3: Implement batched capture

Add batch collation, attention masks, row trimming, retry handling, and metrics while retaining generation directory atomicity and rank-local file naming.

### Step 4: Run tests and verify GREEN

Run the same command. Expected: existing and new feature-capture tests pass.

## Subtask 4: Preparation CLI and full pipeline [INTEGRATION]

**Role:** Assemble the shared planner, regeneration, caching, CLI controls, DDP merge, and observability into the delivered preparation pipeline.

**Implementation:** Wire CLI flags in both modules, validate ranges, make adaptive mode the default for CUDA preparation while retaining `--no-adaptive-batch` for baseline comparison, and print structured rank-0 metrics. Keep CPU/synthetic tests deterministic and leave existing output paths/manifest contracts intact.

**Integration Tests:** Run the complete preparation test set, CPU two-process DDP smoke, static compilation, and a CUDA-only smoke when available. Verify output parity between adaptive and single-example fixtures. **Status: complete on CPU; CUDA host validation remains pending.**

### Step 1: Write integration assertions

Assert CLI parsing exposes the same adaptive controls for both modules and that a disabled adaptive mode retains the current single-example behavior.

### Step 2: Run integration tests and verify RED

Run:

```bash
PYTHONPATH=src pytest -q src/Finetuning/tests/test_teacher_generation.py src/Finetuning/tests/test_features.py src/Finetuning/tests/test_distributed_preparation.py
```

Expected: new CLI/batched integration assertions fail before wiring.

### Step 3: Assemble and document

Update `src/Finetuning/README.md` with adaptive preparation commands, B200 controls, expected metrics, and fallback knobs. Update `src/Finetuning/configs/qwen3_4b.yaml` and `qwen3_8b.yaml` with preparation defaults only if those configs expose preparation sections; otherwise document the CLI as the source of truth.

### Step 4: Run the complete verification set

Run:

```bash
python -m compileall -q src/Finetuning
git diff --check
PYTHONPATH=src pytest -q src/Finetuning/tests
FINETUNING_RUN_DISTRIBUTED_TESTS=1 PYTHONPATH=src pytest -q src/Finetuning/tests/test_distributed_preparation.py
```

Expected: zero failures; CUDA-specific tests are skipped only when CUDA is unavailable.

### Step 5: Record conclusion

Record the actual test counts and explicitly state that B200 throughput/VRAM measurements require execution on the B200 server.
