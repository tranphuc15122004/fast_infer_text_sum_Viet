# Adaptive Regeneration and Feature-Caching Design

## Objective

Use the available VRAM on each B200 efficiently during target regeneration and
hidden-state caching while preserving deterministic teacher outputs, feature
manifest contracts, rank sharding, atomic publication, and resumable artifacts.

## Hypothesis

Length-bucketed adaptive token batching with a real-workload preflight and
OOM backoff will increase samples/second and GPU utilization over the current
single-example loop without changing generated summaries or cached features.

## Scope and constraints

- The source dataset remains streamed from the server; `datasets/` is not a
  training or preparation input.
- One process owns one GPU under `torchrun`; ranks keep the existing
  deterministic round-robin sharding and rank-0 merge behavior.
- Batch sizing is local to a rank and is selected from actual prompt/sequence
  lengths. The target memory ceiling defaults to 90% of reported device VRAM.
- OOM is recoverable through batch reduction; non-OOM runtime errors are never
  hidden.
- Regeneration remains greedy (`do_sample=False`) and caching remains
  `use_cache=False` with selected hidden states copied to CPU before writing.
- Output record identity/order and the existing manifest/prompt contracts are
  invariant.

## Proposed architecture

1. `adaptive_inference.py` provides pure length bucketing, token-budget batch
   planning, and a shared OOM backoff helper. It has no dependency on tests.
2. `generate_targets.py` materializes only a bounded rank-local window,
   tokenizes/length-buckets records, batches padded prompts, and extracts each
   row's generated continuation using its true prompt length.
3. `capture_features.py` uses the same planner and batches padded input IDs and
   attention masks. Each row's selected hidden states are trimmed to its true
   sequence length and atomically written before the next batch.
4. A preflight on representative long examples calibrates the starting batch
   size. During processing, OOM halves the candidate and retries the same
   batch; successful smaller batches are allowed to grow cautiously within the
   configured cap.
5. Rank-local metrics report samples, tokens, peak reserved memory, retries,
   and throughput. Existing rank-0 aggregate output remains the public result.

## Controls

The shared preparation settings are:

```yaml
adaptive:
  enabled: true
  target_memory_fraction: 0.90
  min_batch_size: 1
  max_batch_size: 256
  max_tokens_per_batch: 0
  bucket_window: 512
  probe_batches: 2
  oom_backoff: true
```

`max_tokens_per_batch: 0` means the memory probe controls capacity. A positive
token budget adds a deterministic upper bound and is useful when host RAM is
limited. `bucket_window` bounds CPU buffering and never materializes the full
corpus.

## Validation scope

- Unit tests cover padding/trimming, bucket determinism, token-budget planning,
  OOM-only retry behavior, and configuration validation.
- CPU tests use fake target modules to verify output identity and feature
  shapes without requiring CUDA.
- A two-process DDP smoke verifies rank sharding and atomic merge behavior.
- On a CUDA server, runtime validation compares baseline and adaptive
  samples/second, peak reserved memory, OOM retry count, and output parity on
  the same fixture. The existing training validation remains unchanged.

## Acceptance criteria

- No preparation path processes more than the configured memory target after
  an OOM retry.
- Adaptive regeneration output is byte-equivalent at the JSON payload level
  apart from rank-local temporary naming.
- Adaptive feature records have the same tensors, manifest, dtype, and shapes
  as the single-example path.
- Full existing test suite and preparation DDP smoke remain green.
- Logs contain enough information to identify selected batch size, token
  counts, peak VRAM, retry count, and throughput.
