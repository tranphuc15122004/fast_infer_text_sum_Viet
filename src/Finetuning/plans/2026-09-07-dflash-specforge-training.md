# DFlash SpecForge Training Port Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** Port the SpecForge DFlash offline training process into src/Finetuning/ for Qwen3-4B and Qwen3-8B, with a local JSONL summarization adapter and a runnable single-GPU smoke pipeline.

**Experiment directory:** src/Finetuning/

**Hypothesis:** Reproducing SpecForge's DFlash objective, target-feature contract, and optimizer lifecycle in a self-contained package will produce a trainable Qwen3 DFlash draft while preserving upstream checkpoint and inference semantics.

**Validation scope:** CPU unit/contract tests, synthetic end-to-end flow, bounded tiny-overfit runtime validation, and optional one-step Qwen3-4B local-snapshot smoke. No Vietnamese ROUGE evaluation until a real Vietnamese dataset exists.

**Evaluation design:** A dedicated evaluator exposes one shared in-memory/checkpoint core for loss, supervised accuracy, valid-token count, and simulated acceptance metrics. Training fires evaluation by optimizer-step cadence (training.eval_interval, default 100 for normal configs and 1 for smoke). Each evaluation emits phase-start/end messages, progress, result and efficiency summaries, and fails on missing/unreadable checkpoints, restore errors, empty loaders, aggregation errors, non-finite metrics, or stalls.

**Architecture:** Copy only the DFlash-specific SpecForge seams into a self-contained Finetuning package: Qwen3 DFlash model, DFlash-family objective, feature contract, strategy, single-GPU trainer, scheduler, checkpoint manager, evaluator, typed config, and CLI. Raw document/summary JSONL is converted into the same input_ids/loss_mask/hidden_states contract consumed by the offline trainer. DDP, online/disaggregated capture, Llama DFlash, and MR-DFlash remain later phases.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PyYAML, JSONL, pytest, tqdm; CPU-safe eager/sdpa tests and GPU-compatible flex_attention path when available.

**Spec:** src/Finetuning/plans/2026-09-07-dflash-specforge-training-design.md

## Global Constraints

- Runtime must use the shared Python 3.12 environment; do not create a baseline-specific virtual environment.
- Core runtime code must not import from tests or validation helpers.
- Phase 1 is offline single-GPU; no SGLang, Mooncake, DDP, FSDP, USP, or model parallel launch.
- DFlash target is Qwen3; DFlashDraftModel must retain SpecForge's Qwen3 config, layer, projector, mask, and checkpoint semantics.
- Target embedding and LM head are frozen; only draft parameters are optimized.
- Do not alter externals/SpecForge, src/MR_DFlash, or the user's existing dirty files.
- No internet download is allowed during tests or training; model paths must resolve to local snapshots on the server.
- Training artifacts belong under the configured output directory and must not be committed.
- Synthetic data is for pipeline validation only and must never be reported as Vietnamese summarization quality.
- Every functional subtask includes its own tests and a focused commit; the final assembled pipeline is the only [INTEGRATION] subtask and receives Validation Pyramid checks.

## Shared Scaffold

### Existing infra (do not modify)

- Upstream model/objective reference: externals/SpecForge/specforge/modeling/draft/dflash.py and externals/SpecForge/specforge/algorithms/common/dflash_family_model.py.
- Upstream data/provider reference: externals/SpecForge/specforge/algorithms/common/dflash_family_data.py, externals/SpecForge/specforge/algorithms/dflash/providers.py, and externals/SpecForge/specforge/algorithms/model_providers.py.
- Upstream trainer reference: externals/SpecForge/specforge/training/strategies/base.py, trainer.py, checkpoint.py, schedule.py, and config/schema.py.
- Existing MR implementation: src/MR_DFlash/; use only for comparison when a contract question needs checking, never as a runtime dependency.
- Shared project runtime conventions: AGENTS.md, scripts/common/runtime.sh, and requirements.local.txt for CPU development.

### Needs setup

- Create the self-contained package under src/Finetuning/ with the exact files listed in the subtasks below.
- Create src/Finetuning/plans/ artifacts only; keep checkpoints and logs outside Git.
- Add no dependency beyond packages already present in the shared requirements unless a later plan explicitly approves it.

## Subtask 1: Port the Qwen3 DFlash draft model

**Role:** Provide the Qwen3-compatible draft backbone and checkpoint layout that the objective and trainer consume.

**Implementation files:**

- Create: src/Finetuning/__init__.py
- Create: src/Finetuning/dflash_kernels.py
- Create: src/Finetuning/model.py
- Test: src/Finetuning/tests/test_model.py

**SpecForge source boundaries:** Port the semantics of DFlashKernels, sample, resolve_dflash_attention_layout, apply_rotary_pos_emb, Qwen3DFlashAttention, Qwen3DFlashDecoderLayer, build_target_layer_ids, extract_context_feature, normalize_draft_head_checkpoint_keys, and DFlashDraftModel from the upstream dflash.py and dflash_kernels.py. Do not copy Domino/DSpark subclasses.

**Interfaces:**

- resolve_dflash_attention_layout(config) returns tuple[tuple[str, ...], int | None].
- build_target_layer_ids(num_target_layers: int, num_draft_layers: int) returns list[int].
- extract_context_feature(hidden_states: list[torch.Tensor], layer_ids: list[int]) returns torch.Tensor.
- DFlashDraftModel(config, dflash_kernels=None) exposes forward(position_ids, attention_mask=None, noise_embedding=None, target_hidden=None, past_key_values=None, use_cache=False, **kwargs) and returns the final draft hidden tensor for training.
- DFlashDraftModel.spec_generate(...) preserves the upstream speculative-generation boundary for later inference integration.
- DFlashKernels(make_rms_norm, make_mlp) and DEFAULT_DFLASH_KERNELS preserve the upstream factory boundary.

### Step 1: Write deterministic model contract tests

Create a tiny Qwen3Config fixture in test_model.py with hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2, num_target_layers=4, vocab_size=97, block_size=4, layer_types=['full_attention', 'full_attention'], and dflash_config={'target_layer_ids': [1, 2]}. Tests must cover:

    def test_target_layer_ids_and_context_feature_contract():
        assert build_target_layer_ids(28, 1) == [14]
        states = [torch.full((1, 5, 3), float(i)) for i in range(6)]
        result = extract_context_feature(states, [1, 3])
        assert result.shape == (1, 5, 6)
        assert torch.equal(result[..., :3], states[2])
        assert torch.equal(result[..., 3:], states[4])

    def test_dflash_forward_shape_and_gradient():
        model = DFlashDraftModel(tiny_qwen3_config())
        target_hidden = torch.randn(1, 5, 64)
        noise = torch.randn(1, 8, 32)
        positions = torch.arange(8).view(1, -1)
        output = model(position_ids=positions, noise_embedding=noise,
                       target_hidden=target_hidden)
        assert output.shape == (1, 8, 32)
        output.square().mean().backward()
        assert any(p.grad is not None for p in model.parameters()
                   if p.requires_grad)

    def test_invalid_layer_layout_is_rejected():
        config = tiny_qwen3_config()
        config.layer_types = ['full_attention']
        with pytest.raises(ValueError, match='num_hidden_layers'):
            DFlashDraftModel(config)

Also test that state_dict() contains the draft projector/layers/norm and that the pre-hook normalizes legacy logit_head.* keys without changing normal DFlash keys.

### Step 2: Run the focused tests before implementation

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_model.py

Expected: FAIL because Finetuning model modules do not exist yet.

### Step 3: Implement the model boundary

Port the upstream Qwen3 implementation with these exact invariants:

- context K/V comes from target_hidden and draft K/V comes from noise_embedding;
- fc maps concatenated target layer features to Qwen3 hidden size;
- each layer uses the configured full/sliding layout and the same RoPE positions;
- the DFlash draft has block_size, mask_token_id, target_layer_ids, and config.dflash_config fields;
- DFlashDraftModel.forward() returns normalized hidden states, not vocabulary logits;
- torch.inference_mode() is used only by spec_generate, never by training forward;
- no target model or tokenizer is loaded from this module.

### Step 4: Run focused tests and static checks

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_model.py
    PYTHONPATH=src .venv/bin/python -m compileall -q src/Finetuning

Expected: all model tests pass and compilation produces no output or errors.

### Step 5: Commit the model subtask

    git add src/Finetuning/__init__.py src/Finetuning/dflash_kernels.py src/Finetuning/model.py src/Finetuning/tests/test_model.py
    git commit -m 'experiment: port SpecForge Qwen3 DFlash model'

## Subtask 2: Port the DFlash objective and training strategy

**Role:** Turn feature batches into the faithful block-parallel DFlash loss and expose the strategy contract consumed by the generic trainer.

**Implementation files:**

- Create: src/Finetuning/dflash_family_model.py
- Create: src/Finetuning/strategy.py
- Test: src/Finetuning/tests/test_objective.py
- Test: src/Finetuning/tests/test_strategy.py

**Interfaces:**

- compute_accept_len(pred_ids_4d, target_ids_4d, valid_mask_4d) returns a tensor.
- create_dflash_sdpa_mask(anchor_positions, block_keep_mask, S, block_size, device, sliding_window=None) returns a dense attention mask.
- create_dflash_block_mask(...) returns a Flex Attention BlockMask.
- OnlineDFlashModel(draft_model, target_lm_head, target_embed_tokens, mask_token_id, block_size=16, attention_backend='flex_attention', num_anchors=512, loss_decay_gamma=None, objective_chunk_blocks=128, loss_type='dflash', dpace_alpha=0.5).
- OnlineDFlashModel.forward(input_ids, hidden_states, loss_mask) returns loss, accuracy, and metrics.
- StepOutput(loss, metrics, ratio_metrics, loss_terms).
- StepContext(global_step=0, total_steps=None).
- DFlashTrainStrategy(dflash_model).forward_loss(batch, ctx=None) returns StepOutput.
- DFlashTrainStrategy.checkpoint_state_filter(state_dict) returns a draft-only state dict.

### Step 1: Write objective and strategy tests

Use small CPU tensors and deterministic seeds. The tests must assert actual DFlash semantics rather than only compatible shapes:

    def test_sdpa_full_mask_is_strict_context_and_same_block_noncausal():
        anchors = torch.tensor([[2, 6]])
        keep = torch.tensor([[True, True]])
        mask = create_dflash_sdpa_mask(anchors, keep, S=8, block_size=4,
                                       device=torch.device('cpu'))
        allowed = mask[0, 0, 1].bool()
        assert allowed[0] and allowed[1]
        assert allowed[2] and allowed[3]
        assert allowed[8] and allowed[9] and allowed[10] and allowed[11]
        assert not allowed[12]

        sliding = create_dflash_sdpa_mask(
            anchors, keep, S=8, block_size=4,
            device=torch.device('cpu'), sliding_window=4)
        sliding_allowed = sliding[0, 0, 1].bool()
        assert sliding_allowed[8] and sliding_allowed[9]
        assert not sliding_allowed[10] and not sliding_allowed[11]

    def test_dflash_loss_excludes_anchor_and_uses_loss_mask():
        model = tiny_online_dflash(loss_decay_gamma=None)
        input_ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
        hidden = torch.randn(1, 8,
                             model.draft_model.config.num_target_layers * 32)
        loss_mask = torch.tensor([[0, 1, 1, 1, 0, 0, 1, 1]],
                                 dtype=torch.float32)
        loss, accuracy, metrics = model(input_ids, hidden, loss_mask)
        assert torch.isfinite(loss)
        assert torch.isfinite(accuracy)
        assert metrics['accuracy_denom'].item() > 0
        loss.backward()
        assert all(p.grad is None for p in model.embed_tokens.parameters())
        assert all(p.grad is None for p in model.lm_head.parameters())
        assert any(p.grad is not None for p in model.draft_model.parameters())

    def test_strategy_requires_all_dflash_features_and_filters_draft_keys():
        strategy = DFlashTrainStrategy(tiny_online_dflash())
        with pytest.raises(ValueError, match='missing required features'):
            strategy.validate_batch(
                SimpleBatch({'input_ids': torch.ones(1, 4)})
            )
        filtered = strategy.checkpoint_state_filter({
            'draft_model.fc.weight': torch.ones(2),
            'lm_head.weight': torch.ones(2),
        })
        assert set(filtered) == {'fc.weight'}

Add tests for num_anchors sampling, invalid all-zero loss masks, positional decay, D-PACE weight selection, invalid loss type/backend, objective chunking, and compute_accept_len on ragged blocks. If flex_attention is unavailable, skip only the Flex-specific test; SDPA/eager tests remain mandatory.

### Step 2: Run objective tests to verify they fail

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_objective.py src/Finetuning/tests/test_strategy.py

Expected: FAIL because the DFlash objective and strategy modules are not implemented.

### Step 3: Implement the objective and strategy

Port the upstream DFlash-only sections and preserve these invariants:

- valid anchors require both loss_mask[t] and loss_mask[t+1];
- block offset 0 receives the anchor embedding but never contributes to the loss;
- query context uses strict kv_idx < anchor_pos and draft visibility is limited to the same block; full-attention layers allow all draft offsets in that block, while sliding-attention layers apply causal offset visibility (`kv_offset <= q_offset`), matching SpecForge;
- labels are same-position input_ids[anchor + offset], masked by bounds and original loss_mask;
- default loss_type='dflash' is hard-label cross entropy with optional exponential positional decay;
- target embedding/LM head remain requires_grad=False and the strategy optimizer owns only draft_model;
- additive objective numerator/denominator are returned so accumulation preserves global normalization;
- metric tensors are detached before being exposed to the trainer.

strategy.py must define a small TrainBatch protocol or dataclass containing tensors: dict[str, torch.Tensor]. The core objective must not depend on a test fixture or future data loader implementation.

### Step 4: Run objective tests and gradients

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_objective.py src/Finetuning/tests/test_strategy.py

Expected: all tests pass; target embedding and LM-head gradients remain None, while at least one draft parameter receives a finite gradient.

### Step 5: Commit the objective subtask

    git add src/Finetuning/dflash_family_model.py src/Finetuning/strategy.py src/Finetuning/tests/test_objective.py src/Finetuning/tests/test_strategy.py
    git commit -m 'experiment: port SpecForge DFlash objective'

## Subtask 3: Build the summarization data and offline feature contract

**Role:** Convert future Vietnamese document/summary JSONL into the exact offline tensors expected by the DFlash strategy, while providing synthetic fixtures now.

**Implementation files:**

- Create: src/Finetuning/data.py
- Create: src/Finetuning/features.py
- Create: src/Finetuning/capture_features.py
- Create: src/Finetuning/tests/test_data.py
- Create: src/Finetuning/tests/test_features.py
- Create: src/Finetuning/tests/fixtures/synthetic_summary.jsonl

**Interfaces:**

- SummaryRecord(id: str, document: str, summary: str).
- load_summary_jsonl(path, max_samples=None) returns list[SummaryRecord].
- render_summary_example(record, tokenizer, max_length, chat_template='qwen3') returns tensors.
- build_summary_loss_mask(input_ids, assistant_start, assistant_end) returns a tensor.
- FeatureManifest.to_dict() and FeatureManifest.from_dict(payload).
- validate_feature_record(record, manifest) returns None or raises ValueError.
- OfflineFeatureDataset(root, manifest=None).
- collate_features(features) returns a padded tensor dictionary.
- capture_dataset(target_model_path, prepared_examples, output_dir, target_layer_ids, max_length, device, dtype) returns FeatureManifest.

### Step 1: Write data and feature contract tests

Test local JSONL parsing, Unicode Vietnamese text, deterministic prompt rendering through a fake tokenizer, summary-only loss masks, truncation, and rejected feature artifacts:

    def test_summary_loss_mask_only_supervises_assistant_span():
        mask = build_summary_loss_mask(torch.arange(8), assistant_start=5,
                                       assistant_end=8)
        assert mask.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]

    def test_invalid_feature_width_and_sequence_length_are_rejected(tmp_path):
        manifest = FeatureManifest(model_id='tiny', layer_ids=[1, 3],
                                   hidden_size=4, max_length=8)
        record = {
            'input_ids': torch.ones(7, dtype=torch.long),
            'loss_mask': torch.ones(8),
            'hidden_states': torch.ones(8, 7),
        }
        with pytest.raises(ValueError, match='sequence lengths'):
            validate_feature_record(record, manifest)

    def test_collator_pads_features_without_mixing_feature_width():
        batch = collate_features([
            {'input_ids': torch.tensor([1, 2]),
             'loss_mask': torch.tensor([0, 1]),
             'hidden_states': torch.ones(2, 4)},
            {'input_ids': torch.tensor([3]),
             'loss_mask': torch.tensor([1]),
             'hidden_states': torch.ones(1, 4)},
        ])
        assert batch['input_ids'].shape == (2, 2)
        assert batch['loss_mask'].shape == (2, 2)
        assert batch['hidden_states'].shape == (2, 2, 4)

Add a test that rejects an example with no two consecutive supervised tokens and a manifest round-trip test preserving model id, revision, tokenizer id, layer ids, max length, and feature width.

### Step 2: Run data tests to verify they fail

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_data.py src/Finetuning/tests/test_features.py

Expected: FAIL because the summary adapter and feature contract are absent.

### Step 3: Implement the local JSONL adapter and feature store

Implement these rules:

- accept id, document, and summary as minimal fields, while allowing extra metadata to pass through;
- render a Qwen3-compatible user/assistant sequence through the supplied tokenizer without downloading a template;
- place loss_mask=1 only on summary assistant tokens and set the final sequence position to 0 when required by the causal label shift;
- truncate consistently across input_ids, loss_mask, and captured hidden states;
- reject samples with fewer than two adjacent supervised tokens, matching dflash_min_loss_tokens();
- save one tensor record per feature file plus one manifest using atomic writes and CPU-readable tensors;
- require hidden_states.shape == (sequence_length, len(layer_ids) * target_hidden_size);
- collate by right-padding input_ids/loss_mask and zero-padding hidden_states, preserving integer ids and floating feature dtype.

### Step 4: Implement offline target feature capture

capture_features.py may load a local Transformers target model only for feature generation. It must set eval(), disable gradients, request output_hidden_states=True, select the resolved target layer ids with the same offset as SpecForge, concatenate along the final dimension, and write the manifest before any training loader opens the directory. It must fail if the target snapshot is missing or the produced feature width disagrees with the draft config.

### Step 5: Run data tests and fixture checks

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_data.py src/Finetuning/tests/test_features.py
    PYTHONPATH=src .venv/bin/python - <<'PY'
    from pathlib import Path
    from Finetuning.data import load_summary_jsonl
    records = load_summary_jsonl(
        Path('src/Finetuning/tests/fixtures/synthetic_summary.jsonl')
    )
    assert len(records) >= 2
    assert all(record.document and record.summary for record in records)
    PY

Expected: all tests pass and the fixture loads without internet access.

### Step 6: Commit the data subtask

    git add src/Finetuning/data.py src/Finetuning/features.py src/Finetuning/capture_features.py src/Finetuning/tests/test_data.py src/Finetuning/tests/test_features.py src/Finetuning/tests/fixtures/synthetic_summary.jsonl
    git commit -m 'experiment: add DFlash offline feature contract'

## Subtask 4: Implement the single-GPU trainer, schedule, checkpoints, and evaluator

**Role:** Reproduce SpecForge's optimizer lifecycle around the DFlash strategy without introducing distributed runtime dependencies in phase 1.

**Implementation files:**

- Create: src/Finetuning/schedule.py
- Create: src/Finetuning/checkpoint.py
- Create: src/Finetuning/evaluation.py
- Create: src/Finetuning/metrics.py
- Create: src/Finetuning/trainer.py
- Test: src/Finetuning/tests/test_schedule.py
- Test: src/Finetuning/tests/test_checkpoint.py
- Test: src/Finetuning/tests/test_evaluation.py
- Test: src/Finetuning/tests/test_trainer.py

**Interfaces:**

- resolve_total_steps(total_steps, max_steps, num_samples, batch_size, accumulation_steps, num_epochs) returns int.
- validate_fixed_accumulation_plan(num_samples, batch_size, accumulation_steps, num_epochs, max_steps) returns None or raises ValueError.
- CheckpointManager(output_dir, run_id, max_checkpoints=0) exposes save(step, model, optimizer, scheduler, trainer_state, extra), load(path, map_location='cpu'), latest_dir(), and resolve_resume_dir(path).
- Evaluator.evaluate_in_memory(model, dataloader, device, max_batches=None) returns dict[str, float].
- Evaluator.evaluate_checkpoint(checkpoint_path, build_model, dataloader_factory, device) returns dict[str, float].
- Trainer.fit() returns the completed optimizer step; Trainer.evaluate() returns metrics; Trainer.save_checkpoint() returns a checkpoint Path.

### Step 1: Write lifecycle tests

Tests must cover schedule arithmetic, incomplete accumulation rejection, atomic checkpoint contents, optimizer/scheduler/RNG restoration, evaluation parity between in-memory and checkpoint modes, and a tiny training loop:

    def test_total_steps_matches_optimizer_updates():
        assert resolve_total_steps(None, None, 12, 2, 3, 2) == 4

    def test_partial_accumulation_is_rejected():
        with pytest.raises(ValueError, match='incomplete gradient accumulation'):
            validate_fixed_accumulation_plan(
                num_samples=5, batch_size=2, accumulation_steps=2,
                num_epochs=1, max_steps=None,
            )

    def test_checkpoint_round_trip_restores_training_state(tmp_path):
        manager = CheckpointManager(tmp_path, 'tiny')
        path = manager.save(
            step=3, model=model, optimizer=optimizer, scheduler=scheduler,
            trainer_state={'global_step': 3, 'seed': 42},
            extra={'strategy': 'dflash'},
        )
        restored = manager.load(path)
        assert restored['trainer_state']['global_step'] == 3
        assert restored['extra']['strategy'] == 'dflash'

    def test_checkpoint_and_in_memory_evaluation_share_results(tmp_path):
        in_memory = evaluator.evaluate_in_memory(model, loader, device)
        checkpoint = manager.save(
            step=1, model=model, optimizer=optimizer, scheduler=scheduler,
            trainer_state={}, extra={},
        )
        from_checkpoint = evaluator.evaluate_checkpoint(
            checkpoint, lambda: build_model(), lambda: loader, device,
        )
        assert from_checkpoint.keys() == in_memory.keys()
        for key in in_memory:
            assert from_checkpoint[key] == pytest.approx(
                in_memory[key], rel=1e-5
            )

Add a trainer test asserting that gradient accumulation calls optimizer.step() only at the configured boundary, max_steps stops at the correct optimizer step, logs contain step/loss/grad_norm/lr, and a non-finite loss raises a descriptive error.

### Step 2: Run lifecycle tests to verify they fail

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_schedule.py src/Finetuning/tests/test_checkpoint.py src/Finetuning/tests/test_evaluation.py src/Finetuning/tests/test_trainer.py

Expected: FAIL because the trainer lifecycle modules do not exist.

### Step 3: Implement schedule, checkpoint, evaluator, metrics, and trainer

Implement the following exact lifecycle:

- resolve the effective optimizer horizon before creating the scheduler;
- use AdamW over strategy.trainable_module().parameters() only;
- use the SpecForge warmup plus cosine/constant scheduler state format, with last_epoch restored on resume;
- divide gradients by accumulation steps and clip with max_grad_norm before every optimizer step;
- save draft weights, optimizer, scheduler, RNG state, global/micro step, resolved config, strategy contract, and metric history atomically;
- write metrics.jsonl and a human-readable train.log, and display a tqdm progress bar;
- log loss, accuracy, accuracy_denom, grad_norm, lr, step_time_s, tokens_per_s, and mfu when a hardware peak is configured;
- use a hardware_peak_tflops config value for an explicit MFU estimate and log null when it is not configured, never fabricate a device baseline;
- keep evaluation in evaluation.py; the trainer only decides when it fires;
- make checkpoint-based evaluation reconstruct the model and call the same batch aggregation function as in-memory evaluation;
- fail before training if the validation loader is empty or any aggregate is non-finite.

### Step 4: Run lifecycle tests and a CPU training fixture

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_schedule.py src/Finetuning/tests/test_checkpoint.py src/Finetuning/tests/test_evaluation.py src/Finetuning/tests/test_trainer.py

Expected: all lifecycle tests pass, checkpoint reload reproduces evaluation metrics, and the tiny trainer completes without NaN/Inf.

### Step 5: Commit the trainer subtask

    git add src/Finetuning/schedule.py src/Finetuning/checkpoint.py src/Finetuning/evaluation.py src/Finetuning/metrics.py src/Finetuning/trainer.py src/Finetuning/tests/test_schedule.py src/Finetuning/tests/test_checkpoint.py src/Finetuning/tests/test_evaluation.py src/Finetuning/tests/test_trainer.py
    git commit -m 'experiment: add single GPU DFlash trainer lifecycle'

## Subtask 5: Assemble Qwen3 offline training CLI [INTEGRATION]

**Hypothesis:** The assembled package can execute the complete SpecForge-shaped path from summary records or precomputed features through DFlash loss, backward, optimizer step, evaluation, and checkpoint reload on a single CPU/GPU process.

**Components consumed:**

- Model: src/Finetuning/model.py and dflash_kernels.py.
- Objective/strategy: src/Finetuning/dflash_family_model.py and strategy.py.
- Data/features: src/Finetuning/data.py, features.py, and capture_features.py.
- Lifecycle: src/Finetuning/schedule.py, checkpoint.py, evaluation.py, metrics.py, and trainer.py.

**Implementation files:**

- Create: src/Finetuning/config.py
- Create: src/Finetuning/run_train.py
- Create: src/Finetuning/prepare_data.py
- Create: src/Finetuning/configs/qwen3_4b.yaml
- Create: src/Finetuning/configs/qwen3_8b.yaml
- Create: src/Finetuning/configs/synthetic_smoke.yaml
- Create: src/Finetuning/README.md
- Test: src/Finetuning/tests/test_integration.py
- Test: src/Finetuning/tests/test_qwen3_local_smoke.py

**Interfaces:**

- ModelConfig, DataConfig, TrainingConfig, and RunConfig dataclasses with strict validation for model path, feature source, block size, layer ids, attention backend, batch/accumulation sizes, and positive optimizer settings.
- load_run_config(path) returns RunConfig.
- apply_cli_overrides(config, args) returns RunConfig.
- run_training(config) returns the final checkpoint directory.
- CLI: python -m Finetuning.run_train --config PATH with optional target-model-path, train-data-path, hidden-states-path, output-dir, max-steps, batch-size, device, smoke, and resume-from flags.

### Step 1: Write the integration tests

The synthetic integration test must exercise data -> feature tensors -> model -> DFlash loss -> backward -> optimizer -> evaluation -> checkpoint reload in one process:

    def test_synthetic_end_to_end_training(tmp_path):
        config = RunConfig.from_synthetic(
            output_dir=tmp_path / 'run',
            max_steps=4,
            batch_size=1,
            eval_interval=1,
            attention_backend='eager',
        )
        final_checkpoint = run_training(config)
        assert final_checkpoint.is_dir()
        metrics = read_jsonl(tmp_path / 'run' / 'metrics.jsonl')
        steps = [row for row in metrics if row.get('type') == 'step']
        assert len(steps) == 4
        losses = [row['loss'] for row in steps]
        assert all(math.isfinite(value) for value in losses)
        assert losses[-1] < losses[0]
        restored = evaluate_checkpoint(final_checkpoint, config)
        assert math.isfinite(restored['loss'])

    def test_cli_rejects_online_or_missing_feature_source(tmp_path):
        with pytest.raises(ValueError, match='exactly one'):
            load_run_config(write_yaml(tmp_path, {
                'model': {}, 'data': {}, 'training': {},
            }))

The local Qwen test is opt-in and must never download:

    @pytest.mark.skipif(
        os.environ.get('FINETUNING_RUN_REAL_QWEN3') != '1',
        reason='set FINETUNING_RUN_REAL_QWEN3=1',
    )
    def test_qwen3_4b_one_step_local_snapshot():
        model_path = os.environ['FINETUNING_QWEN3_4B_PATH']
        assert Path(model_path).is_dir()
        result = run_training(load_real_qwen_config(model_path, max_steps=1))
        assert result.is_dir()

### Step 2: Run integration tests to verify they fail

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_integration.py

Expected: FAIL because the CLI/config assembly is not present.

### Step 3: Assemble the pipeline

Implement run_training() in this order:

1. Load/validate config, set Python/PyTorch/CUDA seeds, resolve device and dtype, and reject non-local model paths when offline mode is enabled.
2. Load tokenizer and target config. Resolve Qwen3 target layer ids and draft config from the target config or the checked-in SpecForge-equivalent Qwen3 draft config.
3. If hidden_states_path is set, open and validate the manifest. If train_data_path is set, prepare local summary examples and call capture_dataset() before constructing the training loader. The two sources must not be set simultaneously.
4. Build DFlashDraftModel, load optional draft weights, load only target embedding and LM-head weights into frozen modules, and construct OnlineDFlashModel plus DFlashTrainStrategy.
5. Construct the offline dataset/loader, optional validation loader, Evaluator, CheckpointManager, optimizer, scheduler, and Trainer.
6. Run Trainer.fit(), save the final draft-only checkpoint, run final in-memory evaluation, and print the output path and metric summary.

The config files must be explicit and reproducible:

    model:
      target_model_path: /absolute/path/to/Qwen3-4B
      draft_model_config: null
      torch_dtype: bfloat16
      mask_token_id: null
      attention_backend: eager
    data:
      train_data_path: /absolute/path/to/document_summary.jsonl
      hidden_states_path: null
      max_length: 2048
      chat_template: qwen3
    training:
      strategy: dflash
      num_epochs: 1
      max_steps: 1000
      batch_size: 1
      accumulation_steps: 1
      learning_rate: 0.0006
      warmup_ratio: 0.04
      max_grad_norm: 1.0
      num_anchors: 512
      loss_decay_gamma: 7.0
      objective_chunk_blocks: 128
      attention_backend: eager
      save_interval: 100
      eval_interval: 100
      log_interval: 10
      seed: 42
    output_dir: /absolute/path/to/outputs/qwen3_4b_dflash

The checked-in Qwen3-8B config changes only the target model name and output directory; it does not silently alter DFlash hyperparameters. synthetic_smoke.yaml uses a tiny in-process Qwen3 config and CPU-safe eager attention, so it does not require a model snapshot.

### Step 4: Run integration tests

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_integration.py
    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests/test_qwen3_local_smoke.py
    PYTHONPATH=src .venv/bin/python -m Finetuning.run_train --config src/Finetuning/configs/synthetic_smoke.yaml --device cpu --smoke

Expected: synthetic training completes for four optimizer steps, metrics.jsonl contains finite losses and evaluation rows, final checkpoint reload succeeds, and the command never accesses the network. The Qwen3 local smoke is skipped unless FINETUNING_RUN_REAL_QWEN3=1; when enabled, it must use FINETUNING_QWEN3_4B_PATH and complete one optimizer step.

### Step 5: Write external validation scripts

Create:

- src/Finetuning/tests/validate_static.py: inspect config validation, required artifact files, frozen target parameters, optimizer parameter ownership, metric keys, and absence of NaN/Inf in emitted JSONL without importing test fixtures into core code.
- src/Finetuning/tests/validate_runtime.py: run the synthetic command with a bounded step count, measure time-to-first-progress, verify tqdm/log progress, checkpoint creation, evaluation result, and final checkpoint reload.

The scripts must report named phases config, data, model, train, evaluate, and checkpoint, and return nonzero on missing artifacts, silent progress beyond the timeout budget, or malformed metrics.

### Step 6: Run Validation Pyramid (L0 -> L1)

L0 (ml-static-checks) runs on this integration subtask and checks device/dtype consistency, attention backend selection, frozen target modules, optimizer ownership, scheduler horizon, DataLoader behavior, loss/speed file output, progress indicator, checkpoint/resume, seed setting, and metric output.

L1 (ml-runtime-validator) runs the bounded synthetic flow:

    PYTHONPATH=src .venv/bin/python src/Finetuning/tests/validate_runtime.py --config src/Finetuning/configs/synthetic_smoke.yaml --device cpu --max-steps 8

Expected L1 result: all data/model/loss/backward/optimizer/evaluation/checkpoint phases pass; loss is finite and decreases across the tiny overfit run; no NaN/Inf or crash occurs; first progress appears promptly; metrics and checkpoint are written; in-memory and checkpoint evaluation agree within 1e-5; MFU is null on CPU rather than fabricated.

### Step 7: Run the complete local test suite and self-review

Run:

    PYTHONPATH=src .venv/bin/python -m pytest -q src/Finetuning/tests
    PYTHONPATH=src .venv/bin/python -m compileall -q src/Finetuning
    git diff --check

Review the assembled code against the design doc and confirm that no file imports from src/MR_DFlash, externals/SpecForge at runtime, or test modules; confirm that no output artifact is staged.

### Step 8: Commit the integration subtask

    git add src/Finetuning/config.py src/Finetuning/run_train.py src/Finetuning/prepare_data.py src/Finetuning/configs src/Finetuning/README.md src/Finetuning/tests/test_integration.py src/Finetuning/tests/test_qwen3_local_smoke.py src/Finetuning/tests/validate_static.py src/Finetuning/tests/validate_runtime.py
    git commit -m 'experiment: assemble Qwen3 DFlash offline training'

## Plan self-review

- Spec coverage: Model fidelity is Subtask 1; DFlash masks/loss/strategy are Subtask 2; summary data and feature contract are Subtask 3; lifecycle/checkpoint/evaluation are Subtask 4; config, Qwen3 variants, CLI, integration, and L0/L1 are Subtask 5.
- Scope consistency: Only Qwen3-4B/8B, offline single-GPU, and synthetic validation are in phase 1. Llama 3.1, DDP, online/disaggregated, and real Vietnamese ROUGE are explicitly deferred.
- Interface consistency: DFlashTrainStrategy consumes TrainBatch.tensors; Trainer consumes the strategy and loader; run_training assembles both. Feature width is defined by FeatureManifest and checked before model construction.
- Completeness check: every implementation step specifies exact paths, commands, and expected outcomes.
- Integration uniqueness: Exactly one subtask title ends with [INTEGRATION].
