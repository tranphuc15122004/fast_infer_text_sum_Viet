"""CLI and assembly for the offline single-GPU Qwen3 DFlash trainer."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from .adaptive_batch import AdaptiveBatchSettings, tune_batch_size
from .checkpoint import CheckpointManager, load_draft_initialization
from .config import RunConfig, apply_cli_overrides, load_run_config
from .dflash_family_model import OnlineDFlashModel
from .distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
)
from .evaluation import Evaluator
from .features import FeatureManifest, OfflineFeatureDataset, collate_features
from .model import DFlashDraftModel, build_target_layer_ids
from .strategy import DFlashTrainStrategy
from .trainer import Trainer


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return value


def _resolve_device(
    config: RunConfig,
    distributed_context: DistributedContext | None = None,
) -> torch.device:
    context = distributed_context or DistributedContext()
    if config.device == "auto":
        if context.is_distributed:
            return torch.device("cuda", context.local_rank)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if context.is_distributed and device.type == "cuda":
        return torch.device("cuda", context.local_rank)
    return device


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _make_draft_config(
    target_config: Qwen3Config,
    config: RunConfig,
    target_layer_ids: list[int],
    mask_token_id: int,
    *,
    attention_backend: str | None = None,
) -> Qwen3Config:
    payload = target_config.to_dict()
    payload.update(
        {
            "architectures": ["DFlashDraftModel"],
            "num_hidden_layers": config.model.num_draft_layers,
            "intermediate_size": config.model.draft_intermediate_size
            or int(getattr(target_config, "intermediate_size")),
            "layer_types": config.model.layer_types
            or ["full_attention"] * config.model.num_draft_layers,
            "num_target_layers": int(getattr(target_config, "num_hidden_layers")),
            "block_size": config.model.block_size,
            "dflash_config": {
                "target_layer_ids": target_layer_ids,
                "mask_token_id": mask_token_id,
            },
        }
    )
    draft_config = Qwen3Config.from_dict(payload)
    draft_config._attn_implementation = (
        attention_backend or config.training.attention_backend
    )
    return draft_config


def _resolve_mask_token_id(config: RunConfig, tokenizer: Any, vocab_size: int) -> int:
    if config.model.mask_token_id is not None:
        token_id = int(config.model.mask_token_id)
    else:
        token_id = getattr(tokenizer, "mask_token_id", None)
        if token_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
            candidate = tokenizer.convert_tokens_to_ids("[MASK]")
            token_id = candidate if isinstance(candidate, int) and candidate >= 0 else None
        if token_id is None:
            raise ValueError(
                "model.mask_token_id is required when tokenizer has no mask token"
            )
    if not 0 <= int(token_id) < vocab_size:
        raise ValueError(f"mask_token_id={token_id} is outside vocab_size={vocab_size}")
    return int(token_id)


def _target_layer_ids(config: RunConfig, target_config: Any) -> list[int]:
    if config.model.target_layer_ids is not None:
        return list(config.model.target_layer_ids)
    return build_target_layer_ids(
        int(getattr(target_config, "num_hidden_layers")),
        config.model.num_draft_layers,
    )


def _build_strategy(
    config: RunConfig,
    target_config: Qwen3Config,
    tokenizer: Any,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    device: torch.device,
) -> DFlashTrainStrategy:
    layer_ids = _target_layer_ids(config, target_config)
    mask_token_id = _resolve_mask_token_id(
        config,
        tokenizer,
        int(getattr(target_config, "vocab_size")),
    )
    draft_config = _make_draft_config(
        target_config,
        config,
        layer_ids,
        mask_token_id,
    )
    draft = DFlashDraftModel(draft_config).to(device=device, dtype=_dtype(config.model.torch_dtype))
    target_dtype = _dtype(config.model.torch_dtype)
    model = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=lm_head.to(device=device, dtype=target_dtype),
        target_embed_tokens=embed_tokens.to(device=device, dtype=target_dtype),
        mask_token_id=mask_token_id,
        block_size=config.model.block_size,
        attention_backend=config.training.attention_backend,
        num_anchors=config.training.num_anchors,
        loss_decay_gamma=config.training.loss_decay_gamma,
        objective_chunk_blocks=config.training.objective_chunk_blocks,
        loss_type=config.training.loss_type,
        dpace_alpha=config.training.dpace_alpha,
    ).to(device)
    strategy = DFlashTrainStrategy(model)
    strategy.draft_export_metadata = {
        "target_model_path": config.model.target_model_path,
        "target_layer_ids": layer_ids,
        "block_size": config.model.block_size,
        "mask_token_id": mask_token_id,
        "torch_dtype": config.model.torch_dtype,
        "draft_config": draft_config.to_dict(),
    }
    return strategy


def _loader(
    dataset: OfflineFeatureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    distributed_context: DistributedContext | None = None,
) -> DataLoader:
    context = distributed_context or DistributedContext()
    minimum_samples = batch_size * context.world_size
    if len(dataset) < minimum_samples:
        raise ValueError(
            "feature dataset is too small for the requested per-GPU batch: "
            f"need at least {minimum_samples} samples, got {len(dataset)}"
        )
    sampler = None
    if context.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=shuffle,
            drop_last=True,
        )
        shuffle = False
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "drop_last": True,
        "collate_fn": collate_features,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        **kwargs,
    )


def _same_local_path(left: str, right: str) -> bool:
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(
        strict=False
    )


def _validate_feature_manifest(
    manifest: FeatureManifest,
    config: RunConfig,
    target_layer_ids: list[int],
) -> None:
    """Reject caches not captured for this frozen target and prompt contract."""

    target_path = config.model.target_model_path
    assert target_path is not None
    expected_contract = {
        "chat_template": config.data.chat_template,
        "max_source_tokens": config.data.max_source_tokens,
        "max_summary_tokens": config.data.max_summary_tokens,
        "prompt_template": config.data.prompt_template,
    }
    if not _same_local_path(manifest.model_id, target_path):
        raise ValueError(
            "feature manifest target differs from model.target_model_path: "
            f"{manifest.model_id!r} != {target_path!r}"
        )
    if manifest.tokenizer_id is None or not _same_local_path(
        manifest.tokenizer_id, target_path
    ):
        raise ValueError(
            "feature manifest tokenizer_id does not match model.target_model_path"
        )
    if manifest.layer_ids != target_layer_ids:
        raise ValueError(
            "feature manifest layer_ids do not match the DFlash target-layer selection"
        )
    if manifest.max_length != config.data.max_length:
        raise ValueError(
            "feature manifest max_length does not match data.max_length"
        )
    if manifest.hidden_states_dtype != f"torch.{config.data.feature_dtype}":
        raise ValueError(
            "feature manifest hidden_states_dtype does not match data.feature_dtype"
        )
    if manifest.prompt_contract != expected_contract:
        raise ValueError(
            "feature manifest prompt_contract does not match the configured "
            "chat template and token budgets"
        )


def _build_synthetic(config: RunConfig, device: torch.device):
    target_config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
        attention_dropout=0.0,
    )
    target_config._attn_implementation = config.training.attention_backend
    tokenizer = type("SyntheticTokenizer", (), {"mask_token_id": 0})()
    embed_tokens = nn.Embedding(target_config.vocab_size, target_config.hidden_size)
    lm_head = nn.Linear(target_config.hidden_size, target_config.vocab_size, bias=False)
    strategy = _build_strategy(
        config,
        target_config,
        tokenizer,
        embed_tokens,
        lm_head,
        device,
    )
    torch.manual_seed(config.training.seed + 1)
    sequence_length = config.data.max_length
    feature_width = len(_target_layer_ids(config, target_config)) * target_config.hidden_size
    input_ids = torch.arange(1, sequence_length + 1).remainder(target_config.vocab_size - 1) + 1
    loss_mask = torch.ones(sequence_length, dtype=torch.float32)
    hidden = torch.randn(sequence_length, feature_width, dtype=torch.float32) * 0.05
    batch = {
        "input_ids": input_ids.unsqueeze(0),
        "loss_mask": loss_mask.unsqueeze(0),
        "hidden_states": hidden.unsqueeze(0),
    }
    batches = [
        {key: value.clone() for key, value in batch.items()}
        for _ in range(max(config.training.max_steps or 4, 4))
    ]
    return strategy, batches, [batches[0]]


def _load_real_runtime(
    config: RunConfig,
    device: torch.device,
    distributed_context: DistributedContext | None = None,
):
    target_path = config.model.target_model_path
    assert target_path is not None
    tokenizer = AutoTokenizer.from_pretrained(
        target_path,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.offline,
    )
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=_dtype(config.model.torch_dtype),
        low_cpu_mem_usage=True,
        local_files_only=config.offline,
    ).to(device)
    target.eval()
    target_config = target.config
    embed_tokens = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()
    if embed_tokens is None or lm_head is None:
        raise ValueError("target model must expose input embeddings and lm_head")
    layer_ids = _target_layer_ids(config, target_config)
    train_feature_path = config.data.hidden_states_path
    assert train_feature_path is not None
    train_dataset = OfflineFeatureDataset(train_feature_path)
    _validate_feature_manifest(train_dataset.manifest, config, layer_ids)
    strategy = _build_strategy(
        config,
        target_config,
        tokenizer,
        embed_tokens,
        lm_head,
        device,
    )
    expected_width = strategy.dflash_model.draft_model.fc.in_features
    if train_dataset.manifest.feature_width != expected_width:
        raise ValueError(
            "feature width does not match DFlash draft: "
            f"{train_dataset.manifest.feature_width} != {expected_width}"
        )
    del target
    if config.training.adaptive_batch_size:
        adaptive_result = tune_batch_size(
            strategy,
            train_dataset,
            device=device,
            context=distributed_context,
            settings=AdaptiveBatchSettings(
                enabled=True,
                target_memory_fraction=config.training.target_memory_fraction,
                min_batch_size=config.training.adaptive_min_batch_size,
                max_batch_size=config.training.adaptive_max_batch_size,
                probe_batches=config.training.adaptive_probe_batches,
            ),
        )
        config.training.batch_size = adaptive_result.batch_size
        strategy.adaptive_batch_metadata = adaptive_result.to_dict()
    train_batches = _loader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        distributed_context=distributed_context,
    )
    eval_path = config.data.eval_hidden_states_path
    eval_batches = None
    if eval_path:
        eval_dataset = OfflineFeatureDataset(eval_path)
        _validate_feature_manifest(eval_dataset.manifest, config, layer_ids)
        eval_batches = _loader(
            eval_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory and device.type == "cuda",
            persistent_workers=config.data.persistent_workers,
            prefetch_factor=config.data.prefetch_factor,
            distributed_context=distributed_context,
        )
    return strategy, train_batches, eval_batches


def _assemble(
    config: RunConfig,
    device: torch.device,
    distributed_context: DistributedContext | None = None,
):
    if config.synthetic:
        return _build_synthetic(config, device)
    return _load_real_runtime(config, device, distributed_context)


def _resume_extra(
    config: RunConfig,
    resume_from: str | Path,
) -> dict[str, Any]:
    """Read lightweight checkpoint metadata without loading model/optimizer tensors."""

    manager = CheckpointManager(config.resolved_output_dir, config.run_id)
    root = manager.resolve_resume_dir(resume_from)
    extra_path = root / "extra.json"
    if not extra_path.is_file():
        raise ValueError(f"checkpoint is missing extra.json: {root}")
    payload = json.loads(extra_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint extra.json must contain an object: {extra_path}")
    return payload


def _run_training(
    config: RunConfig,
    *,
    resume_from: str | Path | None,
    distributed_context: DistributedContext,
) -> Path:
    """Run config validation, assembly, optimization, evaluation and save."""

    config.validate()
    resume_adaptive_metadata: dict[str, Any] | None = None
    if resume_from is not None:
        resume_extra = _resume_extra(config, resume_from)
        stored_adaptive = resume_extra.get("adaptive_batch")
        if stored_adaptive is not None:
            if not isinstance(stored_adaptive, dict):
                raise ValueError("checkpoint adaptive_batch metadata must be an object")
            resume_adaptive_metadata = dict(stored_adaptive)
            if config.training.adaptive_batch_size:
                stored_batch = stored_adaptive.get("batch_size")
                if isinstance(stored_batch, bool) or not isinstance(stored_batch, int):
                    raise ValueError(
                        "adaptive resume checkpoint does not contain a valid batch_size"
                    )
                if stored_batch <= 0:
                    raise ValueError(
                        "adaptive resume checkpoint batch_size must be positive"
                    )
                config.training.batch_size = stored_batch
                config.training.adaptive_batch_size = False
        elif config.training.adaptive_batch_size:
            raise ValueError(
                "adaptive_batch_size is enabled for resume, but the checkpoint "
                "does not contain adaptive batch metadata"
            )
    device = _resolve_device(config, distributed_context)
    _seed_everything(config.training.seed, device)
    strategy, train_batches, eval_batches = _assemble(
        config,
        device,
        distributed_context,
    )
    if hasattr(strategy, "configure_distributed"):
        strategy.configure_distributed(distributed_context)
    if resume_from is not None and config.model.draft_init_path is not None:
        raise ValueError("model.draft_init_path cannot be combined with resume_from")
    if config.model.draft_init_path is not None:
        load_draft_initialization(
            config.model.draft_init_path,
            strategy.dflash_model.draft_model,
            strategy.draft_export_metadata,
        )
    adaptive_metadata = getattr(strategy, "adaptive_batch_metadata", None)
    if adaptive_metadata is None:
        adaptive_metadata = resume_adaptive_metadata
    extra_checkpoint_state: dict[str, Any] = {
        "resolved_config": config.to_dict(),
    }
    if adaptive_metadata is not None:
        extra_checkpoint_state["adaptive_batch"] = adaptive_metadata
    trainer = Trainer(
        strategy=strategy,
        train_dataloader=train_batches,
        validation_dataloader=eval_batches,
        output_dir=config.resolved_output_dir,
        run_id=config.run_id,
        batch_size=config.training.batch_size,
        accumulation_steps=config.training.accumulation_steps,
        num_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        scheduler_type=config.training.scheduler_type,
        max_grad_norm=config.training.max_grad_norm,
        save_interval=config.training.save_interval,
        log_interval=config.training.log_interval,
        eval_interval=config.training.eval_interval,
        hardware_peak_tflops=config.training.hardware_peak_tflops,
        device=device,
        distributed_context=distributed_context,
        extra_checkpoint_state=extra_checkpoint_state,
        draft_export_metadata=strategy.draft_export_metadata,
        resume_from=resume_from,
    )
    trainer.fit()
    if eval_batches is not None and trainer._last_eval_step != trainer.global_step:
        metrics = trainer.evaluate()
        with trainer.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"type": "evaluation", "step": trainer.global_step, **metrics},
                    allow_nan=False,
                )
                + "\n"
            )
    return trainer.checkpoint_manager.latest_dir()


def run_training(config: RunConfig, *, resume_from: str | Path | None = None) -> Path:
    """Run training in one process or under a torchrun process group."""

    distributed_context = initialize_distributed(config.device)
    try:
        return _run_training(
            config,
            resume_from=resume_from,
            distributed_context=distributed_context,
        )
    finally:
        cleanup_distributed(distributed_context)


def evaluate_checkpoint(checkpoint_path: str | Path, config: RunConfig) -> dict[str, float]:
    distributed_context = initialize_distributed(config.device)
    try:
        config.validate()
        device = _resolve_device(config, distributed_context)
        _seed_everything(config.training.seed, device)
        strategy, _train_batches, eval_batches = _assemble(
            config,
            device,
            distributed_context,
        )
        if hasattr(strategy, "configure_distributed"):
            strategy.configure_distributed(distributed_context)
        if eval_batches is None:
            raise ValueError("evaluation requires a validation feature source")
        return Evaluator(distributed_context).evaluate_checkpoint(
            str(checkpoint_path),
            lambda: strategy,
            lambda: eval_batches,
            device,
        )
    finally:
        cleanup_distributed(distributed_context)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Qwen3 DFlash draft offline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-model-path")
    parser.add_argument("--train-data-path")
    parser.add_argument("--hidden-states-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--attention-backend")
    parser.add_argument(
        "--adaptive-batch-size",
        action="store_true",
        help="probe a safe fixed per-GPU batch size before CUDA training",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume-from")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_run_config(args.config)
    apply_cli_overrides(
        config,
        target_model_path=args.target_model_path,
        train_data_path=args.train_data_path,
        hidden_states_path=args.hidden_states_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device,
        attention_backend=args.attention_backend,
        adaptive_batch_size=True if args.adaptive_batch_size else None,
        run_id=args.run_id,
    )
    if args.smoke:
        config.training.max_steps = min(config.training.max_steps or 1, 1)
        config.training.attention_backend = "eager"
        config.training.save_interval = 1
        config.training.eval_interval = 1
        config.validate()
    path = run_training(config, resume_from=args.resume_from)
    if os.environ.get("RANK", "0") == "0":
        print(f"final_checkpoint={path}")


if __name__ == "__main__":
    main()


__all__ = ["evaluate_checkpoint", "main", "run_training"]
