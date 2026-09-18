"""Single-process optimizer lifecycle for the self-contained DFlash port."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from .checkpoint import CheckpointManager, load_draft_initialization, restore_rng_state
from .distributed import DistributedContext
from .evaluation import Evaluator
from .schedule import build_scheduler, current_lr, resolve_total_steps, validate_fixed_accumulation_plan
from .strategy import StepContext, StepOutput, TrainBatch


def _as_batch(value: Any) -> TrainBatch:
    if isinstance(value, TrainBatch):
        return value
    if isinstance(value, dict):
        return TrainBatch(tensors=value)
    raise TypeError(f"dataloader item must be a mapping or TrainBatch, got {type(value)!r}")


class Trainer:
    """Train a strategy with fixed gradient accumulation and checkpoints."""

    def __init__(
        self,
        strategy: Any,
        train_dataloader: Iterable[Any],
        output_dir: str | Path,
        run_id: str,
        *,
        validation_dataloader: Iterable[Any] | None = None,
        batch_size: int = 1,
        accumulation_steps: int = 1,
        num_epochs: int = 1,
        max_steps: int | None = None,
        learning_rate: float = 6e-4,
        weight_decay: float = 0.0,
        warmup_steps: int | None = None,
        warmup_ratio: float | None = None,
        scheduler_type: str = "cosine",
        max_grad_norm: float = 1.0,
        save_interval: int = 1,
        log_interval: int = 1,
        eval_interval: int = 0,
        hardware_peak_tflops: float | None = None,
        device: torch.device | str | None = None,
        extra_checkpoint_state: dict[str, Any] | None = None,
        draft_export_metadata: dict[str, Any] | None = None,
        resume_from: str | Path | None = None,
        distributed_context: DistributedContext | None = None,
    ) -> None:
        if batch_size <= 0 or accumulation_steps <= 0 or num_epochs <= 0:
            raise ValueError("batch_size, accumulation_steps and num_epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.strategy = strategy
        self.distributed_context = distributed_context or DistributedContext()
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.max_grad_norm = max_grad_norm
        self.save_interval = max(1, save_interval)
        self.log_interval = max(1, log_interval)
        self.eval_interval = max(0, eval_interval)
        self.hardware_peak_tflops = hardware_peak_tflops
        self.extra_checkpoint_state = extra_checkpoint_state or {}
        self.draft_export_metadata = draft_export_metadata
        self._train_loader = train_dataloader
        if self._loader_length(train_dataloader) == 0:
            raise ValueError("empty training loader")
        self._validation_loader = validation_dataloader
        num_samples = self._loader_length(train_dataloader) * batch_size
        validate_fixed_accumulation_plan(
            num_samples=num_samples,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )
        self.total_steps = resolve_total_steps(
            total_steps=None,
            max_steps=max_steps,
            num_samples=num_samples,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            num_epochs=num_epochs,
        )
        module = strategy.trainable_module()
        module.to(self.device)
        self._trainable_parameter_count = sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        )
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in module.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            scheduler_type=scheduler_type,
        )
        self.checkpoint_manager = CheckpointManager(self.output_dir, run_id)
        self.evaluator = Evaluator(self.distributed_context)
        self.global_step = 0
        self.micro_step = 0
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.log_path = self.output_dir / "train.log"
        self._last_eval_step: int | None = None
        if resume_from is not None:
            self._resume(resume_from)

    @staticmethod
    def _loader_length(value: Iterable[Any]) -> int:
        try:
            length = len(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "trainer requires a sized, re-iterable dataloader; "
                "stream data through a DataLoader instead of a generator"
            ) from exc
        if length < 0:
            raise ValueError("dataloader length must be non-negative")
        return int(length)

    def _write_record(self, record: dict[str, Any]) -> None:
        if not self.distributed_context.is_main_process:
            return
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"step={record.get('step', '?')} loss={record.get('loss', '?')} "
                f"lr={record.get('lr', '?')}\n"
            )

    def _save(self) -> Path:
        path = self.output_dir / f"{self.run_id}-step{self.global_step}"
        if self.distributed_context.is_main_process:
            self.checkpoint_manager.save(
                self.global_step,
                self.strategy,
                self.optimizer,
                self.scheduler,
                {
                    "global_step": self.global_step,
                    "micro_step": self.micro_step,
                },
                {
                    "strategy": getattr(self.strategy, "name", type(self.strategy).__name__),
                    "total_steps": self.total_steps,
                    "distributed_world_size": self.distributed_context.world_size,
                    "global_batch_size": self.distributed_context.global_batch_size(
                        self.batch_size,
                        self.accumulation_steps,
                    ),
                    **self.extra_checkpoint_state,
                },
                draft_export_metadata=self.draft_export_metadata,
            )
        self.distributed_context.barrier()
        return path

    def _resume(self, path: str | Path) -> None:
        state = self.checkpoint_manager.load(path, map_location=self.device)
        stored_world_size = state.get("extra", {}).get("distributed_world_size")
        if (
            stored_world_size is not None
            and int(stored_world_size) != self.distributed_context.world_size
        ):
            raise ValueError(
                "checkpoint distributed_world_size does not match the current run: "
                f"{stored_world_size} != {self.distributed_context.world_size}"
            )
        stored_global_batch_size = state.get("extra", {}).get("global_batch_size")
        if stored_global_batch_size is not None:
            current_global_batch_size = self.distributed_context.global_batch_size(
                self.batch_size,
                self.accumulation_steps,
            )
            if int(stored_global_batch_size) != current_global_batch_size:
                raise ValueError(
                    "checkpoint global_batch_size does not match the current run: "
                    f"{stored_global_batch_size} != {current_global_batch_size}"
                )
        module = self.strategy.trainable_module()
        load_module = module
        if hasattr(self.strategy, "dflash_model"):
            load_module = self.strategy.dflash_model.draft_model
        if self.draft_export_metadata is not None:
            # A DFlash resume must preserve the frozen target, selected target
            # layers and draft architecture.  The embedded export provides a
            # strict state-dict and metadata boundary; silently using
            # ``strict=False`` here could otherwise produce a corrupted run.
            load_draft_initialization(
                state["path"] / "draft_export",
                load_module,
                self.draft_export_metadata,
            )
        else:
            load_module.load_state_dict(state["draft_state_dict"], strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        trainer_state = state.get("trainer_state", {})
        self.global_step = int(trainer_state.get("global_step", 0))
        self.micro_step = int(trainer_state.get("micro_step", 0))
        restore_rng_state(state["rng_state"])

    def save_checkpoint(self) -> Path:
        if self.global_step <= 0:
            raise ValueError("cannot save a checkpoint before the first optimizer step")
        return self._save()

    def evaluate(self) -> dict[str, float]:
        if self._validation_loader is None:
            raise ValueError("validation loader is not configured")
        return self.evaluator.evaluate_in_memory(
            self.strategy,
            self._validation_loader,
            self.device,
        )

    def fit(self) -> int:
        if self._validation_loader is not None and self._loader_length(self._validation_loader) == 0:
            raise ValueError("empty validation loader")
        module = self.strategy.trainable_module()
        module.train()
        self.optimizer.zero_grad(set_to_none=True)
        window_loss_num = 0.0
        window_loss_den = 0.0
        window_uses_loss_terms: bool | None = None
        window_tokens = 0
        step_started: float | None = None
        for epoch in range(self.num_epochs):
            sampler = getattr(self._train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            for item_index, raw_batch in enumerate(self._train_loader):
                if self.global_step >= self.total_steps:
                    break
                started = time.perf_counter()
                if self.micro_step % self.accumulation_steps == 0:
                    step_started = started
                    window_loss_num = 0.0
                    window_loss_den = 0.0
                    window_uses_loss_terms = None
                    window_tokens = 0
                output: StepOutput = self.strategy.forward_loss(
                    _as_batch(raw_batch),
                    StepContext(self.global_step, self.total_steps),
                )
                if not torch.isfinite(output.loss.detach()).all():
                    raise ValueError("non-finite loss encountered")
                uses_loss_terms = output.loss_terms is not None
                if window_uses_loss_terms is None:
                    window_uses_loss_terms = uses_loss_terms
                elif window_uses_loss_terms != uses_loss_terms:
                    raise ValueError("loss_terms must be present for every microbatch in an accumulation window")
                if uses_loss_terms:
                    numerator, denominator = output.loss_terms
                    numerator = torch.as_tensor(numerator).reshape(())
                    denominator = torch.as_tensor(denominator).detach().reshape(())
                    if not torch.isfinite(numerator.detach()) or not torch.isfinite(denominator):
                        raise ValueError("non-finite additive loss term encountered")
                    if float(denominator.cpu()) <= 0:
                        raise ValueError("additive loss denominator must be positive")
                    (numerator / self.accumulation_steps).backward()
                    window_loss_num += float(numerator.detach().cpu())
                    window_loss_den += float(denominator.cpu())
                else:
                    (output.loss / self.accumulation_steps).backward()
                self.micro_step += 1
                at_boundary = self.micro_step % self.accumulation_steps == 0
                if isinstance(raw_batch, dict) and "input_ids" in raw_batch:
                    window_tokens += int(torch.as_tensor(raw_batch["input_ids"]).numel())
                if not at_boundary:
                    continue
                if window_uses_loss_terms:
                    if window_loss_den <= 0:
                        raise ValueError("accumulated additive loss denominator must be positive")
                    global_loss_den = self.distributed_context.all_reduce_sum(
                        torch.tensor(
                            window_loss_den,
                            dtype=torch.float64,
                            device=self.device,
                        )
                    )
                    if float(global_loss_den.detach().cpu()) <= 0:
                        raise ValueError("global additive loss denominator must be positive")
                    scale = (
                        self.accumulation_steps * self.distributed_context.world_size
                        / float(global_loss_den.detach().cpu())
                    )
                    for parameter in module.parameters():
                        if parameter.requires_grad and parameter.grad is not None:
                            parameter.grad.mul_(scale)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in module.parameters() if parameter.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                elapsed = max(time.perf_counter() - (step_started or started), 1e-9)
                if window_uses_loss_terms:
                    global_loss_values = self.distributed_context.all_reduce_sum(
                        torch.tensor(
                            [window_loss_num, window_loss_den],
                            dtype=torch.float64,
                            device=self.device,
                        )
                    ).detach().cpu()
                    loss = float(global_loss_values[0] / global_loss_values[1])
                else:
                    loss = float(output.loss.detach().cpu())
                tokens = window_tokens
                mfu = None
                if self.hardware_peak_tflops is not None:
                    # A transparent 6*N*token estimate; report null unless a
                    # hardware peak was explicitly supplied by the config.
                    mfu = (
                        0.0
                        if not tokens
                        else (6.0 * self._trainable_parameter_count * tokens / elapsed)
                        / (self.hardware_peak_tflops * 1e12)
                    )
                record = {
                    "type": "step",
                    "step": self.global_step,
                    "epoch": epoch,
                    "loss": loss,
                    "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                    "lr": current_lr(self.optimizer),
                    "step_time_s": elapsed,
                    "tokens_per_s": tokens / elapsed if tokens else 0.0,
                    "mfu": mfu,
                }
                for name, value in output.metrics.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        record[name] = float(value.detach().cpu())
                    elif isinstance(value, (int, float)):
                        record[name] = value
                self._write_record(record)
                if self.global_step % self.save_interval == 0:
                    self._save()
                if (
                    self._validation_loader is not None
                    and self.eval_interval > 0
                    and self.global_step % self.eval_interval == 0
                ):
                    eval_metrics = self.evaluate()
                    self._write_record(
                        {"type": "evaluation", "step": self.global_step, **eval_metrics}
                    )
                    self._last_eval_step = self.global_step
                if self.global_step >= self.total_steps:
                    break
            if self.global_step >= self.total_steps:
                break
        if self.global_step <= 0:
            raise ValueError("training completed without an optimizer step")
        if self.distributed_context.is_main_process:
            try:
                latest = self.checkpoint_manager.latest_dir()
            except FileNotFoundError:
                latest = None
            if latest is None or not latest.exists():
                self._save()
            else:
                self.distributed_context.barrier()
        else:
            self.distributed_context.barrier()
        return self.global_step


__all__ = ["Trainer"]
