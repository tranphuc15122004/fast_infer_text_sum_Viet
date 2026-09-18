"""Atomic draft-only checkpoint management for Finetuning."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping

import torch


_DRAFT_EXPORT_STATE = "draft_state_dict.pt"
_DRAFT_EXPORT_METADATA = "draft_metadata.json"
_DRAFT_EXPORT_COMPLETE = "COMPLETE"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _model_state(model: Any) -> dict[str, Any]:
    module = model.trainable_module() if hasattr(model, "trainable_module") else model
    state = module.state_dict()
    if hasattr(model, "checkpoint_state_filter"):
        state = model.checkpoint_state_filter(state)
    return dict(state)


def export_draft(
    output_dir: str | Path,
    draft_model: torch.nn.Module,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically export portable draft weights plus strict provenance."""

    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        torch.save(dict(draft_model.state_dict()), staging / _DRAFT_EXPORT_STATE)
        (staging / _DRAFT_EXPORT_METADATA).write_text(
            json.dumps(dict(metadata), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (staging / _DRAFT_EXPORT_COMPLETE).write_text("ok\n", encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_draft_initialization(
    export_dir: str | Path,
    draft_model: torch.nn.Module,
    expected_metadata: Mapping[str, Any],
) -> None:
    """Strictly load a portable draft export after provenance validation."""

    source = Path(export_dir)
    required = (_DRAFT_EXPORT_STATE, _DRAFT_EXPORT_METADATA, _DRAFT_EXPORT_COMPLETE)
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"portable draft export is incomplete: missing {missing}")
    stored_metadata = json.loads(
        (source / _DRAFT_EXPORT_METADATA).read_text(encoding="utf-8")
    )
    if stored_metadata != dict(expected_metadata):
        raise ValueError("draft export metadata mismatch")
    state = torch.load(source / _DRAFT_EXPORT_STATE, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("portable draft state must be a tensor mapping")
    try:
        draft_model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"draft export state mismatch: {exc}") from exc


class CheckpointManager:
    """Save complete training state in atomically published directories."""

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        max_checkpoints: int = 0,
    ) -> None:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a non-empty directory-safe name")
        if max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.max_checkpoints = max_checkpoints
        self._latest_link = self.output_dir / f"{run_id}-latest"

    def _step_dir(self, step: int) -> Path:
        if not isinstance(step, int) or step <= 0:
            raise ValueError("checkpoint step must be a positive integer")
        return self.output_dir / f"{self.run_id}-step{step}"

    def save(
        self,
        step: int,
        model: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        trainer_state: Mapping[str, Any],
        extra: Mapping[str, Any],
        *,
        draft_export_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        destination = self._step_dir(step)
        if destination.exists():
            shutil.rmtree(destination)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{self.run_id}-step{step}-", dir=self.output_dir)
        )
        try:
            draft_state = _model_state(model)
            torch.save(draft_state, staging / "draft_state_dict.pt")
            torch.save(optimizer.state_dict(), staging / "optimizer.pt")
            torch.save(scheduler.state_dict(), staging / "scheduler.pt")
            torch.save(capture_rng_state(), staging / "rng_state.pt")
            (staging / "trainer_state.json").write_text(
                json.dumps(dict(trainer_state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (staging / "extra.json").write_text(
                json.dumps(dict(extra), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if draft_export_metadata is not None:
                draft_export = staging / "draft_export"
                draft_export.mkdir()
                torch.save(draft_state, draft_export / _DRAFT_EXPORT_STATE)
                (draft_export / _DRAFT_EXPORT_METADATA).write_text(
                    json.dumps(
                        dict(draft_export_metadata),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                (draft_export / _DRAFT_EXPORT_COMPLETE).write_text(
                    "ok\n", encoding="utf-8"
                )
            (staging / "COMPLETE").write_text("ok\n", encoding="utf-8")
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        temporary_link = self.output_dir / f".{self.run_id}-latest.tmp"
        if temporary_link.exists() or temporary_link.is_symlink():
            temporary_link.unlink()
        temporary_link.symlink_to(destination.name)
        os.replace(temporary_link, self._latest_link)
        self._rotate()
        return destination

    def _complete_steps(self) -> list[Path]:
        paths = []
        prefix = f"{self.run_id}-step"
        for path in self.output_dir.glob(f"{prefix}*"):
            if not path.is_dir() or not (path / "COMPLETE").is_file():
                continue
            suffix = path.name.removeprefix(prefix)
            if suffix.isdigit():
                paths.append(path)
        return sorted(paths, key=lambda item: int(item.name.removeprefix(prefix)))

    def _rotate(self) -> None:
        if not self.max_checkpoints:
            return
        complete = self._complete_steps()
        for path in complete[:-self.max_checkpoints]:
            shutil.rmtree(path)

    def latest_dir(self) -> Path:
        if not self._latest_link.is_symlink():
            raise FileNotFoundError(f"latest checkpoint not found: {self._latest_link}")
        target = self._latest_link.resolve()
        if not target.is_dir() or not (target / "COMPLETE").is_file():
            raise FileNotFoundError(f"latest checkpoint is incomplete: {target}")
        return target

    def resolve_resume_dir(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_file():
            candidate = candidate.parent
        if candidate == self.output_dir or candidate.name == self.run_id:
            return self.latest_dir()
        if candidate.name.endswith("-latest") and candidate.is_symlink():
            candidate = candidate.resolve()
        if candidate.is_dir() and (candidate / "COMPLETE").is_file():
            return candidate
        if candidate.is_dir():
            candidates = sorted(
                candidate.glob(f"{self.run_id}-step*"),
                key=lambda item: int(item.name.removeprefix(f"{self.run_id}-step"))
                if item.name.removeprefix(f"{self.run_id}-step").isdigit()
                else -1,
            )
            candidates = [item for item in candidates if (item / "COMPLETE").is_file()]
            if candidates:
                return candidates[-1]
        raise FileNotFoundError(f"cannot resolve checkpoint directory from {path}")

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        root = self.resolve_resume_dir(path)
        required = (
            "draft_state_dict.pt",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pt",
            "trainer_state.json",
            "extra.json",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError(f"checkpoint is incomplete: missing {missing}")
        with (root / "trainer_state.json").open(encoding="utf-8") as handle:
            trainer_state = json.load(handle)
        with (root / "extra.json").open(encoding="utf-8") as handle:
            extra = json.load(handle)
        return {
            "path": root,
            "draft_state_dict": torch.load(root / "draft_state_dict.pt", map_location=map_location, weights_only=True),
            "optimizer": torch.load(root / "optimizer.pt", map_location=map_location, weights_only=True),
            "scheduler": torch.load(root / "scheduler.pt", map_location=map_location, weights_only=True),
            # RNG tensors stay on CPU; torch.set_rng_state expects a CPU byte
            # tensor even when model/optimizer state is restored onto CUDA.
            "rng_state": torch.load(root / "rng_state.pt", map_location="cpu", weights_only=False),
            "trainer_state": trainer_state,
            "extra": extra,
        }


__all__ = [
    "CheckpointManager",
    "capture_rng_state",
    "export_draft",
    "load_draft_initialization",
    "restore_rng_state",
]
