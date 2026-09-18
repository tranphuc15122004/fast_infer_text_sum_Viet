from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import pytest

from Finetuning.distributed import cleanup_distributed, initialize_distributed
from Finetuning.distributed import DistributedContext
from Finetuning.run_train import _loader
from Finetuning.strategy import DFlashTrainStrategy
from Finetuning.trainer import Trainer


class _TinyDFlash(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.draft_model = nn.Linear(1, 1, bias=False)

    def forward(self, *, input_ids, hidden_states, loss_mask):
        del loss_mask
        prediction = self.draft_model(hidden_states).squeeze(-1)
        target = input_ids.float()
        loss = (prediction - target).square().mean()
        return loss, torch.tensor(1.0), {}


def test_strategy_keeps_single_process_forward_path_unchanged() -> None:
    strategy = DFlashTrainStrategy(_TinyDFlash())
    strategy.configure_distributed(DistributedContext())

    assert strategy.forward_model is strategy.dflash_model


class _ToyFeatureDataset:
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        return {
            "input_ids": torch.tensor([index + 1, index + 2]),
            "loss_mask": torch.tensor([0.0, 1.0]),
            "hidden_states": torch.zeros((2, 1)),
        }


def test_distributed_loader_assigns_equal_local_batches() -> None:
    loader = _loader(
        _ToyFeatureDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        distributed_context=DistributedContext(rank=1, local_rank=1, world_size=2),
    )

    assert len(loader) == 2
    assert sum(batch["input_ids"].shape[0] for batch in loader) == 4


def _distributed_worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "INIT_METHOD": init_method,
            "GLOO_SOCKET_IFNAME": "lo",
        }
    )
    context = initialize_distributed("cpu", backend="gloo")
    try:
        strategy = DFlashTrainStrategy(_TinyDFlash())
        strategy.configure_distributed(context)
        batch = {
            "input_ids": torch.full((1, 1), float(rank + 1), dtype=torch.long),
            "hidden_states": torch.ones((1, 1, 1), dtype=torch.float32),
            "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        }
        trainer = Trainer(
            strategy=strategy,
            train_dataloader=[batch],
            output_dir=output_dir,
            run_id="ddp",
            max_steps=1,
            save_interval=1,
            device="cpu",
            distributed_context=context,
        )
        trainer.fit()
        torch.save(strategy.dflash_model.draft_model.weight.detach().cpu(), Path(output_dir) / f"rank{rank}.pt")
    finally:
        cleanup_distributed(context)


@pytest.mark.skipif(
    os.environ.get("FINETUNING_RUN_DISTRIBUTED_TESTS") != "1",
    reason="set FINETUNING_RUN_DISTRIBUTED_TESTS=1 when process-group networking is available",
)
def test_ddp_synchronizes_gradients_and_rank_zero_writes_checkpoint(tmp_path) -> None:
    output_dir = tmp_path / "ddp-run"
    init_file = tmp_path / "process-group"
    mp.spawn(
        _distributed_worker,
        args=(2, f"file://{init_file}", str(output_dir)),
        nprocs=2,
        join=True,
    )

    rank_zero = torch.load(output_dir / "rank0.pt", weights_only=True)
    rank_one = torch.load(output_dir / "rank1.pt", weights_only=True)
    assert torch.equal(rank_zero, rank_one)
    assert (output_dir / "ddp-step1" / "COMPLETE").is_file()
    assert (output_dir / "metrics.jsonl").read_text(encoding="utf-8").count("\n") == 1
