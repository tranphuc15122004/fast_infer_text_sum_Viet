from __future__ import annotations

import torch

from Finetuning.distributed import DistributedContext, shard_indices


def test_single_process_context_is_a_safe_fallback() -> None:
    context = DistributedContext(rank=0, local_rank=0, world_size=1)

    assert context.is_main_process
    assert not context.is_distributed
    assert context.global_batch_size(3, 2) == 6
    value = torch.tensor(3, dtype=torch.int64)
    assert context.all_reduce_min(value).item() == 3
    assert context.all_reduce_max(value).item() == 3


def test_shard_indices_partition_work_without_overlap() -> None:
    shards = [
        list(shard_indices(length=11, rank=rank, world_size=3))
        for rank in range(3)
    ]

    assert shards == [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8]]
    assert sorted(index for shard in shards for index in shard) == list(range(11))
