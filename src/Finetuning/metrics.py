"""Small metric aggregation helpers shared by training and evaluation."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from .distributed import DistributedContext


def _scalar(value: Any) -> float:
    tensor = torch.as_tensor(value).detach().float()
    if tensor.numel() != 1:
        tensor = tensor.sum()
    result = float(tensor.cpu())
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError("non-finite metric encountered")
    return result


def aggregate_step_outputs(
    outputs: Iterable[Any],
    distributed_context: DistributedContext | None = None,
) -> dict[str, float]:
    """Aggregate StepOutput objects using additive numerators where present."""

    collected = list(outputs)
    if not collected:
        raise ValueError("empty validation loader")
    loss_num = 0.0
    loss_den = 0.0
    fallback_losses: list[float] = []
    ratio_totals: dict[str, list[float]] = {}
    metric_values: dict[str, list[float]] = {}
    for output in collected:
        if output.loss_terms is not None:
            loss_num += _scalar(output.loss_terms[0])
            loss_den += _scalar(output.loss_terms[1])
        else:
            fallback_losses.append(_scalar(output.loss))
        for name, pair in output.ratio_metrics.items():
            values = ratio_totals.setdefault(name, [0.0, 0.0])
            values[0] += _scalar(pair[0])
            values[1] += _scalar(pair[1])
        for name, value in output.metrics.items():
            if name == "accuracy_denom":
                continue
            metric_values.setdefault(name, []).append(_scalar(value))

    if distributed_context is not None and distributed_context.is_distributed:
        collective_device = torch.device(
            "cuda" if torch.cuda.is_available() and torch.distributed.get_backend() == "nccl" else "cpu"
        )
        if loss_den > 0:
            reduced = distributed_context.all_reduce_sum(
                torch.tensor([loss_num, loss_den], dtype=torch.float64, device=collective_device)
            ).cpu()
            loss_num, loss_den = float(reduced[0]), float(reduced[1])
        elif fallback_losses:
            reduced = distributed_context.all_reduce_sum(
                torch.tensor(
                    [sum(fallback_losses), len(fallback_losses)],
                    dtype=torch.float64,
                    device=collective_device,
                )
            ).cpu()
            fallback_losses = [float(reduced[0])]
            fallback_count = float(reduced[1])
        else:
            fallback_count = 0.0
        for name, values in ratio_totals.items():
            reduced = distributed_context.all_reduce_sum(
                torch.tensor(values, dtype=torch.float64, device=collective_device)
            ).cpu()
            ratio_totals[name] = [float(reduced[0]), float(reduced[1])]
        for name, values in metric_values.items():
            reduced = distributed_context.all_reduce_sum(
                torch.tensor(
                    [sum(values), len(values)],
                    dtype=torch.float64,
                    device=collective_device,
                )
            ).cpu()
            metric_values[name] = [float(reduced[0] / reduced[1])]
    else:
        fallback_count = float(len(fallback_losses))

    if loss_den > 0:
        result = {"loss": loss_num / loss_den}
    elif fallback_losses:
        result = {"loss": sum(fallback_losses) / fallback_count}
    else:
        raise ValueError("validation produced no positive loss denominator")
    for name, (numerator, denominator) in ratio_totals.items():
        if denominator <= 0:
            raise ValueError(f"metric {name} has no valid denominator")
        result[name] = numerator / denominator
        if name == "acc":
            result["accuracy"] = result[name]
            result["accuracy_denom"] = denominator
    for name, values in metric_values.items():
        if name not in result:
            result[name] = sum(values) / len(values)
    for name, value in result.items():
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"non-finite aggregate metric: {name}")
    return result


__all__ = ["aggregate_step_outputs"]
