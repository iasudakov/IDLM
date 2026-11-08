"""Per-row Shannon entropy of empirical token counts, averaged over rows."""

from __future__ import annotations

import math
from typing import Any, Mapping, NamedTuple

import torch

from gradient_moment.metrics.base import Metric, running_summary
from gradient_moment.stats import RunningStats, init_stats, update_stats_many


def _row_entropy(row: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of one row's empirical token distribution; ``[0, log T]``."""
    T = row.shape[-1]
    counts = torch.bincount(row.long())
    c_per_pos = counts[row.long()]
    return math.log(T) - torch.log(c_per_pos.float()).mean()


def _batched_row_entropy(batch: torch.Tensor) -> torch.Tensor:
    return torch.stack([_row_entropy(row) for row in batch])


class _State(NamedTuple):
    samples: RunningStats
    data: RunningStats


def make_sample_entropy_metric(*, name: str = "sample_entropy") -> Metric:
    """Per-row Shannon entropy averaged over rows, reported for samples and data. ``n`` counts rows."""
    def map_fn(batch_samples: torch.Tensor, batch_data: torch.Tensor):
        return (
            _batched_row_entropy(batch_samples).detach().cpu().numpy(),
            _batched_row_entropy(batch_data).detach().cpu().numpy(),
        )

    def reduce_fn(state: _State, mapped) -> _State:
        rows_samples, rows_data = mapped
        return _State(
            samples=update_stats_many(state.samples, rows_samples),
            data=update_stats_many(state.data, rows_data),
        )

    def finalize(state: _State) -> Mapping[str, Any]:
        return {
            "samples": running_summary(state.samples),
            "data": running_summary(state.data),
        }

    return Metric(
        name=name,
        init=lambda: _State(init_stats(), init_stats()),
        map_fn=map_fn,
        reduce_fn=reduce_fn,
        finalize=finalize,
    )


__all__ = ["make_sample_entropy_metric"]
