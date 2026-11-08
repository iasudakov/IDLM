"""Welford-style running statistics + shared data containers.

Used by the streaming evaluator to fold per-pair scalar estimates without
buffering. Functional / immutable: each ``update_stats`` returns a fresh
``RunningStats``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


BatchPair = tuple[Any, Any]  # (batch_samples, batch_data)


class GMEstimate(NamedTuple):
    """Per-pair estimate. Individual values can be negative; means converge to squared norms."""
    centered: Any            # ||E_samples[grad L] - E_data[grad L]||^2
    uncentered_samples: Any  # ||E_samples[grad L]||^2
    uncentered_data: Any     # ||E_data[grad L]||^2


# ---- Streaming aggregation: pure functional Welford-style stats. ----

class RunningStats(NamedTuple):
    """Immutable running statistics over a stream of scalars."""
    n: int
    sum: float
    sum_sq: float


def init_stats() -> RunningStats:
    return RunningStats(n=0, sum=0.0, sum_sq=0.0)


def update_stats(stats: RunningStats, value: float) -> RunningStats:
    return RunningStats(
        n=stats.n + 1,
        sum=stats.sum + value,
        sum_sq=stats.sum_sq + value * value,
    )


def update_stats_many(stats: RunningStats, values: Any) -> RunningStats:
    """Fold all elements of ``values`` (any array-like) into ``stats`` in one host-side reduction."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return stats
    return RunningStats(
        n=stats.n + int(v.size),
        sum=stats.sum + float(v.sum()),
        sum_sq=stats.sum_sq + float((v * v).sum()),
    )


def stats_mean(stats: RunningStats) -> float:
    if stats.n == 0:
        raise ValueError("Empty stats; call update_stats first.")
    return stats.sum / stats.n


def stats_stderr(stats: RunningStats) -> float:
    """NaN when n < 2."""
    if stats.n < 2:
        return float("nan")
    var = (stats.sum_sq - stats.sum * stats.sum / stats.n) / (stats.n - 1)
    return (max(var, 0.0) / stats.n) ** 0.5
