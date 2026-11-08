"""Map-reduce framework: ``Metric`` + ``evaluate`` driver + ``running_summary`` helper."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, NamedTuple

from gradient_moment.stats import BatchPair, RunningStats, stats_mean, stats_stderr


class Metric(NamedTuple):
    name: str
    init: Callable[[], Any]
    map_fn: Callable[[Any, Any], Any]
    reduce_fn: Callable[[Any, Any], Any]
    finalize: Callable[[Any], Mapping[str, Any]]


def evaluate(
    metrics: Iterable[Metric],
    batches: Iterable[BatchPair],
) -> dict[str, Mapping[str, Any]]:
    """Fold all metrics over one pass of ``batches``; returns ``{name: finalize(state)}``."""
    ms = list(metrics)
    states = [m.init() for m in ms]
    for batch_samples, batch_data in batches:
        for i, m in enumerate(ms):
            states[i] = m.reduce_fn(states[i], m.map_fn(batch_samples, batch_data))
    return {m.name: m.finalize(s) for m, s in zip(ms, states)}


def running_summary(rs: RunningStats) -> dict[str, Any]:
    return {"mean": stats_mean(rs), "stderr": stats_stderr(rs), "n": rs.n}
