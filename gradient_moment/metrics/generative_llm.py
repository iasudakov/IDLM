"""Bundled generative-LLM metrics: GM + CE/PPL + predictive entropy.

Reports each quantity for both ``samples`` (model's generative distribution)
and ``data`` (reference/training distribution). The reference model is
exposed via a ``forward_fn(model, batch) -> (nll, entropy)`` returning two
scalars (mean token NLL and mean per-position predictive entropy). One
forward + backward per sub-batch yields the gradient (for the GM moments)
plus both scalars (for CE / PPL / generative entropy).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, NamedTuple

import torch

from gradient_moment.core import (
    ForwardFn,
    _split_halves,
    _trainable_params,
    tree_inner_product,
)
from gradient_moment.metrics.base import Metric, running_summary
from gradient_moment.stats import (
    RunningStats,
    init_stats,
    stats_mean,
    stats_stderr,
    update_stats,
)


class _Estimate(NamedTuple):
    centered: torch.Tensor
    uncentered_samples: torch.Tensor
    uncentered_data: torch.Tensor
    nll_samples: torch.Tensor
    nll_data: torch.Tensor
    ent_samples: torch.Tensor
    ent_data: torch.Tensor


class _State(NamedTuple):
    centered: RunningStats
    uncentered_samples: RunningStats
    uncentered_data: RunningStats
    nll_samples: RunningStats
    nll_data: RunningStats
    ent_samples: RunningStats
    ent_data: RunningStats


def _forward_with_grad(
    forward_fn: ForwardFn,
    model: torch.nn.Module,
    batch: Any,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """One forward + backward: returns (grad_of_nll, nll_value, entropy_value)."""
    params = _trainable_params(model)
    nll, entropy = forward_fn(model, batch)
    grads = torch.autograd.grad(nll, params)
    return list(grads), nll.detach(), entropy.detach()


def _per_pair_with_aux(
    forward_fn: ForwardFn,
    model: torch.nn.Module,
    batch_samples: Any,
    batch_data: Any,
) -> _Estimate:
    x1_s, x2_s = _split_halves(batch_samples)
    x1_d, x2_d = _split_halves(batch_data)
    g_s1, nll_s1, ent_s1 = _forward_with_grad(forward_fn, model, x1_s)
    g_s2, nll_s2, ent_s2 = _forward_with_grad(forward_fn, model, x2_s)
    g_d1, nll_d1, ent_d1 = _forward_with_grad(forward_fn, model, x1_d)
    g_d2, nll_d2, ent_d2 = _forward_with_grad(forward_fn, model, x2_d)
    diff1 = [a - b for a, b in zip(g_s1, g_d1)]
    diff2 = [a - b for a, b in zip(g_s2, g_d2)]
    return _Estimate(
        centered=tree_inner_product(diff1, diff2),
        uncentered_samples=tree_inner_product(g_s1, g_s2),
        uncentered_data=tree_inner_product(g_d1, g_d2),
        nll_samples=(nll_s1 + nll_s2) / 2,
        nll_data=(nll_d1 + nll_d2) / 2,
        ent_samples=(ent_s1 + ent_s2) / 2,
        ent_data=(ent_d1 + ent_d2) / 2,
    )


def _ppl_summary(nll: RunningStats) -> dict[str, Any]:
    """``exp(mean CE)`` with delta-method stderr ``ppl * stderr(CE)``."""
    if nll.n == 0:
        return {"mean": float("nan"), "stderr": float("nan"), "n": 0}
    mean_ce = stats_mean(nll)
    se_ce = stats_stderr(nll)
    ppl = math.exp(mean_ce)
    return {"mean": ppl, "stderr": ppl * se_ce, "n": nll.n}


def make_generative_llm_metrics(
    forward_fn: ForwardFn,
    model: torch.nn.Module,
    *,
    name: str = "generative_llm_metrics",
) -> Metric:
    """Centered/uncentered GM + generative CE/PPL + generative entropy, in one pass.

    ``forward_fn(model, batch) -> (mean_nll, mean_predictive_entropy)``. The NLL
    must be differentiable w.r.t. the model's trainable parameters; entropy
    just needs to be a detached scalar tensor.
    """
    def _init() -> _State:
        return _State(*(init_stats() for _ in range(7)))

    def map_fn(batch_samples: Any, batch_data: Any) -> _Estimate:
        return _per_pair_with_aux(forward_fn, model, batch_samples, batch_data)

    def reduce_fn(state: _State, est: _Estimate) -> _State:
        return _State(
            centered=update_stats(state.centered, float(est.centered)),
            uncentered_samples=update_stats(state.uncentered_samples, float(est.uncentered_samples)),
            uncentered_data=update_stats(state.uncentered_data, float(est.uncentered_data)),
            nll_samples=update_stats(state.nll_samples, float(est.nll_samples)),
            nll_data=update_stats(state.nll_data, float(est.nll_data)),
            ent_samples=update_stats(state.ent_samples, float(est.ent_samples)),
            ent_data=update_stats(state.ent_data, float(est.ent_data)),
        )

    def finalize(state: _State) -> Mapping[str, Any]:
        return {
            "centered_gm":                running_summary(state.centered),
            "uncentered_gm_samples":      running_summary(state.uncentered_samples),
            "uncentered_gm_data":         running_summary(state.uncentered_data),
            "generative_ce_samples":      running_summary(state.nll_samples),
            "generative_ce_data":         running_summary(state.nll_data),
            "generative_ppl_samples":     _ppl_summary(state.nll_samples),
            "generative_ppl_data":        _ppl_summary(state.nll_data),
            "generative_entropy_samples": running_summary(state.ent_samples),
            "generative_entropy_data":    running_summary(state.ent_data),
        }

    return Metric(name=name, init=_init, map_fn=map_fn, reduce_fn=reduce_fn, finalize=finalize)


__all__ = ["ForwardFn", "make_generative_llm_metrics"]
