"""PyTorch implementation of the Gradient Moment primitives (arXiv:2603.20155, Eq. 14).

Per-pair unbiased estimators of three squared gradient norms. ``samples``
denotes the model's generative distribution (paper notation: ``g``); ``data``
denotes the reference/training distribution (paper notation: ``q``).

    centered            = < grad L(x1_samples) - grad L(x1_data),
                            grad L(x2_samples) - grad L(x2_data) >
    uncentered_samples  = < grad L(x1_samples), grad L(x2_samples) >
    uncentered_data     = < grad L(x1_data),    grad L(x2_data) >.

Each input batch (one ``samples``, one ``data``) is split in half along axis 0
internally; leading axes must be even. Streaming aggregation lives in
:mod:`gradient_moment.metrics`.

Gradients go through ``torch.autograd.grad`` so the model's ``.grad`` buffers
are never mutated and parallel autograd calls don't collide.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch

from gradient_moment.stats import BatchPair, GMEstimate


LossFn = Callable[[torch.nn.Module, Any], torch.Tensor]
ForwardFn = Callable[[torch.nn.Module, Any], tuple[torch.Tensor, torch.Tensor]]


def _trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def tree_inner_product(
    tensors_a: Sequence[torch.Tensor],
    tensors_b: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Sum_i <a_i, b_i> over paired tensors. Empty input returns a 0-d zero tensor."""
    total: torch.Tensor | None = None
    for a, b in zip(tensors_a, tensors_b):
        term = (a * b).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros(())
    return total


def _split_halves(batch: Any) -> tuple[Any, Any]:
    """Split a batch in half along axis 0. Handles tensor / Mapping / sequence."""
    if isinstance(batch, torch.Tensor):
        n = batch.shape[0]
        if n % 2 != 0:
            raise ValueError(f"Batch leading axis must be even, got {n}.")
        h = n // 2
        return batch[:h], batch[h:]
    if isinstance(batch, Mapping):
        firsts: dict[Any, Any] = {}
        seconds: dict[Any, Any] = {}
        for k, v in batch.items():
            a, b = _split_halves(v)
            firsts[k], seconds[k] = a, b
        return type(batch)(firsts), type(batch)(seconds)
    if isinstance(batch, (list, tuple)):
        pairs = [_split_halves(v) for v in batch]
        first = [a for a, _ in pairs]
        second = [b for _, b in pairs]
        return type(batch)(first), type(batch)(second)
    raise TypeError(f"Unsupported batch type for split: {type(batch).__name__}")


def diff_grad(
    loss_fn: LossFn,
    model: torch.nn.Module,
    batch_samples: Any,
    batch_data: Any,
) -> list[torch.Tensor]:
    """grad_params ( loss(model, samples) - loss(model, data) ) in one autograd call."""
    params = _trainable_params(model)
    loss = loss_fn(model, batch_samples) - loss_fn(model, batch_data)
    return list(torch.autograd.grad(loss, params))


def _grad_of_loss(
    loss_fn: LossFn,
    model: torch.nn.Module,
    batch: Any,
) -> list[torch.Tensor]:
    params = _trainable_params(model)
    loss = loss_fn(model, batch)
    return list(torch.autograd.grad(loss, params))


def _per_pair_estimate(
    loss_fn: LossFn,
    model: torch.nn.Module,
    batch_samples: Any,
    batch_data: Any,
) -> GMEstimate:
    x1_s, x2_s = _split_halves(batch_samples)
    x1_d, x2_d = _split_halves(batch_data)
    g1 = _grad_of_loss(loss_fn, model, x1_s)
    g2 = _grad_of_loss(loss_fn, model, x2_s)
    d1 = _grad_of_loss(loss_fn, model, x1_d)
    d2 = _grad_of_loss(loss_fn, model, x2_d)
    diff1 = [a - b for a, b in zip(g1, d1)]
    diff2 = [a - b for a, b in zip(g2, d2)]
    return GMEstimate(
        centered=tree_inner_product(diff1, diff2),
        uncentered_samples=tree_inner_product(g1, g2),
        uncentered_data=tree_inner_product(d1, d2),
    )


def gradient_moment_estimate(
    loss_fn: LossFn,
    model: torch.nn.Module,
    batch_samples: Any,
    batch_data: Any,
) -> GMEstimate:
    """One unbiased per-pair estimate. Each batch is split in half along axis 0."""
    return _per_pair_estimate(loss_fn, model, batch_samples, batch_data)


__all__ = [
    "BatchPair",
    "ForwardFn",
    "GMEstimate",
    "LossFn",
    "diff_grad",
    "gradient_moment_estimate",
    "tree_inner_product",
]
