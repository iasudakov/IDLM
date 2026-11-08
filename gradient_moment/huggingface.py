"""HuggingFace helper (PyTorch). Requires ``pip install "gradient-moment[huggingface]"``.

The canonical reference LM is **gpt2-large**, pinned here as a constant. A
fixed reference is what makes results comparable across runs; if you need a
different reference, drop down to :func:`causal_lm_forward` with a model you
loaded yourself.

CPU policy
----------
The gradient-moment metric backprops through a 770M-parameter model multiple
times per pair, so running it on CPU is impractically slow. The loaders raise
unless CUDA is available, or unless the caller passes ``allow_cpu=True`` (for
tiny test stubs, not the real LM).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, NamedTuple

import numpy as np
import torch

from gradient_moment.core import ForwardFn
from gradient_moment.metrics import (
    Metric,
    evaluate,
    make_generative_llm_metrics,
    make_sample_entropy_metric,
)
from gradient_moment.metrics.base import running_summary
from gradient_moment.stats import RunningStats, init_stats, update_stats_many


GPT2_LARGE_MODEL_ID = "gpt2-large"
GPT2_TOKENIZER_ID = "gpt2"


# --- shared loaders ---------------------------------------------------------


def _resolve_device(device: str | torch.device | None, *, allow_cpu: bool) -> torch.device:
    if device is not None:
        d = torch.device(device)
    elif torch.cuda.is_available():
        d = torch.device("cuda")
    else:
        d = torch.device("cpu")
    if d.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "The gradient-moment metric is too heavy for CPU with a real reference LM. "
            "Run on a CUDA device, or pass allow_cpu=True (intended for tiny test stubs)."
        )
    return d


def _load_gpt2_large(
    *,
    device: str | torch.device | None,
    allow_cpu: bool,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Resolve the device, load gpt2-large to it in eval mode."""
    from transformers import AutoModelForCausalLM  # local: optional dep

    d = _resolve_device(device, allow_cpu=allow_cpu)
    model = AutoModelForCausalLM.from_pretrained(GPT2_LARGE_MODEL_ID, torch_dtype=dtype)
    return model.to(d).eval()


def _load_gpt2_tokenizer():
    from transformers import AutoTokenizer  # local: optional dep

    tok = AutoTokenizer.from_pretrained(GPT2_TOKENIZER_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return type(batch)({k: _move_to_device(v, device) for k, v in batch.items()})
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(v, device) for v in batch)
    return batch


# --- token-id path ----------------------------------------------------------


def causal_lm_forward(model: torch.nn.Module) -> ForwardFn:
    """``(model, input_ids) -> (mean_nll, mean_predictive_entropy)`` for a HF causal LM.

    Both quantities use the standard 1-position shift and the same softmax pass.
    """
    def forward_fn(_model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = _model(input_ids).logits
        shifted = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(shifted, dim=-1)
        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        nll = token_nll.mean()
        with torch.no_grad():
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(-1).mean()
        return nll, entropy
    return forward_fn


def load_gpt2_large_reference(
    *,
    device: str | torch.device | None = None,
    allow_cpu: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, ForwardFn]:
    """Load gpt2-large + a token-id forward_fn."""
    model = _load_gpt2_large(device=device, allow_cpu=allow_cpu, dtype=dtype)
    return model, causal_lm_forward(model)


def gpt2_large_evaluate(
    batches: Iterable[tuple[Any, Any]],
    *,
    device: str | torch.device | None = None,
    allow_cpu: bool = False,
    include_entropy: bool = True,
) -> dict[str, Mapping[str, Any]]:
    """Run the metrics with gpt2-large on token-id batches.

    Each batch is ``(batch_samples, batch_data)`` with even leading axis.
    """
    model, forward_fn = load_gpt2_large_reference(device=device, allow_cpu=allow_cpu)
    d = next(model.parameters()).device
    metrics: list[Metric] = [make_generative_llm_metrics(forward_fn, model)]
    if include_entropy:
        metrics.append(make_sample_entropy_metric())
    return evaluate(metrics, ((_move_to_device(bs, d), _move_to_device(bd, d)) for bs, bd in batches))


# --- text path --------------------------------------------------------------


def _tokenize_batch(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    fallback = tokenizer.eos_token or tokenizer.pad_token or " "
    cleaned = [t.strip() or fallback for t in texts]
    enc = tokenizer(
        cleaned,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"].long(),
        "attention_mask": enc["attention_mask"].float(),
    }


def _paper_loss_mask(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Mask matching FLM's record_generative_perplexity: first EOS + every non-EOS."""
    is_eos = input_ids == eos_token_id
    first_eos_range = is_eos.cumsum(dim=-1) == 1
    return first_eos_range | ~is_eos


def text_causal_lm_forward(model: torch.nn.Module, eos_token_id: int) -> ForwardFn:
    """Forward that takes ``{input_ids, attention_mask}`` and applies the paper's mask."""
    def forward_fn(_model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        logits = _model(input_ids=input_ids, attention_mask=attention_mask).logits
        shifted = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss_mask = _paper_loss_mask(input_ids, eos_token_id)[:, 1:].float()
        log_probs = torch.log_softmax(shifted, dim=-1)
        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        denom = loss_mask.sum().clamp_min(1.0)
        nll = (token_nll * loss_mask).sum() / denom
        with torch.no_grad():
            probs = log_probs.exp()
            per_pos_entropy = -(probs * log_probs).sum(-1)
            entropy = (per_pos_entropy * loss_mask).sum() / denom
        return nll, entropy
    return forward_fn


def load_gpt2_large_text_reference(
    *,
    device: str | torch.device | None = None,
    allow_cpu: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[Any, torch.nn.Module, ForwardFn, int]:
    """gpt2-large + the gpt2 tokenizer wired to ``text_causal_lm_forward``."""
    tokenizer = _load_gpt2_tokenizer()
    model = _load_gpt2_large(device=device, allow_cpu=allow_cpu, dtype=dtype)
    eos_id = int(tokenizer.eos_token_id)
    return tokenizer, model, text_causal_lm_forward(model, eos_id), eos_id


def _row_unigram_entropy(ids: np.ndarray, mask: np.ndarray) -> float:
    values = ids[mask > 0]
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-(probs * np.log(probs)).sum())


class _EntropyState(NamedTuple):
    samples: RunningStats
    data: RunningStats


def _make_masked_sample_entropy_metric(*, name: str = "sample_entropy") -> Metric:
    """Per-row unigram entropy with the HF attention_mask applied (skips padding)."""
    def map_fn(batch_samples, batch_data):
        s_ids = batch_samples["input_ids"].cpu().numpy()
        s_mask = batch_samples["attention_mask"].cpu().numpy()
        d_ids = batch_data["input_ids"].cpu().numpy()
        d_mask = batch_data["attention_mask"].cpu().numpy()
        return (
            np.asarray([_row_unigram_entropy(r, m) for r, m in zip(s_ids, s_mask)]),
            np.asarray([_row_unigram_entropy(r, m) for r, m in zip(d_ids, d_mask)]),
        )

    def reduce_fn(state: _EntropyState, mapped) -> _EntropyState:
        rs, rd = mapped
        return _EntropyState(
            samples=update_stats_many(state.samples, rs),
            data=update_stats_many(state.data, rd),
        )

    def finalize(state: _EntropyState):
        return {
            "samples": running_summary(state.samples),
            "data": running_summary(state.data),
        }

    return Metric(
        name=name,
        init=lambda: _EntropyState(init_stats(), init_stats()),
        map_fn=map_fn,
        reduce_fn=reduce_fn,
        finalize=finalize,
    )


def _wrap_text_metric(metric: Metric, retok_pair) -> Metric:
    """Return a Metric whose ``map_fn`` accepts ``(list[str], list[str])`` and re-tokenizes."""
    def map_fn(samples_texts, data_texts):
        bs, bd = retok_pair(samples_texts, data_texts)
        return metric.map_fn(bs, bd)
    return Metric(
        name=metric.name,
        init=metric.init,
        map_fn=map_fn,
        reduce_fn=metric.reduce_fn,
        finalize=metric.finalize,
    )


def make_gpt2_large_text_metrics(
    *,
    max_length: int = 1024,
    include_entropy: bool = True,
    device: str | torch.device | None = None,
    allow_cpu: bool = False,
) -> list[Metric]:
    """Metrics whose ``map_fn`` accepts ``(list[str], list[str])`` per batch.

    Tokenizes with gpt2 (``padding="max_length"``, ``truncation=True``, ``max_length``),
    scores with gpt2-large under the FLM-paper loss mask (first EOS + non-EOS).
    """
    tokenizer, model, forward_fn, _ = load_gpt2_large_text_reference(
        device=device, allow_cpu=allow_cpu,
    )
    d = next(model.parameters()).device

    def retok_pair(samples_texts, data_texts):
        bs = _tokenize_batch(tokenizer, list(samples_texts), max_length)
        bd = _tokenize_batch(tokenizer, list(data_texts), max_length)
        return _move_to_device(bs, d), _move_to_device(bd, d)

    metrics: list[Metric] = [
        _wrap_text_metric(make_generative_llm_metrics(forward_fn, model), retok_pair),
    ]
    if include_entropy:
        metrics.append(_wrap_text_metric(_make_masked_sample_entropy_metric(), retok_pair))
    return metrics


def gpt2_large_evaluate_texts(
    text_batches: Iterable[tuple[list[str], list[str]]],
    *,
    max_length: int = 1024,
    include_entropy: bool = True,
    device: str | torch.device | None = None,
    allow_cpu: bool = False,
) -> dict[str, Mapping[str, Any]]:
    """Top-level text-input wrapper. Each batch is ``(list[str], list[str])``."""
    metrics = make_gpt2_large_text_metrics(
        max_length=max_length,
        include_entropy=include_entropy,
        device=device,
        allow_cpu=allow_cpu,
    )
    return evaluate(metrics, text_batches)


__all__ = [
    "GPT2_LARGE_MODEL_ID",
    "GPT2_TOKENIZER_ID",
    "causal_lm_forward",
    "gpt2_large_evaluate",
    "gpt2_large_evaluate_texts",
    "load_gpt2_large_reference",
    "load_gpt2_large_text_reference",
    "make_gpt2_large_text_metrics",
    "text_causal_lm_forward",
]
