"""Confidence-ranked commit rules for the masked-diffusion sampler.

Upstream IDLM unmasks by an independent Bernoulli draw per masked position: which
position gets written is a coin flip, not a function of how sure the model is about
it. These are the LLaDA / MaskGIT commit rules, ported from yrOPDLM
(`sample/bd3lm_rl_rollout.py::_diffusion_step_block` and the deployed per-sequence
version in `sample/jetengine_ext/engine/scheduler.py::Scheduler.postprocess`), which
instead rank the still-masked positions by the model's confidence in the token it
would write there and commit the confident ones first.

The two rules budget compute differently, and that difference drives the sampler:

  low_confidence_static   a FIXED schedule -- `k` per step comes from the noise
                          schedule, exactly the expected commit count of the ancestral
                          update it replaces, so `sampling.steps` keeps its meaning
                          and only *which* positions fill each quota changes.
  low_confidence_dynamic  ADAPTIVE -- the threshold is the budget and the step count
                          is an outcome. yrOPDLM sweeps `dynamic_threshold` over
                          0.85/0.9/0.95 and sets its step cap to the one-commit-per-
                          step worst case, then measures the realized forwards
                          (`train/online_eval.py::_budget_grid`).

Everything here is pure tensor code with no model dependency, and batched: yrOPDLM
loops over rows because it decodes a handful at a time, but IDLM decodes 32-64 rows
per call, so the row-varying top-k is vectorised instead.

Callers must pass `confidence` with a value of `-inf` at every non-candidate
position. That is not merely an optimisation: under MDLM's SUBS parameterisation
(`algo.MDLM._process_model_output`) `p_x0` is one-hot at already-unmasked positions,
so their confidence is exactly 1.0 and they would otherwise win every top-k.
"""
import torch

STRATEGIES = ('random', 'low_confidence_static', 'low_confidence_dynamic')

# Strategies whose per-step commit count is decided by the threshold rather than by
# the noise schedule, and which therefore need the adaptive sampler loop.
ADAPTIVE_STRATEGIES = ('low_confidence_dynamic',)


def validate_strategy(strategy):
  """Raise on an unknown `sampling.remasking_strategy`, else return it."""
  if strategy not in STRATEGIES:
    raise ValueError(
      f'sampling.remasking_strategy must be one of {STRATEGIES}, '
      f'got {strategy!r}')
  return strategy


def commit_counts(num_candidates, frac):
  """Per-row commit budget `k = round(n * frac)`, clamped to `[0, n]`.

  Args:
    num_candidates: int torch.Tensor (batch_size,), masked positions per row.
    frac: float torch.Tensor (batch_size,), the fraction of them to commit.

  This is yrOPDLM's `_get_num_transfer_tokens` (bd3lm_rl_rollout.py:153) written
  against IDLM's noise schedule. There `prob = 1 - s/t` on the linear grid; here the
  caller passes `(alpha_s - alpha_t) / (1 - alpha_t)`, which equals it for
  `alpha_t = 1 - t` and stays correct for any other schedule.
  """
  k = torch.round(num_candidates.to(torch.float64) * frac.to(torch.float64))
  return k.clamp(min=0).to(torch.int64).minimum(num_candidates.to(torch.int64))


def select_top_k(confidence, candidates, k):
  """The `k` highest-confidence candidate positions of each row.

  Args:
    confidence: float torch.Tensor (batch_size, length), `-inf` off `candidates`.
    candidates: bool torch.Tensor (batch_size, length), positions eligible to commit.
    k: int torch.Tensor (batch_size,), per-row commit budget.

  Returns:
    bool torch.Tensor (batch_size, length), a subset of `candidates`.

  Vectorised row-varying top-k: `torch.topk` needs one scalar `k` for the whole
  batch, so rank each row by descending confidence and keep the ranks below that
  row's budget. Ties break by position, deterministically -- `torch.topk` in the
  reference does the same.
  """
  ranks = _descending_ranks(confidence)
  return (ranks < k.unsqueeze(-1)) & candidates


def select_threshold(confidence, candidates, threshold):
  """Every candidate whose confidence clears `threshold`.

  Args:
    confidence: float torch.Tensor (batch_size, length), `-inf` off `candidates`.
    candidates: bool torch.Tensor (batch_size, length), positions eligible to commit.
    threshold: float.

  Returns:
    bool torch.Tensor (batch_size, length), a subset of `candidates` with at least
    one entry in every row that has a candidate at all.

  A row where nothing clears the threshold commits its single most confident
  position instead. That fallback is yrOPDLM's (scheduler.py:150), and it is what
  bounds the adaptive loop: every step retires at least one position per unfinished
  row, so decoding terminates in at most `length` steps.
  """
  transfer = (confidence > threshold) & candidates
  stalled = candidates.any(dim=-1) & ~transfer.any(dim=-1)
  if bool(stalled.any()):
    forced = torch.zeros_like(transfer)
    forced.scatter_(-1, confidence.argmax(dim=-1, keepdim=True), True)
    transfer = torch.where(stalled.unsqueeze(-1), forced, transfer)
  return transfer


def _descending_ranks(confidence):
  """(batch_size, length) int tensor: 0 for each row's most confident position."""
  order = confidence.argsort(dim=-1, descending=True)
  ranks = torch.empty_like(order)
  positions = torch.arange(
    confidence.shape[-1], device=confidence.device).expand_as(order)
  ranks.scatter_(-1, order, positions)
  return ranks
