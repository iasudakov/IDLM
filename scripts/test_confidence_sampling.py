"""Tests for the confidence-ranked commit rules (`confidence_sampling`).

CPU-only and checkpoint-free: the sampler's commit decision is a function of
`p_x0` alone, so a stub `forward` returning a fixed distribution exercises it
exactly. Run with

    python scripts/test_confidence_sampling.py

The properties under test, in order:
  1. the default (`random`) path is bit-identical to the pre-change body;
  2. `low_confidence_static` commits exactly `k` positions, and they are the `k`
     most confident *masked* ones (with a negative control);
  3. the static budget closes -- a full decode leaves no MASK behind;
  4. `low_confidence_dynamic` commits exactly the above-threshold set and forces
     one commit when nothing clears it;
  5. the adaptive decode terminates by its early break, and its realized NFE
     tracks the *threshold* while `sampling.steps` is held fixed -- the property
     the separate adaptive loop exists for;
  6. `sampling.temperature=0` decodes greedily while confidence stays the raw
     `p_x0` value.
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import confidence_sampling  # noqa: E402
import trainer_base  # noqa: E402

# Wide enough that a peak of 0.10 still wins: the leftover mass is spread over
# VOCAB - 2 tokens, so the peak is the argmax only while c > 1 / (VOCAB - 1).
VOCAB = 32
MASK = VOCAB - 1
LENGTH = 6


class StubSampler:
  """The real sampler methods over a stub `forward`.

  Everything that decides *which* positions commit is borrowed unbound from the
  production classes, so this tests the shipped code rather than a copy of it.
  Only `forward` (the transformer) and the Lightning plumbing are stubbed.
  """
  _project_prefix = staticmethod(trainer_base.TrainerBase._project_prefix)
  _sigma_from_alphat = trainer_base.Diffusion._sigma_from_alphat
  generate_samples = trainer_base.Diffusion.generate_samples
  _adaptive_generate_samples = trainer_base.Diffusion._adaptive_generate_samples
  _alpha_to_t = trainer_base.Diffusion._alpha_to_t
  prior_sample = trainer_base.AbsorbingState.prior_sample
  _ancestral_update = trainer_base.AbsorbingState._ancestral_update
  _ancestral_greedy_token = trainer_base.AbsorbingState._ancestral_greedy_token
  _propose_tokens = trainer_base.AbsorbingState._propose_tokens
  _confidence_commit = trainer_base.AbsorbingState._confidence_commit

  def __init__(self, p_x0, strategy='random', temperature=1.0,
               threshold=0.9, steps=8, sampler='ancestral_cache',
               noise_removal='ancestral', step_convention='sflm',
               max_steps=None):
    self.p_x0 = p_x0
    self.device = torch.device('cpu')
    self.noise = trainer_base.LogLinear()
    self.mask_index = MASK
    self.num_tokens = p_x0.shape[1]
    self.time_conditioning = False
    self.sampler = sampler
    self.remasking_strategy = confidence_sampling.validate_strategy(strategy)
    self.dynamic_threshold = threshold
    self.sampling_temperature = temperature
    self.forward_calls = 0
    self.config = OmegaConf.create({
      'sampling': {'steps': steps, 'noise_removal': noise_removal,
                   'step_convention': step_convention,
                   'max_steps': max_steps, 'use_float64': True}})

  def forward(self, xt, sigma=None):
    """Log-probabilities, independent of the latent.

    Holding the distribution fixed across steps is what makes the realized step
    count a clean function of the commit rule alone.
    """
    self.forward_calls += 1
    return self.p_x0.clamp_min(1e-30).log()


def _probs_from_confidences(confidences, batch_size=1):
  """(B, L, V) distributions whose argmax at position j has mass confidences[j].

  The peak token is `j % (VOCAB - 1)`, never MASK -- MDLM's SUBS
  parameterisation puts zero mass there, and a stub that did not would let a
  "commit" write a mask back and stall the adaptive loop forever.
  """
  length = len(confidences)
  probs = torch.zeros(length, VOCAB, dtype=torch.float32)
  for j, c in enumerate(confidences):
    peak = j % (VOCAB - 1)
    probs[j] = (1 - c) / (VOCAB - 2)
    probs[j, MASK] = 0.0
    probs[j, peak] = c
  probs = probs / probs.sum(dim=-1, keepdim=True)
  assert torch.allclose(probs.max(dim=-1).values,
                        torch.tensor(confidences, dtype=torch.float32),
                        atol=1e-6), 'peak is not the argmax; raise VOCAB'
  return probs.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def _run(name, fn):
  fn()
  print(f'PASS  {name}')


# --------------------------------------------------------------------------- #

def test_random_default_is_bit_identical():
  """`remasking_strategy=random` must reproduce the pre-change update exactly."""
  probs = _probs_from_confidences([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], batch_size=3)
  model = StubSampler(probs, strategy='random')
  x = torch.full((3, LENGTH), MASK, dtype=torch.int64)
  x[:, 0] = 2  # an already-decoded position, which must never be rewritten
  t = 0.7 * torch.ones(3, 1)
  dt = 0.1

  torch.manual_seed(0)
  _, got = model._ancestral_update(x=x.clone(), t=t, dt=dt, p_x0=None,
                                   noise_removal_step=False)

  # The upstream body, verbatim, as the reference.
  _, alpha_t = model.noise(t)
  _, alpha_s = model.noise(t - dt)
  # Round-trip through log/exp exactly as the sampler does, so any last-ulp
  # difference is the sampler's own and not an artefact of the reference.
  p_x0 = probs.clamp_min(1e-30).log().exp()
  torch.manual_seed(0)
  q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
  q_xs[:, :, MASK] = 1 - alpha_s
  _x = trainer_base.sample_categorical(q_xs)
  copy_flag = (x != MASK).to(x.dtype)
  want = copy_flag * x + (1 - copy_flag) * _x

  assert torch.equal(got, want), (got, want)


def test_static_commits_the_top_k_masked_positions():
  confidences = [0.99, 0.10, 0.95, 0.20, 0.90, 0.30]
  probs = _probs_from_confidences(confidences)
  model = StubSampler(probs, strategy='low_confidence_static', temperature=0.0)

  x = torch.full((1, LENGTH), MASK, dtype=torch.int64)
  x[0, 0] = 5  # decoded already: SUBS would give it confidence 1.0, yet it must
               # not be counted, ranked, or rewritten
  # frac = (alpha_s - alpha_t) / (1 - alpha_t) = 0.5 over 5 candidates -> k = 2.
  alpha_t = torch.tensor([[0.5]])
  alpha_s = torch.tensor([[0.75]])
  out = model._confidence_commit(x=x, p_x0=probs, alpha_s=alpha_s,
                                 alpha_t=alpha_t, noise_removal_step=False)

  committed = (out != x).nonzero()[:, 1].tolist()
  assert committed == [2, 4], committed  # the two most confident *candidates*
  assert out[0, 0].item() == 5


def test_static_negative_control():
  """Reversing the ranking must move the commit set."""
  forward = _probs_from_confidences([0.99, 0.10, 0.95, 0.20, 0.90, 0.30])
  reverse = _probs_from_confidences([0.30, 0.90, 0.20, 0.95, 0.10, 0.99])
  x = torch.full((1, LENGTH), MASK, dtype=torch.int64)
  alpha_t, alpha_s = torch.tensor([[0.5]]), torch.tensor([[0.75]])

  sets = []
  for probs in (forward, reverse):
    model = StubSampler(probs, strategy='low_confidence_static',
                        temperature=0.0)
    out = model._confidence_commit(x=x, p_x0=probs, alpha_s=alpha_s,
                                   alpha_t=alpha_t, noise_removal_step=False)
    sets.append(set((out != x).nonzero()[:, 1].tolist()))

  assert len(sets[0]) == len(sets[1]) == 3, sets  # k = round(6 * 0.5)
  assert sets[0] != sets[1], sets


def test_static_budget_closes():
  """A full fixed-budget decode must leave no MASK token behind."""
  probs = _probs_from_confidences([0.99, 0.10, 0.95, 0.20, 0.90, 0.30],
                                  batch_size=4)
  for convention in ('sflm', 'idlm'):
    model = StubSampler(probs, strategy='low_confidence_static',
                        temperature=0.0, steps=8, step_convention=convention)
    x, meta = model.generate_samples(num_samples=4, return_metadata=True)
    assert not bool((x == MASK).any()), (convention, x)
    assert meta['nfe'] >= 1


def test_dynamic_commits_exactly_the_above_threshold_set():
  confidences = [0.99, 0.10, 0.95, 0.20, 0.90, 0.30]
  probs = _probs_from_confidences(confidences)
  model = StubSampler(probs, strategy='low_confidence_dynamic',
                      temperature=0.0, threshold=0.92)
  x = torch.full((1, LENGTH), MASK, dtype=torch.int64)
  out = model._confidence_commit(x=x, p_x0=probs, alpha_s=None,
                                 alpha_t=torch.tensor([[0.5]]),
                                 noise_removal_step=False)
  committed = (out != x).nonzero()[:, 1].tolist()
  assert committed == [0, 2], committed


def test_dynamic_forces_one_commit_when_nothing_clears():
  """The fallback that bounds the loop: never zero commits on a live row."""
  confidences = [0.30, 0.10, 0.55, 0.20, 0.40, 0.15]
  probs = _probs_from_confidences(confidences)
  model = StubSampler(probs, strategy='low_confidence_dynamic',
                      temperature=0.0, threshold=0.99)
  x = torch.full((1, LENGTH), MASK, dtype=torch.int64)
  out = model._confidence_commit(x=x, p_x0=probs, alpha_s=None,
                                 alpha_t=torch.tensor([[0.5]]),
                                 noise_removal_step=False)
  committed = (out != x).nonzero()[:, 1].tolist()
  assert committed == [2], committed  # the single most confident position


def test_dynamic_ignores_steps_and_reports_realized_nfe():
  """The point of the separate adaptive loop.

  `sampling.steps` is pinned at an absurd 3 throughout; the realized forward
  count must be set by the threshold instead, and must fall as the threshold
  loosens. Termination has to come from the early break, not the cap.
  """
  confidences = [0.99, 0.10, 0.95, 0.20, 0.90, 0.30]
  probs = _probs_from_confidences(confidences, batch_size=2)

  nfes = {}
  for threshold in (0.92, 0.85, 0.05):
    model = StubSampler(probs, strategy='low_confidence_dynamic',
                        temperature=0.0, threshold=threshold, steps=3)
    x, meta = model.generate_samples(num_samples=2, num_steps=3,
                                     return_metadata=True)
    assert not bool((x == MASK).any()), (threshold, x)
    assert meta['steps'] < model.num_tokens, meta  # exited by the early break
    nfes[threshold] = meta['nfe']

  # 0.92 clears 2 positions at once then crawls one at a time -> 5 forwards;
  # 0.85 clears 3 -> 4; 0.05 clears all 6 -> 1.
  assert nfes == {0.92: 5, 0.85: 4, 0.05: 1}, nfes
  assert nfes[0.92] > nfes[0.85] > nfes[0.05], nfes
  # ...and none of them is `sampling.steps`, which was 3 for every run.
  assert 3 not in set(nfes.values()), nfes


def test_ancestral_greedy_token_writes_only_argmax():
  """Greedy-token ancestral: schedule unchanged, token choice no longer random.

  Every position that gets unmasked must receive the argmax of `p_x0`, and no
  already-decoded position may be rewritten.
  """
  confidences = [0.99, 0.30, 0.95, 0.20, 0.90, 0.40]
  probs = _probs_from_confidences(confidences, batch_size=8)
  model = StubSampler(probs, strategy='random', temperature=0.0)
  x = torch.full((8, LENGTH), MASK, dtype=torch.int64)
  x[:, 0] = 5  # already decoded

  torch.manual_seed(0)
  _, out = model._ancestral_update(x=x.clone(), t=0.6 * torch.ones(8, 1),
                                   dt=0.2, p_x0=None,
                                   noise_removal_step=False)

  argmax = probs.argmax(dim=-1)
  changed = out != x
  assert bool((out[changed] == argmax[changed]).all()), 'wrote a non-argmax token'
  assert not bool(changed[:, 0].any()), 'rewrote a decoded position'
  assert bool(changed.any()), 'nothing was unmasked at all'


def test_ancestral_greedy_token_keeps_the_schedule():
  """The unmask *rate* must still be the schedule's, only the token differs.

  (alpha_s - alpha_t) / (1 - alpha_t) with the LogLinear schedule at t=0.6,
  dt=0.3 is 0.5, so about half the masked positions should commit -- checked
  loosely over many rows, since it is a Bernoulli draw.
  """
  probs = _probs_from_confidences([0.99, 0.30, 0.95, 0.20, 0.90, 0.40],
                                  batch_size=512)
  model = StubSampler(probs, strategy='random', temperature=0.0)
  x = torch.full((512, LENGTH), MASK, dtype=torch.int64)
  torch.manual_seed(0)
  _, out = model._ancestral_update(x=x.clone(), t=0.6 * torch.ones(512, 1),
                                   dt=0.3, p_x0=None,
                                   noise_removal_step=False)
  rate = (out != x).float().mean().item()
  assert 0.45 < rate < 0.55, rate

  # And at the noise-removal step every mask must go, greedy or not.
  _, out = model._ancestral_update(x=x.clone(), t=0.6 * torch.ones(512, 1),
                                   dt=0.3, p_x0=None,
                                   noise_removal_step=True)
  assert not bool((out == MASK).any())


def test_greedy_token_is_deterministic_given_the_unmask_draw():
  """Two runs from the same seed agree; the randomness is only in *where*."""
  probs = _probs_from_confidences([0.99, 0.30, 0.95, 0.20, 0.90, 0.40],
                                  batch_size=4)
  outs = []
  for _ in range(2):
    model = StubSampler(probs, strategy='random', temperature=0.0)
    x = torch.full((4, LENGTH), MASK, dtype=torch.int64)
    torch.manual_seed(7)
    _, out = model._ancestral_update(x=x, t=0.6 * torch.ones(4, 1), dt=0.2,
                                     p_x0=None, noise_removal_step=False)
    outs.append(out)
  assert torch.equal(outs[0], outs[1])


def test_per_row_nfe_tracks_when_each_row_finished():
  """The Pareto x-axis: cost attributed per sequence, not per batch.

  Two rows with different amounts of prompt pinned finish at different steps.
  The batch runs until the slowest one is done, so `nfe` is their max -- which
  is exactly why it is the wrong number to average over a dataset.
  """
  probs = _probs_from_confidences([0.99, 0.10, 0.95, 0.20, 0.90, 0.30],
                                  batch_size=2)
  model = StubSampler(probs, strategy='low_confidence_dynamic',
                      temperature=0.0, threshold=0.92)
  # Row 0 has all but its last position pinned; row 1 has none pinned.
  prefix = torch.ones(2, 5, dtype=torch.int64)
  lengths = torch.tensor([5, 0])
  x, meta = model.generate_samples(num_samples=2, return_metadata=True,
                                   prefix_tokens=prefix,
                                   prefix_lengths=lengths)

  per_row = meta['nfe_per_row'].tolist()
  # Row 0: one forced-argmax commit. Row 1: commits 0.99/0.95 together, then
  # crawls through the remaining four one per step.
  assert per_row == [1.0, 5.0], per_row
  assert meta['nfe'] == max(per_row), (meta['nfe'], per_row)
  assert not bool((x == MASK).any())


def test_fixed_grid_reports_uniform_per_row_nfe():
  """A fixed schedule spends the same on every row, so per-row cost is `nfe`."""
  probs = _probs_from_confidences([0.99, 0.10, 0.95, 0.20, 0.90, 0.30],
                                  batch_size=3)
  model = StubSampler(probs, strategy='random', steps=5,
                      step_convention='sflm')
  _, meta = model.generate_samples(num_samples=3, return_metadata=True)
  per_row = meta['nfe_per_row'].tolist()
  assert per_row == [float(meta['nfe'])] * 3, (per_row, meta['nfe'])


def test_pareto_render_returns_an_image():
  import pareto_plot
  history = {
    2000: [{'threshold': 0.5, 'nfe': 6.0, 'accuracy': 0.03},
           {'threshold': 0.9, 'nfe': 48.0, 'accuracy': 0.21}],
    4000: [{'threshold': 0.5, 'nfe': 6.4, 'accuracy': 0.05},
           {'threshold': 0.9, 'nfe': 47.0, 'accuracy': 0.28}],
  }
  reference = [{'threshold': 0.5, 'nfe': 6.1, 'accuracy': 0.02},
               {'threshold': 0.9, 'nfe': 49.0, 'accuracy': 0.16}]
  for ref in (None, reference):
    image = pareto_plot.render(history, reference=ref)
    assert image.ndim == 3 and image.shape[2] == 3, image.shape
    assert image.dtype == np.uint8, image.dtype
    assert image.max() > 0, 'rendered a blank figure'
  # A single checkpoint must not divide by a zero step range.
  pareto_plot.render({0: history[2000]})
  # A configured-but-missing reference file is garnish, not an error.
  assert pareto_plot.load_reference('/nonexistent/reference.json') is None
  assert pareto_plot.load_reference(None) is None


def test_temperature_zero_is_greedy_with_raw_confidence():
  confidences = [0.60, 0.55, 0.51, 0.52, 0.53, 0.54]
  probs = _probs_from_confidences(confidences)
  model = StubSampler(probs, strategy='low_confidence_static', temperature=0.0)
  x0, confidence = model._propose_tokens(probs)

  assert torch.equal(x0, probs.argmax(dim=-1))
  # The confidence is the raw probability, not the 1.0 a tempered distribution
  # would report -- otherwise every position ties and the ranking is vacuous.
  expected = probs.max(dim=-1).values
  assert torch.allclose(confidence, expected, atol=1e-6), (confidence, expected)
  assert float(confidence.max()) < 0.99


def test_rejects_bad_configuration():
  try:
    confidence_sampling.validate_strategy('entropy_bound')
  except ValueError:
    pass
  else:
    raise AssertionError('unknown strategy accepted')


def main():
  torch.manual_seed(0)
  for name, fn in sorted(globals().items()):
    if name.startswith('test_'):
      _run(name, fn)
  print('\nall confidence-sampling tests passed')


if __name__ == '__main__':
  main()
