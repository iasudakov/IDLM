"""CPU-only checks for the optional high-mask distillation mixture.

Run with:

    python scripts/test_high_mask_schedule.py
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import algo  # noqa: E402
import trainer_base  # noqa: E402


class ScheduleStub:
  """Only the state MDLM._sample_distill_generator_t needs."""

  def __init__(self, values, **overrides):
    config = {
      'high_mask_fraction': 0.0,
      'high_mask_min': 0.7,
      'high_mask_max': 1.0,
    }
    config.update(overrides)
    self.config = SimpleNamespace(adversarial_distill=config)
    self.noise = trainer_base.LogLinear()
    self.sampling_eps = 1e-3
    self.values = values

  def _sample_t(self, n, current_accumulation_step):
    del current_accumulation_step
    assert n == self.values.shape[0]
    return self.values.clone()

def sample(stub):
  return algo.MDLM._sample_distill_generator_t(
    stub, stub.values.shape[0], current_accumulation_step=0)


def test_disabled_is_exact_and_consumes_no_rng():
  base = torch.linspace(0.01, 0.99, 32)
  stub = ScheduleStub(base)
  torch.manual_seed(123)
  before = torch.random.get_rng_state()
  actual = sample(stub)
  after = torch.random.get_rng_state()
  assert torch.equal(actual, base)
  assert torch.equal(after, before)


def test_full_mixture_respects_mask_probability_interval():
  stub = ScheduleStub(
    torch.full((4096,), 0.1),
    high_mask_fraction=1.0,
    high_mask_min=0.7,
    high_mask_max=0.9)
  torch.manual_seed(7)
  actual = sample(stub)
  mask_probability = (1.0 - stub.noise.eps) * actual
  assert float(mask_probability.min()) >= 0.7 - 1e-6
  assert float(mask_probability.max()) <= 0.9 + 1e-6


def test_partial_mixture_preserves_requested_share():
  stub = ScheduleStub(
    torch.full((20000,), 0.1),
    high_mask_fraction=0.5,
    high_mask_min=0.7,
    high_mask_max=1.0)
  torch.manual_seed(11)
  actual = sample(stub)
  # Original rows have t=0.1; high-mask rows have t>=0.7/(1-eps), so
  # the two components are separated cleanly for this check.
  realized = (actual > 0.5).float().mean()
  assert abs(float(realized) - 0.5) < 0.02


def test_invalid_settings_fail_loudly():
  invalid = [
    {'high_mask_fraction': -0.1},
    {'high_mask_fraction': 1.1},
    {'high_mask_min': 0.9, 'high_mask_max': 0.8},
    {'high_mask_min': -0.1},
    {'high_mask_max': 1.1},
  ]
  for overrides in invalid:
    try:
      sample(ScheduleStub(torch.full((2,), 0.5), **overrides))
    except ValueError:
      pass
    else:
      raise AssertionError(f'invalid settings were accepted: {overrides}')


if __name__ == '__main__':
  test_disabled_is_exact_and_consumes_no_rng()
  test_full_mixture_respects_mask_probability_interval()
  test_partial_mixture_preserves_requested_share()
  test_invalid_settings_fail_loudly()
  print('4/4 passed')
