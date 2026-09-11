"""Regression tests for sentence-level MMD over draws from one state.

Each sequence is mean-pooled over the scored positions. One clean sentence
embedding is treated as the fixed conditional reference and K generated
sentence embeddings form the sampled side. K > 1 uses an unclamped U-statistic
over the fake draws; K = 1 uses the biased empirical fallback.

Run from the repository root:

    python scripts/test_sentence_draws.py
"""
import math
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmd_rbf
import mmd_reward


def _run(name, fn):
  fn()
  print(f'PASS  {name}')


def _stacked_scalar_features():
  """One active row whose pooled values are real=0, fake={1, 2}."""
  # The third position is deliberately enormous and unselected. If it leaks
  # into pooling, neither hand-computed expected value below can pass.
  clean = torch.tensor([[[-1.0], [1.0], [999.0]],
                        [[7.0], [8.0], [9.0]]])
  draw_1 = torch.tensor([[[0.0], [2.0], [-500.0]],
                         [[1.0], [2.0], [3.0]]])
  draw_2 = torch.tensor([[[1.0], [3.0], [700.0]],
                         [[4.0], [5.0], [6.0]]])
  selected = torch.tensor([[True, True, False],
                           [False, False, False]])
  return {0: torch.cat([clean, draw_1, draw_2], dim=0)}, selected


def test_sentence_pooling_and_linear_unbiased_value():
  features, selected = _stacked_scalar_features()
  reward, active, per_layer = mmd_reward.summed_multi_draw_reward(
    features, batch=2, draw_count=2, length=3, selected=selected,
    indices=[0], weights=[1.0], kernel='linear', level='sentence')

  # k(x,y)=xy here. x=0 and y={1,2}, so the ordered off-diagonal
  # fake term is (1*2 + 2*1)/2 = 2 and the cross term is zero.
  torch.testing.assert_close(reward, torch.tensor([-2.0, 0.0]))
  assert active.tolist() == [True, False]
  torch.testing.assert_close(per_layer[0], reward)


def test_linear_is_mse_with_the_unbiased_variance_correction():
  real = torch.tensor([[0.0, 0.0]])
  one_fake = torch.tensor([[[1.0, 1.0]]])
  # K=1 biased fallback: mean squared error across hidden dimensions.
  torch.testing.assert_close(
    mmd_reward.linear_mmd2_fixed_real(real, one_fake),
    torch.tensor([1.0]))

  two_fakes = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
  mean_mse = torch.tensor([2.25])
  correction = torch.tensor([0.25])  # sample variance 0.5, divided by K=2
  torch.testing.assert_close(
    mmd_reward.linear_mmd2_fixed_real(real, two_fakes),
    mean_mse - correction)


def test_linear_unbiased_estimate_is_not_clamped():
  real = torch.tensor([[0.0]])
  symmetric_fakes = torch.tensor([[[-1.0], [1.0]]])
  # The means match, but the unbiased fake-pair term is -1. A U-statistic may
  # be negative; clamping it would silently turn it back into a biased one.
  torch.testing.assert_close(
    mmd_reward.linear_mmd2_fixed_real(real, symmetric_fakes),
    torch.tensor([-1.0]))


def _dense_fixed_real_rbf(real, fake, alpha):
  kff = torch.exp(-alpha * torch.cdist(fake, fake).square())
  krf = torch.exp(
    -alpha * (fake - real.unsqueeze(1)).square().sum(dim=-1))
  draws = fake.shape[1]
  if draws == 1:
    fake_term = kff[:, 0, 0]
  else:
    diagonal = kff.diagonal(dim1=1, dim2=2).sum(dim=1)
    fake_term = (kff.sum(dim=(1, 2)) - diagonal) / (
      draws * (draws - 1))
  return 1.0 + fake_term - 2.0 * krf.mean(dim=1)


def test_rbf_matches_dense_unbiased_and_biased_formulas():
  alpha = math.log(2.0)
  real = torch.tensor([[0.0]])

  one_fake = torch.tensor([[[1.0]]])
  actual_k1 = mmd_rbf.rbf_mmd2_fixed_real(real, one_fake, alpha)
  torch.testing.assert_close(actual_k1, torch.tensor([1.0]))

  two_fakes = torch.tensor([[[1.0], [2.0]]])
  actual_k2 = mmd_rbf.rbf_mmd2_fixed_real(
    real, two_fakes, alpha, block_size=1)
  expected_k2 = _dense_fixed_real_rbf(real, two_fakes, alpha)
  torch.testing.assert_close(actual_k2, expected_k2, rtol=1e-6, atol=1e-6)
  torch.testing.assert_close(actual_k2, torch.tensor([0.9375]))


def test_rbf_unbiased_estimate_is_not_clamped():
  alpha = math.log(2.0)
  real = torch.tensor([[1.5]])
  fake = torch.tensor([[[1.0], [2.0]]])
  actual = mmd_rbf.rbf_mmd2_fixed_real(real, fake, alpha)
  expected = _dense_fixed_real_rbf(real, fake, alpha)
  torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
  assert float(actual) < 0.0, actual


def test_sentence_rewards_keep_negative_u_statistics():
  # Both kernels have a negative U-statistic for this symmetric pair. The
  # reward is its negation and must therefore be positive, not clamped to 0.
  features = {0: torch.tensor([[[1.5]], [[1.0]], [[2.0]]])}
  selected = torch.tensor([[True]])
  linear, _, _ = mmd_reward.summed_multi_draw_reward(
    features, batch=1, draw_count=2, length=1, selected=selected,
    indices=[0], weights=[1.0], kernel='linear', level='sentence')
  rbf, _, _ = mmd_reward.summed_multi_draw_reward(
    features, batch=1, draw_count=2, length=1, selected=selected,
    indices=[0], weights=[1.0], kernel='rbf', alpha=math.log(2.0),
    level='sentence')
  torch.testing.assert_close(linear, torch.tensor([0.25]))
  assert float(rbf) > 0.0, rbf


def test_sentence_pooling_and_rbf_reward():
  features, selected = _stacked_scalar_features()
  reward, active, _ = mmd_reward.summed_multi_draw_reward(
    features, batch=2, draw_count=2, length=3, selected=selected,
    indices=[0], weights=[1.0], kernel='rbf', alpha=math.log(2.0),
    block_size=1, level='sentence')
  torch.testing.assert_close(reward, torch.tensor([-0.9375, 0.0]))
  assert active.tolist() == [True, False]


def test_auto_bandwidth_uses_only_pooled_real_sentences():
  features, selected = _stacked_scalar_features()
  seen = {}

  def inspect_alpha(real, real_selected, *args, **kwargs):
    del args, kwargs
    seen['real'] = real.clone()
    seen['selected'] = real_selected.clone()
    return 0.25

  with mock.patch.object(mmd_reward, 'median_alpha', inspect_alpha):
    mmd_reward.summed_multi_draw_reward(
      features, batch=2, draw_count=2, length=3, selected=selected,
      indices=[0], weights=[1.0], kernel='rbf', alpha=None,
      level='sentence')
  assert seen['real'].shape == (2, 1, 1)
  torch.testing.assert_close(seen['real'][:, 0, 0], torch.tensor([0.0, 0.0]))
  assert seen['selected'].tolist() == [[True], [False]]


def main():
  for name, fn in sorted(globals().items()):
    if name.startswith('test_'):
      _run(name, fn)
  print('\nall sentence-draw MMD tests passed')


if __name__ == '__main__':
  main()
