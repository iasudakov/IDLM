"""CPU checks for the sentence-draw pseudo-Huber distance-induced MMD kernel.

Run from the repository root: python scripts/test_pseudo_huber_mmd.py
"""
import itertools
import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmd_reward


def rho(left, right):
  """Independent reference, including the historical coordinate normalization."""
  delta = left.double() - right.double()
  return (torch.sqrt(delta.square() + 1e-6) - 0.001).mean(dim=-1)


def dense_estimate(real, fake):
  cross = rho(real[:, None], fake).mean(dim=1)
  count = fake.shape[1]
  if count == 1:
    return cross
  pairs = rho(fake[:, :, None], fake[:, None, :])
  return cross - pairs.sum(dim=(1, 2)) / (2 * count * (count - 1))


def expect_value_error(fn, fragment):
  try:
    fn()
  except ValueError as error:
    assert fragment in str(error), error
  else:
    raise AssertionError('expected ValueError')


def test_k1_exactly_matches_legacy_and_keeps_fp32():
  for dtype in (torch.float32, torch.bfloat16):
    real = torch.tensor([[1e4, 0.001, -3.0], [1.0, 2.0, 3.0]], dtype=dtype)
    fake = torch.tensor([[[-1e4, 0.0, 5.0]], [[1.0, 2.0, 3.0]]], dtype=dtype)
    real.requires_grad_(True)
    fake.requires_grad_(True)
    actual = mmd_reward.pseudo_huber_mmd2_fixed_real(real, fake)
    legacy = -mmd_reward.pseudo_huber_reward(fake[:, 0].float() - real.float())
    torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
    assert actual.dtype == torch.float32 and not actual.requires_grad


def test_k_greater_than_one_matches_dense_reference():
  generator = torch.Generator().manual_seed(71)
  real = torch.randn(4, 7, generator=generator)
  for count in (2, 3, 4):
    fake = torch.randn(4, count, 7, generator=generator)
    actual = mmd_reward.pseudo_huber_mmd2_fixed_real(real, fake)
    torch.testing.assert_close(actual.double(), dense_estimate(real, fake),
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual,
      mmd_reward.pseudo_huber_mmd2_fixed_real(real, fake.flip(1)),
      atol=1e-6, rtol=1e-6)


def test_exact_enumeration_is_unbiased_conditional_on_real():
  support = torch.tensor([[-2.0, 0.5], [0.0, 1.0], [3.0, -1.0]])
  probability = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
  real = torch.tensor([[0.7, -0.2]])
  expected = (probability * rho(real, support)).sum()
  expected -= 0.5 * (probability[:, None] * probability[None, :]
                    * rho(support[:, None], support[None, :])).sum()
  for count in (2, 3):
    indices = torch.tensor(list(itertools.product(range(3), repeat=count)))
    weights = probability[indices].prod(dim=1)
    estimates = mmd_reward.pseudo_huber_mmd2_fixed_real(
      real.expand(indices.shape[0], -1), support[indices])
    torch.testing.assert_close((weights * estimates.double()).sum(), expected,
                               atol=1e-6, rtol=1e-6)


def test_negative_estimate_is_not_clamped():
  actual = mmd_reward.pseudo_huber_mmd2_fixed_real(
    torch.zeros(1, 1), torch.tensor([[[-0.001], [0.001]]]))
  assert actual.item() < 0, actual


def test_pooling_inactive_rows_and_multiple_layers():
  clean = torch.tensor([[[-1.0], [1.0], [1e5]], [[9.0], [8.0], [7.0]]])
  fake = torch.tensor([[[[0.0], [2.0], [-1e5]], [[1.0], [2.0], [3.0]]],
                       [[[1.0], [3.0], [2e5]], [[4.0], [5.0], [6.0]]]])
  selected = torch.tensor([[True, True, False], [False, False, False]])
  stacked = torch.cat([clean, fake.reshape(4, 3, 1)], dim=0)
  features = {0: stacked, 1: 2 * stacked}
  actual, active, layers = mmd_reward.summed_multi_draw_reward(
    features, batch=2, draw_count=2, length=3, selected=selected,
    indices=[0, 1], weights=[0.25, 2.0], kernel='pseudo_huber', level='sentence')
  assert active.tolist() == [True, False]
  for index, factor in ((0, 1.0), (1, 2.0)):
    expected = -dense_estimate(torch.tensor([[0.0]]),
                              factor * torch.tensor([[[1.0], [2.0]]]))
    torch.testing.assert_close(layers[index].double(),
                               torch.tensor([expected.item(), 0.0], dtype=torch.float64),
                               atol=1e-6, rtol=1e-6)
  torch.testing.assert_close(actual, 0.25 * layers[0] + 2 * layers[1])
  legacy, legacy_active = mmd_reward.paired_sequence_reward(
    clean, fake[0], selected, kernel='linear')
  one, one_active, _ = mmd_reward.summed_multi_draw_reward(
    {0: torch.cat([clean, fake[0]])}, batch=2, draw_count=1, length=3,
    selected=selected, indices=[0], weights=[1.0],
    kernel='pseudo_huber', level='sentence')
  torch.testing.assert_close(one, legacy, atol=0, rtol=0)
  assert torch.equal(one_active, legacy_active)


class ToyBackbone(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.blocks = torch.nn.ModuleList([torch.nn.Identity()])

  def forward(self, tokens, sigma):
    assert sigma.shape[0] == tokens.shape[0]
    return self.blocks[0](tokens.float().unsqueeze(-1))


def critic():
  result = mmd_reward.FrozenBackboneFeatures(
    ToyBackbone(), mask_index=9, layer_index=0, noise_level=0.0)
  result.kernel = 'pseudo_huber'
  result.kernel_level = 'sentence'
  return result


def test_production_backbone_wrapper():
  clean = torch.tensor([[0, 0, 7], [3, 3, 8]])
  fake = torch.tensor([[[1, 1, 5], [6, 6, 9]], [[2, 2, 8], [1, 1, 3]]])
  selected = torch.tensor([[True, True, False], [False, False, False]])
  actual, active, layers = critic().multi_draw_reward(
    clean, fake, selected, eos_id=None, sigma=torch.zeros(2, 1),
    return_per_layer=True)
  expected = -dense_estimate(torch.tensor([[0.0]]),
                            torch.tensor([[[1.0], [2.0]]]))
  torch.testing.assert_close(actual.double(),
    torch.tensor([expected.item(), 0.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)
  assert active.tolist() == [True, False]
  torch.testing.assert_close(layers[0], actual)


def test_reinforce_gradient_preserves_tinygsm_scaling():
  batch, length, vocab, group, count = 2, 3, 3, 3, 2
  chosen = torch.tensor([[0, 1], [1, 2], [2, 0]])
  selected = torch.tensor([[True, False, True], [True, False, True]])
  probability = torch.full((batch, length, vocab), 1 / vocab, requires_grad=True)

  def sample(flat, num_samples, replacement):
    assert num_samples == group * count and replacement
    return chosen.flatten().expand(flat.shape[0], -1)

  with mock.patch.object(torch, 'multinomial', sample):
    loss, _ = mmd_reward.mmd_reinforce_loss(
      probability, torch.zeros(batch, length, dtype=torch.long), selected,
      critic(), group_size=group, eos_id=None, kernel_source='draws',
      kernel_draws=count, baseline='group')
  loss.backward()
  rewards = -dense_estimate(torch.zeros(group, 1), chosen[:, :, None])
  advantages = rewards - (rewards.sum() - rewards) / (group - 1)
  expected = torch.zeros_like(probability)
  for candidate in range(group):
    for token in chosen[candidate]:
      expected[..., token] -= (
        advantages[candidate].float() * vocab / (selected.sum() * count)) * selected
  torch.testing.assert_close(probability.grad, expected, atol=1e-6, rtol=1e-6)
  assert torch.isfinite(loss)
  assert probability.grad[selected].abs().sum() > 0


def test_k1_reinforce_matches_legacy_loss_and_gradient():
  outputs = []
  for kernel, level, source in (('linear', 'token', 'rows'),
                                 ('pseudo_huber', 'sentence', 'draws')):
    feature_critic = critic()
    feature_critic.kernel = kernel
    feature_critic.kernel_level = level
    probabilities = torch.full((2, 4, 3), 1 / 3, requires_grad=True)
    torch.manual_seed(92)
    loss, stats = mmd_reward.mmd_reinforce_loss(
      probabilities, torch.zeros(2, 4, dtype=torch.long),
      torch.tensor([[True, False, True, True], [False, True, False, True]]),
      feature_critic, group_size=3, eos_id=None, kernel_source=source,
      kernel_draws=1, baseline='group')
    loss.backward()
    outputs.append((loss, probabilities.grad, stats))
  for old, new in zip(outputs[0], outputs[1]):
    torch.testing.assert_close(old, new, atol=0, rtol=0)


def test_invalid_shapes_and_unsupported_combinations():
  estimate = mmd_reward.pseudo_huber_mmd2_fixed_real
  expect_value_error(lambda: estimate(torch.zeros(1), torch.zeros(1, 1, 1)),
                     'expects')
  expect_value_error(lambda: estimate(torch.zeros(2, 1), torch.zeros(1, 1, 1)),
                     'share batch')
  expect_value_error(lambda: estimate(torch.zeros(1, 2), torch.zeros(1, 1, 1)),
                     'share batch')
  expect_value_error(lambda: estimate(torch.zeros(1, 1), torch.zeros(1, 0, 1)),
                     'at least one')
  for level in ('token', 'sentence'):
    expect_value_error(lambda: mmd_reward.paired_sequence_reward(
      torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.ones(1, 1).bool(),
      kernel='pseudo_huber', level=level), 'mmd_kernel_source=draws')
  expect_value_error(lambda: mmd_reward.summed_multi_draw_reward(
    {0: torch.zeros(2, 1, 1)}, 1, 1, 1, torch.ones(1, 1).bool(), [0], [1.0],
    kernel='pseudo_huber', level='token'), 'mmd_kernel_level=sentence')
  for source, level in (('rows', 'sentence'), ('rows', 'token'), ('draws', 'token')):
    with mock.patch.object(torch, 'multinomial', side_effect=AssertionError('sampled')):
      expect_value_error(lambda: mmd_reward.mmd_reinforce_loss(
        torch.ones(1, 1, 2) / 2, torch.zeros(1, 1).long(), torch.ones(1, 1).bool(),
        SimpleNamespace(kernel='pseudo_huber', kernel_level=level),
        group_size=2, eos_id=None, kernel_source=source),
        'mmd_kernel_source=draws and mmd_kernel_level=sentence')


if __name__ == '__main__':
  for name, fn in sorted(globals().copy().items()):
    if name.startswith('test_'):
      fn()
      print(f'PASS  {name}')
  print('\nall pseudo-Huber MMD tests passed')
