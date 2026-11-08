"""Regression test for group-size-invariant MMD REINFORCE reduction.

The candidates are deliberately identical, so increasing the number of them
must leave both the loss and its gradient unchanged. This isolates the final
`1 / group_size` reduction from changes in sampling variance or the baseline.
"""
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmd_reward


class ConstantReward:
  resolved_layers = [0]

  def paired_reward(self, clean, generated, selected, eos_id,
                    return_per_layer=False):
    del generated, eos_id, return_per_layer
    reward = torch.full(
      (clean.shape[0],), -2.0, dtype=torch.float32, device=clean.device)
    return reward, selected.any(dim=1)


class IndexedReward:
  """Rewards member k with k, making the LOO gradient analytic."""
  resolved_layers = [0]

  def paired_reward(self, clean, generated, selected, eos_id,
                    return_per_layer=False):
    del clean, eos_id, return_per_layer
    # The deterministic sampler below fills every position of member k with k.
    return generated[:, 0].float(), selected.any(dim=1)


def _always_draw_token_zero(probs, num_samples, replacement=False, **kwargs):
  del kwargs
  assert replacement
  return torch.zeros(
    probs.shape[0], num_samples, dtype=torch.long, device=probs.device)


def _draw_member_index(probs, num_samples, replacement=False, **kwargs):
  del kwargs
  assert replacement
  return torch.arange(
    num_samples, dtype=torch.long, device=probs.device).expand(
      probs.shape[0], -1)


def run(group_size):
  batch, length, vocab = 3, 5, 2
  probs = torch.tensor([0.25, 0.75], dtype=torch.float32) \
    .view(1, 1, vocab).repeat(batch, length, 1).requires_grad_()
  x0 = torch.zeros(batch, length, dtype=torch.long)
  selected = torch.ones(batch, length, dtype=torch.bool)

  with mock.patch.object(torch, 'multinomial', _always_draw_token_zero):
    loss, _ = mmd_reward.mmd_reinforce_loss(
      probs, x0, selected, ConstantReward(), group_size=group_size,
      eos_id=None, reward_weight=1.0, baseline='none')
  loss.backward()
  return loss.detach(), probs.grad.detach()


loss_1, grad_1 = run(1)
loss_8, grad_8 = run(8)

torch.testing.assert_close(loss_8, loss_1)
torch.testing.assert_close(grad_8, grad_1)
print('PASS: loss and gradient are invariant to repeated group candidates')

# Exercise the production `baseline=group` path as well. Rewards [0, 1, 2]
# produce leave-one-out advantages [-1.5, 0, 1.5]. With 8 selected positions,
# group size 3 and p(token)=0.2, the expected nonzero probability-gradient
# magnitude is 1.5 / (8 * 3 * 0.2) = 0.3125. Omitting the group average makes
# it 0.9375 instead, exactly three times too large.
batch, length, vocab, group = 2, 4, 5, 3
probs = torch.full(
  (batch, length, vocab), 1.0 / vocab, requires_grad=True)
x0 = torch.zeros(batch, length, dtype=torch.long)
selected = torch.ones(batch, length, dtype=torch.bool)
with mock.patch.object(torch, 'multinomial', _draw_member_index):
  loss, _ = mmd_reward.mmd_reinforce_loss(
    probs, x0, selected, IndexedReward(), group_size=group,
    eos_id=None, reward_weight=1.0, baseline='group')
loss.backward()
expected = torch.zeros_like(probs)
expected[..., 0] = 0.3125
expected[..., 2] = -0.3125
torch.testing.assert_close(probs.grad, expected, rtol=0, atol=0)
print('PASS: leave-one-out group gradient includes the 1 / group_size average')
