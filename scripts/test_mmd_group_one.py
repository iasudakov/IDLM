"""Regression checks for the TinyGSM MMD group-size-one path."""
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmd_reward


class RowReward:
  resolved_layers = [2, 9]

  def paired_reward(self, clean, generated, selected, eos_id,
                    return_per_layer=False):
    del eos_id
    reward = -(clean[:, 0].float() + generated[:, 0].float() + 1.0)
    active = selected.any(dim=1)
    if return_per_layer:
      return reward, active, {2: reward, 9: 2.0 * reward}
    return reward, active


def _draw_member_index(probs, num_samples, replacement=False, **kwargs):
  del kwargs
  assert replacement
  return torch.arange(
    num_samples, dtype=torch.long, device=probs.device).expand(
      probs.shape[0], -1)


def run(baseline, group_size=1):
  batch, length, vocab = 3, 2, 4
  probs = torch.full(
    (batch, length, vocab), 1.0 / vocab, requires_grad=True)
  x0 = torch.arange(batch).unsqueeze(1).expand(-1, length)
  selected = torch.ones(batch, length, dtype=torch.bool)
  with mock.patch.object(torch, 'multinomial', _draw_member_index):
    loss, stats = mmd_reward.mmd_reinforce_loss(
      probs, x0, selected, RowReward(), group_size=group_size, eos_id=None,
      baseline=baseline)
  loss.backward()
  assert torch.isfinite(loss)
  assert torch.isfinite(probs.grad).all()
  assert probs.grad.abs().sum() > 0
  if group_size == 1:
    assert stats['mmd/reward_spread'].item() == 0.0
    assert stats['mmd/reward_spread_l2'].item() == 0.0
    assert stats['mmd/reward_spread_l9'].item() == 0.0
  return loss.detach(), probs.grad.detach()


auto_loss, auto_grad = run('auto')
none_loss, none_grad = run('none')
torch.testing.assert_close(auto_loss, none_loss)
torch.testing.assert_close(auto_grad, none_grad)

batch_loss, batch_grad = run('batch')
assert torch.isfinite(batch_loss)
assert torch.isfinite(batch_grad).all()

auto_group_loss, auto_group_grad = run('auto', group_size=3)
group_loss, group_grad = run('group', group_size=3)
torch.testing.assert_close(auto_group_loss, group_loss)
torch.testing.assert_close(auto_group_grad, group_grad)

reward = torch.tensor([[-1.0], [-100.0], [-3.0]])
active = torch.tensor([True, False, True])
advantage = mmd_reward.batch_advantages(reward, active)
torch.testing.assert_close(advantage[active], torch.tensor([[2.0], [-2.0]]))

probs = torch.full((2, 2, 3), 1.0 / 3, requires_grad=True)
x0 = torch.zeros(2, 2, dtype=torch.long)
selected = torch.ones(2, 2, dtype=torch.bool)
for kwargs, expected in (
    ({'group_size': 0, 'baseline': 'auto'}, 'must be >= 1'),
    ({'group_size': 1, 'baseline': 'group'}, 'needs group_size >= 2'),
    ({'group_size': 1, 'baseline': 'invalid'}, 'auto|group|batch|none')):
  try:
    mmd_reward.mmd_reinforce_loss(
      probs, x0, selected, RowReward(), eos_id=None, **kwargs)
  except ValueError as error:
    assert expected in str(error), str(error)
  else:
    raise AssertionError(f'expected failure for {kwargs}')

print('PASS: TinyGSM MMD supports group_size=1 with explicit baselines')
