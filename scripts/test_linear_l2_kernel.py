"""CPU regressions for negative squared distance between pooled feature means.

Run from the repository root: python scripts/test_linear_l2_kernel.py
No pretrained weights, network access or accelerator are needed.
"""
import math
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmd_reward

# OWT averages the outer REINFORCE group; TinyGSM historically sums it.
AVERAGE_REINFORCE_GROUP = False


def _draw_reward(real, fake, selected, kernel='linear_l2', level='token'):
  count, batch, length, hidden = fake.shape
  return mmd_reward.summed_multi_draw_reward(
    {0: torch.cat([real, fake.reshape(count * batch, length, hidden)])},
    batch=batch, draw_count=count, length=length, selected=selected,
    indices=[0], weights=[1.0], kernel=kernel, level=level)


def test_square_mean_helper_preserves_leading_axes_and_fp32():
  delta = torch.tensor([[[3.0, 4.0], [-0.5, 1.0]],
                        [[0.0, 0.0], [2.0, -2.0]]])
  expected = torch.tensor([[-12.5, -0.625], [0.0, -4.0]])
  for dtype in (torch.float32, torch.bfloat16, torch.float64):
    with torch.autocast('cpu', dtype=torch.bfloat16):
      actual = mmd_reward.mean_squared_reward(delta.to(dtype))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_opposing_token_errors_cancel_before_squaring():
  real = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [999.0, -999.0]]])
  fake = torch.tensor([[[5.0, 7.0], [1.0, 3.0], [-500.0, 500.0]]])
  selected = torch.tensor([[True, True, False]])
  # Aligned token MSE would be 16; both mean vectors are exactly [3, 5].
  for level in ('token', 'sentence'):
    paired, active = mmd_reward.paired_sequence_reward(
      real, fake, selected, kernel='linear_l2', level=level)
    drawn, draw_active, _ = _draw_reward(
      real, fake.unsqueeze(0), selected, level=level)
    torch.testing.assert_close(paired, torch.zeros(1), rtol=0, atol=0)
    torch.testing.assert_close(drawn, paired, rtol=0, atol=0)
    assert active.tolist() == draw_active.tolist() == [True]


def test_rows_grouping_preserves_token_and_sentence_weighting():
  selected = torch.tensor([[True, False, False], [True, True, True],
                           [True, False, False], [False, False, False],
                           [False, False, False], [False, False, False]])
  real = torch.zeros(6, 3, 2)
  shifts = torch.tensor([[0.0, 2.0], [6.0, 4.0], [-2.0, 4.0],
                         [50.0, 50.0], [60.0, 60.0], [70.0, 70.0]])
  fake = shifts[:, None].expand_as(real).clone()
  real[~selected] = 999.0
  fake[~selected] = -500.0
  # First group: token mean delta = [4.5, 3.5], sentence mean = [3, 3].
  # The inactive member of group two shares its active partner's reward.
  for level, first in (('token', -16.25), ('sentence', -9.0)):
    actual, active = mmd_reward.paired_sequence_reward(
      real, fake, selected, kernel='linear_l2', level=level, rows_of=2)
    torch.testing.assert_close(
      actual, torch.tensor([first, first, -10.0, -10.0, 0.0, 0.0]))
    assert active.tolist() == [True, True, True, True, False, False]


def test_draws_compare_means_without_variance_or_per_draw_mse():
  real = torch.zeros(3, 3, 2)
  selected = torch.tensor([[True, True, False], [True, False, False],
                           [False, False, False]])
  fake = torch.zeros(2, 3, 3, 2)
  fake[0] = torch.tensor([0.0, 2.0])
  fake[1] = torch.tensor([4.0, 6.0])
  # Within each draw, opposing token changes also cancel in the mean.
  fake[:, 0, 0] += torch.tensor([3.0, -3.0])
  fake[:, 0, 1] -= torch.tensor([3.0, -3.0])
  real[~selected] = 500.0
  fake[:, ~selected] = -999.0
  for level in ('token', 'sentence'):
    reward, active, layers = _draw_reward(real, fake, selected, level=level)
    # Mean(fake) = [2, 4], hence -10, not mean(per-draw MSE) = -14.
    torch.testing.assert_close(reward, torch.tensor([-10.0, -10.0, 0.0]))
    torch.testing.assert_close(layers[0], reward)
    assert active.tolist() == [True, True, False]
    empty, inactive, _ = _draw_reward(
      real, fake, torch.zeros_like(selected), level=level)
    torch.testing.assert_close(empty, torch.zeros(3), rtol=0, atol=0)
    assert not inactive.any()


def test_symmetric_draws_are_zero_while_legacy_linear_keeps_u_correction():
  real = torch.zeros(1, 1, 2)
  fake = torch.tensor([[[[2.0, 4.0]]], [[[-2.0, -4.0]]]])
  selected = torch.ones(1, 1, dtype=torch.bool)
  for level in ('token', 'sentence'):
    reward, active, _ = _draw_reward(real, fake, selected, level=level)
    torch.testing.assert_close(reward, torch.zeros(1), rtol=0, atol=0)
    assert active.tolist() == [True]
  legacy, _, _ = _draw_reward(real, fake, selected, 'linear', 'sentence')
  torch.testing.assert_close(legacy, torch.tensor([10.0]), rtol=0, atol=0)
  # At K=1 the legacy sentence estimator has no variance correction.
  new_k1, _, _ = _draw_reward(real, fake[:1], selected, level='sentence')
  old_k1, _, _ = _draw_reward(real, fake[:1], selected, 'linear', 'sentence')
  torch.testing.assert_close(new_k1, torch.tensor([-10.0]), rtol=0, atol=0)
  torch.testing.assert_close(new_k1, old_k1, rtol=0, atol=0)


def test_historical_linear_and_pseudo_huber_rewards_are_unchanged():
  real = torch.zeros(1, 1, 2)
  fake = torch.tensor([[[[2.0, 4.0]]], [[[-2.0, -4.0]]]])
  selected = torch.ones(1, 1, dtype=torch.bool)

  def rho(a, b):
    return sum(math.sqrt((x - y) ** 2 + 1e-6) - 0.001
               for x, y in zip(a, b)) / 2

  historical = torch.tensor([-rho([0, 0], [2, 4])])
  for level in ('token', 'sentence'):
    paired, _ = mmd_reward.paired_sequence_reward(
      real, fake[0], selected, kernel='linear', level=level)
    torch.testing.assert_close(paired, historical)
  token, _, _ = _draw_reward(real, fake[:1], selected, 'linear', 'token')
  huber_k1, _, _ = _draw_reward(
    real, fake[:1], selected, 'pseudo_huber', 'sentence')
  torch.testing.assert_close(token, historical)
  torch.testing.assert_close(huber_k1, historical)
  huber_k2, _, _ = _draw_reward(
    real, fake, selected, 'pseudo_huber', 'sentence')
  expected = -rho([0, 0], [2, 4]) + 0.5 * rho([2, 4], [-2, -4])
  torch.testing.assert_close(huber_k2, torch.tensor([expected]), atol=3e-7, rtol=0)


class _Block(torch.nn.Linear):
  def __init__(self, scale):
    super().__init__(2, 2, bias=False)
    with torch.no_grad():
      self.weight.copy_(torch.eye(2) * scale)
    self.seen_dtypes = []

  def forward(self, hidden):
    output = super().forward(hidden)
    self.seen_dtypes.append(output.dtype)
    return output


class _TinyFeatures(torch.nn.Module):
  def __init__(self):
    super().__init__()
    table = torch.tensor([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0],
                          [-2.0, -4.0], [-4.0, -8.0], [1.0, 3.0],
                          [1000.0, -1000.0], [0.0, 0.0]])
    self.embedding = torch.nn.Embedding.from_pretrained(table, freeze=True)
    self.blocks = torch.nn.ModuleList([_Block(1.0), _Block(2.0)])
    self.seen_tokens = []
    self.seen_attention = []

  def forward(self, tokens=None, sigma=None, *, input_ids=None,
              attention_mask=None):
    tokens = input_ids if input_ids is not None else tokens
    if input_ids is None:
      assert sigma.shape == (tokens.shape[0],)
    self.seen_tokens.append(tokens.clone())
    self.seen_attention.append(attention_mask)
    hidden = self.embedding(tokens)
    for block in self.blocks:
      hidden = block(hidden)
    return hidden


def _critic(kind, level, layers=(0, 1), weights=(0.25, 1.5)):
  model = _TinyFeatures()
  if kind == 'ar':
    # Exercise the external-AR wrapper without loading pretrained weights.
    critic = mmd_reward.FrozenARFeatures.__new__(mmd_reward.FrozenARFeatures)
    critic.model = model
    critic.vocab_size = 8
    critic.device = torch.device('cpu')
    critic._resolve_layers(model, list(layers), list(weights))
  else:
    critic = mmd_reward.FrozenBackboneFeatures(
      model, mask_index=7, layer_index=list(layers),
      layer_weights=list(weights), noise_level=0.0)
  critic.kernel = 'linear_l2'
  critic.kernel_level = level
  return critic, model


def test_actual_ar_and_diffusion_hooks_multilayer_masks_and_bf16_forward():
  clean = torch.tensor([[0, 1, 6], [1, 6, 6], [4, 2, 6]])
  draws = torch.tensor([[[1, 2, 5], [2, 5, 5], [0, 0, 0]],
                         [[3, 4, 5], [4, 5, 5], [1, 1, 1]]])
  selected = torch.tensor([[True, True, False], [True, False, False],
                           [False, False, False]])
  for kind in ('ar', 'diffusion'):
    for level in ('token', 'sentence'):
      for source in ('rows', 'draws'):
        critic, model = _critic(kind, level)
        kwargs = {'return_per_layer': True}
        if kind == 'ar':
          kwargs['attention_mask'] = selected
        with torch.autocast('cpu', dtype=torch.bfloat16):
          if source == 'rows':
            reward, active, layers = critic.paired_reward(
              clean, draws[0], selected, eos_id=7, **kwargs)
            expected = torch.tensor([-10.0, -10.0, 0.0])
            expected_rows = 6
          else:
            reward, active, layers = critic.multi_draw_reward(
              clean, draws, selected, eos_id=7, **kwargs)
            expected = torch.tensor([-2.5, -10.0, 0.0])
            expected_rows = 9
        assert reward.dtype == torch.float32 and not reward.requires_grad
        assert active.tolist() == [True, True, False]
        torch.testing.assert_close(layers[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(layers[1], expected * 4, rtol=0, atol=0)
        torch.testing.assert_close(reward, expected * 6.25, rtol=0, atol=0)
        assert len(model.seen_tokens) == 1
        assert model.seen_tokens[0].shape == (expected_rows, 3)
        assert all(block.seen_dtypes == [torch.bfloat16]
                   for block in model.blocks)
        if kind == 'ar':
          torch.testing.assert_close(
            model.seen_attention[0], selected.repeat(expected_rows // 3, 1))


def test_diffusion_corruption_mask_is_shared_with_generated_features():
  critic, model = _critic('diffusion', 'token', layers=(0,), weights=(1.0,))
  critic.noise_level = 0.25
  clean = torch.tensor([[0, 1, 6], [1, 6, 6]])
  fake = torch.tensor([[1, 2, 5], [2, 5, 5]])
  selected = torch.tensor([[True, True, False], [True, False, False]])
  keep = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
  with mock.patch.object(torch, 'rand', return_value=keep):
    reward, active = critic.paired_reward(clean, fake, selected, eos_id=None)
  torch.testing.assert_close(reward, torch.tensor([-2.5, 0.0]))
  assert active.tolist() == [True, True]
  assert model.seen_tokens[0].tolist() == [
    [7, 1, 6], [7, 6, 6], [7, 2, 5], [7, 5, 5]]


def test_reinforce_sampled_gradient_keeps_written_mask_k_and_group_scaling():
  group = 3
  clean = torch.zeros(2, 3, dtype=torch.long)
  selected = torch.tensor([[True, False, True], [False, True, False]])
  scored = torch.ones_like(selected)
  values = torch.tensor([0.2, 0.3, 0.5]).expand(2, 3, 3).clone()
  weight = 0.7
  group_divisor = group if AVERAGE_REINFORCE_GROUP else 1
  # Feature vectors for ids 0,1,2 are [0,0], [2,4], [4,8]. With
  # scored=all, row 0 has mean deltas 0, [4/3,8/3], [8/3,16/3];
  # row 1 has half these deltas. LOO advantages are hand-calculated.
  advantages = [[100 / 9, 40 / 9, -140 / 9],
                [25 / 9, 10 / 9, -35 / 9]]
  expected_gradient = torch.zeros_like(values)
  expected_loss = 0.0
  for row, positions in enumerate(([0, 2], [1])):
    for candidate in range(group):
      coefficient = -weight * advantages[row][candidate] / (3 * group_divisor)
      expected_loss += coefficient * len(positions) * math.log(
        values[row, 0, candidate].item())
      for position in positions:
        expected_gradient[row, position, candidate] = (
          coefficient / values[row, position, candidate])

  for source, count in (('rows', 1), ('draws', 1), ('draws', 2), ('draws', 4)):
    probabilities = values.clone().requires_grad_(True)
    critic, _ = _critic('diffusion', 'sentence', layers=(0,), weights=(1.0,))
    candidate_ids = torch.arange(group).repeat_interleave(count)

    def sample(flat_probs, num_samples, replacement=False):
      assert num_samples == group * count and replacement
      return candidate_ids.expand(flat_probs.shape[0], -1)

    route = 'multi_draw_reward' if source == 'draws' else 'paired_reward'
    with mock.patch.object(torch, 'multinomial', side_effect=sample), \
         mock.patch.object(critic, route, wraps=getattr(critic, route)) as calls:
      loss, stats = mmd_reward.mmd_reinforce_loss(
        probabilities, clean, selected, critic, group_size=group, eos_id=None,
        reward_weight=weight, baseline='group', scored=scored,
        kernel_source=source, kernel_draws=count)
    loss.backward()
    torch.testing.assert_close(loss.detach(), torch.tensor(expected_loss))
    torch.testing.assert_close(probabilities.grad, expected_gradient)
    assert (probabilities.grad[~selected] == 0).all()
    assert calls.call_count == group
    # Rewards: [0,-40/9,-160/9] and [0,-10/9,-40/9].
    torch.testing.assert_close(stats['mmd/reward'], torch.tensor(-125 / 27))


def test_config_accepts_all_feature_routes_and_preserves_layer_settings():
  from scripts.test_feature_mse_config import _model, _feature_stubs

  for feature_source in ('ar', 'diffusion'):
    for kernel_source in ('rows', 'draws'):
      for level in ('token', 'sentence'):
        model = _model(
          mmd_reward_type='feature', mmd_kernel='linear_l2',
          mmd_feature_source=feature_source, mmd_kernel_source=kernel_source,
          mmd_kernel_level=level, mmd_rbf_alpha=0.13, mmd_rbf_block_size=8)
        with _feature_stubs() as stubs:
          critic = model._mmd_critic
          assert critic is stubs.critic and model._mmd_critic is critic
          assert critic.reward_type == 'feature'
          assert critic.kernel == 'linear_l2'
          assert critic.kernel_level == level
          assert critic.kernel_batch_size == 4
          assert critic.rbf_alpha == 0.13
          assert critic.rbf_block_size == 8
          assert stubs.native.call_count == 1
          assert stubs.native.call_args.kwargs['layer_index'] == [0, 3]
          assert stubs.native.call_args.kwargs['layer_weights'] == [0.25, 0.75]


def test_invalid_config_fails_before_loading_feature_models():
  from scripts.test_feature_mse_config import (
    _model, _feature_stubs, _expect_value_error)

  invalid = [dict(mmd_kernel='not-a-kernel'),
             dict(mmd_kernel_level='not-a-level'),
             dict(mmd_kernel_source='not-a-source'),
             dict(mmd_reward_type='idlm'),
             dict(mmd_rbf_exclude_same_sentence=True),
             dict(mmd_rbf_omit_real_real=True),
             dict(mmd_rbf_omit_fake_fake=True)]
  with _feature_stubs() as stubs:
    for overrides in invalid:
      config = dict(mmd_reward_type='feature', mmd_kernel='linear_l2')
      config.update(overrides)
      _expect_value_error(lambda: _model(**config)._mmd_critic)
    stubs.loader.assert_not_called()
    stubs.native.assert_not_called()
    stubs.hf.assert_not_called()


def main():
  for name, fn in sorted(globals().items()):
    if name.startswith('test_'):
      fn()
      print(f'PASS  {name}')
  print('\nall linear_l2 kernel tests passed')


if __name__ == '__main__':
  main()
