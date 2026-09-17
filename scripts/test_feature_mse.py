"""Aligned masked-token feature-MSE reward and score-estimator checks (CPU).

Run from the repository root: python scripts/test_feature_mse.py
"""
import itertools
import math
import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmd_reward


class TinyBackbone(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.scale = torch.nn.Parameter(torch.tensor(1.0))
    self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

  def forward(self, input_ids, sigma=None, attention_mask=None):
    values = input_ids.float() * self.scale
    hidden = torch.stack([values, 2 * values], dim=-1)
    for block in self.blocks:
      hidden = block(hidden)
    return hidden


class BinaryCritic:
  reward_type = 'feature_mse'
  kernel = 'pseudo_huber'  # Irrelevant even in token/rows feature_mse mode.
  kernel_level = 'token'

  def paired_reward(self, clean, generated, selected, eos_id,
                    return_per_layer=False):
    return mmd_reward.aligned_feature_mse_reward(
      clean.unsqueeze(-1), generated.unsqueeze(-1), selected)

  def multi_draw_reward(self, clean, draws, selected, eos_id):
    rewards = [self.paired_reward(clean, item, selected, eos_id)[0]
               for item in draws]
    return torch.stack(rewards).mean(dim=0), selected.any(dim=-1)


class FeatureMseTests(unittest.TestCase):
  def test_aligned_errors_do_not_cancel_when_token_means_match(self):
    real = torch.tensor([[[1., -1.], [-1., 1.]]])
    fake = real.flip(1)
    selected = torch.ones(1, 2, dtype=torch.bool)
    self.assertTrue(torch.equal(real.mean(1), fake.mean(1)))
    reward, active = mmd_reward.aligned_feature_mse_reward(real, fake, selected)
    torch.testing.assert_close(reward, torch.tensor([-4.]))
    self.assertTrue(active.item())

  def test_normalization_masking_nonfinite_and_empty_rows(self):
    real = torch.zeros(4, 3, 2)
    fake = torch.tensor([[[2., -2.], [4., -4.], [float('nan'), float('inf')]],
                         [[2., -2.], [float('inf'), float('nan')], [8., 8.]],
                         [[float('nan'), float('inf')]] * 3,
                         [[2., -2.], [2., -2.], [2., -2.]]])
    selected = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1]],
                            dtype=torch.bool)
    reward, active = mmd_reward.aligned_feature_mse_reward(real, fake, selected)
    torch.testing.assert_close(reward, torch.tensor([-10., -4., 0., -4.]))
    torch.testing.assert_close(active, torch.tensor([True, True, False, True]))
    self.assertTrue(torch.isfinite(reward).all().item())

  def test_fp32_and_detached_features(self):
    real = torch.tensor([[[1e4, 1e4]]], dtype=torch.bfloat16, requires_grad=True)
    fake = -real.detach().clone().requires_grad_()
    reward, _ = mmd_reward.aligned_feature_mse_reward(
      real, fake, torch.ones(1, 1, dtype=torch.bool))
    expected = -(fake.float() - real.float()).square().mean((1, 2))
    torch.testing.assert_close(reward, expected)
    self.assertEqual(reward.dtype, torch.float32)
    self.assertFalse(reward.requires_grad)
    self.assertIsNone(real.grad)

  def test_shapes_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'matching'):
      mmd_reward.aligned_feature_mse_reward(
        torch.zeros(2, 3, 4), torch.zeros(2, 3, 5), torch.ones(2, 3))
    with self.assertRaisesRegex(ValueError, 'selected mask'):
      mmd_reward.aligned_feature_mse_reward(
        torch.zeros(2, 3, 4), torch.zeros(2, 3, 4), torch.ones(2, 4))

  def test_layers_sum_per_row_ignoring_kernel_grouping(self):
    real = torch.zeros(2, 2, 1)
    fake = torch.tensor([[[2.], [7.]], [[4.], [7.]]])
    features = {0: torch.cat([real, fake]), 1: torch.cat([real, fake * 2])}
    reward, active, per_layer = mmd_reward.summed_paired_reward(
      features, 2, torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
      [0, 1], [0.5, 2.0], reward_type='feature_mse', kernel='pseudo_huber',
      level='sentence', rows_of=7, exclude_same_sentence=True)
    torch.testing.assert_close(per_layer[0], torch.tensor([-4., -16.]))
    torch.testing.assert_close(per_layer[1], torch.tensor([-16., -64.]))
    torch.testing.assert_close(reward, torch.tensor([-34., -136.]))
    self.assertTrue(active.all().item())

  def test_draws_average_individual_errors_not_feature_means(self):
    real = torch.zeros(2, 2, 1)
    draws = torch.tensor([[[[2.], [10.]], [[3.], [8.]]],
                          [[[-2.], [-10.]], [[-3.], [-8.]]]])
    features = {0: torch.cat([real, draws.flatten(0, 1)])}
    selected = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    for count in (1, 2):
      stack = {0: torch.cat([real, draws[:count].flatten(0, 1)])}
      reward, active, _ = mmd_reward.summed_multi_draw_reward(
        stack, 2, count, 2, selected, [0], [1.0],
        reward_type='feature_mse', kernel='pseudo_huber', level='token',
        exclude_same_sentence=True)
      torch.testing.assert_close(reward, torch.tensor([-4., 0.]))
      torch.testing.assert_close(active, torch.tensor([True, False]))
    self.assertTrue(torch.equal(draws.mean(0), real))

  def test_both_critic_adapters_thread_mode_for_paired_and_draws(self):
    for adapter in ('native', 'hf'):
      backbone = TinyBackbone()
      if adapter == 'native':
        critic = mmd_reward.FrozenBackboneFeatures(
          backbone, mask_index=19, layer_index=[0, 1], noise_level=0.0,
          layer_weights=[0.5, 0.5])
      else:
        critic = object.__new__(mmd_reward.FrozenARFeatures)
        critic.model = backbone
        critic.vocab_size = 20
        critic._resolve_layers(backbone, [0, 1], [0.5, 0.5])
      critic.reward_type = 'feature_mse'
      critic.kernel = 'pseudo_huber'
      critic.kernel_level = 'token'
      critic.kernel_batch_size = 7
      critic.rbf_exclude_same_sentence = True
      clean = torch.tensor([[1, 2], [3, 4]])
      generated = torch.tensor([[3, 8], [5, 9]])
      selected = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
      paired, active, layers = critic.paired_reward(
        clean, generated, selected, 0, return_per_layer=True)
      multi, multi_active, multi_layers = critic.multi_draw_reward(
        clean, torch.stack([generated, clean]), selected, 0,
        return_per_layer=True)
      torch.testing.assert_close(paired, torch.tensor([-10., 0.]))
      torch.testing.assert_close(multi, paired / 2)
      torch.testing.assert_close(active, multi_active)
      for layer in (0, 1):
        torch.testing.assert_close(layers[layer], paired)
        torch.testing.assert_close(multi_layers[layer], multi)
      self.assertFalse(paired.requires_grad)
      self.assertIsNone(backbone.scale.grad)
      for block in backbone.blocks:
        self.assertFalse(block._forward_hooks)

  def test_reinforce_rejects_widened_scoring(self):
    with self.assertRaisesRegex(ValueError, 'written/masked'):
      mmd_reward.mmd_reinforce_loss(
        torch.full((1, 2, 2), 0.5), torch.zeros(1, 2, dtype=torch.long),
        torch.tensor([[True, False]]), BinaryCritic(), group_size=2,
        eos_id=0, scored=torch.ones(1, 2, dtype=torch.bool))

  def test_reinforce_favors_closer_samples_and_does_not_update_unwritten(self):
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    draws = torch.tensor([[0, 1], [1, 1]])
    with mock.patch.object(torch, 'multinomial', return_value=draws):
      loss, stats = mmd_reward.mmd_reinforce_loss(
        logits.softmax(-1), torch.zeros(1, 2, dtype=torch.long),
        torch.tensor([[True, False]]), BinaryCritic(), group_size=2, eos_id=0)
    loss.backward()
    self.assertLess(logits.grad[0, 0, 0].item(), 0)
    self.assertGreater(logits.grad[0, 0, 1].item(), 0)
    torch.testing.assert_close(logits.grad[0, 1], torch.zeros(2))
    self.assertFalse(stats['mmd/reward'].requires_grad)

  def test_reinforce_empty_mask_has_zero_finite_loss_and_gradient(self):
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    loss, _ = mmd_reward.mmd_reinforce_loss(
      logits.softmax(-1), torch.zeros(1, 2, dtype=torch.long),
      torch.zeros(1, 2, dtype=torch.bool), BinaryCritic(), group_size=2, eos_id=0)
    self.assertEqual(loss.item(), 0)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

  @staticmethod
  def enumerated_gradient(draw_count, group_size, legacy=False):
    probability = 0.3
    expected = 0.0
    critic = BinaryCritic()
    if legacy:
      critic.reward_type = 'feature'
      critic.kernel = 'linear'
    for draws in itertools.product((0, 1), repeat=draw_count * group_size):
      theta = torch.tensor(math.log(probability / (1 - probability)),
                           dtype=torch.float64, requires_grad=True)
      p = theta.sigmoid()
      probs = torch.stack([1 - p, p]).reshape(1, 1, 2)
      mass = math.prod(probability if draw else 1 - probability for draw in draws)
      with mock.patch.object(torch, 'multinomial',
                             return_value=torch.tensor([draws])):
        loss, _ = mmd_reward.mmd_reinforce_loss(
          probs, torch.zeros(1, 1, dtype=torch.long),
          torch.ones(1, 1, dtype=torch.bool), critic, group_size=group_size,
          eos_id=0, kernel_source='draws', kernel_draws=draw_count)
      expected += mass * torch.autograd.grad(loss, theta)[0].item()
    return expected

  def test_exact_expected_gradient_keeps_scale_with_more_draws(self):
    for group_size in (1, 2):
      reference = self.enumerated_gradient(1, group_size)
      self.assertGreater(reference, 0)
      if group_size == 1:
        self.assertAlmostEqual(reference, 0.3 * 0.7, places=7)
      for draw_count in (2, 3):
        with self.subTest(group=group_size, draws=draw_count):
          self.assertAlmostEqual(
            self.enumerated_gradient(draw_count, group_size), reference, places=7)

  def test_legacy_kernel_score_estimator_normalization_is_unchanged(self):
    reference = self.enumerated_gradient(1, 2, legacy=True)
    self.assertAlmostEqual(self.enumerated_gradient(2, 2, legacy=True),
                           reference / 2, places=7)


if __name__ == '__main__':
  unittest.main(verbosity=2)
