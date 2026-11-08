"""Physical forward chunks preserve the complete logical feature reward (CPU).

Run from the repository root: python scripts/test_mmd_forward_batch.py
"""
import inspect
import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmd_reward


class RecordingBackbone(torch.nn.Module):
  """A row-independent stand-in supporting native and HF forward signatures."""
  def __init__(self):
    super().__init__()
    self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    self.calls = []

  def forward(self, input_ids, sigma=None, attention_mask=None):
    self.calls.append(input_ids.shape[0])
    value = input_ids.float()
    if sigma is not None:
      value = value + sigma.reshape(-1, 1)
    if attention_mask is not None:
      value = value + attention_mask.float() * 0.25
    hidden = torch.stack([value.sin(), value.cos(), value.square() * 0.1], -1)
    for index, block in enumerate(self.blocks):
      hidden = block(hidden * (index + 1))
    return hidden


def make_critic(adapter, *, kernel='rbf', level='token', rows=2,
                reward_type='feature', noise=0.0):
  backbone = RecordingBackbone()
  if adapter == 'native':
    critic = mmd_reward.FrozenBackboneFeatures(
      backbone, mask_index=8, layer_index=[0, 1], layer_weights=[0.5, 1.5],
      noise_level=noise)
  else:
    critic = object.__new__(mmd_reward.FrozenARFeatures)
    critic.model, critic.vocab_size = backbone, 7
    critic._resolve_layers(backbone, [0, 1], [0.5, 1.5])
  critic.kernel, critic.kernel_level = kernel, level
  critic.kernel_batch_size, critic.reward_type = rows, reward_type
  critic.rbf_alpha = None
  return critic, backbone


class ForwardBatchTests(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(713)
    self.clean = torch.randint(0, 7, (4, 5))
    self.selected = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1],
                                  [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                                 dtype=torch.bool)

  def assert_nested_equal(self, first, second):
    self.assertEqual(len(first), len(second))
    for left, right in zip(first, second):
      if isinstance(left, dict):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
          torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
      else:
        torch.testing.assert_close(left, right, rtol=0, atol=0)

  def test_native_and_hf_features_preserve_order_sigma_attention_and_hooks(self):
    tokens = torch.tensor([[1, 9, 2], [2, 3, 4], [5, 6, 0], [0, 2, 6]])
    for adapter in ('native', 'hf'):
      critic, backbone = make_critic(adapter)
      if adapter == 'native':
        call = lambda: critic.features(tokens, torch.tensor([0., 0.1, 0.2, 0.3]))
      else:
        mask = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0]])
        call = lambda: critic.features(tokens, 0, attention_mask=mask)
      expected = call()
      for size in (1, 2, 3, 8):
        with self.subTest(adapter=adapter, size=size):
          critic.forward_batch_size = size
          backbone.calls.clear()
          actual = call()
          for index in expected:
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
            self.assertFalse(actual[index].requires_grad)
          self.assertLessEqual(max(backbone.calls), size)
          self.assertEqual(sum(backbone.calls), tokens.shape[0])
          self.assertTrue(all(not block._forward_hooks for block in backbone.blocks))
      critic.forward_batch_size = 0
      with self.assertRaisesRegex(ValueError, 'positive integer'):
        call()

  def test_reward_keeps_logical_auto_bandwidth_and_noise_for_all_adapters(self):
    generated = torch.randint(0, 7, self.clean.shape)
    draws = torch.stack([generated, generated.roll(1, dims=1)])
    for adapter in ('native', 'hf'):
      for source in ('rows', 'draws'):
        for level in ('token', 'sentence'):
          for distance in ('squared_euclidean', 'pseudo_huber'):
            with self.subTest(adapter=adapter, source=source, level=level,
                              distance=distance):
              critic, backbone = make_critic(adapter, level=level, noise=0.25)
              critic.rbf_distance = distance
              critic.rbf_block_size = 2
              def score():
                if source == 'rows':
                  return critic.paired_reward(
                    self.clean, generated, self.selected, 0, return_per_layer=True)
                return critic.multi_draw_reward(
                  self.clean, draws, self.selected, 0, return_per_layer=True)
              original_alpha = mmd_reward.median_alpha
              with mock.patch.object(mmd_reward, 'median_alpha',
                                     wraps=original_alpha) as alpha:
                torch.manual_seed(99)
                expected = score()
                full_inputs = [(call.args[0].clone(), call.args[1].clone())
                               for call in alpha.call_args_list]
                alpha.reset_mock()
                critic.forward_batch_size = 2
                backbone.calls.clear()
                torch.manual_seed(99)
                actual = score()
                chunk_inputs = [(call.args[0], call.args[1])
                                for call in alpha.call_args_list]
              self.assert_nested_equal(expected, actual)
              self.assertEqual(len(full_inputs), 2)  # One bandwidth per layer.
              for full, chunk in zip(full_inputs, chunk_inputs):
                self.assert_nested_equal(full, chunk)
              self.assertLessEqual(max(backbone.calls), 2)
              logical_rows = (2 if source == 'rows' else 3) * len(self.clean)
              self.assertEqual(sum(backbone.calls), logical_rows)

  def test_automatic_subsampling_sees_all_logical_rows(self):
    critic, backbone = make_critic('hf', rows=1)
    clean = torch.arange(32 * 20).reshape(32, 20) % 7
    fake = (clean + 2) % 7
    selected = torch.ones_like(clean, dtype=torch.bool)
    torch.manual_seed(9)
    full = critic.paired_reward(clean, fake, selected, 0)
    critic.forward_batch_size = 8
    backbone.calls.clear()
    torch.manual_seed(9)
    split = critic.paired_reward(clean, fake, selected, 0)
    self.assert_nested_equal(full, split)
    self.assertEqual(backbone.calls, [8] * 8)

  def test_from_samples_matches_loss_stats_and_gradients_with_forward_chunks(self):
    cases = [
      ('feature', 'rbf', 'token', 'rows', 2, 3, 'group'),
      ('feature', 'rbf', 'sentence', 'rows', 2, 3, 'group'),
      ('feature', 'rbf', 'token', 'draws', 2, 3, 'group'),
      ('feature', 'rbf', 'sentence', 'draws', 2, 3, 'group'),
      ('feature', 'pseudo_huber', 'sentence', 'draws', 2, 3, 'group'),
      ('feature_mse', 'linear', 'token', 'draws', 2, 3, 'group'),
      ('feature_mse', 'linear', 'token', 'rows', 1, 1, 'batch'),
      ('feature', 'linear', 'token', 'rows', 1, 1, 'none'),
    ]
    for kind, kernel, level, source, k, group, baseline in cases:
      with self.subTest(kind=kind, kernel=kernel, level=level, source=source,
                        baseline=baseline):
        critic, backbone = make_critic(
          'native', kernel=kernel, level=level, rows=k, reward_type=kind)
        critic.forward_batch_size = 2
        d = group * (k if source == 'draws' else 1)
        raw_draws = torch.randint(0, 7, (len(self.clean) * self.clean.shape[1], d))
        samples = raw_draws.reshape(*self.clean.shape, d).permute(2, 0, 1)
        samples = torch.where(self.selected.unsqueeze(0), samples, self.clean.unsqueeze(0))
        kwargs = dict(group_size=group, eos_id=0, baseline=baseline,
                      kernel_source=source, kernel_draws=k, reward_weight=1.3)
        if 'shuffle_rewards' in inspect.signature(mmd_reward.mmd_reinforce_loss).parameters:
          kwargs['shuffle_rewards'] = True
        logits = torch.randn(4, 5, 7, requires_grad=True)
        direct_logits = logits.detach().clone().requires_grad_()
        torch.manual_seed(123)
        with mock.patch.object(torch, 'multinomial', return_value=raw_draws):
          expected, expected_stats = mmd_reward.mmd_reinforce_loss(
            logits.softmax(-1), self.clean, self.selected, critic, **kwargs)
        expected.backward()

        # The full vocabulary probabilities only exist for one physical chunk.
        gathered = []
        for start in (0, 2):
          logp = direct_logits[start:start+2].softmax(-1).clamp_min(1e-12).log()
          gathered.append(torch.stack([
            logp.gather(-1, drawn[start:start+2].unsqueeze(-1)).squeeze(-1)
            for drawn in samples]))
        sampled_logp = torch.cat(gathered, dim=1)
        torch.manual_seed(123)
        actual, actual_stats = mmd_reward.mmd_reinforce_loss_from_samples(
          sampled_logp, self.clean, self.selected, samples, critic, **kwargs)
        actual.backward()
        self.assert_nested_equal((expected, expected_stats), (actual, actual_stats))
        torch.testing.assert_close(direct_logits.grad, logits.grad, rtol=0, atol=0)
        torch.testing.assert_close(
          direct_logits.grad[~self.selected],
          torch.zeros_like(direct_logits.grad[~self.selected]), rtol=0, atol=0)

  def test_empty_written_span_remains_differentiable_zero(self):
    critic, _ = make_critic('native', reward_type='feature_mse')
    selected = torch.zeros_like(self.selected)
    logp = torch.randn(2, *self.clean.shape, requires_grad=True)
    loss, _ = mmd_reward.mmd_reinforce_loss_from_samples(
      logp, self.clean, selected, self.clean.unsqueeze(0).expand(2, -1, -1),
      critic, group_size=2, eos_id=0)
    loss.backward()
    self.assertEqual(loss.item(), 0.)
    torch.testing.assert_close(logp.grad, torch.zeros_like(logp))

  def test_legacy_does_not_keep_full_vocabulary_logs_during_critic(self):
    critic, _ = make_critic('native', reward_type='feature_mse')
    events = []
    original_score = critic.paired_reward
    original_log = torch.log
    def score(*args, **kwargs):
      events.append('critic')
      return original_score(*args, **kwargs)
    def log(value):
      events.append('full_vocabulary_log')
      return original_log(value)
    logits = torch.randn(4, 5, 7, requires_grad=True)
    with mock.patch.object(critic, 'paired_reward', side_effect=score), \
         mock.patch.object(torch, 'log', side_effect=log):
      loss, _ = mmd_reward.mmd_reinforce_loss(
        logits.softmax(-1), self.clean, self.selected, critic,
        group_size=3, eos_id=0)
    loss.backward()
    self.assertEqual(events, ['critic'] * 3 + ['full_vocabulary_log'])

  def test_supplied_sample_shapes_and_dtypes_are_checked(self):
    critic, _ = make_critic('native')
    samples = self.clean.unsqueeze(0).repeat(2, 1, 1)
    for logp, ids, message in [
        (torch.zeros(1, *self.clean.shape), samples, 'must both be'),
        (torch.zeros_like(samples).float(), samples.float(), 'torch.long'),
        (torch.zeros_like(samples), samples, 'floating point')]:
      with self.subTest(message=message):
        with self.assertRaisesRegex(ValueError, message):
          mmd_reward.mmd_reinforce_loss_from_samples(
            logp, self.clean, self.selected, ids, critic, group_size=2, eos_id=0)


if __name__ == '__main__':
  unittest.main(verbosity=2)
