"""CPU checks that physical forward chunks preserve the logical MMD objective.

Run from the repository root: python scripts/test_mmd_generator_microbatch.py
Uses the actual MDLM methods and feature kernels without importing Lightning,
downloading models, or requiring a GPU. Draws are controlled deliberately:
the contract is mathematical equivalence, not identical random-number streams.
"""
import ast
import os
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmd_reward


class Config(SimpleNamespace):
  def get(self, name, default=None):
    return getattr(self, name, default)


def probe_class():
  path = os.path.join(os.path.dirname(mmd_reward.__file__), 'algo.py')
  with open(path, encoding='utf-8') as source:
    tree = ast.parse(source.read(), filename=path)
  names = {
    '_distill_generator_state', 'multistep_generation', '_mmd_forward_batch_size',
    '_mmd_microbatch_loss', '_mmd_term', '_mmd_content_mask', 'generator_loss',
    '_preserve_prompt_probs', '_preserve_prompt_tokens', 'q_xt_with_probs'}
  model = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'MDLM')
  methods = [node for node in model.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
  found = {node.name for node in methods}
  assert found == names, f'missing methods: {names - found}'
  probe = ast.ClassDef(name='Probe', bases=[], keywords=[], body=methods,
                      decorator_list=[])
  module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
  namespace = {
    'torch': torch, 'F': F, 'checkpoint': checkpoint,
    'trainer_base': SimpleNamespace(sample_categorical=lambda p: p.argmax(-1))}
  exec(compile(module, path, 'exec'), namespace)
  return namespace['Probe']


class FeatureBackbone(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
    self.batch_sizes = []

  def forward(self, tokens, sigma=None):
    self.batch_sizes.append(tokens.shape[0])
    values = tokens.float() / 7
    context = values.mean(-1, keepdim=True).expand_as(values)
    hidden = torch.stack((values, values.sin(), values.cos(), context), -1)
    return self.blocks[0](hidden)


def batch():
  row = torch.arange(32).unsqueeze(-1)
  position = torch.arange(8).unsqueeze(0)
  tokens = 1 + (3 * row + position) % 32
  # The first prompt token identifies each row, including across partitions.
  tokens[:, 0] = 1 + torch.arange(32)
  valid = torch.ones_like(tokens, dtype=torch.bool)
  valid[:, :2] = False
  return tokens, valid


def deterministic_multinomial(probabilities, num_samples, replacement=False,
                              **kwargs):
  del replacement, kwargs
  # Choose fixed probability ranks. Each row gives the same samples whether
  # passed alone, in a full batch, or during checkpoint recomputation.
  ranking = probabilities.argsort(dim=-1, descending=True)
  ranks = torch.arange(num_samples, device=probabilities.device) % ranking.shape[-1]
  return ranking[:, ranks]


def make_model(forward_batch=None, **overrides):
  settings = dict(
    mmd_forward_batch_size=forward_batch, mmd_reward_weight=0.7,
    dmd_weight=0.0, mmd_reward_type='feature', mmd_kernel='rbf',
    mmd_kernel_level='token', mmd_kernel_source='rows',
    mmd_kernel_batch_size=4, mmd_group_size=3, mmd_baseline='group',
    mmd_score_positions='written', mmd_scored_positions='answer',
    mmd_sentence_pooling='mean', mmd_shuffle_rewards=False,
    generator_force_full_mask=False)
  settings.update(overrides)
  cfg = Config(**settings)
  model = probe_class()()
  model.config = Config(adversarial_distill=cfg)
  model.device = torch.device('cpu')
  model.mask_index = 35
  model.tokenizer = SimpleNamespace(eos_token_id=None, pad_token_id=0)
  model.log_dict = mock.Mock()
  model.student_weight = torch.nn.Parameter(
    torch.linspace(-0.8, 0.6, 4 * 36, dtype=torch.float64).reshape(4, 36))
  model.student_bias = torch.nn.Parameter(torch.linspace(0.1, -0.1, 36,
                                                       dtype=torch.float64))
  model.forward_sizes = []
  model._sample_distill_generator_t = lambda n, accumulation: torch.full((n,), 0.5)
  model._sample_t = model._sample_distill_generator_t
  model.noise = lambda t: (-torch.ones_like(t), 1 - t)
  model._sigma_from_alphat = lambda alpha: -alpha.log()
  model._process_sigma = lambda sigma: sigma.squeeze(-1)

  def corrupt(self, x0, alpha, valid_tokens=None):
    del alpha
    row_id = x0[:, :1] - 1
    count = 1 + row_id.div(8, rounding_mode='floor')
    written = ((torch.arange(x0.shape[1]) >= 2)
               & (torch.arange(x0.shape[1]) < count + 2))
    if valid_tokens is not None:
      written &= valid_tokens
    return torch.where(written, self.mask_index, x0)

  def forward(self, tokens, sigma=None):
    del sigma
    self.forward_sizes.append(tokens.shape[0])
    row_id = tokens[:, :1].double() / 32
    position = torch.arange(tokens.shape[1], dtype=torch.float64).unsqueeze(0) / 8
    hidden = torch.stack((row_id.expand_as(tokens), position.expand_as(tokens),
                          tokens.double() / 36, torch.ones_like(tokens)), -1)
    return (hidden @ self.student_weight + self.student_bias).log_softmax(-1)

  model.q_xt = MethodType(corrupt, model)
  model.forward = MethodType(forward, model)
  critic = mmd_reward.FrozenBackboneFeatures(
    FeatureBackbone(), model.mask_index, layer_index=0, noise_level=0.0)
  critic.reward_type = cfg.mmd_reward_type
  critic.kernel = cfg.mmd_kernel
  critic.kernel_level = cfg.mmd_kernel_level
  critic.rbf_alpha = None
  critic.kernel_batch_size = cfg.mmd_kernel_batch_size
  critic.forward_batch_size = forward_batch
  model._mmd_critic = critic
  return model


def evaluate(model, x0, valid):
  with mock.patch.object(torch, 'multinomial', side_effect=deterministic_multinomial):
    loss = model.generator_loss(x0, valid, 0)
    before_backward = len(model.forward_sizes)
    loss.backward()
  gradient = torch.cat([model.student_weight.grad.flatten(),
                        model.student_bias.grad.flatten()])
  return loss.detach(), gradient, before_backward


class GeneratorMicrobatchTests(unittest.TestCase):
  def assert_equivalent(self, **settings):
    x0, valid = batch()
    full = make_model(**settings)
    chunked = make_model(8, **settings)
    full_loss, full_gradient, _ = evaluate(full, x0, valid)
    chunk_loss, chunk_gradient, initial_calls = evaluate(chunked, x0, valid)
    torch.testing.assert_close(chunk_loss, full_loss, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(chunk_gradient, full_gradient, rtol=1e-6, atol=1e-9)
    self.assertGreater(full_gradient.abs().max().item(), 1e-8)
    self.assertEqual(full.forward_sizes, [32])
    self.assertEqual(chunked.forward_sizes[:initial_calls], [8] * 4)
    self.assertGreater(len(chunked.forward_sizes), initial_calls)
    self.assertLessEqual(max(chunked.forward_sizes), 8)
    self.assertLessEqual(max(chunked._mmd_critic.denoiser.batch_sizes), 8)
    # A matching gradient must also result in a matching optimizer update.
    for model in (full, chunked):
      torch.optim.SGD([model.student_weight, model.student_bias], lr=0.1).step()
    torch.testing.assert_close(chunked.student_weight, full.student_weight,
                               rtol=1e-7, atol=1e-10)
    return full_loss, full_gradient

  def test_auto_rbf_rows_preserves_bandwidth_normalization_and_baselines(self):
    for baseline in ('group', 'batch', 'none'):
      with self.subTest(baseline=baseline):
        self.assert_equivalent(mmd_baseline=baseline)

  def test_auto_rbf_draws_preserves_candidate_grouping(self):
    self.assert_equivalent(mmd_kernel_source='draws', mmd_kernel_batch_size=4)

  def test_feature_mse_rows_and_draws_preserve_gradients(self):
    for source in ('rows', 'draws'):
      with self.subTest(source=source):
        self.assert_equivalent(mmd_reward_type='feature_mse',
                               mmd_kernel_source=source)

  def test_nondivisible_forward_limit_preserves_whole_logical_batch(self):
    x0, valid = batch()
    full = make_model()
    chunked = make_model(7)
    full_loss, full_gradient, _ = evaluate(full, x0, valid)
    chunk_loss, chunk_gradient, initial_calls = evaluate(chunked, x0, valid)
    torch.testing.assert_close(chunk_loss, full_loss, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(chunk_gradient, full_gradient, rtol=1e-6, atol=1e-9)
    self.assertEqual(chunked.forward_sizes[:initial_calls], [7, 7, 7, 7, 4])

  def test_checkpoint_replays_random_student_and_samples_without_advancing_rng(self):
    # Compare the checkpointed path against exactly the same physical chunks
    # with activation checkpointing disabled. This exercises real randomness,
    # including the feature critic's random corruption-mask draws.
    x0, valid = batch()
    eager = make_model(8)
    checkpointed = make_model(8)
    for model in (eager, checkpointed):
      original = model.forward

      def stochastic_forward(tokens, sigma=None, original=original):
        logp = original(tokens, sigma)
        noise = 0.2 * torch.rand_like(logp)
        return (logp + noise).log_softmax(-1)

      model.forward = stochastic_forward
    torch.manual_seed(901)
    with mock.patch('torch.utils.checkpoint.checkpoint',
                    side_effect=lambda fn, *args, **kwargs: fn(*args)):
      eager_loss = eager.generator_loss(x0, valid, 0)
      eager_loss.backward()
    real_multinomial = torch.multinomial
    recorded_draws = []

    def record_draws(*args, **kwargs):
      samples = real_multinomial(*args, **kwargs)
      recorded_draws.append(samples.clone())
      return samples

    torch.manual_seed(901)
    with mock.patch.object(torch, 'multinomial', side_effect=record_draws):
      checkpoint_loss = checkpointed.generator_loss(x0, valid, 0)
      forward_count = len(recorded_draws)
      before_backward_rng = torch.random.get_rng_state()
      checkpoint_loss.backward()
    self.assertEqual(forward_count, 4)
    self.assertEqual(len(recorded_draws), 8)
    for initial, repeated in zip(recorded_draws[:4], recorded_draws[4:][::-1]):
      torch.testing.assert_close(repeated, initial)
    torch.testing.assert_close(torch.random.get_rng_state(), before_backward_rng)
    torch.testing.assert_close(checkpoint_loss, eager_loss)
    torch.testing.assert_close(checkpointed.student_weight.grad,
                               eager.student_weight.grad)

  def test_no_selected_tokens_has_zero_gradient_and_skips_critic(self):
    x0, valid = batch()
    model = make_model(8)
    model.q_xt = lambda x0, alpha, valid_tokens=None: x0
    loss, gradient, initial_calls = evaluate(model, x0, valid)
    self.assertEqual(loss.item(), 0.0)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    self.assertEqual(model._mmd_critic.denoiser.batch_sizes, [])
    self.assertGreater(len(model.forward_sizes), initial_calls)

  def test_widened_scoring_and_unconditional_input_keep_same_objective(self):
    x0, _ = batch()
    full = make_model(mmd_score_positions='all')
    chunked = make_model(8, mmd_score_positions='all')
    full_loss, full_gradient, _ = evaluate(full, x0, None)
    chunk_loss, chunk_gradient, _ = evaluate(chunked, x0, None)
    torch.testing.assert_close(chunk_loss, full_loss, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(chunk_gradient, full_gradient, rtol=1e-6, atol=1e-9)

  def test_naive_accumulation_is_not_the_preserved_objective(self):
    # Demonstrate the regression that motivated the physical/logical split.
    # Even linear-kernel reward (no bandwidth issue) depends on the logical
    # masked-token denominator when row mask counts differ.
    x0, valid = batch()
    full = make_model(mmd_kernel='linear', mmd_kernel_batch_size=1)
    small = make_model(mmd_kernel='linear', mmd_kernel_batch_size=1)
    _, full_gradient, _ = evaluate(full, x0, valid)
    with mock.patch.object(torch, 'multinomial', side_effect=deterministic_multinomial):
      for start in range(0, 32, 8):
        (small.generator_loss(x0[start:start + 8], valid[start:start + 8], 0)
         / 4).backward()
    small_gradient = torch.cat([small.student_weight.grad.flatten(),
                               small.student_bias.grad.flatten()])
    self.assertFalse(torch.allclose(full_gradient, small_gradient,
                                    rtol=1e-3, atol=1e-8))

  def test_invalid_forward_limits_fail(self):
    for limit in (0, -1, 1.5, True, '8'):
      with self.subTest(limit=limit), self.assertRaises(ValueError):
        make_model(limit)._mmd_forward_batch_size()

  def test_unsupported_objectives_fail_before_student_forward(self):
    x0, valid = batch()
    for overrides in (dict(dmd_weight=1.0), dict(mmd_reward_type='idlm'),
                      dict(mmd_reward_weight=0.0, dmd_weight=1.0)):
      model = make_model(8, **overrides)
      with self.subTest(overrides=overrides), self.assertRaises(ValueError):
        model.generator_loss(x0, valid, 0)
      self.assertEqual(model.forward_sizes, [])


if __name__ == '__main__':
  unittest.main(verbosity=2)
