"""CPU regressions for the distance-induced pseudo-Huber sentence MMD.

Run from the repository root: python scripts/test_pseudo_huber_mmd.py
No pretrained weights or accelerator are needed.
"""
import ast
import itertools
import math
import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmd_reward


def _distance(x, y):
  """Independent scalar reference, deliberately outside the tensor helper."""
  return sum(math.sqrt((a - b) ** 2 + 1e-6) - 0.001
             for a, b in zip(x, y)) / len(x)


def _dense_reference(real, fake):
  result = []
  for x, ys in zip(real.double().tolist(), fake.double().tolist()):
    count = len(ys)
    cross = sum(_distance(x, y) for y in ys) / count
    if count > 1:
      # The distance-induced kernel uses HALF the ordered fake-pair mean.
      ordered = sum(_distance(ys[i], ys[j])
                    for i in range(count) for j in range(count) if i != j)
      cross -= ordered / (2 * count * (count - 1))
    result.append(cross)
  return torch.tensor(result, dtype=torch.float32)


def _expect_value_error(call):
  try:
    call()
  except ValueError:
    return
  raise AssertionError('invalid inputs/configuration were accepted')


def test_k1_recovers_historical_paired_huber_in_fp32_and_bf16():
  generator = torch.Generator().manual_seed(91)
  selected = torch.tensor([[True, False, True, True],
                           [False, False, False, False],
                           [False, True, False, True]])
  for dtype in (torch.float32, torch.bfloat16):
    clean = torch.randn(3, 4, 7, generator=generator).to(dtype)
    generated = torch.randn(3, 4, 7, generator=generator).to(dtype)
    historical, old_active = mmd_reward.paired_sequence_reward(
      clean, generated, selected, kernel='linear', rows_of=1, level='token')
    new, active, per_layer = mmd_reward.summed_multi_draw_reward(
      {0: torch.cat([clean, generated])}, batch=3, draw_count=1, length=4,
      selected=selected, indices=[0], weights=[1.0],
      kernel='pseudo_huber', level='sentence')
    assert new.dtype == torch.float32
    torch.testing.assert_close(new, historical, rtol=0, atol=0)
    torch.testing.assert_close(active, old_active)
    torch.testing.assert_close(per_layer[0], historical, rtol=0, atol=0)


def test_k2_and_k4_match_independent_dense_distance_formula():
  real = torch.tensor([[0.2, -1.0, 2.0], [-0.1, 3.0, 0.7]])
  fake = torch.tensor([[[1.0, 2.0, 3.0], [-0.5, 1.0, -2.0],
                         [0.2, -1.0, 2.0], [2.0, 0.0, -0.7]],
                        [[0.1, 2.0, 0.8], [-1.0, 1.0, 0.0],
                         [2.0, 1.0, 0.2], [-0.2, 4.0, 0.5]]])
  for count in (2, 4):
    actual = mmd_reward.pseudo_huber_mmd2_fixed_real(real, fake[:, :count])
    torch.testing.assert_close(
      actual, _dense_reference(real, fake[:, :count]), rtol=2e-6, atol=2e-7)


def test_exact_finite_distribution_expectation_is_unbiased_for_k2():
  # Enumerating all ordered samples with their probabilities tests the
  # statistical target itself, including collisions of independent draws.
  support = [[-1.0, 0.4], [0.1, 0.0], [1.7, -0.8]]
  probabilities = [0.2, 0.3, 0.5]
  real = [0.3, -0.2]
  population_cross = sum(p * _distance(real, y)
                         for p, y in zip(probabilities, support))
  population_fake = sum(
    probabilities[i] * probabilities[j] * _distance(support[i], support[j])
    for i, j in itertools.product(range(3), repeat=2))
  population_mmd = population_cross - 0.5 * population_fake
  expected_estimate = 0.0
  for i, j in itertools.product(range(3), repeat=2):
    estimate = mmd_reward.pseudo_huber_mmd2_fixed_real(
      torch.tensor([real]), torch.tensor([[support[i], support[j]]]))
    expected_estimate += probabilities[i] * probabilities[j] * estimate.item()
  assert math.isclose(expected_estimate, population_mmd, rel_tol=2e-6, abs_tol=1e-7)
  # K=1 intentionally retains the historical biased point-to-sample distance.
  assert population_cross > population_mmd + 0.1


def test_negative_estimate_survives_helper_and_reward():
  real = torch.zeros(1, 1)
  fake = torch.tensor([[[-0.0005], [0.0005]]])
  estimate = mmd_reward.pseudo_huber_mmd2_fixed_real(real, fake)
  torch.testing.assert_close(estimate, _dense_reference(real, fake),
                             rtol=2e-6, atol=1e-10)
  assert estimate.item() < 0.0
  reward, _, _ = mmd_reward.summed_multi_draw_reward(
    {0: torch.cat([real[:, None], fake.permute(1, 0, 2)])},
    batch=1, draw_count=2, length=1, selected=torch.ones(1, 1, dtype=torch.bool),
    indices=[0], weights=[1.0], kernel='pseudo_huber', level='sentence')
  torch.testing.assert_close(reward, -estimate, rtol=0, atol=0)
  assert reward.item() > 0.0


def test_selected_pooling_inactive_rows_and_layer_weights():
  clean = torch.tensor([[[-1.0], [1.0], [999.0]],
                        [[7.0], [8.0], [9.0]]])
  first = torch.tensor([[[0.0], [2.0], [-500.0]],
                        [[1.0], [2.0], [3.0]]])
  second = torch.tensor([[[1.0], [3.0], [700.0]],
                         [[4.0], [5.0], [6.0]]])
  hidden = torch.cat([clean, first, second])
  selected = torch.tensor([[True, True, False], [False, False, False]])
  reward, active, per_layer = mmd_reward.summed_multi_draw_reward(
    {0: hidden, 2: hidden * 2 + 0.25}, batch=2, draw_count=2, length=3,
    selected=selected, indices=[0, 2], weights=[0.3, 1.25],
    kernel='pseudo_huber', level='sentence')
  expected_0 = -_dense_reference(
    torch.tensor([[0.0]]), torch.tensor([[[1.0], [2.0]]]))[0]
  expected_2 = -_dense_reference(
    torch.tensor([[0.25]]), torch.tensor([[[2.25], [4.25]]]))[0]
  torch.testing.assert_close(per_layer[0], torch.tensor([expected_0, 0.0]))
  torch.testing.assert_close(per_layer[2], torch.tensor([expected_2, 0.0]))
  torch.testing.assert_close(reward, 0.3 * per_layer[0] + 1.25 * per_layer[2])
  assert active.tolist() == [True, False]


class _ScaleBlock(torch.nn.Module):
  def __init__(self, scale):
    super().__init__()
    self.scale = scale

  def forward(self, hidden):
    return hidden * self.scale


class _TinyBackbone(torch.nn.Module):
  def __init__(self):
    super().__init__()
    table = torch.arange(32, dtype=torch.float32).reshape(8, 4).sin()
    self.embedding = torch.nn.Embedding.from_pretrained(table, freeze=True)
    self.blocks = torch.nn.ModuleList([_ScaleBlock(1.0), _ScaleBlock(1.75)])

  def forward(self, tokens, sigma):
    assert sigma.shape[0] == tokens.shape[0]
    hidden = self.embedding(tokens)
    for block in self.blocks:
      hidden = block(hidden)
    return hidden


def _critic(kernel, level='sentence'):
  critic = mmd_reward.FrozenBackboneFeatures(
    _TinyBackbone(), mask_index=7, layer_index=[0, 1],
    layer_weights=[0.3, 1.25], noise_level=0.0)
  critic.kernel = kernel
  critic.kernel_level = level
  return critic


def test_backbone_wrapper_forwards_multidraw_and_layer_rewards():
  critic = _critic('pseudo_huber')
  clean = torch.tensor([[0, 1, 2], [2, 3, 4]])
  draws = torch.tensor([[[0, 2, 2], [2, 4, 4]],
                         [[1, 1, 2], [3, 3, 4]]])
  selected = torch.tensor([[True, True, False], [False, False, False]])
  reward, active, layers = critic.multi_draw_reward(
    clean, draws, selected, eos_id=None, sigma=torch.zeros(2),
    return_per_layer=True)
  assert active.tolist() == [True, False]
  # Independently form sentence embeddings from the known feature lookup table.
  lookup = critic.denoiser.embedding.weight
  reference = lookup[clean[0, :2]].mean(dim=0).unsqueeze(0)
  generated = lookup[draws[:, 0, :2]].mean(dim=1).unsqueeze(0)
  for index, scale in ((0, 1.0), (1, 1.75)):
    expected = -_dense_reference(reference * scale, generated * scale)[0]
    torch.testing.assert_close(layers[index], torch.tensor([expected, 0.0]))
  torch.testing.assert_close(reward, 0.3 * layers[0] + 1.25 * layers[1])
  assert not reward.requires_grad


def test_k1_reinforce_loss_and_gradient_match_historical_rows():
  clean = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 1], [1, 0, 2, 4]])
  written = torch.tensor([[True, True, False, True],
                          [False, True, True, False],
                          [False, False, False, False]])
  generator = torch.Generator().manual_seed(20)
  values = torch.rand(3, 4, 8, generator=generator) + 0.1
  values /= values.sum(dim=-1, keepdim=True)

  def run(kernel, source, level):
    probabilities = values.clone().requires_grad_(True)
    critic = _critic(kernel, level)
    torch.manual_seed(512)
    loss, stats = mmd_reward.mmd_reinforce_loss(
      probabilities, clean, written, critic, group_size=3, eos_id=None,
      reward_weight=0.7, baseline='group', kernel_source=source, kernel_draws=1)
    loss.backward()
    return loss.detach(), probabilities.grad, stats

  old_loss, old_grad, old_stats = run('linear', 'rows', 'token')
  loss, grad, stats = run('pseudo_huber', 'draws', 'sentence')
  torch.testing.assert_close(loss, old_loss, rtol=0, atol=0)
  torch.testing.assert_close(grad, old_grad, rtol=0, atol=0)
  assert stats.keys() == old_stats.keys()
  for name in stats:
    torch.testing.assert_close(stats[name], old_stats[name], rtol=0, atol=0)
  assert grad[written].abs().sum() > 0
  assert (grad[~written] == 0).all()


def test_invalid_shapes_and_empty_draws_are_rejected():
  cases = [(torch.zeros(2), torch.zeros(1, 2, 2)),
           (torch.zeros(1, 2), torch.zeros(1, 2)),
           (torch.zeros(2, 2), torch.zeros(1, 2, 2)),
           (torch.zeros(1, 3), torch.zeros(1, 2, 2)),
           (torch.zeros(1, 2), torch.zeros(1, 0, 2))]
  for real, fake in cases:
    _expect_value_error(lambda: mmd_reward.pseudo_huber_mmd2_fixed_real(real, fake))


def test_pseudo_huber_kernel_rejects_token_and_rows_modes():
  features = {0: torch.zeros(2, 3, 2)}
  selected = torch.ones(1, 3, dtype=torch.bool)
  _expect_value_error(lambda: mmd_reward.summed_multi_draw_reward(
    features, batch=1, draw_count=1, length=3, selected=selected,
    indices=[0], weights=[1.0], kernel='pseudo_huber', level='token'))
  critic = _critic('pseudo_huber')
  probabilities = torch.full((1, 3, 8), 1 / 8, requires_grad=True)
  _expect_value_error(lambda: mmd_reward.mmd_reinforce_loss(
    probabilities, torch.zeros(1, 3, dtype=torch.long), selected, critic,
    group_size=2, eos_id=None, kernel_source='rows'))


def _critic_getter_without_training_imports():
  # Execute the actual property body while excluding unrelated Lightning,
  # tokenizer and training-model imports from this CPU configuration test.
  path = os.path.join(os.path.dirname(mmd_reward.__file__), 'algo.py')
  with open(path, encoding='utf-8') as source:
    tree = ast.parse(source.read(), filename=path)
  getters = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == '_mmd_critic']
  assert len(getters) == 1
  getters[0].decorator_list = []
  namespace = {'IdlmRatioReward': mock.Mock(
    side_effect=AssertionError('invalid configuration reached IDLM construction'))}
  module = ast.Module(body=getters, type_ignores=[])
  exec(compile(module, path, 'exec'), namespace)
  return namespace['_mmd_critic'], namespace['IdlmRatioReward']


class _Config(SimpleNamespace):
  def get(self, key, default=None):
    return getattr(self, key, default)


def _configured_model(**overrides):
  settings = dict(mmd_reward_type='feature', mmd_kernel='pseudo_huber',
                  mmd_kernel_source='draws', mmd_kernel_level='sentence',
                  mmd_feature_source='ar', mmd_ar_model='stub-owt-ar',
                  mmd_feature_layer=[0, 1], mmd_feature_layer_weights=[0.3, 1.25],
                  mmd_noise_level=0.25, mmd_rbf_alpha=0.13, mmd_rbf_block_size=4,
                  mmd_kernel_batch_size=4, mmd_group_size=3, mmd_reward_weight=0.7)
  settings.update(overrides)
  return SimpleNamespace(
    config=SimpleNamespace(adversarial_distill=_Config(**settings)),
    device=torch.device('cpu'), mask_index=7, teacher=object())


def test_algo_rejects_invalid_kernel_config_before_loading_models():
  getter, idlm_constructor = _critic_getter_without_training_imports()
  invalid = [dict(mmd_kernel='bogus'), dict(mmd_kernel_level='bogus'),
             dict(mmd_kernel_source='bogus'), dict(mmd_kernel_source='rows'),
             dict(mmd_kernel_level='token'), dict(mmd_reward_type='idlm')]
  with mock.patch.object(mmd_reward, 'load_owt_ar_backbone') as loader, \
       mock.patch.object(mmd_reward, 'FrozenBackboneFeatures') as constructor:
    for overrides in invalid:
      _expect_value_error(lambda: getter(_configured_model(**overrides)))
    loader.assert_not_called()
    constructor.assert_not_called()
    idlm_constructor.assert_not_called()


def test_algo_propagates_valid_pseudo_huber_config_and_caches_critic():
  getter, idlm_constructor = _critic_getter_without_training_imports()
  for source in ('ar', 'diffusion'):
    model = _configured_model(mmd_feature_source=source)
    backbone = object()
    critic = SimpleNamespace(layer_description='0 + 1')
    with mock.patch.object(mmd_reward, 'load_owt_ar_backbone',
                           return_value=backbone) as loader, \
         mock.patch.object(mmd_reward, 'FrozenBackboneFeatures',
                           return_value=critic) as constructor:
      assert getter(model) is critic
      assert getter(model) is critic
      if source == 'ar':
        loader.assert_called_once_with('stub-owt-ar', model.device,
                                       weights_dir=None)
        expected_backbone, expected_noise = backbone, 0.0
      else:
        loader.assert_not_called()
        expected_backbone, expected_noise = model.teacher, 0.25
      constructor.assert_called_once_with(
        expected_backbone, 7, layer_index=[0, 1],
        layer_weights=[0.3, 1.25], noise_level=expected_noise)
      assert critic.kernel == 'pseudo_huber'
      assert critic.kernel_level == 'sentence'
      assert critic.kernel_batch_size == 4
      assert critic.rbf_alpha == 0.13
      assert critic.rbf_block_size == 4
    idlm_constructor.assert_not_called()


def main():
  for name, fn in sorted(globals().items()):
    if name.startswith('test_'):
      fn()
      print(f'PASS  {name}')
  print('\nall pseudo-Huber MMD tests passed')


if __name__ == '__main__':
  main()
