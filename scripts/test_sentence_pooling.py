"""CPU regressions for AR sentence features from the last valid token.

Run from the repository root: python scripts/test_sentence_pooling.py
No pretrained weights, network access, or accelerator are needed.
"""
import ast
import math
import os
import sys
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmd_reward


def _expect_value_error(call):
  try:
    call()
  except ValueError:
    return
  raise AssertionError('invalid inputs/configuration were accepted')


def _endpoint_vectors(hidden, selected, sequence_mask=None):
  """Independent reference that does not call the production pooling helper."""
  result = torch.zeros(hidden.shape[0], hidden.shape[-1])
  for row in range(hidden.shape[0]):
    positions = (list(range(hidden.shape[1])) if sequence_mask is None else
                 [i for i, valid in enumerate(sequence_mask[row]) if valid])
    if positions and bool(selected[row].any()):
      result[row] = hidden[row, positions[-1]].float()
  return result


def _fixed_reference(real, fake, kernel, alpha=0.2):
  """Dense scalar reference for the fixed-real, K-fake sentence estimator."""
  values = []
  for x, ys in zip(real.double().tolist(), fake.double().tolist()):
    count = len(ys)

    def product(a, b):
      if kernel == 'linear':
        return sum(i * j for i, j in zip(a, b)) / len(a)
      return math.exp(-alpha * sum((i - j) ** 2 for i, j in zip(a, b)))

    def distance(a, b):
      return sum(math.sqrt((i - j) ** 2 + 1e-6) - 0.001
                 for i, j in zip(a, b)) / len(a)

    if kernel == 'pseudo_huber':
      estimate = sum(distance(x, y) for y in ys) / count
      if count > 1:
        estimate -= sum(distance(ys[i], ys[j]) for i in range(count)
                        for j in range(count) if i != j) / (
                          2 * count * (count - 1))
    else:
      fake_term = (product(ys[0], ys[0]) if count == 1 else
                   sum(product(ys[i], ys[j]) for i in range(count)
                       for j in range(count) if i != j) / (
                         count * (count - 1)))
      estimate = product(x, x) + fake_term - 2 * sum(
        product(x, y) for y in ys) / count
    values.append(-estimate)
  return torch.tensor(values, dtype=torch.float32)


def test_last_token_is_not_last_selected_or_mean():
  hidden = torch.tensor([[[1., 2.], [3., 4.], [7., 11.]]])
  selected = torch.tensor([[True, True, False]])
  last, active = mmd_reward.pool_sentence_features(
    hidden, selected, pooling='last_token')
  mean, _ = mmd_reward.pool_sentence_features(hidden, selected)
  torch.testing.assert_close(last, torch.tensor([[7., 11.]]))
  torch.testing.assert_close(mean, torch.tensor([[2., 3.]]))
  assert active.tolist() == [True]


def test_last_token_padding_holes_and_inactive_rows():
  hidden = torch.arange(50, dtype=torch.float32).reshape(5, 5, 2)
  selected = torch.tensor([[True, False, False, False, False],
                           [False, False, True, False, False],
                           [True, False, False, False, False],
                           [False, False, False, False, False],
                           [True, False, False, False, False]])
  valid = torch.tensor([[True, True, True, False, False],  # right padding
                        [False, False, True, True, True],  # left padding
                        [True, False, True, False, True],  # holes
                        [True, True, True, True, True],    # no scored positions
                        [False, False, False, False, False]])
  for dtype in (torch.float32, torch.bfloat16):
    values, active = mmd_reward.pool_sentence_features(
      hidden.to(dtype), selected, pooling='last_token', sequence_mask=valid)
    assert values.dtype == torch.float32
    torch.testing.assert_close(values, _endpoint_vectors(hidden, selected, valid))
    assert active.tolist() == [True, True, True, False, False]


def test_pooling_shape_guards_and_empty_sequences():
  hidden = torch.zeros(2, 3, 4)
  selected = torch.ones(2, 3, dtype=torch.bool)
  _expect_value_error(lambda: mmd_reward.pool_sentence_features(
    hidden, selected, pooling='bogus'))
  _expect_value_error(lambda: mmd_reward.pool_sentence_features(
    hidden[:, 0], selected, pooling='last_token'))
  _expect_value_error(lambda: mmd_reward.pool_sentence_features(
    hidden, selected[:, :2], pooling='last_token'))
  _expect_value_error(lambda: mmd_reward.pool_sentence_features(
    hidden, selected, pooling='last_token', sequence_mask=selected[:1]))
  value, active = mmd_reward.pool_sentence_features(
    torch.empty(2, 0, 4), torch.empty(2, 0, dtype=torch.bool), pooling='last_token')
  torch.testing.assert_close(value, torch.zeros(2, 4))
  assert active.tolist() == [False, False]


def _draw_fixture():
  rng = torch.Generator().manual_seed(912)
  clean = torch.randn(3, 5, 2, generator=rng)
  draws = torch.randn(4, 3, 5, 2, generator=rng)
  selected = torch.tensor([[True, True, False, False, False],
                           [False, True, False, False, False],
                           [False, False, False, False, False]])
  valid = torch.tensor([[True, True, True, True, False],
                        [False, True, True, True, True],
                        [True, True, True, True, True]])
  return clean, draws, selected, valid


def test_draws_k1_k2_k4_all_kernels_and_weighted_layers():
  clean, draws, selected, valid = _draw_fixture()
  for count in (1, 2, 4):
    hidden = torch.cat([clean, draws[:count].reshape(-1, 5, 2)])
    features = {0: hidden, 2: hidden * 1.75 + 0.2}
    for kernel in ('linear', 'rbf', 'pseudo_huber'):
      reward, active, layers = mmd_reward.summed_multi_draw_reward(
        features, batch=3, draw_count=count, length=5, selected=selected,
        indices=[0, 2], weights=[0.3, 1.25], kernel=kernel, alpha=0.2,
        block_size=1, level='sentence', sentence_pooling='last_token',
        sequence_mask=valid)
      assert active.tolist() == [True, True, False]
      for index in (0, 2):
        endpoints = _endpoint_vectors(
          features[index], selected.repeat(count + 1, 1),
          valid.repeat(count + 1, 1))
        expected = _fixed_reference(
          endpoints[:3], endpoints[3:].reshape(count, 3, 2).permute(1, 0, 2),
          kernel)
        expected[2] = 0
        torch.testing.assert_close(layers[index], expected, rtol=3e-5, atol=1e-6)
      torch.testing.assert_close(reward, 0.3 * layers[0] + 1.25 * layers[2])


def test_rows_last_token_preserves_existing_kernel_statistics():
  rng = torch.Generator().manual_seed(188)
  clean = torch.randn(4, 5, 3, generator=rng)
  generated = torch.randn(4, 5, 3, generator=rng)
  selected = torch.tensor([[True, False, False, False, False]]).repeat(4, 1)
  valid = torch.tensor([[True, True, True, False, False],
                        [False, True, True, True, True],
                        [True, True, True, True, False],
                        [True, False, True, False, True]])
  reference = _endpoint_vectors(clean, selected, valid).unsqueeze(1)
  fake = _endpoint_vectors(generated, selected, valid).unsqueeze(1)
  for kernel, count in (('linear', 1), ('linear', 2), ('rbf', 2), ('rbf', 4)):
    expected, expected_active = mmd_reward.paired_sequence_reward(
      reference, fake, torch.ones(4, 1, dtype=torch.bool), kernel=kernel,
      alpha=0.2, rows_of=count, level='sentence')
    actual, active, layers = mmd_reward.summed_paired_reward(
      {0: torch.cat([clean, generated])}, batch=4, selected=selected,
      indices=[0], weights=[1.0], kernel=kernel, alpha=0.2,
      rows_of=count, level='sentence', sentence_pooling='last_token',
      sequence_mask=valid)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(active, expected_active)
    torch.testing.assert_close(layers[0], expected)
  _expect_value_error(lambda: mmd_reward.paired_sequence_reward(
    clean, generated, selected, kernel='rbf', rows_of=1, level='sentence',
    sentence_pooling='last_token', sequence_mask=valid))
  _expect_value_error(lambda: mmd_reward.paired_sequence_reward(
    clean, generated, selected, kernel='pseudo_huber', level='sentence',
    sentence_pooling='last_token', sequence_mask=valid))


def test_default_and_explicit_mean_preserve_rewards_exactly():
  clean, draws, selected, valid = _draw_fixture()
  original, original_active = mmd_reward.pool_over_selected(clean, selected)
  for kwargs in ({}, {'pooling': 'mean'},
                 {'pooling': 'mean', 'sequence_mask': valid}):
    value, active = mmd_reward.pool_sentence_features(clean, selected, **kwargs)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    torch.testing.assert_close(active, original_active)
  for level in ('token', 'sentence'):
    for kernel in ('linear', 'rbf', 'pseudo_huber'):
      if level == 'token' and kernel == 'pseudo_huber':
        continue
      kwargs = dict(features={0: torch.cat([clean, draws[:2].reshape(-1, 5, 2)])},
                    batch=3, draw_count=2, length=5, selected=selected,
                    indices=[0], weights=[1.0], level=level, kernel=kernel,
                    alpha=0.2)
      old = mmd_reward.summed_multi_draw_reward(**kwargs)
      explicit = mmd_reward.summed_multi_draw_reward(
        **kwargs, sentence_pooling='mean', sequence_mask=valid)
      torch.testing.assert_close(old[0], explicit[0], rtol=0, atol=0)
      torch.testing.assert_close(old[1], explicit[1])
    for kernel in ('linear', 'rbf'):
      kwargs = dict(clean=clean, generated=draws[0], selected=selected,
                    kernel=kernel, level=level, rows_of=3, alpha=0.2)
      old = mmd_reward.paired_sequence_reward(**kwargs)
      explicit = mmd_reward.paired_sequence_reward(
        **kwargs, sentence_pooling='mean', sequence_mask=valid)
      torch.testing.assert_close(old[0], explicit[0], rtol=0, atol=0)
      torch.testing.assert_close(old[1], explicit[1])


def test_auto_rbf_bandwidth_uses_last_real_vectors_only():
  clean, draws, selected, valid = _draw_fixture()
  with mock.patch.object(mmd_reward, 'median_alpha', return_value=0.2) as bandwidth:
    mmd_reward.summed_multi_draw_reward(
      {0: torch.cat([clean, draws[:2].reshape(-1, 5, 2)])},
      batch=3, draw_count=2, length=5, selected=selected,
      indices=[0], weights=[1.0], kernel='rbf', level='sentence',
      sentence_pooling='last_token', sequence_mask=valid)
    real, active = bandwidth.call_args.args[:2]
    torch.testing.assert_close(real[:, 0], _endpoint_vectors(clean, selected, valid))
    assert active.tolist() == [[True], [True], [False]]


class _CumulativeBlock(torch.nn.Module):
  def forward(self, hidden):
    return hidden.cumsum(dim=1)


class _ScaleBlock(torch.nn.Module):
  def forward(self, hidden):
    return hidden * 1.75


class _TinyCausalModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    table = torch.arange(32, dtype=torch.float32).reshape(16, 2).sin()
    self.embedding = torch.nn.Embedding.from_pretrained(table, freeze=True)
    self.blocks = torch.nn.ModuleList([_CumulativeBlock(), _ScaleBlock()])
    self.config = SimpleNamespace(vocab_size=16)
    self.seen_attention = []

  def forward(self, tokens=None, sigma=None, input_ids=None, attention_mask=None):
    del sigma
    tokens = tokens if input_ids is None else input_ids
    self.seen_attention.append(attention_mask)
    hidden = self.embedding(tokens)
    if attention_mask is not None:
      hidden = hidden * attention_mask.unsqueeze(-1)
    for block in self.blocks:
      hidden = block(hidden)
    return hidden


def _critic(generic=False):
  model = _TinyCausalModel()
  if generic:
    critic = mmd_reward.FrozenARFeatures.__new__(mmd_reward.FrozenARFeatures)
    critic.model = model
    critic.vocab_size = 16
    critic.device = 'cpu'
    critic._resolve_layers(model, [0, 1], [0.3, 1.25])
  else:
    critic = mmd_reward.FrozenBackboneFeatures(
      model, mask_index=15, layer_index=[0, 1], noise_level=0.0,
      layer_weights=[0.3, 1.25])
  critic.kernel_level = 'sentence'
  critic.sentence_pooling = 'last_token'
  return critic, model


def test_native_and_generic_wrappers_use_contextual_last_token_features():
  clean = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
  first = torch.tensor([[1, 2, 2, 3], [4, 6, 6, 7]])
  second = torch.tensor([[2, 3, 2, 3], [4, 7, 6, 7]])
  selected = torch.tensor([[True, True, False, False],
                           [False, True, False, False]])
  valid = torch.tensor([[True, True, True, False], [False, True, True, True]])
  for generic in (False, True):
    critic, model = _critic(generic)
    for kernel in ('linear', 'rbf', 'pseudo_huber'):
      critic.kernel = kernel
      critic.rbf_alpha = 0.2
      for count in (1, 2):
        draws = torch.stack([first, second])[:count]
        actual, active, layers = critic.multi_draw_reward(
          clean, draws, selected, eos_id=0, return_per_layer=True,
          sequence_mask=valid)
        embeddings = model.embedding(torch.cat([clean, draws.reshape(-1, 4)]))
        cumulative = embeddings.cumsum(dim=1)
        endpoints = _endpoint_vectors(cumulative, selected.repeat(count + 1, 1),
                                     valid.repeat(count + 1, 1))
        for index, scale in ((0, 1.0), (1, 1.75)):
          expected = _fixed_reference(
            endpoints[:2] * scale,
            endpoints[2:].reshape(count, 2, 2).permute(1, 0, 2) * scale,
            kernel)
          torch.testing.assert_close(layers[index], expected, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(actual, 0.3 * layers[0] + 1.25 * layers[1])
        assert active.all() and not actual.requires_grad
        assert actual.abs().sum() > 0  # Last token IDs themselves never changed.
    critic.kernel = 'linear'
    pair, _, pair_layers = critic.paired_reward(
      clean, first, selected, eos_id=0, sequence_mask=valid, return_per_layer=True)
    assert pair.abs().sum() > 0
    torch.testing.assert_close(pair, 0.3 * pair_layers[0] + 1.25 * pair_layers[1])
    assert all(mask is None for mask in model.seen_attention)


def test_generic_ar_attention_mask_fallback_and_explicit_endpoint_override():
  critic, model = _critic(generic=True)
  clean = torch.tensor([[0, 1, 2, 3]])
  generated = torch.tensor([[1, 1, 2, 3]])
  selected = torch.tensor([[True, False, False, False]])
  attention = torch.tensor([[True, True, True, False]])
  endpoint = torch.tensor([[True, True, False, False]])
  for sequence_mask in (None, endpoint):
    pair, _ = critic.paired_reward(
      clean, generated, selected, eos_id=0, attention_mask=attention,
      sequence_mask=sequence_mask)
    hidden = model.embedding(torch.cat([clean, generated]))
    hidden = (hidden * attention.repeat(2, 1).unsqueeze(-1)).cumsum(1)
    vectors = _endpoint_vectors(
      hidden, selected.repeat(2, 1),
      (attention if sequence_mask is None else sequence_mask).repeat(2, 1))
    delta = vectors[1:] - vectors[:1]
    expected = (0.3 * mmd_reward.pseudo_huber_reward(delta)
                + 1.25 * mmd_reward.pseudo_huber_reward(delta * 1.75))
    torch.testing.assert_close(pair, expected)
    torch.testing.assert_close(model.seen_attention[-1], attention.repeat(2, 1))


def test_reinforce_forwards_endpoint_mask_without_widening_gradients():
  torch.manual_seed(218)
  clean = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
  selected = torch.tensor([[True, False, False, False],
                           [False, True, False, False]])
  valid = torch.tensor([[True, True, True, False], [False, True, True, True]])
  for source in ('rows', 'draws'):
    critic, _ = _critic()
    probabilities = torch.full((2, 4, 16), 1 / 16, requires_grad=True)
    method = 'paired_reward' if source == 'rows' else 'multi_draw_reward'
    with mock.patch.object(critic, method, wraps=getattr(critic, method)) as score:
      loss, _ = mmd_reward.mmd_reinforce_loss(
        probabilities, clean, selected, critic, group_size=2, eos_id=0,
        kernel_source=source, kernel_draws=2, baseline='none', sequence_mask=valid)
      assert score.call_count == 2
      for call in score.call_args_list:
        torch.testing.assert_close(call.kwargs['sequence_mask'], valid)
      loss.backward()
      assert torch.isfinite(loss)
      assert probabilities.grad[selected].abs().sum() > 0
      assert (probabilities.grad[~selected] == 0).all()


class _Config(SimpleNamespace):
  def get(self, key, default=None):
    return getattr(self, key, default)


def _configured_model(**overrides):
  settings = dict(mmd_reward_type='feature', mmd_kernel='linear',
                  mmd_kernel_source='draws', mmd_kernel_level='sentence',
                  mmd_sentence_pooling='last_token', mmd_feature_source='ar',
                  mmd_ar_model='stub-owt-ar', mmd_feature_layer=[0, 1],
                  mmd_feature_layer_weights=[0.3, 1.25], mmd_noise_level=0.25,
                  mmd_kernel_batch_size=4, mmd_group_size=3, mmd_reward_weight=0.7)
  settings.update(overrides)
  return SimpleNamespace(
    config=SimpleNamespace(adversarial_distill=_Config(**settings)),
    device=torch.device('cpu'), mask_index=15, teacher=object())


def _critic_getter():
  path = os.path.join(os.path.dirname(mmd_reward.__file__), 'algo.py')
  with open(path, encoding='utf-8') as source:
    tree = ast.parse(source.read(), filename=path)
  nodes = [node for node in ast.walk(tree)
           if isinstance(node, ast.FunctionDef) and node.name == '_mmd_critic']
  assert len(nodes) == 1
  nodes[0].decorator_list = []
  namespace = {'IdlmRatioReward': mock.Mock(
    side_effect=AssertionError('invalid last-token config reached IDLM loading'))}
  exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
  return namespace['_mmd_critic']


def test_algo_last_token_gate_precedes_all_model_loading():
  getter = _critic_getter()
  invalid = [dict(mmd_sentence_pooling='bogus'), dict(mmd_kernel_level='token'),
             dict(mmd_feature_source='diffusion'), dict(mmd_reward_type='idlm')]
  with mock.patch.object(mmd_reward, 'load_owt_ar_backbone') as loader, \
       mock.patch.object(mmd_reward, 'FrozenBackboneFeatures') as constructor:
    for options in invalid:
      _expect_value_error(lambda: getter(_configured_model(**options)))
    loader.assert_not_called()
    constructor.assert_not_called()


def test_algo_propagates_last_token_for_supported_sources_and_kernels():
  getter = _critic_getter()
  for source in ('rows', 'draws'):
    for kernel in ('linear', 'rbf', 'pseudo_huber'):
      if source == 'rows' and kernel == 'pseudo_huber':
        continue
      model = _configured_model(mmd_kernel_source=source, mmd_kernel=kernel)
      critic = SimpleNamespace(layer_description='0 + 1')
      with mock.patch.object(mmd_reward, 'load_owt_ar_backbone'), \
           mock.patch.object(mmd_reward, 'FrozenBackboneFeatures', return_value=critic):
        assert getter(model) is critic
        assert getter(model) is critic
        assert critic.sentence_pooling == 'last_token'
        assert critic.kernel_level == 'sentence'
        assert critic.kernel == kernel


def test_algo_term_passes_full_packed_sequence_not_written_mask():
  path = os.path.join(os.path.dirname(mmd_reward.__file__), 'algo.py')
  with open(path, encoding='utf-8') as source:
    tree = ast.parse(source.read(), filename=path)
  names = ('_mmd_content_mask', '_mmd_term')
  methods = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in names]
  assert len(methods) == 2
  namespace = {'torch': torch, 'F': torch.nn.functional}
  exec(compile(ast.Module(body=methods, type_ignores=[]), path, 'exec'), namespace)
  # EOS=0 occurs inside packed OWT rows; it must not truncate the endpoint.
  clean = torch.tensor([[0, 2, 0, 3], [0, 4, 5, 0]])
  written = torch.tensor([[False, True, False, False],
                          [False, False, True, False]])
  latent = torch.where(written, 15, clean)
  probabilities = torch.full((2, 4, 16), 1 / 16, requires_grad=True)
  for pooling in ('mean', 'last_token'):
    for score_positions in ('written', 'all'):
      model = _configured_model(mmd_sentence_pooling=pooling,
                                mmd_score_positions=score_positions)
      model.tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=14)
      model._mmd_critic = object()
      model._mmd_content_mask = MethodType(namespace['_mmd_content_mask'], model)
      model.log_dict = mock.Mock()
      with mock.patch.object(mmd_reward, 'mmd_reinforce_loss',
                             return_value=(probabilities.new_zeros(()), {})) as loss:
        namespace['_mmd_term'](model, clean, None, probabilities, latent)
        args, kwargs = loss.call_args
        torch.testing.assert_close(args[2], written)
        torch.testing.assert_close(
          kwargs['scored'], written if score_positions == 'written' else
          torch.ones_like(written))
        if pooling == 'last_token':
          torch.testing.assert_close(kwargs['sequence_mask'], torch.ones_like(written))
        else:
          assert kwargs['sequence_mask'] is None


def main():
  for name, fn in sorted(globals().items()):
    if name.startswith('test_'):
      fn()
      print(f'PASS  {name}')
  print('\nall sentence-pooling MMD tests passed')


if __name__ == '__main__':
  main()
