"""CPU configuration/routing checks for aligned masked feature MSE.

Run from the repository root: python scripts/test_feature_mse_config.py
Executes the actual training methods without importing Lightning or weights.
"""
import ast
from contextlib import contextmanager, ExitStack
import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmd_reward


_IS_OWT = hasattr(mmd_reward, 'load_owt_ar_backbone')


class _Config(SimpleNamespace):
  def get(self, key, default=None):
    return getattr(self, key, default)


def _training_probe():
  path = os.path.join(os.path.dirname(mmd_reward.__file__), 'algo.py')
  with open(path, encoding='utf-8') as source:
    tree = ast.parse(source.read(), filename=path)
  names = {'_mmd_critic', '_needs_fake_phase', '_mmd_content_mask', '_mmd_term'}
  methods = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in names]
  assert len(methods) == len(names)
  probe = ast.ClassDef(name='_Probe', bases=[], keywords=[], body=methods,
                       decorator_list=[])
  namespace = {'torch': torch, 'F': F, 'IdlmRatioReward': mock.Mock(
    side_effect=AssertionError('feature reward constructed an IDLM critic'))}
  module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
  exec(compile(module, path, 'exec'), namespace)
  return namespace['_Probe']()


def _model(**overrides):
  settings = dict(
    mmd_reward_type='feature_mse', mmd_feature_source='diffusion',
    mmd_ar_model='stub-ar' if _IS_OWT else 'stub-ar.ckpt',
    mmd_feature_layer=[0, 3], mmd_feature_layer_weights=[0.25, 0.75],
    mmd_noise_level=0.0, mmd_score_positions='written',
    mmd_scored_positions='answer', mmd_kernel='linear',
    mmd_kernel_level='token', mmd_kernel_source='rows',
    mmd_kernel_batch_size=4, mmd_group_size=3, mmd_reward_weight=0.7,
    mmd_baseline='group', dmd_weight=0.0)
  settings.update(overrides)
  model = _training_probe()
  model.config = SimpleNamespace(adversarial_distill=_Config(**settings),
                                 model_weights_dir='/unused/offline-weights')
  model.device = torch.device('cpu')
  model.teacher = object()
  model.mask_index = 10
  model.tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=9)
  model.log_dict = mock.Mock()
  return model


@contextmanager
def _feature_stubs():
  critic = SimpleNamespace(layer_description='0 + 3')
  backbone = object()
  loader_name = 'load_owt_ar_backbone' if _IS_OWT else 'load_ar_backbone'
  with ExitStack() as stack:
    loader = stack.enter_context(mock.patch.object(
      mmd_reward, loader_name, return_value=backbone))
    native = stack.enter_context(mock.patch.object(
      mmd_reward, 'FrozenBackboneFeatures', return_value=critic))
    # OWT always uses its native AR backbone, while TinyGSM also supports HF AR.
    hf = stack.enter_context(mock.patch.object(
      mmd_reward, 'FrozenARFeatures', return_value=critic))
    yield SimpleNamespace(critic=critic, backbone=backbone, loader=loader,
                           native=native, hf=hf)


def _expect_value_error(call):
  try:
    call()
  except ValueError:
    return
  raise AssertionError('invalid configuration was accepted')


def test_diffusion_and_native_ar_keep_layer_settings_and_cache():
  for source in ('diffusion', 'ar'):
    for kernel_source, expected_batch in (('rows', 1), ('draws', 4)):
      model = _model(mmd_feature_source=source,
                     mmd_kernel_source=kernel_source)
      with _feature_stubs() as stubs:
        assert model._mmd_critic is stubs.critic
        assert model._mmd_critic is stubs.critic
        cfg = model.config.adversarial_distill
        if source == 'diffusion':
          stubs.loader.assert_not_called()
          expected_backbone = model.teacher
        else:
          expected_backbone = stubs.backbone
          if _IS_OWT:
            stubs.loader.assert_called_once_with(
              cfg.mmd_ar_model, model.device,
              weights_dir=model.config.model_weights_dir)
          else:
            stubs.loader.assert_called_once_with(
              cfg.mmd_ar_model, model.config, model.tokenizer, model.device)
        stubs.native.assert_called_once_with(
          expected_backbone, model.mask_index, layer_index=[0, 3],
          layer_weights=[0.25, 0.75], noise_level=0.0)
        stubs.hf.assert_not_called()
        assert stubs.critic.reward_type == 'feature_mse'
        assert stubs.critic.kernel_batch_size == expected_batch


def test_tinygsm_hf_ar_keeps_offline_weights_path():
  if _IS_OWT:
    return
  model = _model(mmd_feature_source='ar', mmd_ar_model='stub/hf-ar')
  with _feature_stubs() as stubs:
    assert model._mmd_critic is stubs.critic
    stubs.loader.assert_not_called()
    stubs.native.assert_not_called()
    stubs.hf.assert_called_once_with(
      'stub/hf-ar', layer_index=[0, 3], layer_weights=[0.25, 0.75],
      device=model.device, weights_dir=model.config.model_weights_dir)
    assert stubs.critic.reward_type == 'feature_mse'


def test_mmd_only_settings_do_not_change_feature_mse_mode():
  # A previous RBF/cross-sentence or pooled-Huber command only needs the new
  # reward selector. These unused settings must not impose MMD constraints.
  for kernel in ('rbf', 'pseudo_huber'):
    model = _model(
      mmd_kernel=kernel, mmd_kernel_level='sentence',
      mmd_kernel_source='rows', mmd_kernel_batch_size=1,
      mmd_rbf_distance='unused-distance', mmd_rbf_huber_delta=-1.0,
      mmd_rbf_exclude_same_sentence=True, mmd_sentence_pooling='last_token')
    with _feature_stubs() as stubs:
      critic = model._mmd_critic
      assert critic is stubs.critic
      assert critic.reward_type == 'feature_mse'
      assert critic.kernel == 'linear'
      assert critic.kernel_level == 'token'
      assert not critic.rbf_exclude_same_sentence
      if _IS_OWT:
        assert critic.sentence_pooling == 'mean'


def test_invalid_reward_source_and_scoring_fail_before_model_loading():
  invalid = [dict(mmd_reward_type='not-a-reward'),
             dict(mmd_feature_source='not-a-model'),
             dict(mmd_kernel_source='not-a-source'),
             dict(mmd_score_positions='all'),
             dict(mmd_score_positions='not-a-mask')]
  with _feature_stubs() as stubs:
    for overrides in invalid:
      _expect_value_error(lambda: _model(**overrides)._mmd_critic)
    stubs.loader.assert_not_called()
    stubs.native.assert_not_called()
    stubs.hf.assert_not_called()


def test_existing_feature_reward_keeps_its_mmd_validation():
  invalid = [dict(mmd_kernel='not-a-kernel'),
             dict(mmd_kernel_level='not-a-level'),
             dict(mmd_kernel='pseudo_huber', mmd_kernel_source='rows'),
             dict(mmd_kernel='rbf', mmd_rbf_huber_delta=0,
                  mmd_rbf_distance='pseudo_huber'),
             dict(mmd_kernel='rbf', mmd_rbf_exclude_same_sentence=True)]
  if _IS_OWT:
    invalid.append(dict(mmd_sentence_pooling='last_token'))
  with _feature_stubs() as stubs:
    for overrides in invalid:
      _expect_value_error(
        lambda: _model(mmd_reward_type='feature', **overrides)._mmd_critic)
    stubs.loader.assert_not_called()
    stubs.native.assert_not_called()
    stubs.hf.assert_not_called()
  model = _model(mmd_reward_type='feature', mmd_kernel='rbf')
  with _feature_stubs() as stubs:
    assert model._mmd_critic is stubs.critic
    assert stubs.critic.reward_type == 'feature'
    assert stubs.critic.kernel == 'rbf'
    assert stubs.critic.kernel_batch_size == 4


def test_feature_mse_does_not_train_fake_unless_dmd_is_enabled():
  for kind, dmd, expected in [('feature_mse', 0.0, False),
                              ('feature_mse', 1.0, True),
                              ('feature', 0.0, False),
                              ('idlm', 0.0, True)]:
    model = _model(mmd_reward_type=kind, dmd_weight=dmd)
    assert model._needs_fake_phase() is expected


def test_training_term_scores_only_masked_answer_content():
  model = _model(mmd_kernel_source='draws')
  x0 = torch.tensor([[0, 4, 5, 6, 0, 9, 9, 9],
                     [0, 4, 5, 6, 0, 9, 9, 9]])
  valid = torch.tensor([[False, False, True, True, True, True, True, True]] * 2)
  xt = x0.clone()
  xt[0] = model.mask_index
  xt[0, 3] = x0[0, 3]
  probabilities = torch.full((2, 8, 11), 1 / 11)
  expected = torch.tensor([[False, False, True, False, True, False, False, False],
                           [False] * 8])
  expected_loss = probabilities.new_tensor(0.125)
  with _feature_stubs(), mock.patch.object(
      mmd_reward, 'mmd_reinforce_loss', return_value=(expected_loss, {})) as loss:
    assert model._mmd_term(x0, valid, probabilities, xt) is expected_loss
    args, kwargs = loss.call_args
    assert args[0] is probabilities
    assert args[1] is x0
    torch.testing.assert_close(args[2], expected)
    torch.testing.assert_close(kwargs['scored'], expected)
    assert args[3].reward_type == 'feature_mse'
    assert kwargs['group_size'] == 3
    assert kwargs['reward_weight'] == 0.7
    assert kwargs['baseline'] == 'group'
    assert kwargs['kernel_source'] == 'draws'
    assert kwargs['kernel_draws'] == 4
    model.log_dict.assert_called_once()


def test_no_masked_tokens_skips_reward_and_feature_loading():
  model = _model()
  x0 = torch.tensor([[0, 4, 5, 6]])
  probabilities = torch.full((1, 4, 11), 1 / 11)
  with _feature_stubs() as stubs, mock.patch.object(
      mmd_reward, 'mmd_reinforce_loss') as loss:
    value = model._mmd_term(x0, None, probabilities, x0)
    assert value.item() == 0.0
    loss.assert_not_called()
    stubs.native.assert_not_called()
    stubs.hf.assert_not_called()
    stubs.loader.assert_not_called()


def main():
  tests = [(name, fn) for name, fn in sorted(globals().items())
           if name.startswith('test_')]
  for name, fn in tests:
    fn()
    print(f'PASS  {name}')
  print(f'\nall {len(tests)} feature-MSE configuration tests passed')


if __name__ == '__main__':
  main()
