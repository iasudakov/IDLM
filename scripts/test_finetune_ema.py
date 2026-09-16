"""CPU checks for EMA-only warm starts without loading a pretrained model.

Run with: python scripts/test_finetune_ema.py
"""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import checkpoint_ema  # noqa: E402

# Use the real EMA implementation without importing the optional CUDA models.
_spec = importlib.util.spec_from_file_location(
  'finetune_test_ema', ROOT / 'models' / 'ema.py')
_ema_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ema_module)
EMA = _ema_module.ExponentialMovingAverage


def config(weights='ema'):
  return OmegaConf.create({
    'mode': 'train',
    'algo': {'name': 'mdlm', 'backbone': 'dit'},
    'adversarial_distill': {'is_distill': True},
    'training': {
      'finetune_weights': weights,
      'finetune_path': 'teacher.ckpt',
      'continue_teacher': False,
      'ema': 0.9999,
    },
  })


class TinyModel(torch.nn.Module):
  def __init__(self, auxiliary=True):
    super().__init__()
    self.backbone = torch.nn.Sequential(
      torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    with torch.no_grad():
      for index, parameter in enumerate(self.backbone.parameters()):
        parameter.fill_(index + 1)
    self.ema = EMA(self.backbone.parameters(), decay=0.93)
    self.ema.num_updates = 173
    for index, shadow in enumerate(self.ema.shadow_params):
      shadow.fill_(index + 21)
    if auxiliary:
      self.teacher = copy.deepcopy(self.backbone)
      self.fake_model = copy.deepcopy(self.backbone)
      with torch.no_grad():
        for parameter in self.teacher.parameters():
          parameter.add_(100)
        for parameter in self.fake_model.parameters():
          parameter.add_(200)
      self.ema_fake = EMA(self.fake_model.parameters(), decay=0.8)
      self.ema_fake.num_updates = 12
      self.teacher.requires_grad_(False)


def state(model):
  return copy.deepcopy({
    'parameters': model.state_dict(),
    'ema': model.ema.state_dict() if model.ema is not None else None,
    'ema_fake': (model.ema_fake.state_dict()
                 if getattr(model, 'ema_fake', None) is not None else None),
  })


class EmaWarmStartTests(unittest.TestCase):
  def assert_tree_equal(self, actual, expected):
    if isinstance(expected, torch.Tensor):
      self.assertTrue(torch.equal(actual, expected))
    elif isinstance(expected, dict):
      self.assertEqual(set(actual), set(expected))
      for key in expected:
        self.assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
      self.assertEqual(len(actual), len(expected))
      for left, right in zip(actual, expected):
        self.assert_tree_equal(left, right)
    else:
      self.assertEqual(actual, expected)

  def assert_all_backbones_ema(self, model):
    for name in ('backbone', 'teacher', 'fake_model'):
      module = getattr(model, name, None)
      if module is not None:
        for parameter, shadow in zip(
            module.parameters(), model.ema.shadow_params):
          self.assertTrue(torch.equal(parameter, shadow), name)
          self.assertNotEqual(parameter.data_ptr(), shadow.data_ptr())

  def test_all_models_use_ema_and_metadata_is_preserved(self):
    model = TinyModel()
    original = state(model)
    self.assertTrue(checkpoint_ema.apply_finetune_weights(model, config()))
    self.assert_all_backbones_ema(model)
    self.assert_tree_equal(model.ema.state_dict(), original['ema'])
    self.assert_tree_equal(model.ema_fake.state_dict(), original['ema'])
    self.assertFalse(next(model.teacher.parameters()).requires_grad)
    self.assertTrue(next(model.backbone.parameters()).requires_grad)
    for student_shadow, fake_shadow in zip(
        model.ema.shadow_params, model.ema_fake.shadow_params):
      self.assertNotEqual(student_shadow.data_ptr(), fake_shadow.data_ptr())
    with torch.no_grad():
      next(model.fake_model.parameters()).add_(8)
      model.ema_fake.shadow_params[0].add_(7)
    self.assert_tree_equal(model.ema.state_dict(), original['ema'])
    self.assertTrue(torch.equal(
      next(model.backbone.parameters()), original['ema']['shadow_params'][0]))
    self.assertTrue(torch.equal(
      next(model.teacher.parameters()), original['ema']['shadow_params'][0]))

  def test_frozen_parameters_are_initialized(self):
    model = TinyModel()
    model.requires_grad_(False)
    checkpoint_ema.initialize_backbones_from_ema(model)
    self.assert_all_backbones_ema(model)
    self.assertTrue(all(not p.requires_grad for p in model.parameters()))

  def test_native_feature_model_without_auxiliary_models(self):
    model = TinyModel(auxiliary=False)
    checkpoint_ema.initialize_backbones_from_ema(model)
    self.assert_all_backbones_ema(model)

  def test_validation_store_restore_retains_ema_initialization(self):
    model = TinyModel()
    checkpoint_ema.initialize_backbones_from_ema(model)
    expected = [p.detach().clone() for p in model.backbone.parameters()]
    model.ema.store(model.backbone.parameters())
    model.ema.copy_to(model.backbone.parameters())
    model.ema.restore(model.backbone.parameters())
    self.assert_tree_equal(list(model.backbone.parameters()), expected)

  def test_raw_mode_is_unchanged(self):
    model = TinyModel()
    original = state(model)
    self.assertFalse(checkpoint_ema.apply_finetune_weights(model, config('raw')))
    self.assert_tree_equal(state(model), original)

  def test_legacy_config_without_option_is_unchanged(self):
    cfg = config()
    del cfg.training.finetune_weights
    model = TinyModel()
    original = state(model)
    self.assertEqual(checkpoint_ema.finetune_weights(cfg), 'raw')
    checkpoint_ema.validate_finetune_weights(cfg)
    self.assertFalse(checkpoint_ema.apply_finetune_weights(model, cfg))
    self.assert_tree_equal(state(model), original)

  def test_full_resume_does_not_reinitialize_models(self):
    model = TinyModel()
    original = state(model)
    self.assertFalse(checkpoint_ema.apply_finetune_weights(
      model, config(), resuming=True))
    self.assert_tree_equal(state(model), original)

  def test_training_entrypoint_initializes_before_validation_and_skips_resume(self):
    import main

    for resuming in (False, True):
      with self.subTest(resuming=resuming):
        cfg = config()
        cfg.checkpointing = {
          'resume_from_ckpt': resuming,
          'resume_ckpt_path': '/test/resume.ckpt',
          'use_periodic_train_batch_checkpoint': False,
        }
        cfg.logger = {'_target_': 'test.logger'}
        cfg.trainer = {'_target_': 'test.trainer'}
        cfg.strategy = {'_target_': 'test.strategy'}
        cfg.eval = {'validate_before_train': True}
        cfg.debug = False
        cfg.callbacks = {}
        model = TinyModel()
        before = state(model)
        factory = mock.Mock()
        factory.load_from_checkpoint.return_value = model
        tracker = SimpleNamespace(defer_finalize=False)
        train_ds, valid_ds = object(), object()
        events = []

        def validate(actual_model, actual_valid_ds):
          self.assertIs(actual_model, model)
          self.assertIs(actual_valid_ds, valid_ds)
          self.assertTrue(tracker.defer_finalize)
          events.append(('validate', state(actual_model)))

        def fit(actual_model, actual_train_ds, actual_valid_ds, *, ckpt_path):
          self.assertIs(actual_model, model)
          self.assertIs(actual_train_ds, train_ds)
          self.assertIs(actual_valid_ds, valid_ds)
          self.assertFalse(tracker.defer_finalize)
          self.assertEqual(ckpt_path, '/test/resume.ckpt' if resuming else None)
          events.append(('fit', state(actual_model)))

        trainer = SimpleNamespace(validate=validate, fit=fit)
        targets = {
          'test.logger': tracker,
          'test.strategy': object(),
          'test.trainer': trainer,
        }

        def instantiate(target_config, **kwargs):
          return targets[target_config['_target_']]

        with mock.patch.object(main.hydra.utils, 'instantiate',
                               side_effect=instantiate), \
             mock.patch.object(main.dataloader, 'get_dataloaders',
                               return_value=(train_ds, valid_ds)), \
             mock.patch.object(main, '_print_batch'), \
             mock.patch.object(main.utils, 'fsspec_exists', return_value=True), \
             mock.patch.object(main.model_assets, 'resolve_checkpoint',
                               return_value='teacher.ckpt'):
          main._train(factory, cfg, mock.Mock(), tokenizer=None)
        factory.load_from_checkpoint.assert_called_once_with(
          'teacher.ckpt', tokenizer=None, config=cfg, weights_only=False)
        self.assertEqual([event[0] for event in events], ['validate', 'fit'])
        if resuming:
          for _, event_state in events:
            self.assert_tree_equal(event_state, before)
        else:
          self.assert_all_backbones_ema(model)
          expected = state(model)
          for _, event_state in events:
            self.assert_tree_equal(event_state, expected)
          self.assertFalse(torch.equal(
            before['parameters']['backbone.0.weight'],
            expected['parameters']['backbone.0.weight']))

  def test_valid_ema_configuration(self):
    cfg = config()
    self.assertEqual(checkpoint_ema.finetune_weights(cfg), 'ema')
    checkpoint_ema.validate_finetune_weights(cfg)

  def test_invalid_option_is_rejected(self):
    with self.assertRaises(ValueError):
      checkpoint_ema.finetune_weights(config('student_ema'))

  def test_unsupported_ema_configurations_are_rejected(self):
    overrides = {
      'training.finetune_path': '',
      'training.continue_teacher': True,
      'training.ema': 0,
      'algo.name': 'duo',
      'algo.backbone': 'hf_dit',
      'adversarial_distill.is_distill': False,
    }
    for path, value in overrides.items():
      with self.subTest(path=path):
        cfg = config()
        OmegaConf.update(cfg, path, value)
        with self.assertRaises(ValueError):
          checkpoint_ema.validate_finetune_weights(cfg)

  def test_raw_mode_does_not_impose_ema_restrictions(self):
    cfg = config('raw')
    cfg.training.finetune_path = ''
    cfg.training.continue_teacher = True
    cfg.training.ema = 0
    cfg.algo.name = 'duo'
    cfg.algo.backbone = 'hf_dit'
    cfg.adversarial_distill.is_distill = False
    checkpoint_ema.validate_finetune_weights(cfg)

  def test_evaluation_config_does_not_require_warm_start(self):
    cfg = config()
    cfg.mode = 'gsm8k_eval'
    cfg.training.finetune_path = ''
    cfg.adversarial_distill.is_distill = False
    checkpoint_ema.validate_finetune_weights(cfg)

  def test_native_ar_loader_selects_its_own_ema_and_freezes_features(self):
    import algo
    import mmd_reward

    for choice in ('raw', 'ema'):
      with self.subTest(choice=choice), tempfile.TemporaryDirectory() as tmp:
        filename = Path(tmp) / 'ar.ckpt'
        filename.touch()
        model = TinyModel(auxiliary=False)
        expected = [p.detach().clone() for p in (
          model.ema.shadow_params if choice == 'ema'
          else model.backbone.parameters())]
        cfg = config(choice)
        tokenizer = object()
        with mock.patch.object(algo.AR, 'load_from_checkpoint',
                               return_value=model) as load:
          backbone = mmd_reward.load_ar_backbone(
            str(filename), cfg, tokenizer, device='cpu')
        self.assertIs(backbone, model.backbone)
        self.assert_tree_equal(list(backbone.parameters()), expected)
        self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))
        self.assertFalse(backbone.training)
        load.assert_called_once()
        loaded_cfg = load.call_args.kwargs['config']
        self.assertEqual(loaded_cfg.algo.name, 'ar')
        self.assertFalse(loaded_cfg.adversarial_distill.is_distill)
        self.assertEqual(cfg.algo.name, 'mdlm')
        self.assertTrue(cfg.adversarial_distill.is_distill)

  def test_native_ar_loader_rejects_missing_ema(self):
    import algo
    import mmd_reward

    with tempfile.TemporaryDirectory() as tmp:
      filename = Path(tmp) / 'ar.ckpt'
      filename.touch()
      model = TinyModel(auxiliary=False)
      model.ema = None
      with mock.patch.object(algo.AR, 'load_from_checkpoint', return_value=model):
        with self.assertRaises(ValueError):
          mmd_reward.load_ar_backbone(str(filename), config(), None, device='cpu')

  def assert_invalid_state_is_atomic(self, model):
    original = state(model)
    with self.assertRaises(ValueError):
      checkpoint_ema.initialize_backbones_from_ema(model)
    self.assert_tree_equal(state(model), original)

  def test_missing_ema_is_rejected_without_changes(self):
    model = TinyModel()
    model.ema = None
    self.assert_invalid_state_is_atomic(model)

  def test_shadow_count_mismatch_is_atomic(self):
    for extra in (False, True):
      with self.subTest(extra=extra):
        model = TinyModel()
        if extra:
          model.ema.shadow_params.append(torch.ones(1))
        else:
          model.ema.shadow_params.pop()
        self.assert_invalid_state_is_atomic(model)

  def test_late_shadow_shape_mismatch_is_atomic(self):
    model = TinyModel()
    model.ema.shadow_params[-1] = torch.ones(3)
    self.assert_invalid_state_is_atomic(model)

  def test_nontensor_shadow_is_rejected_atomically(self):
    model = TinyModel()
    model.ema.shadow_params[-1] = None
    self.assert_invalid_state_is_atomic(model)

  def test_auxiliary_model_shape_mismatch_is_atomic(self):
    model = TinyModel()
    model.teacher[1] = torch.nn.Linear(2, 3)
    self.assert_invalid_state_is_atomic(model)


if __name__ == '__main__':
  unittest.main(verbosity=2)
