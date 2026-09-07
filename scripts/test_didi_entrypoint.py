"""Offline tests for launcher configuration and logger integration boundaries."""

import copy
import ast
import inspect
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from didi_repro.entrypoint import (
  _check_cached_data, install_entrypoint, prepare_config, validate_environment)
from didi_repro.runtime import UPSTREAM_ROOT, load_upstream
from didi_repro.tracking import DiDiUnderdeepLogger


class LauncherTests(unittest.TestCase):
  root = Path(__file__).resolve().parents[1]

  def test_shell_syntax(self):
    for filename in ('train_didi_owt.sh', 'eval_didi_owt.sh'):
      with self.subTest(script=filename):
        subprocess.run(['bash', '-n', str(self.root / 'scripts' / filename)], check=True)

  def test_launchers_preserve_original_presets_without_starting_training(self):
    with tempfile.TemporaryDirectory(prefix='didi-launcher-test-') as directory:
      fixture = Path(directory)
      teacher = fixture / 'teacher.ckpt'
      teacher.touch()
      cache = fixture / 'cache'
      for split in ('openwebtext-train_train', 'openwebtext-valid_validation'):
        (cache / f'{split}_bs1024_wrapped.dat').mkdir(parents=True)
      fake_python = fixture / 'python'
      fake_python.write_text(
        f'#!{sys.executable}\nimport sys\n'
        "print(2 if sys.argv[1] == '-c' else repr(sys.argv[1:]))\n")
      fake_python.chmod(0o755)
      environment = dict(os.environ, PATH=str(fixture) + os.pathsep + os.environ['PATH'],
                         TEACHER_CHECKPOINT=str(teacher), CHECKPOINT=str(teacher),
                         OWT_CACHE=str(cache), SAVE_DIR=str(fixture / 'output'),
                         GPUS='0,1', DIDI_DRY_RUN='1', RUN_NAME='offline-didi-test')
      result = subprocess.run(['bash', str(self.root / 'scripts/train_didi_owt.sh')],
                              env=environment, text=True, capture_output=True, check=True)
      command = shlex.split(result.stdout)
      self.assertEqual(command[:3], ['python', '-m', 'torch.distributed.run'])
      for argument in ('--nproc_per_node=2', 'loader.batch_size=1',
                       'loader.global_batch_size=2', 'trainer.val_check_interval=1000',
                       '+underdeep.name=offline-didi-test'):
        self.assertIn(argument, command)
      self.assertFalse(any(arg.startswith(('optim.', 'algo.', 'sampling.')) for arg in command))
      # Fake Python reports arguments; the real evaluator is never imported.
      result = subprocess.run(['bash', str(self.root / 'scripts/eval_didi_owt.sh')],
                              env=environment, text=True, capture_output=True, check=True)
      command = ast.literal_eval(result.stdout.strip())
      self.assertEqual(command[:2], ['-m', 'didi_repro'])
      for argument in ('mode=sample_eval', 'sampling.steps=16',
                       'sampling.predictor=guided', 'loader.eval_batch_size=20',
                       'sampling.num_sample_batches=2'):
        self.assertIn(argument, command)


class EntryTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.upstream = load_upstream()

  def config(self):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(UPSTREAM_ROOT / 'configs')):
      return hydra.compose(config_name='config', overrides=['trainer.devices=1'])

  def test_integration_defaults_do_not_change_any_upstream_configuration(self):
    original = self.config()
    snapshot = OmegaConf.to_container(original, resolve=False)
    prepared = prepare_config(original)
    for key, value in snapshot.items():
      self.assertEqual(OmegaConf.to_container(prepared, resolve=False)[key], value)
    self.assertTrue(prepared.didi_runtime.strict_environment)
    self.assertTrue(prepared.didi_monitor.gm)
    self.assertEqual(prepared.algo.num_samples_per_prompt, 8)
    self.assertEqual(prepared.optim.lr, 1e-6)
    self.assertEqual(prepared.loader.batch_size, 1)
    self.assertFalse(prepared.didi_eval.before_training)
    self.assertIsNone(prepared.didi_init.student_hf)
    self.assertFalse(prepared.didi_init.local_files_only)

  def test_new_flags_compose_without_changing_upstream_defaults(self):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(UPSTREAM_ROOT / 'configs')):
      config = hydra.compose(config_name='config', overrides=[
        'trainer.devices=1', '+didi_eval.before_training=true',
        '+didi_init.student_hf=haoyangzheng/didi-instruct-small'])
    prepare_config(config)
    self.assertTrue(config.didi_eval.before_training)
    self.assertEqual(config.didi_init.student_hf, 'haoyangzheng/didi-instruct-small')
    self.assertEqual(config.optim.lr, 1e-6)
    self.assertEqual(config.trainer.max_steps, 10000)
    self.assertEqual(config.sampling.steps, 1000)

  def test_strict_environment_fails_on_unrecorded_runtime(self):
    config = prepare_config(self.config())
    good = {'python': '3.12.11', 'torch': '2.5.1+cu121', 'lightning': '2.5.3',
            'pytorch-lightning': '2.5.3', 'flash-attn': '2.8.3', 'transformers': '4.55.4'}
    validate_environment(config, good)
    with self.assertRaisesRegex(RuntimeError, 'dependency mismatch'):
      validate_environment(config, dict(good, torch='2.2.1'))

  def test_logger_calls_do_not_advance_global_rngs(self):
    class Run:
      uid = 'offline-test'
      def log(self, *args, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(3)

    logger = DiDiUnderdeepLogger()
    logger._initialized = True
    logger._run = Run()
    random.seed(80)
    np.random.seed(81)
    torch.manual_seed(82)
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    expected_torch = torch.get_rng_state().clone()
    logger.log_metrics({'loss': 0.5}, step=1)
    logger.log_hyperparams({'lr': 1e-6})
    self.assertEqual(expected_python, random.getstate())
    np.testing.assert_array_equal(expected_numpy[1], np.random.get_state()[1])
    self.assertTrue(torch.equal(expected_torch, torch.get_rng_state()))

  def test_cache_preflight_never_deletes_or_generates(self):
    config = prepare_config(self.config())
    with tempfile.TemporaryDirectory(prefix='didi-cache-check-') as directory:
      config.data.cache_dir = directory
      with mock.patch('shutil.rmtree') as remove, mock.patch('datasets.load_from_disk') as loader:
        with self.assertRaises(FileNotFoundError):
          _check_cached_data(config)
        loader.assert_not_called()
        remove.assert_not_called()
      for dataset, mode in ((config.data.train, 'train'), (config.data.valid, 'validation')):
        (Path(directory) / f'{dataset}_{mode}_bs1024_wrapped.dat').mkdir()
      with mock.patch('shutil.rmtree') as remove, \
           mock.patch('datasets.load_from_disk', side_effect=ValueError('broken cache')):
        with self.assertRaisesRegex(ValueError, 'broken cache'):
          _check_cached_data(config)
        remove.assert_not_called()

  def test_training_adapter_requires_a_teacher_before_any_data_or_run(self):
    config = prepare_config(self.config())
    config.training.finetune_path = '/does-not-exist/didi-teacher.ckpt'
    install_entrypoint(self.upstream)
    with mock.patch('didi_repro.entrypoint.make_underdeep_logger') as logger, \
         mock.patch.object(self.upstream.dataloader, 'get_dataloaders') as data:
      with self.assertRaises(FileNotFoundError):
        self.upstream._train(None, config, None, None)
      logger.assert_not_called()
      data.assert_not_called()

  def test_original_loader_exception_cannot_remove_a_healthy_cache(self):
    install_entrypoint(self.upstream)
    loader = self.upstream.dataloader
    with mock.patch.object(loader.datasets, 'load_from_disk', side_effect=ValueError('disk full')), \
         mock.patch('shutil.rmtree') as remove:
      with self.assertRaisesRegex(RuntimeError, 'cache load failed; preserved'):
        loader.get_dataset('openwebtext-train', None, True, 'train', '/unmodified/cache', ratio=1)
      remove.assert_not_called()

  def test_evaluation_loader_moves_cpu_checkpoint_to_cuda(self):
    install_entrypoint(self.upstream)
    config = prepare_config(self.config())
    model_class = mock.Mock()
    cpu_model = model_class.load_from_checkpoint.return_value
    result = self.upstream._load_from_checkpoint(model_class, config, None)
    cpu_model.to.assert_called_once_with('cuda')
    self.assertIs(result, cpu_model.to.return_value)
    self.assertFalse(model_class.load_from_checkpoint.call_args.kwargs['weights_only'])

  def test_missing_resume_is_not_a_silent_fresh_run(self):
    install_entrypoint(self.upstream)
    config = prepare_config(self.config())
    with tempfile.NamedTemporaryFile(prefix='didi-teacher-placeholder-') as teacher:
      config.training.finetune_path = teacher.name
      config.checkpointing.resume_from_ckpt = True
      config.checkpointing.resume_ckpt_path = '/does-not-exist/didi-resume.ckpt'
      with mock.patch('didi_repro.entrypoint.make_underdeep_logger') as logger:
        with self.assertRaisesRegex(FileNotFoundError, 'Requested resume checkpoint'):
          self.upstream._train(None, config, None, None)
        logger.assert_not_called()

  def test_reject_hf_initialization_with_resume_before_io(self):
    install_entrypoint(self.upstream)
    config = prepare_config(self.config())
    config.didi_init.student_hf = 'haoyangzheng/didi-instruct-small'
    config.checkpointing.resume_from_ckpt = True
    with mock.patch('didi_repro.entrypoint._check_cached_data') as data:
      with self.assertRaisesRegex(ValueError, 'fresh warm start, not a resume'):
        self.upstream._train(None, config, None, None)
      data.assert_not_called()

  def test_reject_pre_eval_with_validation_disabled(self):
    install_entrypoint(self.upstream)
    config = prepare_config(self.config())
    config.didi_eval.before_training = True
    config.trainer.limit_val_batches = 0
    with mock.patch('didi_repro.entrypoint._check_cached_data') as data:
      with self.assertRaisesRegex(ValueError, 'limit_val_batches > 0'):
        self.upstream._train(None, config, None, None)
      data.assert_not_called()

  def test_training_wires_teacher_student_then_evaluation_then_fit(self):
    install_entrypoint(self.upstream)
    config = prepare_config(self.config())
    config.didi_eval.before_training = True
    config.didi_init.student_hf = 'haoyangzheng/didi-instruct-small'
    events = []
    model = SimpleNamespace(initialize_teacher_checkpoint=lambda *args: events.append('teacher'))
    train_loader = object()
    valid_loader = SimpleNamespace(tokenizer=object())
    trainer = mock.Mock()
    trainer.fit.side_effect = lambda *args, **kwargs: events.append('fit')
    tracker = object()

    def instantiate(spec, **kwargs):
      if spec._target_ == 'lightning.Trainer':
        events.append('trainer')
        self.assertIs(kwargs['logger'], tracker)
        return trainer
      return object()

    def evaluate(actual_trainer, actual_model, loader, ckpt_path=None):
      self.assertIs(actual_trainer, trainer)
      self.assertIs(actual_model, model)
      self.assertIs(loader, valid_loader)
      self.assertIsNone(ckpt_path)
      events.append('pre_eval')

    repaired = inspect.getclosurevars(self.upstream._train).nonlocals['patched_train']
    replacements = {
      'make_underdeep_logger': lambda *args, **kwargs: tracker,
      'initialize_underdeep_logger': lambda *args: events.append('logger_ready'),
      'initialize_didi_student': lambda *args: events.append('student'),
      'maybe_evaluate_didi_before_training': evaluate,
      'attach_didi_monitor': lambda *args: None,
      'dataloader': SimpleNamespace(get_dataloaders=lambda *args: (train_loader, valid_loader)),
      '_print_batch': lambda *args: None,
      'hydra': SimpleNamespace(utils=SimpleNamespace(instantiate=instantiate)),
      'utils': SimpleNamespace(fsspec_exists=lambda *args: True, count_parameters=lambda *args: 1),
      'torch': SimpleNamespace(load=lambda *args, **kwargs: {'state_dict': {}}),
    }
    with tempfile.NamedTemporaryFile(prefix='didi-entry-teacher-') as teacher:
      config.training.finetune_path = teacher.name
      with mock.patch.dict(repaired.__globals__, replacements), \
           mock.patch('didi_repro.entrypoint._check_cached_data'):
        self.upstream._train(lambda *args, **kwargs: model, config, mock.Mock(), None)
    self.assertEqual(events, ['teacher', 'student', 'logger_ready', 'trainer', 'pre_eval', 'fit'])
    trainer.fit.assert_called_once_with(model, train_loader, valid_loader, ckpt_path=None)


if __name__ == '__main__':
  torch.set_num_threads(1)
  unittest.main(verbosity=2)
