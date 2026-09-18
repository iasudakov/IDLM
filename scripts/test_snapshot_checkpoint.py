"""CPU snapshot checks with fake Nirvana modules; no external uploads."""
import ast
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import lightning as L
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nirvana_snapshot
from evaluated_checkpoint import EvaluatedCheckpoint
from test_evaluated_checkpoint import ToyModel, loaders, trainer as make_trainer


@contextlib.contextmanager
def fake_runtime():
  package = ModuleType('yt_tools')
  package.__path__ = []
  runtime = ModuleType('yt_tools.nirvana_utils')
  runtime.copy_snapshot_to_out = Mock()
  runtime.copy_out_to_snapshot = Mock()
  package.nirvana_utils = runtime
  with patch.dict(sys.modules, {'yt_tools': package,
                               'yt_tools.nirvana_utils': runtime}):
    yield runtime


def extract_main():
  """Execute actual callback/registration AST without importing heavy main."""
  tree = ast.parse((ROOT / 'main.py').read_text())
  namespace = dict(os=os, L=L, nirvana_snapshot=nirvana_snapshot,
                   EvaluatedCheckpoint=EvaluatedCheckpoint)
  classes = [node for node in tree.body
             if isinstance(node, ast.ClassDef)
             and node.name in ('PeriodicStepCheckpoint', 'SnapshotSync')]
  exec(compile(ast.Module(body=classes, type_ignores=[]),
               str(ROOT / 'main.py'), 'exec'), namespace)
  train = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == '_train')
  prefix = []
  for node in train.body:
    if (isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Tuple)
                and any(isinstance(item, ast.Name) and item.id == 'train_ds'
                        for item in target.elts) for target in node.targets)):
      break
    prefix.append(node)
  assert prefix and len(prefix) < len(train.body)
  return namespace, compile(ast.Module(body=prefix, type_ignores=[]),
                            str(ROOT / 'main.py'), 'exec')


class SnapshotTests(unittest.TestCase):
  def setUp(self):
    self.env = patch.dict(os.environ, {}, clear=True)
    self.env.start()
    self.addCleanup(self.env.stop)

  def test_helper_success_and_disabled_calls(self):
    with tempfile.TemporaryDirectory() as directory, fake_runtime() as runtime:
      path = str(Path(directory) / 'run')
      self.assertTrue(nirvana_snapshot.available())
      self.assertTrue(nirvana_snapshot.restore(path))
      self.assertTrue(Path(path).is_dir())
      runtime.copy_snapshot_to_out.assert_called_once_with(path)
      self.assertTrue(nirvana_snapshot.save(path, reason='test'))
      runtime.copy_out_to_snapshot.assert_called_once_with(path)
      runtime.copy_snapshot_to_out.reset_mock()
      runtime.copy_out_to_snapshot.reset_mock()
      self.assertFalse(nirvana_snapshot.restore(path, enabled=False))
      self.assertFalse(nirvana_snapshot.save(path, enabled=False))
      runtime.copy_snapshot_to_out.assert_not_called()
      runtime.copy_out_to_snapshot.assert_not_called()

  def test_missing_runtime_and_upload_failures_are_nonfatal(self):
    with tempfile.TemporaryDirectory() as directory:
      with patch.dict(sys.modules, {'yt_tools': None,
                                    'yt_tools.nirvana_utils': None}):
        self.assertFalse(nirvana_snapshot.available())
        self.assertFalse(nirvana_snapshot.restore(directory))
        self.assertFalse(nirvana_snapshot.save(directory))
      with fake_runtime() as runtime, contextlib.redirect_stdout(io.StringIO()) as output:
        runtime.copy_snapshot_to_out.side_effect = RuntimeError('restore unavailable')
        runtime.copy_out_to_snapshot.side_effect = RuntimeError('upload unavailable')
        self.assertFalse(nirvana_snapshot.restore(directory))
        self.assertFalse(nirvana_snapshot.save(directory))
        self.assertIn('restore unavailable', output.getvalue())
        self.assertIn('upload unavailable', output.getvalue())

  def test_global_rank_precedes_node_local_rank(self):
    cases = [({'RANK': '1', 'LOCAL_RANK': '0'}, False),
             ({'RANK': '0', 'LOCAL_RANK': '1'}, True),
             ({'SLURM_PROCID': '1', 'LOCAL_RANK': '0'}, False),
             ({'SLURM_PROCID': '0', 'LOCAL_RANK': '1'}, True),
             ({'LOCAL_RANK': '1'}, False), ({}, True)]
    for env, enabled in cases:
      with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
        with tempfile.TemporaryDirectory() as directory, fake_runtime() as runtime:
          self.assertEqual(nirvana_snapshot.restore(directory), enabled)
          self.assertEqual(nirvana_snapshot.save(directory), enabled)
          self.assertEqual(runtime.copy_snapshot_to_out.call_count, int(enabled))
          self.assertEqual(runtime.copy_out_to_snapshot.call_count, int(enabled))

  def test_evaluated_upload_follows_file_write_and_uses_whole_run_directory(self):
    with tempfile.TemporaryDirectory() as directory, fake_runtime() as runtime:
      run = Path(directory)
      (run / 'config.txt').write_text('whole run')
      events = []
      saved_path = run / 'checkpoints' / 'eval-step=00003000.ckpt'
      def write(path, weights_only=False):
        self.assertFalse(weights_only)
        Path(path).write_bytes(b'completed checkpoint')
        events.append('write')
      def upload(path):
        self.assertEqual(path, str(run))
        self.assertEqual(saved_path.read_bytes(), b'completed checkpoint')
        self.assertTrue((Path(path) / 'config.txt').exists())
        events.append('upload')
      runtime.copy_out_to_snapshot.side_effect = upload
      callback = EvaluatedCheckpoint(run / 'checkpoints', start_step=3000,
                                     snapshot_dir=run, snapshot=True)
      trainer = SimpleNamespace(global_step=3000, sanity_checking=False,
          state=SimpleNamespace(fn='fit'), is_global_zero=True,
          strategy=SimpleNamespace(broadcast=lambda value: value),
          save_checkpoint=write)
      callback.on_validation_end(trainer, None)
      self.assertEqual(events, ['write', 'upload'])
      callback.on_validation_end(trainer, None)
      self.assertEqual(events, ['write', 'upload'])
      follower = EvaluatedCheckpoint(run / 'checkpoints', start_step=3000,
                                     snapshot_dir=run, snapshot=True)
      trainer.is_global_zero = False
      trainer.strategy.broadcast = lambda value: (str(saved_path), None)
      trainer.save_checkpoint = Mock()
      follower.on_validation_end(trainer, None)
      trainer.save_checkpoint.assert_called_once()
      self.assertEqual(events, ['write', 'upload'])
      disabled = EvaluatedCheckpoint(run / 'checkpoints', start_step=3000,
                                     snapshot_dir=run, snapshot=False)
      trainer.is_global_zero = True
      trainer.strategy.broadcast = lambda value: value
      trainer.save_checkpoint = write
      disabled.on_validation_end(trainer, None)
      self.assertEqual(events, ['write', 'upload', 'write'])

  def test_actual_periodic_callback_uploads_after_write(self):
    namespace, _ = extract_main()
    with tempfile.TemporaryDirectory() as directory, fake_runtime() as runtime:
      run = Path(directory)
      events = []
      callback = namespace['PeriodicStepCheckpoint'](
          run / 'checkpoints', 2, snapshot_dir=run, snapshot=True)
      def write(path):
        Path(path).write_bytes(b'periodic checkpoint')
        events.append('write')
      def upload(path):
        self.assertEqual(path, str(run))
        self.assertEqual((run / 'checkpoints' / 'step=00000002.ckpt').read_bytes(),
                         b'periodic checkpoint')
        events.append('upload')
      runtime.copy_out_to_snapshot.side_effect = upload
      trainer = SimpleNamespace(is_global_zero=True, save_checkpoint=write)
      callback.on_train_batch_end(trainer, None, None, None, 0)
      self.assertEqual(events, [])
      callback.on_train_batch_end(trainer, None, None, None, 1)
      self.assertEqual(events, ['write', 'upload'])

  def test_real_fit_uploads_only_completed_qualifying_checkpoints(self):
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory)
      callback = EvaluatedCheckpoint(run / 'checkpoints', start_step=3,
                                     snapshot_dir=run, snapshot=True)
      saved_steps = []
      def upload(path, enabled=True, reason=''):
        self.assertEqual(path, str(run))
        self.assertTrue(enabled)
        step = callback.state_dict()['last_saved_step']
        checkpoint = torch.load(run / 'checkpoints' / f'eval-step={step:08d}.ckpt',
                                map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['global_step'], step)
        self.assertTrue(checkpoint['optimizer_states'])
        self.assertEqual(checkpoint['callbacks'][callback.state_key]['last_saved_step'], step)
        self.assertEqual(checkpoint['ema']['shadow_params'][0].item(), 7.)
        self.assertNotEqual(checkpoint['state_dict']['weight'].item(), 7.)
        saved_steps.append(step)
      with patch.object(nirvana_snapshot, 'save', side_effect=upload) as save:
        make_trainer(callback, 6).fit(ToyModel(), *loaders())
        self.assertEqual(save.call_count, 2)
      self.assertEqual(saved_steps, [4, 6])

  def test_actual_registration_restores_before_resume_check_and_logger(self):
    self.assertTrue(OmegaConf.load(ROOT / 'configs/config.yaml').checkpointing.snapshot)
    for enabled in (True, False):
      with self.subTest(snapshot=enabled), tempfile.TemporaryDirectory() as directory:
        namespace, code = extract_main()
        run = Path(directory)
        checkpoint = run / 'checkpoints' / 'last.ckpt'
        events = []
        config = OmegaConf.create(dict(mode='train', seed=1, algo=dict(name='mdlm'),
            training=dict(continue_teacher=False),
            checkpointing=dict(save_dir=str(run), snapshot=enabled,
              resume_from_ckpt=True, resume_ckpt_path=str(checkpoint),
              use_periodic_train_batch_checkpoint=True,
              save_evaluated_checkpoints=True, save_evaluated_from_step=3000),
            logger=dict(marker='logger'), callbacks=dict(
              checkpoint_every_n_steps=dict(every_n_train_steps=2))))
        def restore(path, enabled=True):
          self.assertEqual(path, str(run))
          events.append('restore')
          if enabled:
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'restored')
            (run / 'config_tree.txt').write_text('old snapshot config')
        def exists(path):
          events.append('exists')
          return Path(path).exists()
        def instantiate(cfg, **kwargs):
          if cfg.get('marker') == 'logger':
            events.append('logger')
          return object()
        def print_config(*args, **kwargs):
          events.append('config')
          (run / 'config_tree.txt').write_text('current config')
        def train(**kwargs):
          events.append('train')
          namespace.update(kwargs)
          exec(code, namespace)
        logger = SimpleNamespace(info=Mock())
        namespace.update(config=config, logger=SimpleNamespace(info=Mock()),
            __name__='snapshot_main_ast', L=SimpleNamespace(seed_everything=Mock()),
            nirvana_snapshot=SimpleNamespace(restore=restore),
            utils=SimpleNamespace(fsspec_exists=exists, get_logger=lambda name: logger),
            hydra=SimpleNamespace(utils=SimpleNamespace(instantiate=instantiate,
                                                       get_original_cwd=lambda: str(ROOT))),
            model_assets=SimpleNamespace(configure_model_weights=Mock()),
            _configure_training_mode=Mock(), _print_config=print_config,
            dataloader=SimpleNamespace(get_tokenizer=lambda cfg: object()),
            algo=SimpleNamespace(MDLM=object()), _train=train,
            _generate_samples=Mock(), _eval_ppl=Mock(), _gsm8k_eval=Mock())
        tree = ast.parse((ROOT / 'main.py').read_text())
        entry = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'main')
        entry.decorator_list = []
        exec(compile(ast.Module(body=[entry], type_ignores=[]),
                     str(ROOT / 'main.py'), 'exec'), namespace)
        namespace['main'](config)
        self.assertLess(events.index('restore'), events.index('exists'))
        self.assertLess(events.index('restore'), events.index('logger'))
        self.assertLess(events.index('restore'), events.index('config'))
        self.assertLess(events.index('restore'), events.index('train'))
        self.assertEqual((run / 'config_tree.txt').read_text(), 'current config')
        self.assertEqual(namespace['ckpt_path'], str(checkpoint) if enabled else None)
        savers = [c for c in namespace['callbacks']
                  if isinstance(c, (EvaluatedCheckpoint,
                                    namespace['PeriodicStepCheckpoint']))]
        self.assertEqual(len(savers), 2)
        for saver in savers:
          self.assertEqual(saver.snapshot_dir, str(run))
          self.assertEqual(saver.snapshot, enabled)
        for mode in ('sample_eval', 'ppl_eval', 'gsm8k_eval'):
          events.clear()
          config.mode = mode
          namespace['main'](config)
          self.assertNotIn('restore', events)
          self.assertNotIn('train', events)


if __name__ == '__main__':
  unittest.main()
