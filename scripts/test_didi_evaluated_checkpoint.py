"""Offline evaluated-checkpoint tests using the actual OWT DiDi CPU runtime."""
import copy
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
import torch
from torch.utils.data import DataLoader

import nirvana_snapshot
from didi_repro import UPSTREAM_REVISION
from didi_repro.checkpointing import make_evaluated_checkpoint
from didi_repro.entrypoint import install_entrypoint, prepare_config
from didi_repro.pretrain_eval import maybe_evaluate_before_training
from didi_repro.runtime import UPSTREAM_ROOT, load_upstream, verify_upstream
from evaluated_checkpoint import EvaluatedCheckpoint
from test_didi_runtime import ToyBackbone, ToyDataset, ToyTokenizer


class EvaluatedCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.upstream = load_upstream()

    def config(self, overrides=(), tiny=False):
        with hydra.initialize_config_dir(version_base=None,
                                        config_dir=str(UPSTREAM_ROOT / 'configs')):
            config = hydra.compose(config_name='config',
                                   overrides=['trainer.devices=1', *overrides])
        prepare_config(config, overrides=overrides)
        if tiny:
            config.model.hidden_size = 8
            config.model.length = 12
            config.loader.num_workers = 0
            config.loader.eval_batch_size = 2
            config.trainer.accelerator = 'cpu'
            config.trainer.max_steps = 6
            config.trainer.limit_val_batches = 1
            config.optim.lr = .01
            config.algo.discriminator_lr = .01
            config.algo.lr_scheduler.warmup_steps = 1
            config.algo.save_after_n_steps = 1
            config.algo.print_every = 0
            config.eval.generate_samples = False
        return config

    def model(self, config):
        base = self.upstream.algo.trainer_base
        reference_tokenizer = SimpleNamespace(pad_token='pad', pad_token_id=0)
        with mock.patch.object(base.models.dit, 'DIT', ToyBackbone), \
             mock.patch.object(base.metrics.transformers.AutoTokenizer, 'from_pretrained',
                               return_value=reference_tokenizer):
            model = self.upstream.algo.DiDiInstruct(config, ToyTokenizer())
        teacher = {f'backbone.{name}': value.detach().clone() + .03
                   for name, value in model.backbone.state_dict().items()}
        model.initialize_teacher_checkpoint({'state_dict': teacher})
        return model

    def test_hydra_flags_are_opt_in_and_preserve_upstream_source(self):
        self.assertIsNone(make_evaluated_checkpoint(self.config()))
        config = self.config([
            '+checkpointing.save_evaluated_checkpoints=true',
            '+checkpointing.save_evaluated_from_step=2000',
            '+checkpointing.snapshot=true',
            'callbacks.checkpoint_every_n_steps.save_top_k=0',
            'callbacks.checkpoint_every_n_steps.save_last=false',
            'callbacks.checkpoint_monitor.save_top_k=0',
            'callbacks.checkpoint_monitor.save_last=false'])
        callback = make_evaluated_checkpoint(config)
        self.assertIsInstance(callback, EvaluatedCheckpoint)
        self.assertEqual(callback.start_step, 2000)
        self.assertEqual(callback.snapshot_dir, str(config.checkpointing.save_dir))
        self.assertEqual(callback.dirpath,
                         str(Path(config.checkpointing.save_dir) / 'checkpoints'))
        self.assertTrue(callback.snapshot)
        config.checkpointing.snapshot = False
        self.assertFalse(make_evaluated_checkpoint(config).snapshot)
        config.checkpointing.save_evaluated_checkpoints = False
        self.assertIsNone(make_evaluated_checkpoint(config))
        self.assertEqual(verify_upstream()['commit'], UPSTREAM_REVISION)

    def test_actual_entrypoint_registers_saver_after_evaluation_callbacks(self):
        install_entrypoint(self.upstream)
        config = self.config([
            '+checkpointing.save_evaluated_checkpoints=true',
            '+checkpointing.save_evaluated_from_step=2000',
            '+checkpointing.snapshot=true'])
        model = SimpleNamespace(initialize_teacher_checkpoint=mock.Mock())
        trainer = mock.Mock()
        monitor, pareto = object(), object()
        train_loader, valid_loader = object(), SimpleNamespace(tokenizer=object())
        captured = []

        def instantiate(spec, **kwargs):
            if spec._target_ == 'lightning.Trainer':
                captured.extend(kwargs['callbacks'])
                return trainer
            return object()

        repaired = inspect.getclosurevars(self.upstream._train).nonlocals['patched_train']
        replacements = {
            'make_underdeep_logger': lambda *args, **kwargs: object(),
            'initialize_underdeep_logger': lambda *args: None,
            'initialize_didi_student': lambda *args: None,
            'maybe_evaluate_didi_before_training': lambda *args, **kwargs: None,
            'attach_didi_monitor': lambda *args: monitor,
            'attach_didi_pareto': lambda *args: pareto,
            'dataloader': SimpleNamespace(get_dataloaders=lambda *args: (train_loader, valid_loader)),
            '_print_batch': lambda *args: None,
            'hydra': SimpleNamespace(utils=SimpleNamespace(instantiate=instantiate)),
            'utils': SimpleNamespace(fsspec_exists=lambda *args: True,
                                     count_parameters=lambda *args: 1),
            'torch': SimpleNamespace(load=lambda *args, **kwargs: {'state_dict': {}}),
        }
        with tempfile.NamedTemporaryFile(prefix='didi-evaluated-teacher-') as teacher:
            config.training.finetune_path = teacher.name
            with mock.patch.dict(repaired.__globals__, replacements), \
                 mock.patch('didi_repro.entrypoint._check_cached_data'):
                self.upstream._train(lambda *args, **kwargs: model,
                                     config, mock.Mock(), None)
        savers = [callback for callback in captured
                  if isinstance(callback, EvaluatedCheckpoint)]
        self.assertEqual(len(savers), 1)
        self.assertLess(captured.index(monitor), captured.index(savers[0]))
        self.assertLess(captured.index(pareto), captured.index(savers[0]))
        self.assertEqual(savers[0].start_step, 2000)
        self.assertTrue(savers[0].snapshot)
        trainer.fit.assert_called_once_with(model, train_loader, valid_loader, ckpt_path=None)

    def test_real_didi_fit_saves_complete_evaluated_models_then_uploads(self):
        torch.manual_seed(19)
        config = self.config([
            '+checkpointing.save_evaluated_checkpoints=true',
            '+checkpointing.save_evaluated_from_step=4',
            '+checkpointing.snapshot=true',
            '+didi_eval.before_training=true'], tiny=True)
        with tempfile.TemporaryDirectory(prefix='didi-evaluated-fit-') as directory:
            config.checkpointing.save_dir = directory
            model = self.model(config)
            teacher_before = copy.deepcopy(model.teacher.state_dict())
            saver = make_evaluated_checkpoint(config)
            uploads = []
            trainer = L.Trainer(
                accelerator='cpu', devices=1, precision='32-true', max_steps=6,
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, limit_val_batches=1, num_sanity_val_steps=1,
                val_check_interval=1, default_root_dir=directory, gradient_clip_val=0.0,
                callbacks=[saver])
            loader = DataLoader(ToyDataset(), batch_size=1, shuffle=False)

            def upload(path, enabled=True, reason=''):
                self.assertEqual(path, directory)
                self.assertTrue(enabled)
                step = trainer.global_step
                filename = Path(directory) / 'checkpoints' / f'eval-step={step:08d}.ckpt'
                checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
                self.assertEqual(checkpoint['global_step'], step)
                self.assertEqual(checkpoint['callbacks'][saver.state_key]['last_saved_step'], step)
                self.assertEqual(len(checkpoint['optimizer_states']), 2)
                self.assertTrue(all(state['state'] for state in checkpoint['optimizer_states']))
                metadata = checkpoint['didi_repro']
                self.assertEqual(metadata['upstream_revision'], UPSTREAM_REVISION)
                self.assertEqual(metadata['next_iteration'], step // 2)
                self.assertIsNotNone(metadata['teacher'])
                if model.ema is not None:
                    self.assertIn('ema', checkpoint)
                    self.assertTrue(checkpoint['ema']['shadow_params'])
                state = checkpoint['state_dict']
                for prefix in ('backbone.', 'student_ema.', 'discriminator.', 'teacher.'):
                    self.assertTrue(any(key.startswith(prefix) for key in state))
                for key, tensor in model.state_dict().items():
                    torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
                for key, tensor in teacher_before.items():
                    torch.testing.assert_close(state['teacher.' + key], tensor, rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(value, state['student_ema.' + key])
                                    for key, value in model.backbone.state_dict().items()))
                uploads.append(step)

            with mock.patch.object(nirvana_snapshot, 'save', side_effect=upload):
                maybe_evaluate_before_training(trainer, model, loader)
                self.assertEqual(list(Path(directory).rglob('*.ckpt')), [])
                self.assertEqual(uploads, [])
                trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
            self.assertEqual(trainer.global_step, 6)
            self.assertEqual(model.global_steps, 3)
            self.assertEqual(uploads, [4, 6])
            self.assertEqual(sorted(path.name for path in Path(directory).rglob('*.ckpt')),
                             ['eval-step=00000004.ckpt', 'eval-step=00000006.ckpt'])

    def test_disabled_feature_keeps_native_iteration_zero_save(self):
        config = self.config(tiny=True)
        self.assertIsNone(make_evaluated_checkpoint(config))
        model = self.model(config)
        with tempfile.TemporaryDirectory(prefix='didi-native-save-') as directory:
            trainer = L.Trainer(
                accelerator='cpu', devices=1, precision='32-true', max_steps=2,
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0,
                default_root_dir=directory, gradient_clip_val=0.0)
            with mock.patch.object(nirvana_snapshot, 'save') as upload:
                trainer.fit(model, train_dataloaders=DataLoader(ToyDataset(), batch_size=1))
                upload.assert_not_called()
            checkpoint = torch.load(Path(directory) / 'student_iters_0.ckpt',
                                    map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['global_step'], 2)
            self.assertEqual(checkpoint['didi_repro']['next_iteration'], 1)


if __name__ == '__main__':
    unittest.main()
