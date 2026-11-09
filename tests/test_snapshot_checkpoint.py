"""Snapshot export must follow completed writes and preserve checkpoint I/O."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lightning as L
import torch
from torch.utils.data import DataLoader

from cdlm import CDLM
from cdlm_trajectories import (CDLMTrajectoryDataset, build_cache_identity,
                               prepare_trajectories)
from snapshot_checkpoint import SnapshotCheckpointIO
from test_cdlm_training import CPUDenoiser, Original, Tokenizer, config as make_config


CPU_DEFAULT = object()


class Backend:
    def __init__(self):
        self.events = []

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        self.events.append(('write begin', checkpoint, path, storage_options))
        Path(path).write_text(json.dumps(checkpoint))
        self.events.append(('write complete', path))
        return 'backend saved'

    def load_checkpoint(self, path, map_location=CPU_DEFAULT):
        self.events.append(('load', path, map_location))
        return 'backend loaded'

    def remove_checkpoint(self, path):
        self.events.append(('remove', path))
        return 'backend removed'

    def teardown(self):
        self.events.append(('teardown',))
        return 'backend torn down'


class SnapshotCheckpointTest(unittest.TestCase):
    def test_completed_local_write_precedes_snapshot_with_original_backend_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend()
            wrapper = SnapshotCheckpointIO(backend, directory, namespace='cdlm_tinygsm')
            path = Path(directory) / 'step=1000.ckpt'
            checkpoint, storage = {'global_step': 1000}, {'backend_option': 'kept'}

            def export(run, enabled, reason, namespace):
                self.assertEqual(backend.events[-1], ('write complete', path))
                self.assertEqual(json.loads(path.read_text()), checkpoint)
                self.assertEqual(run, directory)
                self.assertTrue(enabled)
                self.assertIn(str(path), reason)
                self.assertEqual(namespace, 'cdlm_tinygsm')
                backend.events.append(('snapshot',))
                return True

            with patch('snapshot_checkpoint.nirvana_snapshot.save', side_effect=export) as save:
                result = wrapper.save_checkpoint(checkpoint, path, storage_options=storage)
            self.assertIs(wrapper.checkpoint_io, backend)
            self.assertIs(backend.events[0][1], checkpoint)
            self.assertIs(backend.events[0][3], storage)
            self.assertEqual(backend.events[-1], ('snapshot',))
            self.assertEqual(result, 'backend saved')
            save.assert_called_once()

    def test_backend_write_failure_propagates_without_any_snapshot(self):
        backend = Backend()
        wrapper = SnapshotCheckpointIO(backend, 'unused')
        with patch.object(backend, 'save_checkpoint', side_effect=OSError('disk full')), \
                patch('snapshot_checkpoint.nirvana_snapshot.save') as save:
            with self.assertRaisesRegex(OSError, 'disk full'):
                wrapper.save_checkpoint({}, 'failed.ckpt')
        save.assert_not_called()

    def test_export_failure_does_not_fail_a_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.ckpt'
            wrapper = SnapshotCheckpointIO(Backend(), directory)
            output = io.StringIO()
            with patch('snapshot_checkpoint.nirvana_snapshot.save',
                       side_effect=RuntimeError('snapshot service unavailable')), \
                    contextlib.redirect_stdout(output):
                result = wrapper.save_checkpoint({'global_step': 9}, path)
            self.assertEqual(result, 'backend saved')
            self.assertEqual(json.loads(path.read_text()), {'global_step': 9})
            self.assertIn('snapshot service unavailable', output.getvalue())

    def test_disabled_snapshot_does_not_export_and_keeps_local_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.ckpt'
            wrapper = SnapshotCheckpointIO(Backend(), directory, enabled=False)
            with patch('snapshot_checkpoint.nirvana_snapshot.save') as save:
                self.assertEqual(wrapper.save_checkpoint({'global_step': 4}, path), 'backend saved')
            save.assert_not_called()
            self.assertEqual(json.loads(path.read_text())['global_step'], 4)

    def test_non_nirvana_save_result_does_not_change_backend_success(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = SnapshotCheckpointIO(Backend(), directory)
            with patch('snapshot_checkpoint.nirvana_snapshot.save', return_value=False):
                result = wrapper.save_checkpoint({}, Path(directory) / 'local.ckpt')
            self.assertEqual(result, 'backend saved')

    def test_load_preserves_backend_default_and_explicit_map_location(self):
        backend = Backend()
        wrapper = SnapshotCheckpointIO(backend, 'unused')
        path = Path('last.ckpt')
        self.assertEqual(wrapper.load_checkpoint(path), 'backend loaded')
        self.assertIs(backend.events[-1][2], CPU_DEFAULT)
        wrapper.load_checkpoint(path, None)
        self.assertIsNone(backend.events[-1][2])
        wrapper.load_checkpoint(path, map_location='cpu')
        self.assertEqual(backend.events[-1], ('load', path, 'cpu'))

    def test_remove_and_teardown_delegate_without_snapshot_side_effects(self):
        backend = Backend()
        wrapper = SnapshotCheckpointIO(backend, 'unused')
        path = Path('old.ckpt')
        with patch('snapshot_checkpoint.nirvana_snapshot.save') as save:
            self.assertEqual(wrapper.remove_checkpoint(path), 'backend removed')
            self.assertEqual(wrapper.teardown(), 'backend torn down')
        self.assertEqual(backend.events, [('remove', path), ('teardown',)])
        save.assert_not_called()

    def test_cdlm_periodic_last_and_direct_final_checkpoints_export_full_resumable_state(self):
        with patch('models.dit.DIT', CPUDenoiser), tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(12)
            cfg = make_config(directory)
            tokenizer = Tokenizer()
            model = CDLM(cfg, tokenizer)
            torch.save({'state_dict': {'backbone.' + name: tensor.clone()
                                      for name, tensor in model.backbone.state_dict().items()}},
                       cfg.training.finetune_path)
            model.initialize_from_checkpoint(cfg.training.finetune_path)
            teacher_head = {name: tensor.clone() for name, tensor in model.teacher_head.state_dict().items()}
            collector = CDLM(cfg, tokenizer, inference_only=True)
            collector.load_evaluation_weights(cfg.training.finetune_path, use_ema=False)
            prepare_trajectories(collector, cfg, tokenizer, Original())
            identity = build_cache_identity(cfg, tokenizer, Original())
            loader = DataLoader(CDLMTrajectoryDataset(cfg.algo.trajectory_cache_dir, identity),
                                batch_size=2)
            model._cache_fingerprint = identity['fingerprint']
            periodic = L.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(Path(directory) / 'checkpoints'), every_n_train_steps=1,
                save_top_k=-1, save_last=True, save_on_train_epoch_end=True)
            trainer = L.Trainer(accelerator='cpu', devices=1, max_steps=2, max_epochs=-1,
                                accumulate_grad_batches=2, logger=False, callbacks=[periodic],
                                enable_progress_bar=False, enable_model_summary=False,
                                limit_val_batches=0, num_sanity_val_steps=0)
            backend = trainer.strategy.checkpoint_io
            wrapper = SnapshotCheckpointIO(backend, directory, namespace='cdlm_tinygsm')
            trainer.strategy.checkpoint_io = wrapper
            exports = []

            def export(run, enabled, reason, namespace):
                if not reason.startswith('checkpoint written: '):
                    return True  # The module may separately persist final report JSON.
                saved_path = backend.save_checkpoint.call_args.args[1]
                saved = torch.load(saved_path, map_location='cpu', weights_only=False)
                self.assertEqual(run, directory)
                self.assertTrue(enabled)
                self.assertEqual(namespace, 'cdlm_tinygsm')
                self.assertIn(str(saved_path), reason)
                self.assertTrue(saved['optimizer_states'])
                self.assertTrue(saved['lr_schedulers'])
                self.assertTrue(saved['loops'])
                self.assertEqual(saved['cdlm_cache_fingerprint'], identity['fingerprint'])
                for name, expected in teacher_head.items():
                    self.assertTrue(torch.equal(saved['state_dict']['teacher_head.' + name], expected))
                exports.append((Path(saved_path).name, saved['global_step']))
                return True

            with patch.object(backend, 'save_checkpoint', wraps=backend.save_checkpoint), \
                    patch('snapshot_checkpoint.nirvana_snapshot.save', side_effect=export):
                trainer.fit(model, loader)
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual({step for name, step in exports if name != 'last.ckpt'}, {1, 2})
            self.assertGreaterEqual(sum(name == 'last.ckpt' for name, _ in exports), 3)
            self.assertEqual(exports[-1], ('last.ckpt', 2))
            saved = wrapper.load_checkpoint(Path(directory) / 'checkpoints/last.ckpt')
            self.assertEqual(saved['global_step'], 2)
            self.assertEqual(saved['ema']['num_updates'], 2)
            for name, value in model.backbone.state_dict().items():
                self.assertTrue(torch.equal(saved['state_dict']['backbone.' + name], value))
            for parameter, shadow in zip(model._get_parameters(), saved['ema']['shadow_params']):
                self.assertEqual(parameter.shape, shadow.shape)


if __name__ == '__main__':
    unittest.main()
