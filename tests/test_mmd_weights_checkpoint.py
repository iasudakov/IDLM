"""Student-only periodic weights and complete final MMD snapshot on CPU."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import lightning as L
import torch
from torch.utils.data import DataLoader

from mmd_finetune import MMDFinetune
import mmd_weights_checkpoint as exports
from snapshot_checkpoint import SnapshotCheckpointIO
import test_mmd_finetune as fixtures


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 2)
        self.backbone.register_buffer('persistent_count', torch.tensor(8))
        self.teacher = torch.nn.Linear(3, 2)
        self.other = torch.nn.Parameter(torch.ones(4))
        self.ema = SimpleNamespace(shadow_params=[parameter.detach().clone() + 10
                                                 for parameter in self._get_parameters()])
        self.config = {'algo': {'name': 'mmd_finetune'}}

    def _get_parameters(self):
        # Deliberately different from backbone.state_dict() order, with an
        # unrelated tracked parameter, to catch positional EMA mapping bugs.
        return [self.other, *reversed(list(self.backbone.parameters()))]


def fake_trainer(step, *, rank_zero=True, broadcast=None):
    return SimpleNamespace(global_step=step, current_epoch=2, is_global_zero=rank_zero,
                           strategy=SimpleNamespace(broadcast=broadcast or Mock(side_effect=lambda obj, src: obj)))


class MMDWeightsExportTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.module = TinyModule()

    def test_ema_maps_parameter_identity_retains_buffers_and_does_not_mutate_live_weights(self):
        before = {name: value.clone() for name, value in self.module.state_dict().items()}
        payload = exports.student_weights(self.module, global_step=1000, epoch=2)
        self.assertEqual(set(payload), {'state_dict', 'checkpoint_weights', 'global_step',
                                       'epoch', 'hyper_parameters'})
        self.assertEqual(payload['checkpoint_weights'], 'ema')
        self.assertEqual(payload['global_step'], 1000)
        self.assertEqual(payload['epoch'], 2)
        for name, parameter in self.module.backbone.named_parameters():
            self.assertTrue(torch.equal(payload['state_dict']['backbone.' + name], parameter.detach() + 10))
        self.assertEqual(payload['state_dict']['backbone.persistent_count'].item(), 8)
        self.assertTrue(all(name.startswith('backbone.') for name in payload['state_dict']))
        self.assertFalse(any(value.requires_grad for value in payload['state_dict'].values()))
        self.assertTrue(all(value.device.type == 'cpu' for value in payload['state_dict'].values()))
        for name, value in self.module.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        # Export tensors are independent copies, including EMA and buffer tensors.
        payload['state_dict']['backbone.weight'].zero_()
        payload['state_dict']['backbone.persistent_count'].zero_()
        self.assertEqual(self.module.backbone.persistent_count.item(), 8)
        self.assertFalse(torch.equal(self.module.ema.shadow_params[-1],
                                     payload['state_dict']['backbone.weight']))

    def test_no_ema_exports_raw_student_only(self):
        self.module.ema = None
        payload = exports.student_weights(self.module, global_step=9, epoch=1)
        self.assertEqual(payload['checkpoint_weights'], 'raw')
        for name, value in self.module.backbone.state_dict().items():
            self.assertTrue(torch.equal(payload['state_dict']['backbone.' + name], value))
        self.assertNotIn('ema', payload)

    def test_bad_ema_count_or_shape_is_rejected_without_swapping_weights(self):
        original = [shadow.clone() for shadow in self.module.ema.shadow_params]
        self.module.ema.shadow_params = original[:-1]
        with self.assertRaisesRegex(ValueError, 'count'):
            exports.student_weights(self.module, global_step=1, epoch=0)
        self.module.ema.shadow_params = [torch.ones(19), *original[1:]]
        with self.assertRaisesRegex(ValueError, 'shape'):
            exports.student_weights(self.module, global_step=1, epoch=0)

    def test_optimizer_cadence_deduplicates_accumulation_and_retains_numbered_files(self):
        callback = exports.MMDWeightsCheckpoint(self.root, every_n_train_steps=2, snapshot=False)
        with patch.object(exports.nirvana_snapshot, 'save') as save:
            for step in (0, 1, 1, 2, 2, 3, 3, 4, 4):
                trainer = fake_trainer(step)
                callback.on_train_batch_end(trainer, self.module, None, None, step)
            self.assertEqual(save.call_count, 2)
            self.assertTrue(all(not call.kwargs['enabled'] for call in save.call_args_list))
        self.assertEqual(sorted(path.name for path in (self.root / 'checkpoints').iterdir()),
                         ['step=00000002.ckpt', 'step=00000004.ckpt'])
        self.assertEqual(callback.state_dict(), {'last_saved_step': 4})
        restored = exports.MMDWeightsCheckpoint(self.root / 'new-container', every_n_train_steps=2)
        self.assertEqual(callback.state_key, restored.state_key)
        restored.load_state_dict(callback.state_dict())
        with patch.object(exports, '_atomic_save') as write:
            trainer = fake_trainer(4)
            restored.on_train_batch_end(trainer, self.module, None, None, 0)
        write.assert_not_called()
        trainer.strategy.broadcast.assert_not_called()

    def test_atomic_write_failure_propagates_on_both_ranks_without_partial_file(self):
        callback = exports.MMDWeightsCheckpoint(self.root, every_n_train_steps=1)
        message = 'OSError: disk full'
        with patch.object(exports.torch, 'save', side_effect=OSError('disk full')), \
                patch.object(exports.nirvana_snapshot, 'save') as publish:
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                callback.on_train_batch_end(fake_trainer(1), self.module, None, None, 0)
        publish.assert_not_called()
        self.assertEqual(list((self.root / 'checkpoints').iterdir()), [])
        self.assertEqual(callback.last_saved_step, 0)
        peer = exports.MMDWeightsCheckpoint(self.root, every_n_train_steps=1)
        with patch.object(exports, '_atomic_save') as write:
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                peer.on_train_batch_end(fake_trainer(1, rank_zero=False,
                    broadcast=Mock(return_value=message)), self.module, None, None, 0)
        write.assert_not_called()
        self.assertEqual(peer.last_saved_step, 0)

    def test_snapshot_failure_preserves_completed_weights_and_continues(self):
        callback = exports.MMDWeightsCheckpoint(self.root, every_n_train_steps=1)
        with patch.object(exports.nirvana_snapshot, 'save', side_effect=RuntimeError('snapshot unavailable')):
            callback.on_train_batch_end(fake_trainer(1), self.module, None, None, 0)
        saved = torch.load(self.root / 'checkpoints/step=00000001.ckpt', weights_only=False)
        self.assertEqual(saved['global_step'], 1)
        self.assertEqual(saved['checkpoint_weights'], 'ema')
        self.assertEqual(callback.last_saved_step, 1)

    def test_only_rank_zero_writes_and_broadcast_reaches_peers(self):
        callback = exports.MMDWeightsCheckpoint(self.root, every_n_train_steps=1)
        trainer = fake_trainer(1, rank_zero=False)
        with patch.object(exports, '_atomic_save') as write, \
                patch.object(exports.nirvana_snapshot, 'save') as publish:
            callback.on_train_batch_end(trainer, self.module, None, None, 0)
        write.assert_not_called()
        publish.assert_not_called()
        trainer.strategy.broadcast.assert_called_once_with(None, src=0)
        self.assertEqual(callback.last_saved_step, 1)


class MMDWeightsSnapshotIntegrationTest(unittest.TestCase):
    def test_periodic_student_only_and_full_final_checkpoint_survive_snapshot(self):
        for interval in (1, 2):
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as directory, \
                    patch('models.dit.DIT', fixtures.Denoiser):
                torch.manual_seed(97)
                root = Path(directory).resolve()
                run, store = root / 'mmd_run', root / 'snapshot'
                run.mkdir()
                store.mkdir()
                original = store / 'cdlm_tinygsm' / 'checkpoints' / 'last.ckpt'
                original.parent.mkdir(parents=True)
                original.write_bytes(b'original CDLM snapshot must remain untouched')
                original_bytes = original.read_bytes()
                cfg = fixtures.config(str(run))
                cfg.checkpointing.snapshot = True
                cfg.checkpointing.snapshot_namespace = 'mmd_from_cdlm_tinygsm'
                model = MMDFinetune(cfg, fixtures.Tokenizer())
                student, features = root / 'cdlm.ckpt', root / 'features.ckpt'
                torch.save({'state_dict': {'backbone.' + name: value.clone()
                                          for name, value in model.backbone.state_dict().items()},
                            'ema': model.ema.state_dict()}, student)
                feature_weights = {name: value.clone() + .3
                                   for name, value in model.backbone.state_dict().items()}
                torch.save({'state_dict': {'backbone.' + name: value
                                          for name, value in feature_weights.items()}}, features)
                model.initialize_from_checkpoint(str(student), str(features))
                periodic = exports.MMDWeightsCheckpoint(run, every_n_train_steps=interval)
                fit = L.Trainer(accelerator='cpu', devices=1, max_steps=3, max_epochs=-1,
                                accumulate_grad_batches=2, logger=False, callbacks=[periodic],
                                enable_checkpointing=False, enable_progress_bar=False,
                                enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0)
                fit.strategy.checkpoint_io = SnapshotCheckpointIO(
                    fit.strategy.checkpoint_io, str(run), namespace='mmd_from_cdlm_tinygsm')
                publications = []

                def dump(path):
                    self.assertEqual(Path(path).resolve(), store)
                    published = store / 'mmd_from_cdlm_tinygsm' / 'checkpoints'
                    publications.append({p.name: torch.load(p, map_location='cpu', weights_only=False)['global_step']
                                         for p in published.glob('*.ckpt')})

                sdk = SimpleNamespace(snapshot=SimpleNamespace(get_snapshot_path=lambda: str(store),
                                                                dump_snapshot=dump))
                loader = DataLoader([fixtures.Original()[0] for _ in range(16)], batch_size=2)
                with patch('nirvana_snapshot._sdk', return_value=sdk):
                    fit.fit(model, loader)
                self.assertEqual(fit.global_step, 3)
                self.assertEqual(publications[-1]['last.ckpt'], 3)
                expected = [f'step={step:08d}.ckpt' for step in range(interval, 4, interval)]
                numbered = sorted((run / 'checkpoints').glob('step=*.ckpt'))
                self.assertEqual([path.name for path in numbered], expected)
                self.assertEqual(original.read_bytes(), original_bytes)
                for path in numbered:
                    published = store / 'mmd_from_cdlm_tinygsm' / 'checkpoints' / path.name
                    self.assertEqual(path.read_bytes(), published.read_bytes())
                    saved = torch.load(published, map_location='cpu', weights_only=False)
                    self.assertEqual(saved['checkpoint_weights'], 'ema')
                    self.assertEqual(set(saved), {'state_dict', 'checkpoint_weights', 'global_step',
                                                  'epoch', 'hyper_parameters'})
                    self.assertTrue(all(name.startswith('backbone.') for name in saved['state_dict']))
                    inference = MMDFinetune(cfg, fixtures.Tokenizer(), inference_only=True)
                    inference.load_evaluation_weights(str(published), use_ema=True)
                    for name, actual in inference.backbone.state_dict().items():
                        self.assertTrue(torch.equal(actual, saved['state_dict']['backbone.' + name]))
                last = store / 'mmd_from_cdlm_tinygsm/checkpoints/last.ckpt'
                final = torch.load(last, map_location='cpu', weights_only=False)
                self.assertEqual(final['mmd_finetune_schema'], 1)
                self.assertTrue(final['optimizer_states'])
                self.assertTrue(final['lr_schedulers'])
                self.assertTrue(final['loops'])
                self.assertEqual(final['ema']['num_updates'], 3)
                self.assertIn(periodic.state_key, final['callbacks'])
                self.assertEqual(final['callbacks'][periodic.state_key]['last_saved_step'], (3 // interval) * interval)
                for name, expected in feature_weights.items():
                    self.assertTrue(torch.equal(final['state_dict']['teacher.' + name], expected))
                for name, actual in model.backbone.state_dict().items():
                    self.assertTrue(torch.equal(final['state_dict']['backbone.' + name], actual))


if __name__ == '__main__':
    unittest.main()
