"""Fresh initialization and full resume follow different CDLM entry paths."""

from contextlib import ExitStack
import tempfile
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

import main


class WarmStartEntrypointTest(unittest.TestCase):
    def fixture(self, *, resume=False, exists=True):
        cfg = OmegaConf.create({
            'algo': {'name': 'cdlm'}, 'debug': False, 'callbacks': {},
            'logger': {'kind': 'logger'}, 'strategy': {'kind': 'strategy'},
            'trainer': {'kind': 'trainer'},
            'training': {'finetune_path': 'original-mdlm.ckpt',
                         'student_init_path': '/missing-after-resume/mmd.ckpt',
                         'student_init_use_ema': True},
            'checkpointing': {'resume_from_ckpt': resume,
                              'resume_ckpt_path': 'cdlm-last.ckpt',
                              'use_periodic_train_batch_checkpoint': False},
        })
        model, trainer = Mock(), Mock()
        factory = Mock(return_value=model)
        logger = Mock()

        def instantiate(config, *args, **kwargs):
            if config.kind == 'logger':
                return logger
            if config.kind == 'trainer':
                return trainer
            if config.kind == 'strategy':
                return object()
            raise AssertionError('Unexpected Hydra target.')

        context = ExitStack()
        self.addCleanup(context.close)
        context.enter_context(patch('main.hydra.utils.instantiate', side_effect=instantiate))
        context.enter_context(patch('main.utils.fsspec_exists', return_value=exists))
        data = context.enter_context(patch('main.dataloader.get_dataloaders',
                                          return_value=(Mock(), Mock())))
        context.enter_context(patch('main._cdlm_train_loader', return_value=(Mock(), 'teacher-cache-id')))
        context.enter_context(patch('main._print_batch'))
        return cfg, factory, model, trainer, data

    def test_fresh_run_passes_student_source_and_weight_choice(self):
        cfg, factory, model, trainer, _ = self.fixture()
        main._train(factory, cfg, Mock(), object())
        model.initialize_from_checkpoint.assert_called_once_with(
            'original-mdlm.ckpt', student_source='/missing-after-resume/mmd.ckpt',
            student_use_ema=True)
        self.assertIsNone(trainer.fit.call_args.kwargs['ckpt_path'])
        self.assertEqual(model._cache_fingerprint, 'teacher-cache-id')

    def test_full_resume_skips_external_student_initialization(self):
        cfg, factory, model, trainer, _ = self.fixture(resume=True)
        main._train(factory, cfg, Mock(), object())
        model.initialize_from_checkpoint.assert_not_called()
        self.assertEqual(trainer.fit.call_args.kwargs['ckpt_path'], 'cdlm-last.ckpt')

    def test_full_resume_never_downloads_the_original_hub_student(self):
        cfg, factory, model, trainer, _ = self.fixture(resume=True)
        cfg.training.student_init_path = 'hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt'
        with patch('sdtt_checkpoint.hf_hub_download') as download:
            main._train(factory, cfg, Mock(), object())
        download.assert_not_called()
        model.initialize_from_checkpoint.assert_not_called()
        self.assertEqual(trainer.fit.call_args.kwargs['ckpt_path'], 'cdlm-last.ckpt')

    def test_missing_resume_checkpoint_never_falls_back_to_fresh_mmd_weights(self):
        cfg, factory, model, trainer, data = self.fixture(resume=True, exists=False)
        with self.assertRaisesRegex(FileNotFoundError, 'CDLM resume checkpoint is missing'):
            main._train(factory, cfg, Mock(), object())
        model.initialize_from_checkpoint.assert_not_called()
        trainer.fit.assert_not_called()
        data.assert_not_called()

    def test_snapshot_restore_precedes_resume_check_and_installs_active_io_wrapper(self):
        from snapshot_checkpoint import SnapshotCheckpointIO
        cfg, factory, model, trainer, _ = self.fixture(resume=True)
        events = []
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpointing.update({'save_dir': directory, 'snapshot': True,
                                      'snapshot_namespace': 'cdlm_tinygsm'})
            with patch('nirvana_snapshot.restore', side_effect=lambda *a, **k: events.append('restored')) as restore, \
                 patch('main.utils.fsspec_exists', side_effect=lambda path: 'restored' in events):
                main._train(factory, cfg, Mock(), object())
            restore.assert_called_once_with(directory, enabled=True, namespace='cdlm_tinygsm')
            self.assertIsInstance(trainer.strategy.checkpoint_io, SnapshotCheckpointIO)
        self.assertEqual(trainer.fit.call_args.kwargs['ckpt_path'], 'cdlm-last.ckpt')
        model.initialize_from_checkpoint.assert_not_called()

    def test_fresh_snapshot_run_does_not_restore_old_student(self):
        from snapshot_checkpoint import SnapshotCheckpointIO
        cfg, factory, model, trainer, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpointing.update({'save_dir': directory, 'snapshot': True,
                                      'snapshot_namespace': 'cdlm_tinygsm'})
            with patch('nirvana_snapshot.restore') as restore:
                main._train(factory, cfg, Mock(), object())
            restore.assert_not_called()
            self.assertIsInstance(trainer.strategy.checkpoint_io, SnapshotCheckpointIO)
        model.initialize_from_checkpoint.assert_called_once()

    def test_existing_local_checkpoint_is_not_overwritten_by_an_older_snapshot(self):
        from snapshot_checkpoint import SnapshotCheckpointIO
        cfg, factory, model, trainer, _ = self.fixture(resume=True, exists=True)
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpointing.update({'save_dir': directory, 'snapshot': True,
                                      'snapshot_namespace': 'cdlm_tinygsm'})
            with patch('nirvana_snapshot.restore') as restore:
                main._train(factory, cfg, Mock(), object())
            restore.assert_not_called()
            self.assertIsInstance(trainer.strategy.checkpoint_io, SnapshotCheckpointIO)
        model.initialize_from_checkpoint.assert_not_called()
        self.assertEqual(trainer.fit.call_args.kwargs['ckpt_path'], 'cdlm-last.ckpt')

    def test_disabled_snapshot_preserves_existing_checkpoint_backend(self):
        cfg, factory, model, trainer, _ = self.fixture()
        cfg.checkpointing.snapshot = False
        backend = trainer.strategy.checkpoint_io
        with patch('nirvana_snapshot.restore') as restore:
            main._train(factory, cfg, Mock(), object())
        restore.assert_not_called()
        self.assertIs(trainer.strategy.checkpoint_io, backend)


if __name__ == '__main__':
    unittest.main()
