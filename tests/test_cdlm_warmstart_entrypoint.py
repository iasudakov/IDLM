"""Fresh initialization and full resume follow different CDLM entry paths."""

from contextlib import ExitStack
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

    def test_missing_resume_checkpoint_never_falls_back_to_fresh_mmd_weights(self):
        cfg, factory, model, trainer, data = self.fixture(resume=True, exists=False)
        with self.assertRaisesRegex(FileNotFoundError, 'CDLM resume checkpoint is missing'):
            main._train(factory, cfg, Mock(), object())
        model.initialize_from_checkpoint.assert_not_called()
        trainer.fit.assert_not_called()
        data.assert_not_called()


if __name__ == '__main__':
    unittest.main()
