"""CDLM-to-MMD starts a fresh optimizer stage and never enters trajectory prep."""

from unittest.mock import Mock, patch
import unittest

from omegaconf import OmegaConf

import main
import test_cdlm_warmstart_entrypoint as cdlm_entrypoint_tests


class MMDEntrypointTest(unittest.TestCase):
    def fixture(self, *, resume=False, exists=True):
        cfg, factory, model, trainer, data = cdlm_entrypoint_tests.WarmStartEntrypointTest.fixture(
            self, resume=resume, exists=exists)
        cfg.algo.name = 'mmd_finetune'
        cfg.training.student_init_path = '/state/cdlm/checkpoints/last.ckpt'
        cfg.training.student_init_use_ema = False
        cfg.training.teacher_use_ema = False
        cfg.mmd = OmegaConf.create({'retain_student_ema': True})
        cfg.checkpointing.resume_ckpt_path = 'mmd-last.ckpt'
        return cfg, factory, model, trainer, data

    def test_fresh_stage_loads_separate_student_and_feature_backbones(self):
        cfg, factory, model, trainer, _ = self.fixture()
        with patch('main._cdlm_train_loader') as trajectories:
            main._train(factory, cfg, Mock(), object())
        trajectories.assert_not_called()
        factory.load_from_checkpoint.assert_not_called()
        model.initialize_from_checkpoint.assert_called_once_with(
            '/state/cdlm/checkpoints/last.ckpt', feature_source='original-mdlm.ckpt',
            student_use_ema=False, retain_ema=True, feature_use_ema=False)
        self.assertIsNone(trainer.fit.call_args.kwargs['ckpt_path'])

    def test_mmd_resume_does_not_reinitialize_from_cdlm(self):
        cfg, factory, model, trainer, _ = self.fixture(resume=True)
        cfg.training.student_init_path = None
        cfg.training.finetune_path = ''
        with patch('main._cdlm_train_loader') as trajectories:
            main._train(factory, cfg, Mock(), object())
        trajectories.assert_not_called()
        model.initialize_from_checkpoint.assert_not_called()
        self.assertEqual(trainer.fit.call_args.kwargs['ckpt_path'], 'mmd-last.ckpt')

    def test_missing_mmd_resume_fails_without_fresh_fallback(self):
        cfg, factory, model, trainer, data = self.fixture(resume=True, exists=False)
        with self.assertRaisesRegex(FileNotFoundError, 'MMD resume checkpoint is missing'):
            main._train(factory, cfg, Mock(), object())
        data.assert_not_called()
        model.initialize_from_checkpoint.assert_not_called()
        trainer.fit.assert_not_called()

    def test_fresh_stage_requires_selected_student(self):
        cfg, factory, model, trainer, _ = self.fixture()
        cfg.training.student_init_path = None
        with self.assertRaisesRegex(ValueError, 'snapshot preflight'):
            main._train(factory, cfg, Mock(), object())
        model.initialize_from_checkpoint.assert_not_called()
        trainer.fit.assert_not_called()

    def test_inference_loads_only_evaluation_weights(self):
        cfg, factory, model, _, _ = self.fixture()
        cfg.eval = {'checkpoint_path': 'trained-mmd.ckpt', 'disable_ema': False}
        tokenizer = object()
        result = main._load_from_checkpoint(factory, cfg, tokenizer)
        factory.assert_called_once_with(cfg, tokenizer=tokenizer, inference_only=True)
        model.load_evaluation_weights.assert_called_once_with('trained-mmd.ckpt', use_ema=True)
        model.initialize_from_checkpoint.assert_not_called()
        self.assertIs(result, model.to.return_value)


if __name__ == '__main__':
    unittest.main()
