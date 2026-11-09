"""MMD student initialization must preserve CDLM's original teacher targets."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from cdlm import CDLM
from cdlm_trajectories import (CDLMTrajectoryDataset, build_cache_identity,
                               prepare_trajectories)
from test_cdlm_training import (CPUDenoiser, Original, Tokenizer,
                                config as make_config, trainer as make_trainer)


class StudentInitializationTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('models.dit.DIT', CPUDenoiser))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = make_config(str(self.directory))
        self.tokenizer = Tokenizer()
        self.model = CDLM(self.config, self.tokenizer)
        self.teacher = {name: torch.full_like(value, (i + 1) * .125)
                        for i, (name, value) in enumerate(self.model.backbone.state_dict().items())}
        self.student = {name: value + .5 for name, value in self.teacher.items()}
        self.student_path = self.directory / 'trained-mmd.ckpt'
        torch.save({'state_dict': {'backbone.' + name: value
                                  for name, value in self.teacher.items()}},
                   self.config.training.finetune_path)

    def write_student(self, state=None, *, ema=None, ema_only=False):
        state = self.student if state is None else state
        # MMD full checkpoints also contain unrelated teacher and fake tensors;
        # compiled student names must still map strictly onto the native DiT.
        checkpoint = {'state_dict': {
            **{'backbone._orig_mod.' + name: value for name, value in state.items()},
            'teacher_model.unused': torch.tensor(123.),
            'fake_model.unused': torch.tensor(456.)},
            'optimizer_states': [{'old_student_optimizer': True}],
            'global_step': 999}
        if ema is not None:
            checkpoint['ema'] = {'shadow_params': ema, 'num_updates': 777, 'decay': .9}
        if ema_only:
            checkpoint['checkpoint_weights'] = 'ema'
        torch.save(checkpoint, self.student_path)

    def assert_teacher_unchanged(self):
        for name, tensor in self.model.teacher_head.state_dict().items():
            self.assertTrue(torch.equal(tensor, self.teacher['output_layer.linear.' + name]))
        self.assertFalse(self.model.teacher_head.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.teacher_head.parameters()))

    def assert_fresh_student(self, expected):
        for name, tensor in self.model.backbone.state_dict().items():
            self.assertTrue(torch.equal(tensor, expected[name]), name)
        parameters = [p for p in self.model._get_parameters() if p.requires_grad]
        self.assertEqual(len(parameters), len(self.model.ema.shadow_params))
        for parameter, shadow in zip(parameters, self.model.ema.shadow_params):
            self.assertTrue(torch.equal(parameter, shadow))
        self.assertEqual(self.model.ema.num_updates, 0)
        self.assertEqual(self.model.ema.decay, self.config.training.ema)
        self.assert_teacher_unchanged()

    def test_default_initialization_remains_the_raw_original_teacher(self):
        self.model.initialize_from_checkpoint(self.config.training.finetune_path)
        self.assert_fresh_student(self.teacher)

    def test_raw_mmd_student_does_not_replace_original_teacher_head(self):
        self.write_student()
        self.model.ema.num_updates = 31
        self.model.initialize_from_checkpoint(self.config.training.finetune_path,
                                              student_source=str(self.student_path))
        self.assert_fresh_student(self.student)

    def test_mmd_ema_initializes_only_student_and_resets_ema_history(self):
        expected = {name: value + 1. for name, value in self.student.items()}
        shadows = [expected[name] for name, _ in self.model.backbone.named_parameters()]
        self.write_student(ema=shadows)
        self.model.initialize_from_checkpoint(self.config.training.finetune_path,
                                              student_source=str(self.student_path),
                                              student_use_ema=True)
        self.assert_fresh_student(expected)

    def test_native_ema_only_export_requires_explicit_ema_selection(self):
        self.write_student(ema_only=True)
        with self.assertRaisesRegex(ValueError, 'student_init_use_ema=True'):
            self.model.initialize_from_checkpoint(self.config.training.finetune_path,
                                                  student_source=str(self.student_path))
        self.model.initialize_from_checkpoint(self.config.training.finetune_path,
                                              student_source=str(self.student_path),
                                              student_use_ema=True)
        self.assert_fresh_student(self.student)

    def test_requested_ema_must_exist_and_match_student_parameter_shapes(self):
        for shadows, message in [(None, 'no EMA'), ([torch.zeros(1)], 'EMA shapes')]:
            with self.subTest(shadows=shadows):
                self.write_student(ema=shadows)
                with self.assertRaisesRegex(ValueError, message):
                    self.model.initialize_from_checkpoint(self.config.training.finetune_path,
                                                          student_source=str(self.student_path),
                                                          student_use_ema=True)

    def test_incompatible_mmd_backbone_is_rejected_strictly(self):
        incompatible = {**self.student, 'output_layer.linear.weight': torch.zeros(3, 8)}
        self.write_student(state=incompatible)
        with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
            self.model.initialize_from_checkpoint(self.config.training.finetune_path,
                                                  student_source=str(self.student_path))

    def test_student_init_fields_do_not_change_original_teacher_cache_identity(self):
        before = build_cache_identity(self.config, self.tokenizer, Original())
        OmegaConf.update(self.config, 'training.student_init_path',
                         str(self.student_path), force_add=True)
        OmegaConf.update(self.config, 'training.student_init_use_ema', True, force_add=True)
        # No student checkpoint exists: identity construction must not read it.
        self.assertFalse(self.student_path.exists())
        after = build_cache_identity(self.config, self.tokenizer, Original())
        self.assertEqual(before, after)

    def test_full_resume_survives_deleted_mmd_initialization_checkpoint(self):
        self.write_student()
        self.model.initialize_from_checkpoint(self.config.training.finetune_path,
                                              student_source=str(self.student_path))
        collector = CDLM(self.config, self.tokenizer, inference_only=True)
        collector.load_evaluation_weights(self.config.training.finetune_path, use_ema=False)
        prepare_trajectories(collector, self.config, self.tokenizer, Original())
        identity = build_cache_identity(self.config, self.tokenizer, Original())
        loader = DataLoader(CDLMTrajectoryDataset(self.config.algo.trajectory_cache_dir,
                                                 identity), batch_size=2)
        self.model._cache_fingerprint = identity['fingerprint']
        make_trainer(1).fit(self.model, loader)
        checkpoint = self.directory / 'checkpoints/last.ckpt'
        self.student_path.unlink()
        OmegaConf.update(self.config, 'training.student_init_path',
                         str(self.student_path), force_add=True)
        restored = CDLM(self.config, self.tokenizer)
        restored._cache_fingerprint = identity['fingerprint']
        with patch.object(restored, 'initialize_from_checkpoint',
                          side_effect=AssertionError('Resume must not initialize weights.')):
            trainer = make_trainer(2)
            trainer.fit(restored, loader, ckpt_path=str(checkpoint))
        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(restored.ema.num_updates, 2)
        for name, value in restored.teacher_head.state_dict().items():
            self.assertTrue(torch.equal(value, self.teacher['output_layer.linear.' + name]))
        self.assertTrue(torch.load(checkpoint, weights_only=False)['optimizer_states'])


if __name__ == '__main__':
    unittest.main()
