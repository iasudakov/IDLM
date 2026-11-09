"""Offline warm-start paths remain independent of original-teacher assets."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from model_assets import configure_model_weights, resolve_checkpoint


class StudentAssetTest(unittest.TestCase):
    def test_hub_student_uri_is_preserved_without_local_checkpoint_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'bundle'
            bundle.mkdir()
            teacher = bundle / 'teacher.ckpt'
            teacher.touch()
            student = 'hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt'
            evaluation = 'hf://example/evaluation/model.ckpt'
            config = OmegaConf.create({
                'model_weights_dir': 'bundle', 'algo': {'backbone': 'dit'},
                'training': {'finetune_path': 'teacher.ckpt', 'student_init_path': student},
                'eval': {'checkpoint_path': evaluation}, 'adversarial_distill': {},
            })
            with patch('model_assets.resolve_checkpoint', wraps=resolve_checkpoint) as resolve:
                configure_model_weights(config, base_dir=root)
            self.assertEqual(config.training.finetune_path, str(teacher.resolve()))
            self.assertEqual(config.training.student_init_path, student)
            # Only student HF initialization is exempt from the existing path
            # normalizer; evaluation retains its original offline behavior.
            self.assertEqual([call.args[0] for call in resolve.call_args_list],
                             ['teacher.ckpt', evaluation])

    def test_bundle_resolves_original_teacher_and_separate_student_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'bundle'
            student = bundle / 'student/mmd.ckpt'
            student.parent.mkdir(parents=True)
            student.touch()
            teacher = bundle / 'teacher.ckpt'
            teacher.touch()
            config = OmegaConf.create({
                'model_weights_dir': 'bundle', 'algo': {'backbone': 'dit'},
                'training': {'finetune_path': 'teacher.ckpt',
                             'student_init_path': 'student/mmd.ckpt'},
                'eval': {}, 'adversarial_distill': {},
            })
            configure_model_weights(config, base_dir=root)
            self.assertEqual(config.training.finetune_path, str(teacher.resolve()))
            self.assertEqual(config.training.student_init_path, str(student.resolve()))
            # Preparation, resume and evaluation do not need the external
            # initialization checkpoint; fresh training validates it on use.
            student.unlink()
            configure_model_weights(config, base_dir=root)
            self.assertEqual(config.training.student_init_path, str(student.resolve()))


if __name__ == '__main__':
    unittest.main()
