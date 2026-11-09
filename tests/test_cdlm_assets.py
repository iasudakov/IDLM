"""Offline warm-start paths remain independent of original-teacher assets."""

from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf

from model_assets import configure_model_weights


class StudentAssetTest(unittest.TestCase):
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
