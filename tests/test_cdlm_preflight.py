"""CPU tests of checkpoint preflight before distributed workers are started."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra.core.hydra_config import HydraConfig

import cdlm_preflight as preflight
from test_cdlm_protocol import compose_cli


ROOT = Path(__file__).resolve().parents[1]
MMD = 'hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt'


class CheckpointPreflightTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cdlm preflight ')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.code = self.directory / 'code folder'
        self.code.mkdir()
        self.teacher = self.code / 'prepared checkpoints' / 'teacher weights, v=2.ckpt'
        self.teacher.parent.mkdir()
        self.teacher.write_bytes(b'teacher fixture')
        self.student = self.teacher.parent / 'student weights, v=3.ckpt'
        self.student.write_bytes(b'student fixture')
        self.enterContext(patch.dict(os.environ, {'INPUT_PATH': str(self.directory / 'mounted input')}))

    def config(self, *overrides):
        return preflight.compose_config(['+experiment=cdlm_tinygsm_mmd', *overrides],
                                         code_dir=self.code)

    def apply(self, original, resolved):
        return compose_cli('+experiment=cdlm_tinygsm_mmd', *original, *resolved)

    def test_identical_teacher_student_are_prepared_once_with_exact_asset_options(self):
        cfg = self.config('model_weights_dir=relative bundle')
        with patch.object(preflight, 'prepare_checkpoint', return_value=str(self.teacher)) as prepare:
            overrides = preflight.prepare_overrides(cfg, code_dir=self.code)
        prepare.assert_called_once_with(MMD, code_dir=self.code,
            input_dir=str(self.directory / 'mounted input'), weights_dir=self.code / 'relative bundle',
            snapshot_enabled=True)
        final = self.apply([], overrides)
        self.assertEqual(final.training.finetune_path, str(self.teacher))
        self.assertEqual(final.training.student_init_path, str(self.teacher))
        self.assertTrue(final.training.teacher_use_ema)
        self.assertTrue(final.training.student_init_use_ema)
        self.assertEqual(len(overrides), 2)

    def test_custom_teacher_student_paths_and_snapshot_flag_are_preserved(self):
        teacher_source, student_source = '/mounted/teacher weights.ckpt', '/mounted/student weights.ckpt'
        original = ['training.finetune_path=' + teacher_source,
                    'training.student_init_path=' + student_source,
                    'checkpointing.snapshot=False']
        cfg = self.config(*original)
        with patch.object(preflight, 'prepare_checkpoint',
                          side_effect=[str(self.teacher), str(self.student)]) as prepare:
            overrides = preflight.prepare_overrides(cfg, code_dir=self.code)
        self.assertEqual([call.args[0] for call in prepare.call_args_list], [teacher_source, student_source])
        self.assertTrue(all(not call.kwargs['snapshot_enabled'] for call in prepare.call_args_list))
        final = self.apply(original, overrides)
        self.assertEqual(final.training.finetune_path, str(self.teacher))
        self.assertEqual(final.training.student_init_path, str(self.student))
        self.assertFalse(final.checkpointing.snapshot)

    def test_explicit_null_student_stays_null(self):
        original = ['training.student_init_path=null']
        cfg = self.config(*original)
        with patch.object(preflight, 'prepare_checkpoint', return_value=str(self.teacher)) as prepare:
            overrides = preflight.prepare_overrides(cfg, code_dir=self.code)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(len(overrides), 1)
        self.assertIsNone(self.apply(original, overrides).training.student_init_path)

    def test_full_resume_prepares_teacher_but_never_fetches_unused_student(self):
        original = ['checkpointing.resume_from_ckpt=True',
                    'training.student_init_path=hf://unused/student/model.ckpt']
        cfg = self.config(*original)
        with patch.object(preflight, 'prepare_checkpoint', return_value=str(self.teacher)) as prepare:
            overrides = preflight.prepare_overrides(cfg, code_dir=self.code)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(prepare.call_args.args, (MMD,))
        final = self.apply(original, overrides)
        self.assertTrue(final.checkpointing.resume_from_ckpt)
        self.assertEqual(final.training.student_init_path, 'hf://unused/student/model.ckpt')

    def test_final_original_override_wins_before_asset_resolution(self):
        original = ['training.finetune_path=/ignored/first.ckpt',
                    'training.finetune_path=/selected/last teacher.ckpt',
                    'training.student_init_path=null']
        cfg = self.config(*original)
        with patch.object(preflight, 'prepare_checkpoint', return_value=str(self.teacher)) as prepare:
            resolved = preflight.prepare_overrides(cfg, code_dir=self.code)
        self.assertEqual(prepare.call_args.args, ('/selected/last teacher.ckpt',))
        self.assertEqual(self.apply(original, resolved).training.finetune_path, str(self.teacher))

    def test_hydra_runtime_paths_resolve_against_launch_and_output_directories(self):
        prior = HydraConfig.instance().cfg
        cfg = self.config(
            'hydra.run.dir=prepared output',
            'training.finetune_path=${hydra:runtime.cwd}/assets/teacher.ckpt',
            'training.student_init_path=${hydra:runtime.output_dir}/student.ckpt',
            'model_weights_dir=${hydra:runtime.cwd}/bundle')
        self.assertIs(HydraConfig.instance().cfg, prior)
        self.assertEqual(cfg.training.finetune_path, str(self.code / 'assets/teacher.ckpt'))
        self.assertEqual(cfg.training.student_init_path, str(self.code / 'prepared output/student.ckpt'))
        self.assertEqual(cfg.model_weights_dir, str(self.code / 'bundle'))
        with patch.object(preflight, 'prepare_checkpoint',
                          side_effect=[str(self.teacher), str(self.student)]) as prepare:
            preflight.prepare_overrides(cfg, code_dir=self.code)
        self.assertEqual(prepare.call_args_list[0].args[0], str(self.code / 'assets/teacher.ckpt'))
        self.assertEqual(prepare.call_args_list[1].args[0], str(self.code / 'prepared output/student.ckpt'))

    def test_offline_teacher_guard_rejects_hub_source_before_preparing_assets(self):
        cfg = preflight.compose_config(['+experiment=cdlm_tinygsm', 'model_weights_dir=/offline/bundle'],
                                        code_dir=self.code)
        with patch.object(preflight, 'prepare_checkpoint') as prepare:
            with self.assertRaisesRegex(ValueError, 'teacher_allow_hub'):
                preflight.prepare_overrides(cfg, code_dir=self.code)
        prepare.assert_not_called()

    def test_non_cdlm_and_missing_teacher_are_rejected_before_asset_io(self):
        cfg = self.config()
        cfg.algo.name = 'sdtt'
        with patch.object(preflight, 'prepare_checkpoint') as prepare:
            with self.assertRaisesRegex(ValueError, 'algo=cdlm'):
                preflight.prepare_overrides(cfg, code_dir=self.code)
            cfg.algo.name = 'cdlm'
            cfg.training.finetune_path = None
            with self.assertRaisesRegex(ValueError, 'finetune_path'):
                preflight.prepare_overrides(cfg, code_dir=self.code)
        prepare.assert_not_called()

    def test_cli_writes_only_successful_nul_arguments_and_failure_removes_stale_output(self):
        output = self.directory / 'preflight arguments.bin'
        args = ['--output', str(output), '+experiment=cdlm_tinygsm_mmd']
        with patch.object(preflight.Path, 'cwd', return_value=self.code), \
                patch.object(preflight, 'prepare_checkpoint', return_value=str(self.teacher)), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(preflight.main(args), 0)
        encoded = output.read_bytes()
        self.assertTrue(encoded.endswith(b'\0'))
        self.assertNotIn(b'\n', encoded)
        overrides = encoded.decode().split('\0')[:-1]
        final = self.apply([], overrides)
        self.assertEqual(final.training.finetune_path, str(self.teacher))
        self.assertEqual(final.training.student_init_path, str(self.teacher))
        # A separate student's failure must leave no consumable partial/stale result.
        args.append('training.student_init_path=hf://other/student/model.ckpt')
        with patch.object(preflight.Path, 'cwd', return_value=self.code), \
                patch.object(preflight, 'prepare_checkpoint',
                             side_effect=[str(self.teacher), FileNotFoundError('missing student')]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(preflight.main(args), 1)
        self.assertFalse(output.exists())
        self.assertIn('Preflight failed before GPU launch', errors.getvalue())
        self.assertIn('missing student', errors.getvalue())

    def test_composition_in_fresh_process_never_imports_training_or_torch(self):
        script = '''import json, sys
from cdlm_preflight import compose_config
cfg = compose_config(['+experiment=cdlm_tinygsm_mmd'])
print(json.dumps({'teacher': cfg.training.finetune_path,
                  'torch': 'torch' in sys.modules, 'trainer': 'trainer_base' in sys.modules}))
'''
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
        result = subprocess.run([sys.executable, '-c', script], cwd=self.code, env=env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['teacher'], MMD)
        self.assertFalse(report['torch'])
        self.assertFalse(report['trainer'])


if __name__ == '__main__':
    unittest.main()
