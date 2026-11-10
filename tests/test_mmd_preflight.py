"""CDLM snapshot to MMD preflight/launch checks without GPU or training."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import mmd_preflight as preflight
from test_cdlm_protocol import compose_cli, VALIDATION_THRESHOLDS


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = '+experiment=mmd_from_cdlm_tinygsm'
RESOLVED = ('checkpointing.save_dir', 'checkpointing.snapshot_namespace',
            'mmd.cdlm_snapshot_root', 'mmd.cdlm_snapshot_namespace')


def override(key, value):
    return key + '=' + json.dumps(str(value))


class MMDPreflightTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='mmd preflight '))).resolve()
        self.code = self.base / 'code folder'
        self.code.mkdir()
        self.snapshot = self.base / 'input snapshot, v=2'
        self.snapshot.mkdir()
        self.output = self.code / 'new training, v=3'
        self.teacher = self.code / 'checkpoints' / 'MDLM feature weights, v=1.ckpt'
        self.teacher.parent.mkdir()
        torch.save({'state_dict': {'backbone.weight': torch.full((2, 3), 1.)}}, self.teacher)
        self.enterContext(patch.dict(os.environ, {'INPUT_PATH': str(self.base / 'inputs')}))
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))

    def checkpoint(self, name='last.ckpt', step=10, namespace='cdlm_tinygsm'):
        path = self.snapshot / namespace / 'checkpoints' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'global_step': step, 'epoch': 3, 'cdlm_cache_fingerprint': 'fixture-cache',
                    'state_dict': {'backbone.weight': torch.full((2, 3), step * .125)},
                    'hyper_parameters': {'config': {'algo': {'name': 'cdlm'}}}}, path)
        return path

    def arguments(self, *extra):
        return [EXPERIMENT,
                override('mmd.cdlm_snapshot_root', self.snapshot),
                override('training.finetune_path', self.teacher),
                override('checkpointing.save_dir', self.output),
                override('hydra.run.dir', self.output), *extra]

    def config(self, *extra):
        return preflight.compose_config(self.arguments(*extra), code_dir=self.code,
                                         resolved_fields=RESOLVED)

    def test_recipe_matches_final_paper_loss_budget_and_four_gpu_batch(self):
        cfg = self.config()
        self.assertEqual(cfg.algo.name, 'mmd_finetune')
        self.assertEqual(cfg.mmd.kernel, 'rbf')
        self.assertEqual(cfg.mmd.group_size, 2)
        self.assertEqual(cfg.mmd.draws_per_candidate, 1)
        self.assertFalse(cfg.mmd.average_candidates)
        self.assertEqual(cfg.mmd.feature_layer, 6)
        self.assertEqual(cfg.mmd.rbf_alpha, 6e-5)
        self.assertEqual(cfg.mmd.kernel_block_size, 0)
        self.assertEqual(cfg.mmd.feature_batch_size, 8)
        self.assertEqual(cfg.mmd.student_forward_batch_size, 8)
        self.assertTrue(cfg.mmd.student_activation_checkpointing)
        self.assertEqual(cfg.trainer.max_steps, 7000)
        self.assertEqual(cfg.trainer.max_epochs, -1)
        self.assertEqual(cfg.lr_scheduler.num_warmup_steps, 2500)
        self.assertEqual(cfg.optim.lr, 1e-5)
        self.assertEqual(cfg.trainer.devices, 4)
        self.assertEqual(cfg.loader.batch_size, 32)
        self.assertEqual(cfg.trainer.accumulate_grad_batches, 4)
        self.assertEqual(cfg.loader.global_batch_size, 512)
        self.assertEqual(cfg.trainer.devices * cfg.loader.batch_size
                         * cfg.trainer.accumulate_grad_batches, cfg.loader.global_batch_size)
        self.assertFalse(cfg.training.student_init_use_ema)
        self.assertFalse(cfg.training.teacher_use_ema)
        self.assertTrue(cfg.mmd.retain_student_ema)
        self.assertEqual(cfg.training.loss_precision, 'bf16')
        self.assertEqual(cfg.trainer.precision, 'bf16-mixed')
        self.assertFalse(cfg.adversarial_distill.is_distill)
        self.assertFalse(cfg.checkpointing.resume_from_ckpt)
        self.assertFalse(cfg.checkpointing.snapshot_trajectories)
        self.assertTrue(cfg.checkpointing.snapshot)
        self.assertEqual(cfg.checkpointing.snapshot_namespace, 'mmd_from_cdlm_tinygsm')
        periodic = cfg.callbacks.checkpoint_every_n_steps
        self.assertEqual(periodic.every_n_train_steps, 500)
        self.assertEqual(periodic.save_top_k, 0)
        self.assertFalse(periodic.save_last)
        self.assertFalse(periodic.save_on_train_epoch_end)
        weights = cfg.callbacks.mmd_weights
        self.assertEqual(weights._target_, 'mmd_weights_checkpoint.MMDWeightsCheckpoint')
        self.assertEqual(weights.every_n_train_steps, 500)
        self.assertTrue(weights.snapshot)
        self.assertEqual(weights.namespace, 'mmd_from_cdlm_tinygsm')

    def test_recipe_disables_ancestral_preserving_confidence_and_data_protocol(self):
        cfg = self.config()
        self.assertIsNone(cfg.eval.gsm8k_ancestral_steps)
        self.assertIsNone(cfg.eval.gsm8k_ancestral_seeds)
        self.assertFalse(cfg.eval.gsm8k_ancestral_final_denoise)
        self.assertFalse(cfg.eval.gsm8k_greedy_token)
        self.assertEqual(list(cfg.eval.gsm8k_pareto_thresholds), VALIDATION_THRESHOLDS)
        self.assertEqual(cfg.eval.gsm8k_pareto_temperature, 0.)
        self.assertEqual(cfg.trainer.val_check_interval, 2000)
        self.assertEqual(cfg.trainer.limit_val_batches, 1000)
        self.assertEqual(cfg.data.separator, r'\n')
        self.assertFalse(cfg.data.train_on_prompt)
        self.assertTrue(cfg.data.train_on_pad)
        self.assertFalse(cfg.data.wrap)
        self.assertTrue(cfg.data.filter_too_long)
        self.assertEqual(cfg.model.length, 512)

    def test_latest_actual_step_and_separate_feature_teacher_are_recorded(self):
        self.checkpoint(step=10)
        newest = self.checkpoint(name='CDLM step, v=2.ckpt', step=20)
        original = self.arguments()
        values = preflight.prepare_overrides(self.config(), code_dir=self.code)
        final = compose_cli(*original, *values)
        self.assertEqual(final.training.student_init_path, str(newest))
        self.assertEqual(final.training.finetune_path, str(self.teacher))
        manifest = self.output / 'mmd_initialization.json'
        self.assertEqual(final.mmd.initialization_manifest, str(manifest))
        provenance = json.loads(manifest.read_text())
        self.assertEqual(provenance['student']['global_step'], 20)
        self.assertEqual(provenance['student']['path'], str(newest))
        self.assertEqual(provenance['student']['weights'], 'raw')
        self.assertTrue(provenance['student']['retain_ema_history'])
        self.assertEqual(provenance['student']['sha256'], hashlib.sha256(newest.read_bytes()).hexdigest())
        self.assertEqual(provenance['feature_extractor']['path'], str(self.teacher))
        self.assertEqual(provenance['feature_extractor']['sha256'],
                         hashlib.sha256(self.teacher.read_bytes()).hexdigest())
        self.assertEqual(provenance['reference_run'], '3e0ac6b1-7a8f-43c3-af58-d9e21b886da7')
        self.assertEqual(provenance['reference_figure_checkpoint_step'], 7000)
        self.assertTrue(provenance['fresh_optimizer_and_scheduler'])

    def test_cli_roundtrips_nul_paths_with_commas_spaces_and_equals(self):
        newest = self.checkpoint(name='selected checkpoint, step=49.ckpt', step=49)
        output = self.base / 'arguments.bin'
        with patch.object(preflight.Path, 'cwd', return_value=self.code):
            status = preflight.main(['--output', str(output), *self.arguments()])
        self.assertEqual(status, 0)
        raw = output.read_bytes()
        self.assertTrue(raw.endswith(b'\0'))
        self.assertNotIn(b'\n', raw)
        values = raw.decode().split('\0')[:-1]
        self.assertEqual(len(values), 3)
        final = compose_cli(*self.arguments(), *values)
        self.assertEqual(final.training.student_init_path, str(newest))
        self.assertEqual(final.training.finetune_path, str(self.teacher))
        self.assertEqual(final.mmd.initialization_manifest, str(self.output / 'mmd_initialization.json'))

    def test_multiple_source_runs_require_choice_and_output_namespace_is_separate(self):
        first = self.checkpoint(namespace='cdlm_tinygsm', step=100)
        self.checkpoint(namespace='cdlm_tinygsm_ancestral_sweep', step=200)
        with self.assertRaisesRegex(ValueError, 'CDLM_SOURCE_NAMESPACE'):
            preflight.prepare_overrides(self.config(), code_dir=self.code)
        cfg = self.config('mmd.cdlm_snapshot_namespace=cdlm_tinygsm')
        values = preflight.prepare_overrides(cfg, code_dir=self.code)
        final = compose_cli(*self.arguments(), *values)
        self.assertEqual(final.training.student_init_path, str(first))
        for destination in ('cdlm_tinygsm', 'cdlm_tinygsm/child', '.', '../escape', '/absolute'):
            with self.subTest(destination=destination):
                cfg = self.config('mmd.cdlm_snapshot_namespace=cdlm_tinygsm',
                                  override('checkpointing.snapshot_namespace', destination))
                with self.assertRaisesRegex(ValueError, 'separate'):
                    preflight.prepare_overrides(cfg, code_dir=self.code)

    def test_output_directory_cannot_modify_input_snapshot(self):
        self.checkpoint()
        for destination in (self.snapshot, self.snapshot / 'new-mmd'):
            with self.subTest(destination=destination):
                cfg = self.config(override('checkpointing.save_dir', destination))
                with self.assertRaisesRegex(ValueError, 'outside'):
                    preflight.prepare_overrides(cfg, code_dir=self.code)
        self.assertFalse((self.snapshot / 'mmd_initialization.json').exists())

    def test_full_mmd_resume_checks_its_own_state_without_cdlm_selection_or_assets(self):
        cfg = self.config('checkpointing.resume_from_ckpt=True',
                          'training.student_init_path=hf://unused/student/checkpoint.ckpt',
                          'training.finetune_path=hf://unused/teacher/checkpoint.ckpt',
                          'mmd.cdlm_snapshot_root=/missing/previous-state')
        target = str(self.output / 'checkpoints' / 'last.ckpt')
        expected = ['checkpointing.resume_ckpt_path=' + json.dumps(target)]
        with patch.object(preflight, 'validate_resume_checkpoint', return_value=target) as validate, \
                patch.object(preflight, 'select_checkpoint') as select, \
                patch.object(preflight, 'prepare_checkpoint') as prepare:
            self.assertEqual(preflight.prepare_overrides(cfg, code_dir=self.code), expected)
        validate.assert_called_once_with(cfg, code_dir=self.code)
        select.assert_not_called()
        prepare.assert_not_called()
        self.assertFalse(self.output.exists())
        output = self.base / 'resume-arguments.bin'
        with patch.object(preflight.Path, 'cwd', return_value=self.code), \
                patch.object(preflight, 'validate_resume_checkpoint', return_value=target) as validate, \
                patch.object(preflight, 'select_checkpoint') as select, \
                patch.object(preflight, 'prepare_checkpoint') as prepare:
            result = preflight.main(['--output', str(output), *self.arguments(),
                                     'checkpointing.resume_from_ckpt=True'])
        self.assertEqual(result, 0)
        self.assertEqual(output.read_bytes(), (expected[0] + '\0').encode())
        validate.assert_called_once()
        select.assert_not_called()
        prepare.assert_not_called()

    def test_cli_failure_removes_stale_arguments_and_never_writes_manifest(self):
        output = self.base / 'arguments.bin'
        output.write_bytes(b'training.student_init_path=stale\0')
        with patch.object(preflight.Path, 'cwd', return_value=self.code), \
                redirect_stderr(io.StringIO()) as stderr:
            result = preflight.main(['--output', str(output), *self.arguments()])
        self.assertEqual(result, 1)
        self.assertFalse(output.exists())
        self.assertFalse(self.output.exists())
        self.assertIn('Failed before GPU launch', stderr.getvalue())

    def test_explicit_student_and_wrong_modes_fail_before_selection(self):
        with patch.object(preflight, 'select_checkpoint') as select:
            for extra, message in ((['training.student_init_path=/some/weights.ckpt'], 'latest CDLM'),
                                   (['mode=prepare_cdlm'], 'mode=train'),
                                   (['algo=cdlm'], 'algo=mmd_finetune')):
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                    preflight.prepare_overrides(self.config(*extra), code_dir=self.code)
        select.assert_not_called()

    def test_relative_snapshot_root_is_resolved_against_code_directory(self):
        self.checkpoint()
        relative = os.path.relpath(self.snapshot, self.code)
        result = preflight.prepare_overrides(
            self.config(override('mmd.cdlm_snapshot_root', relative)), code_dir=self.code)
        final = compose_cli(*self.arguments(), *result)
        self.assertEqual(Path(final.training.student_init_path).parents[2], self.snapshot)


class MMDLauncherTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='mmd launcher '))).resolve()
        self.calls = self.base / 'calls.jsonl'
        self.fake = self.base / 'fake-python'
        self.fake.write_text('#!' + sys.executable + '\n' + '''import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['CAPTURE_FILE'], 'a') as stream:
    stream.write(json.dumps({'args': args, 'proxy': os.environ.get('CDLM_DOWNLOAD_PROXY')}) + '\\n')
if args[:3] == ['-u', '-m', 'mmd_preflight']:
    if os.environ.get('FAIL_PREFLIGHT') == '1':
        raise SystemExit(23)
    output = pathlib.Path(args[args.index('--output') + 1])
    values = ['training.student_init_path=' + json.dumps('/snapshot/a b,c=d.ckpt'),
              'training.finetune_path=' + json.dumps('/assets/original teacher.ckpt')]
    output.write_bytes(('\\0'.join(values) + '\\0').encode())
elif args[:3] != ['-u', '-m', 'main']:
    raise SystemExit('Unexpected command: ' + repr(args))
''')
        self.fake.chmod(0o755)

    def run_launcher(self, fail=False):
        env = dict(os.environ, PYTHON=str(self.fake), CDLM_CODE_DIR=str(ROOT),
                   CAPTURE_FILE=str(self.calls), CDLM_SNAPSHOT_ROOT='/mounted/previous state, v=3',
                   CDLM_SOURCE_NAMESPACE='cdlm_tinygsm_ancestral_sweep')
        env.pop('CDLM_DOWNLOAD_PROXY', None)
        if fail:
            env['FAIL_PREFLIGHT'] = '1'
        else:
            env.pop('FAIL_PREFLIGHT', None)
        return subprocess.run(['bash', str(ROOT / 'scripts/train_mmd_from_cdlm_tinygsm_nirvana.sh'),
                               'logger.name=cdlm to mmd fixture'], cwd=self.base, env=env,
                              capture_output=True, text=True, timeout=30)

    def test_failed_preflight_stops_before_any_main_or_gpu_command(self):
        result = self.run_launcher(fail=True)
        self.assertEqual(result.returncode, 23, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['args'][:3], ['-u', '-m', 'mmd_preflight'])

    def test_success_starts_one_mmd_main_with_resolved_paths_and_no_cdlm_preparation(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertEqual([call['args'][2] for call in calls], ['mmd_preflight', 'main'])
        prepare_args, train_args = [call['args'] for call in calls]
        prepared = compose_cli(*prepare_args[5:])
        self.assertEqual(prepared.mmd.cdlm_snapshot_root, '/mounted/previous state, v=3')
        self.assertEqual(prepared.mmd.cdlm_snapshot_namespace, 'cdlm_tinygsm_ancestral_sweep')
        self.assertEqual(train_args[-1], 'mode=train')
        self.assertFalse(any('prepare_cdlm' in argument or 'torch.distributed.run' in argument
                             for call in calls for argument in call['args']))
        cfg = compose_cli(*train_args[3:])
        self.assertEqual(cfg.algo.name, 'mmd_finetune')
        self.assertEqual(cfg.training.student_init_path, '/snapshot/a b,c=d.ckpt')
        self.assertEqual(cfg.training.finetune_path, '/assets/original teacher.ckpt')
        self.assertEqual(cfg.logger.name, 'cdlm to mmd fixture')
        self.assertIsNone(calls[1]['proxy'])
        self.assertIn('survival-mode.', calls[0]['proxy'])


if __name__ == '__main__':
    unittest.main()
