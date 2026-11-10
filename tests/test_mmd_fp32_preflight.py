"""Plain MDLM-to-MMD FP32 launch contract, without GPUs or model training."""

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
EXPERIMENT = '+experiment=mmd_tinygsm_fp32'
LAUNCHER = ROOT / 'scripts/train_mmd_tinygsm_fp32_nirvana.sh'
RESOLVED = ('checkpointing.save_dir', 'checkpointing.snapshot_namespace',
            'checkpointing.resume_ckpt_path', 'mmd.cdlm_snapshot_root',
            'mmd.cdlm_snapshot_namespace')


def string_override(key, value):
    return key + '=' + json.dumps(str(value))


class FP32PreflightTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='mmd fp32 '))).resolve()
        self.code = self.root / 'code folder, revision=1'
        self.code.mkdir()
        self.output = self.root / 'run output, precision=32'
        self.teacher = self.code / 'original MDLM, version=2.ckpt'
        torch.save({
            'state_dict': {'backbone.weight': torch.full((2, 3), .25)},
            'ema': {'shadow_params': [torch.full((2, 3), .5)],
                    'decay': .9999, 'num_updates': 10000},
            'global_step': 10000,
        }, self.teacher)
        self.enterContext(patch.dict(os.environ, {'INPUT_PATH': str(self.root / 'inputs')}))
        self.enterContext(redirect_stdout(io.StringIO()))

    def arguments(self, *extra):
        return [EXPERIMENT,
                string_override('training.finetune_path', self.teacher),
                string_override('checkpointing.save_dir', self.output),
                string_override('hydra.run.dir', self.output), *extra]

    def config(self, *extra):
        return preflight.compose_config(self.arguments(*extra), code_dir=self.code,
                                         resolved_fields=RESOLVED)

    def test_full_fp32_recipe_retains_paper_objective_and_four_gpu_update_size(self):
        cfg = self.config()
        self.assertEqual(cfg.algo.name, 'mmd_finetune')
        self.assertEqual(cfg.mmd.initialization_source, 'mdlm')
        self.assertTrue(cfg.model.force_fp32)
        self.assertEqual(cfg.trainer.precision, '32-true')
        self.assertEqual(cfg.training.loss_precision, 'float32')
        self.assertEqual(cfg.mmd.group_size, 8)
        self.assertEqual(cfg.mmd.draws_per_candidate, 1)
        self.assertFalse(cfg.mmd.average_candidates)
        self.assertEqual(cfg.mmd.kernel, 'rbf')
        self.assertEqual(cfg.mmd.feature_layer, 6)
        self.assertEqual(cfg.mmd.rbf_alpha, 6e-5)
        self.assertEqual(cfg.mmd.feature_batch_size, 8)
        self.assertEqual(cfg.mmd.student_forward_batch_size, 8)
        self.assertTrue(cfg.mmd.student_activation_checkpointing)
        self.assertEqual(cfg.trainer.devices, 4)
        self.assertEqual(cfg.loader.batch_size, 32)
        self.assertEqual(cfg.trainer.accumulate_grad_batches, 4)
        self.assertEqual(cfg.loader.global_batch_size, 512)
        self.assertEqual(cfg.trainer.devices * cfg.loader.batch_size
                         * cfg.trainer.accumulate_grad_batches, 512)
        self.assertEqual(cfg.optim.lr, 1e-5)
        self.assertEqual(cfg.lr_scheduler.num_warmup_steps, 2500)
        self.assertEqual(cfg.trainer.max_steps, 7000)
        self.assertEqual(cfg.trainer.max_epochs, -1)
        self.assertFalse(cfg.training.student_init_use_ema)
        self.assertFalse(cfg.training.teacher_use_ema)
        self.assertTrue(cfg.mmd.retain_student_ema)
        self.assertFalse(cfg.adversarial_distill.is_distill)

    def test_eval_and_weights_snapshot_cadence_match_optimizer_updates(self):
        cfg = self.config()
        self.assertTrue(cfg.eval.validate_before_train)
        self.assertTrue(cfg.eval.gsm8k_at_validation)
        self.assertEqual(cfg.eval.gsm8k_every_n_validations, 1)
        self.assertEqual(cfg.trainer.val_check_interval // cfg.trainer.accumulate_grad_batches, 500)
        self.assertEqual(list(cfg.eval.gsm8k_pareto_thresholds), VALIDATION_THRESHOLDS)
        self.assertEqual(cfg.eval.gsm8k_pareto_temperature, 0.)
        self.assertFalse(cfg.eval.gsm8k_greedy_token)
        self.assertIsNone(cfg.eval.gsm8k_ancestral_steps)
        self.assertIsNone(cfg.eval.gsm8k_ancestral_seeds)
        self.assertTrue(cfg.checkpointing.snapshot)
        self.assertFalse(cfg.checkpointing.snapshot_trajectories)
        self.assertFalse(cfg.checkpointing.resume_from_ckpt)
        self.assertEqual(cfg.checkpointing.snapshot_namespace, 'mmd_tinygsm_fp32')
        weights = cfg.callbacks.mmd_weights
        self.assertEqual(weights.every_n_train_steps, 500)
        self.assertTrue(weights.snapshot)
        self.assertEqual(weights.namespace, 'mmd_tinygsm_fp32')
        self.assertEqual(cfg.callbacks.checkpoint_every_n_steps.save_top_k, 0)
        self.assertFalse(cfg.callbacks.checkpoint_every_n_steps.save_last)
        self.assertEqual(cfg.logger.experiment, 'generative-models/tinygsm')
        self.assertEqual(cfg.model.length, 512)
        self.assertEqual(cfg.data.separator, r'\n')
        self.assertFalse(cfg.data.train_on_prompt)
        self.assertTrue(cfg.data.train_on_pad)

    def test_original_mdlm_supplies_both_models_without_a_cdlm_snapshot(self):
        with patch.object(preflight, 'select_checkpoint') as select, \
                patch('nirvana_snapshot._sdk', side_effect=ImportError('No Nirvana SDK')):
            values = preflight.prepare_overrides(self.config(), code_dir=self.code)
        select.assert_not_called()
        final = compose_cli(*self.arguments(), *values)
        self.assertEqual(final.training.student_init_path, str(self.teacher))
        self.assertEqual(final.training.finetune_path, str(self.teacher))
        manifest = self.output / 'mmd_initialization.json'
        self.assertEqual(final.mmd.initialization_manifest, str(manifest))
        provenance = json.loads(manifest.read_text())
        expected_hash = hashlib.sha256(self.teacher.read_bytes()).hexdigest()
        for role in ('student', 'feature_extractor'):
            self.assertEqual(provenance[role]['path'], str(self.teacher))
            self.assertEqual(provenance[role]['sha256'], expected_hash)
            self.assertEqual(provenance[role]['weights'], 'raw')
        self.assertTrue(provenance['student']['retain_ema_history'])
        self.assertTrue(provenance['fresh_optimizer_and_scheduler'])
        self.assertEqual(provenance['reference_run'], '3e0ac6b1-7a8f-43c3-af58-d9e21b886da7')

    def test_cli_preserves_paths_and_removes_stale_output_on_missing_weights(self):
        output = self.root / 'arguments.bin'
        with patch.object(preflight.Path, 'cwd', return_value=self.code):
            status = preflight.main(['--output', str(output), *self.arguments()])
        self.assertEqual(status, 0)
        raw = output.read_bytes()
        self.assertTrue(raw.endswith(b'\0'))
        final = compose_cli(*self.arguments(), *raw.decode().split('\0')[:-1])
        self.assertEqual(final.training.student_init_path, str(self.teacher))
        self.assertEqual(final.training.finetune_path, str(self.teacher))
        self.teacher.unlink()
        with patch.object(preflight.Path, 'cwd', return_value=self.code), \
                patch.object(preflight, 'select_checkpoint') as select, \
                redirect_stderr(io.StringIO()) as stderr:
            status = preflight.main(['--output', str(output), *self.arguments()])
        self.assertEqual(status, 1)
        self.assertFalse(output.exists())
        select.assert_not_called()
        self.assertIn('Failed before GPU launch', stderr.getvalue())

    def test_full_resume_uses_own_state_without_requiring_original_mdlm(self):
        self.teacher.unlink()
        target = str(self.output / 'checkpoints' / 'last.ckpt')
        cfg = self.config('checkpointing.resume_from_ckpt=True')
        with patch.object(preflight, 'validate_resume_checkpoint', return_value=target) as validate, \
                patch.object(preflight, 'select_checkpoint') as select, \
                patch.object(preflight, 'prepare_checkpoint') as assets:
            overrides = preflight.prepare_overrides(cfg, code_dir=self.code)
        self.assertEqual(overrides, [string_override('checkpointing.resume_ckpt_path', target)])
        validate.assert_called_once_with(cfg, code_dir=self.code)
        select.assert_not_called()
        assets.assert_not_called()


class FP32LauncherTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='fp32 launch '))).resolve()
        self.calls = self.root / 'calls.jsonl'
        self.code = self.root / 'code folder, revision=7'
        self.code.mkdir()
        self.fake = self.root / 'fake-python'
        self.fake.write_text('#!' + sys.executable + '\n' + '''import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['CAPTURE_FILE'], 'a') as stream:
    stream.write(json.dumps({'args': args, 'proxy': os.environ.get('CDLM_DOWNLOAD_PROXY')}) + '\\n')
if args[:3] == ['-u', '-m', 'mmd_preflight']:
    if os.environ.get('FAIL_PREFLIGHT') == '1':
        raise SystemExit(23)
    output = pathlib.Path(args[args.index('--output') + 1])
    weights = '/assets/MDLM weights, revision=2.ckpt'
    values = ['training.student_init_path=' + json.dumps(weights),
              'training.finetune_path=' + json.dumps(weights)]
    output.write_bytes(('\\0'.join(values) + '\\0').encode())
elif args[:3] != ['-u', '-m', 'main']:
    raise SystemExit('Unexpected command: ' + repr(args))
''')
        self.fake.chmod(0o755)

    def run_launcher(self, fail=False):
        env = dict(os.environ, PYTHON=str(self.fake), CDLM_CODE_DIR=str(self.code),
                   MMD_CODE_DIR=str(self.code), CAPTURE_FILE=str(self.calls))
        env.pop('CDLM_DOWNLOAD_PROXY', None)
        env.pop('CDLM_SOURCE_NAMESPACE', None)
        env.pop('CDLM_SNAPSHOT_ROOT', None)
        if fail:
            env['FAIL_PREFLIGHT'] = '1'
        else:
            env.pop('FAIL_PREFLIGHT', None)
        return subprocess.run(['bash', str(LAUNCHER),
                               'logger.name=mmd full fp32 fixture', 'optim.lr=2e-5'],
                              cwd=self.root, env=env, capture_output=True, text=True, timeout=30)

    def test_success_starts_plain_mmd_and_preserves_quoted_paths_and_overrides(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual([call['args'][2] for call in calls], ['mmd_preflight', 'main'])
        self.assertFalse(any('prepare_cdlm' in arg or 'torch.distributed.run' in arg
                             for call in calls for arg in call['args']))
        cfg = compose_cli(*calls[1]['args'][3:])
        self.assertEqual(cfg.mmd.initialization_source, 'mdlm')
        self.assertTrue(cfg.model.force_fp32)
        self.assertEqual(cfg.trainer.precision, '32-true')
        self.assertEqual(cfg.training.student_init_path, '/assets/MDLM weights, revision=2.ckpt')
        self.assertEqual(cfg.training.finetune_path, cfg.training.student_init_path)
        self.assertEqual(cfg.data.cache_dir, str(self.code / 'cache_dir'))
        self.assertEqual(cfg.model_weights_dir, str(self.code / 'cache_dir/tinygsm_mmd_offline_bundle'))
        self.assertEqual(cfg.logger.name, 'mmd full fp32 fixture')
        self.assertEqual(cfg.optim.lr, 2e-5)
        self.assertEqual(calls[1]['args'][-1], 'mode=train')
        self.assertIsNone(calls[1]['proxy'])

    def test_failed_preflight_never_starts_main(self):
        result = self.run_launcher(fail=True)
        self.assertEqual(result.returncode, 23, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual([call['args'][2] for call in calls], ['mmd_preflight'])

    def test_actual_launcher_missing_original_mdlm_never_starts_main(self):
        wrapper = self.root / 'real-preflight-wrapper'
        wrapper.write_text('#!' + sys.executable + '\n' + '''import json, os, pathlib, sys
args = sys.argv[1:]
with pathlib.Path(os.environ['CAPTURE_FILE']).open('a') as stream:
    stream.write(json.dumps({'args': args}) + '\\n')
if args[:3] == ['-u', '-m', 'mmd_preflight']:
    os.execv(sys.executable, [sys.executable] + args)
raise SystemExit('MAIN MUST NOT START')
''')
        wrapper.chmod(0o755)
        env = dict(os.environ, PYTHON=str(wrapper), CDLM_CODE_DIR=str(ROOT),
                   MMD_CODE_DIR=str(ROOT), CAPTURE_FILE=str(self.calls),
                   INPUT_PATH=str(self.root / 'empty-inputs'), CDLM_DOWNLOAD_PROXY='',
                   CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        missing = self.root / 'missing-original-mdlm.ckpt'
        result = subprocess.run(
            ['bash', str(LAUNCHER), string_override('training.finetune_path', missing),
             string_override('checkpointing.save_dir', self.root / 'new-run'),
             string_override('hydra.run.dir', self.root / 'new-run')],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('Failed before GPU launch', result.stderr)
        self.assertNotIn('MAIN MUST NOT START', result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual([call['args'][2] for call in calls], ['mmd_preflight'])


if __name__ == '__main__':
    unittest.main()
