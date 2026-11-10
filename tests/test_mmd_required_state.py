"""Missing or non-resumable state must stop MMD before workers/assets start."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import cdlm_snapshot_init as selector
import mmd_preflight as preflight


ROOT = Path(__file__).resolve().parents[1]
RESOLVED = ('checkpointing.save_dir', 'checkpointing.snapshot_namespace',
            'checkpointing.resume_ckpt_path', 'mmd.cdlm_snapshot_root',
            'mmd.cdlm_snapshot_namespace')


def string_override(key, value):
    return key + '=' + json.dumps(str(value))


class RequiredStateTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='required state '))).resolve()
        self.code = self.root / 'code'
        self.code.mkdir()
        self.snapshot = self.root / 'state'
        self.snapshot.mkdir()
        self.output = self.code / 'new MMD, output=1'
        self.namespace = 'mmd_from_cdlm_tinygsm'
        self.target = self.output / 'checkpoints' / 'last.ckpt'
        self.source = self.snapshot / self.namespace / 'checkpoints' / 'last.ckpt'
        self.sdk = SimpleNamespace(snapshot=SimpleNamespace(
            get_snapshot_path=lambda: str(self.snapshot), dump_snapshot=lambda _: None))
        self.enterContext(redirect_stdout(io.StringIO()))

    def arguments(self, *extra):
        return ['+experiment=mmd_from_cdlm_tinygsm',
                string_override('checkpointing.save_dir', self.output),
                string_override('hydra.run.dir', self.output),
                string_override('mmd.cdlm_snapshot_root', self.snapshot),
                'training.finetune_path=/unused/original-mdlm.ckpt', *extra]

    def config(self, *extra):
        return preflight.compose_config(self.arguments(*extra), code_dir=self.code,
                                         resolved_fields=RESOLVED)

    def full_state(self, path=None, **changes):
        payload = {
            'mmd_finetune_schema': 1,
            'global_step': 7000, 'epoch': 1,
            'state_dict': {'backbone.weight': torch.ones(2, 3),
                           'teacher.weight': torch.zeros(2, 3)},
            'optimizer_states': [{'state': {}, 'param_groups': [{'params': [0]}]}],
            'lr_schedulers': [{'last_epoch': 7000}],
            'loops': {'fit_loop': {'epoch_progress': {'current': {'completed': 1}}}},
            'ema': {'shadow_params': [torch.full((2, 3), .5)],
                    'decay': .9999, 'num_updates': 8248},
        }
        payload.update(changes)
        path = Path(path or self.source)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    def cdlm_state(self, root=None, namespace='cdlm_tinygsm'):
        path = Path(root or self.snapshot) / namespace / 'checkpoints' / 'last.ckpt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'global_step': 1248, 'cdlm_cache_fingerprint': 'valid-cdlm',
                    'state_dict': {'backbone.weight': torch.ones(2, 3)}}, path)
        return path

    def test_blank_sdk_root_never_falls_back_to_current_directory(self):
        self.cdlm_state(self.code)
        original_cwd = Path.cwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(self.code)
        for value in (None, '', ' ', '\t'):
            self.sdk.snapshot.get_snapshot_path = lambda value=value: value
            with self.subTest(value=value), patch('nirvana_snapshot._sdk', return_value=self.sdk):
                with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                    selector.select_checkpoint()

    def test_explicit_blank_root_is_rejected_without_querying_sdk(self):
        for value in ('', ' ', '\t'):
            with self.subTest(value=value), patch('nirvana_snapshot._sdk') as sdk:
                with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                    selector.resolve_snapshot_root(value)
                sdk.assert_not_called()

    def test_missing_sdk_does_not_fall_back_to_local_cdlm(self):
        self.cdlm_state(self.code)
        with patch('nirvana_snapshot._sdk', side_effect=ImportError('SDK unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'snapshot|state'):
                selector.select_checkpoint()

    def test_fresh_empty_trajectory_only_or_wrong_namespace_never_prepares_assets(self):
        output_args = self.root / 'arguments.bin'
        for case in ('empty', 'trajectories', 'wrong_namespace'):
            extra = []
            if case == 'trajectories':
                (self.snapshot / 'cdlm_trajectories').mkdir()
            if case == 'wrong_namespace':
                self.cdlm_state(namespace='other_cdlm')
                extra = ['mmd.cdlm_snapshot_namespace=missing_cdlm']
            output_args.write_bytes(b'stale arguments\0')
            with self.subTest(case=case), patch.object(preflight, 'prepare_checkpoint') as assets, \
                    patch.object(preflight.Path, 'cwd', return_value=self.code), \
                    redirect_stderr(io.StringIO()) as errors:
                status = preflight.main(['--output', str(output_args), *self.arguments(*extra)])
            self.assertEqual(status, 1)
            self.assertIn('Failed before GPU launch', errors.getvalue())
            assets.assert_not_called()
            self.assertFalse(output_args.exists())
            self.assertFalse(self.output.exists())

    def test_existing_full_local_state_has_priority_without_sdk(self):
        self.full_state(self.target, global_step=9000)
        cfg = self.config('checkpointing.resume_from_ckpt=True',
                          'mmd.cdlm_snapshot_root=/irrelevant/missing/cdlm')
        with patch('nirvana_snapshot._sdk') as sdk, \
                patch('nirvana_snapshot.restore') as restore:
            actual = preflight.validate_resume_checkpoint(cfg, code_dir=self.code)
        self.assertEqual(actual, str(self.target))
        sdk.assert_not_called()
        restore.assert_not_called()

    def test_corrupt_local_state_does_not_fall_back_to_valid_snapshot(self):
        self.full_state()
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'truncated local checkpoint')
        cfg = self.config('checkpointing.resume_from_ckpt=True')
        with patch('nirvana_snapshot._sdk') as sdk, patch('nirvana_snapshot.restore') as restore:
            with self.assertRaises((ValueError, RuntimeError, EOFError)):
                preflight.validate_resume_checkpoint(cfg, code_dir=self.code)
        sdk.assert_not_called()
        restore.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b'truncated local checkpoint')

    def test_missing_full_mmd_state_fails_before_cdlm_or_assets(self):
        cfg = self.config('checkpointing.resume_from_ckpt=True')
        with patch('nirvana_snapshot._sdk', return_value=self.sdk), \
                patch.object(preflight, 'select_checkpoint') as cdlm, \
                patch.object(preflight, 'prepare_checkpoint') as assets:
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                preflight.prepare_overrides(cfg, code_dir=self.code)
        cdlm.assert_not_called()
        assets.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_snapshot_disabled_does_not_query_sdk_when_local_state_missing(self):
        cfg = self.config('checkpointing.resume_from_ckpt=True', 'checkpointing.snapshot=False')
        with patch('nirvana_snapshot._sdk') as sdk:
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                preflight.validate_resume_checkpoint(cfg, code_dir=self.code)
        sdk.assert_not_called()

    def test_full_resume_rejects_ema_only_and_incomplete_payloads_before_restore(self):
        cfg = self.config('checkpointing.resume_from_ckpt=True')
        changes = [
            {'mmd_finetune_schema': None, 'checkpoint_weights': 'ema'},
            {'state_dict': {'backbone.weight': torch.ones(2, 3)}},
            {'optimizer_states': []}, {'lr_schedulers': []}, {'loops': {}},
            {'global_step': -1}, {'global_step': True}, {'ema': None},
        ]
        for change in changes:
            self.full_state(**change)
            with self.subTest(change=change), patch('nirvana_snapshot._sdk', return_value=self.sdk), \
                    patch('nirvana_snapshot.restore') as restore:
                with self.assertRaises((ValueError, RuntimeError)):
                    preflight.validate_resume_checkpoint(cfg, code_dir=self.code)
            restore.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_valid_snapshot_restores_full_run_and_returns_absolute_override(self):
        source = self.full_state()
        metadata = source.parent.parent / 'underdeep_metadata.pt'
        metadata.write_bytes(b'previous tracker metadata')
        cfg = self.config('checkpointing.resume_from_ckpt=True',
                          'mmd.cdlm_snapshot_root=/missing/cdlm/root',
                          'mmd.cdlm_snapshot_namespace=unused_cdlm')
        with patch('nirvana_snapshot._sdk', return_value=self.sdk), \
                patch.object(preflight, 'select_checkpoint') as cdlm, \
                patch.object(preflight, 'prepare_checkpoint') as assets:
            overrides = preflight.prepare_overrides(cfg, code_dir=self.code)
        self.assertEqual(overrides, [string_override('checkpointing.resume_ckpt_path', self.target)])
        self.assertEqual(self.target.read_bytes(), source.read_bytes())
        self.assertEqual((self.output / metadata.name).read_bytes(), metadata.read_bytes())
        cdlm.assert_not_called()
        assets.assert_not_called()

    def test_restore_failure_or_false_success_without_checkpoint_is_fatal(self):
        self.full_state()
        cfg = self.config('checkpointing.resume_from_ckpt=True')
        for result in (False, True):
            with self.subTest(restore_result=result), \
                    patch('nirvana_snapshot._sdk', return_value=self.sdk), \
                    patch('nirvana_snapshot.restore', return_value=result):
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    preflight.validate_resume_checkpoint(cfg, code_dir=self.code)
            self.assertFalse(self.target.exists())

    def test_missing_explicit_resume_path_outside_save_dir_does_not_guess_a_snapshot_file(self):
        self.full_state()
        cfg = self.config('checkpointing.resume_from_ckpt=True',
                          string_override('checkpointing.resume_ckpt_path', self.root / 'other.ckpt'))
        with patch('nirvana_snapshot._sdk', return_value=self.sdk), \
                patch('nirvana_snapshot.restore') as restore:
            with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
                preflight.validate_resume_checkpoint(cfg, code_dir=self.code)
        restore.assert_not_called()

    def test_actual_launcher_missing_state_never_invokes_main(self):
        calls = self.root / 'calls.txt'
        wrapper = self.root / 'python-wrapper'
        wrapper.write_text('#!' + sys.executable + '\n' + '''import os, pathlib, sys
args = sys.argv[1:]
with pathlib.Path(os.environ['STATE_GUARD_CALLS']).open('a') as stream:
    stream.write(' '.join(args[:3]) + '\\n')
if args[:3] == ['-u', '-m', 'mmd_preflight']:
    os.execv(sys.executable, [sys.executable] + args)
raise SystemExit('MAIN MUST NOT START')
''')
        wrapper.chmod(0o755)
        env = dict(os.environ, PYTHON=str(wrapper), CDLM_CODE_DIR=str(ROOT),
                   CDLM_SNAPSHOT_ROOT=str(self.root / 'missing-state'),
                   STATE_GUARD_CALLS=str(calls), CUDA_VISIBLE_DEVICES='',
                   HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        env.pop('CDLM_SOURCE_NAMESPACE', None)
        result = subprocess.run(
            ['bash', str(ROOT / 'scripts/train_mmd_from_cdlm_tinygsm_nirvana.sh'),
             string_override('checkpointing.save_dir', self.output),
             string_override('hydra.run.dir', self.output)],
            env=env, cwd=self.code, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('Failed before GPU launch', result.stderr)
        self.assertNotIn('MAIN MUST NOT START', result.stderr)
        self.assertEqual(calls.read_text().splitlines(), ['-u -m mmd_preflight'])
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
