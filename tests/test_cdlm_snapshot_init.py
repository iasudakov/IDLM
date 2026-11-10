"""Latest CDLM selection uses checkpoint content, isolated from model training."""

import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import cdlm_snapshot_init as selector


class SnapshotCheckpointSelectionTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()

    def checkpoint(self, name='last.ckpt', *, namespace='cdlm_tinygsm', step=10,
                   algorithm='cdlm', marker=True, **changes):
        path = self.root / namespace / 'checkpoints' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {'global_step': step, 'epoch': 3,
                      'state_dict': {'backbone.weight': torch.ones(2, 3)},
                      'optimizer_states': [{'old': True}]}
        if marker:
            checkpoint['cdlm_cache_fingerprint'] = 'cache123'
        if algorithm is not None:
            checkpoint['hyper_parameters'] = {'config': {'algo': {'name': algorithm}}}
        checkpoint.update(changes)
        torch.save(checkpoint, path)
        return path

    def test_serialized_global_step_beats_filename_and_mtime(self):
        older = self.checkpoint('step=99999999.ckpt', step=3)
        newer = self.checkpoint('step=00000001.ckpt', step=19)
        os.utime(newer, (1, 1))
        os.utime(older, (9999, 9999))
        result = selector.select_checkpoint(self.root)
        self.assertEqual(Path(result['path']), newer)
        self.assertEqual(result['global_step'], 19)
        self.assertEqual(result['namespace'], 'cdlm_tinygsm')
        self.assertEqual(result['epoch'], 3)
        self.assertEqual(result['size_bytes'], newer.stat().st_size)
        self.assertEqual(result['sha256'], hashlib.sha256(newer.read_bytes()).hexdigest())
        self.assertEqual(result['cdlm_cache_fingerprint'], 'cache123')
        self.assertNotIn('optimizer_states', result)
        self.assertNotIn('_mtime_ns', result)

    def test_stale_last_is_not_preferred_to_later_numbered_checkpoint(self):
        self.checkpoint('last.ckpt', step=10)
        newer = self.checkpoint('step=20.ckpt', step=20)
        self.assertEqual(selector.select_checkpoint(self.root)['path'], str(newer))

    def test_last_wins_a_same_step_tie(self):
        last = self.checkpoint('last.ckpt', step=12)
        self.checkpoint('z-step=12.ckpt', step=12)
        self.assertEqual(selector.select_checkpoint(self.root)['path'], str(last))

    def test_independent_namespaces_require_an_explicit_choice(self):
        self.checkpoint(namespace='cdlm_tinygsm', step=900)
        expected = self.checkpoint(namespace='cdlm_tinygsm_ancestral_sweep', step=300)
        with self.assertRaisesRegex(ValueError, 'CDLM_SOURCE_NAMESPACE'):
            selector.select_checkpoint(self.root)
        result = selector.select_checkpoint(self.root, 'cdlm_tinygsm_ancestral_sweep')
        self.assertEqual(result['path'], str(expected))
        self.assertEqual(result['global_step'], 300)

    def test_corrupt_latest_is_reported_then_previous_valid_is_used(self):
        previous = self.checkpoint('step=10.ckpt')
        corrupt = previous.parent / 'last.ckpt'
        corrupt.write_bytes(b'incomplete torch checkpoint')
        with self.assertLogs(selector.LOG, 'WARNING') as logs:
            result = selector.select_checkpoint(self.root)
        self.assertEqual(result['path'], str(previous))
        self.assertTrue(any('last.ckpt' in message and 'Skipping' in message for message in logs.output))

    def test_step_zero_and_marker_only_checkpoint_are_supported(self):
        path = self.checkpoint(step=0, algorithm=None)
        result = selector.select_checkpoint(self.root)
        self.assertEqual(result['global_step'], 0)
        self.assertEqual(result['path'], str(path))

    def test_config_only_cdlm_marker_is_supported(self):
        path = self.checkpoint(marker=False)
        result = selector.select_checkpoint(self.root)
        self.assertEqual(result['path'], str(path))
        self.assertIsNone(result['cdlm_cache_fingerprint'])

    def test_downstream_mmd_is_ignored_even_with_cdlm_provenance(self):
        expected = self.checkpoint(step=12)
        self.checkpoint(namespace='mmd_from_cdlm', algorithm='mmd', step=8000)
        with self.assertLogs(selector.LOG, 'WARNING'):
            result = selector.select_checkpoint(self.root)
        self.assertEqual(result['path'], str(expected))

    def test_missing_and_malformed_steps_do_not_guess_from_filename(self):
        for step in (None, -1, True, '42', 4.5):
            with self.subTest(step=step):
                self.checkpoint('step=0042.ckpt', step=step)
                with self.assertLogs(selector.LOG, 'WARNING'), self.assertRaises(FileNotFoundError):
                    selector.select_checkpoint(self.root)

    def test_no_cdlm_marker_or_weights_is_rejected(self):
        for changes in ({'algorithm': None, 'marker': False},
                        {'state_dict': {'teacher.weight': torch.ones(1)}},
                        {'state_dict': {'backbone.weight': 'not a tensor'}}):
            with self.subTest(changes=changes):
                self.checkpoint(**changes)
                with self.assertLogs(selector.LOG, 'WARNING'), self.assertRaises(FileNotFoundError):
                    selector.select_checkpoint(self.root)

    def test_trajectory_and_asset_namespaces_are_not_scanned(self):
        expected = self.checkpoint(step=7)
        self.checkpoint(namespace='cdlm_trajectories', step=700)
        self.checkpoint(namespace='cdlm_assets', step=800)
        self.assertEqual(selector.select_checkpoint(self.root)['path'], str(expected))

    def test_explicit_nested_namespace_and_root_level_checkpoints(self):
        nested = self.checkpoint(namespace='runs/cdlm_final')
        self.assertEqual(selector.select_checkpoint(self.root, 'runs/cdlm_final')['path'], str(nested))
        legacy = self.checkpoint(namespace='.')
        result = selector.select_checkpoint(self.root)
        self.assertEqual(result['path'], str(legacy))
        self.assertEqual(result['namespace'], '.')

    def test_sdk_default_root_and_useful_missing_sdk_error(self):
        path = self.checkpoint()
        sdk = SimpleNamespace(snapshot=SimpleNamespace(get_snapshot_path=lambda: str(self.root)))
        with patch('nirvana_snapshot._sdk', return_value=sdk):
            self.assertEqual(selector.select_checkpoint()['path'], str(path))
        with patch('nirvana_snapshot._sdk', side_effect=ImportError('missing')):
            with self.assertRaisesRegex(RuntimeError, 'CDLM_SNAPSHOT_ROOT'):
                selector.select_checkpoint()

    def test_invalid_namespace_and_escaping_symlinks_are_rejected(self):
        self.checkpoint()
        for namespace in ('', '../outside', '/tmp', 'valid/../cdlm_tinygsm'):
            with self.subTest(namespace=namespace), self.assertRaises(ValueError):
                selector.select_checkpoint(self.root, namespace)
        outside = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        (self.root / 'escape').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            selector.select_checkpoint(self.root, 'escape')
        # Autodiscovery does not follow the escaping namespace at all.
        self.assertEqual(selector.select_checkpoint(self.root)['namespace'], 'cdlm_tinygsm')

    def test_only_trajectory_state_and_absent_namespace_fail_actionably(self):
        (self.root / 'cdlm_trajectories').mkdir()
        for namespace in (None, 'cdlm_missing'):
            with self.subTest(namespace=namespace), self.assertRaisesRegex(
                    FileNotFoundError, 'trajectory-only state'):
                selector.select_checkpoint(self.root, namespace)
        with self.assertRaisesRegex(FileNotFoundError, 'does not exist'):
            selector.select_checkpoint(self.root / 'missing')


if __name__ == '__main__':
    unittest.main()
