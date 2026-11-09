"""Snapshot real cache files without models, GPUs, network, or Nirvana SDK."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from cdlm_cache import cache_key
from cdlm_trajectory_snapshot import TrajectorySnapshot
import nirvana_snapshot


def identity(teacher='original', weights='raw'):
    parameters = {'schema_version': 2, 'teacher_weights': weights,
                  'teacher_checkpoint_sha256': hashlib.sha256(teacher.encode()).hexdigest()}
    encoded = json.dumps(parameters, sort_keys=True, separators=(',', ':')).encode()
    return {'fingerprint': hashlib.sha256(encoded).hexdigest(), 'parameters': parameters}


class TrajectorySnapshotTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.local, self.store = self.root / 'local', self.root / 'store'
        self.store.mkdir()
        self.identity = identity()
        self.descriptors = [{'file': f'part-{i:08d}.pt', 'start': i, 'count': 1}
                            for i in range(3)]
        self.dumps = []
        self.sdk = ModuleType('nirvana_dl')
        self.sdk.snapshot = SimpleNamespace(get_snapshot_path=lambda: str(self.store),
                                             dump_snapshot=lambda path: self.dumps.append(path))
        self.enterContext(patch.dict(sys.modules, {'nirvana_dl': self.sdk}))
        self.enterContext(patch.dict('os.environ', {}, clear=True))

    def controller(self, directory=None, selected=None, **kwargs):
        return TrajectorySnapshot(directory or self.local, selected or self.identity,
                                  self.descriptors, **kwargs)

    def destination(self, selected=None):
        return self.store / 'cdlm_trajectories' / cache_key(selected or self.identity)

    def manifest(self, directory, selected=None, *, complete=False, label='hf://original/teacher/model.ckpt'):
        selected = selected or self.identity
        directory.mkdir(parents=True, exist_ok=True)
        descriptors = []
        for descriptor in self.descriptors:
            descriptor = dict(descriptor)
            if complete:
                receipt = json.loads((directory / descriptor['file']).with_suffix('.json').read_text())
                descriptor['sha256'] = receipt['sha256']
            descriptors.append(descriptor)
        value = {'schema_version': 2, 'identity': selected,
                 'status': 'complete' if complete else 'building',
                 'num_trajectories': 3, 'shards': descriptors, 'teacher_source': label}
        (directory / 'manifest.json').write_text(json.dumps(value))
        return value

    def pair(self, directory, index, payload=None, selected=None):
        directory.mkdir(parents=True, exist_ok=True)
        descriptor = self.descriptors[index]
        path = directory / descriptor['file']
        contents = payload if payload is not None else f'complete tensor {index}'.encode()
        path.write_bytes(contents)
        receipt = {**descriptor, 'fingerprint': (selected or self.identity)['fingerprint'],
                   'sha256': hashlib.sha256(contents).hexdigest()}
        path.with_suffix('.json').write_text(json.dumps(receipt))
        return path

    def test_partial_restore_then_final_complete_uses_committed_pairs_only(self):
        self.manifest(self.local)
        self.pair(self.local, 0)
        (self.local / 'part-00000001.pt').write_bytes(b'unconfirmed tensor')
        (self.local / 'manifest.lock').touch()
        (self.local / 'part-00000002.pt.random.tmp').write_bytes(b'incomplete write')
        publisher = self.controller()
        self.assertTrue(publisher._publish())
        remote = self.destination()
        self.assertEqual(set(path.name for path in remote.iterdir()),
                         {'manifest.json', 'part-00000000.pt', 'part-00000000.json'})
        manifest = json.loads((remote / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'building')
        self.assertEqual(manifest['teacher_source'], 'hf://original/teacher/model.ckpt')
        restored = self.root / 'restored'
        self.assertTrue(self.controller(restored).restore())
        self.assertEqual((restored / 'part-00000000.pt').read_bytes(), b'complete tensor 0')
        self.assertFalse((restored / 'part-00000001.pt').exists())
        self.assertEqual(json.loads((restored / 'manifest.json').read_text())['status'], 'building')
        for index in (1, 2):
            self.pair(self.local, index)
        self.manifest(self.local, complete=True)
        self.assertTrue(publisher.close())
        final = json.loads((remote / 'manifest.json').read_text())
        self.assertEqual(final['status'], 'complete')
        self.assertTrue(all('sha256' in descriptor for descriptor in final['shards']))

    def test_repeat_publish_copies_only_new_shards_and_retries_failed_sdk_commit(self):
        self.manifest(self.local)
        self.pair(self.local, 0)
        publisher = self.controller()
        with patch.object(publisher, '_copy_verified', wraps=publisher._copy_verified) as copy:
            self.assertTrue(publisher._publish())
            self.assertEqual(copy.call_count, 1)
            self.assertTrue(publisher._publish())
            self.assertEqual(copy.call_count, 1)
            self.assertEqual(len(self.dumps), 1)
            self.pair(self.local, 1)
            with patch.object(self.sdk.snapshot, 'dump_snapshot', side_effect=RuntimeError('upload failed')):
                self.assertFalse(publisher._publish())
            self.assertEqual(copy.call_count, 2)
            self.assertTrue(publisher._publish())
            self.assertEqual(copy.call_count, 2)
            self.assertEqual(len(self.dumps), 2)

    def test_model_and_weight_choices_use_isolated_namespaces(self):
        self.manifest(self.local)
        self.pair(self.local, 0)
        self.assertTrue(self.controller().close())
        original = (self.destination() / 'part-00000000.pt').read_bytes()
        for index, selected in enumerate((identity('mmd', 'ema'), identity('original', 'ema'))):
            directory = self.root / f'teacher-{index}'
            controller = self.controller(directory, selected)
            self.assertFalse(controller.restore())
            self.manifest(directory, selected)
            self.pair(directory, 0, b'another teacher', selected)
            self.assertTrue(controller.close())
            self.assertNotEqual(controller.namespace, self.controller().namespace)
            self.assertEqual((self.destination(selected) / 'part-00000000.pt').read_bytes(),
                             b'another teacher')
        self.assertEqual((self.destination() / 'part-00000000.pt').read_bytes(), original)

    def test_restore_preserves_valid_local_pairs_and_unconfirmed_local_tensor(self):
        remote = self.destination()
        for index in range(3):
            self.pair(remote, index)
        self.manifest(remote, complete=True)
        self.manifest(self.local, label='local teacher path')
        local = self.pair(self.local, 0)
        local_inode = local.stat().st_ino
        (self.local / 'part-00000001.pt').write_bytes(b'local recovery candidate')
        self.assertTrue(self.controller().restore())
        self.assertEqual(local.read_bytes(), b'complete tensor 0')
        self.assertEqual(local.stat().st_ino, local_inode)
        self.assertEqual((self.local / 'part-00000001.pt').read_bytes(), b'local recovery candidate')
        self.assertFalse((self.local / 'part-00000001.json').exists())
        self.assertEqual((self.local / 'part-00000002.pt').read_bytes(), b'complete tensor 2')
        manifest = json.loads((self.local / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'building')
        self.assertEqual(manifest['teacher_source'], 'local teacher path')

    def test_conflicting_valid_immutable_shards_raise_without_overwriting_either_copy(self):
        remote = self.destination()
        self.manifest(remote)
        self.pair(remote, 0, b'saved immutable tensor')
        self.manifest(self.local)
        self.pair(self.local, 0, b'conflicting local tensor')
        for operation in (self.controller().restore, self.controller()._publish):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, 'Conflicting immutable'):
                operation()
            self.assertEqual((remote / 'part-00000000.pt').read_bytes(), b'saved immutable tensor')
            self.assertEqual((self.local / 'part-00000000.pt').read_bytes(), b'conflicting local tensor')
        self.assertEqual(self.dumps, [])

    def test_orphan_snapshot_receipt_is_removed_before_repairing_missing_tensor(self):
        remote = self.destination()
        self.manifest(remote)
        self.pair(remote, 0, b'old orphan receipt')
        (remote / 'part-00000000.pt').unlink()
        self.manifest(self.local)
        self.pair(self.local, 0)
        publisher = self.controller()

        def interrupted(*args):
            self.assertFalse((remote / 'part-00000000.json').exists())
            raise OSError('copy stopped')

        with patch.object(publisher, '_copy_verified', side_effect=interrupted):
            self.assertFalse(publisher._publish())
        self.assertFalse((remote / 'part-00000000.pt').exists())
        self.assertFalse((remote / 'part-00000000.json').exists())
        self.assertEqual(self.dumps, [])
        self.assertTrue(publisher.close())

    def test_orphan_local_receipt_is_removed_before_restoring_missing_tensor(self):
        remote = self.destination()
        self.manifest(remote)
        self.pair(remote, 0)
        self.manifest(self.local)
        self.pair(self.local, 0, b'orphan local receipt')
        (self.local / 'part-00000000.pt').unlink()
        controller = self.controller()

        def interrupted(*args):
            self.assertFalse((self.local / 'part-00000000.json').exists())
            raise OSError('restore stopped')

        with patch.object(controller, '_copy_verified', side_effect=interrupted):
            self.assertFalse(controller.restore())
        self.assertFalse((self.local / 'part-00000000.json').exists())
        self.assertTrue(controller.restore())
        self.assertEqual((self.local / 'part-00000000.pt').read_bytes(), b'complete tensor 0')

    def test_invalid_identity_layout_receipt_or_tensor_never_silently_reuses(self):
        remote = self.destination()
        for case in ('identity', 'layout', 'receipt', 'tensor'):
            with self.subTest(case=case):
                self.pair(remote, 0)
                manifest = self.manifest(remote)
                if case == 'identity':
                    manifest['identity'] = identity('other')
                elif case == 'layout':
                    manifest['shards'][0]['count'] = 99
                elif case == 'receipt':
                    receipt = remote / 'part-00000000.json'
                    value = json.loads(receipt.read_text())
                    value['fingerprint'] = '0' * 64
                    receipt.write_text(json.dumps(value))
                else:
                    (remote / 'part-00000000.pt').write_bytes(b'corrupt tensor')
                (remote / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    self.controller(self.root / ('restore-' + case)).restore()
                self.assertFalse((self.root / ('restore-' + case) / 'part-00000000.pt').exists())

    def test_copy_is_verified_before_atomic_publish_and_sdk_is_not_called_on_failure(self):
        self.manifest(self.local)
        self.pair(self.local, 0)
        publisher = self.controller()

        def corrupted_copy(source, temporary):
            Path(temporary).write_bytes(b'corrupted in transit')

        with patch('cdlm_trajectory_snapshot.shutil.copy2', side_effect=corrupted_copy):
            with self.assertRaisesRegex(ValueError, 'copy checksum mismatch'):
                publisher._publish()
        self.assertFalse((self.destination() / 'part-00000000.pt').exists())
        self.assertEqual(list(self.destination().glob('*.tmp')), [])
        self.assertEqual(self.dumps, [])
        self.assertTrue(publisher.close())

    def test_sdk_absent_or_transfer_failure_is_best_effort(self):
        self.manifest(self.local)
        self.pair(self.local, 0)
        with patch.dict(sys.modules, {'nirvana_dl': None}):
            controller = self.controller()
            with patch.object(controller, '_hash', side_effect=AssertionError('SDK absence must not hash tensors')):
                self.assertFalse(controller.restore())
                self.assertFalse(controller.close())
        publisher = self.controller()
        with patch('cdlm_trajectory_snapshot.shutil.copy2', side_effect=OSError('disk unavailable')):
            self.assertFalse(publisher._publish())
        self.assertTrue(publisher.close())

    def test_rank_and_disabled_guards_never_call_sdk(self):
        for options in ({'rank': 1}, {'enabled': False}):
            with self.subTest(options=options), patch('nirvana_snapshot._sdk') as sdk:
                controller = self.controller(**options)
                self.assertFalse(controller.restore())
                self.assertIs(controller.start(), controller)
                self.assertFalse(controller.close())
                self.assertIsNone(controller._thread)
                sdk.assert_not_called()

    def test_monitor_shutdown_joins_before_single_forced_final_publish(self):
        controller = self.controller(interval_seconds=.01)
        entered = threading.Event()
        calls = []

        def publish(force=False):
            calls.append(force)
            entered.set()
            return True

        with patch.object(controller, '_publish', side_effect=publish):
            self.assertIs(controller.start(), controller)
            self.assertTrue(entered.wait(2))
            self.assertTrue(controller.close())
            self.assertFalse(controller._thread.is_alive())
            self.assertTrue(controller.close())
        self.assertFalse(calls[0])
        self.assertTrue(calls[-1])
        self.assertEqual(calls.count(True), 1)

    def test_monitor_start_failure_still_allows_final_synchronous_export(self):
        self.manifest(self.local)
        self.pair(self.local, 0)
        controller = self.controller()
        with patch('cdlm_trajectory_snapshot.threading.Thread.start',
                   side_effect=RuntimeError('thread resources unavailable')):
            self.assertIs(controller.start(), controller)
        self.assertIsNone(controller._thread)
        self.assertTrue(controller.close())
        self.assertEqual(len(self.dumps), 1)
        self.assertEqual((self.destination() / 'part-00000000.pt').read_bytes(), b'complete tensor 0')

    def test_training_snapshot_waits_until_cache_pair_and_manifest_publication_finishes(self):
        self.manifest(self.local)
        self.pair(self.local, 0)
        entered, release, training_done = threading.Event(), threading.Event(), threading.Event()
        training = self.root / 'training'
        training.mkdir()
        (training / 'last.ckpt').write_bytes(b'complete checkpoint')
        results = []

        def dump(path):
            if threading.current_thread().name == 'cache-publisher':
                entered.set()
                self.assertTrue(release.wait(2))
            self.dumps.append(path)

        def train():
            results.append(nirvana_snapshot.save(training, namespace='cdlm_tinygsm'))
            training_done.set()

        self.sdk.snapshot.dump_snapshot = dump
        cache_thread = threading.Thread(target=lambda: results.append(self.controller()._publish()),
                                        name='cache-publisher')
        cache_thread.start()
        self.assertTrue(entered.wait(2))
        training_thread = threading.Thread(target=train)
        training_thread.start()
        try:
            self.assertFalse(training_done.wait(.05))
            self.assertFalse((self.store / 'cdlm_tinygsm/last.ckpt').exists())
        finally:
            release.set()
            cache_thread.join(2)
            training_thread.join(2)
        self.assertFalse(cache_thread.is_alive())
        self.assertFalse(training_thread.is_alive())
        self.assertEqual(results, [True, True])
        self.assertEqual((self.store / 'cdlm_tinygsm/last.ckpt').read_bytes(), b'complete checkpoint')


if __name__ == '__main__':
    unittest.main()
