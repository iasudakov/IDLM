"""Filesystem-only tests of optional Nirvana snapshot persistence."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import nirvana_snapshot as snapshot


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.run, self.store = self.base / 'run', self.base / 'store'
        self.run.mkdir()
        self.store.mkdir()
        self.get_calls, self.dumps = [], []
        sdk = ModuleType('nirvana_dl')

        def get_path():
            self.get_calls.append(True)
            return str(self.store)

        sdk.snapshot = SimpleNamespace(get_snapshot_path=get_path,
                                       dump_snapshot=lambda path: self.dumps.append(path))
        self.sdk = sdk
        self.enterContext(patch.dict(sys.modules, {'nirvana_dl': sdk}))
        self.enterContext(patch.dict('os.environ', {}, clear=True))
        self.output = self.enterContext(redirect_stdout(io.StringIO()))

    def write_run(self, name, contents):
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def test_namespaced_roundtrip_preserves_checkpoints_metadata_and_other_namespaces(self):
        self.write_run('checkpoints/step=00001000.ckpt', b'numbered checkpoint')
        checkpoint = self.write_run('checkpoints/last.ckpt', b'last checkpoint')
        self.write_run('underdeep_metadata.pt', b'tracker resume')
        self.write_run('pareto_history.json', b'{"curves": {}}')
        checkpoint.unlink()
        checkpoint.symlink_to('step=00001000.ckpt')
        unrelated = self.store / 'other_run' / 'keep.ckpt'
        unrelated.parent.mkdir()
        unrelated.write_bytes(b'keep')
        self.assertTrue(snapshot.available())

        def dump(path):
            # The SDK sees complete files and the root, not the namespace.
            self.assertEqual(path, str(self.store))
            self.assertEqual((self.store / 'cdlm_tinygsm/checkpoints/last.ckpt').read_bytes(),
                             b'numbered checkpoint')
            self.assertEqual((self.store / 'cdlm_tinygsm/underdeep_metadata.pt').read_bytes(),
                             b'tracker resume')
            self.dumps.append(path)

        self.sdk.snapshot.dump_snapshot = dump
        self.assertTrue(snapshot.save(self.run, reason='step=1000', namespace='cdlm_tinygsm'))
        self.assertEqual(self.dumps, [str(self.store)])
        self.assertFalse((self.store / 'cdlm_tinygsm/checkpoints/last.ckpt').is_symlink())
        shutil.rmtree(self.run)
        self.assertTrue(snapshot.restore(self.run, namespace='cdlm_tinygsm'))
        self.assertEqual(checkpoint.read_bytes(), b'numbered checkpoint')
        self.assertFalse(checkpoint.is_symlink())
        self.assertEqual((self.run / 'underdeep_metadata.pt').read_bytes(), b'tracker resume')
        self.assertEqual((self.run / 'pareto_history.json').read_bytes(), b'{"curves": {}}')
        self.assertEqual(unrelated.read_bytes(), b'keep')

    def test_default_namespace_uses_store_root(self):
        self.write_run('last.ckpt', b'checkpoint')
        self.assertTrue(snapshot.save(self.run))
        self.assertEqual((self.store / 'last.ckpt').read_bytes(), b'checkpoint')
        (self.run / 'last.ckpt').unlink()
        self.assertTrue(snapshot.restore(self.run))
        self.assertEqual((self.run / 'last.ckpt').read_bytes(), b'checkpoint')

    def test_disabled_and_nonzero_ranks_never_access_sdk(self):
        self.assertFalse(snapshot.save(self.run, enabled=False))
        self.assertFalse(snapshot.restore(self.run, enabled=False))
        for environment in ({'RANK': '3', 'LOCAL_RANK': '0'},
                            {'SLURM_PROCID': '7', 'LOCAL_RANK': '0'},
                            {'LOCAL_RANK': '1'}, {'RANK': 'invalid', 'SLURM_PROCID': '2'}):
            with self.subTest(environment=environment), patch.dict('os.environ', environment, clear=True):
                self.assertFalse(snapshot.save(self.run))
                self.assertFalse(snapshot.restore(self.run))
        self.assertEqual(self.get_calls, [])
        self.assertEqual(self.dumps, [])

    def test_save_and_restore_hold_the_shared_store_lock_before_copying(self):
        self.write_run('last.ckpt', b'checkpoint')
        (self.store / 'last.ckpt').write_bytes(b'snapshot checkpoint')
        original = snapshot._copy_directory
        for operation in (snapshot.save, snapshot.restore):
            with self.subTest(operation=operation.__name__):
                started, copying = threading.Event(), threading.Event()

                def copy(*args):
                    copying.set()
                    return original(*args)

                def worker():
                    started.set()
                    operation(self.run)

                with patch.object(snapshot, '_copy_directory', side_effect=copy):
                    with snapshot.STORE_LOCK:
                        thread = threading.Thread(target=worker)
                        thread.start()
                        self.assertTrue(started.wait(2))
                        self.assertFalse(copying.wait(.05))
                    thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertTrue(copying.is_set())

    def test_global_rank_zero_takes_precedence_over_local_rank(self):
        self.write_run('last.ckpt', b'checkpoint')
        with patch.dict('os.environ', {'RANK': '0', 'LOCAL_RANK': '8'}, clear=True):
            self.assertTrue(snapshot.save(self.run))
        self.assertEqual(len(self.dumps), 1)

    def test_missing_sdk_and_invalid_sdk_are_clear_noops(self):
        for sdk in (None, ModuleType('nirvana_dl')):
            with self.subTest(sdk=sdk), patch.dict(sys.modules, {'nirvana_dl': sdk}):
                self.assertFalse(snapshot.available())
                self.assertFalse(snapshot.save(self.run))
                self.assertFalse(snapshot.restore(self.run))
        self.assertIn('SDK unavailable', self.output.getvalue())
        self.assertEqual(self.get_calls, [])

    def test_missing_namespace_never_falls_back_to_root(self):
        (self.store / 'last.ckpt').write_bytes(b'another run')
        shutil.rmtree(self.run)
        self.assertFalse(snapshot.restore(self.run, namespace='cdlm_tinygsm'))
        self.assertFalse(self.run.exists())
        self.assertEqual(self.dumps, [])

    def test_interrupted_file_copy_keeps_previous_last_and_cleans_temporary_file(self):
        self.write_run('checkpoints/last.ckpt', b'new checkpoint')
        previous = self.store / 'cdlm_tinygsm/checkpoints/last.ckpt'
        previous.parent.mkdir(parents=True)
        previous.write_bytes(b'previous checkpoint')

        def interrupted(source, target, **kwargs):
            Path(target).write_bytes(b'incomplete')
            raise OSError('copy interrupted')

        with patch.object(snapshot.shutil, 'copy2', side_effect=interrupted):
            self.assertFalse(snapshot.save(self.run, namespace='cdlm_tinygsm'))
        self.assertEqual(previous.read_bytes(), b'previous checkpoint')
        self.assertEqual(list(previous.parent.glob('.*.tmp')), [])
        self.assertEqual(self.dumps, [])
        self.assertIn('copy interrupted', self.output.getvalue())

    def test_interrupted_restore_preserves_existing_local_file(self):
        previous = self.write_run('last.ckpt', b'local checkpoint')
        (self.store / 'last.ckpt').write_bytes(b'snapshot checkpoint')
        with patch.object(snapshot.shutil, 'copy2', side_effect=OSError('read interrupted')):
            self.assertFalse(snapshot.restore(self.run))
        self.assertEqual(previous.read_bytes(), b'local checkpoint')
        self.assertEqual(list(self.run.glob('.*.tmp')), [])

    def test_sdk_path_and_dump_failures_are_best_effort(self):
        self.write_run('last.ckpt', b'checkpoint')
        with patch.object(self.sdk.snapshot, 'get_snapshot_path', side_effect=RuntimeError('SDK path failed')):
            self.assertFalse(snapshot.save(self.run))
            self.assertFalse(snapshot.restore(self.run))
        with patch.object(self.sdk.snapshot, 'dump_snapshot', side_effect=RuntimeError('upload failed')):
            self.assertFalse(snapshot.save(self.run))
        self.assertEqual((self.store / 'last.ckpt').read_bytes(), b'checkpoint')
        self.assertIn('SDK path failed', self.output.getvalue())
        self.assertIn('upload failed', self.output.getvalue())

    def test_escaping_namespace_and_overlapping_directories_are_rejected(self):
        self.write_run('last.ckpt', b'checkpoint')
        for namespace in ('../other', 'a/../../other', str(self.base / 'absolute'), ''):
            with self.subTest(namespace=namespace):
                self.assertFalse(snapshot.save(self.run, namespace=namespace))
                self.assertFalse(snapshot.restore(self.run, namespace=namespace))
        (self.store / 'escape').symlink_to(self.base / 'outside', target_is_directory=True)
        self.assertFalse(snapshot.save(self.run, namespace='escape'))
        for run, root in ((self.run, self.run), (self.run, self.run / 'nested_store'),
                          (self.store / 'nested_run', self.store)):
            with self.subTest(run=run, root=root), \
                    patch.object(self.sdk.snapshot, 'get_snapshot_path', return_value=str(root)):
                self.assertFalse(snapshot.save(run))
                self.assertFalse(snapshot.restore(run))
        self.assertEqual(self.dumps, [])

    def test_source_directory_links_are_rejected_without_recursing(self):
        self.write_run('last.ckpt', b'checkpoint')
        (self.run / 'recursive').symlink_to(self.run, target_is_directory=True)
        self.assertFalse(snapshot.save(self.run))
        self.assertEqual(self.dumps, [])
        self.assertIn('directory cannot be a symlink', self.output.getvalue())

    def test_missing_run_does_not_upload(self):
        shutil.rmtree(self.run)
        self.assertFalse(snapshot.save(self.run, namespace='cdlm_tinygsm'))
        self.assertEqual(self.dumps, [])


if __name__ == '__main__':
    unittest.main()
