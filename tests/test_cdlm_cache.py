"""CPU cache layout, teacher selection, legacy reuse and snapshot lifecycle."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from cdlm_cache import cache_directory, cache_key, cache_root
import cdlm_trajectories as trajectories
from test_cdlm_trajectories import Tokenizer, Teacher, configuration, data


class CacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        (self.base / 'teacher.ckpt').write_bytes(b'original teacher checkpoint fixture')
        self.config = configuration(self.base)
        self.config.algo.trajectory_cache_per_model = True
        self.tokenizer = Tokenizer()

    def identity(self):
        return trajectories.build_cache_identity(self.config, self.tokenizer, data())

    def prepare(self, **kwargs):
        return Path(trajectories.prepare_trajectories(Teacher(), self.config, self.tokenizer, data(), **kwargs))

    def test_key_tracks_generating_teacher_weights_and_full_identity_not_student(self):
        original = self.identity()
        root = cache_root(self.config)
        key = cache_key(original)
        self.assertEqual(key.split('/'), [
            'teacher-' + original['parameters']['teacher_checkpoint_sha256'][:16],
            'raw', original['fingerprint']])
        self.config.training.student_init_path = 'unrelated-student.ckpt'
        self.config.training.student_init_use_ema = True
        self.assertEqual(self.identity(), original)
        self.assertEqual(cache_directory(self.config, original), root / key)
        self.assertEqual(self.config.algo.trajectory_cache_dir, str(root))
        self.config.training.teacher_use_ema = True
        ema = self.identity()
        self.assertEqual(ema['parameters']['teacher_weights'], 'ema')
        self.assertNotEqual(ema['fingerprint'], original['fingerprint'])
        self.assertEqual(cache_key(ema).split('/')[1], 'ema')
        self.assertNotEqual(cache_directory(self.config, ema), cache_directory(self.config, original))
        (self.base / 'teacher.ckpt').write_bytes(b'MMD teacher checkpoint')
        mmd = self.identity()
        self.assertNotEqual(cache_key(mmd).split('/')[0], cache_key(ema).split('/')[0])

    def test_relocated_identical_teacher_keeps_key_and_settings_select_new_child(self):
        identity = self.identity()
        duplicate = self.base / 'moved-teacher.ckpt'
        duplicate.write_bytes((self.base / 'teacher.ckpt').read_bytes())
        self.config.training.finetune_path = str(duplicate)
        self.assertEqual(self.identity(), identity)
        self.config.algo.trajectory_temperatures = [0.]
        changed = self.identity()
        self.assertEqual(cache_key(changed).split('/')[:2], cache_key(identity).split('/')[:2])
        self.assertNotEqual(cache_directory(self.config, changed), cache_directory(self.config, identity))

    def test_root_uses_hydra_launch_directory_and_default_cache_fallback(self):
        self.config.algo.trajectory_cache_dir = 'relative/cache'
        launch = self.base / 'original-launch'
        with patch('hydra.utils.to_absolute_path', side_effect=lambda path: str(launch / path)) as resolve:
            self.assertEqual(cache_root(self.config), launch / 'relative/cache')
        resolve.assert_called_once_with('relative/cache')
        self.config.algo.trajectory_cache_dir = None
        self.assertEqual(cache_root(self.config), self.base / 'cdlm_trajectories')

    def test_matching_flat_legacy_complete_and_partial_cache_are_reused(self):
        self.config.algo.trajectory_cache_per_model = False
        legacy = self.prepare()
        before = (legacy / 'part-00000000.pt').stat().st_mtime_ns
        self.config.algo.trajectory_cache_per_model = True
        identity = self.identity()
        self.assertEqual(cache_directory(self.config, identity), legacy)
        manifest = json.loads((legacy / 'manifest.json').read_text())
        manifest['status'] = 'building'
        (legacy / 'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(cache_directory(self.config, identity), legacy)
        with patch.object(trajectories, 'collect_trajectory_batch', side_effect=AssertionError('must reuse')):
            self.assertEqual(self.prepare(), legacy)
        self.assertEqual((legacy / 'part-00000000.pt').stat().st_mtime_ns, before)
        self.assertEqual(trajectories.read_cache_manifest(legacy)['status'], 'complete')

    def test_different_legacy_teacher_is_retained_while_new_teacher_gets_child(self):
        self.config.algo.trajectory_cache_per_model = False
        legacy = self.prepare()
        old_manifest = (legacy / 'manifest.json').read_bytes()
        old_shard = (legacy / 'part-00000000.pt').read_bytes()
        self.config.algo.trajectory_cache_per_model = True
        (self.base / 'teacher.ckpt').write_bytes(b'other trained teacher')
        selected = cache_directory(self.config, self.identity())
        self.assertNotEqual(selected, legacy)
        self.assertEqual(self.prepare(), selected)
        self.assertEqual((legacy / 'manifest.json').read_bytes(), old_manifest)
        self.assertEqual((legacy / 'part-00000000.pt').read_bytes(), old_shard)
        self.assertEqual(len(trajectories.CDLMTrajectoryDataset(selected, self.identity())), 6)

    def test_keyed_partial_resume_skips_complete_shard_and_recovers_missing_receipt(self):
        real = trajectories.collect_trajectory_batch
        calls = 0

        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('stopped between shards')
            return real(*args, **kwargs)

        with patch.object(trajectories, 'collect_trajectory_batch', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'stopped between shards'):
                self.prepare()
        directory = cache_directory(self.config, self.identity())
        first = directory / 'part-00000000.pt'
        previous_time = first.stat().st_mtime_ns
        first.with_suffix('.json').unlink()
        with patch.object(trajectories, 'collect_trajectory_batch', wraps=real) as collect:
            self.assertEqual(self.prepare(), directory)
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(first.stat().st_mtime_ns, previous_time)
        self.assertEqual(trajectories.read_cache_manifest(directory)['status'], 'complete')

    def test_corrupt_selected_cache_is_rejected_instead_of_silently_rebuilt(self):
        directory = self.prepare()
        identity = self.identity()
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['identity']['parameters']['teacher_checkpoint_sha256'] = '0' * 64
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'identity'):
            cache_directory(self.config, identity)
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.prepare()

    def test_collector_requires_and_records_configured_ema_teacher(self):
        teacher = Teacher()
        teacher._evaluation_weights = 'ema'
        with self.assertRaisesRegex(ValueError, 'configured teacher weights: live'):
            trajectories.prepare_trajectories(teacher, self.config, self.tokenizer, data())
        self.config.training.teacher_use_ema = True
        with self.assertRaisesRegex(ValueError, 'configured teacher weights: ema'):
            self.prepare()
        directory = trajectories.prepare_trajectories(teacher, self.config, self.tokenizer, data())
        manifest = trajectories.read_cache_manifest(directory)
        self.assertEqual(manifest['identity']['parameters']['teacher_weights'], 'ema')
        self.assertEqual(manifest['teacher_source'], str(self.config.training.finetune_path))
        self.assertEqual(Path(directory).parent.name, 'ema')

    def test_cache_ready_callback_reports_selected_path_without_changing_collection(self):
        ready = Mock(side_effect=RuntimeError('optional progress failed'))
        with redirect_stdout(io.StringIO()) as output:
            directory = self.prepare(on_cache_ready=ready)
        ready.assert_called_once_with(str(directory))
        self.assertEqual(trajectories.read_cache_manifest(directory)['status'], 'complete')
        self.assertIn('optional progress failed', output.getvalue())

    def test_snapshot_restore_monitor_final_export_order_and_failure_cleanup(self):
        self.config.checkpointing = {'snapshot': True, 'snapshot_trajectories': True,
                                     'trajectory_snapshot_namespace': 'cdlm_trajectories',
                                     'trajectory_snapshot_interval_seconds': 300}
        directory = cache_directory(self.config, self.identity())
        events = []
        controller = Mock()

        def restored():
            self.assertFalse((directory / 'manifest.json').exists())
            events.append('restore')

        def started():
            self.assertEqual(trajectories.read_cache_manifest(directory)['status'], 'building')
            events.append('start')

        def closed():
            events.append('close:' + trajectories.read_cache_manifest(directory)['status'])

        controller.restore.side_effect = restored
        controller.start.side_effect = started
        controller.close.side_effect = closed
        module = ModuleType('cdlm_trajectory_snapshot')
        module.TrajectorySnapshot = Mock(return_value=controller)
        with patch.dict(sys.modules, {'cdlm_trajectory_snapshot': module}):
            self.prepare(on_cache_ready=lambda path: events.append('ready'))
        self.assertEqual(events, ['restore', 'ready', 'start', 'close:complete'])
        self.assertEqual(module.TrajectorySnapshot.call_args.kwargs['rank'], 0)
        self.assertEqual(module.TrajectorySnapshot.call_args.kwargs['namespace'], 'cdlm_trajectories')
        # A later failed run with a different cache still publishes its partial work.
        self.config.seed += 1
        directory = cache_directory(self.config, self.identity())
        events.clear()
        with patch.dict(sys.modules, {'cdlm_trajectory_snapshot': module}), \
                patch.object(trajectories, 'collect_trajectory_batch', side_effect=RuntimeError('teacher failed')):
            with self.assertRaisesRegex(RuntimeError, 'teacher failed'):
                self.prepare()
        self.assertEqual(events, ['restore', 'start', 'close:building'])

    def test_disabled_snapshot_never_constructs_controller(self):
        self.config.checkpointing = {'snapshot': False, 'snapshot_trajectories': True}
        with patch.dict(sys.modules, {'cdlm_trajectory_snapshot': None}):
            self.prepare()

    def test_invalid_keys_cannot_escape_model_directories(self):
        identity = self.identity()
        identity['parameters']['teacher_weights'] = '../other'
        with self.assertRaisesRegex(ValueError, 'raw/ema'):
            cache_key(identity)


if __name__ == '__main__':
    unittest.main()
