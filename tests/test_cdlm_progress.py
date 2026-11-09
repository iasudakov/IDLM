"""CDLM progress uses tiny receipt fixtures and fake trackers, without models."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from cdlm_progress import PreparationProgress


class Logger:
    def __init__(self):
        self.records, self.finalized, self.threads = [], [], []
        self.first = threading.Event()

    def log_metrics(self, metrics, step=None):
        self.records.append((dict(metrics), step))
        self.threads.append(threading.get_ident())
        self.first.set()

    def finalize(self, status):
        self.finalized.append((status, threading.get_ident()))


def config(directory, target='underdeep_logger.UnderdeepLogger'):
    return OmegaConf.create({
        'algo': {'trajectory_cache_dir': str(directory / 'cache'),
                 'trajectory_num_examples': 8, 'trajectory_temperatures': [0., .5],
                 'trajectory_shard_size': 8},
        'checkpointing': {'save_dir': str(directory / 'training')},
        'logger': {'_target_': target, 'experiment': 'generative-models/tinygsm',
                   'name': 'cdlm run', 'save_dir': str(directory / 'training'),
                   'resume': False, 'force_new_run': True},
    })


def manifest(directory):
    directory.mkdir()
    payload = {'num_trajectories': 16, 'identity': {'fingerprint': 'teacher-data-id'},
               'status': 'building',
               'shards': [{'file': f'part-{i:08d}.pt', 'start': i * 8, 'count': 8}
                          for i in range(2)]}
    (directory / 'manifest.json').write_text(json.dumps(payload))
    return payload


def receipt(directory, descriptor, *, fingerprint='teacher-data-id', tensor=True):
    if tensor:
        (directory / descriptor['file']).write_bytes(b'fixture; no tensor allocation')
    (directory / descriptor['file']).with_suffix('.json').write_text(json.dumps({
        **descriptor, 'fingerprint': fingerprint, 'sha256': 'fixture-hash'}))


class PreparationProgressTest(unittest.TestCase):
    def test_constructor_is_idle_and_nonzero_rank_does_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, patch('cdlm_progress.hydra.utils.instantiate') as create:
            progress = PreparationProgress(config(Path(tmp)), rank=1, interval_seconds=.01)
            self.assertIsNone(progress._thread)
            with redirect_stdout(io.StringIO()) as output:
                self.assertIs(progress.start(), progress)
                progress.close('failed')
            self.assertEqual(output.getvalue(), '')
            create.assert_not_called()

    def test_initial_heartbeat_and_separate_resumable_tracker(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg = config(directory)
            logger = Logger()
            with patch('cdlm_progress.hydra.utils.instantiate', return_value=logger) as create:
                with redirect_stdout(io.StringIO()) as output:
                    progress = PreparationProgress(cfg, interval_seconds=.01)
                    progress.start()
                    self.assertTrue(logger.first.wait(1.))
                    progress.close()
                self.assertIn('[cdlm-prepare] started', output.getvalue())
                self.assertIn('trajectories=0/16 shards=0/2', output.getvalue())
                options = create.call_args.args[0]
                self.assertEqual(options['name'], 'cdlm run [prepare]')
                self.assertEqual(options['save_dir'], str(directory / 'training/cdlm_prepare'))
                self.assertEqual(options['experiment'], 'generative-models/tinygsm')
                self.assertTrue(options['resume'])
                self.assertTrue(options['force_new_run'])
            self.assertEqual(cfg.logger.name, 'cdlm run')
            self.assertFalse(cfg.logger.resume)
            self.assertTrue(cfg.logger.force_new_run)
            self.assertFalse((directory / 'cache').exists())
            self.assertEqual(logger.records[0][0]['prep/cdlm_progress'], 0.)
            self.assertEqual(logger.records[-1][0]['prep/cdlm_complete'], 1)
            self.assertEqual(logger.records[-1][0]['prep/cdlm_failed'], 0)
            self.assertEqual(logger.finalized[0][0], 'complete')
            self.assertNotEqual(logger.finalized[0][1], threading.get_ident())
            self.assertEqual(set(logger.threads), {logger.finalized[0][1]})

    def test_confirmed_receipts_only_and_resumed_work_excluded_from_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg = config(directory)
            cache = directory / 'cache'
            saved = manifest(cache)
            receipt(cache, saved['shards'][0])
            # A stray tensor without its receipt is not confirmed work.
            (cache / saved['shards'][1]['file']).write_bytes(b'unconfirmed')
            logger = Logger()
            with patch('cdlm_progress.hydra.utils.instantiate', return_value=logger), redirect_stdout(io.StringIO()):
                progress = PreparationProgress(cfg, interval_seconds=.01).start()
                self.assertTrue(logger.first.wait(1.))
                initial = logger.records[0][0]
                self.assertEqual(initial['prep/cdlm_trajectories_done'], 8)
                self.assertEqual(initial['prep/cdlm_shards_done'], 1)
                self.assertEqual(initial['prep/cdlm_progress'], .5)
                self.assertEqual(initial['prep/cdlm_trajectories_per_second'], 0.)
                receipt(cache, saved['shards'][1])
                progress.close('failed')
            last = logger.records[-1][0]
            self.assertEqual(last['prep/cdlm_trajectories_done'], 16)
            self.assertEqual(last['prep/cdlm_shards_done'], 2)
            self.assertEqual(last['prep/cdlm_progress'], 1.)
            self.assertAlmostEqual(last['prep/cdlm_trajectories_per_second']
                                   * last['prep/cdlm_elapsed_seconds'], 8.)
            self.assertEqual(last['prep/cdlm_complete'], 0)
            self.assertEqual(last['prep/cdlm_failed'], 1)
            self.assertEqual(logger.finalized[0][0], 'failed')

    def test_corrupt_receipts_wrong_fingerprint_and_missing_tensors_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cache = directory / 'cache'
            saved = manifest(cache)
            progress = PreparationProgress(config(directory))
            receipt(cache, saved['shards'][0], fingerprint='wrong-teacher')
            receipt(cache, saved['shards'][1], tensor=False)
            self.assertEqual(progress._snapshot(), (0, 16, 0, 2))
            (cache / saved['shards'][0]['file']).with_suffix('.json').write_text('{not json')
            self.assertEqual(progress._snapshot(), (0, 16, 0, 2))
            (cache / 'manifest.json').write_text('{not json')
            self.assertEqual(progress._snapshot(), (0, 16, 0, 2))

    def test_other_logger_targets_use_console_only(self):
        with tempfile.TemporaryDirectory() as tmp, patch('cdlm_progress.hydra.utils.instantiate') as create:
            with redirect_stdout(io.StringIO()) as output:
                progress = PreparationProgress(config(Path(tmp), 'some.TensorboardLogger'), interval_seconds=.01)
                progress.start()
                progress.close('failed')
            create.assert_not_called()
            self.assertIn('state=failed', output.getvalue())

    def test_relative_cache_path_uses_hydra_original_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg = config(directory)
            cfg.algo.trajectory_cache_dir = 'relative/cache'
            with patch('hydra.core.hydra_config.HydraConfig.initialized', return_value=True), \
                    patch('hydra.utils.get_original_cwd', return_value=str(directory / 'checkout')), \
                    patch('os.getcwd', return_value=str(directory / 'hydra-output')):
                progress = PreparationProgress(cfg)
            self.assertEqual(progress.directory, directory / 'checkout/relative/cache')
            cfg.algo.trajectory_cache_dir = None
            cfg.data = {'cache_dir': str(directory / 'data-cache')}
            self.assertEqual(PreparationProgress(cfg).directory, directory / 'data-cache/cdlm_trajectories')

    def test_configured_force_new_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(Path(tmp))
            cfg.logger.force_new_run = False
            with patch('cdlm_progress.hydra.utils.instantiate') as create:
                PreparationProgress(cfg)._make_logger()
            self.assertFalse(create.call_args.args[0]['force_new_run'])
            self.assertTrue(create.call_args.args[0]['resume'])

    def test_monitor_thread_start_failure_does_not_abort_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('cdlm_progress.threading.Thread.start', side_effect=RuntimeError('thread unavailable')):
                with redirect_stdout(io.StringIO()) as output:
                    progress = PreparationProgress(config(Path(tmp))).start()
                    progress.close('failed')
            self.assertIsNone(progress._thread)
            self.assertIn('could not start progress monitor: thread unavailable', output.getvalue())

    def test_tracker_failure_is_best_effort(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('cdlm_progress.hydra.utils.instantiate', side_effect=RuntimeError('offline tracker')):
                with redirect_stdout(io.StringIO()) as output:
                    progress = PreparationProgress(config(Path(tmp)), interval_seconds=.01).start()
                    progress.close('failed')
            self.assertIn('tracker setup failed: offline tracker', output.getvalue())
            self.assertIn('state=failed', output.getvalue())

    def test_periodic_zero_part_heartbeat_and_bounded_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = Logger()
            with patch('cdlm_progress.hydra.utils.instantiate', return_value=logger), redirect_stdout(io.StringIO()):
                progress = PreparationProgress(config(Path(tmp)), interval_seconds=.01).start()
                self.assertTrue(logger.first.wait(1.))
                deadline = time.monotonic() + 1.
                while len(logger.records) < 2 and time.monotonic() < deadline:
                    time.sleep(.005)
                progress.close()
            self.assertGreaterEqual(len(logger.records), 3)
            self.assertTrue(all(record['prep/cdlm_trajectories_done'] == 0 for record, _ in logger.records))
            self.assertFalse(progress._thread.is_alive())
            progress.close('failed')
            self.assertEqual(len(logger.finalized), 1)

    def test_blocked_tracker_does_not_block_caller_or_race_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            entered, release = threading.Event(), threading.Event()
            logger = Logger()
            real_log = logger.log_metrics

            def blocked_log(*args, **kwargs):
                entered.set()
                release.wait(5.)
                real_log(*args, **kwargs)

            logger.log_metrics = blocked_log
            with patch('cdlm_progress.hydra.utils.instantiate', return_value=logger), redirect_stdout(io.StringIO()):
                progress = PreparationProgress(config(Path(tmp)), interval_seconds=.01).start()
                self.assertTrue(entered.wait(1.))
                started = time.monotonic()
                progress.close('failed')
                self.assertLess(time.monotonic() - started, 2.5)
                self.assertEqual(logger.finalized, [])
                release.set()
                progress._thread.join(1.)
            self.assertFalse(progress._thread.is_alive())
            self.assertEqual(len(logger.finalized), 1)
            self.assertEqual(logger.finalized[0][0], 'failed')


if __name__ == '__main__':
    unittest.main()
