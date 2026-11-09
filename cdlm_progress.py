"""Best-effort CDLM preparation heartbeats, independent of GPU work and DDP."""

import json
from pathlib import Path
import threading
import time

import hydra.utils
from omegaconf import OmegaConf


class PreparationProgress:
    """Monitor confirmed cache receipts on rank zero without changing the cache.

    Tracker I/O stays in a daemon thread. Preparation resumes its own Underdeep
    metadata under ``cdlm_prepare``; the later training run is independent.
    """

    def __init__(self, config, rank=0, interval_seconds=30):
        if interval_seconds <= 0:
            raise ValueError('interval_seconds must be positive.')
        self.config, self.rank = config, rank
        self.interval_seconds = interval_seconds
        configured_dir = config.algo.get('trajectory_cache_dir')
        if not configured_dir:
            configured_dir = Path(str(config.data.cache_dir)) / 'cdlm_trajectories'
        self.directory = Path(str(configured_dir)).expanduser()
        if not self.directory.is_absolute():
            try:
                self.directory = Path(hydra.utils.to_absolute_path(str(self.directory)))
            except ImportError:
                self.directory = self.directory.absolute()
        count = config.algo.get('trajectory_num_examples')
        self.expected_total = (int(count) * len(config.algo.trajectory_temperatures)
                               if count is not None else 0)
        size = int(config.algo.get('trajectory_shard_size', 8))
        self.expected_shards = (self.expected_total + size - 1) // size
        self._thread = None
        self._stop = threading.Event()
        self._status = 'complete'
        self._started = None
        self._baseline_done = 0

    def _snapshot(self):
        total, shards = self.expected_total, self.expected_shards
        done = completed = 0
        try:
            manifest = json.loads((self.directory / 'manifest.json').read_text())
            total = int(manifest['num_trajectories'])
            descriptors = manifest['shards']
            shards = len(descriptors)
            fingerprint = manifest['identity']['fingerprint']
            for descriptor in descriptors:
                name = descriptor['file']
                if Path(name).name != name:
                    continue
                tensor = self.directory / name
                try:
                    receipt = json.loads(tensor.with_suffix('.json').read_text())
                    if (not tensor.is_file()
                            or receipt.get('fingerprint') != fingerprint
                            or any(receipt.get(key) != descriptor[key]
                                   for key in ('file', 'start', 'count'))
                            or ('sha256' in descriptor and receipt.get('sha256') != descriptor['sha256'])):
                        continue
                    done += int(descriptor['count'])
                    completed += 1
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return done, total, completed, shards

    def _metrics(self, snapshot, status):
        done, total, completed, shards = snapshot
        elapsed = max(0., time.monotonic() - self._started)
        return {
            'prep/cdlm_trajectories_done': done, 'prep/cdlm_trajectories_total': total,
            'prep/cdlm_shards_done': completed, 'prep/cdlm_shards_total': shards,
            'prep/cdlm_progress': min(1., done / total) if total else 0.,
            'prep/cdlm_elapsed_seconds': elapsed,
            'prep/cdlm_trajectories_per_second': max(0, done - self._baseline_done) / elapsed if elapsed else 0.,
            'prep/cdlm_complete': int(status == 'complete'),
            'prep/cdlm_failed': int(status == 'failed'),
        }

    def _make_logger(self):
        configured = self.config.get('logger')
        if not configured or configured.get('_target_') != 'underdeep_logger.UnderdeepLogger':
            return None
        options = OmegaConf.to_container(configured, resolve=True)
        options.update(name=str(options.get('name', 'cdlm')) + ' [prepare]',
                       save_dir=str(Path(str(self.config.checkpointing.save_dir)) / 'cdlm_prepare'),
                       resume=True)
        return hydra.utils.instantiate(options)

    def _emit(self, snapshot, status, logger):
        metrics = self._metrics(snapshot, status)
        done, total, completed, shards = snapshot
        print(f'[cdlm-prepare] state={status} trajectories={done}/{total} '
              f'shards={completed}/{shards} elapsed={metrics["prep/cdlm_elapsed_seconds"]:.1f}s '
              f'new_trajectories_per_second={metrics["prep/cdlm_trajectories_per_second"]:.3f}', flush=True)
        if logger is not None:
            try:
                logger.log_metrics(metrics, step=done)
            except Exception as error:
                print(f'[cdlm-prepare] tracker log failed: {error}', flush=True)

    def _run(self):
        logger = None
        try:
            initial = self._snapshot()
            self._baseline_done = initial[0]
            # Console visibility must not wait for tracker initialization.
            self._emit(initial, 'running', None)
            try:
                logger = self._make_logger()
            except Exception as error:
                print(f'[cdlm-prepare] tracker setup failed: {error}', flush=True)
            self._emit(initial, 'running', logger)
            while not self._stop.wait(self.interval_seconds):
                self._emit(self._snapshot(), 'running', logger)
            self._emit(self._snapshot(), self._status, logger)
        except Exception as error:
            print(f'[cdlm-prepare] progress monitoring failed: {error}', flush=True)
        finally:
            if logger is not None:
                try:
                    logger.finalize(self._status if self._stop.is_set() else 'failed')
                except Exception as error:
                    print(f'[cdlm-prepare] tracker finalization failed: {error}', flush=True)

    def start(self):
        if self.rank != 0 or self._thread is not None or self._stop.is_set():
            return self
        self._started = time.monotonic()
        print(f'[cdlm-prepare] started cache={self.directory} '
              f'heartbeat_seconds={self.interval_seconds:g}', flush=True)
        self._thread = threading.Thread(target=self._run, name='cdlm-preparation-progress', daemon=True)
        try:
            self._thread.start()
        except RuntimeError as error:
            self._thread = None
            print(f'[cdlm-prepare] could not start progress monitor: {error}', flush=True)
        return self

    def close(self, status='complete'):
        if status not in ('complete', 'failed'):
            raise ValueError('status must be complete or failed.')
        if self.rank != 0 or self._stop.is_set():
            return
        self._status = status
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            # A broken tracker cannot hold up process teardown. The daemon is
            # solely responsible for finalization, avoiding concurrent SDK use.
            self._thread.join(timeout=2.)
