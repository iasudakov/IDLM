"""Incremental snapshots of committed CDLM trajectory shards, without GPU I/O."""

import hashlib
import json
import os
from functools import wraps
from pathlib import Path
import shutil
import tempfile
import threading

from cdlm_cache import cache_key
import nirvana_snapshot as snapshot


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _io_errors(method):
    @wraps(method)
    def guarded(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except OSError as error:
            print(f'[trajectory-snapshot] {method.__name__} I/O failed: {error}', flush=True)
            return False
    return guarded


class TrajectorySnapshot:
    def __init__(self, directory, identity, descriptors, enabled=True,
                 namespace='cdlm_trajectories', interval_seconds=300, rank=0):
        if interval_seconds <= 0:
            raise ValueError('Trajectory snapshot interval must be positive.')
        self.directory, self.identity = Path(directory).expanduser(), identity
        self.descriptors = [{key: d[key] for key in ('file', 'start', 'count')}
                            for d in descriptors]
        if any(Path(d['file']).name != d['file'] for d in self.descriptors):
            raise ValueError('Invalid trajectory shard filename.')
        self.total = sum(d['count'] for d in self.descriptors)
        self.namespace = str(Path(namespace) / cache_key(identity))
        self.enabled, self.rank, self.interval = enabled, rank, interval_seconds
        self._verified, self._dirty = {}, False
        self._thread, self._stop = None, threading.Event()
        self._closed, self._result = False, False
        self._labels = {}

    def _manifest(self, directory):
        path = directory / 'manifest.json'
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text())
            layout = [{key: d[key] for key in ('file', 'start', 'count')}
                      for d in value['shards']]
            valid = (value['schema_version'] == self.identity['parameters']['schema_version']
                     and value['identity'] == self.identity
                     and value['num_trajectories'] == self.total
                     and value['status'] in ('building', 'complete')
                     and layout == self.descriptors)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'Invalid trajectory snapshot manifest: {path}') from error
        if not valid:
            raise ValueError(f'Trajectory snapshot identity/layout mismatch: {path}')
        if 'teacher_source' in value:
            self._labels.setdefault('teacher_source', value['teacher_source'])
        return value

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    def _hash(self, path):
        key = self._stamp(path)
        if key not in self._verified:
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            self._verified[key] = digest.hexdigest()
        return self._verified[key]

    def _pair(self, directory, descriptor):
        path = directory / descriptor['file']
        receipt = path.with_suffix('.json')
        if not receipt.is_file():
            return None
        try:
            value = json.loads(receipt.read_text())
            valid = (value['fingerprint'] == self.identity['fingerprint']
                     and all(value[key] == descriptor[key] for key in ('file', 'start', 'count'))
                     and ('sha256' not in descriptor or value['sha256'] == descriptor['sha256']))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'Invalid trajectory shard receipt: {receipt}') from error
        if not valid:
            raise ValueError(f'Trajectory shard receipt mismatch: {receipt}')
        if not path.is_file():
            return None
        if self._hash(path) != value.get('sha256'):
            raise ValueError(f'Trajectory shard checksum mismatch: {path}')
        return value

    def _pairs(self, directory, manifest, strict=True):
        values = [self._pair(directory, d) for d in manifest['shards']]
        if strict and manifest['status'] == 'complete' and any(v is None for v in values):
            raise ValueError(f'Complete trajectory snapshot has missing shards: {directory}')
        return values

    def _copy_verified(self, source, destination, checksum):
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.' + destination.name + '.',
                                          suffix='.tmp', dir=destination.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            shutil.copy2(source, temporary)
            if self._hash(temporary) != checksum:
                raise ValueError(f'Trajectory copy checksum mismatch: {source}')
            os.replace(temporary, destination)
            self._verified[self._stamp(destination)] = checksum
        finally:
            temporary.unlink(missing_ok=True)

    def _write_manifest(self, directory, pairs):
        complete = all(value is not None for value in pairs)
        descriptors = [{**d, **({'sha256': p['sha256']} if complete else {})}
                       for d, p in zip(self.descriptors, pairs)]
        value = {'schema_version': self.identity['parameters']['schema_version'],
                 'identity': self.identity, 'status': 'complete' if complete else 'building',
                 'num_trajectories': self.total, 'shards': descriptors, **self._labels}
        path = directory / 'manifest.json'
        content = _encoded(value)
        if path.is_file() and path.read_bytes() == content:
            return False
        directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.manifest.', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return True

    def _store(self):
        try:
            sdk = snapshot._sdk()
            root = sdk.snapshot.get_snapshot_path()
        except Exception as error:
            print(f'[trajectory-snapshot] SDK unavailable: {error}', flush=True)
            return None
        _, root, target = snapshot._paths(self.directory, root, self.namespace)
        return sdk, root, target

    @_io_errors
    def restore(self):
        if not self.enabled or self.rank != 0:
            return False
        with snapshot.STORE_LOCK:
            store = self._store()
            if store is None:
                return False
            local = self._manifest(self.directory)
            local_pairs = (self._pairs(self.directory, local, strict=False) if local is not None
                           else [self._pair(self.directory, d) for d in self.descriptors])
            _, _, source = store
            remote = self._manifest(source)
            if remote is None:
                return False
            try:
                remote_pairs = self._pairs(source, remote)
                if any(a is not None and b is not None and a['sha256'] != b['sha256']
                       for a, b in zip(local_pairs, remote_pairs)):
                    raise ValueError('Conflicting immutable trajectory shards for the same identity.')
                for descriptor, pair in zip(self.descriptors, remote_pairs):
                    path = self.directory / descriptor['file']
                    if self._pair(self.directory, descriptor) is not None or path.exists():
                        continue  # Preserve valid pairs and unconfirmed local tensors.
                    if pair is not None:
                        path.with_suffix('.json').unlink(missing_ok=True)
                        self._copy_verified(source / descriptor['file'], path, pair['sha256'])
                        snapshot._copy_file((source / descriptor['file']).with_suffix('.json'),
                                            path.with_suffix('.json'))
                pairs = [self._pair(self.directory, d) for d in self.descriptors]
                self._write_manifest(self.directory, pairs)
                print(f'[trajectory-snapshot] restored {source} into {self.directory}', flush=True)
                return True
            except OSError as error:
                print(f'[trajectory-snapshot] restore failed: {error}', flush=True)
                return False

    @_io_errors
    def _publish(self, force=False):
        with snapshot.STORE_LOCK:
            store = self._store()
            if store is None:
                return False
            local = self._manifest(self.directory)
            if local is None:
                return False
            pairs = self._pairs(self.directory, local)
            sdk, root, target = store
            remote = self._manifest(target)
            if remote is not None:
                self._pairs(target, remote)
            previous_pairs = [self._pair(target, d) for d in self.descriptors]
            if any(a is not None and b is not None and a['sha256'] != b['sha256']
                   for a, b in zip(pairs, previous_pairs)):
                raise ValueError('Conflicting immutable trajectory shards for the same identity.')
            try:
                for descriptor, pair, previous in zip(self.descriptors, pairs, previous_pairs):
                    if pair is None:
                        continue
                    if previous is not None and previous['sha256'] == pair['sha256']:
                        continue
                    path = self.directory / descriptor['file']
                    # An orphan receipt cannot certify the next tensor if the
                    # process stops between its tensor and receipt publication.
                    (target / descriptor['file']).with_suffix('.json').unlink(missing_ok=True)
                    self._copy_verified(path, target / descriptor['file'], pair['sha256'])
                    snapshot._copy_file(path.with_suffix('.json'),
                                        (target / descriptor['file']).with_suffix('.json'))
                    self._dirty = True
                saved = [self._pair(target, d) for d in self.descriptors]
                self._dirty |= self._write_manifest(target, saved)
                if self._dirty or force:
                    try:
                        sdk.snapshot.dump_snapshot(str(root))
                    except Exception as error:
                        print(f'[trajectory-snapshot] publication failed: {error}', flush=True)
                        return False
                    self._dirty = False
                    count = sum(p['count'] for p in saved if p is not None)
                    print(f'[trajectory-snapshot] pushed {target}: {count}/{self.total} '
                          f'trajectories, {sum(p is not None for p in saved)}/{len(saved)} shards', flush=True)
                return True
            except OSError as error:
                print(f'[trajectory-snapshot] copy failed: {error}', flush=True)
                return False

    def start(self):
        if self.enabled and self.rank == 0 and self._thread is None and not self._closed:
            self._thread = threading.Thread(target=self._run, name='cdlm-trajectory-snapshot', daemon=True)
            try:
                self._thread.start()
            except RuntimeError as error:
                self._thread = None
                print(f'[trajectory-snapshot] monitor could not start; final export remains enabled: {error}',
                      flush=True)
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                self._publish()
            except Exception as error:
                print(f'[trajectory-snapshot] monitor failed: {error}', flush=True)
            if self._stop.wait(self.interval):
                break

    def close(self):
        if self._closed:
            return self._result
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._closed = True
        if self.enabled and self.rank == 0:
            self._result = self._publish(force=True)
        return self._result
