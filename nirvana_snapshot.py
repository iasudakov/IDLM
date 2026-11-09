"""Best-effort persistence of a run directory in Nirvana's snapshot store.

The SDK is imported lazily. Checkpoint writers call ``save`` after completing
local writes; startup calls ``restore`` before looking for a resume checkpoint.
Only global rank zero copies files or calls the snapshot API.
"""

import importlib
import os
from pathlib import Path
import shutil
import tempfile
import threading


STORE_LOCK = threading.RLock()


def _rank():
    # Global identifiers take precedence over LOCAL_RANK on multi-node jobs.
    for name in ('RANK', 'SLURM_PROCID', 'LOCAL_RANK'):
        try:
            return int(os.environ[name])
        except (KeyError, ValueError):
            continue
    return 0


def _sdk():
    sdk = importlib.import_module('nirvana_dl')
    if not callable(sdk.snapshot.get_snapshot_path) or not callable(sdk.snapshot.dump_snapshot):
        raise ImportError('nirvana_dl.snapshot API is unavailable')
    return sdk


def available():
    """Whether the optional Nirvana snapshot API can be imported."""
    try:
        _sdk()
        return True
    except Exception:
        return False


def _paths(path, snapshot_root, namespace):
    run, root = Path(path).expanduser().resolve(), Path(snapshot_root).expanduser().resolve()
    if run == root or run in root.parents or root in run.parents:
        raise ValueError('Run directory and snapshot store must not overlap')
    target = root
    if namespace is not None:
        relative = Path(namespace)
        if not str(namespace) or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Snapshot namespace must be a relative path without ..')
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise ValueError('Snapshot namespace escapes the store')
    return run, root, target


def _copy_file(source, destination):
    """Publish each complete file atomically; dereference last.ckpt symlinks."""
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp',
                                              dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary, follow_symlinks=True)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _copy_directory(source, destination):
    if not source.is_dir():
        raise FileNotFoundError(f'Snapshot source directory is missing: {source}')
    if destination.is_symlink():
        raise ValueError(f'Snapshot destination directory cannot be a symlink: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        target = destination / entry.name
        if entry.is_dir():
            # File links are portable after dereferencing; directory links can
            # include the snapshot store or recursively include the run itself.
            if entry.is_symlink():
                raise ValueError(f'Snapshot source directory cannot be a symlink: {entry}')
            _copy_directory(entry, target)
        elif entry.is_file():
            _copy_file(entry, target)
        else:
            raise ValueError(f'Unsupported snapshot entry: {entry}')


def save(path, enabled=True, reason='', namespace=None):
    """Copy a run into its snapshot namespace, then publish the store root."""
    if not enabled or _rank() != 0:
        return False
    try:
        sdk = _sdk()
    except Exception as error:
        print(f'[snapshot] Nirvana SDK unavailable; not saving: {error}', flush=True)
        return False
    with STORE_LOCK:
        try:
            run, root, target = _paths(path, sdk.snapshot.get_snapshot_path(), namespace)
            _copy_directory(run, target)
            sdk.snapshot.dump_snapshot(str(root))
            suffix = f' ({reason})' if reason else ''
            print(f'[snapshot] pushed {run} into {target}{suffix}', flush=True)
            return True
        except Exception as error:
            print(f'[snapshot] push failed, continuing: {error}', flush=True)
            return False


def restore(path, enabled=True, namespace=None):
    """Restore the requested namespace before checking the resume checkpoint."""
    if not enabled or _rank() != 0:
        return False
    try:
        sdk = _sdk()
    except Exception as error:
        print(f'[snapshot] Nirvana SDK unavailable; not restoring: {error}', flush=True)
        return False
    with STORE_LOCK:
        try:
            run, _, source = _paths(path, sdk.snapshot.get_snapshot_path(), namespace)
            if not source.is_dir():
                print(f'[snapshot] no saved directory at {source}; not restoring', flush=True)
                return False
            _copy_directory(source, run)
            print(f'[snapshot] restored {source} into {run}', flush=True)
            return True
        except Exception as error:
            print(f'[snapshot] restore failed, continuing: {error}', flush=True)
            return False
