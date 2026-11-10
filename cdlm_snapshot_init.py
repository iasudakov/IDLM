"""Select a completed CDLM checkpoint from an unpacked Nirvana snapshot.

Selection reads checkpoint metadata on CPU. It does not restore training state,
copy tracker metadata, or run the model. Snapshot archives must already be
unpacked, as they are when the PythonDL ``state`` input is made available to
``nirvana_dl.snapshot.get_snapshot_path()``.
"""

from collections.abc import Mapping
import hashlib
import logging
from pathlib import Path


LOG = logging.getLogger(__name__)
_NON_TRAINING_NAMESPACES = frozenset({
    'cdlm_assets', 'cdlm_trajectories', 'cdlm_checkpoint_assets',
})


class _InvalidCheckpoint(ValueError):
    """A readable file that cannot initialize a CDLM-derived student."""


def _read_checkpoint(path):
    # Keep imports of Torch/Hydra out of module discovery and simple path checks.
    from sdtt_checkpoint import read_checkpoint
    return read_checkpoint(str(path))


def _within(path, root):
    return path == root or root in path.parents


def _namespace_path(root, namespace):
    relative = Path(namespace)
    if not str(namespace) or relative.is_absolute() or '..' in relative.parts:
        raise ValueError('CDLM_SOURCE_NAMESPACE must be a relative snapshot path without ..')
    directory = (root / relative).resolve()
    if not _within(directory, root):
        raise ValueError('CDLM_SOURCE_NAMESPACE escapes the snapshot root')
    return directory


def _algorithm_names(checkpoint):
    """Recognize both native Lightning config and standalone export config."""
    configs = [checkpoint.get('config')]
    parameters = checkpoint.get('hyper_parameters')
    if isinstance(parameters, Mapping):
        configs.append(parameters.get('config'))
    names = []
    for config in configs:
        if not isinstance(config, Mapping):
            continue
        algorithm = config.get('algo')
        if isinstance(algorithm, Mapping) and algorithm.get('name') is not None:
            names.append(str(algorithm['name']))
    return names


def _metadata(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise _InvalidCheckpoint('checkpoint payload is not a mapping')
    names = _algorithm_names(checkpoint)
    # A downstream MMD checkpoint must not be accepted just because it happens
    # to retain a provenance field from its CDLM initialization.
    if names and any(name != 'cdlm' for name in names):
        raise _InvalidCheckpoint('checkpoint algorithm is not CDLM')
    if not names and 'cdlm_cache_fingerprint' not in checkpoint:
        raise _InvalidCheckpoint('no CDLM checkpoint marker or CDLM configuration')
    step = checkpoint.get('global_step')
    if type(step) is not int or step < 0:
        raise _InvalidCheckpoint('global_step must be a nonnegative integer')
    state = checkpoint.get('state_dict')
    if not isinstance(state, Mapping):
        raise _InvalidCheckpoint('missing state_dict')
    from torch import Tensor
    if not any(isinstance(key, str) and key.startswith('backbone.')
               and isinstance(value, Tensor) for key, value in state.items()):
        raise _InvalidCheckpoint('missing backbone tensors')
    epoch = checkpoint.get('epoch')
    fingerprint = checkpoint.get('cdlm_cache_fingerprint')
    return {
        'global_step': step,
        'epoch': epoch if type(epoch) is int and epoch >= 0 else None,
        'cdlm_cache_fingerprint': fingerprint if isinstance(fingerprint, str) else None,
    }


def _candidates(directory, root):
    checkpoint_dir = directory / 'checkpoints'
    if not checkpoint_dir.is_dir():
        return []
    if not _within(checkpoint_dir.resolve(), root):
        raise ValueError(f'Checkpoint directory escapes the snapshot root: {checkpoint_dir}')
    candidates = []
    for path in sorted(checkpoint_dir.glob('*.ckpt')):
        try:
            if not _within(path.resolve(), directory):
                raise _InvalidCheckpoint('checkpoint symlink escapes its training namespace')
            before = path.stat()
            checkpoint = _read_checkpoint(path)
            metadata = _metadata(checkpoint)
            del checkpoint
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise _InvalidCheckpoint('checkpoint changed while being inspected')
        except Exception as error:
            # Never echo arbitrary deserializer error text or serialized data.
            detail = str(error) if isinstance(error, _InvalidCheckpoint) else type(error).__name__
            LOG.warning('[cdlm-snapshot] Skipping %s: %s', path, detail)
            continue
        candidates.append({
            **metadata, 'path': str(path.absolute()),
            'size_bytes': after.st_size, '_mtime_ns': after.st_mtime_ns,
        })
    return candidates


def select_checkpoint(snapshot_root=None, namespace=None):
    """Return the latest valid CDLM checkpoint's path and provenance as a dict.

    ``snapshot_root`` defaults to the Nirvana SDK snapshot directory. An explicit
    ``namespace`` selects a relative training directory inside it (nested names
    are supported). Without it, inspect immediate child namespaces and a legacy
    root-level ``checkpoints/`` directory. Multiple valid CDLM namespaces require
    an explicit choice: optimizer steps cannot order independent experiments.

    Within one namespace, rank by the *serialized* ``global_step`` and prefer
    ``last.ckpt`` on ties. Broken or non-CDLM checkpoints are reported and skipped.
    The returned dict contains ``path``, ``snapshot_root``, ``namespace``,
    ``global_step``, ``epoch``, ``size_bytes``, ``sha256`` and
    ``cdlm_cache_fingerprint``. It contains no loaded tensors or old run state.
    """
    if snapshot_root is None:
        import nirvana_snapshot
        try:
            snapshot_root = nirvana_snapshot._sdk().snapshot.get_snapshot_path()
        except Exception as error:
            raise RuntimeError(
                'Cannot locate the Nirvana snapshot. Connect the previous state or '
                'set CDLM_SNAPSHOT_ROOT to an already unpacked snapshot directory.'
            ) from error
    root = Path(snapshot_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'CDLM snapshot directory does not exist: {root}')
    if namespace is not None:
        directories = [_namespace_path(root, namespace)]
    else:
        directories = [root]
        directories.extend(path for path in sorted(root.iterdir())
                           if path.is_dir() and not path.is_symlink()
                           and path.name not in _NON_TRAINING_NAMESPACES)
    available = {}
    for directory in directories:
        candidates = _candidates(directory, root)
        if candidates:
            available[directory] = candidates
    if not available:
        location = str(directories[0]) if namespace is not None else str(root)
        raise FileNotFoundError(
            f'No valid CDLM checkpoint with global_step and backbone weights in {location}. '
            'Connect a snapshot containing training checkpoints; trajectory-only state '
            'cannot initialize the trained CDLM student. For a nested training directory, '
            'set CDLM_SOURCE_NAMESPACE to its relative path.')
    if len(available) > 1:
        choices = ', '.join(repr(str(directory.relative_to(root))) for directory in available)
        raise ValueError(
            f'Multiple CDLM training namespaces are present: {choices}. '
            'Set CDLM_SOURCE_NAMESPACE explicitly; steps from different runs are not comparable.')
    directory, candidates = next(iter(available.items()))
    selected = max(candidates, key=lambda item: (
        item['global_step'], Path(item['path']).name == 'last.ckpt', item['path']))
    path = Path(selected['path'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (selected['size_bytes'], selected.pop('_mtime_ns')):
        raise RuntimeError('Selected CDLM checkpoint changed while hashing; retry with a stable snapshot.')
    return {
        **selected, 'snapshot_root': str(root),
        'namespace': str(directory.relative_to(root)), 'sha256': digest.hexdigest(),
    }
