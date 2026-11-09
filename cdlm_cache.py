"""Stable teacher/weight/configuration directories for CDLM trajectory caches."""

import json
from pathlib import Path
import re


def cache_root(config):
    """Resolve the configured cache root against Hydra's original launch cwd."""
    configured = config.algo.get('trajectory_cache_dir')
    if not configured:
        configured = Path(str(config.data.cache_dir)) / 'cdlm_trajectories'
    directory = Path(str(configured)).expanduser()
    if not directory.is_absolute():
        try:
            from hydra.utils import to_absolute_path
            directory = Path(to_absolute_path(str(directory)))
        except ImportError:
            directory = directory.absolute()
    return directory


def cache_key(identity):
    """The generating teacher and selected weights identify the model folder."""
    parameters = identity['parameters']
    teacher = parameters['teacher_checkpoint_sha256']
    fingerprint = identity['fingerprint']
    weights = parameters['teacher_weights']
    if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
           for value in (teacher, fingerprint)) or weights not in ('raw', 'ema'):
        raise ValueError('CDLM cache identity requires SHA256 hashes and raw/ema teacher weights.')
    return f'teacher-{teacher[:16]}/{weights}/{fingerprint}'


def _check_selected(directory, identity):
    if (directory / 'manifest.json').exists():
        # Import lazily: the collector itself uses this path resolver.
        from cdlm_trajectories import read_cache_manifest
        manifest = read_cache_manifest(directory)
        if manifest['identity']['fingerprint'] != identity['fingerprint']:
            raise ValueError('Selected CDLM trajectory cache fingerprint mismatch with current inputs.')
    return directory


def cache_directory(config, identity):
    """Select an isolated child, retaining a matching flat legacy cache.

    Path relocation and student initialization do not affect this choice. No
    directory/configuration is mutated and unrelated legacy caches are retained.
    """
    root = cache_root(config)
    if not config.algo.get('trajectory_cache_per_model', False):
        return _check_selected(root, identity)
    legacy = root / 'manifest.json'
    if legacy.is_file():
        try:
            candidate = json.loads(legacy.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f'Unreadable legacy CDLM cache manifest: {legacy}') from error
        if candidate.get('identity', {}).get('fingerprint') == identity['fingerprint']:
            return _check_selected(root, identity)
    return _check_selected(root / cache_key(identity), identity)
