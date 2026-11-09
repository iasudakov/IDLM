"""Stage byte-exact CDLM checkpoints before distributed workers are launched.

This module performs filesystem/Hub IO only; checkpoint architecture and EMA
selection are checked by the existing model loader after preflight succeeds.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

# Hub reads these transport switches at import time. The Nirvana launcher sets
# CDLM_DOWNLOAD_PROXY only for the preflight child, leaving training unchanged.
if os.environ.get('CDLM_DOWNLOAD_PROXY'):
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

from huggingface_hub import hf_hub_download

import nirvana_snapshot


MMD_SOURCE = 'hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt'
_CHECKPOINT = 'checkpoint.ckpt'
_METADATA = 'metadata.json'


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def _parse_source(source):
    parts = source[5:].split('/', 2)
    if (len(parts) != 3 or not all(parts)
            or any(part in ('.', '..') for part in parts[:2])
            or PurePosixPath(parts[2]).is_absolute()
            or '..' in PurePosixPath(parts[2]).parts):
        raise ValueError('A checkpoint Hub URI must be hf://owner/repository/filename.')
    return '/'.join(parts[:2]), parts[2]


def _verified(directory, source):
    """Return metadata only for a complete, checksum-verified managed asset."""
    checkpoint, metadata = directory / _CHECKPOINT, directory / _METADATA
    if not checkpoint.exists() or not metadata.exists():
        return None
    try:
        value = json.loads(metadata.read_text())
        if (value['source_uri'] != source or value['filename'] != _CHECKPOINT
                or not _nonempty(checkpoint)
                or value['size_bytes'] != checkpoint.stat().st_size
                or value['sha256'] != _sha256(checkpoint)):
            raise ValueError('source, size or SHA256 does not match')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'Corrupt CDLM checkpoint asset at {directory}; '
                         'remove or replace this managed asset before retrying.') from error
    return value


def _stage(path, directory, source):
    if not _nonempty(path):
        raise ValueError(f'Checkpoint is missing or empty: {path}')
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.checkpoint.', dir=directory)
    digest, size = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, 'wb') as output, path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                output.write(block)
                digest.update(block)
                size += len(block)
            output.flush()
            os.fsync(output.fileno())
        if size == 0:
            raise ValueError(f'Checkpoint is empty: {path}')
        os.replace(temporary, directory / _CHECKPOINT)
        value = dict(schema_version=1, source_uri=source, filename=_CHECKPOINT,
                     sha256=digest.hexdigest(), size_bytes=size)
        descriptor, temporary = tempfile.mkstemp(prefix='.metadata.', dir=directory)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / _METADATA)
        return value
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _asset_lock(directory):
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Outside the managed directory so no transient lock enters the snapshot.
    with (directory.parent / (directory.name + '.lock')).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _snapshot_location(directory, namespace):
    try:
        sdk = nirvana_snapshot._sdk()
        return nirvana_snapshot._paths(
            directory, sdk.snapshot.get_snapshot_path(), namespace)[2]
    except Exception:
        return None


def _persist(directory, source, metadata, enabled, namespace):
    if not enabled:
        return
    with nirvana_snapshot.STORE_LOCK:
        target = _snapshot_location(directory, namespace)
        if target is None:
            return
        try:
            existing = _verified(target, source)
            if existing is not None:
                if existing != metadata:
                    print('[cdlm-assets] Snapshot has different checkpoint bytes; '
                          'keeping the existing snapshot.', flush=True)
                return
            nirvana_snapshot.save(directory, namespace=namespace,
                                  reason='CDLM checkpoint asset')
        except Exception:
            # A valid local checkpoint is usable even if optional persistence fails.
            print('[cdlm-assets] Checkpoint snapshot was not saved; local asset is ready.',
                  flush=True)


def _restore(directory, source, enabled, namespace):
    if not enabled:
        return None
    with nirvana_snapshot.STORE_LOCK:
        target = _snapshot_location(directory, namespace)
        if target is None:
            return None
        # Validate before copying: a wrong source/corrupt snapshot must never be
        # silently interpreted as the requested teacher.
        metadata = _verified(target, source)
        if metadata is None:
            return None
        if not nirvana_snapshot.restore(directory, namespace=namespace):
            return None
        return _verified(directory, source)


def _staged_paths(code_dir, input_dir):
    if input_dir:
        yield input_dir / 'mmd_tinygsm.ckpt'
    yield code_dir / 'checkpoints/tinygsm/mmd_tinygsm.ckpt'
    yield code_dir / 'cdlm_model_assets/mmd_tinygsm.ckpt'


def _check_staged_metadata(path, source):
    provenance = path.with_suffix('.json')
    if not provenance.is_file():
        return
    try:
        value = json.loads(provenance.read_text())
        if (value['source_uri'] != source
                or value['size_bytes'] != path.stat().st_size
                or value['sha256'] != _sha256(path)):
            raise ValueError('packaged checkpoint checksum does not match')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'Invalid staged checkpoint provenance: {provenance}') from error


def _cache_directories(code_dir, input_dir, weights_dir):
    roots = [code_dir / 'cache_dir/cdlm_mmd_hf_cache',
             code_dir / 'cdlm_mmd_hf_cache', code_dir / 'cdlm_model_assets']
    if input_dir:
        roots[:0] = [input_dir / 'cdlm_mmd_hf_cache']
    if weights_dir:
        roots += [weights_dir / 'hf-cache', weights_dir]
    candidates = []
    for root in roots:
        # Accept either an HF hub cache itself or a staged HF_HOME directory.
        candidates.extend([root, root / 'hub'])
    return [path for path in dict.fromkeys(candidates) if path.is_dir()]


def prepare_checkpoint(source, *, code_dir, input_dir=None, weights_dir=None,
                       snapshot_enabled=True):
    """Resolve a local checkpoint or stage an explicitly requested Hub asset.

    A verified managed asset is immutable for its source URI. To intentionally
    refresh a moving Hub revision, remove that managed directory first.
    """
    source = os.fspath(source)
    code_dir = Path(code_dir).expanduser().resolve()
    if weights_dir:
        weights_dir = Path(weights_dir).expanduser()
        weights_dir = (weights_dir if weights_dir.is_absolute()
                       else code_dir / weights_dir).resolve()
    if not source.startswith('hf://'):
        path = Path(source).expanduser()
        candidates = ([path] if path.is_absolute() else [code_dir / path])
        if not path.is_absolute() and weights_dir:
            # Preserve model_assets.resolve_checkpoint's bundle-relative paths;
            # never substitute a basename or a different requested checkpoint.
            candidates.append(weights_dir / path)
        for candidate in candidates:
            candidate = candidate.resolve()
            if source and _nonempty(candidate):
                return str(candidate)
            if candidate.exists():
                break
        raise FileNotFoundError(
            'Explicit checkpoint is missing or empty. Checked: '
            + ', '.join(str(path.resolve()) for path in candidates)
            + '. Supply the requested file; no other weights were selected.')

    repo_id, filename = _parse_source(source)
    input_dir = input_dir or os.environ.get('INPUT_PATH')
    input_dir = Path(input_dir).expanduser().resolve() if input_dir else None
    key = hashlib.sha256(source.encode('utf-8')).hexdigest()
    directory = code_dir / 'cache_dir/cdlm_checkpoint_assets' / key
    namespace = 'cdlm_assets/' + key
    searched = [str(directory)]
    with _asset_lock(directory):
        metadata = _verified(directory, source)
        if metadata is not None:
            _persist(directory, source, metadata, snapshot_enabled, namespace)
            return str(directory / _CHECKPOINT)

        def finish(path):
            value = _stage(Path(path), directory, source)
            _persist(directory, source, value, snapshot_enabled, namespace)
            print(f'[cdlm-assets] Checkpoint ready: {directory / _CHECKPOINT}', flush=True)
            return str(directory / _CHECKPOINT)

        if source == MMD_SOURCE:
            for path in _staged_paths(code_dir, input_dir):
                searched.append(str(path))
                if path.exists():
                    _check_staged_metadata(path, source)
                    return finish(path)
        for cache in [*_cache_directories(code_dir, input_dir, weights_dir), None]:
            searched.append(str(cache) if cache else 'standard Hugging Face cache')
            kwargs = dict(repo_id=repo_id, filename=filename, local_files_only=True)
            if cache is not None:
                kwargs['cache_dir'] = str(cache)
            try:
                path = hf_hub_download(**kwargs)
            except Exception:
                continue
            return finish(path)
        searched.append('Nirvana snapshot ' + namespace)
        metadata = _restore(directory, source, snapshot_enabled, namespace)
        if metadata is not None:
            return str(directory / _CHECKPOINT)
        print(f'[cdlm-assets] Downloading requested checkpoint {source} '
              'from Hugging Face.', flush=True)
        from cdlm_download_proxy import download_proxy
        with download_proxy():
            try:
                downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception:
                # Hub and filesystem errors can contain credential-bearing URLs;
                # provide useful locations without reproducing request details.
                downloaded = None
        if downloaded is not None:
            # A local staging failure (for example, a full disk) is a separate
            # actionable error and must not become a misleading Hub miss.
            return finish(downloaded)
    raise FileNotFoundError(
        f'CDLM checkpoint {source} is unavailable before GPU launch. Searched:\n  '
        + '\n  '.join(searched)
        + '\nProvide a source bundle containing the checkpoint, stage the HF cache '
          'or the exact requested file, or run checkpoint prefetch on a machine '
          'with Hub access. Offline workers cannot download a missing checkpoint. '
          'Hub offline/proxy settings are respected.') from None
