"""Build a portable committed CDLM checkout with the published MMD checkpoint."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


PUBLISHED_MMD = 'hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt'
ASSET_DIRECTORY = 'code/cdlm_model_assets'
CHECKPOINT_MEMBER = ASSET_DIRECTORY + '/mmd_tinygsm.ckpt'
PROVENANCE_MEMBER = ASSET_DIRECTORY + '/mmd_tinygsm.json'


def _prepare_checkpoint(source, **kwargs):
    # Packaging itself needs only the standard library until asset resolution.
    from cdlm_checkpoint_assets import prepare_checkpoint
    return prepare_checkpoint(source, **kwargs)


def _git(directory, *arguments):
    result = subprocess.run(['git', '-C', str(directory), *arguments],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _ensure_clean(directory):
    dirty = _git(directory, 'status', '--porcelain=v1', '--untracked-files=no')
    if dirty:
        raise RuntimeError('Tracked source has uncommitted changes. Commit or restore them '
                           'before packaging HEAD:\n' + dirty)


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_info(name, size, timestamp, *, directory=False):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.mode = 0o755 if directory else 0o644
    info.size = size
    info.mtime = timestamp
    return info


def _append_assets(archive, checkpoint, provenance, timestamp):
    from io import BytesIO

    with tarfile.open(archive, 'a') as bundle:
        names = {member.name.rstrip('/'): member for member in bundle.getmembers()}
        if any(name in names for name in (CHECKPOINT_MEMBER, PROVENANCE_MEMBER)):
            raise RuntimeError('Committed source already contains the packaged MMD asset paths.')
        existing = names.get(ASSET_DIRECTORY)
        if existing is not None and not existing.isdir():
            raise RuntimeError('Committed cdlm_model_assets path must be a directory.')
        if existing is None:
            bundle.addfile(_asset_info(ASSET_DIRECTORY, 0, timestamp, directory=True))
        with checkpoint.open('rb') as source:
            bundle.addfile(_asset_info(CHECKPOINT_MEMBER, checkpoint.stat().st_size,
                                       timestamp), source)
        payload = (json.dumps(provenance, indent=2, sort_keys=True) + '\n').encode('utf-8')
        bundle.addfile(_asset_info(PROVENANCE_MEMBER, len(payload), timestamp), BytesIO(payload))


def package(output, code_dir=None):
    """Archive clean tracked HEAD and exact MMD bytes, returning the output path.

    Untracked and ignored files are excluded. Managed downloads and the source
    tar live in temporary staging, independently of the checkout/output location.
    An existing output is replaced only after the complete archive is written.
    """
    requested_source = Path(code_dir or Path(__file__).resolve().parent).expanduser().resolve()
    source = Path(_git(requested_source, 'rev-parse', '--show-toplevel'))
    _ensure_clean(source)
    commit = _git(source, 'rev-parse', 'HEAD')
    timestamp = int(_git(source, 'show', '-s', '--format=%ct', commit))
    destination = Path(output).expanduser().resolve()
    if destination.is_dir():
        raise ValueError(f'Output must be an archive file: {destination}')
    if destination.is_relative_to(source):
        relative = destination.relative_to(source)
        tracked = _git(source, 'ls-files', '--', str(relative))
        if tracked:
            raise ValueError(f'Output would replace tracked source: {destination}')

    with tempfile.TemporaryDirectory(prefix='cdlm-package-') as temporary:
        staging = Path(temporary)
        asset_code = staging / 'asset-source'
        asset_code.mkdir()
        checkpoint = Path(_prepare_checkpoint(
            PUBLISHED_MMD, code_dir=asset_code, input_dir=os.environ.get('INPUT_PATH'),
            weights_dir=None, snapshot_enabled=False)).expanduser().resolve(strict=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Resolved MMD checkpoint is not a file: {checkpoint}')
        provenance = {'schema_version': 1, 'source_uri': PUBLISHED_MMD,
                      'sha256': _file_digest(checkpoint),
                      'size_bytes': checkpoint.stat().st_size, 'git_commit': commit}
        # Asset resolution may take time; fail if the checkout changed meanwhile.
        _ensure_clean(source)
        if _git(source, 'rev-parse', 'HEAD') != commit:
            raise RuntimeError('Source HEAD changed during checkpoint preparation; package again.')
        archive = staging / 'source.tar'
        with archive.open('wb') as target:
            subprocess.run(['git', '-C', str(source), 'archive', '--format=tar',
                            '--prefix=code/', commit], stdout=target,
                           stderr=subprocess.PIPE, check=True)
        _append_assets(archive, checkpoint, provenance, timestamp)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_output = tempfile.mkstemp(
            prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
        try:
            with os.fdopen(descriptor, 'wb') as target:
                with archive.open('rb') as raw, gzip.GzipFile(
                        filename='', mode='wb', fileobj=target, mtime=0) as compressed:
                    shutil.copyfileobj(raw, compressed, length=8 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_output, destination)
        finally:
            Path(temporary_output).unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True,
                        help='Destination tar.gz containing code/ and the MMD EMA checkpoint.')
    arguments = parser.parse_args(argv)
    try:
        output = package(arguments.output)
    except Exception as error:
        parser.exit(1, f'cdlm_package: {error}\n')
    print(output)


if __name__ == '__main__':
    main()
