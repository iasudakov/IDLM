"""Pristine source selection shared by standalone and integrated evaluation.

Training retains its existing main-branch runtime. These helpers do not import
models or install runtime patches in either release.
"""

import hashlib
import json
from pathlib import Path

from . import UPSTREAM_REVISION


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_REVISION = 'eb9f96c78ddcf3c9d601fbde1744eba45238b3c7'
SOURCES = {
    'main': ('vendor/didi_instruct', 'UPSTREAM.json', UPSTREAM_REVISION),
    'inference': ('vendor/didi_inference', 'INFERENCE_UPSTREAM.json', INFERENCE_REVISION),
}


def verify_source(source='main'):
    """Return the source directory and manifest after verifying the pinned bytes."""
    if source not in SOURCES:
        raise ValueError(f'Standalone DiDi source must be main|inference, got {source!r}')
    relative, manifest_name, revision = SOURCES[source]
    directory = ROOT / relative
    manifest = json.loads((ROOT / 'didi_repro' / manifest_name).read_text())
    if manifest['commit'] != revision or manifest['vendored_path'] != relative:
        raise RuntimeError(f'DiDi {source} source/manifest revision mismatch')
    for name, expected in manifest['files'].items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Vendored DiDi {source} source changed: {path}')
    return directory, manifest
