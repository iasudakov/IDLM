"""Choose the latest CDLM snapshot checkpoint before MMD starts GPU workers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from cdlm_checkpoint_assets import prepare_checkpoint
from cdlm_preflight import compose_config
from cdlm_snapshot_init import select_checkpoint


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_overrides(config, *, code_dir=None):
    if config.algo.name != 'mmd_finetune':
        raise ValueError('MMD snapshot preflight requires algo=mmd_finetune.')
    if config.mode != 'train':
        raise ValueError('The MMD launcher supports mode=train.')
    code_dir = Path(code_dir or Path.cwd()).resolve()
    # Full MMD resumes restore both student and feature network from their own
    # checkpoint. They must not depend on the input CDLM state still existing.
    if config.checkpointing.resume_from_ckpt:
        print('[mmd-init] Full MMD resume: skipping CDLM checkpoint selection.', flush=True)
        return []
    if config.training.student_init_path:
        raise ValueError('This launcher selects the latest CDLM snapshot checkpoint. '
                         'Use mmd.cdlm_snapshot_root / mmd.cdlm_snapshot_namespace '
                         'to select its source, or invoke main directly for an explicit student.')
    source_root = config.mmd.get('cdlm_snapshot_root')
    if source_root:
        source_root = Path(str(source_root)).expanduser()
        if not source_root.is_absolute():
            source_root = code_dir / source_root
    selected = select_checkpoint(source_root, config.mmd.get('cdlm_snapshot_namespace'))
    output_namespace = config.checkpointing.get('snapshot_namespace')
    if config.checkpointing.get('snapshot', False):
        if not output_namespace:
            raise ValueError('MMD snapshots require a separate nonempty snapshot_namespace.')
        output = Path(str(output_namespace))
        source = Path(selected['namespace'])
        if (output.is_absolute() or '..' in output.parts or source == output
                or source in output.parents or output in source.parents):
            raise ValueError('MMD snapshot_namespace must be separate from the input CDLM namespace.')

    feature_source = config.training.finetune_path
    if not feature_source:
        raise ValueError('training.finetune_path must identify the original MDLM feature checkpoint.')
    weights_dir = config.get('model_weights_dir')
    if weights_dir:
        weights_dir = Path(str(weights_dir)).expanduser()
        if not weights_dir.is_absolute():
            weights_dir = code_dir / weights_dir
    if (weights_dir and str(feature_source).startswith('hf://')
            and not config.training.teacher_allow_hub):
        raise ValueError('With an offline bundle, use a local original MDLM checkpoint '
                         'or explicitly enable training.teacher_allow_hub.')
    feature = str(prepare_checkpoint(
        str(feature_source), code_dir=code_dir, input_dir=os.environ.get('INPUT_PATH'),
        weights_dir=weights_dir,
        snapshot_enabled=bool(config.checkpointing.get('snapshot', False))))
    output_dir = Path(str(config.checkpointing.save_dir)).expanduser()
    if not output_dir.is_absolute():
        output_dir = code_dir / output_dir
    snapshot_root = Path(selected['snapshot_root']).resolve()
    destination = output_dir.resolve()
    if destination == snapshot_root or snapshot_root in destination.parents:
        raise ValueError('MMD save_dir must be outside the input snapshot directory.')
    manifest = output_dir / 'mmd_initialization.json'
    provenance = {
        'schema_version': 1,
        'reference_run': '3e0ac6b1-7a8f-43c3-af58-d9e21b886da7',
        'reference_figure_checkpoint_step': 7000,
        'student': {**selected, 'weights': 'ema' if config.training.student_init_use_ema else 'raw',
                    'retain_ema_history': bool(config.mmd.retain_student_ema)},
        'feature_extractor': {'path': feature, 'sha256': _hash(feature),
                              'weights': 'ema' if config.training.teacher_use_ema else 'raw'},
        'fresh_optimizer_and_scheduler': True,
    }
    _atomic_write(manifest, (json.dumps(provenance, indent=2) + '\n').encode())
    values = {'training.student_init_path': selected['path'],
              'training.finetune_path': feature,
              'mmd.initialization_manifest': str(manifest.resolve())}
    print(f"[mmd-init] CDLM step={selected['global_step']} from {selected['path']}", flush=True)
    print(f'[mmd-init] Frozen MDLM features: {feature}; provenance: {manifest}', flush=True)
    return [f'{key}={json.dumps(value, ensure_ascii=False)}' for key, value in values.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='NUL-separated Hydra overrides.')
    options, overrides = parser.parse_known_args(argv)
    if overrides[:1] == ['--']:
        overrides = overrides[1:]
    output = Path(options.output).expanduser()
    try:
        output.unlink(missing_ok=True)
        config = compose_config(overrides, resolved_fields=(
            'checkpointing.save_dir', 'checkpointing.snapshot_namespace',
            'mmd.cdlm_snapshot_root', 'mmd.cdlm_snapshot_namespace'))
        values = prepare_overrides(config)
        data = ('\0'.join(values) + '\0').encode() if values else b''
        _atomic_write(output, data)
        return 0
    except Exception as error:
        print(f'[mmd-init] Failed before GPU launch: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
