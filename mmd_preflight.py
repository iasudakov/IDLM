"""Prepare CDLM or original-MDLM initialization before MMD starts GPU workers."""

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from cdlm_checkpoint_assets import prepare_checkpoint
from cdlm_preflight import compose_config
from cdlm_snapshot_init import resolve_snapshot_root, select_checkpoint
from full_precision import validate_full_fp32_config


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


def _local_path(value, code_dir, label):
    if value is None or not str(value).strip() or '://' in str(value):
        raise ValueError(f'{label} must identify a local path for the Nirvana launcher.')
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else code_dir / path


def _validate_full_mmd_state(path, require_ema):
    from sdtt_checkpoint import read_checkpoint
    from torch import Tensor

    try:
        checkpoint = read_checkpoint(str(path))
    except Exception as error:
        raise ValueError(f'Cannot read MMD resume state at {path}: {type(error).__name__}') from error
    error_message = (f'A full MMD training state is required at {path}. '
                     'Use the final last.ckpt; periodic student-only weights cannot resume training.')
    if not isinstance(checkpoint, Mapping) or checkpoint.get('mmd_finetune_schema') != 1:
        raise ValueError(error_message)
    step = checkpoint.get('global_step')
    state = checkpoint.get('state_dict')
    if type(step) is not int or step < 0 or not isinstance(state, Mapping):
        raise ValueError(error_message)
    for prefix in ('backbone.', 'teacher.'):
        if not any(isinstance(key, str) and key.startswith(prefix) and isinstance(value, Tensor)
                   for key, value in state.items()):
            raise ValueError(error_message)
    for field in ('optimizer_states', 'lr_schedulers'):
        values = checkpoint.get(field)
        if not isinstance(values, list) or not values or not all(isinstance(v, Mapping) for v in values):
            raise ValueError(error_message)
    if not isinstance(checkpoint.get('loops'), Mapping) or not checkpoint['loops']:
        raise ValueError(error_message)
    if require_ema:
        ema = checkpoint.get('ema')
        if (not isinstance(ema, Mapping) or not isinstance(ema.get('shadow_params'), (list, tuple))
                or not ema['shadow_params'] or not all(isinstance(value, Tensor) for value in ema['shadow_params'])):
            raise ValueError(error_message + ' EMA weights are missing.')
    return step


def validate_resume_checkpoint(config, *, code_dir=None):
    """Validate and restore full MMD state before main can launch GPU workers."""
    code_dir = Path(code_dir or Path.cwd()).resolve()
    target = _local_path(config.checkpointing.get('resume_ckpt_path'), code_dir,
                         'checkpointing.resume_ckpt_path')
    require_ema = config.training.ema > 0
    if target.exists() or target.is_symlink():
        # A corrupt explicitly requested local state must not fall back to an
        # older snapshot. Preserve the same local-first policy as main._train.
        if not target.is_file():
            raise FileNotFoundError(f'MMD resume state is not a readable file: {target}')
        step = _validate_full_mmd_state(target, require_ema)
    else:
        if not config.checkpointing.get('snapshot', False):
            raise FileNotFoundError(f'MMD resume state is missing: {target}; snapshot restore is disabled.')
        import nirvana_snapshot
        run = _local_path(config.checkpointing.save_dir, code_dir, 'checkpointing.save_dir').resolve()
        try:
            relative = target.resolve().relative_to(run)
        except ValueError as error:
            raise FileNotFoundError(
                f'MMD resume state is missing: {target}. Snapshot restoration only supplies files '
                'inside checkpointing.save_dir.') from error
        try:
            root = resolve_snapshot_root()
        except (RuntimeError, FileNotFoundError) as error:
            raise FileNotFoundError(
                f'MMD resume state is missing: {target}; no usable Nirvana snapshot is connected. '
                'Connect the MMD state or set checkpointing.resume_ckpt_path to an existing '
                'local full checkpoint.') from error
        namespace = config.checkpointing.get('snapshot_namespace')
        _, _, source_dir = nirvana_snapshot._paths(run, root, namespace)
        source = source_dir / relative
        if not source.resolve().is_relative_to(source_dir):
            raise ValueError('MMD resume checkpoint escapes its snapshot namespace.')
        if not source.is_file():
            raise FileNotFoundError(
                f'MMD resume state is missing from the connected snapshot: {source}. '
                'Connect the MMD state containing the full final last.ckpt and select its '
                'checkpointing.snapshot_namespace; a trajectory-only or periodic-weights state is insufficient.')
        step = _validate_full_mmd_state(source, require_ema)
        if not nirvana_snapshot.restore(str(run), enabled=True, namespace=namespace) or not target.is_file():
            raise RuntimeError(f'MMD state restoration failed: {source} -> {target}. Training was not started.')
    print(f'[mmd-init] Full MMD state verified: step={step}, checkpoint={target}', flush=True)
    return str(target)


def prepare_overrides(config, *, code_dir=None):
    if config.algo.name != 'mmd_finetune':
        raise ValueError('MMD snapshot preflight requires algo=mmd_finetune.')
    if config.mode != 'train':
        raise ValueError('The MMD launcher supports mode=train.')
    validate_full_fp32_config(config)
    code_dir = Path(code_dir or Path.cwd()).resolve()
    # Full MMD resumes restore both student and feature network from their own
    # checkpoint. They must not depend on the input CDLM state still existing.
    if config.checkpointing.resume_from_ckpt:
        target = validate_resume_checkpoint(config, code_dir=code_dir)
        return ['checkpointing.resume_ckpt_path=' + json.dumps(target, ensure_ascii=False)]
    source_kind = config.mmd.get('initialization_source', 'cdlm_snapshot')
    if source_kind not in ('cdlm_snapshot', 'mdlm'):
        raise ValueError('mmd.initialization_source must be cdlm_snapshot or mdlm.')
    if source_kind == 'mdlm':
        if config.training.student_init_path:
            raise ValueError('The MDLM recipe uses training.finetune_path for both student and features; '
                             'leave training.student_init_path=null.')
        feature = _prepare_feature_checkpoint(config, code_dir)
        selected = {'path': feature, 'sha256': _hash(feature),
                    'source_kind': 'mdlm', 'source': str(config.training.finetune_path)}
        print(f'[mmd-init] Original MDLM selected for student and frozen features: {feature}', flush=True)
        return _write_initialization(config, code_dir, selected, feature)
    if config.training.student_init_path:
        raise ValueError('This launcher selects the latest CDLM snapshot checkpoint. '
                         'Use mmd.cdlm_snapshot_root / mmd.cdlm_snapshot_namespace '
                         'to select its source, or invoke main directly for an explicit student.')
    source_root = config.mmd.get('cdlm_snapshot_root')
    if source_root is not None:
        if not str(source_root).strip():
            raise FileNotFoundError('CDLM snapshot state path is empty; connect the previous state.')
        source_root = Path(str(source_root)).expanduser()
        if not source_root.is_absolute():
            source_root = code_dir / source_root
    print('[mmd-init] Checking required CDLM snapshot state before loading model assets.', flush=True)
    selected = select_checkpoint(source_root, config.mmd.get('cdlm_snapshot_namespace'))
    print(f"[mmd-init] CDLM state verified: step={selected['global_step']}, checkpoint={selected['path']}",
          flush=True)
    output_namespace = config.checkpointing.get('snapshot_namespace')
    if config.checkpointing.get('snapshot', False):
        if not output_namespace:
            raise ValueError('MMD snapshots require a separate nonempty snapshot_namespace.')
        output = Path(str(output_namespace))
        source = Path(selected['namespace'])
        if (output.is_absolute() or '..' in output.parts or source == output
                or source in output.parents or output in source.parents):
            raise ValueError('MMD snapshot_namespace must be separate from the input CDLM namespace.')

    feature = _prepare_feature_checkpoint(config, code_dir)
    return _write_initialization(config, code_dir, selected, feature)


def _prepare_feature_checkpoint(config, code_dir):
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
    return str(prepare_checkpoint(
        str(feature_source), code_dir=code_dir, input_dir=os.environ.get('INPUT_PATH'),
        weights_dir=weights_dir,
        snapshot_enabled=bool(config.checkpointing.get('snapshot', False))))


def _write_initialization(config, code_dir, selected, feature):
    if config.checkpointing.get('snapshot', False):
        namespace = config.checkpointing.get('snapshot_namespace')
        if (not namespace or not Path(str(namespace)).parts
                or Path(str(namespace)).is_absolute() or '..' in Path(str(namespace)).parts):
            raise ValueError('MMD snapshots require a nonempty relative snapshot_namespace.')
    output_dir = Path(str(config.checkpointing.save_dir)).expanduser()
    if not output_dir.is_absolute():
        output_dir = code_dir / output_dir
    destination = output_dir.resolve()
    if selected.get('snapshot_root'):
        snapshot_root = Path(selected['snapshot_root']).resolve()
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
        'precision': {'trainer': str(config.trainer.precision),
                      'full_fp32': bool(config.model.get('force_fp32', False)),
                      'tf32': False if config.model.get('force_fp32', False) else None},
    }
    _atomic_write(manifest, (json.dumps(provenance, indent=2) + '\n').encode())
    values = {'training.student_init_path': selected['path'],
              'training.finetune_path': feature,
              'mmd.initialization_manifest': str(manifest.resolve())}
    if selected.get('global_step') is not None:
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
            'checkpointing.save_dir', 'checkpointing.snapshot_namespace', 'checkpointing.resume_ckpt_path',
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
