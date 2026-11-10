"""Evaluate the published TinyGSM MMD checkpoint with the CDLM comparison protocol.

Only checkpoint preparation may download the explicitly selected weights. The
shared GPU worker runs offline, without a teacher, training or trajectories.
"""

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import sys

from cdlm_checkpoint_assets import MMD_SOURCE, prepare_checkpoint
from cdlm_checkpoint_eval import compose_evaluation_config, execute_plan


def inspect_checkpoint(path):
    """Read native checkpoint metadata/EMA on CPU without constructing the model.

    Reject malformed exports and missing EMA before torchrun. Parameter names,
    exact architecture and EMA ordering are subsequently checked by the shared
    worker's strict backbone loader; arbitrary buffers are not EMA parameters.
    """
    from sdtt_checkpoint import read_checkpoint, backbone_state
    from torch import Tensor

    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'MMD evaluation checkpoint missing or empty: {path}')
    before = path.stat()
    try:
        checkpoint = read_checkpoint(str(path))
    except Exception as error:
        # Serialized objects/deserializer errors can contain unrelated private
        # fields. Report only the requested local file and exception class.
        raise ValueError(f'Cannot read MMD evaluation checkpoint {path}: {type(error).__name__}') from None
    if not isinstance(checkpoint, Mapping):
        raise ValueError('MMD evaluation checkpoint must be a mapping.')
    step = checkpoint.get('global_step')
    if type(step) is not int or step < 0:
        raise ValueError('MMD evaluation checkpoint must contain a nonnegative integer global_step; '
                         'the evaluation step is never inferred from its filename.')
    raw_state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(raw_state, Mapping) or not all(isinstance(key, str) for key in raw_state):
        raise ValueError('MMD evaluation checkpoint state_dict must be a string-keyed mapping.')
    state = backbone_state(checkpoint)
    if not all(isinstance(value, Tensor) for value in state.values()):
        raise ValueError('MMD evaluation checkpoint backbone entries must be tensors.')
    if not any(value.is_floating_point() and value.numel() for value in state.values()):
        raise ValueError('MMD evaluation checkpoint has no nonempty floating backbone weights.')
    marker = checkpoint.get('checkpoint_weights')
    if marker not in (None, 'raw', 'ema'):
        raise ValueError(f'Unsupported checkpoint_weights marker: {marker!r}')
    ema_count = None
    if marker == 'ema':
        checkpoint_format = 'ema_weights_only'
    else:
        ema = checkpoint.get('ema')
        shadows = ema.get('shadow_params') if isinstance(ema, Mapping) else None
        if (not isinstance(shadows, (list, tuple)) or not shadows
                or not all(isinstance(value, Tensor) and value.is_floating_point()
                           and value.numel() for value in shadows)):
            raise ValueError('MMD evaluation requires EMA inference weights: supply an EMA-only export '
                             'or a full checkpoint with nonempty ema.shadow_params. Raw weights are not substituted.')
        available = Counter(tuple(value.shape) for value in state.values() if value.is_floating_point())
        needed = Counter(tuple(value.shape) for value in shadows)
        if needed - available:
            raise ValueError('MMD checkpoint EMA shapes/counts cannot match its backbone weights.')
        ema_count = len(shadows)
        checkpoint_format = 'full_checkpoint_with_ema'
    epoch = checkpoint.get('epoch')
    metadata = {'global_step': step, 'epoch': epoch if type(epoch) is int else None,
                'checkpoint_format': checkpoint_format, 'ema_tensor_count': ema_count,
                'size_bytes': before.st_size}
    del state, checkpoint
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('MMD evaluation checkpoint changed during preflight; retry with stable input weights.')
    metadata['sha256'] = digest.hexdigest()
    return metadata


def build_single_plan(checkpoint_path, source, output_dir, *, snapshot_namespace='mmd_checkpoint_eval'):
    """Build a one-checkpoint plan compatible with the existing comparison worker."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError('An explicit nonempty MMD checkpoint source is required.')
    if output_dir is None or not str(output_dir).strip():
        raise ValueError('An evaluation output directory is required.')
    if (not isinstance(snapshot_namespace, str) or not snapshot_namespace
            or Path(snapshot_namespace).is_absolute()
            or '..' in Path(snapshot_namespace).parts or snapshot_namespace == '.'):
        raise ValueError('Evaluation snapshot namespace must be a nonempty relative path without ..')
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if checkpoint == output or output in checkpoint.parents:
        raise ValueError('Evaluation output must not contain the input MMD checkpoint.')
    metadata = inspect_checkpoint(checkpoint)
    source_id = ('hf-yresearch-mdlm-mmd-tinygsm' if source == MMD_SOURCE else
                 'mmd-checkpoint-' + hashlib.sha256(source.encode('utf-8')).hexdigest()[:12])
    step = metadata['global_step']
    return {
        'schema_version': 1, 'output_dir': str(output),
        'snapshot_namespace': snapshot_namespace, 'checkpoint_source': source,
        'checkpoint_metadata': metadata,
        'entries': [{'source_run': source_id, 'family': 'MDLM to MMD',
                     'checkpoint': str(checkpoint), 'step': step,
                     'output_dir': str(output / source_id / f'step={step:08d}')}],
        'skipped_checkpoints': [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=os.environ.get('MMD_EVAL_CHECKPOINT', MMD_SOURCE),
                        help='Published hf:// checkpoint (default) or explicit local .ckpt path.')
    parser.add_argument('--output-dir', default=os.environ.get(
        'MMD_EVAL_OUTPUT_DIR', str(Path.cwd() / 'mmd_checkpoint_eval')))
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--no-restore-results', action='store_true')
    options, overrides = parser.parse_known_args(argv)
    if overrides[:1] == ['--']:
        overrides = overrides[1:]
    if not options.checkpoint or not str(options.checkpoint).strip():
        raise ValueError('--checkpoint / MMD_EVAL_CHECKPOINT must not be empty.')
    if not options.output_dir or not str(options.output_dir).strip():
        raise ValueError('--output-dir / MMD_EVAL_OUTPUT_DIR must not be empty.')
    if options.devices <= 0 or any('=' not in value or value.startswith('--') for value in overrides):
        raise ValueError('Use a positive --devices value and Hydra key=value overrides.')
    code_dir, output = Path.cwd().resolve(), Path(options.output_dir).expanduser().resolve()
    defaults = ['logger.name="mmd checkpoint evaluation"',
                'checkpointing.snapshot_namespace=mmd_checkpoint_eval']
    # Fail on missing offline tokenizer/GSM8K assets before spending time or
    # bandwidth on the explicitly requested model download.
    config = compose_evaluation_config(defaults + overrides, code_dir=code_dir,
                                       output_dir=output, devices=options.devices)
    print('[mmd-checkpoint-eval] Preparing the selected MMD weights before GPU workers.', flush=True)
    checkpoint = prepare_checkpoint(
        options.checkpoint, code_dir=code_dir, input_dir=os.environ.get('INPUT_PATH'),
        weights_dir=config.model_weights_dir,
        snapshot_enabled=bool(config.checkpointing.get('snapshot', False)))
    plan = build_single_plan(checkpoint, options.checkpoint, output,
                             snapshot_namespace=config.checkpointing.snapshot_namespace)
    metadata = plan['checkpoint_metadata']
    print(f'[mmd-checkpoint-eval] Verified saved step={metadata["global_step"]}; '
          f'{metadata["checkpoint_format"]}; SHA256={metadata["sha256"]}.', flush=True)
    # Asset preparation above respects the caller's offline/proxy settings.
    # Fresh worker processes are always offline and receive only local paths.
    worker_env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', TOKENIZERS_PARALLELISM='false')
    return execute_plan(plan, config, devices=options.devices, plan_only=options.plan_only,
                        restore_results=not options.no_restore_results, worker_env=worker_env)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f'[mmd-checkpoint-eval] Preflight/evaluation failed: {error}', file=sys.stderr, flush=True)
        raise SystemExit(2)
