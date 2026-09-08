#!/usr/bin/env python3
"""Map the authors' safetensors into a checkpoint for UNMODIFIED DiDi main.py.

This is an evaluation-only format adapter, not training or a model conversion:
the released generator is written under both backbone.* (used by the original
sampler) and student_ema.*, and the complete discriminator state is preserved.
No teacher, optimizer, IDLM runtime patches, or tracker is involved.

CUDA serialization is intentional. Original Lightning checkpoint loading moves
the model to the first saved tensor's device; a CPU checkpoint would leave the
original sampler on CPU, where its FlashAttention implementation cannot run.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from didi_repro import UPSTREAM_REVISION
from didi_repro.initialization import (
    AUTHORS_STUDENT_REPO, WEIGHT_FILES, _resolve_files, _sha256, _validate_state)
from didi_repro.runtime import UPSTREAM_ROOT, verify_upstream


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--student-hf', default=AUTHORS_STUDENT_REPO,
                        help='HF repository or local directory containing both safetensors files')
    result.add_argument('--revision', default=None,
                        help='Immutable HF revision; defaults to the pinned authors revision')
    result.add_argument('--cache-dir', default=None)
    result.add_argument('--local-files-only', action='store_true',
                        help='Require cached/local weights and tokenizers; do not download')
    result.add_argument('--output', type=Path, required=True,
                        help='New .ckpt path; neither this file nor its .json sidecar may exist')
    result.add_argument('--device', default='cuda:0', choices=('cuda', 'cuda:0', 'cpu'),
                        help='Saved tensor device; CPU is for inspection/tests, not original GPU evaluation')
    return result


def output_paths(output):
    output = Path(output).expanduser().absolute()
    sidecar = output.with_name(output.name + '.json')
    for path in (output, sidecar):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    return output, sidecar


def saved_device(requested, torch_module):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Prepare the checkpoint in one process, not with torchrun.')
    if requested == 'cpu':
        return torch_module.device('cpu')
    if requested not in ('cuda', 'cuda:0'):
        raise ValueError('Use cuda:0 with exactly one visible GPU, or cpu for inspection.')
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise ValueError('CUDA output requires exactly one visible GPU. '
                         'Set CUDA_VISIBLE_DEVICES to the selected GPU before launching.')
    return torch_module.device('cuda:0')


def pristine_model():
    """Build the original native architecture on CPU without installing repairs."""
    verify_upstream()
    for name in ('main', 'algo', 'models', 'trainer_base', 'dataloader', 'metrics', 'utils'):
        module = sys.modules.get(name)
        if module is not None:
            location = Path(getattr(module, '__file__', '') or '').resolve()
            if not location.is_relative_to(UPSTREAM_ROOT):
                raise RuntimeError(f'Conflicting {name} import; run this script in its own process.')
    sys.path.insert(0, str(UPSTREAM_ROOT))
    upstream = importlib.import_module('main')
    if getattr(upstream.algo.DiDiInstruct, '_didi_runtime_installed', False):
        raise RuntimeError('Refusing a patched DiDi runtime; preparation requires pristine upstream.')
    import hydra
    from omegaconf import OmegaConf
    import torch
    with hydra.initialize_config_dir(
            version_base=None, config_dir=str(UPSTREAM_ROOT / 'configs')):
        config = hydra.compose(config_name='config', overrides=[
            'mode=sample_eval', 'model=small', 'algo=didi_instruct',
            'algo.backbone=dit', 'sampling.predictor=guided', 'trainer.devices=1'])
    # No dataloaders are constructed. This only obtains the vocabulary used by
    # the original model and its metrics tokenizer during model construction.
    tokenizer = upstream.dataloader.get_tokenizer(config)
    with torch.device('cpu'):
        model = upstream.algo.DiDiInstruct(config, tokenizer=tokenizer)
    architecture = {
        'model': OmegaConf.to_container(config.model, resolve=True),
        'backbone': str(config.algo.backbone),
        'vocabulary_size': model.vocab_size,
        'tokenizer': str(config.data.tokenizer_name_or_path),
        'model_class': f'{type(model).__module__}.{type(model).__name__}',
        'runtime_patches_installed': False,
    }
    return model, architecture


def mapped_state(model, generator, discriminator, device):
    """Validate every learned tensor/buffer before transferring or writing any."""
    _validate_state(model.backbone, generator, 'generator')
    _validate_state(model.student_ema, generator, 'student EMA')
    _validate_state(model.discriminator, discriminator, 'discriminator')
    state = {}
    for name, value in generator.items():
        tensor = value.detach().to(device=device)
        # Both names deliberately alias one unchanged tensor. torch.save keeps
        # the shared storage, avoiding a duplicate generator in the output.
        state[f'backbone.{name}'] = tensor
        state[f'student_ema.{name}'] = tensor
    for name, value in discriminator.items():
        state[f'discriminator.{name}'] = value.detach().to(device=device)
    return state


def write_checkpoint(output, state, metadata, lightning_version):
    """Exclusive files: never overwrite an existing checkpoint or provenance."""
    import torch
    output, sidecar = output_paths(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'state_dict': state, 'epoch': 0, 'global_step': 0,
        'pytorch-lightning_version': lightning_version,
        'didi_authors_checkpoint': metadata,
    }
    with output.open('xb') as stream:
        torch.save(checkpoint, stream)
    record = dict(metadata)
    record['checkpoint'] = {
        'path': str(output), 'sha256': _sha256(output),
        'size_bytes': output.stat().st_size,
    }
    with sidecar.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return output, sidecar


def main(argv=None):
    options = parser().parse_args(argv)
    output_paths(options.output)  # Fail before downloads or model allocation.
    if options.local_files_only:
        # Set before importing HF/Transformers, including upstream imports.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    device = saved_device(options.device, torch)
    files, source = _resolve_files({
        'student_hf': options.student_hf, 'revision': options.revision,
        'cache_dir': options.cache_dir, 'local_files_only': options.local_files_only})
    from safetensors.torch import load_file
    generator = load_file(str(files[WEIGHT_FILES[0]]), device='cpu')
    discriminator = load_file(str(files[WEIGHT_FILES[1]]), device='cpu')
    source['files'] = {
        name: {'path': str(path), 'sha256': _sha256(path)} for name, path in files.items()}
    print('[didi-checkpoint] Validating against the pristine native CPU model.', flush=True)
    model, architecture = pristine_model()
    state = mapped_state(model, generator, discriminator, device)
    del model
    lightning_version = importlib.metadata.version('lightning')
    metadata = {
        'kind': 'authors_safetensors_for_unmodified_upstream_evaluation',
        'upstream_revision': UPSTREAM_REVISION,
        'source': source, 'architecture': architecture,
        'remapping': {
            'model.safetensors': ['backbone.*', 'student_ema.*'],
            'discriminator.safetensors': ['discriminator.*'],
            'generator_aliases_share_storage': True,
            'values_and_dtypes_changed': False,
            'spectral_normalization_buffers': 'preserved',
            'omitted_unused_state': ['teacher', 'legacy EMA', 'optimizers', 'training progress'],
        },
        'saved_tensor_device': str(device),
        'intended_use': ('unmodified vendor/didi_instruct/main.py mode=sample_eval'
                         if device.type == 'cuda' else 'CPU inspection only; reprepare on CUDA for original GPU evaluation'),
        'environment': {
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'lightning': lightning_version, 'torch_cuda': torch.version.cuda,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        },
    }
    output, sidecar = write_checkpoint(options.output, state, metadata, lightning_version)
    print(f'[didi-checkpoint] Saved {output}', flush=True)
    print(f'[didi-checkpoint] Provenance {sidecar}', flush=True)
    print('[didi-checkpoint] No training, sampling, or tracker writes were performed.', flush=True)


if __name__ == '__main__':
    main()
