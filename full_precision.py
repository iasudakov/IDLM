"""Opt-in IEEE FP32 execution settings, applied before model construction."""

import os

import torch


def validate_full_fp32_config(config):
    """Validate the full-FP32 request without changing global runtime settings."""
    requested = config.get('model', {}).get('force_fp32', False)
    if type(requested) is not bool:
        raise ValueError('model.force_fp32 must be a boolean.')
    if not requested:
        return False

    precision = config.get('trainer', {}).get('precision')
    if not ((type(precision) is int and precision == 32)
            or (isinstance(precision, str) and precision in {'32', '32-true'})):
        raise ValueError('model.force_fp32=True requires trainer.precision=32-true '
                         '(or 32); mixed precision would change the training arithmetic.')
    if config.get('training', {}).get('loss_precision', 'float32') != 'float32':
        raise ValueError('model.force_fp32=True requires training.loss_precision=float32.')
    if os.environ.get('NVIDIA_TF32_OVERRIDE', '').strip() == '1':
        raise ValueError('Full FP32 conflicts with NVIDIA_TF32_OVERRIDE=1. '
                         'Unset it or set NVIDIA_TF32_OVERRIDE=0 before launching Python.')
    return True


def configure_full_fp32(config):
    """Enable full precision for this process; leave existing recipes untouched."""
    if not validate_full_fp32_config(config):
        return False

    # PyTorch 2.9+ forbids mixing these newer controls with allow_tf32. Explicit
    # operator settings also override any precision choices made by a caller.
    if hasattr(torch.backends, 'fp32_precision'):
        torch.backends.fp32_precision = 'ieee'
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.fp32_precision = 'ieee'
        for name in ('conv', 'rnn'):
            backend = getattr(torch.backends.cudnn, name, None)
            if backend is not None and hasattr(backend, 'fp32_precision'):
                backend.fp32_precision = 'ieee'
    else:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print('[precision] Full FP32: IEEE matmul and TF32 disabled.', flush=True)
    return True
