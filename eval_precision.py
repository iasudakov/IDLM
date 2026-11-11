"""Scoped FP32 prediction normalization and paired evaluation randomness.

The backbone keeps its configured dtype. Cast raw logits BEFORE SUBS masking
and logsumexp, rather than casting already rounded BF16 log probabilities.
"""

from contextlib import contextmanager
import random

import numpy as np
import torch


def capture_rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _restore_rng_state(state):
    python, numpy, cpu, cuda = state
    random.setstate(python)
    np.random.set_state(numpy)
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


@contextmanager
def replay_rng_state(state):
    """Replay a sweep's starting RNGs without advancing the caller's RNGs."""
    current = capture_rng_state()
    try:
        _restore_rng_state(state)
        yield
    finally:
        _restore_rng_state(current)


@contextmanager
def fp32_prediction_normalization(model):
    """Temporarily wrap the normalizer; preserve weights, forward and modes.

    Used only by the additional no-grad benchmark sweep, never training/NLL.
    Restore the exact previous instance/class lookup even after an exception.
    """
    original = model._process_model_output
    had_override = '_process_model_output' in vars(model)
    previous = vars(model).get('_process_model_output')

    def normalize(model_output, xt, sigma):
        with torch.autocast(device_type=model_output.device.type, enabled=False):
            return original(model_output=model_output.float(), xt=xt, sigma=sigma)

    model._process_model_output = normalize
    try:
        yield
    finally:
        if had_override:
            model._process_model_output = previous
        else:
            del model._process_model_output
