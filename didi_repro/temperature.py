"""Temporary denoiser temperature for evaluation-only DiDi Pareto sweeps.

The released sampler, discriminator scores, and noise schedule are untouched.
Temperature is applied to p(x0 | xt) before either ancestral sampling or the
guided sampler's gradient tilting/candidate reranking, including final cleanup.
"""

from contextlib import contextmanager
from functools import wraps
import math

import torch


def _temper(log_probabilities, tokens, mask_index, temperature):
    """Normalize masked-token predictions without reviving SUBS exclusions."""
    work = log_probabilities
    limits = torch.finfo(work.dtype)
    if temperature < limits.tiny or temperature > limits.max:
        # Avoid rounding a finite positive Python float to zero/infinity in
        # lower precision. Ordinary sweep temperatures keep the original dtype.
        work = work.double()
    shifted = work - work.amax(dim=-1, keepdim=True)
    shifted[..., mask_index] = -torch.inf
    tempered = (shifted / temperature).log_softmax(dim=-1)
    tempered = tempered.to(log_probabilities.dtype)
    # Upstream represents forbidden tokens using a finite negative sentinel.
    # Keep observed-token rows byte-for-byte; exclude MASK explicitly on masked
    # rows, so large temperatures cannot accidentally restore its probability.
    return torch.where((tokens != mask_index).unsqueeze(-1),
                       log_probabilities, tempered)


@contextmanager
def sampling_temperature(model, temperature):
    """Set an absolute, temporary sampling temperature; restore on every exit.

T=1 is an exact no-op outside a nested temperature override. Nested contexts
override (rather than multiply) the outer temperature. This context changes
neither RNG state, module modes, discriminator/noise, nor parameter gradients.
It is not a training option and refuses entry during a DiDi training step.
"""
    try:
        temperature = float(temperature)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('Sampling temperature must be finite and greater than zero.') from error
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Sampling temperature must be finite and greater than zero.')
    if getattr(model, '_didi_in_training_step', False):
        raise RuntimeError('Sampling temperature overrides are evaluation-only, not training options.')

    current_forward = model.forward
    base_forward = getattr(current_forward, '__dict__', {}).get(
        '__didi_temperature_base_forward__', current_forward)
    if temperature == 1.0 and base_forward is current_forward:
        yield model
        return

    had_instance_forward = 'forward' in model.__dict__
    instance_forward = model.__dict__.get('forward')
    if temperature == 1.0:
        replacement = base_forward
    else:
        @wraps(base_forward)
        def replacement(*args, **kwargs):
            if getattr(model, '_didi_in_training_step', False):
                raise RuntimeError('Sampling temperature overrides are evaluation-only, not training options.')
            tokens = kwargs.get('xt', kwargs.get('x', args[0] if args else None))
            if tokens is None:
                raise TypeError('DiDi temperature sampling requires the input token sequence.')
            return _temper(base_forward(*args, **kwargs), tokens,
                           model.mask_index, temperature)

        replacement.__didi_temperature_base_forward__ = base_forward

    model.forward = replacement
    try:
        yield model
    finally:
        if had_instance_forward:
            model.forward = instance_forward
        else:
            delattr(model, 'forward')
