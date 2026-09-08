"""Temporary denoiser temperature for evaluation-only DiDi Pareto sweeps.

The released sampler, discriminator scores, and noise schedule are untouched.
Temperature is applied to p(x0 | xt) before either ancestral sampling or the
guided sampler's gradient tilting/candidate reranking, including final cleanup.
The inference release reads raw student-EMA logits directly, so that path uses
logits/T without introducing the main release's SUBS exclusions or sigma change.
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


def _temper_raw_logits(logits, temperature):
    """Scale inference logits without changing their original support.

    Ordinary sweep temperatures are exactly raw logits/T in their native dtype.
    Extreme Python floats use float64 arithmetic. If a row cannot represent the
    scaled values in the original dtype, subtracting its maximum preserves its
    softmax distribution; saturate negative differences to finite dtype limits.
    Existing -inf exclusions remain excluded, but no new MASK exclusion is added.
    """
    limits = torch.finfo(logits.dtype)
    if limits.eps <= temperature <= 1 / limits.eps:
        return logits / temperature
    work = logits.double()
    scaled = work / temperature
    finite = torch.isfinite(work)
    overflow = (finite & ((scaled.abs() > limits.max) | ~torch.isfinite(scaled))).any(
        dim=-1, keepdim=True)
    centered = ((work - work.amax(dim=-1, keepdim=True)) / temperature).clamp(
        min=limits.min, max=0)
    centered = torch.where(torch.isneginf(work), work, centered)
    return torch.where(overflow, centered, scaled).to(logits.dtype)


def _base_forward(forward):
    return getattr(forward, '__dict__', {}).get('__didi_temperature_base_forward__', forward)


def _training_guard(model):
    if getattr(model, '_didi_in_training_step', False):
        raise RuntimeError('Sampling temperature overrides are evaluation-only, not training options.')


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
    _training_guard(model)

    current_forward = model.forward
    base_forward = _base_forward(current_forward)
    inference = getattr(model, '_didi_sampler_source', 'main') == 'inference'
    ema = model.student_ema if inference else None
    current_ema_forward = ema.forward if inference else None
    base_ema_forward = _base_forward(current_ema_forward) if inference else None
    if (temperature == 1.0 and base_forward is current_forward
            and (not inference or base_ema_forward is current_ema_forward)):
        yield model
        return

    # Both wrappers share this context-local depth. model.forward already yields
    # processed log probabilities, so any EMA invocation it makes must not also
    # scale raw logits. Nested contexts get independent depths and absolute T.
    state = {'processed_depth': 0}
    if temperature == 1.0:
        replacement = base_forward
        ema_replacement = base_ema_forward
    else:
        @wraps(base_forward)
        def replacement(*args, **kwargs):
            _training_guard(model)
            tokens = kwargs.get('xt', kwargs.get('x', args[0] if args else None))
            if tokens is None:
                raise TypeError('DiDi temperature sampling requires the input token sequence.')
            state['processed_depth'] += 1
            try:
                output = base_forward(*args, **kwargs)
            finally:
                state['processed_depth'] -= 1
            return _temper(output, tokens, model.mask_index, temperature)

        replacement.__didi_temperature_base_forward__ = base_forward
        if inference:
            @wraps(base_ema_forward)
            def ema_replacement(*args, **kwargs):
                _training_guard(model)
                logits = base_ema_forward(*args, **kwargs)
                if state['processed_depth']:
                    return logits
                return _temper_raw_logits(logits, temperature)

            ema_replacement.__didi_temperature_base_forward__ = base_ema_forward

    replacements = [(model, replacement)]
    if inference:
        replacements.append((ema, ema_replacement))
    changes = []
    try:
        for module, forward in replacements:
            changes.append((module, 'forward' in module.__dict__, module.__dict__.get('forward')))
            module.forward = forward
        yield model
    finally:
        for module, had_instance_forward, instance_forward in reversed(changes):
            if had_instance_forward:
                module.forward = instance_forward
            else:
                delattr(module, 'forward')
