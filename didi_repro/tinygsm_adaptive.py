"""Reference TinyGSM threshold decoding, independent of DiDi guidance.

Ports ``Diffusion._adaptive_generate_samples``, ``_alpha_to_t``,
``AbsorbingState._propose_tokens`` / ``_confidence_commit``, and
``confidence_sampling.select_threshold`` from tinygsm-upstream-repro.
The threshold is strict (confidence > threshold), proposals use the original
categorical race, and confidence always comes from untempered SUBS probabilities.

The complete batch is forwarded every iteration, including already-finished
rows, preserving the reference's seeded stochastic draws. ``nfe_per_row`` is
the reference's active-until-completion cost, NOT physical row forwards;
``nfe`` counts actual full-batch forwards, including any capped-loop flush.
"""

from contextlib import contextmanager
import math
import numbers

import torch


DECODER_PROTOCOL = 'tinygsm-reference-low-confidence-dynamic-v1'


def _number(value, name, minimum, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number, not a boolean.')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite number.') from error
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        interval = f'[{minimum}, {maximum}]' if maximum is not None else f'>= {minimum}'
        raise ValueError(f'{name} must be finite and in {interval}.')
    return value


def _sample_categorical(probabilities):
    # Bit-identical to the reference trainer_base.sample_categorical, including
    # drawing for fixed positions and completed rows.
    gumbel_norm = 1e-10 - (torch.rand_like(probabilities) + 1e-10).log()
    return (probabilities / gumbel_norm).argmax(dim=-1)


def _propose_tokens(probabilities, temperature):
    if temperature == 0:
        tokens = probabilities.argmax(dim=-1)
    elif temperature == 1:
        tokens = _sample_categorical(probabilities)
    else:
        sampling_probabilities = torch.softmax(
            probabilities.clamp_min(1e-30).log() / temperature, dim=-1)
        if not bool(torch.isfinite(sampling_probabilities).all()):
            raise RuntimeError('Adaptive temperature produced nonfinite sampling probabilities; '
                               'use temperature=0 for greedy decoding or a representable positive value.')
        tokens = _sample_categorical(sampling_probabilities)
    confidence = probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, confidence.float()


def _select_threshold(confidence, candidates, threshold):
    # Exact reference strictness and tie-breaking (first argmax position).
    transfer = (confidence > threshold) & candidates
    stalled = candidates.any(dim=-1) & ~transfer.any(dim=-1)
    if bool(stalled.any()):
        forced = torch.zeros_like(transfer)
        forced.scatter_(-1, confidence.argmax(dim=-1, keepdim=True), True)
        transfer = torch.where(stalled.unsqueeze(-1), forced, transfer)
    return transfer


@contextmanager
def _evaluation(model):
    if getattr(model, '_didi_in_training_step', False):
        raise RuntimeError('TinyGSM adaptive decoding is evaluation-only.')
    if getattr(model, '_didi_inference_sampling_depth', 0):
        raise RuntimeError('Adaptive decoding cannot run inside the guided inference sampler.')
    if getattr(model.forward, '__didi_temperature_base_forward__', None) is not None:
        raise RuntimeError('Pass adaptive temperature directly; do not wrap it in sampling_temperature.')
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        # This routes standard SUBS forward to the configured raw student or
        # student EMA. It does NOT invoke the inference release's raw-logit
        # sampler, nor any discriminator. No parameters are swapped or copied.
        with model._didi_evaluation_context():
            yield
    finally:
        for module, training in modes:
            module.training = training


def _initial_state(model, prefix_tokens, prefix_lengths):
    if prefix_tokens is None or prefix_lengths is None:
        raise ValueError('Adaptive TinyGSM decoding needs prefix_tokens and prefix_lengths.')
    prefixes = torch.as_tensor(prefix_tokens, device=model.device).clone()
    lengths = torch.as_tensor(prefix_lengths, device=model.device).clone()
    if prefixes.ndim != 2 or prefixes.shape[0] < 1:
        raise ValueError('prefix_tokens must have a nonempty [batch, prefix_width] shape.')
    if lengths.ndim != 1 or lengths.shape[0] != prefixes.shape[0]:
        raise ValueError('prefix_lengths must have shape [batch].')
    if (prefixes.is_floating_point() or prefixes.is_complex() or prefixes.dtype == torch.bool
            or lengths.is_floating_point() or lengths.is_complex() or lengths.dtype == torch.bool):
        raise ValueError('Prefix tokens and lengths must be integers.')
    width = int(model.num_tokens)
    if (bool((lengths <= 0).any()) or bool((lengths >= width).any())
            or bool((lengths > prefixes.shape[1]).any())):
        raise ValueError('Every prefix must fit and leave at least one position to generate.')
    fixed_positions = torch.arange(width, device=model.device)[None, :] < lengths[:, None]
    x = torch.full((prefixes.shape[0], width), model.mask_index,
                   dtype=torch.long, device=model.device)
    supplied_width = min(width, prefixes.shape[1])
    x[:, :supplied_width] = prefixes[:, :supplied_width].long()
    x.masked_fill_(~fixed_positions, model.mask_index)
    fixed_ids = x[fixed_positions]
    if bool(((fixed_ids < 0) | (fixed_ids >= model.vocab_size)).any()):
        raise ValueError('Prefix token IDs are outside the model vocabulary.')
    if bool((fixed_ids == model.mask_index).any()):
        raise ValueError('A fixed prefix cannot contain MASK tokens.')
    return x, fixed_positions


def generate_adaptive(model, prefix_tokens, prefix_lengths, threshold,
                      temperature=0.0, return_metadata=True, max_steps=None):
    """Decode complete suffixes with the reference confidence-threshold rule.

    ``max_steps=None`` uses ``model.length`` (one-commit-per-step worst case).
    An explicit positive integer caps threshold iterations, not total NFE:
    if masks remain, one extra forward commits every remaining position at the
    requested temperature, exactly like the reference final flush.

    EOS/PAD do not stop decoding. No gold answer boundary or suffix token is
    used. RNG consumption is intentionally not restored here; the outer sweep
    owns reproducible seeding and preservation of training RNG state.

    Returns ``(tokens, metadata)`` by default, or tokens only when requested.
    Unlike the reference's unchecked edge case, selected MASK proposals raise
    an actionable error. This preserves ordinary reference draws while avoiding
    silent stalled decoding or falsely completed outputs at extreme temperature.
    """
    threshold = _number(threshold, 'threshold', 0.0, 1.0)
    temperature = _number(temperature, 'temperature', 0.0)
    if max_steps is None:
        max_steps = int(model.num_tokens)
    elif isinstance(max_steps, bool) or not isinstance(max_steps, numbers.Integral) or max_steps <= 0:
        raise ValueError('max_steps must be a positive integer or None.')
    max_steps = int(max_steps)
    if getattr(model, 'parameterization', None) != 'subs':
        raise ValueError('The reference TinyGSM adaptive decoder requires SUBS parameterization.')
    weights = getattr(model, '_didi_evaluation_weights', None)
    if weights not in ('backbone', 'student_ema'):
        raise ValueError('Adaptive decoding requires explicit DiDi backbone or student_ema evaluation weights.')

    with torch.inference_mode(False), torch.no_grad(), _evaluation(model):
        x, fixed_positions = _initial_state(model, prefix_tokens, prefix_lengths)
        fixed_tokens = x.clone()
        nfe_per_row = torch.zeros(x.shape[0], device=model.device)
        nfe = steps = fallback_nfe = 0
        eps = 1e-5

        def update(tokens, t, flush=False):
            alpha_t = model.noise(t)[1]
            sigma = model._sigma_from_alphat(alpha_t)
            probabilities = model.forward(tokens, sigma).exp()
            if probabilities.shape != (*tokens.shape, model.vocab_size):
                raise ValueError('Adaptive decoder expected [B,L,vocab_size] SUBS probabilities.')
            if not bool(torch.isfinite(probabilities).all()):
                raise RuntimeError('Adaptive decoder received nonfinite denoiser probabilities.')
            if not bool((probabilities.sum(dim=-1) > 0).all()):
                raise RuntimeError('Adaptive decoder received an empty denoiser distribution.')
            proposals, confidence = _propose_tokens(probabilities, temperature)
            candidates = tokens == model.mask_index
            confidence = torch.where(candidates, confidence,
                                     torch.full_like(confidence, -float('inf')))
            transfer = candidates if flush else _select_threshold(confidence, candidates, threshold)
            if bool(((proposals == model.mask_index) & transfer).any()):
                raise RuntimeError('Adaptive decoder selected MASK as a committed token. '
                                   'The reference temperature floor can restore forbidden-token '
                                   'mass at extreme temperatures; lower temperature (0 is greedy). '
                                   'Refusing stalled or incomplete benchmark output.')
            result = torch.where(transfer, proposals, tokens)
            return torch.where(fixed_positions, fixed_tokens, result)

        for _ in range(max_steps):
            unfinished = (x == model.mask_index).any(dim=-1)
            if not bool(unfinished.any()):
                break
            nfe_per_row += unfinished.to(nfe_per_row.dtype)
            # Keep the reference operation order, full-width denominator, and
            # inversion of the linear alpha schedule; prefix is already known.
            alpha_t = 1 - (x == model.mask_index).to(torch.float32).mean(dim=-1, keepdim=True)
            t = ((1 - alpha_t) / (1 - model.noise.eps)).clamp(eps, 1.0)
            x = update(x, t)
            nfe += 1
            steps += 1

        stragglers = (x == model.mask_index).any(dim=-1)
        if bool(stragglers.any()):
            x = update(x, eps * torch.ones(x.shape[0], 1, device=model.device), flush=True)
            nfe += 1
            fallback_nfe = 1
            nfe_per_row += stragglers.to(nfe_per_row.dtype)
        if bool((x == model.mask_index).any()):
            raise RuntimeError('Adaptive decoding did not complete; refusing partial benchmark output.')
        if not return_metadata:
            return x
        return x, {
            'nfe': nfe,
            'steps': steps,
            'nfe_per_row': nfe_per_row,
            'physical_row_forwards': nfe * x.shape[0],
            'fallback_nfe': fallback_nfe,
            'max_steps': max_steps,
            'complete': True,
            'threshold': threshold,
            'temperature': temperature,
            'weights': weights,
            'remasking_strategy': 'low_confidence_dynamic',
            'decoder': DECODER_PROTOCOL,
            'confidence_comparison': 'strict-greater-than',
            'proposal_guard': 'raise-on-selected-MASK',
        }
