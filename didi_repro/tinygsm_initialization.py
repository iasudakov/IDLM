"""Select raw or EMA native TinyGSM MDLM weights for fresh initialization.

The reference EMA saves an unnamed list in backbone.parameters() followed by
noise.parameters() order, excluding frozen parameters. The supported native
DiT has all backbone parameters trainable, no aliases, parameter-free noise,
and one rotary buffer which is NOT an EMA entry. Do not zip shadows with a
state_dict or filter the frozen destination student_ema by requires_grad.
"""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch


def student_ema_source(config):
    source = config.get('didi_init', {}).get('tinygsm_student_ema', 'checkpoint')
    if source not in ('checkpoint', 'raw'):
        raise ValueError('didi_init.tinygsm_student_ema must be checkpoint or raw.')
    return source


def training_weights_source(config):
    source = config.get('didi_init', {}).get('tinygsm_weights', 'raw')
    if source not in ('raw', 'ema'):
        raise ValueError('didi_init.tinygsm_weights must be raw or ema.')
    if source == 'ema' and student_ema_source(config) != 'checkpoint':
        raise ValueError('didi_init.tinygsm_weights=ema initializes every model copy '
                         'from checkpoint EMA; tinygsm_student_ema must be checkpoint, '
                         'not raw.')
    return source


def _tensor(value, expected, label):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type == 'meta' or value.is_complex()):
        raise ValueError(f'{label} must be a materialized, dense real tensor.')
    if value.shape != expected.shape:
        raise ValueError(f'{label} shape {tuple(value.shape)} != {tuple(expected.shape)}.')
    if value.is_floating_point() != expected.is_floating_point():
        raise ValueError(f'{label} has incompatible dtype {value.dtype}.')
    if value.is_floating_point():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{label} contains nonfinite values.')
        # A finite FP64 checkpoint can still overflow a FP32/FP16 destination.
        if value.dtype != expected.dtype and not bool(torch.isfinite(value.to(dtype=expected.dtype)).all()):
            raise ValueError(f'{label} overflows the destination dtype {expected.dtype}.')


def validate_ema_destinations(model, state):
    """Check every destination before applying the shared EMA initializer.

    The discriminator has a newly initialized classification head in place of
    the MDLM vocabulary head. Validate/copy only its shared backbone tensors,
    including frozen parameters and non-EMA buffers.
    """
    for label in ('backbone', 'teacher', 'student_ema', 'discriminator'):
        expected = getattr(model, label).state_dict()
        selected = state
        if label == 'discriminator':
            expected = {key: value for key, value in expected.items()
                        if not key.startswith('output_layer.')}
            selected = {key: value for key, value in state.items()
                        if not key.startswith('output_layer.')}
        if set(selected) != set(expected):
            raise ValueError(f'Native EMA/{label} state mismatch: '
                             f'missing={sorted(set(expected) - set(selected))}, '
                             f'unexpected={sorted(set(selected) - set(expected))}.')
        for name, value in selected.items():
            _tensor(value, expected[name], f'Native EMA/{label}.{name}')


def prepare_student_ema(model, checkpoint):
    """Return a validated full EMA state and JSON-safe provenance; never mutate.

    Native EMA lists carry no parameter names. This validates the pinned
    parameter layout and shapes, but cannot detect a corrupt permutation of
    equal-shaped shadows. Only trusted native MDLM checkpoints are supported.
    No source EMA decay/update counter is applied to DiDi's training schedule.
    """
    source = student_ema_source(model.config)
    training_source = training_weights_source(model.config)
    state = checkpoint.get('state_dict')
    if not isinstance(state, Mapping):
        raise ValueError('Native TinyGSM checkpoint needs a state_dict mapping.')
    raw = {key[len('backbone.'):]: value for key, value in state.items()
           if isinstance(key, str) and key.startswith('backbone.')}
    expected = model.student_ema.state_dict()
    if set(raw) != set(expected):
        raise ValueError('Native TinyGSM backbone state mismatch: '
                         f'missing={sorted(set(expected) - set(raw))}, '
                         f'unexpected={sorted(set(raw) - set(expected))}.')
    for name, value in raw.items():
        _tensor(value, expected[name], f'Native backbone.{name}')
    metadata = {
        'kind': 'native_tinygsm_initialization',
        'training_weights': training_source,
        'backbone': 'raw backbone.*',
        'teacher': 'raw backbone.*; unchanged',
        'discriminator': 'raw backbone trunk; newly initialized head unchanged',
        'student_ema': {'source': 'raw_backbone'},
        'ema_update': 'released DiDi algo.ema_beta; no checkpoint EMA schedule imported',
    }
    if source == 'raw':
        return raw, metadata

    prefix = ('Native TinyGSM checkpoint EMA is incompatible. '
              'Use a complete native MDLM checkpoint, or explicitly select '
              'didi_init.tinygsm_weights=raw and tinygsm_student_ema=raw '
              'for raw-only initialization. ')
    ema = checkpoint.get('ema')
    if not isinstance(ema, Mapping):
        raise ValueError(prefix + "Missing checkpoint['ema'] mapping.")
    shadows = ema.get('shadow_params')
    if not isinstance(shadows, (list, tuple)):
        raise ValueError(prefix + 'ema.shadow_params must be a list or tuple.')
    parameters = list(model.backbone.named_parameters())
    if (len(list(model.backbone.named_parameters(remove_duplicate=False))) != len(parameters)
            or any(not parameter.requires_grad for _, parameter in parameters)
            or any(True for _ in model.noise.parameters())):
        raise ValueError(prefix + 'Expected untied, fully trainable native backbone and parameter-free noise.')
    names = [name for name, _ in parameters]
    parameter_names = set(names)
    if [name for name in raw if name in parameter_names] != names:
        raise ValueError(prefix + 'Native backbone parameter order differs from the pinned EMA layout.')
    if len(shadows) != len(parameters):
        raise ValueError(prefix + f'Expected {len(parameters)} EMA parameters, got {len(shadows)}; '
                         'buffers must not occur in the EMA list.')
    decay, updates = ema.get('decay'), ema.get('num_updates')
    if (isinstance(decay, bool) or not isinstance(decay, Real)
            or not math.isfinite(decay) or not 0 <= decay <= 1):
        raise ValueError(prefix + 'ema.decay must be finite and in [0, 1].')
    if 'num_updates' not in ema or (updates is not None and (
            isinstance(updates, bool) or not isinstance(updates, Integral) or updates < 0)):
        raise ValueError(prefix + 'ema.num_updates must be a nonnegative integer or null.')
    overlay, different = dict(raw), 0
    for (name, parameter), shadow in zip(parameters, shadows):
        _tensor(shadow, parameter, f'Native ema.shadow_params[{name}]')
        different += int(not torch.equal(raw[name], shadow))
        overlay[name] = shadow
    metadata['student_ema'] = {
        'source': 'checkpoint_ema',
        'source_field': 'ema.shadow_params',
        'parameter_order': 'native backbone.named_parameters(); buffers excluded',
        'parameter_count': len(parameters),
        'raw_different_parameter_count': different,
        'buffers': 'raw backbone buffers; not EMA parameters',
        'checkpoint_decay': float(decay),
        'checkpoint_num_updates': None if updates is None else int(updates),
    }
    if training_source == 'ema':
        validate_ema_destinations(model, overlay)
        metadata.update({
            'backbone': 'checkpoint EMA parameters; raw backbone buffers',
            'teacher': 'checkpoint EMA parameters; raw backbone buffers',
            'discriminator': 'checkpoint EMA trunk parameters; raw backbone buffers; '
                             'newly initialized head unchanged',
        })
    return overlay, metadata
