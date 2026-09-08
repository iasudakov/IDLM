"""Opt-in float64 *sampling arithmetic* diagnostic for the pinned inference code.

This is not the untouched authors' sampler or full-float64 neural inference.
Network forwards, the guidance-gradient path, timesteps, and final denoising
retain their original precision. Only post-network sampling arithmetic changes.
"""

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from .standalone import verify_source


REPLACEMENTS = (
    ('log_p_x0 = log_p_x0 + h * grad_clipped.detach()',
     "log_p_x0 = _didi_double(log_p_x0, 'tilt_logits') + h * "
     "_didi_double(grad_clipped.detach(), 'tilt_gradient')"),
    ('p_x0 = log_p_x0.softmax(dim=-1)',
     "p_x0 = _didi_double(log_p_x0, 'probability_logits').softmax(dim=-1)"),
    ('_, alpha_t = self.noise(t)',
     "_, alpha_t = self.noise(t)\nalpha_t = _didi_double(alpha_t, 'alpha_t')"),
    ('_, alpha_s = self.noise(t - dt)',
     "_, alpha_s = self.noise(t - dt)\nalpha_s = _didi_double(alpha_s, 'alpha_s')"),
    ('reward_flat = discriminator_scores_flat.sum(dim=[1, 2])',
     "reward_flat = _didi_double(discriminator_scores_flat, 'rerank_scores').sum(dim=[1, 2])"),
)


def sample_categorical_reusing_buffer(categorical_probs):
    """Original categorical equation/RNG layout, with one reusable workspace.

    No chunking, extra draws, RNG reseeding, or reciprocal approximation. The
    original float64 B=20, M=4 vocabulary-sized temporaries exceed an 80GB GPU;
    out/in-place operations preserve the same elementwise operation order.
    Inputs (including zero-stride expanded views) are never modified.
    """
    import torch

    noise = torch.rand_like(categorical_probs)
    noise.add_(1e-10).log_()
    torch.sub(1e-10, noise, out=noise)
    torch.div(categorical_probs, noise, out=noise)
    return noise.argmax(dim=-1)


def _patched_function():
    directory, manifest = verify_source('inference')
    path = directory / 'algo.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    owner, = [node for node in tree.body
              if isinstance(node, ast.ClassDef) and node.name == 'DiDiInstruct']
    original, = [node for node in owner.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'reward_guided_ancestral_step']
    function = copy.deepcopy(original)
    for before, after in REPLACEMENTS:
        target, = ast.parse(before).body
        matches = []
        for node in ast.walk(function):
            for _, statements in ast.iter_fields(node):
                if not isinstance(statements, list):
                    continue
                for index, statement in enumerate(statements):
                    if (isinstance(statement, ast.stmt) and
                            ast.dump(statement, include_attributes=False) ==
                            ast.dump(target, include_attributes=False)):
                        matches.append((statements, index))
        if len(matches) != 1:
            raise RuntimeError(f'Precision diagnostic expected exactly one match: {before}')
        statements, index = matches[0]
        statements[index:index + 1] = ast.parse(after).body
    ast.fix_missing_locations(function)
    return function, original, path, manifest


def precision_metadata():
    function, original, path, manifest = _patched_function()
    return {
        'diagnostic': 'float64_sampling_arithmetic',
        'not_full_float64_model_or_evaluation': True,
        'upstream_revision': manifest['commit'],
        'upstream_algo_sha256': manifest['files']['algo.py'],
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method': function.name,
        'replacements': [{'before': before, 'after': after} for before, after in REPLACEMENTS],
        'original_method': ast.unparse(original),
        'modified_method': ast.unparse(function),
        'categorical': 'Original rand_like shape and equation; reuse random buffer in place.',
        'unchanged': ['checkpoint', 'network forwards', 'guidance gradient calculation',
                      'timestep grid and noise inputs', 'guidance/candidate schedule',
                      'final denoising', 'generation batch size', 'perplexity scorer'],
        'rng_note': 'Same seed, but dtype changes alter draws; samples are not paired-identical.',
    }


def load_precision_sampler(upstream_algo, observer=None):
    """Return one unbound diagnostic method; never mutate the supplied module."""
    function, original, path, _ = _patched_function()
    namespace = dict(vars(upstream_algo))
    required = {'torch', 'F', 'trainer_base'}
    if not required.issubset(namespace):
        raise ValueError(f'Missing inference globals: {sorted(required - namespace.keys())}')
    torch = namespace['torch']

    def promote(tensor, stage):
        if observer is not None:
            observer(stage + '_input', tensor)
        result = tensor.to(torch.float64)
        if observer is not None:
            observer(stage + '_output', result)
        return result

    def categorical(probabilities):
        if probabilities.dtype != torch.float64:
            raise RuntimeError('Diagnostic guided categorical probabilities must be float64.')
        if observer is not None:
            observer('categorical', probabilities)
        return sample_categorical_reusing_buffer(probabilities)

    namespace['_didi_double'] = promote
    namespace['trainer_base'] = SimpleNamespace(sample_categorical=categorical)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path) + ':float64-diagnostic', 'exec'), namespace)
    result = namespace[function.name]
    result.__didi_original_source__ = ast.unparse(original)
    result.__didi_precision_source__ = ast.unparse(function)
    return result, precision_metadata()
