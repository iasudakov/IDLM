"""Load the pinned authors' inference methods without importing another runtime.

The methods retain the inference branch's EMA calls, timestep conditioning,
phase-local guidance schedule, candidate selection, and terminal denoising.
Only the input-gradient calculation is repaired: autograd.grad returns the same
gradient as backward() without accumulating discriminator parameter gradients.
Mode restoration, evaluation-weight routing, and temperature belong to the
calling runtime, not to this source-preserving loader.
"""

import ast
import copy

from .standalone import verify_source


METHOD_NAMES = ('reward_guided_ancestral_step', 'generate_samples')
SAMPLING_REPAIRS = (
    ('reward.backward()\ngrad = log_p_x0_for_grad.grad',
     'grad = torch.autograd.grad(reward, log_p_x0_for_grad, only_inputs=True)[0]'),
)


def _same_ast(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _repair_input_gradient(function):
    """Replace one exact adjacent statement pair; reject an unexpected source."""
    before, after = SAMPLING_REPAIRS[0]
    old_statements = ast.parse(before).body
    replacement = ast.parse(after).body[0]
    matches = []
    for node in ast.walk(function):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index in range(len(value) - 1):
                if (isinstance(value[index], ast.stmt)
                        and isinstance(value[index + 1], ast.stmt)
                        and _same_ast(value[index], old_statements[0])
                        and _same_ast(value[index + 1], old_statements[1])):
                    matches.append((value, index))
    if len(matches) != 1:
        raise RuntimeError('Pinned DiDi inference input-gradient repair no longer matches once.')
    statements, index = matches[0]
    replacement = ast.copy_location(replacement, statements[index + 1])
    statements[index:index + 2] = [replacement]


def load_inference_sampler(upstream_algo):
    """Return ``(unbound_functions, metadata)`` in the main algorithm namespace.

``upstream_algo`` is the already loaded original/main ``algo`` module. Its
globals are copied, not mutated, so both samplers can coexist in one process.
Every call verifies the complete separately pinned inference source tree.
"""
    directory, manifest = verify_source('inference')
    path = directory / 'algo.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    owners = [node for node in tree.body
              if isinstance(node, ast.ClassDef) and node.name == 'DiDiInstruct']
    if len(owners) != 1:
        raise RuntimeError('Pinned DiDi inference class no longer matches once.')
    functions = []
    originals = {}
    for name in METHOD_NAMES:
        candidates = [node for node in owners[0].body
                      if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(candidates) != 1:
            raise RuntimeError(f'Pinned DiDi inference method no longer matches once: {name}')
        function = copy.deepcopy(candidates[0])
        originals[name] = ast.unparse(function)
        if name == 'reward_guided_ancestral_step':
            _repair_input_gradient(function)
        functions.append(function)

    namespace = dict(vars(upstream_algo))
    required = {'torch', 'F', 'trainer_base', 'MDLM'}
    if not required.issubset(namespace):
        raise ValueError(f'DiDi main algorithm namespace is missing: {sorted(required - namespace.keys())}')
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    result = {}
    for node in functions:
        function = namespace[node.name]
        function.__didi_original_source__ = originals[node.name]
        function.__didi_repaired_source__ = ast.unparse(node)
        result[node.name] = function
    metadata = {
        'source': 'inference',
        'repository': manifest['repository'],
        'commit': manifest['commit'],
        'algo_sha256': manifest['files']['algo.py'],
        'methods': list(METHOD_NAMES),
        'repairs': [{'before': before, 'after': after} for before, after in SAMPLING_REPAIRS],
    }
    return result, metadata
