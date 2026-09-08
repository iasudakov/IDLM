#!/usr/bin/env python3
"""Validate and summarize the preregistered five-seed DiDi guidance control.

Only reads existing experiment artifacts. No model imports, downloads, GPU,
tracking, pooled perplexities, significance tests, or incomplete-cell summaries.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from didi_repro.standalone import INFERENCE_REVISION, verify_source


CHECKPOINT_SHA256 = '31db40c7d9db3d4c4de721b54d4f0a45d88735df0dc896d700edb59ac083ebbe'
SCORER_REVISION = '32b71b12589c2f8d625668d2335a01cac3249519'
SEEDS = [1, 2, 3, 4, 5]
CONDITIONS = {
    'off': {'guidance_scale_start': 0.0, 'guidance_scale_end': 0.0,
            'num_candidates': 1, 'rerank_steps_ratio': 0.5},
    'on': {'guidance_scale_start': 30.0, 'guidance_scale_end': 40.0,
           'num_candidates': 4, 'rerank_steps_ratio': 0.5},
}
ARTIFACTS = ('samples.json', 'all40.json', 'authors_eval_environment.json', '.hydra/config.yaml')
ALLOWED_CONFIG_DIFFERENCES = (
    'seed', 'eval.generated_samples_path', 'checkpointing.save_dir',
    'algo.guidance_scale_start', 'algo.guidance_scale_end', 'algo.num_candidates',
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'{path}: duplicate JSON key {key}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f'{path}: nonfinite JSON constant {value}')

    with path.open() as stream:
        result = json.load(stream, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return result


def field(mapping, dotted, label):
    current = mapping
    for name in dotted.split('.'):
        if not isinstance(current, dict) or name not in current:
            raise ValueError(f'{label}: missing {dotted}')
        current = current[name]
    return current


def expect(actual, expected, label):
    equal = actual == expected
    if isinstance(expected, bool):
        equal = equal and isinstance(actual, bool)
    elif isinstance(expected, (int, float)):
        equal = equal and isinstance(actual, (int, float)) and not isinstance(actual, bool)
    if not equal:
        raise ValueError(f'{label}: expected {expected!r}, got {actual!r}')


def check(mapping, expected, label):
    for path, value in expected.items():
        expect(field(mapping, path, label), value, f'{label}.{path}')


def finite_number(value, label, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f'{label}: expected a finite {"positive " if positive else ""}number')
    return float(value)


def require_digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{label}: expected a SHA256 digest')


def comparable_config(config):
    result = copy.deepcopy(config)
    for path in ALLOWED_CONFIG_DIFFERENCES:
        parts = path.split('.')
        owner = result
        for name in parts[:-1]:
            owner = owner[name]
        del owner[parts[-1]]
    return result


def require_same(actual, expected, label):
    if actual != expected:
        # Avoid dumping full installed-package lists or configuration contents.
        raise ValueError(f'{label}: differs between cells outside the declared control changes')


def validate_protocol(protocol):
    check(protocol, {
        'checkpoint_sha256': CHECKPOINT_SHA256, 'upstream_source': 'inference',
        'upstream_revision': INFERENCE_REVISION, 'seeds': SEEDS, 'steps': 8,
        'samples_per_seed_per_condition': 40, 'generation_batch_size': 20,
        'generation_batches': 2, 'length': 1024, 'sampler': 'guided', 'cleanup': 'ancestral',
        'original_perplexity_batch_size': 8, 'scorer_revision': SCORER_REVISION,
        'precision': 'Unchanged upstream mixed precision; float64 diagnostic disabled.',
        'temperature': 'Unchanged upstream; no temperature override.',
    }, 'protocol')
    if any(type(seed) is not int for seed in protocol['seeds']):
        raise ValueError('protocol.seeds: expected integer seed IDs')
    checkpoint = field(protocol, 'checkpoint', 'protocol')
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise ValueError('protocol.checkpoint: expected an absolute checkpoint path')
    for condition, settings in CONDITIONS.items():
        expect(field(protocol, 'guidance_' + condition, 'protocol'), settings,
               'protocol.guidance_' + condition)


def validate_cell(root, condition, seed, protocol, source_root, manifest):
    cell = root / f'{condition}_seed{seed}'
    label = cell.name
    for name in ARTIFACTS:
        if not (cell / name).is_file():
            raise ValueError(f'{label}: missing artifact {name}; all ten completed cells are required')
    samples = read_json(cell / 'samples.json')
    scored = read_json(cell / 'all40.json')
    provenance = read_json(cell / 'authors_eval_environment.json')
    with (cell / '.hydra/config.yaml').open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f'{label}: expected a configuration mapping')
    check(config, {
        'mode': 'sample_eval', 'seed': seed, 'sampling.steps': 8,
        'sampling.predictor': 'guided', 'sampling.noise_removal': 'ancestral',
        'sampling.num_sample_batches': 2, 'sampling.use_float64': True,
        'sampling.semi_ar': False, 'loader.batch_size': 20, 'loader.eval_batch_size': 20,
        'model.name': 'small', 'model.length': 1024, 'algo.name': 'didi_instruct',
        'algo.backbone': 'dit', 'eval.disable_ema': False, 'eval.perplexity_batch_size': 8,
        'eval.gen_ppl_eval_model_name_or_path': 'gpt2-large',
        'eval.checkpoint_path': protocol['checkpoint'],
        'eval.generated_samples_path': str(cell / 'samples.json'),
        'checkpointing.save_dir': str(cell),
        **{'algo.' + name: value for name, value in CONDITIONS[condition].items()},
    }, label + '.config')
    check(provenance, {
        'checkpoint': protocol['checkpoint'], 'checkpoint_sha256': CHECKPOINT_SHA256,
        'upstream_source': 'inference', 'upstream_revision': INFERENCE_REVISION,
        'core_environment_guard_strict': True, 'precision_diagnostic': None,
        'entrypoint': str(source_root / 'main.py') + '; no runtime patches',
        'checkpoint_validation.complete_tensor_counts': {
            'backbone': 131, 'student_ema': 131, 'discriminator': 134},
        'checkpoint_validation.sampling_saved_devices': ['cuda:0'],
        'checkpoint_validation.first_state_tensor.saved_device': 'cuda:0',
    }, label + '.provenance')
    for name in ('installed_packages', 'environment'):
        value = field(provenance, name, label)
        if not isinstance(value, dict) or not value:
            raise ValueError(f'{label}: missing nonempty {name} provenance')
    texts = field(samples, 'generated_seqs', label)
    if not isinstance(texts, list) or len(texts) != 40 or not all(isinstance(text, str) for text in texts):
        raise ValueError(f'{label}: expected exactly 40 generated text samples')
    original_ppl = finite_number(field(samples, 'generative_ppl', label), label + '.original_ppl', True)
    entropy = finite_number(field(samples, 'entropy', label), label + '.entropy')
    if entropy < 0:
        raise ValueError(f'{label}: entropy cannot be negative')
    sample_hash = sha256(cell / 'samples.json')
    check(scored, {
        'samples_path': str(cell / 'samples.json'), 'samples_sha256': sample_hash,
        'original.generative_ppl': original_ppl, 'original.entropy': entropy,
        'original.generated_sample_count': 40, 'original.inferred_scored_sample_count': 32,
        'original.count_assumptions.generation_batch_size': 20,
        'original.count_assumptions.perplexity_batch_size': 8,
        'all_samples.scored_sample_count': 40, 'all_samples.entropy': entropy,
        'scoring.batch_size': 4, 'scoring.max_length': 1024,
        'scoring.model.resolved_revision': SCORER_REVISION,
        'scoring.model.source': 'gpt2-large', 'scoring.model.local_files_only': True,
        'scoring.metric_equations': 'unmodified vendored metrics.py',
        'scoring.model_dtype_policy': 'unmodified upstream from_pretrained; no dtype override',
        'scoring.accumulator_dtype': 'torch.float32',
        'upstream.commit': INFERENCE_REVISION,
        'upstream.metrics_sha256': manifest['files']['metrics.py'],
        'upstream.metrics_path': str(source_root / 'metrics.py'),
    }, label + '.all40')
    ppl = finite_number(field(scored, 'all_samples.generative_ppl', label), label + '.all40_ppl', True)
    weight = finite_number(field(scored, 'all_samples.scored_token_weight', label), label + '.token_weight', True)
    require_digest(field(scored, 'script_sha256', label), label + '.scoring_script_sha256')
    expect(scored['script_sha256'], sha256(REPO_ROOT / 'scripts/score_didi_authors_samples.py'),
           label + '.scoring_script_sha256')
    require_digest(field(scored, 'scoring.model.config_sha256', label), label + '.scorer_config_sha256')
    if not isinstance(field(scored, 'environment', label), dict) or not scored['environment']:
        raise ValueError(f'{label}: missing scorer environment')
    comparison = {
        'config': comparable_config(config),
        'generation_environment': provenance['environment'],
        'installed_packages': provenance['installed_packages'],
        'cuda_allocator_config': provenance.get('cuda_allocator_config'),
        'checkpoint_validation': provenance['checkpoint_validation'],
        'scoring': scored['scoring'], 'scorer_environment': scored['environment'],
        'metric_provenance': scored['upstream'], 'scoring_script_sha256': scored['script_sha256'],
    }
    return {
        'condition': condition, 'seed': seed, 'directory': str(cell),
        'generative_ppl': ppl, 'entropy': entropy, 'sample_count': 40,
        'scored_token_weight': weight, 'original_32_sample_ppl': original_ppl,
        'artifact_sha256': {name: sha256(cell / name) for name in ARTIFACTS},
    }, comparison


def summarize(root):
    root = Path(root).resolve()
    protocol = read_json(root / 'protocol.json')
    validate_protocol(protocol)
    source_root, manifest = verify_source('inference')
    cells = []
    reference = None
    for seed in SEEDS:
        for condition in CONDITIONS:
            cell, comparison = validate_cell(root, condition, seed, protocol, source_root, manifest)
            if reference is None:
                reference = comparison
            else:
                for name, value in comparison.items():
                    require_same(value, reference[name], f'{condition}_seed{seed}.{name}')
            cells.append(cell)
    aggregates = {}
    for condition in CONDITIONS:
        selected = [cell for cell in cells if cell['condition'] == condition]
        aggregates[condition] = {
            'seed_count': len(selected), 'generated_sample_count': 200, 'scored_sample_count': 200,
            **{metric: {'mean': statistics.mean(cell[metric] for cell in selected),
                        'sample_sd': statistics.stdev(cell[metric] for cell in selected)}
               for metric in ('generative_ppl', 'entropy')},
        }
    effects = {metric: aggregates['off'][metric]['mean'] - aggregates['on'][metric]['mean']
               for metric in ('generative_ppl', 'entropy')}
    return {
        'schema_version': 1, 'root': str(root), 'protocol': protocol,
        'protocol_sha256': sha256(root / 'protocol.json'), 'validation': 'all ten cells passed',
        'aggregation': 'Arithmetic mean and sample standard deviation (ddof=1) of five seed-level values.',
        'cells': cells, 'conditions': aggregates, 'off_minus_on': effects,
        'effect_interpretation': 'Positive PPL off-minus-on means guidance improves PPL; entropy is reported separately.',
        'shared_provenance': reference,
        'limitations': ['Same seed IDs do not imply identical random draws; guidance changes RNG consumption.',
                        'The comparison tests combined tilting and reranking, not their separate effects.',
                        'Reported SD describes seed variability, not SEM or a confidence interval.',
                        'Per-seed PPL values are averaged arithmetically, not recomputed from pooled token NLLs.'],
        'summary_script_sha256': sha256(Path(__file__)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error(f'Refusing to overwrite {args.output}')
    if not args.output.parent.is_dir():
        parser.error(f'Output parent directory does not exist: {args.output.parent}')
    result = summarize(args.root)
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with args.output.open('x') as stream:
        stream.write(serialized)
    print(json.dumps({'output': str(args.output), 'conditions': result['conditions'],
                      'off_minus_on': result['off_minus_on']}, indent=2))


if __name__ == '__main__':
    main()
