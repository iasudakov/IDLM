"""CPU-only synthetic artifact tests; no models, GPU, tracking, or downloads."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest

import yaml

import summarize_didi_guidance_control as summary


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def update_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


def fixture(root):
    source, manifest = summary.verify_source('inference')
    checkpoint = '/fixtures/authors.ckpt'
    protocol = {
        'checkpoint': checkpoint, 'checkpoint_sha256': summary.CHECKPOINT_SHA256,
        'upstream_source': 'inference', 'upstream_revision': summary.INFERENCE_REVISION,
        'seeds': summary.SEEDS, 'steps': 8, 'samples_per_seed_per_condition': 40,
        'generation_batch_size': 20, 'generation_batches': 2, 'length': 1024,
        'sampler': 'guided', 'cleanup': 'ancestral', 'original_perplexity_batch_size': 8,
        'scorer_revision': summary.SCORER_REVISION,
        'precision': 'Unchanged upstream mixed precision; float64 diagnostic disabled.',
        'temperature': 'Unchanged upstream; no temperature override.',
        **{'guidance_' + name: settings for name, settings in summary.CONDITIONS.items()},
    }
    write_json(root / 'protocol.json', protocol)
    for seed in summary.SEEDS:
        for condition, settings in summary.CONDITIONS.items():
            cell = root / f'{condition}_seed{seed}'
            ppl = (100 if condition == 'off' else 80) + seed
            entropy = (5.4 if condition == 'off' else 5.2) + seed / 100
            samples = {'generative_ppl': ppl - 1, 'entropy': entropy,
                       'generated_seqs': [f'{condition} seed {seed} sample {i}' for i in range(40)]}
            write_json(cell / 'samples.json', samples)
            config = {
                'mode': 'sample_eval', 'seed': seed,
                'sampling': {'steps': 8, 'predictor': 'guided', 'noise_removal': 'ancestral',
                             'num_sample_batches': 2, 'use_float64': True, 'semi_ar': False},
                'loader': {'batch_size': 20, 'eval_batch_size': 20},
                'model': {'name': 'small', 'length': 1024},
                'algo': {'name': 'didi_instruct', 'backbone': 'dit',
                         'time_conditioning': False, **settings},
                'eval': {'disable_ema': False, 'perplexity_batch_size': 8,
                         'gen_ppl_eval_model_name_or_path': 'gpt2-large',
                         'checkpoint_path': checkpoint, 'generated_samples_path': str(cell / 'samples.json')},
                'checkpointing': {'save_dir': str(cell), 'resume_ckpt_path': '${.save_dir}/checkpoints/last.ckpt'},
                'trainer': {'devices': '${device_count:}', 'precision': 'bf16'},
            }
            (cell / '.hydra').mkdir()
            (cell / '.hydra/config.yaml').write_text(yaml.safe_dump(config))
            provenance = {
                'checkpoint': checkpoint, 'checkpoint_sha256': summary.CHECKPOINT_SHA256,
                'upstream_source': 'inference', 'upstream_revision': summary.INFERENCE_REVISION,
                'core_environment_guard_strict': True, 'precision_diagnostic': None,
                'entrypoint': str(source / 'main.py') + '; no runtime patches',
                'checkpoint_validation': {
                    'complete_tensor_counts': {'backbone': 131, 'student_ema': 131, 'discriminator': 134},
                    'sampling_saved_devices': ['cuda:0'],
                    'first_state_tensor': {'name': 'reward_baseline', 'saved_device': 'cuda:0'}},
                'environment': {'python': '3.12.11', 'devices': ['fixture GPU']},
                'installed_packages': {'torch': '2.5.1', 'transformers': '4.55.4'},
                'cuda_allocator_config': 'expandable_segments:True',
            }
            write_json(cell / 'authors_eval_environment.json', provenance)
            scored = {
                'samples_path': str(cell / 'samples.json'), 'samples_sha256': summary.sha256(cell / 'samples.json'),
                'original': {'generative_ppl': ppl - 1, 'entropy': entropy,
                             'generated_sample_count': 40, 'inferred_scored_sample_count': 32,
                             'count_assumptions': {'generation_batch_size': 20, 'perplexity_batch_size': 8}},
                'all_samples': {'generative_ppl': ppl, 'entropy': entropy, 'scored_sample_count': 40,
                                'scored_token_weight': 40000 + seed},
                'scoring': {'batch_size': 4, 'max_length': 1024,
                            'metric_equations': 'unmodified vendored metrics.py',
                            'model_dtype_policy': 'unmodified upstream from_pretrained; no dtype override',
                            'accumulator_dtype': 'torch.float32',
                            'model': {'resolved_revision': summary.SCORER_REVISION, 'source': 'gpt2-large',
                                      'local_files_only': True, 'config_sha256': 'a' * 64}},
                'upstream': {'commit': summary.INFERENCE_REVISION,
                             'metrics_sha256': manifest['files']['metrics.py'],
                             'metrics_path': str(source / 'metrics.py')},
                'environment': {'torch': '2.5.1', 'device': 'cuda:0'},
                'script_sha256': summary.sha256(summary.REPO_ROOT / 'scripts/score_didi_authors_samples.py'),
            }
            write_json(cell / 'all40.json', scored)


class GuidanceSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='didi-guidance-summary-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        fixture(self.root)
        self.output = self.root / 'summary.json'

    def assert_rejected(self, expression=''):
        with self.assertRaisesRegex(ValueError, expression), redirect_stdout(io.StringIO()):
            summary.main(['--root', str(self.root), '--output', str(self.output)])
        self.assertFalse(self.output.exists())

    def test_complete_summary_arithmetic_mean_sample_sd_and_preserved_inputs(self):
        before = {str(path): summary.sha256(path) for path in self.root.rglob('*') if path.is_file()}
        with redirect_stdout(io.StringIO()):
            summary.main(['--root', str(self.root), '--output', str(self.output)])
        report = json.loads(self.output.read_text())
        self.assertEqual(len(report['cells']), 10)
        self.assertEqual(report['conditions']['off']['generative_ppl']['mean'], 103)
        self.assertEqual(report['conditions']['on']['generative_ppl']['mean'], 83)
        self.assertAlmostEqual(report['conditions']['off']['generative_ppl']['sample_sd'],
                               statistics.stdev([101, 102, 103, 104, 105]))
        self.assertEqual(report['conditions']['off']['scored_sample_count'], 200)
        self.assertEqual(report['off_minus_on']['generative_ppl'], 20)
        self.assertAlmostEqual(report['off_minus_on']['entropy'], 0.2)
        self.assertEqual(before, {name: summary.sha256(Path(name)) for name in before})

    def test_output_is_exclusive(self):
        self.output.write_text('keep this file')
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            summary.main(['--root', str(self.root), '--output', str(self.output)])
        self.assertEqual(self.output.read_text(), 'keep this file')

    def test_missing_cell_fails_without_partial_summary(self):
        (self.root / 'on_seed5/all40.json').unlink()
        self.assert_rejected('all ten completed cells')

    def test_tampered_samples_hash_fails(self):
        update_json(self.root / 'on_seed2/samples.json', lambda data:
                    data['generated_seqs'].__setitem__(0, 'altered sample'))
        self.assert_rejected('samples_sha256')

    def test_wrong_provenance_flags_fail(self):
        path = self.root / 'on_seed2/authors_eval_environment.json'
        original = json.loads(path.read_text())
        changes = {
            'checkpoint_sha256': 'c' * 64, 'upstream_revision': 'd' * 40,
            'upstream_source': 'main', 'core_environment_guard_strict': False,
            'precision_diagnostic': {'diagnostic': 'float64_sampling_arithmetic'},
            'entrypoint': '/wrong/runtime.py',
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                write_json(path, {**original, key: value})
                self.assert_rejected(key)
        write_json(path, original)

    def test_wrong_scores_counts_or_revision_fail(self):
        path = self.root / 'off_seed3/all40.json'
        original = json.loads(path.read_text())
        mutations = [
            ('original', 'generative_ppl', 999), ('original', 'entropy', 999),
            ('all_samples', 'entropy', 999), ('all_samples', 'scored_sample_count', 32),
            ('scoring', 'batch_size', 8), ('scoring', 'max_length', 512),
        ]
        for group, key, value in mutations:
            with self.subTest(group=group, key=key):
                changed = copy.deepcopy(original)
                changed[group][key] = value
                write_json(path, changed)
                self.assert_rejected(key)
        changed = copy.deepcopy(original)
        changed['scoring']['model']['resolved_revision'] = 'f' * 40
        write_json(path, changed)
        self.assert_rejected('resolved_revision')

    def test_cross_cell_package_environment_metric_script_and_hidden_config_drift_fail(self):
        provenance_path = self.root / 'on_seed4/authors_eval_environment.json'
        original = json.loads(provenance_path.read_text())
        for name in ('installed_packages', 'environment'):
            with self.subTest(name=name):
                changed = copy.deepcopy(original)
                changed[name]['unexpected_change'] = True
                write_json(provenance_path, changed)
                self.assert_rejected('differs between cells')
        write_json(provenance_path, original)
        scored_path = self.root / 'on_seed4/all40.json'
        scored = json.loads(scored_path.read_text())
        for name, value in (('script_sha256', 'c' * 64), ('environment', {'torch': 'changed'})):
            with self.subTest(name=name):
                write_json(scored_path, {**scored, name: value})
                self.assert_rejected('script_sha256' if name == 'script_sha256' else 'differs between cells')
        write_json(scored_path, scored)
        config_path = self.root / 'on_seed4/.hydra/config.yaml'
        config = yaml.safe_load(config_path.read_text())
        config['algo']['time_conditioning'] = True
        config_path.write_text(yaml.safe_dump(config))
        self.assert_rejected('config')

    def test_wrong_seed_guidance_ratio_or_batch_fails(self):
        path = self.root / 'off_seed2/.hydra/config.yaml'
        original = yaml.safe_load(path.read_text())
        for group, key, value in (('algo', 'guidance_scale_start', 30),
                                  ('algo', 'num_candidates', 4),
                                  ('algo', 'rerank_steps_ratio', 0.75),
                                  ('loader', 'eval_batch_size', 10),
                                  ('sampling', 'predictor', 'ancestral')):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed[group][key] = value
                path.write_text(yaml.safe_dump(changed))
                self.assert_rejected(key)
        changed = copy.deepcopy(original)
        changed['seed'] = 1
        path.write_text(yaml.safe_dump(changed))
        self.assert_rejected('seed')

    def test_int_and_float_guidance_values_are_equivalent(self):
        path = self.root / 'on_seed2/.hydra/config.yaml'
        config = yaml.safe_load(path.read_text())
        config['algo']['guidance_scale_start'] = 30
        config['algo']['guidance_scale_end'] = 40
        path.write_text(yaml.safe_dump(config))
        self.assertEqual(len(summary.summarize(self.root)['cells']), 10)

    def test_protocol_mismatch_rejected_before_cells(self):
        update_json(self.root / 'protocol.json', lambda value: value.__setitem__('steps', 16))
        self.assert_rejected('protocol.steps')

    def test_nonfinite_metric_and_wrong_text_count_fail(self):
        path = self.root / 'on_seed1/all40.json'
        original = json.loads(path.read_text())
        changed = copy.deepcopy(original)
        changed['all_samples']['generative_ppl'] = float('nan')
        write_json(path, changed)
        self.assert_rejected('nonfinite')
        write_json(path, original)
        update_json(self.root / 'on_seed1/samples.json', lambda data: data['generated_seqs'].pop())
        self.assert_rejected('exactly 40')


if __name__ == '__main__':
    unittest.main(verbosity=2)
