"""CPU checks for the S-FLM random-position ancestral T=0 comparison.

The independent oracle uses archived IDLM 14594e sampler equations. Tests use
small context-dependent distributions, never checkpoints, GPUs, or tracker IO.
"""
from contextlib import ExitStack
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

import sdtt_eval as validation


def categorical_module():
    """Load the real stateless sampler without importing optional training/GPU dependencies."""
    path = Path(__file__).resolve().parents[1] / 'trainer_base.py'
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'sample_categorical')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(sample_categorical=namespace['sample_categorical'])


class Tokenizer:
    bos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[1] * len(text))

    def decode(self, tokens, **kwargs):
        return 'generated code'


class Logger:
    def __init__(self):
        self.metrics = []

    def log_metrics(self, values, step=None):
        self.metrics.append((values, step))


class Noise:
    eps = .001

    def __call__(self, t):
        return -(1 - self.eps), 1 - (1 - self.eps) * t


class Model:
    device = torch.device('cpu')
    mask_index, num_tokens = 6, 11
    time_conditioning = False
    noise = Noise()

    def __init__(self, directory=None):
        self.seen, self.logged = [], []
        self.global_step = 500
        self.tokenizer = Tokenizer()
        logger = Logger()
        self.trainer = SimpleNamespace(logger=logger, loggers=[logger], global_rank=0,
                                       world_size=1, ckpt_path=None, sanity_checking=False)
        self.config = OmegaConf.create({
            'seed': 1, 'data': {'train': 'tiny_gsm', 'separator': '\n'},
            'sampling': {'steps': 4, 'predictor': 'ancestral_cache', 'num_sample_log': 0},
            'eval': {'gsm8k_pareto_thresholds': [.49], 'gsm8k_batch_size': 2,
                     'gsm8k_grading_workers': 1, 'gsm8k_at_validation': True,
                     'gsm8k_greedy_token': True, 'gsm8k_steps': None,
                     'validate_before_train': True, 'gsm8k_every_n_validations': 1},
            'checkpointing': {'save_dir': directory},
        })

    def prior_sample(self, *shape):
        return torch.full(shape, self.mask_index, dtype=torch.long)

    def _sigma_from_alphat(self, alpha):
        return -alpha.log()

    def forward(self, x, sigma):
        self.seen.append(x.clone())
        positions = torch.arange(x.shape[1]).expand_as(x)
        context = x.clamp_max(self.mask_index - 1).sum(-1, keepdim=True)
        winner = (context + positions + (sigma.reshape(-1, 1) * 3).long()) % 5 + 1
        probabilities = torch.full((*x.shape, self.mask_index + 1), .02)
        probabilities.scatter_(-1, winner.unsqueeze(-1), .8)
        probabilities[..., self.mask_index] = 0.
        return (probabilities / probabilities.sum(-1, keepdim=True)).log()

    def log(self, name, value, **kwargs):
        self.logged.append((name, float(value)))


@torch.no_grad()
def reference_sflm(model, prompts, steps):
    """Archived random-position T=0 equations, independent of the new helper."""
    state = torch.full((len(prompts), model.num_tokens), model.mask_index, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        state[row, :len(prompt)] = torch.tensor(prompt)
    grid = torch.linspace(1., 1e-5, steps + 1)
    width = (1 - 1e-5) / steps
    distribution, forwards = None, 0
    for index in range(steps):
        time = grid[index] * torch.ones(len(prompts), 1)
        current_alpha = model.noise(time)[1]
        next_alpha = (torch.ones_like(current_alpha) if index + 1 == steps
                      else model.noise(time - width)[1])
        if distribution is None:
            distribution = model.forward(state, -current_alpha.log()).exp()
            forwards += 1
        chance = ((next_alpha - current_alpha) / (1 - current_alpha).clamp_min(1e-12)).clamp(0, 1)
        positions = (state == model.mask_index) & (torch.rand(state.shape) < chance)
        next_state = state.clone()
        next_state[positions] = distribution.argmax(-1)[positions]
        if (model.config.sampling.predictor == 'ancestral' or model.time_conditioning
                or not torch.equal(state, next_state)):
            distribution = None
        state = next_state
        for row, prompt in enumerate(prompts):
            state[row, :len(prompt)] = torch.tensor(prompt)
    return state, forwards


def examples():
    return [{'input_ids': [0, 2] if index % 2 == 0 else [0, 2, 2],
             'answer': 'yes' if index % 2 == 0 else 'no', 'question': f'Q{index}'}
            for index in range(5)]


class AncestralDecoderTest(unittest.TestCase):
    def test_independent_oracle_1_2_4_steps_cache_and_conditioning(self):
        prompts = [[0, 2], [0, 3, 1]]
        for seed in (1, 37):
            for steps in (1, 2, 4):
                for predictor, conditioning in [('ancestral_cache', False), ('ancestral', False),
                                                 ('ancestral_cache', True)]:
                    with self.subTest(seed=seed, steps=steps, predictor=predictor, conditioning=conditioning):
                        expected_model, actual_model = Model(), Model()
                        for model in (expected_model, actual_model):
                            model.config.sampling.predictor = predictor
                            model.time_conditioning = conditioning
                        torch.manual_seed(seed)
                        expected, nfe = reference_sflm(expected_model, prompts, steps)
                        expected_rng = torch.get_rng_state().clone()
                        torch.manual_seed(seed)
                        actual, info = validation.generate_greedy_samples(actual_model, prompts, steps)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertEqual(info['nfe'], nfe)
                        self.assertEqual(info['nfe_per_row'].tolist(), [nfe, nfe])
                        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                        self.assertEqual(len(actual_model.seen), len(expected_model.seen))
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual_model.seen, expected_model.seen)))
                        self.assertFalse(bool((actual == actual_model.mask_index).any()))
                        for state in actual_model.seen + [actual]:
                            for row, prompt in enumerate(prompts):
                                self.assertEqual(state[row, :len(prompt)].tolist(), prompt)

    def test_position_draws_remain_random_at_temperature_zero_and_cache_skips_forwards(self):
        model = Model()
        # No early reveal: high uniforms preserve masks until the scheduled final step.
        with patch.object(torch, 'rand', side_effect=lambda shape, **kwargs: torch.full(shape, .999)):
            output, info = validation.generate_greedy_samples(model, [[0, 2]], 4)
        self.assertEqual(info['nfe'], 1)
        self.assertEqual(len(model.seen), 1)
        self.assertFalse(bool((output == model.mask_index).any()))
        uncached = Model()
        uncached.config.sampling.predictor = 'ancestral'
        with patch.object(torch, 'rand', side_effect=lambda shape, **kwargs: torch.full(shape, .999)):
            _, info = validation.generate_greedy_samples(uncached, [[0, 2]], 4)
        self.assertEqual(info['nfe'], 4)  # Exactly the budget; no appended final forward.

    def test_budget_fallback_and_invalid_inputs(self):
        model = Model()
        model.config.sampling.steps = 64
        with patch.object(torch, 'rand', side_effect=lambda shape, **kwargs: torch.full(shape, .999999)):
            _, metadata = validation.generate_greedy_samples(model, [[0, 2]])
        self.assertEqual(metadata['nfe'], 1)
        self.assertEqual(validation._greedy_protocol(model)['steps'], 64)
        for steps in (0, -1, True, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                validation.generate_greedy_samples(Model(), [[0]], steps)
        for prompts in ([], [[0] * 11], [[6]]):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                validation.generate_greedy_samples(Model(), prompts, 2)
        model.config.eval.gsm8k_steps = 0
        with self.assertRaises(ValueError):
            validation.generate_greedy_samples(model, [[0]])
        model.config.sampling.predictor = 'analytic'
        with self.assertRaises(ValueError):
            validation.generate_greedy_samples(model, [[0]], 2)


class AncestralTemperatureTest(unittest.TestCase):
    def test_tempered_proposals_and_final_cleanup_use_categorical_sampling(self):
        for temperature in (.1, 1.):
            with self.subTest(temperature=temperature):
                model = Model()
                probabilities = torch.tensor([.1, .2, .3, .4, 0., 0., 0.])
                model.forward = lambda x, sigma: probabilities.log().repeat(*x.shape, 1)
                seen = []

                def categorical(weights):
                    seen.append(weights.clone())
                    # Force a non-argmax token, including the final cleanup.
                    return torch.ones(weights.shape[:-1], dtype=torch.long)

                with patch.dict('sys.modules', {'trainer_base': SimpleNamespace(sample_categorical=categorical)}), \
                        patch.object(torch, 'rand', side_effect=lambda shape, **kw: torch.ones(shape)):
                    samples, metadata = validation.generate_greedy_samples(
                        model, [[0, 2]], 2, final_denoise=True, temperature=temperature)
                self.assertEqual(samples[0, :2].tolist(), [0, 2])
                self.assertEqual(samples[0, 2:].tolist(), [1] * (model.num_tokens - 2))
                self.assertEqual(metadata['nfe'], 2)  # Cached transitions plus cleanup.
                self.assertEqual(len(seen), 3)  # Resample each transition, then cleanup.
                expected = probabilities.pow(1 / temperature)
                expected /= expected.sum()
                for weights in seen:
                    self.assertTrue(torch.allclose(weights[0, 0], expected, atol=1e-6))
                    self.assertEqual(float(weights[..., model.mask_index].sum()), 0.)

    def test_invalid_temperature_fails_before_model_forward(self):
        for temperature in (-1., float('nan'), float('inf')):
            model = Model()
            with self.subTest(temperature=temperature), self.assertRaisesRegex(ValueError, 'temperature'):
                validation.generate_greedy_samples(model, [[0]], 4, temperature=temperature)
            self.assertEqual(model.seen, [])


class AncestralSweepTest(unittest.TestCase):
    def test_repeatability_preserves_training_and_confidence_rng(self):
        model = Model()
        torch.manual_seed(5551)
        before = torch.get_rng_state().clone()
        with patch.object(validation, 'grade_sample', return_value=True), \
                patch.object(torch.cuda, 'get_rng_state', side_effect=AssertionError('CPU touched CUDA')):
            first = validation.sweep_greedy(model, examples())
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            second = validation.sweep_greedy(model, examples())
            self.assertEqual(first, second)
            confidence = validation.sweep(model, examples())
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            validation.sweep_greedy(model, examples())
            self.assertEqual(validation.sweep(model, examples()), confidence)

    def test_distributed_pooling_includes_empty_ranks(self):
        data, world = examples(), 7
        for rank in range(world):
            with self.subTest(rank=rank):
                model = Model()
                model.config.sampling.predictor = 'ancestral'
                model.config.sampling.steps = 2
                model.trainer.global_rank, model.trainer.world_size = rank, world
                local = data[rank::world]
                expected_local = [sum(row['answer'] == 'yes' for row in local), len(local), 2 * len(local)]
                reductions = []

                def reduce(tensor, op):
                    reductions.append(tensor.tolist())
                    tensor.copy_(torch.tensor([3., 5., 10.], dtype=tensor.dtype))

                def gather(out, error):
                    out[:] = [error] * world

                with ExitStack() as stack:
                    for name, value in [('is_available', True), ('is_initialized', True),
                                        ('get_rank', rank), ('get_world_size', world)]:
                        stack.enter_context(patch.object(validation.dist, name, return_value=value))
                    stack.enter_context(patch.object(validation.dist, 'all_gather_object', side_effect=gather))
                    stack.enter_context(patch.object(validation.dist, 'all_reduce', side_effect=reduce))
                    grade = stack.enter_context(patch.object(validation, 'grade_sample',
                        side_effect=lambda sample, answer, **kwargs: answer == 'yes'))
                    result = validation.sweep_greedy(model, data)
                self.assertEqual(reductions, [expected_local])
                self.assertEqual(grade.call_count, len(local))
                self.assertEqual(result['count'], 5)
                self.assertEqual(result['correct'], 3)
                self.assertEqual(result['accuracy'], .6)
                self.assertEqual(result['nfe'], 2.)

    def test_worker_failure_reaches_error_boundary_before_reduce(self):
        model = Model()
        with patch.object(validation, 'grade_sample', side_effect=RuntimeError('grader failed')), \
                patch.object(validation.dist, 'all_reduce') as reduce:
            with self.assertRaisesRegex(RuntimeError, 'grader failed'):
                validation.sweep_greedy(model, examples())
        reduce.assert_not_called()


class AncestralLoggingTest(unittest.TestCase):
    def test_initial_periodic_final_logging_and_history_keep_separate_frontier(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Model(directory)
            data = Path(directory) / 'gsm8k.json'
            data.write_text(json.dumps([{'prompt': 'Q', 'response_ground_truth': 'yes'}]))
            model.config.eval.gsm8k_data_path = str(data)
            curve = [{'threshold': .49, 'accuracy': .5, 'nfe': 3.}]
            greedy = {'accuracy': .6, 'nfe': 4., 'correct': 3, 'count': 5,
                      'steps': 4, 'sampler': 'random', 'temperature': 0.}
            with patch.object(validation, 'sweep_greedy', return_value=greedy) as ancestral, \
                    patch.object(validation, 'sweep', return_value=curve), \
                    patch.object(validation, 'render_frontier', return_value=np.zeros((2, 2, 3))):
                model.global_step = 0
                validation.log_validation_gsm8k(model, initial=True)
                self.assertEqual(model.logged, [])
                self.assertEqual(model.trainer.logger.metrics[0][0]['val/gsm8k_acc_greedy'], .6)
                model.global_step = 500
                validation.log_validation_gsm8k(model)
                metrics = dict(model.logged)
                self.assertAlmostEqual(metrics['val/gsm8k_acc_greedy'], .6, places=6)
                self.assertEqual(metrics['val/gsm8k_nfe_greedy'], 4.)
                self.assertNotIn('val/gsm8k_acc', metrics)  # No misleading legacy alias.
                model.logged.clear()
                model.global_step = 501
                validation.log_validation_gsm8k(model, final=True)
                self.assertEqual(model.logged, [])
                final_metrics = [values for values, step in model.trainer.logger.metrics
                                 if step == 501 and 'val/gsm8k_acc_greedy' in values]
                self.assertEqual(len(final_metrics), 1)
                validation.log_validation_gsm8k(model, final=True)
                self.assertEqual(ancestral.call_count, 3)  # Duplicate final step is skipped.
            report = json.loads((Path(directory) / 'pareto_history.json').read_text())
            self.assertEqual(report['greedy'], greedy)
            self.assertEqual(set(report['curves']), {'0', '500', '501'})
            self.assertEqual(report['curves']['501'], curve)
            self.assertEqual(report['protocol']['sampler'], 'low_confidence_dynamic')
            self.assertEqual(report['protocol']['greedy']['temperature'], 0.)
            self.assertEqual(report['protocol']['greedy']['epsilon'], 1e-5)
            self.assertEqual(report['protocol']['greedy']['steps'], 4)
            # The fixed ancestral point never enters confidence frontier metric coordinates.
            native = [(values, step) for values, step in model.trainer.logger.metrics
                      if any(key.startswith('pareto/gsm8k_step_') for key in values)]
            self.assertTrue(all(step == 300 for values, step in native))

    def test_disabled_toggle_clears_previous_result_and_emits_no_greedy_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Model(directory)
            model.config.eval.gsm8k_greedy_token = False
            model._gsm8k_examples = examples()
            model._gsm8k_protocol = {'greedy': {'temperature': 0.}}
            model._gsm8k_greedy_result = {'accuracy': 1., 'nfe': 4.}
            curve = [{'threshold': .49, 'accuracy': .5, 'nfe': 3.}]
            with patch.object(validation, 'sweep_greedy') as ancestral, \
                    patch.object(validation, 'sweep', return_value=curve):
                validation.log_validation_gsm8k(model)
            ancestral.assert_not_called()
            self.assertIsNone(model._gsm8k_greedy_result)
            self.assertNotIn('greedy', model._gsm8k_protocol)
            self.assertFalse(any('greedy' in name for name, value in model.logged))

    def test_standalone_report_includes_greedy_before_thresholds_without_changing_confidence_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Model(directory)
            data, output = Path(directory) / 'data.json', Path(directory) / 'results.json'
            data.write_text(json.dumps([{'prompt': 'Q', 'response_ground_truth': 'yes'}]))
            config = OmegaConf.create(OmegaConf.to_container(model.config, resolve=True))
            config.eval.update({'gsm8k_data_path': str(data), 'checkpoint_path': 'student.ckpt',
                                'disable_ema': False, 'generated_samples_path': str(output)})
            streams = []

            def confidence(*args, **kwargs):
                streams.append(torch.rand(4))
                return 1, 1, 3, 3

            with patch.object(validation, '_evaluate_threshold', side_effect=confidence), \
                    patch.object(validation, 'grade_sample', return_value=True):
                config.eval.gsm8k_greedy_token = False
                plain = validation.evaluate(model, config)
                self.assertNotIn('greedy', json.loads(output.read_text()))
                config.eval.gsm8k_greedy_token = True
                actual = validation.evaluate(model, config)
            self.assertEqual(actual, plain)
            self.assertTrue(torch.equal(streams[0], streams[1]))
            report = json.loads(output.read_text())
            self.assertEqual(report['results'], actual)
            self.assertEqual(report['greedy']['count'], 1)
            self.assertEqual(report['greedy']['accuracy'], 1.)
            self.assertLessEqual(report['greedy']['nfe'], 4.)
            self.assertEqual(report['greedy']['sampler'], 'random')
            self.assertEqual(report['greedy_protocol']['epsilon'], 1e-5)


if __name__ == '__main__':
    unittest.main()
