"""CPU checks for TinyGSM validation decoding, grading and distributed counts.

The fixed-grid oracle follows the archived 14594e S-FLM random-position/T=0
sampler equations, independently of tinygsm_validation's implementation.
No model weights, network services, or GPUs are used.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tinygsm_validation as validation
import tinygsm_sandbox_gsm8k as grader


class Tokenizer:
    bos_token_id, eos_token_id = 0, 1

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        return SimpleNamespace(input_ids=[1] * len(text))

    def decode(self, tokens, skip_special_tokens=False):
        return 'def simple_math_problem():\n    return 42'


class Logger:
    def __init__(self):
        self.metrics, self.tables, self.images = [], [], []

    def log_metrics(self, metrics, step=None):
        self.metrics.append((metrics, step))

    def log_table(self, key, columns, data, step=None):
        self.tables.append((key, columns, data, step))

    def log_image(self, key, image, step=None):
        self.images.append((key, image, step))


class Noise:
    eps = .001

    def __call__(self, t):
        return -(1 - self.eps), 1 - (1 - self.eps) * t


class FixedModel:
    """Tied .5 confidences exercise strict thresholds and forced reveals."""
    device = torch.device('cpu')
    mask_index, num_tokens = 4, 5
    noise = Noise()
    time_conditioning = False

    def __init__(self, directory=None):
        self.seen, self.logged = [], []
        self.global_step = 500
        self.tokenizer = Tokenizer()
        logger = Logger()
        self.trainer = SimpleNamespace(logger=logger, loggers=[logger], global_rank=0,
                                       world_size=1, ckpt_path=None, sanity_checking=False)
        self.config = OmegaConf.create({
            'seed': 1,
            'data': {'train': 'tiny_gsm', 'separator': '\n', 'cache_dir': directory},
            'sampling': {'steps': 3, 'predictor': 'ancestral_cache',
                         'noise_removal': 'ancestral', 'num_sample_log': 2},
            'eval': {'gsm8k_pareto_thresholds': [.49, .9], 'gsm8k_batch_size': 2,
                     'gsm8k_grading_workers': 1, 'tinygsm_exec': True,
                     'gsm8k_at_validation': True, 'gsm8k_greedy_token': True,
                     'validate_before_train': True, 'gsm8k_every_n_validations': 1},
            'checkpointing': {'save_dir': directory},
        })

    def log(self, name, value, **kwargs):
        self.logged.append((name, float(value), kwargs))

    def prior_sample(self, *shape):
        return torch.full(shape, self.mask_index, dtype=torch.long)

    def _sigma_from_alphat(self, alpha):
        return -alpha.log()

    def forward(self, x, sigma):
        self.seen.append(x.clone())
        probs = torch.zeros(*x.shape, self.mask_index + 1)
        probs[..., 1:3] = .5
        return probs.log()


class StatefulModel(FixedModel):
    """Predictions depend on the progressively revealed context and time."""
    mask_index, num_tokens = 6, 11

    def forward(self, x, sigma):
        self.seen.append(x.clone())
        position = torch.arange(x.shape[1]).expand_as(x)
        context = x.clamp_max(self.mask_index - 1).sum(-1, keepdim=True)
        time = (sigma.reshape(-1, 1) * 3).long()
        winner = (context + position + time).remainder(self.mask_index - 1) + 1
        probs = torch.full((*x.shape, self.mask_index + 1), .02)
        probs.scatter_(-1, winner.unsqueeze(-1), .8)
        probs[..., self.mask_index] = 0.
        probs /= probs.sum(-1, keepdim=True)
        return probs.log()


@torch.no_grad()
def reference_sflm_greedy(model, prompts, steps, eps=1e-5):
    """Independent fixed-grid oracle: scheduled Bernoulli reveals + argmax.

    The source of the equations is archived trainer_base.py at 14594e,
    generate_samples / _ancestral_update / _ancestral_greedy_token. The last
    scheduled transition uses alpha_s=1 and does not append another forward.
    """
    latent = torch.full((len(prompts), model.num_tokens), model.mask_index, dtype=torch.long)
    for row, prefix in enumerate(prompts):
        latent[row, :len(prefix)] = torch.tensor(prefix)
    grid = torch.linspace(1., eps, steps + 1)
    spacing = (1. - eps) / steps
    distribution, forwards = None, 0
    for index in range(steps):
        time = grid[index] * torch.ones(len(prompts), 1)
        alpha_t = model.noise(time)[1]
        alpha_s = (torch.ones_like(alpha_t) if index == steps - 1
                   else model.noise(time - spacing)[1])
        if distribution is None:
            distribution = model.forward(latent, -alpha_t.log()).exp()
            forwards += 1
        reveal_probability = ((alpha_s - alpha_t)
                              / (1. - alpha_t).clamp_min(1e-12)).clamp(0, 1)
        reveal = torch.rand(latent.shape) < reveal_probability
        new_latent = latent.clone()
        written = (latent == model.mask_index) & reveal
        new_latent[written] = distribution.argmax(-1)[written]
        if (model.config.sampling.predictor == 'ancestral' or model.time_conditioning
                or not torch.equal(new_latent, latent)):
            distribution = None
        latent = new_latent
        for row, prefix in enumerate(prompts):
            latent[row, :len(prefix)] = torch.tensor(prefix)
    return latent, forwards


def examples():
    return [{'input_ids': [0, 2] if index % 2 == 0 else [0, 2, 2],
             'answer': '#### 42' if index % 2 == 0 else '#### 0',
             'question': f'question {index}'} for index in range(5)]


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dynamic_strict_threshold_pins_prompts_and_counts_only_active_rows(self):
        model = FixedModel()
        prompts = [[0, 2], [0, 2, 2]]
        samples, info = validation.generate_samples(model, prompts, threshold=.5)
        self.assertEqual(info['nfe'], 3)
        self.assertEqual(info['nfe_per_row'].tolist(), [3, 2])
        self.assertEqual(samples.tolist(), [[0, 2, 1, 1, 1], [0, 2, 2, 1, 1]])
        # Generated EOS1 fills every answer slot; it does not stop the row.
        self.assertFalse(bool((samples == model.mask_index).any()))
        for state in model.seen:
            for row, prompt in enumerate(prompts):
                self.assertEqual(state[row, :len(prompt)].tolist(), prompt)

    def test_dynamic_low_threshold_and_capped_fallback_nfe(self):
        one, metadata = validation.generate_samples(FixedModel(), [[0]], .49)
        self.assertEqual(metadata['nfe_per_row'].tolist(), [1])
        self.assertEqual(one.tolist(), [[0, 1, 1, 1, 1]])
        fallback, metadata = validation.generate_samples(FixedModel(), [[0]], 1., max_steps=1)
        self.assertEqual(metadata['nfe'], 2)
        self.assertEqual(metadata['nfe_per_row'].tolist(), [2])
        self.assertTrue(torch.equal(fallback, one))

    def test_seeded_greedy_matches_legacy_equations_states_nfe_and_rng(self):
        prompts = [[0, 2], [0, 3, 1]]
        for seed in (1, 37):
            for steps in (1, 7, 64):
                for predictor, conditioning in (('ancestral_cache', False),
                                                ('ancestral', False),
                                                ('ancestral_cache', True)):
                    with self.subTest(seed=seed, steps=steps, predictor=predictor,
                                      conditioning=conditioning):
                        old, new = StatefulModel(), StatefulModel()
                        for model in (old, new):
                            model.config.sampling.predictor = predictor
                            model.time_conditioning = conditioning
                        torch.manual_seed(seed)
                        expected, nfe = reference_sflm_greedy(old, prompts, steps)
                        expected_rng = torch.get_rng_state().clone()
                        torch.manual_seed(seed)
                        actual, metadata = validation.generate_greedy_samples(new, prompts, steps)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertEqual(metadata['nfe'], nfe)
                        self.assertEqual(metadata['nfe_per_row'].tolist(), [nfe, nfe])
                        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                        self.assertEqual(len(old.seen), len(new.seen))
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old.seen, new.seen)))
                        self.assertFalse(bool((actual == new.mask_index).any()))

    def test_greedy_uses_sflm_budget_without_extra_final_forward(self):
        model = FixedModel()
        model.config.sampling.predictor = 'ancestral'
        model.config.eval.gsm8k_steps = 4
        output, metadata = validation.generate_greedy_samples(model, [[0, 2]])
        self.assertEqual(metadata['nfe'], 4)
        self.assertEqual(len(model.seen), 4)
        self.assertEqual(output.tolist(), [[0, 2, 1, 1, 1]])

    def test_invalid_decoder_inputs_fail(self):
        for prompts, threshold, temperature, cap in [
            ([], .5, 0., None), ([[0] * 5], .5, 0., None),
            ([[4]], .5, 0., None), ([[0]], float('nan'), 0., None),
            ([[0]], .5, -1., None), ([[0]], .5, 0., 0),
        ]:
            with self.subTest(prompts=prompts, threshold=threshold):
                with self.assertRaises(ValueError):
                    validation.generate_samples(FixedModel(), prompts, threshold, temperature, cap)
        for steps in (0, -1, True, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                validation.generate_greedy_samples(FixedModel(), [[0]], steps)
        with self.assertRaises(ValueError):
            validation.generate_greedy_samples(FixedModel(), [[4]], 2)


class PromptAndGraderTests(unittest.TestCase):
    def test_reference_grid_and_literal_separator(self):
        self.assertEqual(len(validation.THRESHOLDS), 27)
        self.assertEqual(validation.THRESHOLDS[:3], [.325, .35, .375])
        self.assertEqual(validation.THRESHOLDS[-2:], [.95, .98])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'examples.json'
            path.write_text(json.dumps([{'prompt': ' Q ', 'response_ground_truth': '#### 42'}]))
            tokenizer = Tokenizer()
            data = validation.load_examples(path, tokenizer, r'\n', 8)
            self.assertEqual(tokenizer.calls, [(r'\n', False), ('Q', False)])
            self.assertEqual(data[0]['input_ids'], [0, 1, 1, 1])
            with self.assertRaises(ValueError):
                validation.load_examples(path, tokenizer, r'\n', 4)

    def test_explicit_missing_path_does_not_fall_back_and_bundle_data_is_reused(self):
        with self.assertRaises(FileNotFoundError):
            validation.resolve_data_path(OmegaConf.create({
                'eval': {'gsm8k_data_path': '/no/such/tinygsm-test-data.json'}}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data' / 'gsm8k_test.json'
            path.parent.mkdir()
            path.write_text('[]')
            config = OmegaConf.create({'model_weights_dir': directory, 'eval': {},
                                       'data': {'cache_dir': str(Path(directory) / 'other')}})
            self.assertEqual(validation.resolve_data_path(config), path)

    def test_grader_requires_executable_correct_answer(self):
        self.assertTrue(grader.evaluate_samples(
            '```python\ndef simple_math_problem():\n    return 42\n```', '#### 42', 1.))
        self.assertTrue(grader.evaluate_samples(
            'def simple_math_problem():\n    print("Answer: 42")', '#### 42', 1.))
        self.assertFalse(grader.evaluate_samples('The answer is 42', '#### 42', 1.))
        self.assertFalse(grader.evaluate_samples(
            'def simple_math_problem():\n    return 41', '#### 42', 1.))

    def test_grader_subprocess_and_timeout(self):
        self.assertTrue(validation.grade_sample(
            'def simple_math_problem():\n    return 42', '#### 42', .2))
        self.assertFalse(validation.grade_sample(
            'def simple_math_problem():\n    while True: pass', '#### 42', .05))


class SweepAndLoggingTests(unittest.TestCase):
    def test_both_sweeps_preserve_cpu_training_rng_without_touching_cuda(self):
        model = FixedModel()
        torch.manual_seed(5551)
        before = torch.get_rng_state().clone()
        with patch.object(validation, 'grade_sample', return_value=True), \
                patch.object(torch.cuda, 'get_rng_state', side_effect=AssertionError('CPU eval touched CUDA')):
            first = validation.sweep_greedy(model, examples())
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            second = validation.sweep_greedy(model, examples())
            self.assertEqual(first, second)
            validation.sweep(model, examples())
            self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_every_rank_including_empty_shards_contributes_counts(self):
        data, world = examples(), 7
        for kind in ('pareto', 'greedy'):
            for rank in range(world):
                with self.subTest(kind=kind, rank=rank):
                    model = FixedModel()
                    model.config.eval.gsm8k_pareto_thresholds = [.9]
                    model.config.sampling.predictor = 'ancestral'
                    model.config.sampling.steps = 2
                    model.trainer.global_rank, model.trainer.world_size = rank, world
                    local = data[rank::world]
                    expected_nfe = (2 * len(local) if kind == 'greedy'
                                    else sum(5 - len(row['input_ids']) for row in local))
                    expected_local = [sum(row['answer'] == '#### 42' for row in local),
                                      len(local), expected_nfe]
                    global_totals = [3, 5, 10 if kind == 'greedy' else 13]
                    reductions = []

                    def reduce(tensor, op):
                        reductions.append(tensor.tolist())
                        tensor.copy_(torch.tensor(global_totals, dtype=tensor.dtype))

                    def gather(out, error):
                        out[:] = [error] * world

                    with ExitStack() as stack:
                        for name, value in [('is_available', True), ('is_initialized', True),
                                            ('get_rank', rank), ('get_world_size', world)]:
                            stack.enter_context(patch.object(validation.dist, name, return_value=value))
                        stack.enter_context(patch.object(validation.dist, 'all_gather_object', side_effect=gather))
                        stack.enter_context(patch.object(validation.dist, 'all_reduce', side_effect=reduce))
                        grade = stack.enter_context(patch.object(
                            validation, 'grade_sample', side_effect=lambda sample, answer, *args, **kw: answer == '#### 42'))
                        result = (validation.sweep_greedy(model, data) if kind == 'greedy'
                                  else validation.sweep(model, data)[0])
                    self.assertEqual(reductions, [expected_local])
                    self.assertEqual(grade.call_count, len(local))
                    self.assertEqual(result['count'], 5)
                    self.assertEqual(result['correct'], 3)
                    self.assertEqual(result['accuracy'], .6)
                    self.assertEqual(result['nfe'], global_totals[-1] / 5)

    def test_remote_rank_failure_is_reported_before_metric_reduction(self):
        def gather(out, value):
            out[:] = [value, 'worker failed']
        with patch.object(validation.dist, 'is_available', return_value=True), \
                patch.object(validation.dist, 'is_initialized', return_value=True), \
                patch.object(validation.dist, 'get_world_size', return_value=2), \
                patch.object(validation.dist, 'all_gather_object', side_effect=gather):
            with self.assertRaisesRegex(RuntimeError, 'across ranks'):
                validation._raise_if_failed(None)

    def test_sample_logging_reuses_generation_and_grading(self):
        model = FixedModel()
        model.config.sampling.num_sample_log = 0
        with patch.object(validation, 'grade_sample', return_value=True):
            without_samples = validation.sweep(model, examples())
        seen = model.seen
        self.assertEqual(model._gsm8k_sample_records, [])
        model.seen = []
        model.config.sampling.num_sample_log = 2
        with patch.object(validation, 'grade_sample', return_value=True) as grade:
            with_samples = validation.sweep(model, examples())
        self.assertEqual(with_samples, without_samples)
        self.assertEqual(grade.call_count, 10)
        self.assertEqual(len(model.seen), len(seen))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.seen, seen)))
        self.assertEqual(len(model._gsm8k_sample_records), 2)
        self.assertTrue(all(row['threshold'] == .9 for row in model._gsm8k_sample_records))

    def test_publish_logs_greedy_and_pareto_keys_native_frontier_and_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FixedModel(directory)
            old = [{'threshold': .9, 'accuracy': .4, 'nfe': 15.}]
            curve = [{'threshold': .35, 'accuracy': .3, 'nfe': 4.5},
                     {'threshold': .9, 'accuracy': .6, 'nfe': 10.}]
            model._gsm8k_pareto_history = {0: old}
            model._gsm8k_greedy_result = {'accuracy': .31, 'nfe': 64., 'count': 1319}
            model._gsm8k_sample_records = [
                {'question': 'Q', 'generated': 'code', 'reference': '#### 42',
                 'grade': True, 'nfe': 10, 'threshold': .9}]
            image = np.zeros((2, 3, 3), dtype=np.uint8)
            with patch.object(validation, 'render_frontier', return_value=image) as render:
                validation.publish(model, curve)
            logged = {name: value for name, value, _ in model.logged}
            self.assertAlmostEqual(logged['val/gsm8k_acc_greedy'], .31)
            self.assertEqual(logged['val/gsm8k_nfe_greedy'], 64.)
            self.assertAlmostEqual(logged['val/gsm8k_acc_thr035'], .3)
            self.assertAlmostEqual(logged['pareto/gsm8k_acc_thr09'], .6)
            self.assertEqual(logged['val/gsm8k_nfe_thr09'], 10.)
            logger = model.trainer.logger
            self.assertIn(({'pareto/gsm8k_step_500': .3}, 450), logger.metrics)
            self.assertIn(({'pareto/gsm8k_step_500': .6}, 1000), logger.metrics)
            render.assert_called_once_with({0: old, 500: curve}, 500)
            self.assertIs(logger.images[0][1], image)
            self.assertEqual(logger.tables[0][0], 'val/samples')
            self.assertEqual(logger.tables[0][2], [['Q', 'code', '#### 42', True, 10, .9]])
            history = json.loads((Path(directory) / 'pareto_history.json').read_text())
            self.assertEqual(set(history['curves']), {'0', '500'})
            self.assertEqual(history['greedy'], model._gsm8k_greedy_result)

    def test_initial_metrics_use_step_zero_without_lightning_epoch_context(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FixedModel(directory)
            model.global_step = 0
            model._gsm8k_greedy_result = {'accuracy': .2, 'nfe': 64.}
            curve = [{'threshold': .325, 'accuracy': .1, 'nfe': 4.}]
            with patch.object(validation, 'render_frontier', return_value=np.zeros((2, 2, 3))):
                validation.publish(model, curve, initial=True)
            self.assertEqual(model.logged, [])
            metrics, step = model.trainer.logger.metrics[0]
            self.assertEqual(step, 0)
            self.assertEqual(metrics['val/gsm8k_acc_thr0325'], .1)
            self.assertEqual(metrics['val/gsm8k_acc_greedy'], .2)
            self.assertEqual(metrics['val/gsm8k_nfe_greedy'], 64.)

    def test_validation_hook_caches_data_runs_both_samplers_and_honors_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gsm8k.json'
            path.write_text('[]')
            model = FixedModel(directory)
            model.global_step = 0
            model.config.eval.gsm8k_every_n_validations = 2
            greedy = {'accuracy': .2, 'nfe': 64.}
            curve = [{'threshold': .9, 'accuracy': .5, 'nfe': 10.}]
            with patch.object(validation, 'resolve_data_path', return_value=path) as resolve, \
                    patch.object(validation, 'load_examples', return_value=examples()), \
                    patch.object(validation, 'sweep_greedy', return_value=greedy) as fixed, \
                    patch.object(validation, 'sweep', return_value=curve) as adaptive, \
                    patch.object(validation, 'publish') as publish:
                validation.log_validation_gsm8k(model, initial=True)
                self.assertIs(model._gsm8k_greedy_result, greedy)
                model.global_step = 500
                validation.log_validation_gsm8k(model)
                model.global_step = 1000
                validation.log_validation_gsm8k(model)
            self.assertEqual(resolve.call_count, 1)
            self.assertEqual(fixed.call_count, 2)
            self.assertEqual(adaptive.call_count, 2)
            self.assertEqual(publish.call_args_list[0].kwargs, {'initial': True})
            self.assertEqual(publish.call_args_list[1].kwargs, {'initial': False})

    def test_frontier_renderer_returns_nonempty_rgb_image(self):
        image = validation.render_frontier({0: [{'nfe': 10., 'accuracy': .2}],
                                             500: [{'nfe': 8., 'accuracy': .4}]}, 500)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[-1], 3)
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(image.shape[0], 100)
        self.assertGreater(float(image.std()), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
