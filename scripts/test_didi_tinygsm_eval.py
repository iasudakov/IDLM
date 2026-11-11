"""Offline CPU checks for conditional TinyGSM evaluation, including Gloo DDP.

Run: CUDA_VISIBLE_DEVICES='' python scripts/test_didi_tinygsm_eval.py
No model/data downloads, real program execution, or tracking services.
"""

from datetime import timedelta
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.multiprocessing as mp

from didi_repro import tinygsm_eval as evaluation
from test_eval_prediction_precision import Model as PrecisionModel


def config(**options):
    return OmegaConf.create({
        'eval': {'generate_samples': True, 'tinygsm_exec': True,
                 'gsm8k_at_validation': True, 'gsm8k_every_n_validations': 1,
                 'gsm8k_steps': [8, 16, 32], 'gsm8k_plot': False,
                 'gsm8k_pareto_fp32': False, **options},
        'sampling': {'steps': 16, 'num_sample_batches': 2, 'num_sample_log': 2, 'predictor': 'guided'},
        'loader': {'eval_batch_size': 2},
    })


class Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(value) for value in ids if value != 0)


class FakeModel(torch.nn.Module):
    _process_model_output = PrecisionModel._process_model_output
    neg_infinity = -1e6
    num_tokens = 6
    mask_index = 99
    device = torch.device('cpu')

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.ema = torch.nn.Linear(2, 2)
        self.ema.requires_grad_(False)
        self.ema.eval()
        self.tokenizer = Tokenizer()
        self.calls = []
        self.failure = None
        self.change_prefix = False
        self.epoch_flags = []

    def generate_samples(self, num_samples, num_steps, prefix_tokens, prefix_lengths):
        if self.failure:
            raise RuntimeError(self.failure)
        self.calls.append((num_steps, prefix_tokens.clone(), prefix_lengths.clone()))
        assert not self.training
        assert not torch.is_inference_mode_enabled()
        random.random()
        np.random.rand()
        torch.rand(1)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
            parameter.grad = torch.ones_like(parameter)
        self.train()
        samples = torch.zeros(num_samples, self.num_tokens, dtype=torch.long)
        for index, length in enumerate(prefix_lengths.tolist()):
            samples[index, :length] = prefix_tokens[index, :length]
            samples[index, length:length + 2] = 2
        if self.change_prefix:
            samples[0, 0] += 1
        return samples

    def on_validation_epoch_end(self):
        self.epoch_flags.append(self.config.eval.generate_samples)
        if self.failure:
            raise RuntimeError(self.failure)


class Grader:
    def __init__(self, failure=None):
        self.failure = failure
        self.tiny_calls = []
        self.benchmark_calls = []

    def evaluate_tinygsm_sample(self, generated, gold, timeout_s):
        if self.failure:
            raise RuntimeError(self.failure)
        self.tiny_calls.append((generated, gold, timeout_s))
        return generated == gold and gold != 'ungradable', gold != 'ungradable'

    def evaluate_samples(self, generated, gold, timeout_s):
        if self.failure:
            raise RuntimeError(self.failure)
        self.benchmark_calls.append((generated, gold, timeout_s))
        return generated == gold


class Logger:
    def __init__(self, failure=None):
        self.metrics, self.tables = [], []
        self.failure = failure

    def log_metrics(self, metrics, step=None):
        if self.failure:
            raise RuntimeError(self.failure)
        self.metrics.append((metrics, step))

    def log_table(self, key, columns, data, step=None):
        self.tables.append((key, columns, data, step))


class AdaptiveGenerator:
    def __init__(self, failure=None, mutation=None, fp32_failure=False):
        self.calls = []
        self.failure = failure
        self.mutation = mutation
        self.fp32_failure = fp32_failure

    def __call__(self, model, prefix_tokens, prefix_lengths, threshold, temperature,
                 max_steps, return_metadata):
        if self.failure:
            raise RuntimeError(self.failure)
        prediction = model._process_model_output(
            torch.ones(1, 1, 100, dtype=torch.bfloat16), torch.tensor([[99]]), None)
        if self.fp32_failure and prediction.dtype == torch.float32:
            raise RuntimeError('FP32 adaptive failure')
        self.calls.append({'threshold': threshold, 'temperature': temperature, 'max_steps': max_steps,
                           'dtype': str(prediction.dtype),
                           'randomness': (random.random(), np.random.rand(), torch.rand(1).item()),
                           'prefix_tokens': prefix_tokens.clone(), 'prefix_lengths': prefix_lengths.clone()})
        assert return_metadata
        samples = model.generate_samples(len(prefix_tokens), 999, prefix_tokens, prefix_lengths)
        per_row = torch.tensor([{7: 1, 8: 3, 4: 2}[int(row[0])] + int(threshold > .5)
                                for row in prefix_tokens], dtype=torch.float32)
        metadata = {'nfe': int(per_row.max()), 'nfe_per_row': per_row, 'complete': True,
                    'fallback_nfe': 0, 'max_steps': max_steps or model.num_tokens,
                    'weights': getattr(model, '_didi_evaluation_weights', 'backbone'),
                    'remasking_strategy': 'low_confidence_dynamic'}
        if self.mutation:
            self.mutation(samples, metadata)
        return samples, metadata


class ThresholdPublisher:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def __call__(self, trainer, model, curve, **kwargs):
        if self.failure:
            raise RuntimeError(self.failure)
        self.calls.append((curve, kwargs))


def trainer(logger=None, **overrides):
    logger = logger or Logger()
    return SimpleNamespace(sanity_checking=False, global_step=42, world_size=1,
                           logger=logger, loggers=[logger], **overrides)


def dataset():
    return [{'input_ids': [7], 'response_ground_truth': '2 2'},
            {'input_ids': [8, 9], 'response_ground_truth': '3'},
            {'input_ids': [4, 5, 6], 'response_ground_truth': '2 2'}]


def batch():
    return {'input_ids': torch.tensor([[7, 2, 2, 0, 0, 0], [8, 9, 2, 2, 0, 0]]),
            'prompt_len': torch.tensor([1, 2]), 'attention_mask': torch.ones(2, 6)}


def make_callback(cfg=None, scorer=None, data=None, **kwargs):
    return evaluation.DiDiTinyGSMEvaluation(
        cfg or config(), dataset_factory=lambda config, tokenizer: dataset() if data is None else data,
        grader_factory=lambda: scorer or Grader(), **kwargs)


def buffer(callback, host, model, value=None):
    callback.on_validation_epoch_start(host, model)
    callback.on_validation_batch_end(host, model, None, batch() if value is None else value, 0)


class TinyGSMEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_conditional_and_three_fixed_budgets_use_prefixes_counts_and_tables(self):
        scorer = Grader()
        callback = make_callback(scorer=scorer)
        model, host = FakeModel(), trainer()
        buffer(callback, host, model)
        callback.on_validation_epoch_end(host, model)
        metrics, step = host.logger.metrics[0]
        self.assertEqual(step, 42)
        self.assertEqual(metrics['val/answer_acc'], 1)
        self.assertEqual(metrics['val/answer_gradable_count'], 2)
        self.assertEqual(metrics['val/answer_exact_match'], 1)
        self.assertEqual(metrics['val/answer_count'], 2)
        for suffix in ('', '_s8', '_s16', '_s32'):
            self.assertEqual(metrics['val/gsm8k_acc' + suffix], 2 / 3)
            self.assertEqual(metrics['val/gsm8k_count' + suffix], 3)
        tables = {key: (columns, rows) for key, columns, rows, _ in host.logger.tables}
        self.assertEqual(tables['val/samples'][1], [['7', '2 2', '2 2'], ['8 9', '2 2', '2 2']])
        self.assertEqual([row[0] for row in tables['pareto/gsm8k_fixed_steps'][1]], [8, 16, 32])
        self.assertEqual(len(scorer.benchmark_calls), 9)
        self.assertEqual([call[0] for call in model.calls], [16, 16, 16, 8, 8, 32, 32])
        self.assertEqual(callback.batches, [])

    def test_rng_gradients_requires_grad_and_mixed_modes_are_restored(self):
        callback = make_callback()
        model, host = FakeModel(), trainer()
        for parameter in model.backbone.parameters():
            parameter.grad = torch.randn_like(parameter)
        modes = [(module, module.training) for module in model.modules()]
        params = [(parameter, parameter.grad, parameter.requires_grad, parameter.detach().clone())
                  for parameter in model.parameters()]
        buffer(callback, host, model)
        random.seed(13)
        np.random.seed(14)
        torch.manual_seed(15)
        before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with torch.inference_mode():
            callback.on_validation_epoch_end(host, model)
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        torch.testing.assert_close(torch.get_rng_state(), before[2], rtol=0, atol=0)
        self.assertEqual([(module, module.training) for module, _ in modes], modes)
        for parameter, gradient, enabled, value in params:
            self.assertIs(parameter.grad, gradient)
            self.assertEqual(parameter.requires_grad, enabled)
            torch.testing.assert_close(parameter, value)

    def test_answer_entropy_omits_prompt_and_counts_generated_tail(self):
        model, host = FakeModel(), trainer()
        callback = make_callback(config(gsm8k_at_validation=False, tinygsm_exec=False))
        buffer(callback, host, model)
        callback.on_validation_epoch_end(host, model)
        expected = (evaluation._answer_entropy(torch.tensor([2, 2, 0, 0, 0])) +
                    evaluation._answer_entropy(torch.tensor([2, 2, 0, 0]))) / 2
        self.assertAlmostEqual(host.logger.metrics[0][0]['val/sample_entropy'], expected)
        self.assertNotIn('val/answer_acc', host.logger.metrics[0][0])

    def test_no_gradable_references_does_not_log_fake_zero_accuracy(self):
        callback = make_callback()
        result = callback._conditional_metrics([
            {'generated': '2', 'gold': 'ungradable', 'entropy': 1.},
        ])
        self.assertEqual(result['val/answer_gradable'], 0)
        self.assertNotIn('val/answer_acc', result)

    def test_rank_zero_only_style_execution_exception_propagates_and_restores(self):
        callback = make_callback(scorer=Grader(failure='scorer exploded'))
        model, host = FakeModel(), trainer()
        buffer(callback, host, model)
        before = [(module, module.training) for module in model.modules()]
        with self.assertRaisesRegex(RuntimeError, 'conditional grading.*scorer exploded'):
            callback.on_validation_epoch_end(host, model)
        self.assertEqual([(module, module.training) for module, _ in before], before)
        self.assertEqual(host.logger.metrics, [])
        self.assertEqual(callback.batches, [])

    def test_generation_logger_and_prefix_failures_are_not_silent_zero(self):
        for kind in ('generation', 'logger', 'prefix'):
            with self.subTest(kind=kind):
                callback = make_callback()
                model, host = FakeModel(), trainer()
                if kind == 'generation':
                    model.failure = 'generation broke'
                elif kind == 'logger':
                    host.logger.failure = 'logger broke'
                else:
                    model.change_prefix = True
                buffer(callback, host, model)
                with self.assertRaises(RuntimeError):
                    callback.on_validation_epoch_end(host, model)
                self.assertEqual(callback.batches, [])

    def test_missing_explicit_boundary_does_not_guess_from_loss_mask(self):
        callback = make_callback()
        model, host = FakeModel(), trainer()
        value = batch()
        del value['prompt_len']
        buffer(callback, host, model, value)
        with self.assertRaisesRegex(RuntimeError, 'explicit prompt_len'):
            callback.on_validation_epoch_end(host, model)

    def test_empty_dataset_and_no_validation_rows_fail(self):
        for callback in (make_callback(data=[]), make_callback(config(gsm8k_at_validation=False))):
            model, host = FakeModel(), trainer()
            if callback.benchmark:
                buffer(callback, host, model)
            with self.assertRaisesRegex(RuntimeError, 'empty|No conditional validation rows'):
                callback.on_validation_epoch_end(host, model)

    def test_small_dataset_limit_tail_and_cached_dataset(self):
        callback = make_callback(config(generate_samples=False, gsm8k_num_examples=1, gsm8k_steps=[8]))
        with mock.patch.object(callback, 'dataset_factory', wraps=callback.dataset_factory) as factory:
            model, host = FakeModel(), trainer()
            callback.on_validation_epoch_end(host, model)
            callback.on_validation_epoch_end(host, model)
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(all(len(prefix) == 1 for _, prefix, _ in model.calls))
        self.assertEqual(host.logger.metrics[0][0]['val/gsm8k_count'], 1)

    def test_sanity_and_cadence_do_not_execute_or_advance_wrongly(self):
        callback = make_callback(config(generate_samples=False, gsm8k_every_n_validations=2))
        model, host = FakeModel(), trainer()
        host.sanity_checking = True
        callback.on_validation_epoch_end(host, model)
        self.assertEqual(callback.validations_seen, 0)
        host.sanity_checking = False
        callback.on_validation_epoch_end(host, model)
        callback.on_validation_epoch_end(host, model)
        self.assertEqual(len(host.logger.metrics), 1)
        state = callback.state_dict()
        resumed = make_callback(config(generate_samples=False, gsm8k_every_n_validations=2))
        resumed.load_state_dict(state)
        resumed.on_validation_epoch_end(host, model)
        self.assertEqual(len(host.logger.metrics), 2)

    def test_buffer_limit_and_copy_ownership(self):
        callback = make_callback()
        host, model = trainer(), FakeModel()
        value = batch()
        for index in range(4):
            callback.on_validation_batch_end(host, model, None, value, index)
        value['input_ids'].fill_(99)
        self.assertEqual(len(callback.batches), 2)
        self.assertEqual(callback.batches[0]['input_ids'][0, 0], 7)

    def test_resume_evaluation_step_overrides_trainer_zero(self):
        callback = make_callback(config(generate_samples=False))
        model, host = FakeModel(), trainer()
        model._didi_evaluation_step = {'value': 1000}
        callback.on_validation_epoch_end(host, model)
        self.assertEqual(host.logger.metrics[0][1], 1000)
        self.assertTrue(all(row[-1] == 1000 for row in host.logger.tables))

    def test_attachment_only_disables_original_generation_and_restores_on_error(self):
        model, cfg = FakeModel(), config()
        model.config = cfg
        callback = evaluation.attach_tinygsm_evaluation(model, cfg)
        self.assertTrue(callback.conditional)
        model.on_validation_epoch_end()
        self.assertEqual(model.epoch_flags, [False])
        self.assertTrue(cfg.eval.generate_samples)
        model.failure = 'original hook failed'
        with self.assertRaisesRegex(RuntimeError, 'original hook failed'):
            model.on_validation_epoch_end()
        self.assertTrue(cfg.eval.generate_samples)
        with self.assertRaisesRegex(RuntimeError, 'already attached'):
            evaluation.attach_tinygsm_evaluation(model, cfg)

    def test_config_and_missing_distributed_group_rejected(self):
        for name, value in (('gsm8k_steps', [0]), ('gsm8k_steps', [True]), ('gsm8k_steps', '8,16'),
                            ('gsm8k_batch_size', 0), ('gsm8k_num_examples', 0),
                            ('tinygsm_exec_timeout', float('nan')), ('gsm8k_every_n_validations', 0)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                make_callback(config(**{name: value}))
        host = trainer()
        host.world_size = 2
        with self.assertRaisesRegex(RuntimeError, 'initialized distributed process group'):
            make_callback().on_validation_epoch_end(host, FakeModel())

    def test_gathered_index_duplicates_rejected(self):
        callback = make_callback()
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            callback._benchmark_metrics([{'index': 0}, {'index': 0}], 2, 8)


class TinyGSMThresholdTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def callback(self, generator=None, publisher=None, **options):
        options = {'generate_samples': False, 'gsm8k_steps': [8],
                   'gsm8k_pareto_thresholds': [.75, .25], **options}
        return make_callback(config(**options), adaptive_generator=generator or AdaptiveGenerator(),
                             threshold_publisher=publisher or ThresholdPublisher())

    def test_additional_fp32_sweep_has_separate_publication_and_paired_rng(self):
        generator, publisher = AdaptiveGenerator(), ThresholdPublisher()
        callback = self.callback(generator, publisher, gsm8k_pareto_fp32=True)
        model, host = FakeModel(), trainer()
        callback.on_validation_epoch_end(host, model)
        self.assertEqual(len(publisher.calls), 2)
        self.assertNotIn('suffix', publisher.calls[0][1])
        self.assertEqual(publisher.calls[1][1]['suffix'], '_fp32norm')
        self.assertEqual([call['dtype'] for call in generator.calls],
                         ['torch.bfloat16'] * 4 + ['torch.float32'] * 4)
        self.assertEqual([call['randomness'] for call in generator.calls[:4]],
                         [call['randomness'] for call in generator.calls[4:]])
        self.assertNotIn('_process_model_output', vars(model))
        self.assertEqual(len(host.logger.metrics), 1)  # Primary fixed-budget metrics unchanged.

    def test_fp32_failure_restores_normalizer_and_rng(self):
        callback = self.callback(AdaptiveGenerator(fp32_failure=True), gsm8k_pareto_fp32=True)
        model = FakeModel()
        before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'FP32 adaptive failure'):
            callback.on_validation_epoch_end(trainer(), model)
        self.assertNotIn('_process_model_output', vars(model))
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        torch.testing.assert_close(torch.get_rng_state(), before[2], rtol=0, atol=0)

    def test_fp32_flag_requires_boolean(self):
        for value in ('false', 1, None):
            with self.assertRaisesRegex(ValueError, 'gsm8k_pareto_fp32'):
                self.callback(gsm8k_pareto_fp32=value)

    def test_threshold_sweep_adds_weighted_actual_nfe_without_changing_fixed_budgets(self):
        generator, publisher = AdaptiveGenerator(), ThresholdPublisher()
        callback = self.callback(generator, publisher, gsm8k_pareto_temperature=.3,
                                 gsm8k_pareto_reference='/optional/reference.json', gsm8k_pareto_plot=True)
        model, host = FakeModel(), trainer()
        callback.on_validation_epoch_end(host, model)
        curve, kwargs = publisher.calls[0]
        self.assertEqual([point['threshold'] for point in curve], [.25, .75])
        self.assertEqual([point['nfe'] for point in curve], [2., 3.])
        self.assertEqual([point['total_batch_forwards'] for point in curve], [5, 7])
        self.assertEqual([point['total_example_forwards'] for point in curve], [6, 9])
        self.assertEqual([point['physical_row_forwards'] for point in curve], [8, 11])
        self.assertTrue(all(point['generation_batches'] == 2 and point['count'] == 3 for point in curve))
        self.assertTrue(all(point['accuracy'] == 2 / 3 for point in curve))
        self.assertEqual(kwargs, {'reference': '/optional/reference.json', 'plot': True})
        self.assertEqual([call[0] for call in model.calls[:4]], [16, 16, 8, 8])
        self.assertEqual([call['threshold'] for call in generator.calls], [.25, .25, .75, .75])
        self.assertTrue(all(call['temperature'] == .3 for call in generator.calls))
        metrics = host.logger.metrics[0][0]
        self.assertEqual(metrics['val/gsm8k_acc'], 2 / 3)
        self.assertFalse(any('thr' in key for key in metrics))  # Dedicated publisher owns threshold series.
        protocol = dict(next(rows for key, _, rows, _ in host.logger.tables if key == 'didi/tinygsm_protocol'))
        self.assertIn('weighted by global example count', json.loads(protocol['threshold_sweep'])['nfe'])

    def test_disabled_thresholds_never_call_adaptive_sampler_or_publisher(self):
        generator, publisher = AdaptiveGenerator(failure='must not run'), ThresholdPublisher(failure='must not run')
        for disabled in (None, []):
            self.callback(generator, publisher, gsm8k_pareto_thresholds=disabled).on_validation_epoch_end(trainer(), FakeModel())
            self.callback(generator, publisher, gsm8k_pareto_thresholds=disabled, gsm8k_at_validation=False)
        self.assertFalse(generator.calls)
        self.assertFalse(publisher.calls)

    def test_threshold_validation_rejects_bad_lists_ranges_and_disabled_benchmark(self):
        for value in (.5, '', '0.5', {}, [False], [True], [float('nan')], [float('inf')],
                      [-.01], [1.01], [.5, .5], ['0.5'], [None]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.callback(gsm8k_pareto_thresholds=value)
        with self.assertRaisesRegex(ValueError, 'gsm8k_at_validation=true'):
            self.callback(gsm8k_at_validation=False)
        valid = self.callback(gsm8k_pareto_thresholds=[1., 0.])
        self.assertEqual(valid.thresholds, [0., 1.])

    def test_temperature_and_cap_validation_and_forwarding(self):
        for temperature in (False, -.01, float('nan'), float('inf'), '0'):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                self.callback(gsm8k_pareto_temperature=temperature)
        for cap in (False, 0, -1, 1.5, 2.0, '2', float('nan'), float('inf')):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.callback(gsm8k_pareto_max_steps=cap)
        generator, publisher = AdaptiveGenerator(), ThresholdPublisher()
        self.callback(generator, publisher, gsm8k_pareto_max_steps=2).on_validation_epoch_end(trainer(), FakeModel())
        self.assertTrue(all(call['max_steps'] == 2 for call in generator.calls))
        self.assertTrue(all(point['max_steps'] == 2 for point in publisher.calls[0][0]))

    def test_threshold_cadence_follows_primary_benchmark_and_skips_sanity(self):
        publisher = ThresholdPublisher()
        callback = self.callback(publisher=publisher, gsm8k_every_n_validations=2)
        host, model = trainer(), FakeModel()
        host.sanity_checking = True
        callback.on_validation_epoch_end(host, model)
        host.sanity_checking = False
        for _ in range(3):
            callback.on_validation_epoch_end(host, model)
        self.assertEqual(len(publisher.calls), 2)

    def test_invalid_adaptive_metadata_incomplete_or_changed_prefix_prevents_publishing(self):
        mutations = {
            'incomplete': lambda samples, meta: meta.update(complete=False),
            'residual_mask': lambda samples, meta: samples.__setitem__((0, -1), 99),
            'changed_prefix': lambda samples, meta: samples.__setitem__((0, 0), 11),
            'row_shape': lambda samples, meta: meta.update(nfe_per_row=torch.tensor([1, 2, 3])),
            'row_nan': lambda samples, meta: meta.update(nfe_per_row=torch.tensor([float('nan'), 1.])),
            'row_bool': lambda samples, meta: meta.update(nfe_per_row=torch.tensor([True, True])),
            'row_fraction': lambda samples, meta: meta.update(nfe_per_row=torch.tensor([1.5, 2.])),
            'row_gt_batch': lambda samples, meta: meta.update(nfe_per_row=torch.tensor([100., 1.])),
            'invalid_batch_nfe': lambda samples, meta: meta.update(nfe=0),
            'invalid_fallback': lambda samples, meta: meta.update(fallback_nfe=2),
            'wrong_sampler': lambda samples, meta: meta.update(remasking_strategy='guided'),
            'wrong_weights': lambda samples, meta: meta.update(weights='teacher'),
            'physical_forwards': lambda samples, meta: meta.update(physical_row_forwards=999),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                publisher = ThresholdPublisher()
                callback = self.callback(AdaptiveGenerator(mutation=mutation), publisher)
                with self.assertRaisesRegex(RuntimeError, 'threshold generation'):
                    callback.on_validation_epoch_end(trainer(), FakeModel())
                self.assertEqual(publisher.calls, [])

    def test_generator_and_publisher_failure_restore_rng_modes_and_gradient_references(self):
        for failure in ('generate', 'publish', None):
            with self.subTest(failure=failure):
                generator = AdaptiveGenerator(failure='adaptive failure' if failure == 'generate' else None)
                publisher = ThresholdPublisher(failure='publisher failure' if failure == 'publish' else None)
                callback = self.callback(generator, publisher)
                model, host = FakeModel(), trainer()
                for parameter in model.backbone.parameters():
                    parameter.grad = torch.ones_like(parameter)
                parameters = [(parameter, parameter.grad, parameter.requires_grad) for parameter in model.parameters()]
                modes = [(module, module.training) for module in model.modules()]
                before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
                with torch.inference_mode():
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, 'failure'):
                            callback.on_validation_epoch_end(host, model)
                    else:
                        callback.on_validation_epoch_end(host, model)
                self.assertEqual(random.getstate(), before[0])
                np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
                torch.testing.assert_close(torch.get_rng_state(), before[2], rtol=0, atol=0)
                for parameter, gradient, required in parameters:
                    self.assertIs(parameter.grad, gradient)
                    self.assertEqual(parameter.requires_grad, required)
                self.assertEqual([(module, module.training) for module in model.modules()], modes)

    def test_fallback_counts_remain_separate_from_per_sequence_nfe(self):
        generator = AdaptiveGenerator(mutation=lambda samples, meta: meta.update(fallback_nfe=1))
        publisher = ThresholdPublisher()
        self.callback(generator, publisher).on_validation_epoch_end(trainer(), FakeModel())
        curve = publisher.calls[0][0]
        self.assertEqual(curve[0]['fallback_batches'], 2)
        self.assertEqual(curve[0]['nfe'], 2)


def _ddp_worker(rank, init_file, result_dir, failure):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                                        world_size=2, timeout=timedelta(seconds=40))
    scorer = Grader(failure='rank-zero scorer failure' if failure and rank == 0 else None)
    callback = make_callback(config(generate_samples=False, gsm8k_steps=[8]), scorer=scorer, data=dataset()[:1])
    model, host = FakeModel(), trainer()
    host.world_size = 2
    error = None
    try:
        callback.on_validation_epoch_end(host, model)
    except RuntimeError as exception:
        error = str(exception)
    finally:
        result = {'rank': rank, 'error': error, 'generation_calls': len(model.calls),
                  'grading_calls': len(scorer.benchmark_calls), 'metrics': host.logger.metrics}
        Path(result_dir, f'{rank}.json').write_text(json.dumps(result))
        torch.distributed.destroy_process_group()


def _threshold_ddp_worker(rank, init_file, result_dir, scenario):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                                        world_size=2, timeout=timedelta(seconds=40))
    generator = AdaptiveGenerator(failure='rank-one adaptive failure' if scenario == 'failure' and rank == 1 else None,
                                  fp32_failure=scenario == 'fp32_failure' and rank == 1)
    publisher = ThresholdPublisher()
    data = dataset()[:1] if scenario == 'empty_rank' else dataset()
    callback = make_callback(config(generate_samples=False, gsm8k_steps=None, gsm8k_pareto_thresholds=[.25],
                                    gsm8k_pareto_fp32=True),
                             data=data, adaptive_generator=generator, threshold_publisher=publisher)
    host, model = trainer(), FakeModel()
    host.world_size = 2
    error = None
    try:
        callback.on_validation_epoch_end(host, model)
    except RuntimeError as exception:
        error = str(exception)
    finally:
        result = {'error': error, 'adaptive_calls': len(generator.calls),
                  'published': publisher.calls, 'metrics': host.logger.metrics}
        Path(result_dir, f'{rank}.json').write_text(json.dumps(result))
        torch.distributed.destroy_process_group()


class TinyGSMDistributedTests(unittest.TestCase):
    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), 'requires Gloo')
    def test_one_example_two_ranks_and_synchronized_rank_zero_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix='didi-tinygsm-gloo-') as directory:
                mp.spawn(_ddp_worker, args=(str(Path(directory, 'init')), directory, failure), nprocs=2, join=True)
                first, second = [json.loads(Path(directory, f'{rank}.json').read_text()) for rank in range(2)]
                self.assertEqual(second['generation_calls'], 0)
                self.assertEqual(second['grading_calls'], 0)
                self.assertEqual(second['metrics'], [])
                if failure:
                    self.assertIn('rank-zero scorer failure', first['error'])
                    self.assertEqual(first['error'], second['error'])
                    self.assertEqual(first['metrics'], [])
                else:
                    self.assertIsNone(first['error'])
                    self.assertIsNone(second['error'])
                    self.assertEqual(first['generation_calls'], 2)
                    self.assertEqual(first['metrics'][0][0]['val/gsm8k_count'], 1)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), 'requires Gloo')
    def test_threshold_global_weighting_empty_rank_and_synchronized_generation_failure(self):
        for scenario in ('uneven', 'empty_rank', 'failure', 'fp32_failure'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory(prefix='didi-threshold-gloo-') as directory:
                mp.spawn(_threshold_ddp_worker, args=(str(Path(directory, 'init')), directory, scenario), nprocs=2, join=True)
                first, second = [json.loads(Path(directory, f'{rank}.json').read_text()) for rank in range(2)]
                self.assertEqual(second['published'], [])
                if scenario in ('failure', 'fp32_failure'):
                    self.assertIn('rank-one adaptive failure' if scenario == 'failure' else 'FP32 adaptive failure',
                                  first['error'])
                    self.assertEqual(first['error'], second['error'])
                    self.assertEqual(first['published'], [])
                    self.assertEqual(first['metrics'], [])
                else:
                    self.assertIsNone(first['error'])
                    self.assertIsNone(second['error'])
                    self.assertEqual(len(first['published']), 2)
                    self.assertEqual(first['published'][1][1]['suffix'], '_fp32norm')
                    point = first['published'][0][0][0]
                    if scenario == 'empty_rank':
                        self.assertEqual(second['adaptive_calls'], 0)
                        self.assertEqual(point['nfe'], 1.)
                        self.assertEqual(point['count'], 1)
                    else:
                        self.assertEqual(point['nfe'], 2.)  # Not unweighted rank mean (1.5 + 3) / 2.
                        self.assertEqual(point['total_batch_forwards'], 5)
                        self.assertEqual(point['total_example_forwards'], 6)
                        self.assertEqual(point['count'], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
