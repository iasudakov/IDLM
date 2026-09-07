"""Offline DiDi Pareto tests: actual tiny upstream samplers, local evaluators.

    conda run -n mdlm python scripts/test_didi_pareto.py
    conda run --no-capture-output -n mdlm torchrun --standalone --nproc_per_node=2 \
        scripts/test_didi_pareto.py --ddp-smoke
"""

import copy
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
from omegaconf import OmegaConf, open_dict
import torch

from didi_repro import pareto
from didi_repro.monitor import _root_metrics_class
import test_didi_runtime as runtime_tests


class Logger:
    version = 'test-run-a'

    def __init__(self):
        self.metrics, self.tables, self.images = [], [], []

    def log_metrics(self, metrics, step=None):
        self.metrics.append((dict(metrics), step))

    def log_table(self, key, columns, data, step=None):
        self.tables.append((key, columns, data, step))

    def log_image(self, key, image, step=None):
        self.images.append((key, image.shape, step))


class MeanMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sync_on_compute = True
        self.reset()

    def reset(self):
        self.values = []

    def compute(self):
        return torch.tensor(sum(self.values) / len(self.values))


class Reference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.devices = []

    def to(self, device):
        self.devices.append(str(device))
        return self


class Evaluator:
    def __init__(self, **kwargs):
        self.gen_ppl = MeanMetric()
        self._eval_model = Reference()
        self.batch_counts = []

    def to(self, device):
        return self

    def record_generative_perplexity(self, texts, **kwargs):
        assert not torch.is_grad_enabled()
        self.batch_counts.append(len(texts))
        self.gen_ppl.values.extend(sum(map(int, text.split())) + 1 for text in texts)


class ReferenceTokenizer:
    pad_token = 'pad'
    pad_token_id = 0
    eos_token_id = 8

    def __call__(self, texts, **kwargs):
        ids = torch.tensor([list(map(int, text.split())) for text in texts])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


class UniformLM(torch.nn.Module):
    def forward(self, tokens, attention_mask):
        return (torch.zeros(*tokens.shape, 9),)


def uniform_root_factory(**kwargs):
    """Real TorchMetrics-backed evaluator, with only its LM/tokenizer mocked."""
    root_class = _root_metrics_class()
    with mock.patch('transformers.AutoTokenizer.from_pretrained', return_value=ReferenceTokenizer()):
        evaluator = root_class(**kwargs)
    evaluator._eval_model = UniformLM()
    return evaluator


class ParetoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime_tests.DiDiRuntimeTests.setUpClass()
        cls.helpers = runtime_tests.DiDiRuntimeTests()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='didi-pareto-test-')
        self.addCleanup(self.directory.cleanup)

    def model(self, sampler='ancestral_cache', **overrides):
        cfg = self.helpers.config(tiny=True)
        cfg.sampling.predictor = sampler
        cfg.eval.perplexity_batch_size = 3
        cfg.checkpointing.save_dir = self.directory.name
        with open_dict(cfg):
            cfg.didi_pareto = {
                'every_n_validations': 1, 'temperatures': [.75, 1.0],
                'steps': [2, 4], 'seeds': 2, 'num_batches': 2,
                'batch_size': 2, 'keep_samples': 2, 'reference': None, 'sampler': None,
                **overrides,
            }
        model = self.helpers.model(cfg)
        self.helpers.initialize(model)
        return model

    def trainer(self, step=123):
        return SimpleNamespace(sanity_checking=False, global_step=step,
                               logger=Logger(), is_global_zero=True)

    def callback(self, model):
        callback = pareto.attach_pareto(model, model.config)
        callback.metrics_factory = Evaluator
        return callback

    def fast_plots(self):
        return mock.patch.multiple('owt_pareto_plot',
            render_temperature=mock.Mock(return_value=np.zeros((2, 3, 3), dtype=np.uint8)),
            render_checkpoints=mock.Mock(return_value=np.zeros((2, 3, 3), dtype=np.uint8)))

    def history(self):
        return json.loads((Path(self.directory.name) / 'didi_pareto_temp_history.json').read_text())

    def test_disabled_and_strict_preflight(self):
        disabled = OmegaConf.create({'didi_pareto': {'every_n_validations': 0}})
        self.assertIsNone(pareto.attach_pareto(None, disabled))
        pareto.validate_options(disabled)
        model = self.model()
        for key, value in [('seeds', 1.5), ('num_batches', 0), ('batch_size', 2.5),
                           ('steps', [2.5]), ('temperatures', [0]), ('temperatures', [float('nan')]),
                           ('every_n_validations', -.5), ('keep_samples', -1), ('sampler', 'analytic')]:
            cfg = copy.deepcopy(model.config)
            cfg.didi_pareto[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                pareto.validate_options(cfg)
        cfg = copy.deepcopy(model.config)
        cfg.trainer.limit_val_batches = 0
        with self.assertRaisesRegex(ValueError, 'limit_val_batches'):
            pareto.validate_options(cfg)
        cfg.trainer.limit_val_batches = 1
        cfg.didi_pareto.sampler = 'guided'
        cfg.didi_pareto.steps = [1]
        with self.assertRaisesRegex(ValueError, 'at least two'):
            pareto.validate_options(cfg)

    def test_real_sampling_counts_paired_seeds_and_tracker_keys(self):
        model = self.model()
        callback, trainer = self.callback(model), self.trainer()
        seed_calls = []
        original = model.generate_samples

        def record(*args, **kwargs):
            seed_calls.append(torch.initial_seed())
            return original(*args, **kwargs)

        model.metrics.reset = mock.Mock(side_effect=AssertionError('upstream metrics must not be reset'))
        with mock.patch.object(model, 'generate_samples', side_effect=record), self.fast_plots():
            callback.on_validation_end(trainer, model)
        self.assertEqual(len(seed_calls), 2 * 2 * 2 * 2)
        self.assertEqual(seed_calls[:8], seed_calls[8:])
        self.assertEqual(seed_calls[:4], [9973 + 2, 9973 + 2, 2 * 9973 + 2, 2 * 9973 + 2])
        self.assertEqual(callback.evaluator.batch_counts, [3, 1] * 8)
        self.assertFalse(callback.evaluator.gen_ppl.sync_on_compute)
        self.assertFalse(callback.evaluator.gen_ppl._to_sync)
        model.metrics.reset.assert_not_called()
        history = self.history()
        self.assertEqual(history['protocol']['samples_per_seed'], 4)
        for curve in history['curves']['123'].values():
            for point in curve:
                self.assertEqual(point['total_samples'], 8)
                self.assertEqual(point['seeds'], 2)
                self.assertGreaterEqual(point['gen_ppl_std'], 0)
        values = next(values for values, _ in trainer.logger.metrics if 'val/gen_ppl_s2_t1' in values)
        self.assertIn('val/entropy_s4_t0.75', values)
        self.assertIn('val/gen_ppl_std_s2_t1', values)
        self.assertTrue(all(step == 123 for _, step in trainer.logger.metrics))
        self.assertTrue(all(step == 123 for _, _, _, step in trainer.logger.tables))
        self.assertIn('pareto/temperature_values', [key for key, _, _, _ in trainer.logger.tables])
        self.assertIn('pareto/temperature_frontier', [key for key, _, _ in trainer.logger.images])
        self.assertEqual(callback.evaluator._eval_model.devices[-1], 'cpu')

    def test_state_rng_and_parameter_gradients_preserved_for_guided_sampler(self):
        model = self.model(sampler='guided', temperatures=[.75, 1.0], seeds=1, num_batches=1, steps=[2])
        model.teacher.eval()
        for parameter in model.backbone.parameters():
            parameter.grad = torch.full_like(parameter, .25)
        modes = [module.training for module in model.modules()]
        gradients = [parameter.grad for parameter in model.parameters()]
        weights = copy.deepcopy(model.state_dict())
        torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        callback = self.callback(model)
        with self.fast_plots(), torch.inference_mode():
            callback.on_validation_end(self.trainer(), model)
        self.helpers.assert_same(model.state_dict(), weights)
        self.assertEqual(modes, [module.training for module in model.modules()])
        for before, parameter in zip(gradients, model.parameters()):
            self.assertIs(before, parameter.grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        self.assertEqual(model.sampler, 'guided')
        self.assertNotIn('forward', model.__dict__)

    def test_cadence_resume_state_and_baseline_step(self):
        model = self.model(every_n_validations=3, temperatures=[1], steps=[2], seeds=1, num_batches=1)
        callback, trainer = self.callback(model), self.trainer(step=0)
        trainer.sanity_checking = True
        callback.on_validation_end(trainer, model)
        self.assertEqual(callback.validations_seen, 0)
        trainer.sanity_checking = False
        model._didi_evaluation_step = {'value': 200}
        with self.fast_plots():
            callback.on_validation_end(trainer, model)
            self.assertIn('200', self.history()['curves'])
            self.assertEqual(callback.validations_seen, 0)
            del model._didi_evaluation_step
            trainer.global_step = 250
            callback.on_validation_end(trainer, model)  # first regular validation
            trainer.global_step = 300
            callback.on_validation_end(trainer, model)  # skipped
            state = callback.state_dict()
            resumed = self.callback(model)
            resumed.load_state_dict(state)
            trainer.global_step = 350
            resumed.on_validation_end(trainer, model)  # skipped
            trainer.global_step = 400
            resumed.on_validation_end(trainer, model)  # fourth regular validation
        self.assertEqual(set(self.history()['curves']), {'200', '250', '400'})
        self.assertEqual(resumed.validations_seen, 4)
        self.assertIn('pareto/temperature_by_checkpoint', [key for key, _, _ in trainer.logger.images])

    def test_incompatible_run_or_protocol_history_is_archived_not_stitched(self):
        model = self.model(temperatures=[1], steps=[2], seeds=1, num_batches=1)
        trainer = self.trainer()
        with self.fast_plots():
            self.callback(model).on_validation_end(trainer, model)
            trainer.logger.version = 'test-run-b'
            trainer.global_step = 200
            self.callback(model).on_validation_end(trainer, model)
            self.assertEqual(set(self.history()['curves']), {'200'})
            self.assertEqual(len(list(Path(self.directory.name).glob('*.previous.*.json'))), 1)
            trainer.global_step = 300
            model.config.didi_pareto.seeds = 2
            self.callback(model).on_validation_end(trainer, model)
        self.assertEqual(set(self.history()['curves']), {'300'})
        self.assertEqual(len(list(Path(self.directory.name).glob('*.previous.*.json'))), 2)

    def test_generation_failure_restores_modes_sampler_forward_and_gradients(self):
        model = self.model(sampler='ancestral_cache')
        model.config.didi_pareto.sampler = 'guided'
        callback, trainer = self.callback(model), self.trainer()
        parameter = next(model.backbone.parameters())
        parameter.grad = torch.full_like(parameter, 7)
        original_gradient = parameter.grad
        flags = [module.training for module in model.modules()]
        rng = torch.get_rng_state()

        def fail(*args, **kwargs):
            parameter.grad = torch.zeros_like(parameter)
            model.train()
            torch.rand(4)
            raise RuntimeError('injected generation failure')

        with mock.patch.object(model, 'generate_samples', side_effect=fail):
            callback.on_validation_end(trainer, model)
        self.assertEqual(trainer.logger.metrics[-1][0]['didi_pareto/failed'], 1)
        self.assertEqual(flags, [module.training for module in model.modules()])
        self.assertEqual(model.sampler, 'ancestral_cache')
        self.assertIs(parameter.grad, original_gradient)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertNotIn('forward', model.__dict__)
        self.assertFalse((Path(self.directory.name) / 'didi_pareto_temp_history.json').exists())

    def test_evaluator_failure_reports_failure_and_offloads(self):
        model = self.model(temperatures=[1], steps=[2], seeds=1, num_batches=1)
        callback, trainer = self.callback(model), self.trainer()
        evaluator = Evaluator()
        evaluator.record_generative_perplexity = mock.Mock(side_effect=RuntimeError('reference unavailable'))
        callback.metrics_factory = lambda **kwargs: evaluator
        callback.on_validation_end(trainer, model)
        self.assertEqual(trainer.logger.metrics[-1][0]['didi_pareto/failed'], 1)
        self.assertEqual(evaluator._eval_model.devices[-1], 'cpu')

    def test_evaluator_transfer_failure_still_attempts_offload(self):
        model = self.model(temperatures=[1], steps=[2], seeds=1, num_batches=1)
        callback, trainer = self.callback(model), self.trainer()
        evaluator = Evaluator()
        evaluator._eval_model.to = mock.Mock(side_effect=[RuntimeError('transfer failed'), evaluator._eval_model])
        callback.metrics_factory = lambda **kwargs: evaluator
        callback.on_validation_end(trainer, model)
        self.assertEqual(trainer.logger.metrics[-1][0]['didi_pareto/failed'], 1)
        self.assertEqual(evaluator._eval_model.to.call_args_list[-1], mock.call('cpu'))
        self.assertEqual(evaluator._eval_model.to.call_count, 2)

    def test_distinct_close_temperatures_keep_distinct_curve_and_metric_keys(self):
        model = self.model(temperatures=[1, 1.000001], steps=[2], seeds=1, num_batches=1)
        callback, trainer = self.callback(model), self.trainer()
        with self.fast_plots():
            callback.on_validation_end(trainer, model)
        self.assertEqual(set(self.history()['curves']['123']), {'1', '1.000001'})
        values = next(data for data, _ in trainer.logger.metrics if 'val/gen_ppl_s2_t1' in data)
        self.assertIn('val/gen_ppl_s2_t1.000001', values)

    def test_real_root_evaluator_and_real_plots_with_tiny_reference(self):
        model = self.model(temperatures=[.75, 1], steps=[2], seeds=1, num_batches=1)
        callback, trainer = self.callback(model), self.trainer()

        callback.metrics_factory = uniform_root_factory
        callback.on_validation_end(trainer, model)
        for point in self.history()['curves']['123'].values():
            self.assertAlmostEqual(point[0]['gen_ppl'], 9.0, places=5)
        image = next(shape for key, shape, _ in trainer.logger.images if key == 'pareto/temperature_frontier')
        self.assertEqual(len(image), 3)
        self.assertEqual(image[-1], 3)

    def test_sample_standard_deviation(self):
        self.assertEqual(pareto._mean_std([5]), (5, 0))
        self.assertEqual(pareto._mean_std([1, 3, 5]), (3, 2))


def ddp_smoke():
    """Test success and rank-local generation/evaluator failures without GPUs."""
    torch.distributed.init_process_group('gloo', timeout=timedelta(seconds=90))
    ParetoTests.setUpClass()
    case = ParetoTests()
    case.setUp()
    try:
        rank = torch.distributed.get_rank()
        torch.manual_seed(55)
        model = case.model(temperatures=[.75, 1], steps=[2], seeds=1, num_batches=1)
        trainer = case.trainer()
        trainer.is_global_zero = rank == 0
        callback = case.callback(model)
        callback.metrics_factory = uniform_root_factory
        seeds = []
        original = model.generate_samples

        def capture(*args, **kwargs):
            seeds.append(torch.initial_seed())
            return original(*args, **kwargs)

        with mock.patch.object(model, 'generate_samples', side_effect=capture), case.fast_plots():
            callback.on_validation_end(trainer, model)
        assert seeds == [9973 + 2 + 104729 * rank] * 2
        if rank == 0:
            assert case.history()['protocol']['samples_per_seed'] == 4
            assert trainer.logger.metrics[-1][0]['didi_pareto/failed'] == 0
        # A generation failure on rank one must stop rank zero before gather.
        callback = case.callback(model)
        with mock.patch.object(model, 'generate_samples', side_effect=(
                RuntimeError('rank-one sample failure') if rank == 1 else original)):
            callback.on_validation_end(trainer, model)
        if rank == 0:
            assert trainer.logger.metrics[-1][0]['didi_pareto/failed'] == 1
        # A reference-LM failure only on rank zero must stop rank one too.
        callback = case.callback(model)
        callback.metrics_factory = mock.Mock(side_effect=RuntimeError('rank-zero reference failure'))
        callback.on_validation_end(trainer, model)
        if rank == 0:
            assert trainer.logger.metrics[-1][0]['didi_pareto/failed'] == 1
        # Root-only post-gather/aggregation exceptions must not shift the
        # sequence of collectives followed by the other rank.
        original_gather = pareto._gather_tokens

        def failed_post_gather(tokens):
            result = original_gather(tokens)
            if rank == 0:
                raise RuntimeError('rank-zero post-gather failure')
            return result

        callback = case.callback(model)
        with mock.patch.object(pareto, '_gather_tokens', side_effect=failed_post_gather):
            callback.on_validation_end(trainer, model)
        if rank == 0:
            assert trainer.logger.metrics[-1][0]['didi_pareto/failed'] == 1
        callback = case.callback(model)
        with mock.patch.object(pareto, '_mean_std', side_effect=RuntimeError('aggregation failure')):
            callback.on_validation_end(trainer, model)
        if rank == 0:
            assert trainer.logger.metrics[-1][0]['didi_pareto/failed'] == 1
            print('PASS: two-rank Pareto pooling, paired seeds, actual TorchMetrics, '
                  'and synchronized generation/evaluation/gather/aggregation failures')
    finally:
        torch.distributed.destroy_process_group()
        case.doCleanups()


if __name__ == '__main__':
    if '--ddp-smoke' in sys.argv:
        ddp_smoke()
    else:
        unittest.main(verbosity=2)
