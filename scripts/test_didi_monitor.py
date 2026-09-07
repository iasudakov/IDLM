"""Offline tests for isolated DiDi diagnostics; no checkpoint or tracker needed.

    python scripts/test_didi_monitor.py
"""

import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import types
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from didi_repro import monitor


class Logger:
    def __init__(self):
        self.tables = {}
        self.metrics = []

    def log_table(self, key, columns, data, step=None):
        self.tables[key] = {'columns': columns, 'data': data, 'step': step}

    def log_metrics(self, metrics, step=None):
        self.metrics.append((metrics, step))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.weight.grad = torch.tensor(7.0)
        self.num_tokens = 4
        self.device = torch.device('cpu')
        self.sampler = 'guided'
        self.calls = 0
        self.tokenizer = SimpleNamespace(batch_decode=lambda xs: [str(row.tolist()) for row in xs])

    def generate_samples(self, num_samples, **kwargs):
        self.calls += 1
        return torch.randint(0, 3, (num_samples, self.num_tokens))


class SumMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0

    def compute(self):
        return torch.tensor(float(self.count))


class Evaluator:
    def __init__(self, **kwargs):
        self.gen_ppl = SumMetric()

    def to(self, device):
        pass

    def record_generative_perplexity(self, texts, **kwargs):
        # Exercise RNG isolation, including the first lazy evaluator call.
        random.random()
        np.random.random()
        torch.rand(1)
        assert not torch.is_grad_enabled()
        self.gen_ppl.count += len(texts)


class TestDiDiMonitor(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = OmegaConf.create({
            'checkpointing': {'save_dir': self.directory.name},
            'eval': {'perplexity_batch_size': 2,
                     'gen_ppl_eval_model_name_or_path': 'test/reference',
                     'compute_generative_perplexity': False},
            'sampling': {'num_sample_log': 2, 'steps': 32},
            'algo': {'name': 'didi_instruct', 'path': '/a path/teacher.ckpt',
                     'values': [1, 'text', None], 'empty': {}, 'null': None},
            'didi_monitor': {'enabled': True, 'gen_ppl': True, 'gm': False},
        })
        self.model = Model()
        self.trainer = SimpleNamespace(is_global_zero=True, sanity_checking=False,
                                       logger=Logger(), global_step=17)

    def test_full_config_strings_lists_nulls_and_revision(self):
        callback = monitor.attach_monitor(self.model, self.config, 'revision-123')
        callback.on_fit_start(self.trainer, self.model)
        rows = dict(self.trainer.logger.tables['didi/config']['data'])
        self.assertEqual(json.loads(rows['algo.path']), '/a path/teacher.ckpt')
        self.assertEqual(json.loads(rows['algo.values']), [1, 'text', None])
        self.assertEqual(json.loads(rows['algo.empty']), {})
        self.assertIsNone(json.loads(rows['algo.null']))
        self.assertEqual(json.loads(rows['didi_upstream_revision']), 'revision-123')
        payload = json.loads((Path(self.directory.name) / 'didi_monitor_config.json').read_text())
        self.assertEqual(payload['upstream_revision'], 'revision-123')
        self.assertEqual(payload['config'], OmegaConf.to_container(self.config, resolve=True))

    def test_implicit_hyperparameters_teacher_and_environment_provenance(self):
        self.model._didi_teacher_metadata = {
            'path': '/original/mdlm.ckpt', 'sha256': 'abc123', 'weights': 'raw backbone.*'}
        self.model._didi_student_metadata = {
            'source': 'authors/student', 'revision': 'immutable-hf-revision',
            'files': {'model.safetensors': {'sha256': 'student-sha'}}}
        self.model._didi_effective_hyperparameters = {
            'two_step': {'kl_hi_coef': 0.01, 'logprob_mode': 'prev_is_mask'},
            'sampling_eps': 0.02,
        }
        directory = Path(self.directory.name)
        (directory / 'didi_environment.json').write_text(json.dumps({
            'python': '3.12.11', 'torch': '2.5.1', 'devices': ['A100']}))
        (directory / 'didi_provenance.json').write_text(json.dumps({
            'upstream_revision': 'revision-123', 'overrides': ['training.finetune_path=/original/mdlm.ckpt']}))
        callback = monitor.attach_monitor(self.model, self.config, 'revision-123')
        callback.on_fit_start(self.trainer, self.model)
        rows = dict(self.trainer.logger.tables['didi/config']['data'])
        self.assertEqual(json.loads(rows['didi_provenance.teacher_checkpoint.sha256']), 'abc123')
        self.assertEqual(json.loads(rows['didi_provenance.teacher_checkpoint.weights']), 'raw backbone.*')
        self.assertEqual(json.loads(rows['didi_provenance.student_initialization.revision']),
                         'immutable-hf-revision')
        self.assertEqual(json.loads(rows['didi_provenance.effective_hyperparameters.two_step.kl_hi_coef']), 0.01)
        self.assertEqual(json.loads(rows['didi_provenance.environment.devices']), ['A100'])
        self.assertEqual(json.loads(rows['didi_provenance.entrypoint_provenance.overrides']),
                         ['training.finetune_path=/original/mdlm.ckpt'])
        payload = json.loads((directory / 'didi_monitor_config.json').read_text())
        self.assertEqual(payload['teacher_checkpoint'], self.model._didi_teacher_metadata)
        self.assertEqual(payload['student_initialization'], self.model._didi_student_metadata)
        self.assertEqual(payload['effective_hyperparameters'], self.model._didi_effective_hyperparameters)
        self.assertEqual(payload['environment']['torch'], '2.5.1')

    def test_capture_keeps_original_sampling_rng_and_result(self):
        torch.manual_seed(18)
        expected = self.model.generate_samples(3)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(18)
        callback = monitor.attach_monitor(self.model, self.config)
        callback.on_validation_start(self.trainer, self.model)
        output = self.model.generate_samples(3)
        self.assertTrue(torch.equal(output, expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertEqual(self.model.calls, 2)
        output.zero_()
        self.assertTrue(torch.equal(callback.samples[0], expected))

    def test_metric_values_and_rng_modes_gradients_unchanged(self):
        callback = monitor.attach_monitor(self.model, self.config)
        callback.metrics_factory = Evaluator
        callback.on_validation_start(self.trainer, self.model)
        # Three rows exercise the evaluator's smaller final batch.
        callback.samples = [torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 2, 2]])]
        torch_state = torch.get_rng_state()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        with torch.inference_mode():
            callback.on_validation_end(self.trainer, self.model)
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(self.model.training)
        self.assertEqual(float(self.model.weight.grad), 7.0)
        values, step = self.trainer.logger.metrics[0]
        self.assertAlmostEqual(values['val/sample_entropy'], 2 * math.log(2) / 3, places=6)
        self.assertEqual(values['val/gen_ppl'], 3.0)
        self.assertEqual(values['didi_monitor/sample_count'], 3)
        self.assertEqual(step, 17)
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(callback.samples, [])
        self.assertIn('didi/diagnostics', self.trainer.logger.tables)

    def test_reference_capture_gm_and_disabled_inference(self):
        self.config.didi_monitor.gm = True
        self.config.didi_monitor.max_reference_samples = 2
        callback = monitor.attach_monitor(self.model, self.config)
        callback.metrics_factory = Evaluator
        callback.on_validation_start(self.trainer, self.model)
        batch = {'input_ids': torch.arange(12).view(3, 4)}
        callback.on_validation_batch_end(self.trainer, self.model, None, batch, 0)
        callback.on_validation_batch_end(self.trainer, self.model, None, batch, 1)
        self.assertEqual(sum(len(x) for x in callback.references), 2)
        self.model.generate_samples(3)

        def gm(evaluator, samples, refs, length, batch_size, device):
            self.assertFalse(torch.is_inference_mode_enabled())
            self.assertEqual(len(refs), 2)
            return {'val/gm': -0.25, 'val/gen_ppl_gm': 12.0}

        with mock.patch.object(monitor, '_gradient_moment', side_effect=gm), torch.inference_mode():
            callback.on_validation_end(self.trainer, self.model)
        values = self.trainer.logger.metrics[0][0]
        self.assertEqual(values['val/gm'], -0.25)
        self.assertEqual(values['val/gen_ppl_gm'], 12.0)

    def test_disabled_and_empty_paths_do_not_load_evaluator(self):
        original = self.model.generate_samples
        self.config.didi_monitor.enabled = False
        self.assertIsNone(monitor.attach_monitor(self.model, self.config))
        self.assertEqual(self.model.generate_samples, original)
        self.config.didi_monitor.enabled = True
        callback = monitor.attach_monitor(self.model, self.config)
        callback.metrics_factory = mock.Mock(side_effect=AssertionError('no evaluation expected'))
        callback.on_validation_start(self.trainer, self.model)
        callback.on_validation_end(self.trainer, self.model)
        callback.metrics_factory.assert_not_called()
        self.assertEqual(self.trainer.logger.metrics[0][0]['didi_monitor/sample_count'], 0)

    def test_failure_logs_status_and_restores_rng(self):
        callback = monitor.attach_monitor(self.model, self.config)
        callback.on_validation_start(self.trainer, self.model)
        self.model.generate_samples(2)
        torch_state = torch.get_rng_state()

        def failed_evaluator(**kwargs):
            torch.rand(2)
            raise RuntimeError('simulated unavailable reference checkpoint')

        callback.metrics_factory = failed_evaluator
        callback.on_validation_end(self.trainer, self.model)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(self.trainer.logger.metrics[-1][0]['didi_monitor/failed'], 1)

    def test_distributed_gather_merges_all_ranks(self):
        callback = monitor.attach_monitor(self.model, self.config)
        callback.metrics_factory = Evaluator
        callback.on_validation_start(self.trainer, self.model)
        callback.samples = [torch.zeros(2, 4, dtype=torch.long)]

        def gather(payload, destination, dst):
            destination[:] = [payload, ([torch.ones(3, 4, dtype=torch.long)], [])]

        with mock.patch.object(torch.distributed, 'is_initialized', return_value=True), \
             mock.patch.object(torch.distributed, 'get_world_size', return_value=2), \
             mock.patch.object(torch.distributed, 'gather_object', side_effect=gather), \
             mock.patch.object(torch.distributed, 'barrier') as barrier:
            callback.on_validation_end(self.trainer, self.model)
        self.assertEqual(self.trainer.logger.metrics[0][0]['didi_monitor/sample_count'], 5)
        self.assertEqual(self.trainer.logger.metrics[0][0]['val/gen_ppl'], 5.0)
        barrier.assert_called_once()

    def test_nonzero_rank_does_not_construct_metrics(self):
        self.trainer.is_global_zero = False
        callback = monitor.attach_monitor(self.model, self.config)
        callback.metrics_factory = mock.Mock(side_effect=AssertionError('rank zero only'))
        with mock.patch.object(torch.distributed, 'is_initialized', return_value=True), \
             mock.patch.object(torch.distributed, 'gather_object') as gather, \
             mock.patch.object(torch.distributed, 'barrier') as barrier:
            callback.on_validation_end(self.trainer, self.model)
        callback.metrics_factory.assert_not_called()
        gather.assert_called_once()
        barrier.assert_called_once()

    def test_root_metrics_import_does_not_replace_upstream_module(self):
        upstream_metrics = types.ModuleType('metrics')
        with mock.patch.dict(sys.modules, {'metrics': upstream_metrics}):
            root_metrics = monitor._root_metrics_class()
            self.assertIs(sys.modules['metrics'], upstream_metrics)
            self.assertEqual(root_metrics.__module__, '_idlm_didi_monitor_metrics')

    def test_real_gm_estimator_keeps_grad_buffers_and_identical_samples_match(self):
        class ReferenceLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(3))

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, 3))

        class Tokenizer:
            eos_token_id = 2

            def __call__(self, texts, **kwargs):
                ids = torch.tensor([[0, 1, 0, 1] for _ in texts])
                return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}

        reference = ReferenceLM()
        reference.logits.grad = torch.full_like(reference.logits, 9)
        evaluator = SimpleNamespace(tokenizer=Tokenizer(), _ensure_eval_model=lambda device: reference)
        with torch.no_grad():
            values = monitor._gradient_moment(evaluator, ['x'] * 4, ['x'] * 4,
                                              max_length=4, batch_size=2, device='cpu')
        self.assertEqual(values['val/gm'], 0.0)
        self.assertAlmostEqual(values['val/gen_ppl_gm'], 3.0, places=5)
        self.assertEqual(values['val/gm_pairs'], 1)
        self.assertTrue(torch.equal(reference.logits.grad, torch.full_like(reference.logits, 9)))


if __name__ == '__main__':
    unittest.main()
