"""Check reference tracker outputs using CPU rollouts and an in-memory logger."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

import sdtt_eval
from underdeep_logger import UnderdeepLogger


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


class Model:
    device = torch.device('cpu')
    mask_index = 4
    num_tokens = 5
    noise = Noise()

    def __init__(self, directory):
        self.global_step = 500
        self.seen = []
        self.logged = []
        logger = Logger()
        self.trainer = SimpleNamespace(logger=logger, loggers=[logger], global_rank=0,
                                       world_size=1, ckpt_path=None)
        self.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: 'generated code')
        self.config = OmegaConf.create({
            'seed': 1, 'sampling': {'num_sample_log': 2},
            'eval': {'gsm8k_pareto_thresholds': [.49, .9], 'gsm8k_batch_size': 2,
                     'gsm8k_grading_workers': 1},
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
        probs = torch.zeros(*x.shape, 5)
        probs[..., 1:3] = .5
        return probs.log()


class TrackerParityTest(unittest.TestCase):
    def test_samples_reuse_existing_generation_and_grades(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Model(directory)
            examples = [
                {'input_ids': [0, 2], 'answer': 'gold 1', 'question': 'question 1'},
                {'input_ids': [0, 2, 2], 'answer': 'gold 2', 'question': 'question 2'},
                {'input_ids': [0, 2], 'answer': 'gold 3', 'question': 'question 3'},
            ]
            model.config.sampling.num_sample_log = 0
            with patch.object(sdtt_eval, 'grade_sample', return_value=True):
                without_samples = sdtt_eval.sweep(model, examples)
            states_without_samples = model.seen
            self.assertEqual(model._gsm8k_sample_records, [])
            model.seen = []
            model.config.sampling.num_sample_log = 2
            with patch.object(sdtt_eval, 'grade_sample', return_value=True) as grade:
                with_samples = sdtt_eval.sweep(model, examples)
            self.assertEqual(with_samples, without_samples)
            self.assertEqual(len(model.seen), len(states_without_samples))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.seen, states_without_samples)))
            # Only the original two thresholds x three examples are graded.
            self.assertEqual(grade.call_count, 6)
            rows = model._gsm8k_sample_records
            self.assertEqual(len(rows), 2)
            self.assertEqual([row['question'] for row in rows], ['question 1', 'question 2'])
            self.assertTrue(all(row['threshold'] == .9 for row in rows))
            self.assertTrue(all(row['generated'] == 'generated code' and row['grade'] for row in rows))
            self.assertEqual([row['nfe'] for row in rows], [3, 2])

    def test_publish_preserves_original_keys_native_frontier_image_and_table(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Model(directory)
            old = [{'threshold': .9, 'accuracy': .4, 'nfe': 15.}]
            curve = [{'threshold': .35, 'accuracy': .3, 'nfe': 4.5},
                     {'threshold': .9, 'accuracy': .6, 'nfe': 10.}]
            model._gsm8k_pareto_history = {0: old}
            model._gsm8k_sample_records = [
                {'question': 'How many?', 'generated': 'return 42', 'reference': '#### 42',
                 'grade': True, 'nfe': 10, 'threshold': .9}]
            image = np.zeros((2, 3, 3), dtype=np.uint8)
            with patch.object(sdtt_eval, 'render_frontier', return_value=image) as render:
                sdtt_eval.publish(model, curve)
            logged = {name: value for name, value, _ in model.logged}
            self.assertAlmostEqual(logged['val/gsm8k_acc_thr035'], .3)
            self.assertEqual(logged['val/gsm8k_nfe_thr09'], 10.)
            self.assertAlmostEqual(logged['pareto/gsm8k_acc_thr09'], .6)
            logger = model.trainer.logger
            self.assertEqual(logger.metrics,
                             [({'pareto/gsm8k_step_500': .3}, 450),
                              ({'pareto/gsm8k_step_500': .6}, 1000)])
            render.assert_called_once_with({0: old, 500: curve}, 500)
            self.assertEqual(logger.images[0][0], 'pareto/frontier')
            self.assertIs(logger.images[0][1], image)
            self.assertEqual(logger.images[0][2], 500)
            table = logger.tables[0]
            self.assertEqual(table[0], 'val/samples')
            self.assertEqual(table[1], ['question', 'generated', 'reference', 'grade', 'NFE', 'threshold'])
            self.assertEqual(table[2], [['How many?', 'return 42', '#### 42', True, 10, .9]])
            self.assertEqual(table[3], 500)
            history = json.loads((Path(directory) / 'pareto_history.json').read_text())
            self.assertEqual(set(history['curves']), {'0', '500'})

    def test_initial_metrics_use_logger_step_zero_without_lightning_epoch_context(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Model(directory)
            model.global_step = 0
            curve = [{'threshold': .325, 'accuracy': .1, 'nfe': 4.}]
            with patch.object(sdtt_eval, 'render_frontier', return_value=np.zeros((2, 2, 3))):
                sdtt_eval.publish(model, curve, initial=True)
            self.assertEqual(model.logged, [])
            metrics, step = model.trainer.logger.metrics[0]
            self.assertEqual(step, 0)
            self.assertEqual(metrics['val/gsm8k_acc_thr0325'], .1)
            self.assertEqual(metrics['val/gsm8k_nfe_thr0325'], 4.)
            self.assertIn(({'pareto/gsm8k_step_0': .1}, 400), model.trainer.logger.metrics)

    def test_frontier_renderer_returns_rgb_history_image(self):
        image = sdtt_eval.render_frontier({0: [{'nfe': 10., 'accuracy': .2}],
                                          500: [{'nfe': 8., 'accuracy': .4}]}, 500)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[-1], 3)
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(image.shape[0], 100)
        self.assertGreater(float(image.std()), 0)


class UnderdeepTableTest(unittest.TestCase):
    def test_table_payload_matches_existing_underdeep_adapter(self):
        class UTable:
            def __init__(self, value):
                self.value = value

        data_types = types.ModuleType('underdeep.runs.data_types')
        data_types.UTable = UTable
        calls = []
        run = SimpleNamespace(log=lambda payload, step=None: calls.append((payload, step)))
        logger = UnderdeepLogger()
        logger._initialized = True
        logger._run = run
        with patch.dict(sys.modules, {'underdeep.runs.data_types': data_types}):
            logger.log_table('val/samples', ['question', 'grade'], [['Q', True]], step=500)
        self.assertEqual(calls[0][0]['val/samples'].value, [{'question': 'Q', 'grade': True}])
        self.assertEqual(calls[0][1], 500)
        # Empty tables do not create empty tracker artifacts.
        logger.log_table('val/samples', ['question'], [], step=500)
        self.assertEqual(len(calls), 1)

    def test_missing_tracker_keeps_existing_noop_behavior(self):
        logger = UnderdeepLogger()
        logger._initialized = True
        logger._run = None
        logger.log_table('val/samples', ['question'], [['Q']], step=0)


if __name__ == '__main__':
    unittest.main()
