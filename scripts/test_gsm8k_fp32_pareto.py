"""CPU checks for the two production sweeps and independent tracker/history output."""

import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from test_eval_prediction_precision import Model, ROOT, source_method
from eval_precision import capture_rng_state, fp32_prediction_normalization, replay_rng_state


def trainer_method(name):
    return source_method(ROOT / 'trainer_base.py', 'TrainerBase', name, globals())


_NFE_X_SCALE = 100


class SweepModel(Model):
    _log_gsm8k_pareto = trainer_method('_log_gsm8k_pareto')
    _run_gsm8k_pareto = trainer_method('_run_gsm8k_pareto')
    _publish_pareto = trainer_method('_publish_pareto')

    def __init__(self, directory, fp32=True):
        super().__init__()
        self.config = OmegaConf.create({'eval': {'gsm8k_pareto_thresholds': [.925],
                                                'gsm8k_pareto_fp32': fp32},
                                       'sampling': {'steps': 16},
                                       'checkpointing': {'save_dir': directory}})
        self.trainer = SimpleNamespace(global_rank=0, logger=SimpleNamespace(
            version='run-a', log_metrics=mock.Mock(), log_image=mock.Mock()))
        self.device = torch.device('cpu')
        self.global_step = 100
        self.remasking_strategy, self.dynamic_threshold, self.sampling_temperature = 'random', .7, 1.
        self.log = mock.Mock()
        self.draws, self.dtypes = [], []
        self.fail_fp32 = False

    def _gsm8k_generate(self, dataset, indices, batch_size, steps, convention):
        self.draws.append((random.random(), np.random.rand(), torch.rand(1).item()))
        output = self(torch.full((1, 3), self.mask_index))
        self.dtypes.append(output.dtype)
        if self.fail_fp32 and output.dtype == torch.float32:
            raise RuntimeError('FP32 generation failed')
        nfe = 1 if output.exp()[0, 0, 0] > self.dynamic_threshold else 3
        return [('answer', 'gold')], [nfe], [nfe]

    def _gsm8k_grade(self, rows):
        return 1, 1.


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = SweepModel(self.directory.name)

    def run_sweep(self):
        with mock.patch('pareto_plot.render', return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
            self.model._log_gsm8k_pareto([], [], 2)

    def test_both_precisions_paired_rng_independent_history_and_metrics(self):
        initial = capture_rng_state()
        with replay_rng_state(initial):
            random.random(), np.random.rand(), torch.rand(1)
            expected_next = random.random(), np.random.rand(), torch.rand(1).item()
        self.run_sweep()
        self.assertEqual(self.model.dtypes, [torch.bfloat16, torch.float32])
        self.assertEqual(self.model.draws[0], self.model.draws[1])
        self.assertEqual((random.random(), np.random.rand(), torch.rand(1).item()), expected_next)
        for suffix, nfe in (('', 1), ('_fp32norm', 3)):
            stored = json.loads(Path(self.directory.name, f'pareto_history{suffix}.json').read_text())
            self.assertEqual(stored['curves']['100'][0]['nfe'], nfe)
            self.assertEqual(getattr(self.model, f'_pareto_history{suffix}')[100][0]['nfe'], nfe)
        keys = [call.args[0] for call in self.model.log.call_args_list]
        self.assertEqual(set(keys), {'val/gsm8k_acc_thr0925', 'val/gsm8k_nfe_thr0925',
                                    'val/gsm8k_acc_thr0925_fp32norm', 'val/gsm8k_nfe_thr0925_fp32norm'})
        logger = self.model.trainer.logger
        self.assertEqual([next(iter(call.args[0])) for call in logger.log_metrics.call_args_list],
                         ['pareto/gsm8k_step_100', 'pareto/gsm8k_step_100_fp32norm'])
        self.assertEqual([call.args[0] for call in logger.log_image.call_args_list],
                         ['pareto/frontier', 'pareto/frontier_fp32norm'])
        self.assertNotIn('_process_model_output', vars(self.model))

    def test_disabled_fp32_and_disabled_sweep(self):
        self.model.config.eval.gsm8k_pareto_fp32 = False
        self.run_sweep()
        self.assertEqual(self.model.dtypes, [torch.bfloat16])
        self.assertFalse(Path(self.directory.name, 'pareto_history_fp32norm.json').exists())
        self.model.config.eval.gsm8k_pareto_thresholds = []
        self.run_sweep()
        self.assertEqual(len(self.model.dtypes), 1)

    def test_failure_restores_normalizer_sampler_and_training_rng(self):
        self.model.fail_fp32 = True
        initial = capture_rng_state()
        with replay_rng_state(initial):
            random.random(), np.random.rand(), torch.rand(1)
            expected_next = random.random(), np.random.rand(), torch.rand(1).item()
        with self.assertRaisesRegex(RuntimeError, 'FP32 generation failed'):
            self.run_sweep()
        self.assertNotIn('_process_model_output', vars(self.model))
        self.assertEqual((self.model.remasking_strategy, self.model.dynamic_threshold,
                          self.model.sampling_temperature), ('random', .7, 1.))
        self.assertEqual((random.random(), np.random.rand(), torch.rand(1).item()), expected_next)


if __name__ == '__main__':
    unittest.main(verbosity=2)
