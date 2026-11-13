"""CPU regression tests for explicit weight selection and validation isolation."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('ema_under_test', ROOT / 'models/ema.py')
ema_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ema_module)
EMA = ema_module.ExponentialMovingAverage
from training_gsm8k import run_weighted_benchmark, pass_at_k, _passk
from eval_precision import capture_rng_state


class Config(dict):
    __getattr__ = dict.__getitem__


class FakeModel:
    def __init__(self, out):
        self.p = torch.nn.Parameter(torch.tensor([2.]))
        self.ema = EMA([self.p], 0.9999)
        self.ema.shadow_params[0].fill_(7.)
        self.p.requires_grad_(False)
        self.ema.store([self.p])
        self.ema.copy_to([self.p])
        self.remasking_strategy, self.dynamic_threshold, self.sampling_temperature = 'random', .9, 1.
        self.global_step = 500
        self.logged, self.draws = [], []
        self.trainer = SimpleNamespace(global_rank=0, world_size=1,
            logger=SimpleNamespace(log_metrics=lambda metrics, step: self.logged.append((metrics, step))))
        self.config = Config(eval=Config(gsm8k_weights=['raw', 'ema'], gsm8k_passk=True,
            gsm8k_passk_samples=2, gsm8k_passk_ks=[1, 2], gsm8k_passk_budgets=[4]),
            trainer=Config(precision='bf16'), sampling=Config(predictor='ancestral_cache'),
            checkpointing=Config(save_dir=out))

    def _get_parameters(self):
        return iter([self.p])

    def _process_model_output(self, model_output, xt, sigma):
        return model_output

    def _gsm8k_generate(self, dataset, indices, batch_size, steps, convention):
        assert convention == 'sflm'
        self.draws.append((float(self.p), steps, torch.rand(1).item(), random.random(), np.random.rand()))
        return [(str(int(self.p)), 'gold')] * 1319, [steps], [steps] * 1319

    def _gsm8k_grade(self, rows):
        return 1319, 1.

    def _run_gsm8k_pareto(self, dataset, indices, batch_size, suffix):
        self.logged.append(({suffix: float(self.p)}, self.global_step))


class Tests(unittest.TestCase):
    def test_estimator(self):
        self.assertEqual(pass_at_k(32, 0, 32), 0.)
        self.assertEqual(pass_at_k(32, 1, 32), 1.)
        self.assertAlmostEqual(pass_at_k(32, 8, 1), .25)
        self.assertAlmostEqual(pass_at_k(4, 1, 2), .5)
        with self.assertRaises(ValueError): pass_at_k(2, 1, 3)

    def test_ema_frozen_membership_and_legacy_state(self):
        p = torch.nn.Parameter(torch.tensor([2.]))
        excluded = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
        ema = EMA([excluded, p], .9)
        ema.load_state_dict(dict(decay=.9, num_updates=10, shadow_params=[torch.tensor([8.])]))
        p.requires_grad_(False)
        ema.copy_to([excluded, p])
        self.assertEqual(p.item(), 8.)
        self.assertEqual(excluded.item(), 3.)
        with self.assertRaises(ValueError): ema.copy_to([p])

    def test_paired_eval_and_restoration(self):
        with tempfile.TemporaryDirectory() as out:
            model = FakeModel(out)
            rng = capture_rng_state()
            grader = SimpleNamespace(evaluate_samples=lambda response, gold, timeout_s: response == '7')
            with patch.dict(sys.modules, {'sandbox_gsm8k': grader}):
                run_weighted_benchmark(model, range(1319), list(range(1319)), 16, 64)
            self.assertEqual(float(model.p), 7.)  # outer validation still uses EMA
            self.assertFalse(model.p.requires_grad)
            self.assertEqual(model.ema.collected_params[0].item(), 2.)
            self.assertEqual(model.ema.num_updates, 0)
            self.assertNotIn('_process_model_output', vars(model))

            self.assertEqual(model.sampling_temperature, 1.)
            after = capture_rng_state()
            self.assertEqual(after[0], rng[0])
            np.testing.assert_array_equal(after[1][1], rng[1][1])
            self.assertTrue(torch.equal(after[2], rng[2]))
            # Same random streams, different weights, for all paired draws.
            self.assertEqual(len(model.draws), 8)
            for a, b in zip(model.draws[:4], model.draws[4:]):
                self.assertEqual(a[0], 2.)
                self.assertEqual(b[0], 7.)
                self.assertEqual(a[1:], b[1:])
            for weight, expected in [('raw', 0.), ('ema', 1.)]:
                result = json.loads((Path(out)/'passk/step_00000500'/weight/'results.json').read_text())
                self.assertTrue(result['complete'])
                self.assertEqual(result['results'][0]['pass_at_k'], {'1': expected, '2': expected})
            model.ema.restore(model._get_parameters())
            self.assertEqual(float(model.p), 2.)

    def test_exception_restores_weights_and_rng(self):
        with tempfile.TemporaryDirectory() as out:
            model = FakeModel(out)
            rng = torch.get_rng_state().clone()
            def fail(*args, **kwargs):
                torch.rand(3)
                raise RuntimeError('test failure')
            model._gsm8k_generate = fail
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                run_weighted_benchmark(model, range(1319), list(range(1319)), 16, 64)
            self.assertEqual(float(model.p), 7.)
            self.assertEqual(model.ema.collected_params[0].item(), 2.)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertNotIn('_process_model_output', vars(model))


def ddp_worker(rank, init_file, out):
    torch.distributed.init_process_group('gloo', init_method='file://' + init_file,
                                        rank=rank, world_size=2)
    try:
        model = FakeModel(out)
        model.trainer.global_rank, model.trainer.world_size = rank, 2
        def generate(dataset, indices, batch_size, steps, convention):
            local = [(str(i), 'gold') for i in indices]
            gathered = [None, None]
            torch.distributed.all_gather_object(gathered, local)
            rows = [r for shard in gathered for r in shard]
            return rows, [steps], [steps] * len(rows)
        model._gsm8k_generate = generate
        grader = SimpleNamespace(evaluate_samples=lambda r, g, timeout_s: int(r) % 3 == 0)
        with patch.dict(sys.modules, {'sandbox_gsm8k': grader}):
            _passk(model, range(1319), list(range(rank, 1319, 2)), 16, 'ema')
        # A rank-zero grading error must reach both workers without deadlock.
        grader.evaluate_samples = lambda *args, **kw: (_ for _ in ()).throw(ValueError('grader error'))
        try:
            with patch.dict(sys.modules, {'sandbox_gsm8k': grader}):
                _passk(model, range(1319), list(range(rank, 1319, 2)), 16, 'raw')
        except RuntimeError as error:
            assert 'grader error' in str(error)
        else:
            raise AssertionError('Expected grading error on every rank')
    finally:
        torch.distributed.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_uneven_shards_and_failure_propagation(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as out:
            mp.spawn(ddp_worker, args=(str(Path(out)/'init'), out), nprocs=2, join=True)
            result = json.loads((Path(out)/'passk/step_00000500/ema/results.json').read_text())
            counts = result['results'][0]['correct_per_question']
            self.assertEqual(counts, [2 if i % 3 == 0 else 0 for i in range(1319)])
            self.assertAlmostEqual(result['results'][0]['pass_at_k']['1'], 440 / 1319)


if __name__ == '__main__':
    unittest.main()
