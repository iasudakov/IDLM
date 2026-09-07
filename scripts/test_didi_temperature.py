"""Offline temperature-context tests with the actual tiny upstream runtime.

Run separately: conda run -n mdlm python scripts/test_didi_temperature.py
"""

import copy
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from didi_repro.temperature import sampling_temperature
import test_didi_runtime as runtime_tests


class TemperatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime_tests.DiDiRuntimeTests.setUpClass()
        cls.helpers = runtime_tests.DiDiRuntimeTests()

    def model(self, sampler='ancestral_cache'):
        cfg = self.helpers.config(tiny=True)
        cfg.sampling.predictor = sampler
        model = self.helpers.model(cfg)
        self.helpers.initialize(model)
        return model

    def input(self, model):
        tokens = model.prior_sample(2, model.num_tokens)
        tokens[:, :3] = torch.tensor([1, 2, 3])
        sigma = torch.ones(2, 1)
        return tokens, sigma

    def test_one_is_exact_identity_and_seeded_samples_match(self):
        for sampler in ('ancestral_cache', 'guided'):
            with self.subTest(sampler=sampler):
                torch.manual_seed(42)
                model = self.model(sampler)
                self.assertNotIn('forward', model.__dict__)
                tokens, sigma = self.input(model)
                reference = model.forward(tokens, sigma)
                with sampling_temperature(model, 1.0):
                    self.assertNotIn('forward', model.__dict__)
                    torch.testing.assert_close(model.forward(tokens, sigma), reference, rtol=0, atol=0)
                torch.manual_seed(101)
                expected = model.generate_samples(8, num_steps=4)
                torch.manual_seed(101)
                with sampling_temperature(model, 1.0):
                    actual = model.generate_samples(8, num_steps=4)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_masked_distributions_tempered_observed_rows_preserved(self):
        model = self.model()
        tokens, sigma = self.input(model)
        original = model.forward(tokens, sigma)
        with sampling_temperature(model, 2.0):
            actual = model.forward(xt=tokens, sigma=sigma)
        observed = tokens != model.mask_index
        torch.testing.assert_close(actual[observed], original[observed], rtol=0, atol=0)
        expected = (original / 2).log_softmax(-1)
        torch.testing.assert_close(actual[~observed][:, :model.mask_index],
                                   expected[~observed][:, :model.mask_index])
        self.assertTrue(torch.isneginf(actual[~observed][:, model.mask_index]).all())
        self.assertFalse(torch.allclose(actual[~observed][:, :model.mask_index],
                                        original[~observed][:, :model.mask_index]))
        torch.testing.assert_close(actual.exp().sum(-1), torch.ones_like(tokens, dtype=actual.dtype))

    def test_extreme_positive_temperatures_keep_support_and_no_nan(self):
        model = self.model()
        tokens, sigma = self.input(model)
        original = model.forward(tokens, sigma)
        observed = tokens != model.mask_index
        for temperature in (1e9, 1e308, 1e-300):
            with self.subTest(temperature=temperature):
                with sampling_temperature(model, temperature):
                    actual = model.forward(tokens, sigma)
                self.assertFalse(torch.isnan(actual).any())
                torch.testing.assert_close(actual[observed], original[observed], rtol=0, atol=0)
                self.assertTrue((actual[~observed][:, model.mask_index].exp() == 0).all())
                torch.testing.assert_close(actual.exp().sum(-1), torch.ones_like(tokens, dtype=actual.dtype))

    def test_nonunit_temperature_reaches_both_samplers_and_terminal_cleanup(self):
        for sampler in ('ancestral_cache', 'guided'):
            with self.subTest(sampler=sampler):
                torch.manual_seed(17)
                model = self.model(sampler)
                torch.manual_seed(103)
                expected = model.generate_samples(24, num_steps=4)
                calls = []
                original = model.forward

                def record_forward(*args, **kwargs):
                    calls.append(args[0] if args else kwargs['xt'])
                    return original(*args, **kwargs)

                with mock.patch.object(model, 'forward', side_effect=record_forward):
                    torch.manual_seed(103)
                    with sampling_temperature(model, .2):
                        actual = model.generate_samples(24, num_steps=4)
                self.assertEqual(len(calls), 5)  # Four transitions plus terminal denoiser.
                self.assertTrue((actual != expected).any())
                self.assertFalse((actual == model.mask_index).any())

    def test_context_alone_changes_no_rng_modes_gradients_or_parameters(self):
        model = self.model()
        model.teacher.eval()
        for parameter in model.backbone.parameters():
            parameter.grad = torch.full_like(parameter, .25)
        parameter_state = copy.deepcopy(model.state_dict())
        modes = {name: module.training for name, module in model.named_modules()}
        gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with sampling_temperature(model, .75):
            pass
        self.assertEqual(random.getstate(), python_rng)
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_rng[0])
        np.testing.assert_array_equal(current_numpy[1], numpy_rng[1])
        self.assertEqual(current_numpy[2:], numpy_rng[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        self.assertEqual(modes, {name: module.training for name, module in model.named_modules()})
        for name, parameter in model.named_parameters():
            self.assertIs(parameter.grad, gradients[name])
        self.helpers.assert_same(model.state_dict(), parameter_state)
        self.assertNotIn('forward', model.__dict__)

    def test_restoration_after_failure_and_nested_absolute_temperatures(self):
        model = self.model()
        tokens, sigma = self.input(model)
        baseline = model.forward(tokens, sigma)
        with sampling_temperature(model, 2.0):
            outer = model.forward
            outer_values = model.forward(tokens, sigma)
            with sampling_temperature(model, 1.0):
                torch.testing.assert_close(model.forward(tokens, sigma), baseline, rtol=0, atol=0)
            self.assertIs(model.forward, outer)
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with sampling_temperature(model, .5):
                    expected = (baseline[0, 3, :model.mask_index] / .5).log_softmax(-1)
                    torch.testing.assert_close(model.forward(tokens, sigma)[0, 3, :model.mask_index], expected)
                    raise RuntimeError('test failure')
            self.assertIs(model.forward, outer)
            torch.testing.assert_close(model.forward(tokens, sigma), outer_values, rtol=0, atol=0)
        self.assertNotIn('forward', model.__dict__)
        custom = lambda *args, **kwargs: baseline
        model.forward = custom
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with sampling_temperature(model, 3.0):
                raise RuntimeError('test failure')
        self.assertIs(model.forward, custom)

    def test_invalid_inputs_and_training_guard(self):
        model = self.model()
        for temperature in (0, -1, float('nan'), float('inf'), -float('inf'), None, 'bad'):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, 'finite and greater than zero'):
                    with sampling_temperature(model, temperature):
                        pass
        model._didi_in_training_step = True
        for temperature in (1, 2):
            with self.assertRaisesRegex(RuntimeError, 'evaluation-only'):
                with sampling_temperature(model, temperature):
                    pass
        model._didi_in_training_step = False
        tokens, sigma = self.input(model)
        with sampling_temperature(model, 2):
            model._didi_in_training_step = True
            with self.assertRaisesRegex(RuntimeError, 'evaluation-only'):
                model.forward(tokens, sigma)
        self.assertNotIn('forward', model.__dict__)


if __name__ == '__main__':
    unittest.main(verbosity=2)
