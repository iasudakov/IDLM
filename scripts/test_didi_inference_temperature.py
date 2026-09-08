"""CPU-only temperature tests for direct-EMA official inference sampling."""

import copy
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from didi_repro.temperature import _temper_raw_logits, sampling_temperature


class TinyDenoiser(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, 1.3, -0.7, 0.6]) + offset)
        self.last_sigma = None

    def forward(self, x, sigma):
        self.last_sigma = sigma
        result = self.logits.expand(*x.shape, -1)
        # A genuine nonzero-sigma dependence that the temperature hook must not alter.
        addition = torch.zeros_like(result)
        addition[..., 0] = sigma.reshape(x.shape[0], -1)[:, :1]
        return result + addition


class TinyInference(torch.nn.Module):
    def __init__(self, processed_ema=False):
        super().__init__()
        self._didi_sampler_source = 'inference'
        self.mask_index = 3
        self.student_ema = TinyDenoiser(0.0)
        self.backbone = TinyDenoiser(0.5)
        self.discriminator = torch.nn.Linear(4, 1)
        self.processed_ema = processed_ema

    def forward(self, xt, sigma):
        denoiser = self.student_ema if self.processed_ema else self.backbone
        logits = denoiser(xt, sigma).clone()
        logits[..., self.mask_index] = -1e6
        result = logits.log_softmax(-1)
        observed = torch.full_like(result, -1e6)
        observed.scatter_(-1, xt.unsqueeze(-1), 0.0)
        return torch.where((xt == self.mask_index).unsqueeze(-1), result, observed)

    @torch.no_grad()
    def generate(self, tokens, sigma):
        first = torch.multinomial(self.student_ema(tokens, sigma).softmax(-1).reshape(-1, 4), 1)
        tokens = first.reshape_as(tokens)
        final = self.forward(tokens, sigma)
        return torch.multinomial(final.exp().reshape(-1, 4), 1).reshape_as(tokens)


class InferenceTemperatureTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyInference()
        self.tokens = torch.tensor([[3, 1, 3, 2], [0, 3, 3, 3]])
        self.sigma = torch.tensor([[0.25], [0.75]])

    def test_one_is_strict_noop_for_both_forwards_samples_and_rng(self):
        reference_raw = self.model.student_ema(self.tokens, self.sigma)
        reference_processed = self.model(self.tokens, self.sigma)
        with sampling_temperature(self.model, 1):
            self.assertNotIn('forward', self.model.__dict__)
            self.assertNotIn('forward', self.model.student_ema.__dict__)
            torch.testing.assert_close(self.model.student_ema(self.tokens, self.sigma),
                                       reference_raw, rtol=0, atol=0)
            torch.testing.assert_close(self.model(self.tokens, self.sigma),
                                       reference_processed, rtol=0, atol=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            reference = self.model.generate(self.tokens, self.sigma)
            rng = torch.get_rng_state()
            torch.manual_seed(43)
            with sampling_temperature(self.model, 1):
                actual = self.model.generate(self.tokens, self.sigma)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_direct_ema_is_raw_logits_over_temperature_with_full_support_and_sigma(self):
        original = self.model.student_ema(self.tokens, self.sigma)
        for temperature in (0.75, 2.0):
            with self.subTest(temperature=temperature), sampling_temperature(self.model, temperature):
                actual = self.model.student_ema(x=self.tokens, sigma=self.sigma)
                torch.testing.assert_close(actual, original / temperature, rtol=0, atol=0)
                self.assertIs(self.model.student_ema.last_sigma, self.sigma)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue((actual.softmax(-1)[..., self.model.mask_index] > 0).all())
                self.assertFalse(torch.equal(actual[self.tokens != 3], original[self.tokens != 3]))

    def test_processed_cleanup_remains_subs_aware_and_not_double_tempered(self):
        for processed_ema in (False, True):
            with self.subTest(processed_ema=processed_ema):
                model = TinyInference(processed_ema=processed_ema)
                original = model(self.tokens, self.sigma)
                original_raw = model.student_ema(self.tokens, self.sigma)
                with sampling_temperature(model, 2.0):
                    actual = model(self.tokens, self.sigma)
                    raw = model.student_ema(self.tokens, self.sigma)
                observed = self.tokens != model.mask_index
                torch.testing.assert_close(actual[observed], original[observed], rtol=0, atol=0)
                expected = (original[~observed][:, :3] / 2).log_softmax(-1)
                torch.testing.assert_close(actual[~observed][:, :3], expected)
                self.assertTrue(torch.isneginf(actual[~observed][:, 3]).all())
                torch.testing.assert_close(raw, original_raw / 2, rtol=0, atol=0)

    def test_main_source_does_not_wrap_or_change_direct_ema_calls(self):
        self.model._didi_sampler_source = 'main'
        original = self.model.student_ema(self.tokens, self.sigma)
        with sampling_temperature(self.model, 0.75):
            self.assertNotIn('forward', self.model.student_ema.__dict__)
            torch.testing.assert_close(self.model.student_ema(self.tokens, self.sigma),
                                       original, rtol=0, atol=0)
        del self.model._didi_sampler_source
        with sampling_temperature(self.model, 0.75):
            self.assertNotIn('forward', self.model.student_ema.__dict__)

    def test_nested_temperatures_are_absolute_for_raw_and_processed_paths(self):
        original_raw = self.model.student_ema(self.tokens, self.sigma)
        original_processed = self.model(self.tokens, self.sigma)
        with sampling_temperature(self.model, 2):
            outer_forward = self.model.forward
            outer_ema = self.model.student_ema.forward
            outer_processed = self.model(self.tokens, self.sigma)
            with sampling_temperature(self.model, 1):
                torch.testing.assert_close(self.model.student_ema(self.tokens, self.sigma),
                                           original_raw, rtol=0, atol=0)
                torch.testing.assert_close(self.model(self.tokens, self.sigma),
                                           original_processed, rtol=0, atol=0)
            self.assertIs(self.model.forward, outer_forward)
            self.assertIs(self.model.student_ema.forward, outer_ema)
            with sampling_temperature(self.model, 0.5):
                torch.testing.assert_close(self.model.student_ema(self.tokens, self.sigma),
                                           original_raw / 0.5, rtol=0, atol=0)
                expected = (original_processed[0, 0, :3] / 0.5).log_softmax(-1)
                torch.testing.assert_close(self.model(self.tokens, self.sigma)[0, 0, :3], expected)
            self.assertIs(self.model.forward, outer_forward)
            self.assertIs(self.model.student_ema.forward, outer_ema)
            torch.testing.assert_close(self.model(self.tokens, self.sigma), outer_processed,
                                       rtol=0, atol=0)
        self.assertNotIn('forward', self.model.__dict__)
        self.assertNotIn('forward', self.model.student_ema.__dict__)

    def test_preexisting_overrides_are_restored_even_on_nested_exception(self):
        original_forward = self.model.forward
        original_ema = self.model.student_ema.forward

        def custom_forward(*args, **kwargs):
            return original_forward(*args, **kwargs)

        def custom_ema(*args, **kwargs):
            return original_ema(*args, **kwargs) + 0.125

        self.model.forward = custom_forward
        self.model.student_ema.forward = custom_ema
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with sampling_temperature(self.model, 2):
                outer_forward, outer_ema = self.model.forward, self.model.student_ema.forward
                try:
                    with sampling_temperature(self.model, 0.5):
                        raise RuntimeError('test failure')
                finally:
                    self.assertIs(self.model.forward, outer_forward)
                    self.assertIs(self.model.student_ema.forward, outer_ema)
        self.assertIs(self.model.forward, custom_forward)
        self.assertIs(self.model.student_ema.forward, custom_ema)

    def test_processed_failure_does_not_leave_ema_temperature_suppressed(self):
        model = TinyInference(processed_ema=True)
        original_raw = model.student_ema(self.tokens, self.sigma)

        def failing_forward(*args, **kwargs):
            model.student_ema(*args, **kwargs)
            raise RuntimeError('test failure')

        model.forward = failing_forward
        with sampling_temperature(model, 2):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                model(self.tokens, self.sigma)
            torch.testing.assert_close(model.student_ema(self.tokens, self.sigma),
                                       original_raw / 2, rtol=0, atol=0)

    def test_extreme_positive_temperatures_are_stable_and_keep_existing_support(self):
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.tensor([[1., 2., -1., 0.25], [-float('inf'), 1., 2., 0.25]],
                                  dtype=dtype)
            for temperature in (1e9, 1e308, 1e-300, 5e-324):
                with self.subTest(dtype=dtype, temperature=temperature):
                    actual = _temper_raw_logits(logits, temperature)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(torch.isfinite(actual[torch.isfinite(logits)]).all())
                    self.assertTrue(torch.isneginf(actual[1, 0]))
                    probabilities = actual.float().softmax(-1)
                    self.assertFalse(torch.isnan(probabilities).any())
                    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
                    if temperature < 1:
                        self.assertEqual(probabilities[0].argmax().item(), logits[0].argmax().item())
                        self.assertEqual(probabilities[1].argmax().item(), logits[1].argmax().item())

    def test_context_alone_changes_no_rng_modes_parameters_or_gradient_references(self):
        self.model.student_ema.eval()
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        parameters = copy.deepcopy(self.model.state_dict())
        gradients = {name: value.grad for name, value in self.model.named_parameters()}
        modes = {name: module.training for name, module in self.model.named_modules()}
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with sampling_temperature(self.model, 0.75):
            pass
        self.assertEqual(random.getstate(), rng[0])
        self.assertEqual(np.random.get_state()[0], rng[1][0])
        np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
        self.assertEqual(np.random.get_state()[2:], rng[1][2:])
        torch.testing.assert_close(torch.get_rng_state(), rng[2], rtol=0, atol=0)
        self.assertEqual(modes, {name: module.training for name, module in self.model.named_modules()})
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
        for name, value in self.model.named_parameters():
            self.assertIs(value.grad, gradients[name])

    def test_entry_and_both_call_paths_reject_training(self):
        self.model._didi_in_training_step = True
        for temperature in (1, 2):
            with self.assertRaisesRegex(RuntimeError, 'evaluation-only'):
                with sampling_temperature(self.model, temperature):
                    pass
        self.model._didi_in_training_step = False
        with sampling_temperature(self.model, 2):
            self.model._didi_in_training_step = True
            for call in (self.model, self.model.student_ema):
                with self.assertRaisesRegex(RuntimeError, 'evaluation-only'):
                    call(self.tokens, self.sigma)
        self.assertNotIn('forward', self.model.__dict__)
        self.assertNotIn('forward', self.model.student_ema.__dict__)


if __name__ == '__main__':
    unittest.main(verbosity=2)
