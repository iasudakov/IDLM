"""CPU regressions for TinyGSM teacher/policy corrections; no downloads.

Run this script in its own process, as with the existing DiDi runtime tests.
The toy backbone is instrumented for sigma and dropout, which the older
prompt-mask tests intentionally do not model.
"""

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

import test_didi_tinygsm_runtime as tiny_fixtures
from didi_repro.runtime import verify_upstream
from didi_repro.tinygsm_runtime import TRAINING_CORRECTIONS


class TinyGSMTrainingCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiny_fixtures.TinyGSMRuntimeTests.setUpClass()
        cls.fixture = tiny_fixtures.TinyGSMRuntimeTests()
        cls.upstream = cls.fixture.upstream

    @classmethod
    def tearDownClass(cls):
        tiny_fixtures.TinyGSMRuntimeTests.tearDownClass()

    def setUp(self):
        torch.manual_seed(17)

    def model(self, time_conditioning=False):
        config = self.fixture.config()
        config.algo.time_conditioning = time_conditioning
        model = self.fixture.fixture.model(config)
        self.fixture.fixture.initialize(model)
        return model

    def masked_batch(self, model):
        batch = self.fixture.batch(True)
        target = batch['attention_mask'].bool()
        tokens = torch.where(target, model.mask_index, batch['input_ids'])
        return tokens, target

    @staticmethod
    def constant_head(backbone, mask_index):
        with torch.no_grad():
            head = backbone.output_layer.linear
            head.weight.zero_()
            head.bias.copy_(torch.linspace(-.4, .4, head.bias.numel()))
            head.bias[mask_index] = 80.0

    def test_teacher_sigma_matches_student_for_conditioned_and_unconditioned_models(self):
        for conditioned in (False, True):
            model = self.model(time_conditioning=conditioned)
            tokens, _ = self.masked_batch(model)
            for sigma in (torch.tensor([1.25, 3.5]),
                          torch.tensor([[1.0, 1.5], [3.0, 4.0]])):
                with self.subTest(time_conditioning=conditioned, sigma_shape=tuple(sigma.shape)):
                    with mock.patch.object(model.teacher, 'forward',
                                           wraps=model.teacher.forward) as forward:
                        result = model._tinygsm_teacher_log_probs(tokens, sigma)
                    arguments, keywords = forward.call_args
                    received = keywords.get('sigma', arguments[1] if len(arguments) > 1 else None)
                    expected = torch.tensor([1.25, 3.5]) if conditioned else torch.zeros(2)
                    torch.testing.assert_close(received, expected, rtol=0, atol=0)
                    self.assertFalse(result.requires_grad)

    def test_teacher_subs_excludes_mask_and_preserves_visible_tokens(self):
        model = self.model()
        tokens, target = self.masked_batch(model)
        self.constant_head(model.teacher, model.mask_index)
        with torch.no_grad():
            raw = model.teacher(tokens, torch.zeros(2)).softmax(-1)
        self.assertGreater(float(raw[..., model.mask_index].min()), .99)
        probability = model._tinygsm_teacher_log_probs(tokens, torch.ones(2)).exp()
        self.assertTrue(bool((probability[..., model.mask_index] == 0).all()))
        torch.testing.assert_close(probability.sum(-1), torch.ones_like(tokens, dtype=torch.float32))
        fixed = F.one_hot(tokens[~target], num_classes=model.vocab_size).float()
        torch.testing.assert_close(probability[~target], fixed, rtol=0, atol=0)

    def test_train_mode_keeps_frozen_teacher_and_ema_in_eval(self):
        model = self.model()
        self.assertIs(model.train(), model)
        self.assertTrue(model.backbone.training)
        self.assertTrue(model.discriminator.training)
        self.assertTrue(all(not part.training for part in model.teacher.modules()))
        self.assertTrue(all(not part.training for part in model.student_ema.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.teacher.parameters()))
        tokens, _ = self.masked_batch(model)
        model.teacher.train()  # Defensive helper invariant, even after direct submodule changes.
        result = model._tinygsm_teacher_log_probs(tokens, torch.ones(2))
        self.assertFalse(model.teacher.training)
        self.assertFalse(result.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.teacher.parameters()))

    def test_identical_native_distributions_have_zero_kl_even_with_large_raw_mask_logit(self):
        model = self.model().eval()
        self.constant_head(model.backbone, model.mask_index)
        model.teacher.load_state_dict(model.backbone.state_dict())
        first, first_mask = self.masked_batch(model)
        second = first.clone()
        second[:, 6] = 1
        second_mask = second == model.mask_index
        sigma_hi = torch.full((2, 1), 6.9)
        sigma_lo = torch.full((2,), .7)
        logits_hi = (model.forward(first, sigma_hi).exp() + 1e-10).log()
        logits_lo = (model.forward(second, sigma_lo[:, None]).exp() + 1e-10).log()
        loss, entropy = model._get_regularization(
            2, first, second, sigma_lo,
            logits_hi, first_mask, logits_lo, second_mask)
        self.assertLess(abs(float(loss)), 1e-7)
        self.assertGreater(float(entropy), 0)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.teacher.parameters()))
        gradients = [parameter.grad for parameter in model.backbone.parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
        self.assertLess(max(float(value.abs().max()) for value in gradients), 1e-6)

    def test_regularization_uses_corrected_teacher_for_both_trajectory_states(self):
        model = self.model()
        first, first_mask = self.masked_batch(model)
        second = first.clone()
        second[:, 6] = 1
        second_mask = second == model.mask_index
        logits_hi = torch.randn(2, 12, model.vocab_size, requires_grad=True)
        logits_lo = torch.randn_like(logits_hi, requires_grad=True)
        with mock.patch.object(model, '_tinygsm_teacher_log_probs',
                               wraps=model._tinygsm_teacher_log_probs) as teacher:
            loss, entropy = model._get_regularization(
                2, first, second, torch.full((2,), .7),
                logits_hi, first_mask, logits_lo, second_mask)
        self.assertEqual(teacher.call_count, 2)
        self.assertIs(teacher.call_args_list[0].args[0], first)
        self.assertIs(teacher.call_args_list[1].args[0], second)
        (loss - .0005 * entropy).backward()
        for gradient, selected in ((logits_hi.grad, first_mask),
                                   (logits_lo.grad, second_mask)):
            self.assertTrue(bool((gradient[~selected] == 0).all()))
            self.assertGreater(float(gradient[selected].abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.teacher.parameters()))

    def test_dropout_sampling_and_live_probabilities_share_one_forward(self):
        model = self.model().train()
        tokens, target = self.masked_batch(model)
        tokens[:, 6] = 1  # Include an already-unmasked answer token as well as the prompt.
        original_forward = model.backbone.forward
        sampled_probabilities = []

        def dropout_forward(inputs, sigma):
            return F.dropout(original_forward(inputs, sigma), p=.5, training=True)

        def select_nonmask(probability):
            self.assertFalse(probability.requires_grad)
            sampled_probabilities.append(probability.clone())
            candidates = probability.clone()
            candidates[..., model.mask_index] = -1
            return candidates.argmax(-1)

        times, intervals = torch.tensor([.7, .8]), torch.tensor([.4, .3])
        with mock.patch.object(model.backbone, 'forward', side_effect=dropout_forward) as forward, \
                mock.patch.object(self.upstream.algo.trainer_base, 'sample_categorical',
                                  side_effect=select_nonmask):
            probability, generated = model._get_ancestral_update(tokens, times, intervals)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(len(sampled_probabilities), 1)
        self.assertTrue(probability.requires_grad)
        _, alpha_t = model.noise(times)
        _, alpha_s = model.noise(times - intervals)
        expected = probability.detach() * (alpha_s - alpha_t)[:, None, None]
        expected[..., model.mask_index] = (1 - alpha_s)[:, None]
        torch.testing.assert_close(sampled_probabilities[0], expected, rtol=0, atol=0)
        visible = tokens != model.mask_index
        torch.testing.assert_close(generated[visible], tokens[visible], rtol=0, atol=0)
        torch.testing.assert_close(generated[~target], tokens[~target], rtol=0, atol=0)
        score = (probability + 1e-10).log().gather(-1, generated.unsqueeze(-1)).squeeze(-1)
        loss = -(score * ~visible).sum()
        loss.backward()
        gradients = [parameter.grad for parameter in model.backbone.parameters()
                     if parameter.grad is not None]
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0)

    def test_supplied_probabilities_skip_forward_and_preserve_noise_removal_equations(self):
        model = self.model()
        tokens, _ = self.masked_batch(model)
        times, intervals = torch.tensor([.7, .8]), torch.tensor([.4, .3])
        for noise_removal in (False, True):
            for live in (False, True):
                with self.subTest(noise_removal=noise_removal, live=live):
                    logits = torch.randn(2, 12, model.vocab_size, requires_grad=live)
                    probability = model._process_model_output(logits.clone(), tokens, None).exp()
                    original = probability.detach().clone()
                    captured = []

                    def sample(q):
                        captured.append(q.clone())
                        return torch.ones_like(tokens)

                    with mock.patch.object(model, 'forward', side_effect=AssertionError('unexpected forward')), \
                            mock.patch.object(self.upstream.algo.trainer_base, 'sample_categorical',
                                              side_effect=sample):
                        returned, generated = model._get_ancestral_update(
                            tokens, times, intervals, p_x0=probability,
                            noise_removal_step=noise_removal)
                    self.assertIs(returned, probability)
                    torch.testing.assert_close(probability, original, rtol=0, atol=0)
                    _, alpha_t = model.noise(times)
                    alpha_s = torch.ones_like(alpha_t) if noise_removal else model.noise(times - intervals)[1]
                    expected = original * (alpha_s - alpha_t)[:, None, None]
                    expected[..., model.mask_index] = (1 - alpha_s)[:, None]
                    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
                    self.assertFalse(captured[0].requires_grad)
                    visible = tokens != model.mask_index
                    torch.testing.assert_close(generated[visible], tokens[visible], rtol=0, atol=0)
                    if noise_removal:
                        self.assertTrue(bool((captured[0][..., model.mask_index] == 0).all()))

    def test_legacy_training_checkpoint_cannot_silently_resume_changed_objective(self):
        model = self.model()
        checkpoint = {'didi_tinygsm': model._conditional_checkpoint_metadata()}
        with self.assertRaisesRegex(ValueError, 'training corrections differ'):
            model.on_load_checkpoint(checkpoint)
        checkpoint['didi_tinygsm_training'] = {
            'corrections': dict(TRAINING_CORRECTIONS, version=0)}
        with self.assertRaisesRegex(ValueError, 'training corrections differ'):
            model.on_load_checkpoint(checkpoint)
        verify_upstream()


if __name__ == '__main__':
    unittest.main()
