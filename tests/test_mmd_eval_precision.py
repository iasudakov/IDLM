"""CPU regression for CDLM/MMD confidence parity and unchanged MMD training."""

from contextlib import nullcontext
import tempfile
import unittest
from unittest.mock import patch

import torch

from algo import MDLM_with_AR
from cdlm import CDLM
from mmd_finetune import MMDFinetune
from sdtt_eval import generate_samples
from test_cdlm_training import Tokenizer
from test_mmd_finetune import config


class BF16Denoiser(torch.nn.Module):
    """Fixed logits near a confidence threshold, returned in DiT's output dtype."""

    def __init__(self, config, vocab_size):
        super().__init__()
        logits = torch.full((vocab_size,), -20.)
        logits[:2] = torch.tensor([8., 5.5])
        self.logits = torch.nn.Parameter(logits)

    def forward(self, tokens, sigma):
        return self.logits.to(torch.bfloat16).expand(*tokens.shape, -1).clone()


class EvaluationPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('models.dit.DIT', BF16Denoiser))
        directory = self.enterContext(tempfile.TemporaryDirectory())
        cfg = config(directory)
        self.mmd = MMDFinetune(cfg, Tokenizer(), inference_only=True)
        self.cdlm = CDLM(cfg, Tokenizer(), inference_only=True)
        self.tokens = torch.tensor([[0, 1, self.mmd.mask_index,
                                     self.mmd.mask_index, self.mmd.mask_index,
                                     self.mmd.mask_index]])
        self.sigma = torch.zeros(1, 1)

    def test_step_zero_eval_uses_backbone_mode_and_matches_cdlm_probabilities(self):
        # Initial/final evaluation switches only backbone/noise via _eval_mode.
        # The LightningModule itself deliberately remains in training mode.
        for model in (self.mmd, self.cdlm):
            self.assertTrue(model.training)
            model._eval_mode()
            self.assertTrue(model.training)
            self.assertFalse(model.backbone.training)
        mmd = self.mmd(self.tokens, self.sigma)
        cdlm = self.cdlm(self.tokens, self.sigma)
        self.assertEqual(mmd.dtype, torch.float32)
        torch.testing.assert_close(mmd, cdlm, atol=0, rtol=0)
        probs = mmd.exp()
        torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs[..., 0]),
                                   atol=2e-7, rtol=0)
        self.assertAlmostEqual(probs[0, 2].max().item(), .9241417, places=6)
        self.assertLess(probs[0, 2].max().item(), .925)

    def test_shared_confidence_sampler_matches_tokens_and_nfe_in_both_eval_modes(self):
        prompts = [[0, 1], [0, 1, 2]]
        for mode in ('_eval_mode', 'eval'):
            for autocast in (False, True):
                with self.subTest(mode=mode, cpu_bf16_autocast=autocast):
                    for model in (self.mmd, self.cdlm):
                        getattr(model, mode)()
                    context = (torch.autocast('cpu', dtype=torch.bfloat16)
                               if autocast else nullcontext())
                    with context:
                        actual, actual_meta = generate_samples(self.mmd, prompts, .925)
                        expected, expected_meta = generate_samples(self.cdlm, prompts, .925)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.tolist(), [[0, 1, 0, 0, 0, 0],
                                                       [0, 1, 2, 0, 0, 0]])
                    self.assertEqual(actual_meta['nfe'], expected_meta['nfe'])
                    self.assertEqual(actual_meta['nfe'], 4)
                    self.assertEqual(actual_meta['nfe_per_row'].tolist(), [4, 3])
                    self.assertTrue(torch.equal(actual_meta['nfe_per_row'],
                                                expected_meta['nfe_per_row']))
                    if mode == '_eval_mode':
                        for model in (self.mmd, self.cdlm):
                            model._train_mode()

    def test_return_from_eval_preserves_bf16_training_even_under_no_grad(self):
        # Repeated initial/final-style evaluations must not leave training in FP32.
        for _ in range(2):
            self.mmd._eval_mode()
            self.assertEqual(self.mmd(self.tokens, self.sigma).dtype, torch.float32)
            self.mmd._train_mode()
            self.assertTrue(self.mmd.backbone.training)
            with torch.no_grad():
                actual = self.mmd(self.tokens, self.sigma)
                expected = MDLM_with_AR._process_model_output(
                    self.mmd, self.mmd.backbone(self.tokens, self.sigma),
                    self.tokens, self.sigma)
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(actual.exp()[0, 2].max().item(), .9375)

    def test_training_values_and_gradients_match_original_mdlm_normalization(self):
        self.mmd.train()
        actual = self.mmd(self.tokens, self.sigma)
        expected = MDLM_with_AR._process_model_output(
            self.mmd, self.mmd.backbone(self.tokens, self.sigma),
            self.tokens, self.sigma)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        mask = self.tokens.eq(self.mmd.mask_index)
        actual_grad, = torch.autograd.grad(actual[..., 0][mask].sum(),
                                           self.mmd.backbone.logits)
        expected_grad, = torch.autograd.grad(expected[..., 0][mask].sum(),
                                             self.mmd.backbone.logits)
        self.assertGreater(actual_grad.abs().max().item(), 0)
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
