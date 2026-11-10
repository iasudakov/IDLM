"""Real small DiT CPU checks for the opt-in full-FP32 forward/backward path."""

from contextlib import nullcontext
import math
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from models.dit import (DIT, Rotary, regular_attention_multi_headed,
                        split_and_apply_rotary_pos_emb,
                        split_and_apply_rotary_pos_emb_fp32)


def config(force_fp32=True, causal=False):
    return OmegaConf.create({
        'algo': {'causal_attention': causal},
        'model': {'hidden_size': 16, 'cond_dim': 8, 'n_blocks': 2,
                  'n_heads': 2, 'dropout': 0., 'scale_by_sigma': True,
                  'force_fp32': force_fp32},
    })


def initialized_model(force_fp32=True):
    model = DIT(config(force_fp32), vocab_size=13)
    # DiT initializes the output projection and conditioning gates to zero.
    # Make the test exercise gradients all the way through attention/embedding.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'adaLN_modulation' in name or name.startswith('output_layer.linear'):
                parameter.normal_(std=.07)
    return model


class FullFP32DiTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)

    def test_complete_forward_and_backward_stay_fp32_under_outer_autocast(self):
        model = initialized_model()
        tokens = torch.randint(0, 13, (2, 6))
        sigma = torch.tensor([.25, .75])
        for training in (True, False):
            for autocast in (False, True):
                with self.subTest(training=training, outer_cpu_bf16=autocast):
                    model.train(training)
                    observed = []

                    def capture(module, inputs, output):
                        values = output if isinstance(output, tuple) else (output,)
                        observed.extend((type(module).__name__, value.dtype)
                                        for value in values
                                        if isinstance(value, torch.Tensor)
                                        and value.is_floating_point())

                    handles = [module.register_forward_hook(capture)
                               for module in model.modules()]
                    context = (torch.autocast('cpu', dtype=torch.bfloat16)
                               if autocast else nullcontext())
                    try:
                        with context:
                            output = model(tokens, sigma)
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertTrue(observed)
                    self.assertTrue(all(dtype == torch.float32 for _, dtype in observed), observed)
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(output).all())
                    model.zero_grad(set_to_none=True)
                    output.square().mean().backward()
                    for name, parameter in model.named_parameters():
                        self.assertEqual(parameter.dtype, torch.float32, name)
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertEqual(parameter.grad.dtype, torch.float32, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    for name in ('vocab_embed.embedding', 'blocks.0.attn_qkv.weight',
                                 'sigma_map.mlp.0.weight', 'output_layer.linear.weight'):
                        self.assertGreater(dict(model.named_parameters())[name].grad.abs().max().item(), 0)

    def test_checkpoint_keys_and_forward_match_existing_full_attention(self):
        original = initialized_model(force_fp32=False).eval()
        fp32 = initialized_model().eval()
        self.assertEqual(set(original.state_dict()), set(fp32.state_dict()))
        fp32.load_state_dict(original.state_dict(), strict=True)
        tokens = torch.randint(0, 13, (2, 6))
        sigma = torch.tensor([.25, .75])
        # CPU execution of the original path disables its CUDA-only autocast,
        # so it serves as an FP32 reference with the original rotary helper.
        with torch.no_grad():
            expected = original(tokens, sigma)
            actual = fp32(tokens, sigma)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_fp32_rotary_matches_existing_helper_and_its_gradient(self):
        qkv = torch.randn(2, 6, 3, 2, 8, requires_grad=True)
        rotary = Rotary(8)(torch.zeros(2, 6, 16))
        expected = split_and_apply_rotary_pos_emb(qkv, rotary)
        actual = split_and_apply_rotary_pos_emb_fp32(qkv, rotary)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        weights = [torch.randn_like(value) for value in actual]
        actual_grad, = torch.autograd.grad(sum((value * weight).sum()
            for value, weight in zip(actual, weights)), qkv)
        expected_grad, = torch.autograd.grad(sum((value * weight).sum()
            for value, weight in zip(expected, weights)), qkv)
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'FP32 Q/K/V'):
            split_and_apply_rotary_pos_emb_fp32(qkv.bfloat16(), rotary)

    def test_attention_uses_math_backend_and_matches_dense_reference(self):
        q, k, v = [torch.randn(2, 6, 2, 8, requires_grad=True) for _ in range(3)]
        original_sdpa = F.scaled_dot_product_attention
        before = (torch.backends.cuda.flash_sdp_enabled(),
                  torch.backends.cuda.mem_efficient_sdp_enabled(),
                  torch.backends.cuda.math_sdp_enabled())

        def checked_sdpa(*args, **kwargs):
            self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
            self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
            self.assertTrue(torch.backends.cuda.math_sdp_enabled())
            return original_sdpa(*args, **kwargs)

        with patch('models.dit.F.scaled_dot_product_attention', side_effect=checked_sdpa):
            actual = regular_attention_multi_headed(q, k, v, force_fp32=True)
        self.assertEqual(before, (torch.backends.cuda.flash_sdp_enabled(),
                                  torch.backends.cuda.mem_efficient_sdp_enabled(),
                                  torch.backends.cuda.math_sdp_enabled()))
        scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)
        expected = (torch.softmax(scores / math.sqrt(q.shape[-1]), dim=-1)
                    @ v.transpose(1, 2)).transpose(1, 2).reshape(2, 6, 16)
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-6)
        weight = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual * weight).sum(), (q, k, v))
        expected_grads = torch.autograd.grad((expected * weight).sum(), (q, k, v))
        for left, right in zip(actual_grads, expected_grads):
            torch.testing.assert_close(left, right, atol=8e-7, rtol=1e-5)

    def test_default_preserves_existing_mode_and_unsupported_mode_fails(self):
        cfg = config()
        del cfg.model.force_fp32
        model = DIT(cfg, 13)
        self.assertFalse(model.force_fp32)
        self.assertTrue(all(not block.force_fp32 for block in model.blocks))
        with self.assertRaisesRegex(ValueError, 'full-attention'):
            DIT(config(causal=True), 13)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            DIT(config(force_fp32='true'), 13)


if __name__ == '__main__':
    unittest.main()
