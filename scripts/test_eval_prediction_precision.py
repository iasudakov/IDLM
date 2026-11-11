"""Offline CPU regression checks for FP32 normalization, without model downloads."""

import ast
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from eval_precision import capture_rng_state, fp32_prediction_normalization, replay_rng_state


def source_method(path, class_name, method_name, namespace=None):
    """Execute the actual method without importing GPU-only model dependencies."""
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {'torch': torch, **(namespace or {})}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method_name]


class Model(torch.nn.Module):
    _process_model_output = source_method(ROOT / 'algo.py', 'MDLM', '_process_model_output')
    mask_index = 4
    neg_infinity = -1e6

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([8., 5.5, -20., -20., -30.]))
        self.forward_dtypes = []

    def forward(self, xt, sigma=None):
        raw = self.weight.to(torch.bfloat16)[None, None].expand(*xt.shape, -1).clone()
        self.forward_dtypes.append(raw.dtype)
        return self._process_model_output(raw, xt, sigma)


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cast_precedes_normalization_and_changes_threshold_decision(self):
        model = Model()
        tokens = torch.tensor([[0, 4, 4]])
        # Outer autocast must not pull the normalization back into BF16.
        with torch.autocast('cpu', dtype=torch.bfloat16):
            legacy = model(tokens).exp()
            with fp32_prediction_normalization(model):
                precise = model(tokens).exp()
        self.assertEqual(legacy.dtype, torch.bfloat16)
        self.assertEqual(precise.dtype, torch.float32)
        self.assertGreater(legacy[0, 1, 0].item(), .925)
        self.assertLess(precise[0, 1, 0].item(), .925)
        torch.testing.assert_close(precise.sum(-1), torch.ones(1, 3))
        self.assertEqual(precise[0, 0, 0].item(), 1.)
        self.assertEqual(precise[..., 4].count_nonzero().item(), 0)
        self.assertTrue(all(dtype == torch.bfloat16 for dtype in model.forward_dtypes))
        self.assertTrue(model.training)  # No dependence on Lightning's training flag.
        self.assertNotIn('_process_model_output', vars(model))
        torch.testing.assert_close(model(tokens).exp(), legacy, rtol=0, atol=0)

    def test_exception_nested_context_and_training_gradient_restoration(self):
        model = Model()
        tokens = torch.full((1, 3), 4)
        (-model(tokens)[..., 0].mean()).backward()
        expected = model.weight.grad.clone()
        model.weight.grad = None
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with fp32_prediction_normalization(model):
                outer = model._process_model_output
                with fp32_prediction_normalization(model):
                    self.assertEqual(model(tokens).dtype, torch.float32)
                self.assertIs(model._process_model_output, outer)
                raise RuntimeError('test failure')
        self.assertNotIn('_process_model_output', vars(model))
        (-model(tokens)[..., 0].mean()).backward()
        torch.testing.assert_close(model.weight.grad, expected, rtol=0, atol=0)

    def test_paired_rng_does_not_advance_training_even_on_failure(self):
        def draw():
            return random.random(), np.random.rand(), torch.rand(2)
        def same(left, right):
            self.assertEqual(left[:2], right[:2])
            torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)
        initial = capture_rng_state()
        first = draw()
        after_legacy = capture_rng_state()
        expected_next = draw()
        with replay_rng_state(after_legacy):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with replay_rng_state(initial):
                    same(draw(), first)
                    draw()
                    raise RuntimeError('test failure')
            same(draw(), expected_next)


if __name__ == '__main__':
    unittest.main(verbosity=2)
