"""CPU regressions for cosine MMD helpers; no models or downloads required.

Run from the repository root: python scripts/test_cosine_kernel.py
"""

import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mmd_cosine


def normalized(features):
    # The reference deliberately uses float64 and the direct definition rather
    # than the production max-scaling normalization or centered-moment algebra.
    values = features.detach().double()
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def dense_unequal(real, fake, real_mask, fake_mask):
    real, fake = normalized(real), normalized(fake)
    estimates, active = [], []
    for xs, ys, wx, wy in zip(real, fake, real_mask, fake_mask):
        xs, ys = xs[wx.bool()], ys[wy.bool()]
        n, m = len(xs), len(ys)
        enabled = n >= 2 and m >= 2
        active.append(enabled)
        if not enabled:
            estimates.append(0.0)
            continue
        xx, yy, xy = xs @ xs.T, ys @ ys.T, xs @ ys.T
        estimates.append(((xx.sum() - xx.diag().sum()) / (n * (n - 1))
                          + (yy.sum() - yy.diag().sum()) / (m * (m - 1))
                          - 2 * xy.mean()).item())
    return torch.tensor(estimates), torch.tensor(active)


def dense_fixed(real, fake):
    real, fake = normalized(real), normalized(fake)
    estimates = []
    for x, ys in zip(real, fake):
        count = len(ys)
        yy = ys @ ys.T
        fake_term = yy[0, 0] if count == 1 else (
            yy.sum() - yy.diag().sum()) / (count * (count - 1))
        estimates.append((x.dot(x) + fake_term - 2 * (ys @ x).mean()).item())
    return torch.tensor(estimates)


class CosineKernelTests(unittest.TestCase):
    def test_unequal_masked_sets_match_dense_gram_reference(self):
        rng = torch.Generator().manual_seed(611)
        real = torch.randn(4, 5, 7, generator=rng)
        fake = torch.randn(4, 11, 7, generator=rng)
        real_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 1, 0, 1, 1],
                                  [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        fake_mask = torch.ones(4, 11, dtype=torch.bool)
        fake_mask[1, 1::2] = False
        actual, active = mmd_cosine.cosine_mmd2_unequal(real, fake, real_mask, fake_mask)
        expected, expected_active = dense_unequal(real, fake, real_mask, fake_mask)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(active, expected_active)

    def test_fixed_real_k1_k2_k4_match_dense_gram_reference(self):
        rng = torch.Generator().manual_seed(612)
        real = torch.randn(3, 8, generator=rng)
        fake = torch.randn(3, 4, 8, generator=rng)
        for count in (1, 2, 4):
            with self.subTest(count=count):
                actual = mmd_cosine.cosine_mmd2_fixed_real(real, fake[:, :count])
                torch.testing.assert_close(actual, dense_fixed(real, fake[:, :count]),
                                           rtol=2e-5, atol=1e-7)

    def test_k1_is_squared_unit_distance_not_cosine_distance_or_dimension_mean(self):
        real = torch.tensor([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        fake = torch.tensor([[[0., 1., 0.]], [[-1., 0., 0.]], [[2., 0., 0.]]])
        torch.testing.assert_close(mmd_cosine.cosine_mmd2_fixed_real(real, fake),
                                   torch.tensor([2., 4., 0.]), rtol=0, atol=0)

    def test_each_feature_is_invariant_to_positive_rescaling_above_epsilon(self):
        rng = torch.Generator().manual_seed(613)
        real = torch.randn(2, 4, 5, generator=rng)
        fake = torch.randn(2, 9, 5, generator=rng)
        real_scale = torch.exp(torch.randn(2, 4, 1, generator=rng) * 3)
        fake_scale = torch.exp(torch.randn(2, 9, 1, generator=rng) * 3)
        wx, wy = torch.ones(2, 4), torch.ones(2, 9)
        expected, _ = mmd_cosine.cosine_mmd2_unequal(real, fake, wx, wy)
        actual, _ = mmd_cosine.cosine_mmd2_unequal(real * real_scale, fake * fake_scale, wx, wy)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
        expected = mmd_cosine.cosine_mmd2_fixed_real(real[:, 0], fake[:, :4])
        actual = mmd_cosine.cosine_mmd2_fixed_real(
            real[:, 0] * real_scale[:, 0], fake[:, :4] * fake_scale[:, :4])
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)

    def test_large_finite_features_do_not_overflow_normalization(self):
        real = torch.tensor([[[1., 2., 3.], [-3., -2., -1.]]]) * 1e38
        fake = torch.tensor([[[3., 2., 1.], [-1., -2., -3.], [2., -1., 2.]]]) * 1e38
        wx, wy = torch.ones(1, 2), torch.ones(1, 3)
        self.assertTrue(torch.isfinite(real).all() and torch.isfinite(fake).all())
        actual, _ = mmd_cosine.cosine_mmd2_unequal(real, fake, wx, wy)
        expected, _ = dense_unequal(real, fake, wx, wy)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
        fixed = mmd_cosine.cosine_mmd2_fixed_real(real[:, 0], fake)
        torch.testing.assert_close(fixed, dense_fixed(real[:, 0], fake), rtol=2e-5, atol=1e-7)

    def test_zero_and_sub_epsilon_vectors_use_actual_self_norms(self):
        real = torch.tensor([[[1., 0.], [0., 0.]], [[1e-14, 0.], [0., 0.]]])
        fake = torch.tensor([[[0., 1.], [0., 0.]], [[0., 1e-14], [0., 0.]]])
        mask = torch.ones(2, 2)
        actual, active = mmd_cosine.cosine_mmd2_unequal(real, fake, mask, mask)
        expected, _ = dense_unequal(real, fake, mask, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
        self.assertTrue(active.all())
        torch.testing.assert_close(actual, torch.zeros(2), rtol=0, atol=1e-7)
        fixed_real = torch.tensor([[0., 0.], [1e-14, 0.], [0., 0.]])
        fixed_fake = torch.tensor([[[1., 0.]], [[0., 1e-14]], [[0., 0.]]])
        actual = mmd_cosine.cosine_mmd2_fixed_real(fixed_real, fixed_fake)
        torch.testing.assert_close(actual, dense_fixed(fixed_real, fixed_fake), rtol=2e-5, atol=1e-9)
        torch.testing.assert_close(actual, torch.tensor([1., 2e-4, 0.]), rtol=2e-5, atol=1e-9)
        for count in (2, 4):
            zeros = mmd_cosine.cosine_mmd2_fixed_real(torch.zeros(1, 3), torch.zeros(1, count, 3))
            torch.testing.assert_close(zeros, torch.zeros(1), rtol=0, atol=0)

    def test_unbiased_negative_estimates_are_not_clamped(self):
        values = torch.tensor([[[1., 0.], [-1., 0.]]])
        mask = torch.ones(1, 2)
        actual, active = mmd_cosine.cosine_mmd2_unequal(values, values, mask, mask)
        torch.testing.assert_close(actual, torch.tensor([-2.]), rtol=0, atol=0)
        self.assertTrue(active.item())
        real = torch.tensor([[1., 0.]])
        fake = torch.tensor([[[0.5, math.sqrt(0.75)], [0.5, -math.sqrt(0.75)]]])
        actual = mmd_cosine.cosine_mmd2_fixed_real(real, fake)
        torch.testing.assert_close(actual, torch.tensor([-0.5]), rtol=1e-6, atol=1e-7)

    def test_mask_count_not_draw_count_controls_unequal_activity(self):
        real, fake = torch.ones(3, 2, 2), torch.ones(3, 4, 2)
        wx = torch.tensor([[1, 0], [1, 1], [0, 0]])
        wy = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
        actual, active = mmd_cosine.cosine_mmd2_unequal(real, fake, wx, wy)
        torch.testing.assert_close(actual, torch.zeros(3), rtol=0, atol=0)
        self.assertEqual(active.tolist(), [False, False, False])
        actual, active = mmd_cosine.cosine_mmd2_unequal(
            torch.empty(2, 0, 3), torch.ones(2, 2, 3), torch.empty(2, 0), torch.ones(2, 2))
        torch.testing.assert_close(actual, torch.zeros(2), rtol=0, atol=0)
        self.assertFalse(active.any())

    def test_float32_detached_results_for_all_feature_precisions(self):
        rng = torch.Generator().manual_seed(614)
        real = torch.randn(2, 3, 4, generator=rng)
        fake = torch.randn(2, 5, 4, generator=rng)
        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x = real.to(dtype).detach().clone().requires_grad_()
                y = fake.to(dtype).detach().clone().requires_grad_()
                actual, _ = mmd_cosine.cosine_mmd2_unequal(x, y, torch.ones(2, 3), torch.ones(2, 5))
                fixed = mmd_cosine.cosine_mmd2_fixed_real(x[:, 0], y)
                for value in (actual, fixed):
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertFalse(value.requires_grad)
                    self.assertIsNone(value.grad_fn)
                self.assertIsNone(x.grad)
                self.assertIsNone(y.grad)

    def test_no_pairwise_gram_operations_are_needed(self):
        with mock.patch.object(torch, 'bmm', side_effect=AssertionError('unexpected Gram matrix')), \
                mock.patch.object(torch, 'matmul', side_effect=AssertionError('unexpected Gram matrix')), \
                mock.patch.object(torch, 'cdist', side_effect=AssertionError('unexpected pair distances')):
            real, fake = torch.ones(1, 3, 4), torch.ones(1, 7, 4)
            result, _ = mmd_cosine.cosine_mmd2_unequal(real, fake, torch.ones(1, 3), torch.ones(1, 7))
            torch.testing.assert_close(result, torch.zeros(1), rtol=0, atol=0)
            result = mmd_cosine.cosine_mmd2_fixed_real(real[:, 0], fake)
            torch.testing.assert_close(result, torch.zeros(1), rtol=0, atol=0)

    def test_invalid_shapes_empty_draws_and_nonbinary_masks_are_rejected(self):
        real, fake = torch.ones(1, 2, 3), torch.ones(1, 4, 3)
        cases = [(real[:, 0], fake, torch.ones(1, 2), torch.ones(1, 4)),
                 (real, fake[:, 0], torch.ones(1, 2), torch.ones(1, 4)),
                 (real, fake.repeat(2, 1, 1), torch.ones(1, 2), torch.ones(2, 4)),
                 (real, fake[..., :2], torch.ones(1, 2), torch.ones(1, 4)),
                 (real, fake, torch.ones(1, 1), torch.ones(1, 4)),
                 (real, fake, torch.ones(1, 2), torch.ones(1, 3))]
        for arguments in cases:
            with self.subTest(shapes=[tuple(x.shape) for x in arguments]), self.assertRaises(ValueError):
                mmd_cosine.cosine_mmd2_unequal(*arguments)
        for invalid in (0.5, -1., float('nan'), float('inf')):
            with self.subTest(weight=invalid), self.assertRaisesRegex(ValueError, 'binary'):
                mmd_cosine.cosine_mmd2_unequal(real, fake, torch.tensor([[1., invalid]]), torch.ones(1, 4))
        for x, y in ((real, fake), (real[:, 0], fake[:, 0]),
                     (real[:, 0], fake[:, :0]), (real[:, 0], fake[..., :2])):
            with self.subTest(real_shape=tuple(x.shape), fake_shape=tuple(y.shape)), self.assertRaises(ValueError):
                mmd_cosine.cosine_mmd2_fixed_real(x, y)


if __name__ == '__main__':
    unittest.main(verbosity=2)
