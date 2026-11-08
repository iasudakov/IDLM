"""CPU references for removing only the real-real component of RBF MMD.

Run with: OMP_NUM_THREADS=2 python -B scripts/test_rbf_omit_real_real.py
"""

from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mmd_rbf


def dense_kernel(x, y, alpha, distance, delta):
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    if distance == 'squared_euclidean':
        d = diff.square().sum(-1)
    else:
        d = (torch.sqrt(diff.square() + delta ** 2) - delta).sum(-1)
    return torch.exp(-alpha * d)


def dense_unequal_without_real(real, fake, real_mask, fake_mask, alpha, distance, delta):
    estimates = []
    for x, y, wx, wy in zip(real.double(), fake.double(), real_mask, fake_mask):
        x, y = x[wx.bool()], y[wy.bool()]
        if len(x) < 2 or len(y) < 2:
            estimates.append(torch.tensor(0., dtype=torch.float64))
            continue
        yy = dense_kernel(y, y, alpha, distance, delta)
        xy = dense_kernel(x, y, alpha, distance, delta)
        estimates.append((yy.sum() - yy.diag().sum()) / (len(y) * (len(y) - 1))
                         - 2 * xy.mean())
    return torch.stack(estimates)


class OmitRealRealRBFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        gen = torch.Generator().manual_seed(891)
        self.real = torch.randn(5, 5, 7, generator=gen)
        self.fake = torch.randn(5, 4, 5, 7, generator=gen)
        self.mask = torch.tensor([[1, 0, 1, 1, 1], [0, 1, 1, 0, 0],
                                  [1, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1]], dtype=torch.bool)
        self.alpha, self.delta = .13, .37

    def test_unequal_k1_k2_k4_masks_and_tiles_match_dense(self):
        for distance in ('squared_euclidean', 'pseudo_huber'):
            for k in (1, 2, 4):
                fake = self.fake[:, :k].flatten(1, 2)
                fake_mask = self.mask.repeat(1, k)
                # Independent mask counts, including an insufficient fake set.
                fake_mask[-1] = False
                fake_mask[-1, 0] = True
                expected = dense_unequal_without_real(
                    self.real, fake, self.mask, fake_mask, self.alpha, distance, self.delta)
                for tile in (0, 1, 3, 99):
                    with self.subTest(distance=distance, k=k, tile=tile):
                        value, active = mmd_rbf.rbf_mmd2_unequal(
                            self.real, fake, self.mask, fake_mask, self.alpha,
                            block_size=tile, distance=distance, huber_delta=self.delta,
                            omit_real_real=True)
                        torch.testing.assert_close(value.double(), expected, atol=2e-6, rtol=2e-5)
                        self.assertEqual(active.tolist(), [True, True, False, False, False])
                        self.assertFalse(value.requires_grad)

    def test_pair_sums_values_and_fake_gradients_are_unchanged(self):
        real, initial_fake = self.real.double(), self.fake[:, 0].double()
        weights = self.mask.double()
        for distance in ('squared_euclidean', 'pseudo_huber'):
            expected_y = initial_fake.clone().requires_grad_()
            pair_weights = weights[:, :, None] * weights[:, None, :]
            expected = tuple((dense_kernel(a, b, self.alpha, distance, self.delta)
                              * pair_weights).sum((1, 2))
                             for a, b in ((expected_y, expected_y), (real, expected_y)))
            (.3 * expected[0].sum() - .7 * expected[1].sum()).backward()
            for tile in (0, 1, 3, 99):
                with self.subTest(distance=distance, tile=tile):
                    full_y = initial_fake.clone().requires_grad_()
                    full = mmd_rbf.rbf_pair_sums(
                        real, full_y, weights, self.alpha, block_size=tile,
                        distance=distance, huber_delta=self.delta)
                    (full[0].sum() + .3 * full[1].sum() - .7 * full[2].sum()).backward()
                    x, y = real.clone().requires_grad_(), initial_fake.clone().requires_grad_()
                    omitted = mmd_rbf.rbf_pair_sums(
                        x, y, weights, self.alpha, block_size=tile, distance=distance,
                        huber_delta=self.delta, omit_real_real=True)
                    (omitted[0].sum() + .3 * omitted[1].sum() - .7 * omitted[2].sum()).backward()
                    torch.testing.assert_close(omitted[0], torch.zeros_like(omitted[0]), atol=0, rtol=0)
                    for got, unchanged, reference in zip(omitted[1:], full[1:], expected):
                        torch.testing.assert_close(got, unchanged, atol=0, rtol=0)
                        torch.testing.assert_close(got, reference, atol=2e-11, rtol=2e-10)
                    self.assertIsNone(x.grad)
                    torch.testing.assert_close(y.grad, full_y.grad, atol=0, rtol=0)
                    torch.testing.assert_close(y.grad, expected_y.grad, atol=2e-11, rtol=2e-9)

    def test_fixed_real_removes_only_reference_one_including_k1(self):
        real, fake = self.real[:, 0], self.fake[:, :, 0]
        for distance in ('squared_euclidean', 'pseudo_huber'):
            for k in (1, 2, 4):
                yy = dense_kernel(fake[:, :k].double(), fake[:, :k].double(),
                                  self.alpha, distance, self.delta)
                ff = (yy[:, 0, 0] if k == 1 else
                      (yy.sum((1, 2)) - yy.diagonal(dim1=1, dim2=2).sum(1)) / (k * (k - 1)))
                rf = dense_kernel(real.double().unsqueeze(1), fake[:, :k].double(),
                                  self.alpha, distance, self.delta).mean((1, 2))
                expected = ff - 2 * rf
                for tile in (0, 1, 3, 99):
                    with self.subTest(distance=distance, k=k, tile=tile):
                        kwargs = dict(block_size=tile, distance=distance, huber_delta=self.delta)
                        full = mmd_rbf.rbf_mmd2_fixed_real(real, fake[:, :k], self.alpha, **kwargs)
                        omitted = mmd_rbf.rbf_mmd2_fixed_real(
                            real, fake[:, :k], self.alpha, omit_real_real=True, **kwargs)
                        torch.testing.assert_close(omitted.double(), expected, atol=2e-6, rtol=2e-5)
                        torch.testing.assert_close(omitted, full - 1, atol=2e-7, rtol=2e-6)

    def test_real_real_kernel_blocks_are_not_evaluated(self):
        # Distinct first-coordinate markers identify the real and fake sets
        # even when low-level paths cast, detach, and slice their inputs.
        real, fake = self.real.clone(), self.fake[:, 0].clone()
        real[:, :, 0], fake[:, :, 0] = 10., -10.
        for distance, kernel_name in (('squared_euclidean', '_kernel_block'),
                                      ('pseudo_huber', '_pseudo_huber_kernel_block')):
            original = getattr(mmd_rbf, kernel_name)
            calls = []

            def observe(a, b, *args, **kwargs):
                rr = bool((a[..., 0] == 10).all() and (b[..., 0] == 10).all())
                calls.append(rr)
                return original(a, b, *args, **kwargs)

            for function, args in (
                    (mmd_rbf.rbf_pair_sums, (real, fake, self.mask.float(), self.alpha)),
                    (mmd_rbf.rbf_mmd2_unequal, (real, fake, self.mask, self.mask, self.alpha)),
                    (mmd_rbf.rbf_mmd2_fixed_real, (real[:, 0], fake[:, :2], self.alpha))):
                with self.subTest(distance=distance, function=function.__name__):
                    with mock.patch.object(mmd_rbf, kernel_name, side_effect=observe):
                        calls.clear()
                        function(*args, block_size=2, distance=distance, huber_delta=self.delta,
                                 omit_real_real=True)
                        self.assertGreater(len(calls), 0)
                        self.assertFalse(any(calls), 'no real-real kernel tile may be evaluated')
                        calls.clear()
                        function(*args, block_size=2, distance=distance, huber_delta=self.delta)
                        if distance == 'squared_euclidean' and function is mmd_rbf.rbf_mmd2_fixed_real:
                            self.assertFalse(any(calls))  # This path already uses analytic k(x,x)=1.
                        else:
                            self.assertTrue(any(calls), 'the detector must identify legacy RR work')

    def test_sharded_fake_cross_values_and_gradients_remain_equal(self):
        real, initial_fake, weights = self.real.double(), self.fake[:, 0].double(), self.mask.double()
        for distance in ('squared_euclidean', 'pseudo_huber'):
            kwargs = dict(block_size=2, distance=distance, huber_delta=self.delta, omit_real_real=True)
            whole_y = initial_fake.clone().requires_grad_()
            whole = mmd_rbf.rbf_pair_sums(real, whole_y, weights, self.alpha, **kwargs)
            (whole[1].sum() - 2 * whole[2].sum()).backward()
            partial_values, partial_gradients = [], []
            for rank in range(3):
                with mock.patch.object(mmd_rbf, '_shard_ctx', return_value=(rank, 3)), \
                        mock.patch.object(mmd_rbf.dist, 'all_reduce'):
                    y = initial_fake.clone().requires_grad_()
                    part = mmd_rbf.rbf_pair_sums(real, y, weights, self.alpha, shard=True, **kwargs)
                    (part[1].sum() - 2 * part[2].sum()).backward()
                    partial_values.append(torch.stack(part))
                    partial_gradients.append(y.grad)
                    full_part = mmd_rbf.rbf_pair_sums(
                        real, y.detach(), weights, self.alpha, shard=True,
                        **{**kwargs, 'omit_real_real': False})
                    for got, expected in zip(part[1:], full_part[1:]):
                        torch.testing.assert_close(got, expected, atol=0, rtol=0)
            torch.testing.assert_close(sum(partial_values), torch.stack(whole), atol=2e-11, rtol=2e-10)
            torch.testing.assert_close(sum(partial_gradients), whole_y.grad, atol=2e-11, rtol=2e-10)

    def test_default_matches_explicit_false_exactly(self):
        for distance in ('squared_euclidean', 'pseudo_huber'):
            for function, args in (
                    (mmd_rbf.rbf_pair_sums, (self.real, self.fake[:, 0], self.mask.float(), self.alpha)),
                    (mmd_rbf.rbf_mmd2_unequal,
                     (self.real, self.fake[:, 0], self.mask, self.mask, self.alpha)),
                    (mmd_rbf.rbf_mmd2_fixed_real, (self.real[:, 0], self.fake[:, :, 0], self.alpha))):
                kwargs = dict(block_size=2, distance=distance, huber_delta=self.delta)
                default = function(*args, **kwargs)
                explicit = function(*args, omit_real_real=False, **kwargs)
                if not isinstance(default, tuple):
                    default, explicit = (default,), (explicit,)
                for a, b in zip(default, explicit):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
