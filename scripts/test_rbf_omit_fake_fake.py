"""CPU references and kernel-work checks for independent fake-fake omission.

Run: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -B scripts/test_rbf_omit_fake_fake.py
"""

import itertools
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mmd_rbf


DISTANCES = ('squared_euclidean', 'pseudo_huber')


def dense_kernel(x, y, alpha, distance, delta):
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    d = (diff.square().sum(-1) if distance == 'squared_euclidean' else
         (torch.sqrt(diff.square() + delta ** 2) - delta).sum(-1))
    return torch.exp(-alpha * d)


def dense_unequal(real, fake, wx, wy, alpha, distance, delta, omit_rr):
    values = []
    for x, y, mx, my in zip(real.double(), fake.double(), wx, wy):
        x, y = x[mx.bool()], y[my.bool()]
        if len(x) < 2 or len(y) < 2:
            values.append(real.new_zeros((), dtype=torch.float64))
            continue
        value = -2 * dense_kernel(x, y, alpha, distance, delta).mean()
        if not omit_rr:
            xx = dense_kernel(x, x, alpha, distance, delta)
            value += (xx.sum() - xx.diag().sum()) / (len(x) * (len(x) - 1))
        values.append(value)
    return torch.stack(values)


class OmitFakeFakeRBFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        generator = torch.Generator().manual_seed(3901)
        self.real = torch.randn(5, 5, 7, generator=generator)
        self.fake = torch.randn(5, 4, 5, 7, generator=generator)
        self.mask = torch.tensor([[1, 1, 0, 1, 1], [0, 1, 1, 0, 0],
                                  [0, 0, 1, 0, 0], [0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1]], dtype=torch.bool)
        self.alpha, self.delta = .13, .37

    def kwargs(self, distance, omit_rr=False, tile=2):
        return dict(block_size=tile, distance=distance, huber_delta=self.delta,
                    omit_real_real=omit_rr, omit_fake_fake=True)

    def test_unequal_dense_k1_k2_k4_and_unchanged_minimum_counts(self):
        for distance, count, omit_rr, tile in itertools.product(
                DISTANCES, (1, 2, 4), (False, True), (0, 1, 3, 99)):
            with self.subTest(distance=distance, count=count, omit_rr=omit_rr, tile=tile):
                fake = self.fake[:, :count].flatten(1, 2)
                fake_mask = self.mask.repeat(1, count)
                fake_mask[-1] = False
                fake_mask[-1, 0] = True
                expected = dense_unequal(self.real, fake, self.mask, fake_mask,
                                         self.alpha, distance, self.delta, omit_rr)
                actual, active = mmd_rbf.rbf_mmd2_unequal(
                    self.real, fake, self.mask, fake_mask, self.alpha,
                    **self.kwargs(distance, omit_rr, tile))
                torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
                self.assertEqual(active.tolist(), [True, True, False, False, False])
                self.assertFalse(actual.requires_grad)

    def test_pair_sums_dense_values_and_cross_only_fake_gradients(self):
        for distance, omit_rr, length, tile in itertools.product(
                DISTANCES, (False, True), (1, 2, 4, 5), (0, 1, 3)):
            with self.subTest(distance=distance, omit_rr=omit_rr, length=length, tile=tile):
                x = self.real[:, :length].double().requires_grad_()
                initial_y = self.fake[:, 0, :length].double()
                w = self.mask[:, :length].double()
                y_reference = initial_y.clone().requires_grad_()
                weights = w.unsqueeze(2) * w.unsqueeze(1)
                xy = (dense_kernel(x.detach(), y_reference, self.alpha, distance, self.delta)
                      * weights).sum((1, 2))
                xx = (torch.zeros_like(xy) if omit_rr else
                      (dense_kernel(x.detach(), x.detach(), self.alpha, distance, self.delta)
                       * weights).sum((1, 2)))
                upstream = torch.tensor([.7, -.3, .9, 1.1, -.2], dtype=torch.float64)
                (upstream * xy).sum().backward()
                y = initial_y.clone().requires_grad_()
                actual = mmd_rbf.rbf_pair_sums(
                    x, y, w, self.alpha, **self.kwargs(distance, omit_rr, tile))
                # Nonzero upstream for FF ensures it is suppressed by the flag,
                # even if a caller still differentiates the returned zero sum.
                (.3 * actual[0].sum() + .8 * actual[1].sum()
                 + (upstream * actual[2]).sum()).backward()
                for got, expected in zip(actual, (xx, torch.zeros_like(xx), xy)):
                    torch.testing.assert_close(got, expected, atol=2e-11, rtol=2e-10)
                torch.testing.assert_close(y.grad, y_reference.grad, atol=2e-11, rtol=2e-9)
                self.assertIsNone(x.grad)
                self.assertEqual(y.grad[3].count_nonzero().item(), 0)

    def test_fixed_real_dense_removes_full_term_including_k1_fallback(self):
        real, fake = self.real[:, 0], self.fake[:, :, 0]
        for distance, omit_rr, count, tile in itertools.product(
                DISTANCES, (False, True), (1, 2, 4), (0, 1, 3, 99)):
            with self.subTest(distance=distance, omit_rr=omit_rr, count=count, tile=tile):
                xy = dense_kernel(real.double().unsqueeze(1), fake[:, :count].double(),
                                  self.alpha, distance, self.delta).mean((1, 2))
                expected = (0. if omit_rr else 1.) - 2 * xy
                actual = mmd_rbf.rbf_mmd2_fixed_real(
                    real, fake[:, :count], self.alpha, **self.kwargs(distance, omit_rr, tile))
                torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
                self.assertFalse(actual.requires_grad)
        # Identical draws make the formerly retained fake fallback exactly 1.
        for distance, count, omit_rr in itertools.product(DISTANCES, (1, 2, 4), (False, True)):
            actual = mmd_rbf.rbf_mmd2_fixed_real(
                torch.zeros(2, 3), torch.zeros(2, count, 3), self.alpha,
                **self.kwargs(distance, omit_rr))
            torch.testing.assert_close(actual, torch.full((2,), -2. if omit_rr else -1.))

    def test_cross_sentence_dense_empty_masks_and_k_constraint(self):
        for distance, count, tile in itertools.product(DISTANCES, (2, 4), (0, 1, 3, 99)):
            with self.subTest(distance=distance, count=count, tile=tile):
                expected = []
                for x, draws, selected in zip(self.real, self.fake[:, :count], self.mask):
                    if not selected.any():
                        expected.append(0.)
                    else:
                        xy = dense_kernel(x[selected].double(),
                                          draws[:, selected].flatten(0, 1).double(),
                                          self.alpha, distance, self.delta)
                        expected.append(float(-2 * xy.mean()))
                actual, active = mmd_rbf.rbf_mmd2_cross_sentence(
                    self.real, self.fake[:, :count], self.mask, self.alpha,
                    block_size=tile, distance=distance, huber_delta=self.delta,
                    omit_fake_fake=True)
                torch.testing.assert_close(actual, torch.tensor(expected), atol=2e-6, rtol=2e-5)
                self.assertEqual(active.tolist(), [True, True, True, False, True])
                self.assertFalse(actual.requires_grad)
        for distance, count in itertools.product(DISTANCES, (0, 1)):
            with self.assertRaisesRegex(ValueError, 'at least 2'):
                mmd_rbf.rbf_mmd2_cross_sentence(
                    self.real, self.fake[:, :count], self.mask, self.alpha,
                    distance=distance, omit_fake_fake=True)

    def test_forward_and_backward_skip_ff_and_preserve_rr_cross_work(self):
        real, fake = self.real.clone(), self.fake.clone()
        real[..., 0], fake[..., 0] = 1., -1.
        for distance, name in zip(DISTANCES, ('_kernel_block', '_pseudo_huber_kernel_block')):
            original = getattr(mmd_rbf, name)
            calls = []

            def observe(a, b, *args, **kwargs):
                # Markers survive all casts and slicing used by either backend.
                left = 'r' if (a[..., 0] == 1).all() else 'f'
                right = 'r' if (b[..., 0] == 1).all() else 'f'
                calls.append(left + right)
                return original(a, b, *args, **kwargs)

            for omit_rr in (False, True):
                for function, args in (
                        (mmd_rbf.rbf_pair_sums, (real, fake[:, 0], self.mask.float(), self.alpha)),
                        (mmd_rbf.rbf_mmd2_unequal,
                         (real, fake[:, 0], self.mask, self.mask, self.alpha)),
                        (mmd_rbf.rbf_mmd2_fixed_real, (real[:, 0], fake[:, :, 0], self.alpha)),
                        (mmd_rbf.rbf_mmd2_cross_sentence, (real, fake, self.mask, self.alpha))):
                    with self.subTest(distance=distance, omit_rr=omit_rr, function=function.__name__):
                        options = self.kwargs(distance, omit_rr)
                        if function is mmd_rbf.rbf_mmd2_cross_sentence:
                            options.pop('omit_real_real')
                        with mock.patch.object(mmd_rbf, name, side_effect=observe):
                            calls.clear()
                            function(*args, **{**options, 'omit_fake_fake': False})
                            full_calls = list(calls)
                            self.assertIn('ff', full_calls)
                            calls.clear()
                            function(*args, **options)
                            self.assertEqual(calls, [term for term in full_calls if term != 'ff'])
                            self.assertIn('rf', calls)

                with mock.patch.object(mmd_rbf, name, side_effect=observe):
                    for omit_ff in (False, True):
                        y = fake[:, 0].clone().requires_grad_()
                        pair = mmd_rbf.rbf_pair_sums(
                            real, y, self.mask.float(), self.alpha,
                            **{**self.kwargs(distance, omit_rr), 'omit_fake_fake': omit_ff})
                        calls.clear()
                        (pair[0].sum() + pair[1].sum() - 2 * pair[2].sum()).backward()
                        if omit_ff:
                            self.assertEqual(calls, [term for term in full_backward if term != 'ff'])
                            self.assertIn('fr', calls)
                        else:
                            full_backward = list(calls)
                            self.assertIn('ff', full_backward)

    def test_sharding_preserves_remaining_tile_assignments_and_gradients(self):
        real, initial_y, weights = self.real.double(), self.fake[:, 0].double(), self.mask.double()
        for distance, omit_rr in itertools.product(DISTANCES, (False, True)):
            kwargs = self.kwargs(distance, omit_rr)
            y = initial_y.clone().requires_grad_()
            whole = mmd_rbf.rbf_pair_sums(real, y, weights, self.alpha, **kwargs)
            (whole[0].sum() + .4 * whole[1].sum() - .7 * whole[2].sum()).backward()
            values, gradients = [], []
            for rank in range(4):
                with self.subTest(distance=distance, omit_rr=omit_rr, rank=rank):
                    with mock.patch.object(mmd_rbf, '_shard_ctx', return_value=(rank, 4)), \
                            mock.patch.object(mmd_rbf.dist, 'all_reduce'):
                        local_y = initial_y.clone().requires_grad_()
                        partial = mmd_rbf.rbf_pair_sums(
                            real, local_y, weights, self.alpha, shard=True, **kwargs)
                        (partial[0].sum() + .4 * partial[1].sum() - .7 * partial[2].sum()).backward()
                        values.append(torch.stack(partial))
                        gradients.append(local_y.grad)
                        # With FF's upstream zero, the legacy per-rank gradient
                        # isolates the original RF backward tile assignment.
                        full_y = initial_y.clone().requires_grad_()
                        full = mmd_rbf.rbf_pair_sums(
                            real, full_y, weights, self.alpha, shard=True,
                            **{**kwargs, 'omit_fake_fake': False})
                        (full[0].sum() - .7 * full[2].sum()).backward()
                        for index in (0, 2):
                            torch.testing.assert_close(partial[index], full[index], atol=0, rtol=0)
                        torch.testing.assert_close(local_y.grad, full_y.grad, atol=0, rtol=0)
            torch.testing.assert_close(sum(values), torch.stack(whole), atol=2e-11, rtol=2e-10)
            torch.testing.assert_close(sum(gradients), y.grad, atol=2e-11, rtol=2e-10)

    def test_omitted_output_has_zero_derivative_and_unused_fake_backward_arity(self):
        for distance, omit_rr in itertools.product(DISTANCES, (False, True)):
            y = self.fake[:, 0].clone().requires_grad_()
            _, ff, _ = mmd_rbf.rbf_pair_sums(
                self.real, y, self.mask.float(), self.alpha, **self.kwargs(distance, omit_rr))
            ff.sum().backward()
            torch.testing.assert_close(y.grad, torch.zeros_like(y), atol=0, rtol=0)
            x = self.real.clone().requires_grad_()
            sums = mmd_rbf.rbf_pair_sums(
                x, self.fake[:, 0], self.mask.float(), self.alpha, **self.kwargs(distance, omit_rr))
            sum(s.sum() for s in sums).backward()
            self.assertIsNone(x.grad)

    def test_default_matches_explicit_false_exactly(self):
        for distance, omit_rr in itertools.product(DISTANCES, (False, True)):
            for function, args in (
                    (mmd_rbf.rbf_pair_sums,
                     (self.real, self.fake[:, 0], self.mask.float(), self.alpha)),
                    (mmd_rbf.rbf_mmd2_unequal,
                     (self.real, self.fake[:, 0], self.mask, self.mask, self.alpha)),
                    (mmd_rbf.rbf_mmd2_fixed_real, (self.real[:, 0], self.fake[:, :, 0], self.alpha)),
                    (mmd_rbf.rbf_mmd2_cross_sentence, (self.real, self.fake, self.mask, self.alpha))):
                kwargs = self.kwargs(distance, omit_rr)
                kwargs.pop('omit_fake_fake')
                if function is mmd_rbf.rbf_mmd2_cross_sentence:
                    kwargs.pop('omit_real_real')
                default = function(*args, **kwargs)
                explicit = function(*args, **kwargs, omit_fake_fake=False)
                if not isinstance(default, tuple):
                    default, explicit = (default,), (explicit,)
                for a, b in zip(default, explicit):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
