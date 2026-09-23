"""CPU oracles for excluding equal-position fake pairs across sentence draws."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmd_rbf import rbf_mmd2_cross_sentence


def oracle(real, fake, selected, alpha, *, distance='squared_euclidean',
           huber_delta=0.37, omit_fake_fake=False, exclude_same_position=True):
    """Explicit ordered draw/position pairs, with all arithmetic in FP64."""
    real, fake = real.detach().double(), fake.detach().double()
    alpha = torch.as_tensor(alpha).detach().double().reshape(-1)
    values, active = [], []
    draws = fake.shape[1]
    for b in range(real.shape[0]):
        positions = selected[b].nonzero(as_tuple=False).flatten().tolist()
        n = len(positions)
        usable = n >= (2 if exclude_same_position else 1)
        active.append(usable)
        if not usable:
            values.append(0.0)
            continue
        coefficient = alpha[0 if alpha.numel() == 1 else b].item()

        def kernel(x, y):
            diff = x - y
            if distance == 'squared_euclidean':
                d = diff.square().sum().item()
            else:
                # sqrt(diff**2 + delta**2) - delta, evaluated stably.
                d = (diff.square() / (
                    torch.sqrt(diff.square() + huber_delta ** 2)
                    + huber_delta)).sum().item()
            return math.exp(-coefficient * d)

        fake_sum = 0.0
        if not omit_fake_fake:
            for r in range(draws):
                for s in range(draws):
                    if r == s:
                        continue
                    for i in positions:
                        for j in positions:
                            if exclude_same_position and i == j:
                                continue
                            fake_sum += kernel(fake[b, r, i], fake[b, s, j])
        cross_sum = sum(kernel(real[b, i], fake[b, r, j])
                        for r in range(draws)
                        for i in positions for j in positions)
        fake_pairs = n * (n - 1) if exclude_same_position else n * n
        values.append(fake_sum / (draws * (draws - 1) * fake_pairs)
                      - 2.0 * cross_sum / (draws * n * n))
    return torch.tensor(values, dtype=torch.float64), torch.tensor(active)


def inputs():
    """Strided tensors with zero, one, two, three, and five scored positions."""
    generator = torch.Generator().manual_seed(327)
    real = torch.randn(5, 10, 14, generator=generator)[:, ::2, ::2]
    fake = torch.randn(5, 3, 10, 14, generator=generator)[:, :, ::2, ::2]
    selected = torch.zeros(5, 10, dtype=torch.bool)[:, ::2]
    selected.copy_(torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
    ], dtype=torch.bool))
    assert not real.is_contiguous()
    assert not fake.is_contiguous()
    assert not selected.is_contiguous()
    return real, fake, selected


class ExcludeSamePositionTests(unittest.TestCase):
    def assert_oracle(self, real, fake, selected, alpha, **kwargs):
        expected, expected_active = oracle(real, fake, selected, alpha,
                                          **kwargs)
        for block_size in (0, 1, 2, 3, 100):
            with self.subTest(block_size=block_size):
                actual, active = rbf_mmd2_cross_sentence(
                    real, fake, selected, alpha, block_size=block_size,
                    **kwargs)
                torch.testing.assert_close(actual.double(), expected,
                                           atol=3e-7, rtol=3e-6)
                self.assertTrue(torch.equal(active, expected_active))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertFalse(actual.requires_grad)

    def test_explicit_fp64_oracle_for_both_distances_draw_counts_and_masks(self):
        real, fake, selected = inputs()
        for draws in (2, 3):
            for distance in ('squared_euclidean', 'pseudo_huber'):
                for omit_fake_fake in (False, True):
                    with self.subTest(draws=draws, distance=distance,
                                      omit_fake_fake=omit_fake_fake):
                        self.assert_oracle(
                            real, fake[:, :draws], selected, 0.19,
                            distance=distance, huber_delta=0.37,
                            omit_fake_fake=omit_fake_fake,
                            exclude_same_position=True)

    def test_broadcast_alpha_and_outer_autocast_are_detached_fp32(self):
        real, fake, selected = inputs()
        real.requires_grad_()
        fake.requires_grad_()
        for alpha in (
                torch.tensor(0.19, requires_grad=True),
                torch.tensor([0.11, 0.14, 0.19, 0.23, 0.31],
                             requires_grad=True).reshape(5, 1, 1)):
            for distance in ('squared_euclidean', 'pseudo_huber'):
                with self.subTest(alpha_shape=tuple(alpha.shape), distance=distance):
                    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                        self.assert_oracle(
                            real, fake, selected, alpha, distance=distance,
                            huber_delta=0.37, exclude_same_position=True)

    def test_default_is_all_positions_and_explicit_false_is_exact(self):
        real, fake, selected = inputs()
        for distance in ('squared_euclidean', 'pseudo_huber'):
            for omit_fake_fake in (False, True):
                for block_size in (1, 3, 100):
                    with self.subTest(distance=distance,
                                      omit_fake_fake=omit_fake_fake,
                                      block_size=block_size):
                        options = dict(distance=distance, huber_delta=0.37,
                                       omit_fake_fake=omit_fake_fake,
                                       block_size=block_size)
                        default, active = rbf_mmd2_cross_sentence(
                            real, fake, selected, 0.19, **options)
                        explicit, explicit_active = rbf_mmd2_cross_sentence(
                            real, fake, selected, 0.19,
                            exclude_same_position=False, **options)
                        self.assertTrue(torch.equal(default, explicit))
                        self.assertTrue(torch.equal(active, explicit_active))
                        expected, expected_active = oracle(
                            real, fake, selected, 0.19, distance=distance,
                            huber_delta=0.37, omit_fake_fake=omit_fake_fake,
                            exclude_same_position=False)
                        torch.testing.assert_close(default.double(), expected,
                                                   atol=3e-7, rtol=3e-6)
                        self.assertTrue(torch.equal(active, expected_active))

    def test_rf_keeps_all_positions_and_is_exactly_unchanged_on_active_rows(self):
        real, fake, selected = inputs()
        for draws in (2, 3):
            for distance in ('squared_euclidean', 'pseudo_huber'):
                with self.subTest(draws=draws, distance=distance):
                    options = dict(distance=distance, huber_delta=0.37,
                                   block_size=2, omit_fake_fake=True)
                    old, old_active = rbf_mmd2_cross_sentence(
                        real, fake[:, :draws], selected, 0.19, **options)
                    new, active = rbf_mmd2_cross_sentence(
                        real, fake[:, :draws], selected, 0.19,
                        exclude_same_position=True, **options)
                    self.assertTrue(torch.equal(old[active], new[active]))
                    self.assertEqual(new[0].item(), 0.0)
                    self.assertEqual(new[1].item(), 0.0)
                    self.assertFalse(active[1].item())
                    self.assertTrue(old_active[1].item())
                    self.assertLess(old[1].item(), 0.0)

    def test_position_exclusion_is_not_alignment_or_unlabelled_token_pooling(self):
        # RF vanishes. Matching equal-position FF pairs must be removed;
        # swapping one draw makes its off-diagonal FF pairs match instead.
        real = torch.full((1, 2, 1), 1000.0)
        fake = torch.tensor([[[[0.0], [100.0]], [[0.0], [100.0]]]])
        selected = torch.ones(1, 2, dtype=torch.bool)
        swapped = fake.clone()
        swapped[:, 1] = swapped[:, 1].flip(1)
        for distance in ('squared_euclidean', 'pseudo_huber'):
            with self.subTest(distance=distance):
                options = dict(distance=distance, huber_delta=0.37, block_size=1)
                before, _ = rbf_mmd2_cross_sentence(
                    real, fake, selected, 1.0, exclude_same_position=True,
                    **options)
                after, _ = rbf_mmd2_cross_sentence(
                    real, swapped, selected, 1.0, exclude_same_position=True,
                    **options)
                pooled, _ = rbf_mmd2_cross_sentence(
                    real, fake, selected, 1.0, **options)
                pooled_swapped, _ = rbf_mmd2_cross_sentence(
                    real, swapped, selected, 1.0, **options)
                self.assertAlmostEqual(before.item(), 0.0, places=7)
                self.assertAlmostEqual(after.item(), 1.0, places=7)
                self.assertAlmostEqual(pooled.item(), 0.5, places=7)
                self.assertTrue(torch.equal(pooled, pooled_swapped))

    def test_at_least_two_draws_are_still_required(self):
        real, fake, selected = inputs()
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            rbf_mmd2_cross_sentence(real, fake[:, :1], selected, 0.19,
                                   exclude_same_position=True)


if __name__ == '__main__':
    unittest.main()
