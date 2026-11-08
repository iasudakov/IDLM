"""CPU references for exp(-alpha * sum_d pseudo_huber_delta(x_d-y_d)).

Run with: OMP_NUM_THREADS=2 python -B scripts/test_rbf_pseudo_huber.py
No checkpoints, downloads, CUDA, or tracker are needed. Dense references use
the defining coordinatewise expression, independently of the tiled kernel.
"""

import ast
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mmd_rbf
import mmd_reward


def dense_distance(a, b, delta, distance='pseudo_huber'):
    diff = a.unsqueeze(-2) - b.unsqueeze(-3)
    if distance == 'squared_euclidean':
        return diff.square().sum(-1)
    return (torch.sqrt(diff.square() + delta ** 2) - delta).sum(-1)


def dense_kernel(a, b, alpha, delta, distance='pseudo_huber'):
    return torch.exp(-alpha * dense_distance(a, b, delta, distance))


def dense_sums(x, y, weights, alpha, delta, distance='pseudo_huber'):
    pair_weights = weights.unsqueeze(-1) * weights.unsqueeze(-2)
    return tuple((dense_kernel(a, b, alpha, delta, distance)
                  * pair_weights).sum((-2, -1))
                 for a, b in ((x, x), (y, y), (x, y)))


def dense_unequal(x, y, alpha, delta):
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xx = dense_kernel(x, x, alpha, delta)
    yy = dense_kernel(y, y, alpha, delta)
    xy = dense_kernel(x, y, alpha, delta)
    return float((xx.sum() - xx.diag().sum()) / (len(x) * (len(x) - 1))
                 + (yy.sum() - yy.diag().sum()) / (len(y) * (len(y) - 1))
                 - 2 * xy.mean())


def dense_fixed(x, y, alpha, delta):
    yy = dense_kernel(y, y, alpha, delta)
    fake_term = (yy[0, 0] if len(y) == 1 else
                 (yy.sum() - yy.diag().sum()) / (len(y) * (len(y) - 1)))
    return float(1 + fake_term - 2 * dense_kernel(x[None], y, alpha, delta).mean())


class PseudoHuberRBFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        generator = torch.Generator().manual_seed(417)
        self.real = torch.randn(4, 5, 7, generator=generator)
        self.fake = torch.randn(3, 4, 5, 7, generator=generator)
        self.selected = torch.tensor([[1, 1, 0, 1, 1], [0, 1, 1, 0, 0],
                                      [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                                     dtype=torch.bool)
        self.alpha, self.delta = .13, .37
        self.options = dict(distance='pseudo_huber', huber_delta=self.delta)

    def test_pair_sums_and_fake_only_backward_match_dense_for_ragged_tiles(self):
        generator = torch.Generator().manual_seed(816)
        # More than one hidden-dimension chunk as well as ragged token tiles.
        x0 = torch.randn(2, 5, 67, dtype=torch.float64, generator=generator) / 3
        y0 = torch.randn(2, 5, 67, dtype=torch.float64, generator=generator) / 3
        weights = torch.tensor([[1, 0, 1, 1, 1], [0, 1, 0, 1, 0]], dtype=x0.dtype)
        coefficients = torch.tensor([[.3, -.7], [1.1, .4], [-.2, .9]], dtype=x0.dtype)
        reference_y = y0.clone().requires_grad_()
        expected = dense_sums(x0, reference_y, weights, self.alpha, self.delta)
        sum((term * coefficients[i]).sum() for i, term in enumerate(expected)).backward()
        for tile in (0, 1, 2, 3, 20):
            with self.subTest(tile=tile):
                x, y = x0.clone().requires_grad_(), y0.clone().requires_grad_()
                actual = mmd_rbf.rbf_pair_sums(
                    x, y, weights, self.alpha, block_size=tile, **self.options)
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, atol=2e-11, rtol=2e-10)
                sum((term * coefficients[i]).sum() for i, term in enumerate(actual)).backward()
                self.assertIsNone(x.grad, 'the frozen real branch must not receive a gradient')
                torch.testing.assert_close(y.grad, reference_y.grad, atol=2e-11, rtol=2e-9)
                self.assertEqual(float(y.grad[weights == 0].abs().max()), 0.0)

    def test_coordinatewise_formula_has_no_delta_multiplier_or_dimension_mean(self):
        x = torch.zeros(1, 2)
        y = torch.tensor([[[3., 4.]]])
        delta, alpha = .7, .2
        distance = math.sqrt(9 + delta ** 2) + math.sqrt(16 + delta ** 2) - 2 * delta
        expected = 2 - 2 * math.exp(-alpha * distance)
        value = mmd_rbf.rbf_mmd2_fixed_real(
            x, y, alpha, distance='pseudo_huber', huber_delta=delta)
        self.assertAlmostEqual(float(value), expected, places=6)
        for incorrect in (distance * delta, distance / 2,
                          math.sqrt(25 + delta ** 2) - delta):
            self.assertGreater(abs(float(value) - (2 - 2 * math.exp(-alpha * incorrect))), .01)

    def test_unequal_masks_inactive_rows_and_tiles_match_dense(self):
        fake = self.fake.permute(1, 0, 2, 3).reshape(4, 15, 7)
        fake_mask = self.selected.repeat(1, 3)
        expected = torch.tensor([
            dense_unequal(x[mask].double(), y[ymask].double(), self.alpha, self.delta)
            for x, y, mask, ymask in zip(self.real, fake, self.selected, fake_mask)])
        for tile in (0, 1, 2, 4, 20):
            value, active = mmd_rbf.rbf_mmd2_unequal(
                self.real, fake, self.selected, fake_mask, self.alpha,
                block_size=tile, **self.options)
            self.assertEqual(active.tolist(), [True, True, False, False])
            self.assertFalse(value.requires_grad)
            torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-5)

    def test_fixed_real_k1_and_larger_draw_sets_match_dense_without_clamping(self):
        real = self.real[:, 0]
        fake = self.fake[:, :, 0].transpose(0, 1)
        for count in (1, 2, 3):
            expected = torch.tensor([dense_fixed(x.double(), ys[:count].double(),
                                                self.alpha, self.delta)
                                     for x, ys in zip(real, fake)])
            for tile in (0, 1, 2, 10):
                value = mmd_rbf.rbf_mmd2_fixed_real(
                    real, fake[:, :count], self.alpha, block_size=tile, **self.options)
                torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-5)
        value = mmd_rbf.rbf_mmd2_fixed_real(
            torch.zeros(1, 1), torch.tensor([[[-.1], [.1]]]), 1.,
            distance='pseudo_huber', huber_delta=1.)
        self.assertLess(float(value), 0., 'negative finite-sample estimates must survive')

    def test_zero_alpha_and_identical_points_have_finite_zero_gradients(self):
        for alpha in (0., .5):
            x = torch.zeros(2, 5, 3, dtype=torch.float64)
            y = x.clone().requires_grad_()
            weights = torch.ones(2, 5, dtype=x.dtype)
            values = mmd_rbf.rbf_pair_sums(x, y, weights, alpha, block_size=2, **self.options)
            sum(term.sum() for term in values).backward()
            for value in values:
                torch.testing.assert_close(value, torch.full_like(value, 25.))
            torch.testing.assert_close(y.grad, torch.zeros_like(y))

    def test_half_features_use_finite_float32_pair_sums(self):
        for dtype in (torch.float16, torch.bfloat16):
            x = self.real[:2].to(dtype)
            y = self.fake[0, :2].to(dtype).requires_grad_()
            weights = self.selected[:2].to(dtype)
            actual = mmd_rbf.rbf_pair_sums(
                x, y, weights, self.alpha, block_size=2, **self.options)
            expected = dense_sums(x.double(), y.detach().double(), weights.double(),
                                  self.alpha, self.delta)
            for value, reference in zip(actual, expected):
                self.assertEqual(value.dtype, torch.float32)
                torch.testing.assert_close(value.double(), reference, atol=2e-6, rtol=2e-5)
            (actual[1].sum() - 2 * actual[2].sum()).backward()
            self.assertTrue(bool(torch.isfinite(y.grad).all()))

    def test_explicit_large_block_respects_scratch_budget_in_forward_and_backward(self):
        generator = torch.Generator().manual_seed(508)
        x = torch.randn(2, 5, 67, generator=generator, dtype=torch.float64) / 3
        initial_y = torch.randn(2, 5, 67, generator=generator, dtype=torch.float64) / 3
        weights = torch.ones(2, 5, dtype=torch.float64)
        reference_y = initial_y.clone().requires_grad_()
        expected = dense_sums(x, reference_y, weights, self.alpha, self.delta)
        (expected[1].sum() - 2 * expected[2].sum()).backward()
        y = initial_y.clone().requires_grad_()
        scratch_bytes, shapes = 8192, []
        original_hypot = torch.hypot

        def observe_hypot(values, delta, *args, **kwargs):
            if values.ndim == 4:
                shapes.append(tuple(values.shape))
                # The layout budgets four live coordinatewise scratch buffers.
                self.assertLessEqual(4 * values.numel() * values.element_size(), scratch_bytes)
            return original_hypot(values, delta, *args, **kwargs)

        with mock.patch.object(mmd_rbf, '_PSEUDO_HUBER_SCRATCH_BYTES', scratch_bytes):
            with mock.patch.object(torch, 'hypot', side_effect=observe_hypot):
                actual = mmd_rbf.rbf_pair_sums(
                    x, y, weights, self.alpha, block_size=99999, **self.options)
                forward_calls = len(shapes)
                (actual[1].sum() - 2 * actual[2].sum()).backward()
        self.assertGreater(forward_calls, 0)
        self.assertGreater(len(shapes), forward_calls, 'backward must recompute bounded tiles')
        self.assertLess(max(max(shape[1:3]) for shape in shapes), x.shape[1])
        self.assertLess(max(shape[3] for shape in shapes), x.shape[2])
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, atol=2e-11, rtol=2e-10)
        torch.testing.assert_close(y.grad, reference_y.grad, atol=2e-11, rtol=2e-9)

    def test_bandwidth_uses_only_selected_real_robust_distances(self):
        real = torch.tensor([[[0., 0.], [2., 1.], [7., 3.], [1e8, -1e8]]])
        selected = torch.tensor([[True, True, True, False]])
        points = real[0, :3].double()
        matrix = dense_distance(points, points, self.delta)
        median = matrix[~torch.eye(3, dtype=torch.bool)].median()
        expected = float(1 / (2 * median))
        actual = mmd_reward.median_alpha(real, selected, **self.options)
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertGreater(abs(actual - mmd_reward.median_alpha(real, selected)), .01)
        for mask in (torch.zeros_like(selected), torch.tensor([[True, False, False, False]])):
            self.assertEqual(mmd_reward.median_alpha(real, mask, **self.options), 1.)
        self.assertEqual(mmd_reward.median_alpha(torch.zeros_like(real), selected, **self.options), 1.)

    def test_default_squared_euclidean_path_remains_identical(self):
        x, y, weights = self.real.double(), self.fake[0].double(), self.selected.double()
        actual = mmd_rbf.rbf_pair_sums(x, y, weights, self.alpha, block_size=2)
        explicit = mmd_rbf.rbf_pair_sums(x, y, weights, self.alpha, block_size=2,
                                       distance='squared_euclidean')
        expected = dense_sums(x, y, weights, self.alpha, self.delta, 'squared_euclidean')
        for value, unchanged, reference in zip(actual, explicit, expected):
            torch.testing.assert_close(value, unchanged, atol=0, rtol=0)
            torch.testing.assert_close(value, reference, atol=2e-11, rtol=2e-10)
        for function, args in (
            (mmd_rbf.rbf_mmd2_unequal, (x, y, weights, weights, self.alpha)),
            (mmd_rbf.rbf_mmd2_fixed_real, (x[:, 0], y, self.alpha))):
            default = function(*args, block_size=2)
            explicit = function(*args, block_size=2, distance='squared_euclidean')
            if isinstance(default, tuple):
                default, explicit = default[0], explicit[0]
            torch.testing.assert_close(default, explicit, atol=0, rtol=0)

    def test_paired_and_draw_wrappers_forward_distance_and_delta(self):
        expected = torch.tensor([
            -dense_unequal(x[mask].double(), y[mask].double(), self.alpha, self.delta)
            for x, y, mask in zip(self.real, self.fake[0], self.selected)])
        value, active = mmd_reward.paired_sequence_reward(
            self.real, self.fake[0], self.selected, kernel='rbf', alpha=self.alpha,
            block_size=2, **self.options)
        torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(active.tolist(), [True, True, False, False])
        features = {0: torch.cat([self.real, self.fake.flatten(0, 1)])}
        for level in ('token', 'sentence'):
            expected = []
            for row, mask in enumerate(self.selected):
                x = self.real[row, mask].double()
                y = self.fake[:, row, mask].double()
                if level == 'token':
                    estimate = dense_unequal(x, y.flatten(0, 1), self.alpha, self.delta)
                else:
                    estimate = (dense_fixed(x.mean(0), y.mean(1), self.alpha, self.delta)
                                if bool(mask.any()) else 0.)
                expected.append(-estimate)
            value, _, per_layer = mmd_reward.summed_multi_draw_reward(
                features, 4, 3, 5, self.selected, [0], [1.], kernel='rbf',
                level=level, alpha=self.alpha, block_size=2, **self.options)
            torch.testing.assert_close(value, torch.tensor(expected), atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(value, per_layer[0])

    def test_invalid_mode_and_nonpositive_or_nonfinite_delta_are_rejected(self):
        functions = (
            lambda **kw: mmd_rbf.rbf_pair_sums(self.real, self.fake[0], self.selected.float(), self.alpha, **kw),
            lambda **kw: mmd_rbf.rbf_mmd2_unequal(self.real, self.fake[0], self.selected, self.selected, self.alpha, **kw),
            lambda **kw: mmd_rbf.rbf_mmd2_fixed_real(self.real[:, 0], self.fake[0], self.alpha, **kw),
            lambda **kw: mmd_reward.median_alpha(self.real, self.selected, **kw))
        for function in functions:
            with self.assertRaises(ValueError):
                function(distance='not_a_distance')
            for delta in (0., -.1, float('inf'), float('-inf'), float('nan')):
                with self.subTest(delta=delta), self.assertRaises(ValueError):
                    function(distance='pseudo_huber', huber_delta=delta)

    def test_hydra_overrides_reach_lazy_critic_without_loading_a_model(self):
        from hydra import compose, initialize_config_dir

        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
            config = compose(config_name='config', overrides=[
                'algo=mdlm', 'adversarial_distill.mmd_kernel=rbf',
                'adversarial_distill.mmd_feature_source=diffusion',
                'adversarial_distill.mmd_rbf_distance=pseudo_huber',
                'adversarial_distill.mmd_rbf_huber_delta=0.37'])
        self.assertEqual(config.adversarial_distill.mmd_rbf_distance, 'pseudo_huber')
        self.assertEqual(config.adversarial_distill.mmd_rbf_huber_delta, .37)
        # Exercise the production property without importing Lightning, Flash
        # Attention, or creating teacher/student checkpoint copies.
        path = root / 'algo.py'
        tree = ast.parse(path.read_text(), filename=str(path))
        nodes = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == '_mmd_critic']
        self.assertEqual(len(nodes), 1)
        nodes[0].decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        teacher = object()
        host = SimpleNamespace(config=config, teacher=teacher, mask_index=12)
        fake_critic = SimpleNamespace(layer_description='CPU fixture')
        with mock.patch.object(mmd_reward, 'FrozenBackboneFeatures', return_value=fake_critic) as constructor:
            with redirect_stdout(io.StringIO()):
                critic = namespace['_mmd_critic'](host)
        self.assertIs(constructor.call_args.args[0], teacher)
        self.assertIs(critic, fake_critic)
        self.assertEqual(critic.rbf_distance, 'pseudo_huber')
        self.assertEqual(critic.rbf_huber_delta, .37)
        self.assertEqual(critic.kernel, 'rbf')


if __name__ == '__main__':
    unittest.main(verbosity=2)
