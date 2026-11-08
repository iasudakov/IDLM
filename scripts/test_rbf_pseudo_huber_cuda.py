"""Optional fused pseudo-Huber CUDA regression tests and bounded benchmark.

Run: OMP_NUM_THREADS=2 python -B scripts/test_rbf_pseudo_huber_cuda.py
Benchmark: CUDA_VISIBLE_DEVICES=<free GPU> python -B \
    scripts/test_rbf_pseudo_huber_cuda.py --benchmark

GPU cases skip when CUDA/Triton is unavailable. CPU fallback is tested in a
fresh process which rejects every Triton import. No models or downloads needed.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mmd_laplace_cuda as fused
import mmd_rbf


def reference_distance(a, b, delta):
    """Dense FP64 definition, rationalized to retain near-zero distances."""
    difference = a.double().unsqueeze(-2) - b.double().unsqueeze(-3)
    return (difference.square()
            / (torch.sqrt(difference.square() + delta ** 2) + delta)).sum(-1)


def reference_kernel(a, b, alpha, delta):
    return torch.exp(-alpha * reference_distance(a, b, delta))


def reference_weighted_gradient(a, b, weights, delta):
    # Differentiate the dense defining distance, rather than duplicate the
    # fused gradient's formula or its block traversal.
    a = a.detach().double().requires_grad_()
    loss = (reference_distance(a, b.detach(), delta) * weights.double()).sum()
    return torch.autograd.grad(loss, a)[0]


class OptionalCudaImportTests(unittest.TestCase):
    def test_cpu_fallback_never_imports_triton(self):
        program = r'''
import builtins
import torch
original_import = builtins.__import__
def no_triton(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        raise AssertionError("CPU fallback attempted to import optional Triton")
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_triton
import mmd_laplace_cuda
import mmd_rbf
x = torch.tensor([[[0., 1.], [2., 3.]]])
y = torch.tensor([[[.25, -.5], [1.5, 2.5]]], requires_grad=True)
assert not mmd_laplace_cuda.supports_pseudo_huber_cuda(x)
result = mmd_rbf.pairwise_distance(x, y, distance="pseudo_huber", huber_delta=.3)
diff = x.double()[:, :, None] - y.detach().double()[:, None]
reference = (torch.sqrt(diff.square() + .3 ** 2) - .3).sum(-1)
torch.testing.assert_close(result.double(), reference, atol=1e-6, rtol=1e-6)
sums = mmd_rbf.rbf_pair_sums(x, y, torch.ones(1, 2), .2,
                           distance="pseudo_huber", huber_delta=.3)
(sums[1] - 2 * sums[2]).sum().backward()
assert torch.isfinite(y.grad).all()
'''
        completed = subprocess.run([sys.executable, '-B', '-c', program],
                                   cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class FusedPseudoHuberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA is unavailable')
        if not fused.supports_pseudo_huber_cuda(torch.empty(0, device='cuda')):
            raise unittest.SkipTest('optional Triton kernels are unavailable')
        torch.set_num_threads(2)

    def setUp(self):
        self.generator = torch.Generator().manual_seed(918)

    def random(self, *shape):
        return torch.randn(*shape, generator=self.generator).cuda()

    def assert_close(self, actual, expected, atol=3e-5, rtol=3e-5):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual.double(), expected.double(), atol=atol, rtol=rtol)

    def test_distances_ragged_feature_tiles_and_strided_inputs(self):
        # H=131 and 1057 cover the tail of a hidden tile and the >1024 loop.
        for hidden in (7, 131, 1057):
            a = self.random(4, 18, hidden * 2)[::2, 1::2, ::2]
            b = self.random(4, 26, hidden * 2)[::2, 1::2, ::2]
            for delta in (.001, .37, 10.):
                with self.subTest(hidden=hidden, delta=delta):
                    actual = fused.pseudo_huber_pairwise_distance_cuda(a, b, delta)
                    self.assert_close(actual, reference_distance(a, b, delta))
                    self.assertEqual(actual.shape, (2, 9, 13))
                    self.assertFalse(actual.requires_grad)

    def test_weighted_gradients_contiguous_transposed_and_strided_weights(self):
        a = self.random(4, 10, 134)[::2, 1::2, ::2]
        b = self.random(4, 138, 134)[::2, 1::2, ::2]
        weights = (
            self.random(2, 5, 69),
            self.random(2, 69, 5).transpose(1, 2),
            self.random(4, 10, 138)[::2, ::2, ::2],
        )
        for index, k in enumerate(weights):
            with self.subTest(layout=index):
                actual = fused.pseudo_huber_weighted_grad_sum(a, b, k, .17)
                self.assert_close(actual, reference_weighted_gradient(a, b, k, .17),
                                  atol=8e-5, rtol=5e-5)
                self.assertTrue(actual.is_contiguous())
                self.assertFalse(actual.requires_grad)

    def test_zero_small_and_large_residuals_and_deltas(self):
        cases = (
            (0., .001), (1e-8, .001), (1e-3, 1e-3),
            (1e20, .001), (1e-4, 1e20), (1., 1e-30),
        )
        for scale, delta in cases:
            a = torch.tensor([[[0., scale, -scale], [scale, 0., scale]]], device='cuda')
            b = torch.tensor([[[0., 0., 0.], [-scale, scale, 0.]]], device='cuda')
            k = torch.tensor([[[1., -.25], [.5, 2.]]], device='cuda')
            with self.subTest(scale=scale, delta=delta):
                actual = fused.pseudo_huber_pairwise_distance_cuda(a, b, delta)
                self.assert_close(actual, reference_distance(a, b, delta),
                                  atol=1e-35, rtol=3e-5)
                gradient = fused.pseudo_huber_weighted_grad_sum(a, b, k, delta)
                self.assert_close(gradient, reference_weighted_gradient(a, b, k, delta),
                                  atol=1e-30, rtol=4e-5)

    def test_empty_dimensions(self):
        for rows, count_i, count_j, hidden in ((0, 3, 4, 7), (2, 0, 4, 7),
                                               (2, 3, 0, 7), (2, 3, 4, 0)):
            a = torch.zeros(rows, count_i, hidden, device='cuda')
            b = torch.zeros(rows, count_j, hidden, device='cuda')
            k = torch.ones(rows, count_i, count_j, device='cuda')
            with self.subTest(shape=(rows, count_i, count_j, hidden)):
                distance = fused.pseudo_huber_pairwise_distance_cuda(a, b, .001)
                gradient = fused.pseudo_huber_weighted_grad_sum(a, b, k, .001)
                self.assertEqual(distance.shape, (rows, count_i, count_j))
                self.assertEqual(gradient.shape, a.shape)
                self.assertEqual(int(torch.count_nonzero(distance)), 0)
                self.assertEqual(int(torch.count_nonzero(gradient)), 0)

    def test_existing_laplace_entrypoints_keep_l1_semantics(self):
        a, b = self.random(2, 5, 11), self.random(2, 7, 11)
        k = self.random(2, 7, 5).transpose(1, 2)
        difference = a.double()[:, :, None] - b.double()[:, None]
        self.assert_close(fused.l1_pairwise_distance_cuda(a, b), difference.abs().sum(-1))
        self.assert_close(fused.l1_weighted_sign_sum(a, b, k),
                          (difference.sign() * k.double().unsqueeze(-1)).sum(2))

    def test_integrated_distance_and_fake_only_backward_against_dense(self):
        x, initial_y = self.random(2, 7, 67) / 3, self.random(2, 7, 67) / 3
        weights = torch.tensor([[1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 1]],
                               dtype=torch.float32, device='cuda')
        coefficients = self.random(3, 2)
        alpha, delta = .07, .31
        reference_y = initial_y.double().requires_grad_()
        pair_weights = weights.double()[:, :, None] * weights.double()[:, None]
        reference = tuple((reference_kernel(a, b, alpha, delta) * pair_weights).sum((1, 2))
                          for a, b in ((x, x), (reference_y, reference_y), (x, reference_y)))
        expected_gradient = torch.autograd.grad(
            sum((term * coefficients[index]).sum() for index, term in enumerate(reference)),
            reference_y)[0]
        for block in (0, 3):
            with self.subTest(block=block):
                real, fake = x.clone().requires_grad_(), initial_y.clone().requires_grad_()
                with mock.patch.object(fused, 'pseudo_huber_pairwise_distance_cuda',
                                       wraps=fused.pseudo_huber_pairwise_distance_cuda) as distance:
                    with mock.patch.object(fused, 'pseudo_huber_weighted_grad_sum',
                                           wraps=fused.pseudo_huber_weighted_grad_sum) as gradient:
                        actual = mmd_rbf.rbf_pair_sums(
                            real, fake, weights, alpha, block_size=block,
                            distance='pseudo_huber', huber_delta=delta)
                        sum((term * coefficients[index]).sum()
                            for index, term in enumerate(actual)).backward()
                self.assertGreater(distance.call_count, 0, 'integrated forward must use fused kernel')
                self.assertGreater(gradient.call_count, 0, 'integrated backward must use fused kernel')
                for value, expected in zip(actual, reference):
                    self.assert_close(value, expected)
                self.assert_close(fake.grad, expected_gradient, atol=4e-6, rtol=4e-5)
                self.assertIsNone(real.grad)
                self.assertEqual(int(torch.count_nonzero(fake.grad[weights == 0])), 0)

    def test_unequal_fixed_real_and_cross_sentence_estimators(self):
        x, draws = self.random(3, 5, 9), self.random(3, 3, 5, 9)
        selected = torch.tensor([[1, 1, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                                dtype=torch.bool, device='cuda')
        alpha, delta = .09, .13
        options = dict(distance='pseudo_huber', huber_delta=delta, block_size=3)
        actual, active = mmd_rbf.rbf_mmd2_unequal(
            x, draws.flatten(1, 2), selected, selected.repeat(1, 3), alpha, **options)
        expected_unequal, expected_cross = [], []
        for real, fake, mask in zip(x, draws, selected):
            real, fake = real[mask].double(), fake[:, mask].double()
            n, m = len(real), fake.numel() // fake.shape[-1]
            flat = fake.flatten(0, 1)
            if n >= 2:
                xx, yy = reference_kernel(real, real, alpha, delta), reference_kernel(flat, flat, alpha, delta)
                xy = reference_kernel(real, flat, alpha, delta)
                expected_unequal.append((xx.sum() - xx.diag().sum()) / (n * (n - 1))
                                        + (yy.sum() - yy.diag().sum()) / (m * (m - 1)) - 2 * xy.mean())
            else:
                expected_unequal.append(x.new_tensor(0.).double())
            if n:
                within = sum(reference_kernel(fake[i], fake[j], alpha, delta).mean()
                             for i in range(3) for j in range(3) if i != j) / 6
                cross = sum(reference_kernel(real, draw, alpha, delta).mean() for draw in fake) / 3
                expected_cross.append(within - 2 * cross)
            else:
                expected_cross.append(x.new_tensor(0.).double())
        self.assert_close(actual, torch.stack(expected_unequal), atol=2e-6)
        self.assertEqual(active.tolist(), [True, False, False])
        actual, active = mmd_rbf.rbf_mmd2_cross_sentence(x, draws, selected, alpha, **options)
        self.assert_close(actual, torch.stack(expected_cross), atol=2e-6)
        self.assertEqual(active.tolist(), [True, True, False])
        for count in (1, 3):
            real, fake = x[:, 0], draws[:, :count, 0]
            expected = []
            for a, b in zip(real, fake):
                yy = reference_kernel(b, b, alpha, delta)
                yy_term = 1. if count == 1 else (yy.sum() - yy.diag().sum()) / (count * (count - 1))
                expected.append(1 + yy_term - 2 * reference_kernel(a[None], b, alpha, delta).mean())
            self.assert_close(mmd_rbf.rbf_mmd2_fixed_real(real, fake, alpha, **options),
                              torch.stack(expected), atol=2e-6)

    def test_missing_optional_backend_keeps_cuda_fallback(self):
        x, y = self.random(2, 5, 11), self.random(2, 9, 11)
        with mock.patch.object(fused, 'supports_pseudo_huber_cuda', return_value=False):
            with mock.patch.object(fused, 'pseudo_huber_pairwise_distance_cuda',
                                   side_effect=AssertionError('fused path must not run')):
                actual = mmd_rbf.pairwise_distance(x, y, distance='pseudo_huber',
                                                   huber_delta=.2, block_size=3)
        self.assert_close(actual, reference_distance(x, y, .2))

    def test_float64_uses_portable_precision_and_bfloat16_promotes(self):
        x, y = self.random(1, 5, 11), self.random(1, 7, 11)
        with mock.patch.object(fused, 'pseudo_huber_pairwise_distance_cuda',
                               side_effect=AssertionError('FP64 must retain portable precision')):
            actual = mmd_rbf.pairwise_distance(x.double(), y.double(),
                                               distance='pseudo_huber', huber_delta=.2)
        self.assertEqual(actual.dtype, torch.float64)
        self.assert_close(actual, reference_distance(x, y, .2), atol=2e-12, rtol=2e-12)
        x, y = x.bfloat16(), y.bfloat16()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            actual = mmd_rbf.pairwise_distance(x, y, distance='pseudo_huber', huber_delta=.2)
        self.assertEqual(actual.dtype, torch.float32)
        self.assert_close(actual, reference_distance(x, y, .2))


def benchmark(repeats):
    """Time the token-draw MMD estimator with bounded scratch, no dense baseline."""
    if not torch.cuda.is_available() or not fused.supports_pseudo_huber_cuda(torch.empty(0, device='cuda')):
        raise SystemExit('Benchmark requires a free CUDA GPU and Triton')
    torch.manual_seed(930)
    x = torch.randn(1, 512, 768, device='cuda')
    y = torch.randn(1, 2048, 768, device='cuda')
    wx, wy = torch.ones(1, 512, device='cuda'), torch.ones(1, 2048, device='cuda')
    options = dict(distance='pseudo_huber', huber_delta=.001, block_size=0)
    outputs, timings = {}, {}
    for backend in ('fused_cuda', 'torch_fallback'):
        context = (nullcontext() if backend == 'fused_cuda' else
                   mock.patch.object(fused, 'supports_pseudo_huber_cuda', return_value=False))
        with context, torch.no_grad():
            # Warm-up includes JIT compilation; reported time excludes it.
            mmd_rbf.rbf_mmd2_unequal(x, y, wx, wy, 1 / 768, **options)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            started = time.perf_counter()
            for _ in range(repeats):
                output, _ = mmd_rbf.rbf_mmd2_unequal(x, y, wx, wy, 1 / 768, **options)
            torch.cuda.synchronize()
            timings[backend] = {
                'seconds_per_call': (time.perf_counter() - started) / repeats,
                'peak_extra_allocated_mib': (torch.cuda.max_memory_allocated() - baseline) / 2 ** 20,
            }
            outputs[backend] = output.clone()
    torch.testing.assert_close(outputs['fused_cuda'], outputs['torch_fallback'], atol=3e-6, rtol=3e-4)
    print(json.dumps({
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'shape': {'rows': 1, 'real_tokens': 512, 'fake_tokens': 2048, 'hidden': 768},
        'block_size': options['block_size'], 'repeats': repeats, 'timings': timings,
        'speedup': timings['torch_fallback']['seconds_per_call'] / timings['fused_cuda']['seconds_per_call'],
        'absolute_result_difference': float((outputs['fused_cuda'] - outputs['torch_fallback']).abs().max()),
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    arguments, remaining = parser.parse_known_args()
    if arguments.benchmark:
        if arguments.repeats < 1:
            parser.error('--repeats must be positive')
        benchmark(arguments.repeats)
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
