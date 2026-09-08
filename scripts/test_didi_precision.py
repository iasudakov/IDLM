"""CPU checks for the opt-in float64 sampling-arithmetic diagnostic.

Run: CUDA_VISIBLE_DEVICES='' python scripts/test_didi_precision.py
No model downloads, training, tracker, or GPU are used. References do not use
the production replacement table or categorical implementation.
"""

import ast
from contextlib import nullcontext
import copy
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from didi_repro import inference_precision as precision
from didi_repro.standalone import INFERENCE_REVISION, verify_source


def literal_method(name, namespace):
    path = ROOT / 'vendor/didi_inference/algo.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == 'DiDiInstruct')
    node = next(node for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    globals_ = dict(vars(namespace))
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]),
                 str(path), 'exec'), globals_)
    return globals_[name]


def literal_categorical(probabilities):
    noise = 1e-10 - (torch.rand_like(probabilities) + 1e-10).log()
    return (probabilities / noise).argmax(dim=-1)


def algorithm_namespace():
    return SimpleNamespace(torch=torch, F=F, MDLM=object,
                           trainer_base=SimpleNamespace(sample_categorical=literal_categorical))


class TinyGenerator(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer('logits', torch.tensor([.103, 1.003, -.9, .7], dtype=dtype))
        self.calls = []

    def forward(self, x, sigma):
        self.calls.append((x.clone(), sigma.clone(), x.dtype, sigma.dtype))
        return self.logits.expand(*x.shape, -1).clone()


class TinyDiscriminator(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(7.5))
        self.dtype = dtype
        self.calls = []

    def forward(self, x, sigma):
        self.calls.append((x.detach().clone(), sigma.clone(), x.dtype, sigma.dtype))
        coefficients = torch.tensor([.25, -.75, .5, .1])
        if x.ndim == 3:
            scores = (x.softmax(-1) * coefficients).sum(-1, keepdim=True)
        else:
            scores = coefficients[x].unsqueeze(-1)
        return (scores * self.gain).to(self.dtype)


class TinyModel:
    def __init__(self, dtype=torch.bfloat16):
        self.student_ema = TinyGenerator(dtype)
        self.discriminator = TinyDiscriminator(dtype)
        self.backbone = torch.nn.Linear(1, 1)
        self.mask_index, self.vocab_size = 3, 4
        self.noise_calls, self.sigma_calls, self.cleanup_calls = [], [], []
        self.device, self.num_tokens = torch.device('cpu'), 4
        self.sampler = 'guided'
        self.guidance_scale_start, self.guidance_scale_end = 30., 40.
        self.rerank_steps_ratio, self.num_candidates = .5, 4
        self.config = SimpleNamespace(mode='cpu_test', sampling=SimpleNamespace(
            steps=8, noise_removal='ancestral'))

    def noise(self, t):
        self.noise_calls.append(t.detach().clone())
        return torch.full_like(t, -.999), 1 - .999 * t

    def _sigma_from_alphat(self, alpha):
        self.sigma_calls.append(alpha.detach().clone())
        return -alpha.log()

    def prior_sample(self, batch, length):
        return torch.full((batch, length), self.mask_index, dtype=torch.long)

    def _denoiser_update(self, x, t):
        self.cleanup_calls.append((x.clone(), t.clone()))
        return x.masked_fill(x == self.mask_index, 0)


@torch.no_grad()
def reference_step(model, x, t, dt, h, candidates, draw):
    """Independent reference: original networks/gradient, float64 afterwards."""
    sigma = model._sigma_from_alphat(model.noise(t)[1]).squeeze(-1)
    logits = model.student_ema(x, sigma=sigma)
    if h > 0:
        mask = x == model.mask_index
        with torch.enable_grad():
            leaf = logits.clone().requires_grad_()
            observed = F.one_hot(x.clamp(min=0), model.vocab_size).float() * 1e9
            discriminator_input = torch.where(mask.unsqueeze(-1), leaf, observed)
            scores = model.discriminator(discriminator_input, sigma=sigma)
            objective = (scores.squeeze(-1) * mask).sum()
            objective.backward()
            gradient = leaf.grad
        gradient = torch.where(mask.unsqueeze(-1), gradient, torch.zeros_like(gradient))
        gradient = gradient.clamp_(-1., 1.)
        logits = logits.double() + h * gradient.detach().double()
    probabilities = logits.double().softmax(-1)
    _, alpha_t = model.noise(t)
    _, alpha_s = model.noise(t - dt)
    alpha_t, alpha_s = alpha_t.double(), alpha_s.double()
    transition = probabilities * (alpha_s - alpha_t).unsqueeze(-1)
    transition[:, :, model.mask_index] = 1 - alpha_s
    if candidates <= 1:
        sampled = draw(transition)
        return torch.where(x != model.mask_index, x, sampled)
    batch, length = x.shape
    sampled = draw(transition.unsqueeze(1).expand(-1, candidates, -1, -1))
    candidate_tokens = torch.where((x != model.mask_index).unsqueeze(1), x.unsqueeze(1), sampled)
    sigma_s = model._sigma_from_alphat(model.noise(t - dt)[1]).squeeze(-1)
    scores = model.discriminator(candidate_tokens.reshape(batch * candidates, length),
                                 sigma=sigma_s.repeat_interleave(candidates))
    rewards = scores.double().sum(dim=[1, 2]).view(batch, candidates)
    selected = draw(rewards.softmax(-1)).view(batch, 1, 1).expand(-1, -1, length)
    return candidate_tokens.gather(1, selected).squeeze(1)


class CategoricalBufferTests(unittest.TestCase):
    def test_matches_literal_samples_rng_and_does_not_mutate_any_layout(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            base = torch.linspace(.01, 2., 2 * 3 * 17, dtype=dtype).reshape(2, 3, 17)
            layouts = [base, base.transpose(0, 1), base.unsqueeze(1).expand(-1, 4, -1, -1)]
            for probabilities in layouts:
                with self.subTest(dtype=dtype, shape=probabilities.shape, stride=probabilities.stride()):
                    before, base_before = probabilities.clone(), base.clone()
                    torch.manual_seed(539)
                    expected = literal_categorical(probabilities)
                    rng = torch.get_rng_state()
                    torch.manual_seed(539)
                    actual = precision.sample_categorical_reusing_buffer(probabilities)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                    torch.testing.assert_close(probabilities, before, rtol=0, atol=0)
                    torch.testing.assert_close(base, base_before, rtol=0, atol=0)

    def test_exactly_one_full_shape_random_draw_including_expanded_candidates(self):
        base = torch.rand(2, 3, 7, dtype=torch.float64)
        probabilities = base.unsqueeze(1).expand(-1, 4, -1, -1)
        rand_like = torch.rand_like
        with mock.patch.object(torch, 'rand_like', wraps=rand_like) as random:
            actual = precision.sample_categorical_reusing_buffer(probabilities)
        random.assert_called_once_with(probabilities)
        self.assertEqual(actual.shape, (2, 4, 3))
        self.assertEqual(actual.dtype, torch.long)

    def test_literal_equivalence_for_zero_probabilities_and_boundary_random_values(self):
        probabilities = torch.tensor([[0., 1., 0., 2.], [0., 0., 0., 0.]], dtype=torch.float64)
        uniforms = torch.tensor([[0., .5, 1., 1.], [0., .5, 1., .1]], dtype=torch.float64)
        with mock.patch.object(torch, 'rand_like', side_effect=lambda x: uniforms.clone()):
            expected = literal_categorical(probabilities)
            actual = precision.sample_categorical_reusing_buffer(probabilities)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class PrecisionStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def call(self, function, model, *args, **kwargs):
        with mock.patch.object(torch.amp, 'autocast', return_value=nullcontext()):
            return function(model, *args, **kwargs)

    def test_float64_probabilities_match_independent_reference_for_all_phases(self):
        x = torch.tensor([[3, 0, 3, 1], [2, 3, 3, 3]])
        t = torch.tensor([[.65], [.37]])
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for h, candidates in ((0, 1), (30, 1), (40, 1), (0, 4), (30, 4)):
                with self.subTest(dtype=dtype, h=h, candidates=candidates):
                    reference, actual = TinyModel(dtype), TinyModel(dtype)
                    expected_probabilities, observed = [], []

                    def draw(probabilities):
                        expected_probabilities.append(probabilities.clone())
                        return literal_categorical(probabilities)

                    function, _ = precision.load_precision_sampler(
                        algorithm_namespace(), observer=lambda stage, value: observed.append(
                            (stage, value.detach().clone())))
                    torch.manual_seed(93)
                    expected = reference_step(reference, x, t, .125, h, candidates, draw)
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(93)
                    result = self.call(function, actual, x, t, .125, h, candidates)
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                    categorical = [value for stage, value in observed if stage == 'categorical']
                    self.assertEqual(len(categorical), 1 if candidates <= 1 else 2)
                    for value, expected in zip(categorical, expected_probabilities):
                        self.assertEqual(value.dtype, torch.float64)
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    self.assertEqual([v.dtype for stage, v in observed if stage.endswith('_output')],
                                     [torch.float64] * sum(s.endswith('_output') for s, _ in observed))
                    torch.testing.assert_close(result[x != 3], x[x != 3], rtol=0, atol=0)

    def test_noise_sigma_network_inputs_and_original_backward_are_unchanged(self):
        x = torch.tensor([[3, 0, 3, 1], [2, 3, 3, 3]])
        t = torch.tensor([[.65], [.37]])
        for dtype in (torch.float16, torch.bfloat16):
            reference, diagnostic = TinyModel(dtype), TinyModel(dtype)
            namespace = algorithm_namespace()
            original = literal_method('reward_guided_ancestral_step', namespace)
            function, _ = precision.load_precision_sampler(namespace)
            leaves = []
            original_requires_grad = torch.Tensor.requires_grad_

            def capture(tensor, *args, **kwargs):
                result = original_requires_grad(tensor, *args, **kwargs)
                leaves.append(result)
                return result

            with mock.patch.object(torch.Tensor, 'requires_grad_', capture), \
                    mock.patch.object(torch.autograd, 'grad', side_effect=AssertionError('gradient repair invoked')):
                self.call(original, reference, x, t, .125, 30, 1)
                self.call(function, diagnostic, x, t, .125, 30, 1)
            self.assertEqual(len(leaves), 2)
            self.assertEqual(leaves[1].dtype, dtype)
            torch.testing.assert_close(leaves[1].grad, leaves[0].grad, rtol=0, atol=0)
            self.assertGreater(float(leaves[1].grad.abs().sum()), 0)
            self.assertTrue((leaves[1].grad[x != 3] == 0).all())
            self.assertIsNotNone(diagnostic.discriminator.gain.grad)
            torch.testing.assert_close(diagnostic.discriminator.gain.grad,
                                       reference.discriminator.gain.grad, rtol=0, atol=0)
            for module in ('student_ema', 'discriminator'):
                calls = [getattr(model, module).calls for model in (reference, diagnostic)]
                self.assertEqual(len(calls[0]), len(calls[1]))
                for before, after in zip(*calls):
                    self.assertEqual(before[2:], after[2:])
                    torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
                    torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)
            for name in ('noise_calls', 'sigma_calls'):
                self.assertEqual(len(getattr(reference, name)), len(getattr(diagnostic, name)))
                for before, after in zip(getattr(reference, name), getattr(diagnostic, name)):
                    self.assertEqual(after.dtype, torch.float32)
                    torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_rerank_reduces_scores_in_double_before_low_precision_overflow(self):
        observed = []
        function, _ = precision.load_precision_sampler(
            algorithm_namespace(), observer=lambda stage, value: observed.append((stage, value.clone())))
        model = TinyModel(torch.float16)
        scores = torch.tensor([[[65504.], [65504.]], [[65504.], [65472.]]], dtype=torch.float16)
        model.discriminator = mock.Mock(return_value=scores)
        self.call(function, model, torch.full((1, 2), 3), torch.tensor([[.5]]), .25, 0, 2)
        categorical = [value for stage, value in observed if stage == 'categorical']
        expected = scores.double().sum([1, 2]).view(1, 2).softmax(-1)
        self.assertTrue(torch.isfinite(categorical[1]).all())
        torch.testing.assert_close(categorical[1], expected, rtol=0, atol=0)
        self.assertFalse(torch.isfinite(scores.sum([1, 2]).softmax(-1)).all())
        sigma = model.discriminator.call_args.kwargs['sigma']
        self.assertEqual(sigma.dtype, torch.float32)

    def test_unmodified_generate_keeps_guidance_grid_and_cleanup(self):
        namespace = algorithm_namespace()
        generate = literal_method('generate_samples', namespace)
        function, _ = precision.load_precision_sampler(namespace)
        for steps in (8, 16, 32):
            model, schedule = TinyModel(), []

            def step(x, t, dt, h, M):
                schedule.append((h, M, t.clone(), dt))
                return function(model, x, t, dt, h, M)

            model.reward_guided_ancestral_step = step
            self.call(generate, model, 2, num_steps=steps)
            half = steps // 2
            expected = [(30 + 10 * i / (half - 1), 1) for i in range(half)]
            expected += [(0, 4)] * (steps - half)
            for i, ((h, candidates, t, dt), (expected_h, expected_m)) in enumerate(zip(schedule, expected)):
                self.assertAlmostEqual(h, expected_h)
                self.assertEqual(candidates, expected_m)
                self.assertEqual(t.dtype, torch.float32)
                torch.testing.assert_close(t, torch.linspace(1, 1e-5, steps + 1)[i].expand(2, 1))
                self.assertAlmostEqual(dt, (1 - 1e-5) / steps)
            self.assertEqual(len(schedule), steps)
            self.assertEqual(len(model.cleanup_calls), 1)
            self.assertEqual(model.cleanup_calls[0][0].dtype, torch.long)
            self.assertEqual(model.cleanup_calls[0][1].dtype, torch.float32)
            self.assertTrue(model.backbone.training)
            self.assertTrue(model.discriminator.training)
            self.assertFalse(model.student_ema.training)


class PrecisionIsolationTests(unittest.TestCase):
    def test_ast_changes_only_five_approved_arithmetic_sites(self):
        function, record = precision.load_precision_sampler(algorithm_namespace())
        expected = function.__didi_original_source__
        # Independent whitelist, deliberately not sourced from REPLACEMENTS.
        edits = [
            ('log_p_x0 = log_p_x0 + h * grad_clipped.detach()',
             "log_p_x0 = _didi_double(log_p_x0, 'tilt_logits') + h * _didi_double(grad_clipped.detach(), 'tilt_gradient')"),
            ('p_x0 = log_p_x0.softmax(dim=-1)',
             "p_x0 = _didi_double(log_p_x0, 'probability_logits').softmax(dim=-1)"),
            ('_, alpha_t = self.noise(t)',
             "_, alpha_t = self.noise(t)\n    alpha_t = _didi_double(alpha_t, 'alpha_t')"),
            ('_, alpha_s = self.noise(t - dt)',
             "_, alpha_s = self.noise(t - dt)\n    alpha_s = _didi_double(alpha_s, 'alpha_s')"),
            ('reward_flat = discriminator_scores_flat.sum(dim=[1, 2])',
             "reward_flat = _didi_double(discriminator_scores_flat, 'rerank_scores').sum(dim=[1, 2])"),
        ]
        for before, after in edits:
            before = ast.unparse(ast.parse(before).body[0])
            self.assertEqual(expected.count(before), 1)
            expected = expected.replace(before, after)
        self.assertEqual(ast.dump(ast.parse(function.__didi_precision_source__)),
                         ast.dump(ast.parse(expected)))
        self.assertIn('reward.backward()', function.__didi_precision_source__)
        self.assertEqual(record['upstream_revision'], INFERENCE_REVISION)
        self.assertEqual(record['method'], 'reward_guided_ancestral_step')
        self.assertTrue(record['not_full_float64_model_or_evaluation'])

    def test_loading_preserves_original_namespace_and_verifies_pinned_source(self):
        namespace = algorithm_namespace()
        before = dict(vars(namespace))
        original_categorical = namespace.trainer_base.sample_categorical
        with mock.patch.object(precision, 'verify_source', wraps=verify_source) as verify:
            function, _ = precision.load_precision_sampler(namespace)
        self.assertTrue(verify.called)
        self.assertTrue(all(call == mock.call('inference') for call in verify.call_args_list))
        self.assertEqual(set(vars(namespace)), set(before))
        for key, value in before.items():
            self.assertIs(getattr(namespace, key), value)
        self.assertIs(namespace.trainer_base.sample_categorical, original_categorical)
        self.assertIsNot(inspect.unwrap(function).__globals__['trainer_base'], namespace.trainer_base)

    def test_unavailable_or_changed_source_and_missing_globals_fail_closed(self):
        with mock.patch.object(precision, 'verify_source', side_effect=RuntimeError('changed source')):
            with self.assertRaisesRegex(RuntimeError, 'changed source'):
                precision.load_precision_sampler(algorithm_namespace())
        with self.assertRaises(ValueError):
            precision.load_precision_sampler(SimpleNamespace(torch=torch))
        with mock.patch.object(precision, 'REPLACEMENTS',
                               (('nonexistent = original', 'nonexistent = changed'),)):
            with self.assertRaisesRegex(RuntimeError, 'exactly one match'):
                precision.load_precision_sampler(algorithm_namespace())

    def test_diagnostic_proxy_rejects_non_double_probabilities(self):
        function, _ = precision.load_precision_sampler(algorithm_namespace())
        categorical = inspect.unwrap(function).__globals__['trainer_base'].sample_categorical
        with self.assertRaisesRegex(RuntimeError, 'float64'):
            categorical(torch.ones(2, 3, dtype=torch.float32))


class PrecisionLauncherTests(unittest.TestCase):
    def test_precision_inference_and_base_launchers_have_valid_shell_syntax(self):
        for name in ('eval_didi_precision.sh', 'eval_didi_inference.sh', 'eval_didi_authors.sh'):
            subprocess.run(['bash', '-n', str(ROOT / 'scripts' / name)],
                           check=True, capture_output=True, text=True)

    def test_precision_wrapper_selects_float64_and_delegates_to_inference(self):
        with tempfile.TemporaryDirectory(prefix='didi-precision-shell-test-') as folder:
            directory = Path(folder) / 'scripts'
            directory.mkdir()
            for name in ('eval_didi_precision.sh', 'eval_didi_inference.sh'):
                shutil.copyfile(ROOT / 'scripts' / name, directory / name)
            (directory / 'eval_didi_authors.sh').write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$DIDI_INFERENCE_PRECISION" '
                '"$DIDI_AUTHORS_SOURCE" "$@"\n')
            environment = {**os.environ, 'DIDI_INFERENCE_PRECISION': 'upstream',
                           'DIDI_AUTHORS_SOURCE': 'main'}
            result = subprocess.run(
                ['bash', str(directory / 'eval_didi_precision.sh'), 'sampling.steps=8', 'seed=17'],
                env=environment, check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.splitlines(),
                             ['float64', 'inference', 'sampling.steps=8', 'seed=17'])

    def test_invalid_precision_and_float64_main_reject_before_checkpoint_checks(self):
        for selection, source, message in (
                ('bogus', 'inference', 'must be upstream or float64'),
                ('float64', 'main', 'requires the inference source')):
            environment = {**os.environ, 'DIDI_INFERENCE_PRECISION': selection,
                           'DIDI_AUTHORS_SOURCE': source}
            environment.pop('CHECKPOINT', None)
            environment.pop('SAVE_DIR', None)
            result = subprocess.run(['bash', str(ROOT / 'scripts/eval_didi_authors.sh')],
                                    env=environment, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
            self.assertNotIn('set CHECKPOINT', result.stderr)

    def test_default_precision_remains_upstream(self):
        environment = {**os.environ, 'DIDI_AUTHORS_SOURCE': 'main'}
        for key in ('CHECKPOINT', 'SAVE_DIR', 'DIDI_INFERENCE_PRECISION'):
            environment.pop(key, None)
        result = subprocess.run(['bash', str(ROOT / 'scripts/eval_didi_authors.sh')],
                                env=environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('set CHECKPOINT', result.stderr)
        self.assertNotIn('precision diagnostic requires', result.stderr)


if __name__ == '__main__':
    unittest.main()
