"""CPU equivalence tests for inference-branch methods inside the main runtime.

No GPU, downloaded model, dataset, or tracker is used. Run in a separate process:
python scripts/test_didi_inference_sampler.py
"""

import ast
from contextlib import nullcontext
import copy
from pathlib import Path
import sys
from types import MethodType
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import open_dict

from didi_repro import inference_sampler
from didi_repro.standalone import INFERENCE_REVISION, verify_source
from didi_repro.temperature import sampling_temperature
import test_didi_runtime as runtime_tests


def literal_methods(namespace):
    """Independently extract the two original methods with zero AST repairs."""
    directory, _ = verify_source('inference')
    path = directory / 'algo.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == 'DiDiInstruct')
    nodes = [copy.deepcopy(node) for node in owner.body
             if isinstance(node, ast.FunctionDef)
             and node.name in inference_sampler.METHOD_NAMES]
    environment = dict(vars(namespace))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), environment)
    return {node.name: environment[node.name] for node in nodes}


class InferenceSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime_tests.DiDiRuntimeTests.setUpClass()
        cls.helpers = runtime_tests.DiDiRuntimeTests()
        cls.upstream = cls.helpers.upstream
        cls.functions, cls.metadata = inference_sampler.load_inference_sampler(cls.upstream.algo)
        cls.literal = literal_methods(cls.upstream.algo)

    def model(self, source=None):
        configuration = self.helpers.corrected_config()
        configuration.sampling.predictor = 'guided'
        configuration.sampling.steps = 4
        configuration.sampling.noise_removal = 'ancestral'
        configuration.algo.guidance_scale_start = 30
        configuration.algo.guidance_scale_end = 40
        configuration.algo.num_candidates = 4
        configuration.algo.rerank_steps_ratio = .5
        if source is not None:
            with open_dict(configuration):
                configuration.didi_eval.sampler_source = source
        model = self.helpers.model(configuration)
        self.helpers.initialize(model)
        model.eval()
        return model

    def call(self, function, model, *args, **kwargs):
        # The original code explicitly requests CUDA autocast. On CPU its
        # disabled context is an identity; suppress only that device warning.
        with mock.patch.object(torch.amp, 'autocast', return_value=nullcontext()):
            return function(model, *args, **kwargs)

    def test_metadata_and_only_approved_ast_change(self):
        self.assertEqual(self.metadata['source'], 'inference')
        self.assertEqual(self.metadata['commit'], INFERENCE_REVISION)
        self.assertEqual(self.metadata['methods'], list(inference_sampler.METHOD_NAMES))
        for name, function in self.functions.items():
            expected = ast.parse(function.__didi_original_source__)
            actual = ast.parse(function.__didi_repaired_source__)
            if name == 'reward_guided_ancestral_step':
                # Independent textual replacement, followed by AST comparison.
                source = function.__didi_original_source__
                before = 'reward.backward()\n            grad = log_p_x0_for_grad.grad'
                self.assertEqual(source.count(before), 1)
                source = source.replace(
                    before, 'grad = torch.autograd.grad(reward, log_p_x0_for_grad, only_inputs=True)[0]')
                expected = ast.parse(source)
            self.assertEqual(ast.dump(actual), ast.dump(expected))
        self.assertEqual(len(self.metadata['repairs']), 1)

    def test_load_verifies_every_time_and_does_not_mutate_main_namespace(self):
        original_globals = dict(vars(self.upstream.algo))
        with mock.patch.object(inference_sampler, 'verify_source', wraps=verify_source) as verify:
            first, _ = inference_sampler.load_inference_sampler(self.upstream.algo)
            second, _ = inference_sampler.load_inference_sampler(self.upstream.algo)
        self.assertEqual(verify.call_args_list, [mock.call('inference'), mock.call('inference')])
        self.assertEqual(set(vars(self.upstream.algo)), set(original_globals))
        for name, value in original_globals.items():
            self.assertIs(vars(self.upstream.algo)[name], value)
        self.assertIsNot(first['generate_samples'], second['generate_samples'])

    def test_altered_source_and_unexpected_gradient_pattern_fail_closed(self):
        with mock.patch.object(inference_sampler, 'verify_source',
                               side_effect=RuntimeError('source changed')):
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                inference_sampler.load_inference_sampler(self.upstream.algo)
        function = ast.parse('def f():\n    pass').body[0]
        with self.assertRaisesRegex(RuntimeError, 'no longer matches once'):
            inference_sampler._repair_input_gradient(function)
        function = ast.parse('def f():\n    reward.backward()\n    grad = log_p_x0_for_grad.grad\n'
                             '    reward.backward()\n    grad = log_p_x0_for_grad.grad').body[0]
        with self.assertRaisesRegex(RuntimeError, 'no longer matches once'):
            inference_sampler._repair_input_gradient(function)

    def test_seeded_guidance_and_reranking_exactly_match_original_t1(self):
        torch.manual_seed(17)
        initial = self.model()
        draw = self.upstream.algo.trainer_base.sample_categorical
        for h, candidates in ((0, 1), (30, 1), (40, 1), (0, 4), (30, 4)):
            with self.subTest(h=h, candidates=candidates):
                reference, repaired = copy.deepcopy(initial), copy.deepcopy(initial)
                tokens = initial.prior_sample(2, initial.num_tokens)
                tokens[:, :3] = torch.tensor([1, 2, 3])
                t = torch.full((2, 1), .65)
                observed = []
                results = []
                for model, functions in ((reference, self.literal), (repaired, self.functions)):
                    probabilities = []

                    def record(probability):
                        probabilities.append(probability.detach().clone())
                        return draw(probability)

                    torch.manual_seed(341)
                    with mock.patch.object(self.upstream.algo.trainer_base, 'sample_categorical',
                                           side_effect=record):
                        results.append(self.call(functions['reward_guided_ancestral_step'],
                                                 model, tokens, t, dt=.2, h=h, M=candidates))
                    observed.append(probabilities)
                torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
                self.assertEqual(len(observed[0]), len(observed[1]))
                for expected, actual in zip(*observed):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(results[1][:, :3], tokens[:, :3], rtol=0, atol=0)
                self.assertTrue(all(parameter.grad is None for parameter in repaired.parameters()))
                if h > 0:
                    self.assertTrue(any(parameter.grad is not None
                                        for parameter in reference.discriminator.parameters()))

    def test_guidance_preserves_existing_parameter_gradients_and_weights(self):
        model = self.model()
        for index, parameter in enumerate(model.parameters()):
            if parameter.requires_grad and index % 2:
                parameter.grad = torch.full_like(parameter, .123)
        gradients = {name: (parameter.grad, None if parameter.grad is None else parameter.grad.clone())
                     for name, parameter in model.named_parameters()}
        weights = copy.deepcopy(model.state_dict())
        self.call(self.functions['reward_guided_ancestral_step'], model,
                  model.prior_sample(2, model.num_tokens), torch.full((2, 1), .6),
                  dt=.2, h=30, M=1)
        for name, parameter in model.named_parameters():
            reference, values = gradients[name]
            self.assertIs(parameter.grad, reference)
            if values is not None:
                torch.testing.assert_close(parameter.grad, values, rtol=0, atol=0)
        self.helpers.assert_same(model.state_dict(), weights)

    def test_unclipped_input_gradient_equals_original_backward_leaf_gradient(self):
        model = self.model()
        reference, repaired = copy.deepcopy(model), copy.deepcopy(model)
        tokens = model.prior_sample(2, model.num_tokens)
        tokens[:, :2] = torch.tensor([1, 2])
        t = torch.full((2, 1), .65)
        original_requires_grad = torch.Tensor.requires_grad_
        original_grad = torch.autograd.grad
        leaves, gradients = [], []

        def record_leaf(tensor, *args, **kwargs):
            value = original_requires_grad(tensor, *args, **kwargs)
            leaves.append(value)
            return value

        def record_gradient(*args, **kwargs):
            values = original_grad(*args, **kwargs)
            gradients.append(values[0].detach().clone())
            return values

        with mock.patch.object(torch.Tensor, 'requires_grad_', record_leaf):
            self.call(self.literal['reward_guided_ancestral_step'], reference,
                      tokens, t, dt=.2, h=30, M=1)
        with mock.patch.object(torch.autograd, 'grad', side_effect=record_gradient):
            self.call(self.functions['reward_guided_ancestral_step'], repaired,
                      tokens, t, dt=.2, h=30, M=1)
        self.assertEqual(len(leaves), 1)
        self.assertEqual(len(gradients), 1)
        self.assertIsNotNone(leaves[0].grad)
        torch.testing.assert_close(gradients[0], leaves[0].grad, rtol=0, atol=0)
        self.assertTrue((gradients[0][:, 2:].abs() > 0).any())
        self.assertTrue((gradients[0][:, :2] == 0).all())

    def test_actual_nonzero_sigma_reaches_ema_not_processed_forward(self):
        model = self.model()
        self.assertFalse(model.time_conditioning)
        tokens = model.prior_sample(2, model.num_tokens)
        t = torch.full((2, 1), .65)
        with mock.patch.object(model.student_ema, 'forward', wraps=model.student_ema.forward) as ema, \
                mock.patch.object(model, 'forward', side_effect=AssertionError('processed forward called')), \
                mock.patch.object(model.backbone, 'forward', side_effect=AssertionError('raw backbone called')):
            self.call(self.functions['reward_guided_ancestral_step'], model, tokens, t,
                      dt=.2, h=30, M=1)
        sigma = ema.call_args.kwargs['sigma']
        expected = model._sigma_from_alphat(model.noise(t)[1]).squeeze(-1)
        torch.testing.assert_close(sigma, expected, rtol=0, atol=0)
        self.assertTrue((sigma > 0).all())

    def test_full_seeded_samples_sigma_schedule_and_raw_cleanup_match_original(self):
        torch.manual_seed(59)
        initial = self.model()
        for steps in (4, 8, 16, 32):
            with self.subTest(steps=steps):
                outputs, schedules = [], []
                for functions in (self.literal, self.functions):
                    model = copy.deepcopy(initial)
                    calls = []
                    method = functions['reward_guided_ancestral_step']

                    def record(x, t, dt, h, M):
                        calls.append((h, M))
                        return method(model, x, t, dt, h, M)

                    model.reward_guided_ancestral_step = record
                    # Outside the runtime's eval context, the authors' terminal
                    # _denoiser_update uses raw backbone, not student_ema.
                    with mock.patch.object(model.backbone, 'forward', wraps=model.backbone.forward) as raw, \
                            mock.patch.object(model.student_ema, 'forward', wraps=model.student_ema.forward) as ema:
                        torch.manual_seed(687)
                        outputs.append(self.call(functions['generate_samples'], model, 2, num_steps=steps))
                    self.assertEqual(ema.call_count, steps)
                    self.assertEqual(raw.call_count, 1)
                    self.assertTrue(model.backbone.training)
                    self.assertTrue(model.discriminator.training)
                    self.assertFalse(model.student_ema.training)
                    schedules.append(calls)
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                self.assertFalse((outputs[1] == initial.mask_index).any())
                self.assertEqual(schedules[0], schedules[1])
                half = steps // 2
                expected = [(30 + (i / (half - 1)) * 10, 1) for i in range(half)]
                expected += [(0, 4)] * (steps - half)
                self.assertEqual(schedules[1], expected)

    def test_h_zero_still_reranks_using_unscaled_reward_softmax(self):
        model = self.model()
        candidates = torch.tensor([[[0, 1], [1, 0], [1, 1], [0, 0]]])
        probabilities = []

        def draw(probability):
            probabilities.append(probability.clone())
            return candidates.clone() if probability.ndim == 4 else torch.tensor([1])

        with mock.patch.object(self.upstream.algo.trainer_base, 'sample_categorical', side_effect=draw), \
                mock.patch.object(model.discriminator, 'forward',
                                  side_effect=lambda x, sigma: x.float().unsqueeze(-1)):
            result = self.call(self.functions['reward_guided_ancestral_step'], model,
                               model.prior_sample(1, 2), torch.tensor([[.5]]),
                               dt=.25, h=0, M=4)
        self.assertEqual(len(probabilities), 2)
        torch.testing.assert_close(probabilities[1], torch.tensor([[1., 1., 2., 0.]]).softmax(-1))
        torch.testing.assert_close(result, candidates[:, 1], rtol=0, atol=0)

    def test_real_runtime_t1_matches_literal_with_distinct_raw_ema_and_inference_mode(self):
        torch.manual_seed(43)
        initial = self.model(source='inference')
        with torch.no_grad():
            for parameter in initial.backbone.parameters():
                parameter.add_(.15)
        self.assertTrue(any(not torch.equal(raw, ema) for raw, ema in zip(
            initial.backbone.parameters(), initial.student_ema.parameters())))
        for steps in (4, 8):
            with self.subTest(steps=steps):
                reference, integrated = copy.deepcopy(initial), copy.deepcopy(initial)
                reference.reward_guided_ancestral_step = MethodType(
                    self.literal['reward_guided_ancestral_step'], reference)
                torch.manual_seed(302)
                expected = self.call(self.literal['generate_samples'], reference, 3, num_steps=steps)
                before = copy.deepcopy(integrated.state_dict())
                modes = [module.training for module in integrated.modules()]
                with mock.patch.object(integrated.backbone, 'forward',
                                       wraps=integrated.backbone.forward) as raw, \
                        mock.patch.object(integrated.student_ema, 'forward',
                                          wraps=integrated.student_ema.forward) as ema, \
                        mock.patch.object(torch.amp, 'autocast', return_value=nullcontext()):
                    torch.manual_seed(302)
                    with torch.inference_mode(), sampling_temperature(integrated, 1):
                        actual = integrated.generate_samples(3, num_steps=steps)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(raw.call_count, 1)
                self.assertEqual(ema.call_count, steps)
                self.assertEqual(integrated._didi_inference_sampling_depth, 0)
                self.assertEqual(integrated._didi_evaluation_depth, 0)
                self.assertEqual(modes, [module.training for module in integrated.modules()])
                self.assertTrue(all(parameter.grad is None for parameter in integrated.parameters()))
                self.helpers.assert_same(integrated.state_dict(), before)

    def test_real_runtime_restores_depth_and_mixed_modes_after_nested_failure(self):
        model = self.model(source='inference')
        model.backbone.train()
        model.teacher.eval()
        model.discriminator.train()
        model.student_ema.eval()
        modes = [module.training for module in model.modules()]
        with model._didi_evaluation_context():
            self.assertEqual(model._didi_evaluation_depth, 1)
            with mock.patch.object(model.discriminator, 'forward',
                                   side_effect=RuntimeError('test discriminator failure')), \
                    mock.patch.object(torch.amp, 'autocast', return_value=nullcontext()), \
                    self.assertRaisesRegex(RuntimeError, 'test discriminator failure'):
                model.generate_samples(2, num_steps=4)
            self.assertEqual(model._didi_evaluation_depth, 1)
            self.assertEqual(model._didi_inference_sampling_depth, 0)
        self.assertEqual(model._didi_evaluation_depth, 0)
        self.assertEqual(modes, [module.training for module in model.modules()])


if __name__ == '__main__':
    unittest.main(verbosity=2)
