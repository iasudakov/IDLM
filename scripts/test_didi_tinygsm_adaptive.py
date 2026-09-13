"""CPU-only reference threshold-decoding tests; no network, GPU or checkpoints."""

import ast
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from didi_repro.tinygsm_adaptive import (
    _propose_tokens, _select_threshold, generate_adaptive,
)


class LinearNoise(nn.Module):
    eps = 1e-3

    def forward(self, t):
        return -(1 - self.eps), 1 - (1 - self.eps) * t


def probabilities(confidences, vocab=16):
    values = torch.zeros(len(confidences), vocab)
    for position, confidence in enumerate(confidences):
        values[position, :-1] = (1 - confidence) / (vocab - 2)
        values[position, position % (vocab - 1)] = confidence
    return values


class FixedBackbone(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer('logits', values.log())
        self.weight = nn.Parameter(torch.tensor(0.))
        self.calls = []

    def forward(self, tokens, sigma):
        self.calls.append({'tokens': tokens.clone(), 'sigma': sigma.clone(),
                           'grad_enabled': torch.is_grad_enabled(),
                           'training': self.training})
        return self.logits[None].expand(tokens.shape[0], -1, -1).clone() + self.weight * 0


class ForbiddenDiscriminator(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError('Adaptive reference decoding must never call the discriminator.')


class Model(nn.Module):
    def __init__(self, confidences=None, weights='student_ema', time_conditioning=False):
        super().__init__()
        confidences = confidences or [.99, .9, .8, .6, .4, .2]
        self.num_tokens = len(confidences)
        self.vocab_size = 16
        self.mask_index = 15
        self.device = torch.device('cpu')
        self.parameterization = 'subs'
        self.noise = LinearNoise()
        self.backbone = FixedBackbone(probabilities([.95] * self.num_tokens))
        self.student_ema = FixedBackbone(probabilities(confidences))
        self.discriminator = ForbiddenDiscriminator()
        self.time_conditioning = time_conditioning
        self._didi_evaluation_weights = weights
        self._didi_in_training_step = False
        self._didi_inference_sampling_depth = 0
        self._didi_evaluation_depth = 0

    @contextmanager
    def _didi_evaluation_context(self):
        self._didi_evaluation_depth += 1
        try:
            yield
        finally:
            self._didi_evaluation_depth -= 1

    def _sigma_from_alphat(self, alpha_t):
        return -alpha_t.log()

    def forward(self, tokens, sigma):
        sigma = sigma.mean(-1).reshape(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        model = self.student_ema if self._didi_evaluation_weights == 'student_ema' else self.backbone
        logits = model(tokens, sigma)
        logits[..., self.mask_index] -= 1e6
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        observed = tokens != self.mask_index
        logits[observed] = -1e6
        logits[observed, tokens[observed]] = 0
        return logits


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.prefixes = torch.tensor([[7, 8, 6, 6], [8, 7, 6, 5]])
        self.lengths = torch.tensor([2, 4])

    def sample(self, model=None, **kwargs):
        return generate_adaptive(model or Model(), self.prefixes, self.lengths, **kwargs)

    def test_threshold_is_strict_and_stall_forces_first_argmax_tie(self):
        confidence = torch.tensor([[.8, .8, .7], [1., .9, .8]])
        candidates = torch.tensor([[True, True, True], [False, True, True]])
        confidence = confidence.masked_fill(~candidates, -torch.inf)
        got = _select_threshold(confidence, candidates, .8)
        self.assertTrue(torch.equal(got, torch.tensor([[True, False, False], [False, True, False]])))
        got = _select_threshold(confidence, candidates, 1.)
        self.assertTrue(torch.equal(got, torch.tensor([[True, False, False], [False, True, False]])))

    def test_completed_row_is_never_forced(self):
        candidates = torch.tensor([[False, False], [False, True]])
        confidence = torch.tensor([[-torch.inf, -torch.inf], [-torch.inf, .2]])
        self.assertTrue(torch.equal(_select_threshold(confidence, candidates, .9), candidates))

    def test_threshold_one_terminates_in_suffix_length_and_reports_real_nfe(self):
        model = Model()
        output, metadata = self.sample(model, threshold=1.)
        self.assertEqual(metadata['nfe'], 4)
        self.assertEqual(metadata['steps'], 4)
        self.assertEqual(metadata['fallback_nfe'], 0)
        self.assertEqual(metadata['nfe_per_row'].tolist(), [4., 2.])
        self.assertEqual(metadata['physical_row_forwards'], 8)
        self.assertEqual(len(model.student_ema.calls), 4)
        self.assertTrue(all(call['tokens'].shape[0] == 2 for call in model.student_ema.calls))
        self.assertFalse(bool((output == model.mask_index).any()))

    def test_zero_threshold_commits_entire_suffix_in_one_forward(self):
        output, metadata = self.sample(threshold=0.)
        self.assertEqual(metadata['nfe'], 1)
        self.assertEqual(metadata['nfe_per_row'].tolist(), [1., 1.])
        self.assertTrue(metadata['complete'])
        self.assertEqual(metadata['max_steps'], 6)

    def test_optional_cap_adds_one_real_flush_and_only_active_row_cost(self):
        output, metadata = self.sample(threshold=1., max_steps=2)
        self.assertEqual(metadata['max_steps'], 2)
        self.assertEqual(metadata['steps'], 2)
        self.assertEqual(metadata['nfe'], 3)
        self.assertEqual(metadata['fallback_nfe'], 1)
        self.assertEqual(metadata['nfe_per_row'].tolist(), [3., 2.])
        self.assertEqual(metadata['physical_row_forwards'], 6)
        self.assertFalse(bool((output == 15).any()))

    def test_proposals_use_raw_confidence_at_greedy_and_nonzero_temperature(self):
        values = probabilities([.9, .4, .2])[None]
        greedy, confidence = _propose_tokens(values, 0.)
        torch.testing.assert_close(confidence, torch.tensor([[.9, .4, .2]]))
        self.assertTrue(torch.equal(greedy, torch.tensor([[0, 1, 2]])))
        for temperature in (1., .7, 1.2):
            torch.manual_seed(17)
            proposals, confidence = _propose_tokens(values, temperature)
            torch.testing.assert_close(confidence, values.gather(-1, proposals[..., None]).squeeze(-1))

    def test_greedy_draw_consumes_no_rng_and_fixed_prefixes_are_untouched(self):
        torch.manual_seed(31)
        before = torch.get_rng_state().clone()
        output, _ = self.sample(threshold=.7)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for row, length in enumerate(self.lengths):
            self.assertTrue(torch.equal(output[row, :length], self.prefixes[row, :length]))
        # Changing junk beyond each prompt cannot change decoding.
        other = self.prefixes.clone()
        other[0, 2:] = torch.tensor([10, 11])
        reference, _ = generate_adaptive(Model(), other, self.lengths, threshold=.7)
        self.assertTrue(torch.equal(reference, output))

    def test_only_selected_weights_are_used_and_no_discriminator(self):
        for weights in ('backbone', 'student_ema'):
            with self.subTest(weights=weights):
                model = Model(weights=weights)
                _, metadata = self.sample(model, threshold=.7)
                selected = getattr(model, weights)
                unused = model.student_ema if weights == 'backbone' else model.backbone
                self.assertEqual(metadata['nfe'], len(selected.calls))
                self.assertEqual(len(unused.calls), 0)
                self.assertEqual(metadata['weights'], weights)
                self.assertTrue(all(not call['grad_enabled'] and not call['training'] for call in selected.calls))
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_sigma_uses_full_width_mask_fraction_and_configured_time_conditioning(self):
        model = Model(time_conditioning=True)
        self.sample(model, threshold=1., max_steps=1)
        first, flush = model.student_ema.calls
        # Four unknown of six positions in row 0, two of six in row 1.
        torch.testing.assert_close(first['sigma'], -torch.tensor([2/6, 4/6]).log())
        expected_flush = -torch.tensor(1 - (1 - model.noise.eps) * 1e-5).log()
        torch.testing.assert_close(flush['sigma'], expected_flush.expand(2))
        unconditional_time = Model(time_conditioning=False)
        self.sample(unconditional_time, threshold=.9)
        self.assertTrue(all(bool((call['sigma'] == 0).all()) for call in unconditional_time.student_ema.calls))

    def test_restores_mixed_module_modes_after_success_and_failure(self):
        model = Model()
        model.train()
        model.student_ema.eval()
        before = [module.training for module in model.modules()]
        with torch.inference_mode():
            self.sample(model, threshold=.9)
        self.assertEqual(before, [module.training for module in model.modules()])
        with mock.patch.object(model, 'forward', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                self.sample(model, threshold=.9)
        self.assertEqual(before, [module.training for module in model.modules()])
        self.assertEqual(model._didi_evaluation_depth, 0)

    def test_preserves_existing_parameter_gradients(self):
        model = Model()
        model.backbone.weight.grad = torch.tensor(7.)
        model.student_ema.weight.grad = torch.tensor(3.)
        backbone_grad = model.backbone.weight.grad
        ema_grad = model.student_ema.weight.grad
        self.sample(model, threshold=.7)
        self.assertIs(model.backbone.weight.grad, backbone_grad)
        self.assertIs(model.student_ema.weight.grad, ema_grad)
        self.assertEqual(float(backbone_grad), 7.)
        self.assertEqual(float(ema_grad), 3.)

    def test_invalid_options_fail_before_any_forward(self):
        for kwargs in ({'threshold': -1}, {'threshold': 1.1}, {'threshold': float('nan')},
                       {'threshold': True}, {'threshold': .5, 'temperature': -1},
                       {'threshold': .5, 'temperature': float('inf')},
                       {'threshold': .5, 'max_steps': 0}, {'threshold': .5, 'max_steps': 1.5},
                       {'threshold': .5, 'max_steps': True}):
            model = Model()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sample(model, **kwargs)
            self.assertEqual(len(model.student_ema.calls), 0)

    def test_invalid_prefixes_and_training_context_are_rejected(self):
        model = Model()
        for prefixes, lengths in ((None, self.lengths), (self.prefixes, torch.tensor([0, 4])),
                                  (self.prefixes, torch.tensor([2, 6])),
                                  (self.prefixes.float(), self.lengths),
                                  (torch.full_like(self.prefixes, 15), self.lengths)):
            with self.assertRaises(ValueError):
                generate_adaptive(model, prefixes, lengths, .9)
        model._didi_in_training_step = True
        with self.assertRaisesRegex(RuntimeError, 'evaluation-only'):
            self.sample(model, threshold=.9)

    def test_nonfinite_probabilities_and_selected_mask_raise_not_partial_output(self):
        model = Model()
        with mock.patch.object(model, 'forward', return_value=torch.full((2, 6, 16), float('nan'))):
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                self.sample(model, threshold=.9)
        proposals = torch.full((2, 6), model.mask_index)
        confidence = torch.ones(2, 6)
        with mock.patch('didi_repro.tinygsm_adaptive._propose_tokens', return_value=(proposals, confidence)):
            with self.assertRaisesRegex(RuntimeError, 'selected MASK'):
                self.sample(model, threshold=.9, temperature=1e10)
        with self.assertRaisesRegex(RuntimeError, 'nonfinite sampling probabilities'):
            self.sample(model, threshold=.9, temperature=1e-300)

    def test_eos_and_pad_tokens_do_not_stop_remaining_positions(self):
        model = Model()
        # Every proposed token is the same EOS/PAD-like zero, not a special
        # termination instruction. All four/two suffix positions still count.
        values = torch.zeros(6, 16)
        values[:, :15] = .4 / 14
        values[:, 0] = .6
        model.student_ema.logits = values.log()
        output, metadata = self.sample(model, threshold=1.)
        self.assertEqual(metadata['nfe_per_row'].tolist(), [4., 2.])
        self.assertTrue(bool((output[0, 2:] == 0).all()))

    def test_optional_tokens_only_result(self):
        result = self.sample(threshold=.9, return_metadata=False)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2, 6))

    def test_real_didi_runtime_uses_requested_weights_without_guidance(self):
        import test_didi_tinygsm_runtime as fixtures
        fixtures.TinyGSMRuntimeTests.setUpClass()
        try:
            helpers = fixtures.TinyGSMRuntimeTests()
            for weights in ('backbone', 'student_ema'):
                model = helpers.model(source='inference', sampler='guided')
                model._didi_evaluation_weights = weights
                counts = {'backbone': 0, 'student_ema': 0}
                handles = []
                def counted(name):
                    def hook(*args):
                        counts[name] += 1
                    return hook
                for name in counts:
                    handles.append(getattr(model, name).register_forward_hook(counted(name)))
                try:
                    with mock.patch.object(model.discriminator, 'forward', side_effect=AssertionError('discriminator')):
                        result, metadata = generate_adaptive(
                            model, torch.tensor([[1, 2, 3], [3, 2, 1]]), torch.tensor([2, 3]),
                            threshold=.9, max_steps=2)
                    self.assertEqual(counts[weights], metadata['nfe'])
                    self.assertEqual(counts['backbone' if weights == 'student_ema' else 'student_ema'], 0)
                    self.assertFalse(bool((result == model.mask_index).any()))
                finally:
                    for handle in handles:
                        handle.remove()
        finally:
            fixtures.TinyGSMRuntimeTests.tearDownClass()

    def test_bit_exact_reference_helpers_and_seeded_full_decode(self):
        """Execute the pinned reference source in an isolated AST namespace.

        No reference flat modules or GPUs are imported. A shallow deployment
        lacking that historical Git object skips this optional cross-branch
        equivalence test; all standalone behavioral tests still run.
        """
        from didi_repro.tinygsm import REFERENCE_REVISION
        sources = {}
        for filename in ('trainer_base.py', 'confidence_sampling.py'):
            result = subprocess.run(['git', 'show', f'{REFERENCE_REVISION}:{filename}'],
                                    cwd=ROOT, capture_output=True, text=True, check=False)
            if result.returncode:
                self.skipTest('Pinned TinyGSM reference Git object is unavailable.')
            sources[filename] = ast.parse(result.stdout)
        namespace = {'torch': torch}
        def compile_function(filename, class_name, name):
            nodes = sources[filename].body
            if class_name is not None:
                nodes = next(node.body for node in nodes
                             if isinstance(node, ast.ClassDef) and node.name == class_name)
            function = next(node for node in nodes
                            if isinstance(node, ast.FunctionDef) and node.name == name)
            module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
            exec(compile(module, '<pinned-tinygsm-reference>', 'exec'), namespace)
            compiled = namespace[name]
            # This extracted method is assigned directly to an instance;
            # unwrap the class descriptor (not callable before Python 3.10).
            return compiled.__func__ if isinstance(compiled, staticmethod) else compiled
        select = compile_function('confidence_sampling.py', None, 'select_threshold')
        namespace['confidence_sampling'] = types.SimpleNamespace(
            select_threshold=select, ADAPTIVE_STRATEGIES=('low_confidence_dynamic',))
        compile_function('trainer_base.py', None, 'sample_categorical')
        methods = {}
        for owner, names in (
                ('Diffusion', ('_adaptive_generate_samples', '_alpha_to_t')),
                ('AbsorbingState', ('prior_sample', '_ancestral_update', '_propose_tokens', '_confidence_commit'))):
            for name in names:
                methods[name] = compile_function('trainer_base.py', owner, name)
        projection = compile_function('trainer_base.py', 'TrainerBase', '_project_prefix')
        for temperature in (0., 1., .7, 1.2):
            for cap in (None, 2):
                with self.subTest(temperature=temperature, max_steps=cap):
                    reference = Model()
                    for name, method in methods.items():
                        setattr(reference, name, types.MethodType(method, reference))
                    reference._project_prefix = projection
                    reference.remasking_strategy = 'low_confidence_dynamic'
                    reference.dynamic_threshold = .75
                    reference.sampling_temperature = temperature
                    reference.config = types.SimpleNamespace(sampling={'max_steps': cap})
                    torch.manual_seed(991)
                    expected, ref_meta = reference._adaptive_generate_samples(
                        2, prefix_tokens=self.prefixes, prefix_lengths=self.lengths, return_metadata=True)
                    expected_rng = torch.get_rng_state().clone()
                    torch.manual_seed(991)
                    actual, metadata = self.sample(threshold=.75, temperature=temperature, max_steps=cap)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertEqual(metadata['nfe'], ref_meta['nfe'])
                    self.assertEqual(metadata['steps'], ref_meta['steps'])
                    torch.testing.assert_close(metadata['nfe_per_row'], ref_meta['nfe_per_row'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
