"""Offline CPU and shell checks for original-precision guidance controls.

Run: CUDA_VISIBLE_DEVICES='' python scripts/test_didi_guidance_control.py
The sampler, generator loop, and final denoising methods are extracted unchanged
from the separately pinned inference source; neural models are tiny CPU fakes.
"""

import ast
from contextlib import nullcontext
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from didi_repro.standalone import verify_source


class GuidanceLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='didi-guidance-shell-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / 'scripts'
        self.directory.mkdir()
        self.wrapper = self.directory / 'eval_didi_guidance_control.sh'
        shutil.copyfile(ROOT / 'scripts' / self.wrapper.name, self.wrapper)
        (self.directory / 'eval_didi_authors.sh').write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$DIDI_AUTHORS_SOURCE" '
            '"$DIDI_INFERENCE_PRECISION" "$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" "$@"\n')

    def launch(self, mode=None, arguments=(), precision=None, **extra):
        environment = dict(os.environ)
        for key in ('DIDI_GUIDANCE_CONTROL', 'DIDI_INFERENCE_PRECISION',
                    'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'):
            environment.pop(key, None)
        if mode is not None:
            environment['DIDI_GUIDANCE_CONTROL'] = mode
        if precision is not None:
            environment['DIDI_INFERENCE_PRECISION'] = precision
        environment.update(extra)
        return subprocess.run(['bash', str(self.wrapper), *arguments], env=environment,
                              capture_output=True, text=True)

    def test_shell_syntax(self):
        subprocess.run(['bash', '-n', str(self.wrapper)], check=True, capture_output=True, text=True)

    def test_on_and_off_select_exact_protocol_and_preserve_allowed_arguments(self):
        for mode, start, end, candidates in (('on', '30.0', '40.0', '4'),
                                              ('off', '0.0', '0.0', '1')):
            arguments = ('sampling.steps=8', 'seed=3', 'eval.perplexity_batch_size=4')
            result = self.launch(mode, arguments, DIDI_AUTHORS_SOURCE='main')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [
                'inference', 'upstream', '1',
                f'algo.guidance_scale_start={start}', f'algo.guidance_scale_end={end}',
                f'algo.num_candidates={candidates}', 'algo.rerank_steps_ratio=0.5', *arguments])

    def test_control_choice_is_required_and_case_sensitive(self):
        for choice in (None, '', 'unknown', 'ON', 'OFF'):
            result = self.launch(choice)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Set DIDI_GUIDANCE_CONTROL to on or off', result.stderr)
            self.assertEqual(result.stdout, '')

    def test_precision_changes_are_rejected_before_delegation(self):
        for mode in ('on', 'off'):
            for selection in ('float64', 'bogus', 'FLOAT64'):
                result = self.launch(mode, precision=selection)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('require original precision', result.stderr)
                self.assertEqual(result.stdout, '')

    def test_no_unapproved_hydra_override_can_change_the_control(self):
        for argument in ('sampling.predictor=ancestral', 'algo.guidance_scale_start=5',
                         'algo.num_candidates=7', '+algo.rerank_steps_ratio=0',
                         'model=medium', 'loader.eval_batch_size=1', 'trainer.devices=8',
                         'eval.checkpoint_path=/unused/checkpoint.ckpt', 'mode=train',
                         '++sampling.steps=8', '~sampling.predictor'):
            result = self.launch('off', (argument,))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Unsupported guidance-control override', result.stderr)
            self.assertEqual(result.stdout, '')

    def test_no_args_leave_sampling_batching_and_seed_defaults_to_base(self):
        result = self.launch('off', precision='upstream', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[:3], ['inference', 'upstream', '0'])
        self.assertEqual(len(lines[3:]), 4)
        self.assertTrue(all(line.startswith('algo.') for line in lines[3:]))


def original_methods(draw):
    """Compile original AST nodes, never applying the runtime/precision adapter."""
    directory, _ = verify_source('inference')
    selections = {
        'algo.py': {'DiDiInstruct': ('generate_samples', 'reward_guided_ancestral_step'),
                    'MDLM': ('_process_model_output', '_get_score')},
        'trainer_base.py': {'TrainerBase': ('forward',), 'Diffusion': ('_process_sigma',),
                            'AbsorbingState': ('_denoiser_update', '_staggered_score',
                                               '_transp_transition')},
    }
    namespace = {'torch': torch, 'F': F, 'MDLM': object,
                 'trainer_base': SimpleNamespace(sample_categorical=draw)}
    methods = {}
    for filename, owners in selections.items():
        path = directory / filename
        tree = ast.parse(path.read_text(), filename=str(path))
        nodes = [copy.deepcopy(node) for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in ('_unsqueeze', 'sample_categorical')]
        for owner in tree.body:
            if isinstance(owner, ast.ClassDef) and owner.name in owners:
                nodes.extend(copy.deepcopy(node) for node in owner.body
                             if isinstance(node, ast.FunctionDef) and node.name in owners[owner.name])
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        methods.update({node.name: namespace[node.name] for node in nodes})
    return methods


class RecordingNetwork(torch.nn.Module):
    def __init__(self, values, dtype):
        super().__init__()
        self.register_buffer('logits', torch.tensor(values, dtype=dtype))
        self.calls = []

    def forward(self, x, sigma):
        self.calls.append((x.clone(), sigma.clone()))
        return self.logits.expand(*x.shape, -1).clone()


class ForbiddenDiscriminator(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError('Guidance-off must never run the discriminator')


class OriginalGuidanceProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def model(self, methods):
        initial = torch.tensor([[0, 3, 3, 1], [3, 2, 3, 3]])
        model = SimpleNamespace(
            config=SimpleNamespace(mode='cpu_test', sampling=SimpleNamespace(
                steps=8, noise_removal='ancestral', use_float64=True)),
            mask_index=3, vocab_size=4, num_tokens=4, device=torch.device('cpu'),
            neg_infinity=-1e6, time_conditioning=False, sampler='guided',
            guidance_scale_start=0., guidance_scale_end=0., num_candidates=1,
            rerank_steps_ratio=.5,
            student_ema=RecordingNetwork([.2, 1.1, -.5, .3], torch.bfloat16),
            backbone=RecordingNetwork([3., -1., .5, 0.], torch.float32),
            discriminator=ForbiddenDiscriminator(),
            prior_sample=lambda batch, length: initial.clone(),
            noise=lambda t: (torch.full_like(t, -.999), 1 - .999 * t),
            _sigma_from_alphat=lambda alpha: -alpha.log())
        for name, function in methods.items():
            if name.startswith('_') or name == 'forward':
                setattr(model, name, MethodType(function, model))
        return model, initial

    def test_off_executes_all_eight_original_steps_without_discriminator_and_keeps_cleanup(self):
        probabilities, schedule, cleanup = [], [], []

        def draw(q):
            probabilities.append(q.clone())
            # Retain masks until cleanup so its raw-backbone behavior is observable.
            return torch.full(q.shape[:-1], 3, dtype=torch.long)

        methods = original_methods(draw)
        model, initial = self.model(methods)

        def step(x, t, dt, h, M):
            schedule.append((h, M, x.clone(), t.clone(), dt))
            return methods['reward_guided_ancestral_step'](model, x, t, dt, h, M)

        original_cleanup = model._denoiser_update

        def finish(x, t):
            self.assertEqual(len(model.backbone.calls), 0)
            cleanup.append((x.clone(), t.clone()))
            return original_cleanup(x, t)

        model.reward_guided_ancestral_step = step
        model._denoiser_update = finish
        with mock.patch.object(torch.amp, 'autocast', return_value=nullcontext()), \
                mock.patch.object(torch.Tensor, 'backward', side_effect=AssertionError('backward called')):
            torch.manual_seed(13)
            result = methods['generate_samples'](model, 2, num_steps=8)
        self.assertEqual(len(schedule), 8)
        self.assertEqual(len(probabilities), 8)
        self.assertEqual(len(model.student_ema.calls), 8)
        grid = torch.linspace(1, 1e-5, 9)
        for index, ((h, candidates, x, t, dt), q, (ema_x, ema_sigma)) in enumerate(
                zip(schedule, probabilities, model.student_ema.calls)):
            self.assertEqual((h, candidates), (0, 1))
            self.assertEqual(q.shape, (2, 4, 4))
            self.assertEqual(q.dtype, torch.float32)  # Original bf16 softmax, fp32 alpha arithmetic.
            self.assertEqual(t.dtype, torch.float32)
            torch.testing.assert_close(x, initial, rtol=0, atol=0)
            torch.testing.assert_close(ema_x, initial, rtol=0, atol=0)
            torch.testing.assert_close(t, grid[index].expand(2, 1), rtol=0, atol=0)
            torch.testing.assert_close(ema_sigma, -(1 - .999 * t).log().squeeze(-1), rtol=0, atol=0)
            self.assertTrue((ema_sigma > 0).all())
            self.assertAlmostEqual(dt, (1 - 1e-5) / 8)
        self.assertEqual(len(cleanup), 1)
        torch.testing.assert_close(cleanup[0][0], initial, rtol=0, atol=0)
        torch.testing.assert_close(cleanup[0][1], grid[-1].expand(2, 1), rtol=0, atol=0)
        self.assertEqual(len(model.backbone.calls), 1)
        torch.testing.assert_close(model.backbone.calls[0][0], initial, rtol=0, atol=0)
        torch.testing.assert_close(model.backbone.calls[0][1], torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(result[initial != 3], initial[initial != 3], rtol=0, atol=0)
        self.assertFalse((result == 3).any())
        self.assertTrue(model.backbone.training)
        self.assertTrue(model.discriminator.training)
        self.assertFalse(model.student_ema.training)

    def test_on_schedule_remains_gradient_half_then_reranking_half(self):
        methods = original_methods(lambda q: q.argmax(-1))
        model, _ = self.model(methods)
        model.guidance_scale_start, model.guidance_scale_end, model.num_candidates = 30., 40., 4
        schedule = []

        def step(x, t, dt, h, M):
            schedule.append((h, M))
            return x

        model.reward_guided_ancestral_step = step
        model._denoiser_update = lambda x, t: x
        methods['generate_samples'](model, 2, num_steps=8)
        expected = [(30 + 10 * i / 3, 1) for i in range(4)] + [(0, 4)] * 4
        for (actual_h, actual_m), (expected_h, expected_m) in zip(schedule, expected):
            self.assertAlmostEqual(actual_h, expected_h)
            self.assertEqual(actual_m, expected_m)
        self.assertEqual(len(schedule), 8)


if __name__ == '__main__':
    unittest.main()
