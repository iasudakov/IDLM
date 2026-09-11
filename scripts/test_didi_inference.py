"""Offline CPU checks for the pinned official DiDi inference evaluation path.

Run from the repository root: python scripts/test_didi_inference.py
The sampler tests execute untouched methods extracted by AST, using tiny CPU
fixtures instead of importing the heavyweight upstream training environment.
"""

import ast
from contextlib import nullcontext
import copy
import hashlib
import json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from didi_repro import UPSTREAM_REVISION
from didi_repro import standalone
from didi_repro.runtime import UPSTREAM_ROOT, verify_upstream
import prepare_didi_authors_checkpoint as prepare
import score_didi_authors_samples as score


INFERENCE_REVISION = 'eb9f96c78ddcf3c9d601fbde1744eba45238b3c7'


def upstream_method(name, **namespace):
    """Compile the exact method body, without modifying any source or imports."""
    path = ROOT / 'vendor' / 'didi_inference' / 'algo.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == 'DiDiInstruct')
    function = next(node for node in owner.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    environment = {'torch': torch, 'F': torch.nn.functional, **namespace}
    exec(compile(ast.Module(body=[copy.deepcopy(function)], type_ignores=[]),
                 str(path), 'exec'), environment)
    return environment[name]


def guided_config():
    from omegaconf import OmegaConf
    return OmegaConf.load(ROOT / 'vendor' / 'didi_inference' /
                          'configs' / 'algo' / 'didi_instruct.yaml')


class SourceSelectionTests(unittest.TestCase):
    def test_main_default_remains_existing_training_source(self):
        path, manifest = standalone.verify_source()
        explicit_path, explicit = standalone.verify_source('main')
        self.assertEqual(path, UPSTREAM_ROOT)
        self.assertEqual(path, explicit_path)
        self.assertEqual(manifest, explicit)
        self.assertEqual(manifest, verify_upstream())
        self.assertEqual(manifest['commit'], UPSTREAM_REVISION)

    def test_inference_manifest_is_pinned_separately_and_every_file_matches(self):
        path, manifest = standalone.verify_source('inference')
        self.assertEqual(path, ROOT / 'vendor' / 'didi_inference')
        self.assertEqual(manifest['commit'], INFERENCE_REVISION)
        self.assertEqual(manifest['branch'], 'inference')
        self.assertEqual(len(manifest['files']), 42)
        for filename, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((path / filename).read_bytes()).hexdigest(),
                             expected, filename)
        _, main = standalone.verify_source('main')
        self.assertNotEqual(manifest['files']['algo.py'], main['files']['algo.py'])

    def test_invalid_source_is_rejected(self):
        for source in ('bogus', '../didi_instruct', '', 'MAIN'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                standalone.verify_source(source)

    def test_changed_source_file_is_rejected_without_editing_vendor(self):
        original_read = Path.read_bytes
        for source in ('main', 'inference'):
            path, _ = standalone.verify_source(source)
            target = path / 'algo.py'

            def altered_read(path):
                return b'not the pinned source' if path == target else original_read(path)

            with self.subTest(source=source), mock.patch.object(
                    Path, 'read_bytes', altered_read), self.assertRaises(RuntimeError):
                standalone.verify_source(source)
        # The patches above only changed reads, never files on disk.
        standalone.verify_source('main')
        standalone.verify_source('inference')

    def test_missing_source_file_is_rejected_without_removing_it(self):
        original_is_file = Path.is_file
        path, _ = standalone.verify_source('inference')
        target = path / 'algo.py'
        with mock.patch.object(Path, 'is_file',
                               lambda path: False if path == target else original_is_file(path)):
            with self.assertRaises(RuntimeError):
                standalone.verify_source('inference')
        self.assertTrue(target.is_file())

    def test_manifest_revision_and_directory_mismatches_are_rejected(self):
        manifest_path = ROOT / 'didi_repro' / 'INFERENCE_UPSTREAM.json'
        manifest = json.loads(manifest_path.read_text())
        original_read = Path.read_text
        for key, invalid in (('commit', '0' * 40), ('vendored_path', 'vendor/didi_instruct')):
            changed = {**manifest, key: invalid}

            def altered_read(path, *args, **kwargs):
                return (json.dumps(changed) if path == manifest_path else
                        original_read(path, *args, **kwargs))

            with self.subTest(key=key), mock.patch.object(Path, 'read_text', altered_read):
                with self.assertRaisesRegex(RuntimeError, 'revision mismatch'):
                    standalone.verify_source('inference')

    def test_checkpoint_and_rescorer_cli_preserve_main_default(self):
        preparation = ['--output', '/unused/test-didi-inference.ckpt']
        rescoring = ['--samples', '/unused/samples.json', '--output', '/unused/rescore.json']
        for parser, arguments in ((prepare.parser(), preparation), (score.parser(), rescoring)):
            self.assertEqual(parser.parse_args(arguments).upstream, 'main')
            self.assertEqual(parser.parse_args(arguments + ['--upstream', 'inference']).upstream,
                             'inference')
            with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                parser.parse_args(arguments + ['--upstream', 'bogus'])

    def test_preparation_rejects_mixed_flat_imports_before_model_construction(self):
        conflicting = SimpleNamespace(__file__='/not/this/repository/main.py')
        for source in ('main', 'inference'):
            with self.subTest(source=source), mock.patch.dict(sys.modules, {'main': conflicting}), \
                    mock.patch.object(prepare.importlib, 'import_module') as importer:
                with self.assertRaisesRegex(RuntimeError, 'Conflicting main import'):
                    prepare.pristine_model(source)
                importer.assert_not_called()

    def test_rescorer_loads_metrics_from_selected_release_with_matching_provenance(self):
        for source in ('main', 'inference'):
            directory, manifest = standalone.verify_source(source)
            sentinel = object()
            module = SimpleNamespace(Metrics=sentinel)
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=mock.Mock()))
            with self.subTest(source=source), mock.patch.dict(sys.modules), \
                    mock.patch.object(score.importlib.util, 'spec_from_file_location',
                                      return_value=spec) as loader, \
                    mock.patch.object(score.importlib.util, 'module_from_spec', return_value=module):
                metric, record = score.load_pristine_metrics(source)
                self.assertIs(metric, sentinel)
                self.assertEqual(loader.call_args.args[1], directory / 'metrics.py')
                self.assertEqual(record['commit'], manifest['commit'])
                self.assertEqual(record['metrics_path'], str(directory / 'metrics.py'))
                self.assertEqual(record['metrics_sha256'], manifest['files']['metrics.py'])
                spec.loader.exec_module.assert_called_once_with(module)


class CheckpointPreflightTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            backbone=torch.nn.Linear(3, 4), student_ema=torch.nn.Linear(3, 4),
            discriminator=torch.nn.utils.spectral_norm(torch.nn.Linear(3, 1)))
        self.state = {}
        for prefix in ('backbone', 'student_ema', 'discriminator'):
            for name, value in getattr(self.model, prefix).state_dict().items():
                self.state[f'{prefix}.{name}'] = value.detach().clone()
        self.state['student_ema.weight'] = self.state['backbone.weight'] + 0.25
        self.checkpoint = {'state_dict': self.state, 'didi_authors_checkpoint': {'source': 'fixture'}}

    def _validate(self, checkpoint=None, devices=('cuda:0',)):
        payload = self.checkpoint if checkpoint is None else checkpoint

        def load(path, map_location, weights_only):
            self.assertEqual(path, '/unused/native.ckpt')
            self.assertFalse(weights_only)
            for device in devices:
                storage = object()
                self.assertIs(map_location(storage, device), storage)
            return payload

        with mock.patch.object(torch, 'load', side_effect=load):
            return prepare.validate_native_checkpoint('/unused/native.ckpt', self.model)

    def test_complete_cuda_checkpoint_preserves_distinct_raw_ema_and_spectral_buffers(self):
        before_state = copy.deepcopy(self.state)
        before_model = {prefix: copy.deepcopy(getattr(self.model, prefix).state_dict())
                        for prefix in ('backbone', 'student_ema', 'discriminator')}
        result = self._validate()
        self.assertEqual(result['saved_devices'], ['cuda:0'])
        self.assertEqual(result['complete_tensor_counts'],
                         {prefix: len(state) for prefix, state in before_model.items()})
        self.assertEqual(result['source_metadata'], {'source': 'fixture'})
        self.assertIn('discriminator.weight_u', self.state)
        self.assertIn('discriminator.weight_v', self.state)
        for name, value in before_state.items():
            torch.testing.assert_close(self.state[name], value, rtol=0, atol=0)
        for prefix, weights in before_model.items():
            for name, value in weights.items():
                torch.testing.assert_close(getattr(self.model, prefix).state_dict()[name],
                                           value, rtol=0, atol=0)
        self.assertFalse(torch.equal(self.state['backbone.weight'], self.state['student_ema.weight']))

    def test_missing_raw_ema_discriminator_or_spectral_state_fails(self):
        for prefix in ('backbone', 'student_ema', 'discriminator'):
            state = {key: value for key, value in self.state.items()
                     if not key.startswith(prefix + '.')}
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, prefix):
                self._validate({'state_dict': state})
        for name in ('discriminator.weight_u', 'discriminator.weight_v'):
            state = {key: value for key, value in self.state.items() if key != name}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'discriminator'):
                self._validate({'state_dict': state})

    def test_wrong_shape_and_unexpected_learned_keys_fail(self):
        for change in ({'student_ema.weight': torch.zeros(1)}, {'backbone.extra': torch.zeros(1)}):
            with self.subTest(change=list(change)), self.assertRaisesRegex(ValueError, 'Incompatible'):
                self._validate({'state_dict': {**self.state, **change}})

    def test_cpu_other_cuda_mixed_and_missing_storage_tags_fail(self):
        for devices in ((), ('cpu',), ('cuda:1',), ('cuda:0', 'cpu')):
            with self.subTest(devices=devices), self.assertRaisesRegex(ValueError, 'logical cuda:0'):
                self._validate(devices=devices)

    def test_non_lightning_or_non_mapping_state_fails(self):
        for checkpoint in ({}, [], {'state_dict': None}, {'state_dict': []}):
            with self.subTest(checkpoint=checkpoint), self.assertRaisesRegex(ValueError, 'state_dict'):
                self._validate(checkpoint)


class InferenceProtocolTests(unittest.TestCase):
    def test_released_config_matches_phase_protocol_and_differs_from_main(self):
        from omegaconf import OmegaConf
        inference = guided_config()
        main = OmegaConf.load(UPSTREAM_ROOT / 'configs' / 'algo' / 'didi_instruct.yaml')
        self.assertEqual(inference.guidance_scale_start, 30.0)
        self.assertEqual(inference.guidance_scale_end, 40.0)
        self.assertEqual(inference.rerank_steps_ratio, 0.5)
        self.assertEqual(inference.num_candidates, 4)
        self.assertEqual(main.guidance_scale_start, 0.2)
        self.assertEqual(main.guidance_scale_end, 1.0)

    def test_unmodified_generate_samples_uses_30_to_40_first_half_then_h0_m4(self):
        configuration = guided_config()
        generate = upstream_method('generate_samples')
        for steps in (8, 16, 32):
            with self.subTest(steps=steps):
                calls = []
                backbone = mock.Mock()
                model = SimpleNamespace(
                    config=SimpleNamespace(mode='cpu_test', sampling=SimpleNamespace(
                        steps=steps, noise_removal='ancestral')),
                    student_ema=mock.Mock(), discriminator=mock.Mock(), backbone=backbone,
                    num_tokens=5, device=torch.device('cpu'), sampler='guided',
                    guidance_scale_start=configuration.guidance_scale_start,
                    guidance_scale_end=configuration.guidance_scale_end,
                    rerank_steps_ratio=configuration.rerank_steps_ratio,
                    num_candidates=configuration.num_candidates,
                    prior_sample=lambda batch, length: torch.zeros(batch, length, dtype=torch.long),
                    _denoiser_update=mock.Mock(side_effect=lambda x, t: x + 7))

                def step(x, t, dt, h, M):
                    calls.append((h, M, t.clone(), dt))
                    return x + 1

                model.reward_guided_ancestral_step = step
                result = generate(model, num_samples=2, num_steps=steps)
                self.assertEqual(len(calls), steps)
                half = steps // 2
                for index, (h, candidates, t, dt) in enumerate(calls):
                    self.assertEqual(candidates, 1 if index < half else 4)
                    expected = 30 + 10 * index / (half - 1) if index < half else 0
                    self.assertAlmostEqual(h, expected)
                    self.assertEqual(tuple(t.shape), (2, 1))
                    self.assertAlmostEqual(dt, (1 - 1e-5) / steps)
                model.student_ema.eval.assert_called_once()
                model.discriminator.eval.assert_called_once()
                model._denoiser_update.assert_called_once()
                backbone.assert_not_called()
                torch.testing.assert_close(result, torch.full((2, 5), steps + 7))

    def _step_model(self):
        logits = torch.tensor([[[1., 4., -2.], [3., 1., -2.]]])
        model = SimpleNamespace(
            student_ema=mock.Mock(return_value=logits),
            backbone=mock.Mock(side_effect=AssertionError('raw backbone must not be sampled')),
            discriminator=mock.Mock(), mask_index=2, vocab_size=3,
            noise=lambda t: (None, 1 - 0.5 * t), _sigma_from_alphat=lambda alpha: alpha)
        return model, logits

    def test_unmodified_guided_step_reads_student_ema_and_preserves_unmasked_tokens(self):
        draw = mock.Mock(side_effect=lambda probabilities: probabilities.argmax(-1))
        function = upstream_method('reward_guided_ancestral_step',
                                   trainer_base=SimpleNamespace(sample_categorical=draw))
        model, _ = self._step_model()
        result = function(model, torch.tensor([[2, 0]]), torch.tensor([[0.5]]),
                          dt=0.25, h=0, M=1)
        model.student_ema.assert_called_once()
        model.backbone.assert_not_called()
        model.discriminator.assert_not_called()
        self.assertEqual(draw.call_count, 1)
        self.assertEqual(result[0, 1].item(), 0)
        self.assertEqual(result.dtype, torch.long)

    def test_unmodified_reranking_samples_softmax_reward_not_argmax_or_h_scaled(self):
        recorded = []
        candidates = torch.tensor([[[0, 1], [1, 0], [1, 1], [0, 0]]])

        def draw(probabilities):
            recorded.append(probabilities.detach().clone())
            if probabilities.ndim == 4:
                return candidates.clone()
            # Deliberately choose a candidate other than the maximum-reward one.
            return torch.tensor([1])

        function = upstream_method('reward_guided_ancestral_step',
                                   trainer_base=SimpleNamespace(sample_categorical=draw))
        model, _ = self._step_model()
        model.discriminator.side_effect = lambda x, sigma: x.float().unsqueeze(-1)
        with mock.patch.object(torch.amp, 'autocast', return_value=nullcontext()):
            result = function(model, torch.tensor([[2, 2]]), torch.tensor([[0.5]]),
                              dt=0.25, h=0, M=4)
        model.student_ema.assert_called_once()
        model.backbone.assert_not_called()
        model.discriminator.assert_called_once()
        self.assertEqual(len(recorded), 2)
        torch.testing.assert_close(recorded[1], torch.softmax(torch.tensor([[1., 1., 2., 0.]]), -1))
        torch.testing.assert_close(result, candidates[:, 1])


class LauncherTests(unittest.TestCase):
    def test_launchers_are_valid_shell(self):
        for name in ('eval_didi_authors.sh', 'eval_didi_inference.sh'):
            subprocess.run(['bash', '-n', str(ROOT / 'scripts' / name)], check=True,
                           capture_output=True, text=True)

    def test_inference_wrapper_forces_source_and_forwards_cli_arguments(self):
        with tempfile.TemporaryDirectory(prefix='didi-inference-shell-test-') as folder:
            directory = Path(folder) / 'scripts'
            directory.mkdir()
            wrapper = directory / 'eval_didi_inference.sh'
            shutil.copyfile(ROOT / 'scripts' / wrapper.name, wrapper)
            delegate = directory / 'eval_didi_authors.sh'
            delegate.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$DIDI_AUTHORS_SOURCE" "$@"\n')
            environment = {**os.environ, 'DIDI_AUTHORS_SOURCE': 'main'}
            result = subprocess.run(['bash', str(wrapper), 'sampling.steps=8', 'seed=17'],
                                    env=environment, check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.splitlines(), ['inference', 'sampling.steps=8', 'seed=17'])


if __name__ == '__main__':
    unittest.main()
