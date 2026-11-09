"""Offline integration checks for the opt-in SDTT feature source.

Run: OMP_NUM_THREADS=2 python -B scripts/test_sdtt_feature_routing.py

The actual routing method is extracted with AST so Lightning, CUDA, Hub
weights, and network access are unnecessary. Legacy numerical checks use the
immutable historical baseline (override SDTT_ROUTING_BASELINE if needed).
"""
import ast
from contextlib import contextmanager, redirect_stdout
from functools import lru_cache
import io
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mmd_reward

BASELINE = os.environ.get('SDTT_ROUTING_BASELINE', '634176b')


@lru_cache(maxsize=None)
def historical_source(path):
    return subprocess.run(
        ['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class TinyDenoiser(torch.nn.Module):
    """A deterministic native backbone with real feature hooks."""
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
        self.blocks = torch.nn.ModuleList([
            torch.nn.Identity(), torch.nn.Identity()])

    def forward(self, tokens, sigma):
        values = tokens.float() + sigma.reshape(-1, 1)
        hidden = torch.stack((values.sin(), values.cos(), values / 7.), -1)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden * self.scale * (index + 1))
        return hidden


def routing_class(source):
    tree = ast.parse(source, filename='algo.py')
    mdlm = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'MDLM')
    method = next(node for node in mdlm.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == '_mmd_critic')
    probe = ast.parse('class RoutingProbe(torch.nn.Module):\n    pass').body[0]
    probe.body = [method]
    namespace = {'torch': torch, 'IdlmRatioReward': mock.Mock(
        side_effect=AssertionError('feature source constructed an IDLM critic'))}
    module = ast.fix_missing_locations(ast.Module(
        body=[probe], type_ignores=[]))
    exec(compile(module, 'algo.py', 'exec'), namespace)
    return namespace['RoutingProbe']


def make_model(*, source=None, **overrides):
    settings = dict(
        mmd_reward_type='feature', mmd_feature_source='diffusion',
        mmd_ar_model='unused/ar', mmd_feature_layer=[0, 1],
        mmd_feature_layer_weights=[.25, .75], mmd_noise_level=.15,
        mmd_kernel='rbf', mmd_kernel_level='token',
        mmd_kernel_source='draws', mmd_kernel_batch_size=2,
        mmd_group_size=3, mmd_reward_weight=.7,
        mmd_rbf_alpha=.2, mmd_rbf_exclude_same_sentence=False,
        mmd_rbf_exclude_same_position=False,
        mmd_sdtt_model='jdeschena/sdtt',
        mmd_sdtt_revision='teacher_400k_large_step_400000',
        mmd_sdtt_path=None, mmd_sdtt_local_files_only=False)
    settings.update(overrides)
    cls = routing_class(source if source is not None
                        else (ROOT / 'algo.py').read_text())
    model = cls()
    model.config = SimpleNamespace(
        adversarial_distill=Config(**settings),
        model_weights_dir='/unused/offline-bundle')
    model.device = torch.device('cpu')
    model.tokenizer = SimpleNamespace(
        eos_token_id=0, pad_token_id=7, vocab_size=8)
    model.mask_index = 8
    model.backbone = TinyDenoiser(1.3)
    model.teacher = TinyDenoiser().requires_grad_(False).eval()
    return model


@contextmanager
def reject_sdtt_import():
    original = __import__

    def guarded(name, *args, **kwargs):
        if name == 'mmd_sdtt_features' or name.startswith('mmd_sdtt_features.'):
            raise AssertionError('legacy route imported the SDTT loader')
        return original(name, *args, **kwargs)

    with mock.patch('builtins.__import__', side_effect=guarded):
        yield


def construct(model):
    with redirect_stdout(io.StringIO()):
        return model._mmd_critic


class SdttFeatureRoutingTests(unittest.TestCase):
    def test_sdtt_dependency_is_not_imported_at_algo_module_load(self):
        tree = ast.parse((ROOT / 'algo.py').read_text())
        parents = {child: parent for parent in ast.walk(tree)
                   for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            imports_sdtt = (
                isinstance(node, ast.ImportFrom)
                and (node.module or '').startswith('mmd_sdtt_features')) or (
                isinstance(node, ast.Import)
                and any(alias.name.startswith('mmd_sdtt_features')
                        for alias in node.names))
            if imports_sdtt:
                ancestors, current = [], node
                while current in parents:
                    current = parents[current]
                    ancestors.append(current)
                self.assertTrue(any(isinstance(parent, ast.FunctionDef)
                                    for parent in ancestors),
                                'SDTT dependency must be imported lazily')

    def test_all_existing_adversarial_defaults_remain_unchanged(self):
        old = OmegaConf.to_container(OmegaConf.create(
            historical_source('configs/config.yaml')).adversarial_distill,
            resolve=False)
        new = OmegaConf.to_container(OmegaConf.load(
            ROOT / 'configs/config.yaml').adversarial_distill, resolve=False)
        additions = {name: value for name, value in new.items()
                     if name.startswith(('mmd_sdtt_', 'mmd_sdllm_'))}
        self.assertEqual(additions, {
            'mmd_sdtt_model': 'jdeschena/sdtt',
            'mmd_sdtt_revision': 'teacher_400k_large_step_400000',
            'mmd_sdtt_path': None, 'mmd_sdtt_local_files_only': False,
            'mmd_sdllm_model': 'nvidia/SDLLM-MDLM-1.7B-Base',
            'mmd_sdllm_revision': '4007e0b2a23465a31adf282d843a61172932af2a',
            'mmd_sdllm_path': None, 'mmd_sdllm_local_files_only': False})
        self.assertEqual({key: value for key, value in new.items()
                          if key not in additions}, old)
        self.assertEqual(new['mmd_feature_source'], old['mmd_feature_source'])

    def test_diffusion_uses_identical_teacher_without_sdtt_import_and_caches(self):
        model = make_model()
        teacher, student = model.teacher, model.backbone
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with reject_sdtt_import(), mock.patch.object(
                mmd_reward, 'load_owt_ar_backbone', side_effect=AssertionError(
                    'diffusion route loaded another model')):
            critic = construct(model)
            self.assertIs(critic, construct(model))
        self.assertIs(critic.denoiser, teacher)
        self.assertIs(model.backbone, student)
        self.assertIs(model.teacher, teacher)
        self.assertEqual(set(model.state_dict()), set(before))
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value,
                                       atol=0, rtol=0)

    def test_ar_route_does_not_import_sdtt(self):
        model = make_model(mmd_feature_source='ar')
        loaded = TinyDenoiser(2.).requires_grad_(False).eval()
        with reject_sdtt_import(), mock.patch.object(
                mmd_reward, 'load_owt_ar_backbone', return_value=loaded) as load:
            critic = construct(model)
        self.assertIs(critic.denoiser, loaded)
        load.assert_called_once_with(
            'unused/ar', model.device, weights_dir='/unused/offline-bundle')
        self.assertEqual(critic.noise_level, 0.)

    def test_sdtt_loads_once_and_keeps_separate_frozen_feature_backbone(self):
        for local_path, offline in ((None, False), ('/offline/sdtt', True)):
            with self.subTest(local_path=local_path, offline=offline):
                model = make_model(mmd_feature_source='sdtt_mdlm',
                                   mmd_sdtt_path=local_path,
                                   mmd_sdtt_local_files_only=offline)
                teacher, student = model.teacher, model.backbone
                before = {key: value.clone()
                          for key, value in model.state_dict().items()}
                feature_model = TinyDenoiser(2.).requires_grad_(False).eval()
                loader = mock.Mock(return_value=feature_model)
                stub = ModuleType('mmd_sdtt_features')
                stub.load_sdtt_feature_backbone = loader
                with mock.patch.dict(sys.modules, {'mmd_sdtt_features': stub}):
                    critic = construct(model)
                    self.assertIs(critic, construct(model))
                loader.assert_called_once_with(
                    'jdeschena/sdtt', model.device, tokenizer=model.tokenizer,
                    mask_index=model.mask_index,
                    feature_layer=[0, 1],
                    revision='teacher_400k_large_step_400000',
                    local_path=local_path, local_files_only=offline)
                self.assertIs(critic.denoiser, feature_model)
                self.assertIsNot(feature_model, teacher)
                self.assertIsNot(feature_model, student)
                self.assertFalse(feature_model.training)
                self.assertTrue(all(not parameter.requires_grad
                                    for parameter in feature_model.parameters()))
                self.assertIs(model.teacher, teacher)
                self.assertIs(model.backbone, student)
                self.assertEqual(critic.resolved_layers, [0, 1])
                self.assertEqual(critic.layer_weights, [.25, .75])
                self.assertEqual(critic.noise_level, .15)
                self.assertEqual(critic.kernel, 'rbf')
                self.assertEqual(critic.rbf_alpha, .2)
                self.assertEqual(critic.kernel_batch_size, 2)
                self.assertEqual(set(model.state_dict()), set(before))
                for key, value in before.items():
                    torch.testing.assert_close(model.state_dict()[key], value,
                                               atol=0, rtol=0)

    def test_invalid_source_lists_sdtt_option_without_loading_it(self):
        model = make_model(mmd_feature_source='unsupported-source')
        with reject_sdtt_import(), self.assertRaisesRegex(
                ValueError, r'ar\|diffusion\|sdtt_mdlm'):
            construct(model)

    def numerical_result(self, source, *, kernel_source, noise):
        torch.manual_seed(819)
        model = make_model(source=source, mmd_kernel_source=kernel_source,
                           mmd_noise_level=noise)
        initial_rng = torch.get_rng_state().clone()
        with reject_sdtt_import():
            critic = construct(model)
        # Construction of the unchanged frozen-teacher route consumes no RNG.
        torch.testing.assert_close(torch.get_rng_state(), initial_rng,
                                   atol=0, rtol=0)
        clean = torch.arange(4 * 5).reshape(4, 5) % 7
        selected = torch.tensor([
            [1, 1, 0, 1, 1], [1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        logits = torch.linspace(-1.7, 1.3, 4 * 5 * 8).reshape(
            4, 5, 8).requires_grad_()
        draws = []
        multinomial = torch.multinomial

        def record_draws(*args, **kwargs):
            result = multinomial(*args, **kwargs)
            draws.append(result.clone())
            return result

        with mock.patch.object(torch, 'multinomial', side_effect=record_draws):
            loss, stats = mmd_reward.mmd_reinforce_loss(
                logits.softmax(-1), clean, selected, critic,
                group_size=3, eos_id=0, reward_weight=.7,
                baseline='group', kernel_source=kernel_source, kernel_draws=2)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(logits.grad.abs().sum() > 0))
        self.assertTrue(all(not block._forward_hooks
                            for block in model.teacher.blocks))
        return (loss.detach(), stats, logits.grad.clone(), draws,
                torch.get_rng_state().clone())

    def test_legacy_samples_rewards_loss_gradients_and_rng_match_baseline_exactly(self):
        current = (ROOT / 'algo.py').read_text()
        historical = historical_source('algo.py')
        for kernel_source in ('rows', 'draws'):
            for noise in (0., .25):
                with self.subTest(kernel_source=kernel_source, noise=noise):
                    expected = self.numerical_result(
                        historical, kernel_source=kernel_source, noise=noise)
                    actual = self.numerical_result(
                        current, kernel_source=kernel_source, noise=noise)
                    for index in (0, 2, 4):
                        torch.testing.assert_close(actual[index], expected[index],
                                                   atol=0, rtol=0)
                    self.assertEqual(actual[1].keys(), expected[1].keys())
                    for key in actual[1]:
                        torch.testing.assert_close(actual[1][key], expected[1][key],
                                                   atol=0, rtol=0)
                    self.assertEqual(len(actual[3]), len(expected[3]))
                    for actual_draw, expected_draw in zip(actual[3], expected[3]):
                        torch.testing.assert_close(actual_draw, expected_draw,
                                                   atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
