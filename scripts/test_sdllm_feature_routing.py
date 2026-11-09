"""Offline AST integration checks for the SDLLM native-token feature source.

The actual MDLM route and startup hook are exercised with tiny frozen models
and stub external asset loaders. No Lightning construction, network or GPU is
required. Numerical kernel/retokenization tests live in the dedicated suites.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import mmd_reward
import test_sdtt_feature_routing as legacy_routing
import test_sdtt_startup as legacy_startup


MODEL = 'nvidia/SDLLM-MDLM-1.7B-Base'
REVISION = '4007e0b2a23465a31adf282d843a61172932af2a'


def make_model(**overrides):
    settings = dict(
        mmd_feature_source='sdllm_mdlm', mmd_reward_type='feature',
        mmd_kernel='rbf', mmd_kernel_level='token', mmd_kernel_source='draws',
        mmd_kernel_batch_size=2, mmd_score_positions='all', mmd_noise_level=0.,
        mmd_sentence_pooling='mean', mmd_rbf_exclude_same_sentence=True,
        mmd_rbf_exclude_same_position=True,
        mmd_sdllm_model=MODEL, mmd_sdllm_revision=REVISION,
        mmd_sdllm_path=None, mmd_sdllm_local_files_only=False)
    settings.update(overrides)
    model = legacy_routing.make_model(**settings)
    model.fake_model = legacy_routing.TinyDenoiser(1.1)
    return model


@contextmanager
def external_stubs():
    feature_model = legacy_routing.TinyDenoiser(2.7).requires_grad_(False).eval()
    feature_loader = mock.Mock(return_value=feature_model)
    critic = SimpleNamespace(layer_description='0 + 1', resolved_layers=[0, 1],
                             layer_weights=[.25, .75])
    critic_constructor = mock.Mock(return_value=critic)
    feature_module = ModuleType('mmd_sdllm_features')
    feature_module.load_sdllm_feature_model = feature_loader
    reward_module = ModuleType('mmd_sdllm_reward')
    reward_module.FrozenSdllmFeatures = critic_constructor
    with mock.patch.dict(sys.modules, {'mmd_sdllm_features': feature_module,
                                      'mmd_sdllm_reward': reward_module}):
        yield feature_model, feature_loader, critic, critic_constructor


@contextmanager
def reject_sdllm_import():
    original = __import__
    def guarded(name, *args, **kwargs):
        if name == 'mmd_sdllm_features' or name.startswith('mmd_sdllm_features.'):
            raise AssertionError('legacy/invalid source imported SDLLM loader')
        if name == 'mmd_sdllm_reward' or name.startswith('mmd_sdllm_reward.'):
            raise AssertionError('legacy/invalid source imported SDLLM reward')
        return original(name, *args, **kwargs)
    with mock.patch('builtins.__import__', side_effect=guarded):
        yield


class SDLLMFeatureRoutingTests(unittest.TestCase):
    def test_sdllm_dependencies_are_imported_only_inside_methods(self):
        tree = ast.parse((ROOT / 'algo.py').read_text())
        parents = {child: parent for parent in ast.walk(tree)
                   for child in ast.iter_child_nodes(parent)}
        count = 0
        for node in ast.walk(tree):
            source = ''
            if isinstance(node, ast.ImportFrom):
                source = node.module or ''
            elif isinstance(node, ast.Import):
                source = ','.join(alias.name for alias in node.names)
            if 'mmd_sdllm_' not in source:
                continue
            count += 1
            current, ancestors = node, []
            while current in parents:
                current = parents[current]
                ancestors.append(current)
            self.assertTrue(any(isinstance(parent, ast.FunctionDef) for parent in ancestors))
        self.assertEqual(count, 2)

    def test_sdllm_route_loads_once_and_threads_all_asset_and_layer_arguments(self):
        for path, offline in ((None, False), ('/explicit/local release', True)):
            with self.subTest(path=path, offline=offline):
                model = make_model(mmd_sdllm_path=path, mmd_sdllm_local_files_only=offline)
                with external_stubs() as (feature_model, loader, critic, constructor):
                    actual = legacy_routing.construct(model)
                    self.assertIs(actual, legacy_routing.construct(model))
                self.assertIs(actual, critic)
                loader.assert_called_once_with(
                    MODEL, model.device, tokenizer=model.tokenizer,
                    mask_index=model.mask_index, revision=REVISION,
                    local_path=path, local_files_only=offline, feature_layer=[0, 1])
                constructor.assert_called_once_with(
                    feature_model, model.tokenizer, model.mask_index,
                    layer_index=[0, 1], layer_weights=[.25, .75], noise_level=0.)
                self.assertFalse(feature_model.training)
                self.assertTrue(all(not p.requires_grad for p in feature_model.parameters()))

    def test_common_reward_settings_reach_native_critic_unchanged(self):
        model = make_model(mmd_rbf_alpha=.012, mmd_rbf_distance='pseudo_huber',
                           mmd_rbf_huber_delta=.03, mmd_rbf_block_size=7,
                           mmd_rbf_omit_real_real=True, mmd_rbf_omit_fake_fake=True,
                           mmd_kernel_batch_size=4)
        with external_stubs() as (_, _, critic, _):
            actual = legacy_routing.construct(model)
        self.assertIs(actual, critic)
        expected = dict(
            reward_type='feature', kernel='rbf', kernel_level='token',
            sentence_pooling='mean', rbf_alpha=.012,
            rbf_distance='pseudo_huber', rbf_huber_delta=.03,
            rbf_exclude_same_sentence=True, rbf_exclude_same_position=True,
            rbf_omit_real_real=True, rbf_omit_fake_fake=True,
            rbf_block_size=7, kernel_batch_size=4)
        for name, value in expected.items():
            self.assertEqual(getattr(actual, name), value, name)

    def test_feature_initialization_does_not_register_or_modify_student_teacher_or_optimizer(self):
        model = make_model()
        student, teacher, fake = model.backbone, model.teacher, model.fake_model
        optimizer = torch.optim.SGD(model.backbone.parameters(), lr=.1)
        optimizer_parameters = [id(p) for group in optimizer.param_groups for p in group['params']]
        model_parameters = [id(p) for p in model.parameters()]
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with external_stubs() as (_, loader, _, _):
            legacy_routing.construct(model)
        self.assertEqual(loader.call_count, 1)
        self.assertIs(model.backbone, student)
        self.assertIs(model.teacher, teacher)
        self.assertIs(model.fake_model, fake)
        self.assertEqual([id(p) for p in model.parameters()], model_parameters)
        self.assertEqual([id(p) for group in optimizer.param_groups for p in group['params']], optimizer_parameters)
        self.assertEqual(set(before), set(model.state_dict()))
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_unsupported_native_modes_fail_before_sdllm_import_or_download(self):
        cases = [
            {'mmd_reward_type': 'feature_mse'}, {'mmd_kernel': 'linear'},
            {'mmd_kernel_level': 'sentence'}, {'mmd_kernel_source': 'rows'},
            {'mmd_score_positions': 'written'}, {'mmd_noise_level': .1},
            {'mmd_sentence_pooling': 'last_token'},
            {'mmd_kernel_batch_size': 1}, {'mmd_rbf_distance': 'unknown'},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                model = make_model(**changes)
                with reject_sdllm_import(), self.assertRaises(ValueError):
                    legacy_routing.construct(model)

    def test_existing_diffusion_ar_and_sdtt_routes_never_import_sdllm(self):
        for source in ('diffusion', 'ar', 'sdtt_mdlm'):
            with self.subTest(source=source):
                model = legacy_routing.make_model(mmd_feature_source=source)
                sdtt_backbone = legacy_routing.TinyDenoiser(2.).requires_grad_(False).eval()
                sdtt_module = ModuleType('mmd_sdtt_features')
                sdtt_module.load_sdtt_feature_backbone = mock.Mock(return_value=sdtt_backbone)
                ar_backbone = legacy_routing.TinyDenoiser(2.1).requires_grad_(False).eval()
                with mock.patch.dict(sys.modules, {'mmd_sdtt_features': sdtt_module}):
                    with mock.patch.object(mmd_reward, 'load_owt_ar_backbone', return_value=ar_backbone):
                        with reject_sdllm_import():
                            result = legacy_routing.construct(model)
                self.assertIs(result.denoiser, model.teacher if source == 'diffusion'
                              else ar_backbone if source == 'ar' else sdtt_backbone)

    def test_sdllm_configuration_does_not_override_idlm_ratio_route(self):
        model = make_model(mmd_reward_type='idlm', mmd_rbf_exclude_same_sentence=False,
                           mmd_rbf_exclude_same_position=False)
        ratio = SimpleNamespace(reward_batch_size=8)
        factory = mock.Mock(return_value=ratio)
        route_globals = type(model)._mmd_critic.fget.__globals__
        with mock.patch.dict(route_globals, {'IdlmRatioReward': factory}), reject_sdllm_import():
            self.assertIs(legacy_routing.construct(model), ratio)
        factory.assert_called_once_with(model)


class SDLLMStartupTests(unittest.TestCase):
    def make(self, **settings):
        return legacy_startup.SDTTStartupTests().make(
            mmd_feature_source='sdllm_mdlm', **settings)

    def test_prepares_native_teacher_before_parent_initial_evaluation(self):
        model = self.make()
        model.on_train_start()
        self.assertEqual(model.events, ['noise schedule', 'feature model', 'parent'])

    def test_disabled_zero_weight_or_ratio_modes_do_not_prepare_sdllm(self):
        for changes in ({'mmd_reward_weight': 0.}, {'mmd_reward_type': 'idlm'}):
            with self.subTest(changes=changes):
                model = self.make(**changes)
                model.on_train_start()
                self.assertNotIn('feature model', model.events)
        model = self.make()
        model.is_adversarial_distill = False
        model.on_train_start()
        self.assertEqual(model.events, ['parent'])

    def test_preparation_failure_stops_startup_before_parent_evaluation(self):
        model = self.make()
        model.download_error = RuntimeError('native release unavailable')
        with self.assertRaisesRegex(RuntimeError, 'native release unavailable'):
            model.on_train_start()
        self.assertEqual(model.events, ['noise schedule', 'feature model'])

    def test_legacy_sources_do_not_activate_external_startup(self):
        for source in ('ar', 'diffusion'):
            with self.subTest(source=source):
                model = legacy_startup.SDTTStartupTests().make(mmd_feature_source=source)
                model.on_train_start()
                self.assertNotIn('feature model', model.events)


if __name__ == '__main__':
    unittest.main(verbosity=2)
