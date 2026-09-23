"""CPU routing checks for RBF excluding same-position cross-draw fake pairs.

Run: OMP_NUM_THREADS=2 python -B scripts/test_rbf_exclude_same_position_routing.py
No checkpoints, model downloads, network calls, or GPU are needed.
"""
from contextlib import redirect_stdout
import inspect
import io
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mmd_rbf
import mmd_reward
from test_feature_mse_config import _feature_stubs
from test_rbf_cross_sentence import critic_getter


def config(**changes):
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = compose(config_name='config', overrides=[
            'algo=mdlm', 'adversarial_distill.mmd_reward_type=feature',
            'adversarial_distill.mmd_feature_source=diffusion',
            'adversarial_distill.mmd_ar_model=stub-ar.ckpt',
            'adversarial_distill.mmd_kernel=rbf',
            'adversarial_distill.mmd_kernel_source=draws',
            'adversarial_distill.mmd_kernel_level=token',
            'adversarial_distill.mmd_kernel_batch_size=3',
            'adversarial_distill.mmd_rbf_exclude_same_sentence=true',
            'adversarial_distill.mmd_rbf_exclude_same_position=true'])
    return OmegaConf.merge(cfg, {'adversarial_distill': changes})


def excluded_position_reference(real, fake, selected, alpha, distance, delta,
                                omit_fake_fake=False):
    """Independent FP64 enumeration; real B,L,D and fake K,B,L,D.

    Fake-fake averages distinct draws AND distinct selected positions.
    Real-fake retains every selected real/fake position pair. Rows require
    at least two selected positions even when fake-fake is omitted.
    """
    def kernel(left, right):
        diff = left.double() - right.double()
        if distance == 'squared_euclidean':
            cost = diff.square().sum()
        else:
            cost = (torch.sqrt(diff.square() + delta ** 2) - delta).sum()
        return torch.exp(-alpha * cost)

    count, batch = fake.shape[:2]
    rewards = []
    for row in range(batch):
        positions = selected[row].nonzero().flatten().tolist()
        n = len(positions)
        if n < 2:
            rewards.append(0.)
            continue
        cross = sum(kernel(real[row, i], fake[k, row, j])
                    for k in range(count) for i in positions for j in positions)
        cross = cross / (count * n * n)
        within = 0. if omit_fake_fake else sum(
            kernel(fake[k, row, i], fake[l, row, j])
            for k in range(count) for l in range(count) if k != l
            for i in positions for j in positions if i != j
        ) / (count * (count - 1) * n * (n - 1))
        rewards.append(float(2 * cross - within))
    return torch.tensor(rewards, dtype=torch.float32)


class TinyBackbone(torch.nn.Module):
    """A deterministic token embedding map with real layer hooks."""
    def __init__(self):
        super().__init__()
        values = torch.arange(9, dtype=torch.float32)
        self.register_buffer('embeddings', torch.stack(
            [values / 3., values.square() / 7., values.sin()], dim=-1))
        self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.calls = []

    def forward(self, input_ids, sigma=None, attention_mask=None):
        self.calls.append(input_ids.clone())
        hidden = self.embeddings[input_ids]
        hidden = self.blocks[0](hidden)
        return self.blocks[1](hidden * 1.7)


def make_critic(adapter, noise=0.):
    backbone = TinyBackbone()
    if adapter == 'native':
        critic = mmd_reward.FrozenBackboneFeatures(
            backbone, mask_index=8, layer_index=[0, 1],
            layer_weights=[.4, 1.3], noise_level=noise)
    else:
        critic = mmd_reward.FrozenARFeatures.__new__(mmd_reward.FrozenARFeatures)
        critic.model, critic.vocab_size = backbone, 9
        critic._resolve_layers(backbone, [0, 1], [.4, 1.3])
    critic.kernel, critic.kernel_level = 'rbf', 'token'
    critic.kernel_batch_size, critic.forward_batch_size = 3, 2
    critic.rbf_alpha, critic.rbf_huber_delta = .13, .3
    critic.rbf_block_size = 2
    critic.rbf_exclude_same_sentence, critic.rbf_exclude_same_position = True, True
    return critic, backbone


class ExcludeSamePositionConfigTests(unittest.TestCase):
    def construct(self, cfg, route=None):
        model = SimpleNamespace(config=cfg, teacher=object(), mask_index=8,
                                device=torch.device('cpu'),
                                tokenizer=SimpleNamespace(eos_token_id=0, pad_token_id=9))
        with _feature_stubs() as stubs, redirect_stdout(io.StringIO()):
            actual = critic_getter()(model)
            self.assertIs(actual, stubs.critic)
            if route == 'diffusion':
                stubs.loader.assert_not_called()
                stubs.hf.assert_not_called()
                stubs.native.assert_called_once()
                self.assertIs(stubs.native.call_args.args[0], model.teacher)
            elif route == 'native_ar':
                stubs.loader.assert_called_once_with(
                    cfg.adversarial_distill.mmd_ar_model, cfg,
                    model.tokenizer, model.device)
                stubs.hf.assert_not_called()
                stubs.native.assert_called_once()
                self.assertIs(stubs.native.call_args.args[0], stubs.backbone)
                self.assertEqual(stubs.native.call_args.kwargs['noise_level'], 0.)
            elif route == 'hf_ar':
                stubs.loader.assert_not_called()
                stubs.native.assert_not_called()
                stubs.hf.assert_called_once()
                self.assertEqual(stubs.hf.call_args.args[0],
                                 cfg.adversarial_distill.mmd_ar_model)
        return actual

    def test_hydra_default_and_explicit_override_reach_diffusion_and_ar(self):
        raw = OmegaConf.load(ROOT / 'configs/config.yaml')
        self.assertIs(raw.adversarial_distill.mmd_rbf_exclude_same_position, False)
        for route, source, model in (
                ('diffusion', 'diffusion', 'stub-ar.ckpt'),
                ('native_ar', 'ar', 'stub-ar.ckpt'),
                ('hf_ar', 'ar', 'stub/hf-ar')):
            with self.subTest(route=route):
                cfg = config(mmd_feature_source=source, mmd_ar_model=model)
                self.assertIs(cfg.adversarial_distill.mmd_rbf_exclude_same_position, True)
                critic = self.construct(cfg, route=route)
                self.assertTrue(critic.rbf_exclude_same_position)
                self.assertTrue(critic.rbf_exclude_same_sentence)
                self.assertEqual(critic.kernel_batch_size, 3)

    def test_legacy_config_without_option_preserves_disabled_default(self):
        cfg = config(mmd_rbf_exclude_same_position=False)
        with open_dict(cfg.adversarial_distill):
            del cfg.adversarial_distill['mmd_rbf_exclude_same_position']
        self.assertFalse(self.construct(cfg).rbf_exclude_same_position)
        self.assertIs(mmd_reward._LayeredFeatures.rbf_exclude_same_position, False)
        for function in (mmd_reward.paired_sequence_reward,
                         mmd_reward.summed_paired_reward,
                         mmd_reward.summed_multi_draw_reward,
                         mmd_rbf.rbf_mmd2_cross_sentence):
            self.assertIs(inspect.signature(function).parameters['exclude_same_position'].default,
                          False)

    def test_unsupported_contexts_reject_before_loading_features(self):
        cases = [dict(mmd_kernel=k) for k in
                 ('linear', 'linear_l2', 'pseudo_huber')]
        cases += [dict(mmd_kernel_source='rows'), dict(mmd_kernel_level='sentence'),
                  dict(mmd_kernel_batch_size=1), dict(mmd_rbf_exclude_same_sentence=False),
                  dict(mmd_reward_type='idlm')]
        for changes in cases:
            with self.subTest(changes=changes), _feature_stubs() as stubs:
                model = SimpleNamespace(config=config(**changes), teacher=object(),
                                        mask_index=8, device=torch.device('cpu'))
                with self.assertRaises(ValueError):
                    critic_getter()(model)
                stubs.loader.assert_not_called()
                stubs.native.assert_not_called()
                stubs.hf.assert_not_called()

    def test_feature_mse_resets_kernel_only_flags(self):
        critic = self.construct(config(
            mmd_reward_type='feature_mse', mmd_kernel='pseudo_huber',
            mmd_kernel_source='rows', mmd_kernel_level='sentence',
            mmd_kernel_batch_size=1))
        self.assertEqual(critic.reward_type, 'feature_mse')
        self.assertFalse(critic.rbf_exclude_same_position)
        self.assertFalse(critic.rbf_exclude_same_sentence)
        self.assertEqual(critic.kernel, 'linear')
        self.assertEqual(critic.kernel_level, 'token')


class ExcludeSamePositionRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.clean = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0],
                                   [1, 2, 3, 4], [4, 3, 2, 1]])
        self.draws = torch.stack([self.clean.roll(1, 1), (self.clean + 2) % 7,
                                 self.clean.flip(1)])
        self.selected = torch.tensor([[1, 1, 0, 1], [0, 0, 1, 0],
                                      [0, 1, 0, 1], [0, 0, 0, 0]], dtype=torch.bool)
        self.uniform = torch.tensor([[.1, .8, .1, .8], [.9, .9, .1, .9],
                                     [.9, .8, .2, .1], [.1, .1, .1, .1]])
        generator = torch.Generator().manual_seed(931)
        self.real = torch.randn(4, 4, 3, generator=generator)
        self.fake = torch.randn(3, 4, 4, 3, generator=generator)

    def test_weighted_layers_match_independent_excluded_position_formula(self):
        hidden = torch.cat([self.real, self.fake.flatten(0, 1)])
        features = {0: hidden, 2: hidden * 1.7}
        for distance, omit_fake in itertools.product(
                ('squared_euclidean', 'pseudo_huber'), (False, True)):
            with self.subTest(distance=distance, omit_fake=omit_fake):
                total, active, layers = mmd_reward.summed_multi_draw_reward(
                    features, 4, 3, 4, self.selected, [0, 2], [.4, 1.3],
                    kernel='rbf', level='token', alpha=.13, distance=distance,
                    huber_delta=.3, block_size=2, exclude_same_sentence=True,
                    exclude_same_position=True, omit_fake_fake=omit_fake)
                for index, scale in ((0, 1.), (2, 1.7)):
                    expected = excluded_position_reference(
                        self.real * scale, self.fake * scale, self.selected,
                        .13, distance, .3, omit_fake_fake=omit_fake)
                    torch.testing.assert_close(layers[index], expected, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(total, .4 * layers[0] + 1.3 * layers[2])
                self.assertEqual(active.tolist(), [True, False, True, False])

    def test_fake_positions_excluded_but_real_fake_positions_remain_all_pairs(self):
        selected = torch.ones(1, 2, dtype=torch.bool)
        # Across the two draws, only same-position fake features coincide.
        # Excluding those pairs removes the entire surviving fake-fake term.
        real = torch.full((1, 2, 1), 1000.)
        fake = torch.tensor([[[[0.], [100.]]], [[[0.], [100.]]]])
        hidden = torch.cat([real, fake.flatten(0, 1)])
        args = ({0: hidden}, 1, 2, 2, selected, [0], [1.])
        actual = mmd_reward.summed_multi_draw_reward(
            *args, kernel='rbf', alpha=1., exclude_same_sentence=True,
            exclude_same_position=True)
        torch.testing.assert_close(actual[0], torch.zeros(1), atol=0, rtol=0)
        self.assertTrue(actual[1].item())
        # With reversed fake positions, same-position real-fake pairs disagree,
        # but half of ALL real-fake pairs agree: 2 * E[k(real,fake)] = 1.
        real = torch.tensor([[[0.], [100.]]])
        fake = torch.tensor([[[[100.], [0.]]], [[[100.], [0.]]]])
        hidden = torch.cat([real, fake.flatten(0, 1)])
        actual = mmd_reward.summed_multi_draw_reward(
            {0: hidden}, 1, 2, 2, selected, [0], [1.], kernel='rbf', alpha=1.,
            exclude_same_sentence=True, exclude_same_position=True,
            omit_fake_fake=True)
        torch.testing.assert_close(actual[0], torch.ones(1), atol=0, rtol=0)

    def test_one_position_stays_inactive_even_when_fake_term_is_omitted(self):
        hidden = torch.cat([self.real, self.fake.flatten(0, 1)])
        for omit_fake in (False, True):
            with self.subTest(omit_fake=omit_fake):
                result = mmd_reward.summed_multi_draw_reward(
                    {0: hidden}, 4, 3, 4, self.selected, [0], [1.],
                    kernel='rbf', alpha=.13, exclude_same_sentence=True,
                    exclude_same_position=True, omit_fake_fake=omit_fake)
                self.assertEqual(result[1].tolist(), [True, False, True, False])
                torch.testing.assert_close(result[0][[1, 3]], torch.zeros(2),
                                           atol=0, rtol=0)

    def test_clean_noisy_diffusion_and_ar_route_exclusion(self):
        for (adapter, noise), distance in itertools.product(
                [('native', 0.), ('native', .4), ('hf', 0.)],
                ('squared_euclidean', 'pseudo_huber')):
            with self.subTest(adapter=adapter, noise=noise, distance=distance):
                critic, backbone = make_critic(adapter, noise=noise)
                critic.rbf_distance = distance
                helper = mmd_reward.summed_multi_draw_reward
                with mock.patch.object(torch, 'rand', return_value=self.uniform), \
                     mock.patch.object(mmd_reward, 'summed_multi_draw_reward',
                                       wraps=helper) as routed:
                    result = critic.multi_draw_reward(
                        self.clean, self.draws, self.selected, 0, return_per_layer=True)
                self.assertIs(routed.call_args.kwargs['exclude_same_position'], True)
                drop = (self.uniform < noise) & self.selected
                clean = torch.where(drop, 8, self.clean)
                draws = torch.where(drop.unsqueeze(0), 8, self.draws)
                expected_layers = {}
                for index, scale in ((0, 1.), (1, 1.7)):
                    expected_layers[index] = excluded_position_reference(
                        backbone.embeddings[clean] * scale,
                        backbone.embeddings[draws] * scale, self.selected,
                        .13, distance, .3)
                    torch.testing.assert_close(result[2][index], expected_layers[index],
                                               atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(result[0], .4 * expected_layers[0] +
                                           1.3 * expected_layers[1], atol=2e-6, rtol=2e-5)
                self.assertEqual(result[1].tolist(), [True, False, True, False])
                self.assertLessEqual(max(len(call) for call in backbone.calls), 2)
                torch.testing.assert_close(torch.cat(backbone.calls), torch.cat(
                    [clean, draws.flatten(0, 1)]), atol=0, rtol=0)
                self.assertTrue(all(not layer._forward_hooks for layer in backbone.blocks))

    def test_auto_alpha_uses_real_features_and_reaches_exclusion_core(self):
        hidden = torch.cat([self.real, self.fake.flatten(0, 1)])
        core = mmd_rbf.rbf_mmd2_cross_sentence
        for omit_fake in (False, True):
            with self.subTest(omit_fake=omit_fake), \
                 mock.patch.object(mmd_reward, 'median_alpha', return_value=.13) as bandwidth, \
                 mock.patch.object(mmd_rbf, 'rbf_mmd2_cross_sentence', wraps=core) as routed:
                actual = mmd_reward.summed_multi_draw_reward(
                    {0: hidden}, 4, 3, 4, self.selected, [0], [1.],
                    kernel='rbf', alpha=None, exclude_same_sentence=True,
                    exclude_same_position=True, omit_fake_fake=omit_fake)
            bandwidth.assert_called_once()
            torch.testing.assert_close(bandwidth.call_args.args[0], self.real, atol=0, rtol=0)
            torch.testing.assert_close(bandwidth.call_args.args[1], self.selected)
            routed.assert_called_once()
            self.assertIs(routed.call_args.kwargs['exclude_same_position'], True)
            self.assertIs(routed.call_args.kwargs['omit_fake_fake'], omit_fake)
            expected = excluded_position_reference(self.real, self.fake, self.selected,
                                         .13, 'squared_euclidean', .001,
                                         omit_fake_fake=omit_fake)
            torch.testing.assert_close(actual[0], expected, atol=2e-6, rtol=2e-5)

    def test_rows_routes_reject_instead_of_silently_ignoring_flag(self):
        with self.assertRaises(ValueError):
            mmd_reward.paired_sequence_reward(
                self.real, self.fake[0], self.selected, kernel='rbf',
                alpha=.13, exclude_same_position=True)
        with self.assertRaises(ValueError):
            mmd_reward.summed_paired_reward(
                {0: torch.cat([self.real, self.fake[0]])}, 4,
                self.selected, [0], [1.], kernel='rbf', alpha=.13,
                exclude_same_position=True)
        for adapter, noise in [('native', 0.), ('native', .4), ('hf', 0.)]:
            with self.subTest(adapter=adapter, noise=noise):
                critic, _ = make_critic(adapter, noise=noise)
                # Isolate the new flag: the older exclusion flag cannot be
                # responsible for this rows rejection.
                critic.rbf_exclude_same_sentence = False
                with self.assertRaises(ValueError):
                    critic.paired_reward(self.clean, self.draws[0], self.selected, 0)

    def test_lower_draws_reject_unsupported_contexts(self):
        for changes in [dict(kernel='linear'), dict(kernel='linear_l2'),
                        dict(kernel='pseudo_huber'),
                        dict(level='sentence'), dict(exclude_same_sentence=False),
                        dict(draw_count=1)]:
            settings = dict(kernel='rbf', level='token', exclude_same_sentence=True,
                            draw_count=3)
            settings.update(changes)
            count = settings.pop('draw_count')
            hidden = torch.cat([self.real, self.fake[:count].flatten(0, 1)])
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                mmd_reward.summed_multi_draw_reward(
                    {0: hidden}, 4, count, 4, self.selected, [0], [1.],
                    alpha=.13, exclude_same_position=True, **settings)

    def test_feature_mse_helpers_ignore_kernel_only_flag(self):
        paired_args = ({0: torch.cat([self.real, self.fake[0]])},
                       4, self.selected, [0], [1.])
        draw_args = ({0: torch.cat([self.real, self.fake.flatten(0, 1)])},
                     4, 3, 4, self.selected, [0], [1.])
        for helper, args in [(mmd_reward.summed_paired_reward, paired_args),
                             (mmd_reward.summed_multi_draw_reward, draw_args)]:
            with self.subTest(helper=helper.__name__):
                expected = helper(*args, reward_type='feature_mse')
                actual = helper(*args, reward_type='feature_mse', exclude_same_position=True,
                                exclude_same_sentence=True, kernel='linear')
                torch.testing.assert_close(expected[0], actual[0], atol=0, rtol=0)
                torch.testing.assert_close(expected[1], actual[1])

    def test_disabled_flag_preserves_old_rows_and_draws_values(self):
        paired = ({0: torch.cat([self.real, self.fake[0]])}, 4,
                  self.selected, [0], [1.])
        draws = ({0: torch.cat([self.real, self.fake.flatten(0, 1)])},
                 4, 3, 4, self.selected, [0], [1.])
        for helper, args, excluded in [
                (mmd_reward.summed_paired_reward, paired, False),
                (mmd_reward.summed_multi_draw_reward, draws, False),
                (mmd_reward.summed_multi_draw_reward, draws, True)]:
            with self.subTest(helper=helper.__name__, excluded=excluded):
                options = dict(kernel='rbf', alpha=.13, exclude_same_sentence=excluded)
                default = helper(*args, **options)
                explicit = helper(*args, exclude_same_position=False, **options)
                torch.testing.assert_close(default[0], explicit[0], atol=0, rtol=0)
                torch.testing.assert_close(default[1], explicit[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
