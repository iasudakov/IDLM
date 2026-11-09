"""Offline CPU checks for native SDLLM-token MMD rewards and policy gradients.

No model assets, network, GPU, or tokenizer downloads are needed. The tiny
deterministic feature model deliberately returns unequal native token lengths.
The mathematical oracles below use dense float64 pair differences rather than
the production blocked kernels.

Run: python scripts/test_sdllm_reward.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mmd_rbf
import mmd_reward
import mmd_sdllm_reward
from mmd_sdllm_reward import FrozenSdllmFeatures, ragged_cross_sentence_mmd2


def dense_kernel(left, right, alpha, distance='squared_euclidean', delta=.3):
    differences = left.detach().double()[:, None] - right.detach().double()[None]
    if distance == 'squared_euclidean':
        distances = differences.square().sum(-1)
    else:
        distances = (torch.sqrt(differences.square() + delta ** 2) - delta).sum(-1)
    return torch.exp(-float(alpha) * distances)


def dense_cross_sentence(real, fake, alpha, *, exclude_position=False,
                         omit_fake=False, distance='squared_euclidean', delta=.3):
    """Each independent draw pair has equal weight, regardless of length."""
    cross = torch.stack([dense_kernel(real, row, alpha, distance, delta).mean()
                         for row in fake]).mean()
    fake_term = real.new_zeros((), dtype=torch.float64)
    if not omit_fake:
        terms = []
        for r, left in enumerate(fake):
            for s, right in enumerate(fake):
                if r == s:
                    continue
                kernel = dense_kernel(left, right, alpha, distance, delta)
                if exclude_position:
                    keep = (torch.arange(len(left))[:, None]
                            != torch.arange(len(right))[None])
                    terms.append(kernel[keep].mean())
                else:
                    terms.append(kernel.mean())
        fake_term = torch.stack(terms).mean()
    return fake_term - 2 * cross


def dense_ordinary(real, fake, alpha, *, omit_real=False, omit_fake=False):
    cross = dense_kernel(real, fake, alpha).mean()
    statistic = -2 * cross
    for left, omit in ((real, omit_real), (fake, omit_fake)):
        if not omit:
            keep = ~torch.eye(len(left), dtype=torch.bool)
            statistic = statistic + dense_kernel(left, left, alpha)[keep].mean()
    return statistic


def native_features(ids, layer, device='cpu'):
    """Feature oracle shared only with the deterministic test teacher."""
    values = torch.tensor(ids, dtype=torch.float32, device=device) * .1
    position = torch.arange(len(ids), device=device, dtype=torch.float32)
    base = torch.stack((values, values.square(), position * .2,
                        values.mean().expand_as(values)), dim=-1)
    if layer == -1:
        return base
    return (layer + 2) * base + torch.sin(base * (layer + 1))


class NativeFeatureModel(torch.nn.Module):
    """Toy native retokenizer plus exact-length physical forward buckets.

    Zero is a source piece that decodes to empty text; all other source IDs
    become one native piece. This makes source and native lengths differ
    without making tests depend on a downloaded tokenizer.
    """
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.vocab_embed = torch.nn.Identity()
        self.gain = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.forward_calls = []

    def retokenize(self, tokens, source_tokenizer, source_mask_index):
        del source_tokenizer
        if bool((tokens == source_mask_index).any()):
            raise ValueError('masked source token')
        return [[int(token) for token in row if token != 0] for row in tokens.tolist()]

    def features(self, rows, layers, forward_batch_size=None):
        output = {layer: [None] * len(rows) for layer in layers}
        size = len(rows) if forward_batch_size is None else int(forward_batch_size)
        if size < 1:
            raise ValueError('physical forward batch must be positive')
        for length in sorted(set(map(len, rows))):
            indices = [i for i, row in enumerate(rows) if len(row) == length]
            for start in range(0, len(indices), size):
                chunk = indices[start:start + size]
                self.forward_calls.append((length, len(chunk)))
                for i in chunk:
                    for layer in layers:
                        output[layer][i] = native_features(rows[i], layer, self.gain.device) * self.gain
        return output


class SdllmRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.clean = torch.tensor([[1, 2, 0, 3, 4],
                                   [2, 4, 3, 5, 0],
                                   [3, 1, 2, 0, 0]])
        self.draws = torch.tensor([
            [[2, 0, 1, 3, 5], [1, 0, 3, 0, 4], [3, 2, 4, 1, 0]],
            [[3, 2, 4, 1, 6], [2, 4, 2, 6, 0], [0, 3, 1, 2, 0]],
            [[4, 1, 2, 0, 0], [3, 5, 1, 7, 2], [2, 3, 0, 0, 0]],
        ])
        self.scored = torch.ones_like(self.clean, dtype=torch.bool)

    def critic(self, *, layers=0, weights=None, cross=True, position=True, **settings):
        teacher = NativeFeatureModel()
        critic = FrozenSdllmFeatures(teacher, object(), 99,
                                     layer_index=layers, layer_weights=weights)
        critic.kernel = 'rbf'
        critic.rbf_alpha = .27
        critic.rbf_exclude_same_sentence = cross
        critic.rbf_exclude_same_position = position
        critic.rbf_block_size = 2
        for key, value in settings.items():
            setattr(critic, key, value)
        return critic

    def native_rows(self, tokens, layer=0):
        return [native_features([int(x) for x in row if x != 0], layer)
                for row in tokens.tolist()]

    def assert_close(self, actual, expected, *, atol=2e-6, rtol=2e-6):
        torch.testing.assert_close(actual.double(), expected.double(), atol=atol, rtol=rtol)

    def test_equal_length_matches_legacy_cross_sentence_for_all_flags(self):
        generator = torch.Generator().manual_seed(318)
        real = torch.randn(2, 5, 4, generator=generator)
        fake = torch.randn(2, 3, 5, 4, generator=generator)
        selected = torch.ones(2, 5, dtype=torch.bool)
        for block in (1, 2, 0, 16):
            for position in (False, True):
                for omit in (False, True):
                    with self.subTest(block=block, position=position, omit=omit):
                        legacy, active = mmd_rbf.rbf_mmd2_cross_sentence(
                            real, fake, selected, .27, block_size=block,
                            exclude_same_position=position, omit_fake_fake=omit)
                        for i in range(len(real)):
                            actual, valid = ragged_cross_sentence_mmd2(
                                real[i], list(fake[i]), .27, block_size=block,
                                exclude_same_position=position, omit_fake_fake=omit)
                            self.assert_close(actual, legacy[i])
                            self.assertEqual(bool(valid), bool(active[i]))

    def test_unequal_lengths_match_independent_dense_oracle(self):
        generator = torch.Generator().manual_seed(179)
        real = torch.randn(4, 3, generator=generator)
        fake = [torch.randn(n, 3, generator=generator) for n in (3, 5, 7)]
        for block in (1, 2, 0, 10):
            for position in (False, True):
                for omit in (False, True):
                    with self.subTest(block=block, position=position, omit=omit):
                        actual, active = ragged_cross_sentence_mmd2(
                            real, fake, .21, block_size=block,
                            exclude_same_position=position, omit_fake_fake=omit)
                        expected = dense_cross_sentence(
                            real, fake, .21, exclude_position=position, omit_fake=omit)
                        self.assert_close(actual, expected)
                        self.assertTrue(bool(active))

    def test_pseudo_huber_ragged_distance_matches_independent_oracle(self):
        real = self.native_rows(self.clean)[0]
        fake = [self.native_rows(draw)[0] for draw in self.draws]
        for position in (False, True):
            actual, active = ragged_cross_sentence_mmd2(
                real, fake, .6, block_size=2, distance='pseudo_huber',
                huber_delta=.3, exclude_same_position=position)
            expected = dense_cross_sentence(
                real, fake, .6, distance='pseudo_huber', delta=.3,
                exclude_position=position)
            self.assert_close(actual, expected)
            self.assertTrue(bool(active))

    def test_signed_statistic_is_not_clamped_and_equal_draws_keep_equal_weight(self):
        real = torch.zeros(3, 2)
        fake = [torch.zeros(2, 2), torch.zeros(5, 2)]
        statistic, active = ragged_cross_sentence_mmd2(
            real, fake, .2, exclude_same_position=True)
        self.assertEqual(float(statistic), -1.)
        self.assertTrue(bool(active))
        real = torch.zeros(2, 1)
        fake = [torch.zeros(2, 1), torch.full((8, 1), 3.)]
        statistic, _ = ragged_cross_sentence_mmd2(real, fake, 1.)
        self.assert_close(statistic, dense_cross_sentence(real, fake, 1.))
        pooled_cross = torch.cat([dense_kernel(real, row, 1.).flatten()
                                  for row in fake]).mean()
        equal_draw_cross = torch.stack([dense_kernel(real, row, 1.).mean()
                                       for row in fake]).mean()
        self.assertGreater(float((pooled_cross - equal_draw_cross).abs()), .1)

    def test_helper_inactive_and_insufficient_draw_rules(self):
        real = torch.ones(3, 2)
        with self.assertRaises(ValueError):
            ragged_cross_sentence_mmd2(real, [real], .2)
        for fake in ([real[:0], real], [real[:1], real[:1]]):
            for omit in (False, True):
                value, active = ragged_cross_sentence_mmd2(
                    real, fake, .2, exclude_same_position=True, omit_fake_fake=omit)
                self.assertFalse(bool(active))
                self.assertEqual(float(value), 0.)

    def test_native_multidraw_reward_matches_dense_oracle(self):
        for position in (False, True):
            for omit in (False, True):
                critic = self.critic(position=position, rbf_omit_fake_fake=omit)
                actual, active = critic.multi_draw_reward(
                    self.clean, self.draws, self.scored, eos_id=None)
                real = self.native_rows(self.clean)
                fake = [self.native_rows(draw) for draw in self.draws]
                expected = torch.stack([-dense_cross_sentence(
                    row, [draw[i] for draw in fake], .27,
                    exclude_position=position, omit_fake=omit)
                    for i, row in enumerate(real)])
                self.assert_close(actual, expected)
                self.assertTrue(bool(active.all()))
                self.assertFalse(actual.requires_grad)

    def test_ordinary_paired_and_multidraw_rewards_use_true_native_lengths(self):
        for omit_real in (False, True):
            for omit_fake in (False, True):
                critic = self.critic(cross=False, position=False,
                                     rbf_omit_real_real=omit_real,
                                     rbf_omit_fake_fake=omit_fake)
                real = self.native_rows(self.clean)
                for paired in (False, True):
                    if paired:
                        reward, active = critic.paired_reward(
                            self.clean, self.draws[0], self.scored, None)
                        fake = self.native_rows(self.draws[0])
                    else:
                        reward, active = critic.multi_draw_reward(
                            self.clean, self.draws, self.scored, None)
                        draws = [self.native_rows(row) for row in self.draws]
                        fake = [torch.cat([draw[i] for draw in draws])
                                for i in range(len(real))]
                    expected = torch.stack([-dense_ordinary(
                        left, right, .27, omit_real=omit_real, omit_fake=omit_fake)
                        for left, right in zip(real, fake)])
                    self.assert_close(reward, expected)
                    self.assertTrue(bool(active.all()))

    def test_paired_kernel_row_groups_concatenate_native_features(self):
        clean, generated = self.clean[:2], self.draws[0, :2]
        critic = self.critic(cross=False, position=False, kernel_batch_size=2)
        reward, active = critic.paired_reward(
            clean, generated, torch.ones_like(clean, dtype=torch.bool), None)
        real = torch.cat(self.native_rows(clean))
        fake = torch.cat(self.native_rows(generated))
        self.assert_close(reward, -dense_ordinary(real, fake, .27).expand(2))
        self.assertTrue(bool(active.all()))

    def test_physical_forward_chunks_preserve_rewards_rng_and_logical_alpha(self):
        for alpha in (.27, None):
            baseline = None
            for size in (None, 1, 2, 4, 100):
                critic = self.critic(layers=[-1, 0, 1], weights=[.3, .4, 1.2],
                                     forward_batch_size=size, rbf_alpha=alpha)
                torch.manual_seed(331)
                output = critic.multi_draw_reward(
                    self.clean, self.draws, self.scored, None, return_per_layer=True)
                state = torch.get_rng_state()
                if baseline is None:
                    baseline = (output, state)
                else:
                    self.assert_close(output[0], baseline[0][0], atol=0, rtol=0)
                    self.assertTrue(torch.equal(output[1], baseline[0][1]))
                    for layer in (-1, 0, 1):
                        self.assert_close(output[2][layer], baseline[0][2][layer], atol=0, rtol=0)
                    self.assertTrue(torch.equal(state, baseline[1]))
                self.assertTrue(critic.denoiser.forward_calls)
                if size is not None:
                    self.assertTrue(all(batch <= size for _, batch
                                        in critic.denoiser.forward_calls))

    def test_autoalpha_uses_only_all_valid_real_native_tokens(self):
        critic = self.critic(rbf_alpha=None)
        real = self.native_rows(self.clean)
        pooled = torch.cat(real).double()
        differences = pooled[:, None] - pooled[None]
        distances = differences.square().sum(-1)
        off_diagonal = distances[~torch.eye(len(pooled), dtype=torch.bool)]
        expected_alpha = float(1 / (2 * off_diagonal.median()))
        calls = []
        original = mmd_sdllm_reward.median_alpha

        def capture(features, valid, **kwargs):
            calls.append((features.clone(), valid.clone(), original(features, valid, **kwargs)))
            return calls[-1][2]

        with mock.patch.object(mmd_sdllm_reward, 'median_alpha', side_effect=capture):
            reward, _ = critic.multi_draw_reward(self.clean, self.draws, self.scored, None)
            critic.multi_draw_reward(self.clean, self.draws + 10, self.scored, None)
        self.assertEqual(len(calls), 2)
        for packed, valid, alpha in calls:
            self.assertEqual(int(valid.sum()), sum(map(len, real)))
            self.assert_close(packed[valid], torch.cat(real), atol=0, rtol=0)
            self.assertAlmostEqual(alpha, expected_alpha, places=6)
        fixed = self.critic(rbf_alpha=calls[0][2])
        expected, _ = fixed.multi_draw_reward(self.clean, self.draws, self.scored, None)
        self.assert_close(reward, expected, atol=0, rtol=0)

    def test_autoalpha_subsampling_is_independent_of_physical_chunks(self):
        # More than 512 real positions exercises median_alpha's RNG subsample.
        clean = torch.arange(780).reshape(3, 260).remainder(19).add(1)
        draws = torch.stack((clean.roll(1, 1), clean.roll(2, 1) + 1))
        selected = torch.ones_like(clean, dtype=torch.bool)
        previous = None
        for size in (None, 1, 2):
            critic = self.critic(forward_batch_size=size, rbf_alpha=None,
                                 rbf_block_size=64)
            torch.manual_seed(99)
            result = critic.multi_draw_reward(clean, draws, selected, None)
            state = torch.get_rng_state()
            if previous is not None:
                self.assert_close(result[0], previous[0], atol=0, rtol=0)
                self.assertTrue(torch.equal(state, previous[1]))
            previous = (result[0], state)

    def test_weighted_multilayer_rewards_match_separate_layer_rewards(self):
        layers, weights = [-1, 0, 1], [.2, 1.3, .7]
        critic = self.critic(layers=layers, weights=weights)
        total, active, breakdown = critic.multi_draw_reward(
            self.clean, self.draws, self.scored, None, return_per_layer=True)
        expected = torch.zeros_like(total)
        for layer, weight in zip(layers, weights):
            separate, layer_active = self.critic(layers=layer).multi_draw_reward(
                self.clean, self.draws, self.scored, None)
            self.assert_close(breakdown[layer], separate, atol=0, rtol=0)
            self.assertTrue(torch.equal(active, layer_active))
            expected += weight * separate
        self.assert_close(total, expected, atol=0, rtol=0)

    def test_fixed_alpha_preserves_exactly_one_noise_zero_rng_draw_per_call(self):
        for paired in (False, True):
            critic = self.critic(cross=not paired, position=not paired)
            torch.manual_seed(628)
            before = torch.get_rng_state()
            torch.rand(self.clean.shape)
            expected = torch.get_rng_state()
            torch.set_rng_state(before)
            if paired:
                critic.paired_reward(self.clean, self.draws[0], self.scored, None)
            else:
                critic.multi_draw_reward(self.clean, self.draws, self.scored, None)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected))

    def test_unsupported_settings_and_partial_masks_fail_explicitly(self):
        invalid = [
            {'reward_type': 'feature_mse'}, {'kernel': 'linear'},
            {'kernel_level': 'sentence'}, {'sentence_pooling': 'last_token'},
            {'rbf_exclude_same_sentence': False}, {'rbf_distance': 'invalid'},
            {'rbf_distance': 'pseudo_huber', 'rbf_huber_delta': 0},
            {'rbf_alpha': float('nan')}, {'rbf_alpha': float('inf')}, {'rbf_alpha': -1},
        ]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.critic(**settings).multi_draw_reward(
                    self.clean, self.draws, self.scored, None)
        with self.assertRaises(ValueError):
            FrozenSdllmFeatures(NativeFeatureModel(), object(), 99, noise_level=.1)
        partial = self.scored.clone()
        partial[0, 1] = False
        for mask in (partial, self.scored[:, :2]):
            with self.subTest(mask_shape=mask.shape), self.assertRaises(ValueError):
                self.critic().multi_draw_reward(self.clean, self.draws, mask, None)
        with self.assertRaises(ValueError):
            self.critic().multi_draw_reward(self.clean, self.draws, self.scored, None,
                                            sequence_mask=partial)
        with self.assertRaises(ValueError):
            self.critic().multi_draw_reward(self.clean, self.draws[:1], self.scored, None)
        with self.assertRaises(ValueError):
            self.critic().paired_reward(self.clean, self.draws[0], self.scored, None)
        with self.assertRaises(ValueError):
            self.critic(cross=False, position=False, kernel_batch_size=2).paired_reward(
                self.clean, self.draws[0], self.scored, None)

    def test_all_valid_masks_are_accepted_but_short_native_rows_are_rejected(self):
        critic = self.critic()
        reward, active = critic.multi_draw_reward(
            self.clean, self.draws, self.scored, None, sequence_mask=self.scored)
        self.assertTrue(bool(active.all()))
        self.assertTrue(bool(torch.isfinite(reward).all()))
        for length in (0, 1):
            for omit in (False, True):
                broken = self.draws.clone()
                broken[1, 2] = 0
                broken[1, 2, :length] = 1
                critic.rbf_omit_fake_fake = omit
                with self.subTest(native_length=length, omit=omit), self.assertRaises(ValueError):
                    critic.multi_draw_reward(self.clean, broken, self.scored, None)

    def test_actual_reinforce_gradient_preserves_source_mask_group_and_draw_scaling(self):
        clean = self.clean[:2]
        written = torch.tensor([[False, True, True, False, False],
                                [True, False, True, False, False]])
        group, draws = 3, 2
        replacements = torch.stack([
            (clean + 1 + g * 2 + k).remainder(8) + 1
            for g in range(group) for k in range(draws)])
        samples = torch.where(written.unsqueeze(0), replacements, clean.unsqueeze(0))
        logp = torch.linspace(-2., -.3, samples.numel()).reshape(samples.shape).requires_grad_()
        weights, layers, reward_weight = [.4, 1.3], [0, 1], 1.7
        critic = self.critic(layers=layers, weights=weights)

        # Construct each candidate reward independently, then the exact
        # leave-one-out advantage and source-token score-function derivative.
        oracle_reward = torch.zeros(len(clean), group, dtype=torch.float64)
        oracle_layers = {}
        for layer, weight in zip(layers, weights):
            real = self.native_rows(clean, layer)
            per_layer = torch.zeros_like(oracle_reward)
            for g in range(group):
                native_draws = [self.native_rows(samples[g * draws + j], layer)
                                for j in range(draws)]
                for i, row in enumerate(real):
                    per_layer[i, g] = -dense_cross_sentence(
                        row, [native[i] for native in native_draws], .27,
                        exclude_position=True)
            oracle_layers[layer] = per_layer
            oracle_reward += weight * per_layer
        advantage = oracle_reward - ((oracle_reward.sum(-1, keepdim=True)
                                      - oracle_reward) / (group - 1))
        denominator = int(written.sum()) * group * draws
        expected_gradient = torch.zeros_like(logp, dtype=torch.float64)
        for g in range(group):
            expected_gradient[g * draws:(g + 1) * draws] = (
                -reward_weight * advantage[:, g, None] * written / denominator)
        expected_loss = (expected_gradient * logp.detach().double()).sum()

        torch.manual_seed(498)
        before = torch.get_rng_state()
        for _ in range(group):
            torch.rand(clean.shape)
        expected_state = torch.get_rng_state()
        torch.set_rng_state(before)
        loss, stats = mmd_reward.mmd_reinforce_loss_from_samples(
            logp, clean, written, samples, critic, group_size=group, eos_id=None,
            reward_weight=reward_weight, baseline='auto',
            scored=torch.ones_like(written), kernel_source='draws', kernel_draws=draws)
        loss.backward()
        self.assert_close(loss, expected_loss)
        self.assert_close(logp.grad, expected_gradient)
        self.assertTrue(bool((logp.grad[:, ~written] == 0).all()))
        self.assertGreater(float(logp.grad.abs().sum()), 0.)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_state))
        self.assertIsNone(critic.denoiser.gain.grad)
        self.assert_close(stats['mmd/reward'], oracle_reward.mean())
        for layer in layers:
            self.assert_close(stats[f'mmd/reward_l{layer}'], oracle_layers[layer].mean())


if __name__ == '__main__':
    unittest.main(verbosity=2)
