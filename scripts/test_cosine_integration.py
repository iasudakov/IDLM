"""CPU integration regressions for true cosine-kernel MMD rewards.

Run from the repository root: python scripts/test_cosine_integration.py
References explicitly form normalized-dot Gram matrices, independently of the
production cosine helpers. No weights, network, tracker, or GPU are required.
"""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import mmd_reward as reward
import test_sentence_pooling as pooling_fixture


def normalized(values):
    return F.normalize(values.double(), dim=-1, eps=1e-12)


def unequal_reference(real, fake):
    """Off-diagonal estimator for two active sample sets, without 1/D."""
    if len(real) < 2 or len(fake) < 2:
        return 0.
    x, y = normalized(real), normalized(fake)
    xx, yy, xy = x @ x.T, y @ y.T, x @ y.T
    return float((xx.sum() - xx.diag().sum()) / (len(x) * (len(x) - 1))
                 + (yy.sum() - yy.diag().sum()) / (len(y) * (len(y) - 1))
                 - 2 * xy.mean())


def fixed_reference(real, fake):
    x, y = normalized(real), normalized(fake)
    yy = y @ y.T
    fake_term = (yy[0, 0] if len(y) == 1 else
                 (yy.sum() - yy.diag().sum()) / (len(y) * (len(y) - 1)))
    return float(x.dot(x) + fake_term - 2 * (y @ x).mean())


def sentence_reference(hidden, selected, pooling='mean', valid=None):
    """Independent existing pooling semantics, before any normalization."""
    vectors, active = [], []
    for index, row in enumerate(hidden):
        positions = (torch.arange(len(row)) if valid is None else
                     valid[index].nonzero().flatten())
        enabled = bool(selected[index].any()) and (pooling == 'mean' or len(positions) > 0)
        if not enabled:
            vectors.append(torch.zeros(row.shape[-1], dtype=row.dtype))
        elif pooling == 'mean':
            vectors.append(row[selected[index]].mean(dim=0))
        else:
            vectors.append(row[positions[-1]])
        active.append(enabled)
    return torch.stack(vectors), active


def draw_reward(real, fake, selected, **kwargs):
    count, batch, length, hidden = fake.shape
    features = {0: torch.cat([real, fake.reshape(-1, length, hidden)])}
    return reward.summed_multi_draw_reward(
        features, batch, count, length, selected, [0], [1.],
        kernel='cosine', **kwargs)


class CosineRewardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(815)
        self.real = torch.randn(4, 5, 3, generator=generator)
        self.fake = torch.randn(4, 4, 5, 3, generator=generator)
        self.selected = torch.tensor([[True, True, False, True, False],
                                      [False, True, True, False, False],
                                      [True, False, False, False, False],
                                      [False, False, False, False, False]])

    def test_rows_token_matches_dense_and_normalizes_each_token(self):
        for rows_of in (1, 2, 4):
            actual, active = reward.paired_sequence_reward(
                self.real, self.fake[0], self.selected, kernel='cosine', rows_of=rows_of)
            expected, enabled = [], []
            for start in range(0, len(self.real), rows_of):
                mask = self.selected[start:start + rows_of].flatten()
                clean = self.real[start:start + rows_of].flatten(0, 1)[mask]
                generated = self.fake[0, start:start + rows_of].flatten(0, 1)[mask]
                expected.extend([-unequal_reference(clean, generated)] * rows_of)
                enabled.extend([len(clean) > 1] * rows_of)
            torch.testing.assert_close(actual, torch.tensor(expected), atol=2e-6, rtol=2e-5)
            self.assertEqual(active.tolist(), enabled)
        # Independent positive rescaling at EACH token must not change token kernels.
        scales = torch.tensor([.01, 100., .2, 13., .8])[None, :, None]
        base, _ = reward.paired_sequence_reward(
            self.real, self.fake[0], self.selected, kernel='cosine')
        scaled, _ = reward.paired_sequence_reward(
            self.real * scales, self.fake[0] / scales, self.selected, kernel='cosine')
        torch.testing.assert_close(base, scaled, atol=2e-6, rtol=2e-5)

    def test_rows_sentence_pools_each_sequence_then_normalizes(self):
        for rows_of in (2, 4):
            actual, active = reward.paired_sequence_reward(
                self.real, self.fake[0], self.selected, kernel='cosine',
                rows_of=rows_of, level='sentence')
            clean, enabled = sentence_reference(self.real, self.selected)
            generated, _ = sentence_reference(self.fake[0], self.selected)
            expected, expected_active = [], []
            for start in range(0, len(clean), rows_of):
                indices = [i for i in range(start, start + rows_of) if enabled[i]]
                estimate = unequal_reference(clean[indices], generated[indices])
                expected.extend([-estimate] * rows_of)
                expected_active.extend([len(indices) > 1] * rows_of)
            torch.testing.assert_close(actual, torch.tensor(expected), atol=2e-6, rtol=2e-5)
            self.assertEqual(active.tolist(), expected_active)
        with self.assertRaises(ValueError):
            reward.paired_sequence_reward(self.real, self.fake[0], self.selected,
                                          kernel='cosine', level='sentence', rows_of=1)

    def test_sentence_normalization_is_after_mean_pooling_not_before(self):
        clean = torch.tensor([[[100., 0.], [0., 1.]]])
        fake = torch.tensor([[[[1., 0.], [0., 100.]]]])
        selected = torch.ones(1, 2, dtype=torch.bool)
        actual, active, _ = draw_reward(clean, fake, selected, level='sentence')
        expected = -fixed_reference(clean.mean(1)[0], fake.mean(2)[:, 0])
        torch.testing.assert_close(actual, torch.tensor([expected]), atol=2e-6, rtol=2e-5)
        self.assertTrue(active.all())
        self.assertLess(float(actual), -1.9)
        # Normalizing tokens before pooling would give identical means and zero.
        torch.testing.assert_close(normalized(clean).mean(1), normalized(fake).mean(2)[0])

    def test_draws_token_k1_k2_k4_unequal_sample_counts_and_inactivity(self):
        for count in (1, 2, 4):
            actual, active, layers = draw_reward(
                self.real, self.fake[:count], self.selected, level='token')
            expected = []
            for row, mask in enumerate(self.selected):
                real = self.real[row, mask]
                fake = self.fake[:count, row][:, mask].reshape(-1, self.real.shape[-1])
                expected.append(-unequal_reference(real, fake))
            torch.testing.assert_close(actual, torch.tensor(expected), atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(layers[0], actual)
            self.assertEqual(active.tolist(), [True, True, False, False])

    def test_draws_sentence_k1_biased_and_k2_k4_unbiased(self):
        clean, enabled = sentence_reference(self.real, self.selected)
        for count in (1, 2, 4):
            fake = torch.stack([sentence_reference(draw, self.selected)[0]
                                for draw in self.fake[:count]])
            actual, active, _ = draw_reward(self.real, self.fake[:count], self.selected,
                                             level='sentence')
            expected = [-fixed_reference(clean[row], fake[:, row]) if enabled[row] else 0.
                        for row in range(len(clean))]
            torch.testing.assert_close(actual, torch.tensor(expected), atol=2e-6, rtol=2e-5)
            self.assertEqual(active.tolist(), enabled)

    def test_no_huber_bandwidth_or_hidden_dimension_division(self):
        # Orthogonal nonzero vectors give cosine-MMD^2=2 in every dimension.
        with mock.patch.object(reward, 'pseudo_huber_reward', side_effect=AssertionError('Huber called')), \
                mock.patch.object(reward, 'median_alpha', side_effect=AssertionError('RBF bandwidth called')):
            for hidden in (2, 7, 33):
                clean = torch.zeros(1, 2, hidden)
                fake = torch.zeros(1, 1, 2, hidden)
                clean[..., 0], fake[..., 1] = 2., 9.
                selected = torch.ones(1, 2, dtype=torch.bool)
                sentence, _, _ = draw_reward(clean, fake, selected, level='sentence')
                token, _, _ = draw_reward(clean, fake, selected, level='token')
                rows, _ = reward.paired_sequence_reward(clean, fake[0], selected, kernel='cosine')
                for result in (sentence, token, rows):
                    torch.testing.assert_close(result, torch.tensor([-2.]), atol=1e-6, rtol=1e-6)

    def test_negative_u_statistics_are_not_clamped(self):
        points = torch.tensor([[[1., 0.], [-1., 0.]]])
        selected = torch.ones(1, 2, dtype=torch.bool)
        actual, _ = reward.paired_sequence_reward(points, points, selected, kernel='cosine')
        torch.testing.assert_close(actual, torch.tensor([2.]))
        real = torch.tensor([[[1., 0.]]])
        fake = torch.tensor([[[[.5, math.sqrt(.75)]]], [[[.5, -math.sqrt(.75)]]]])
        actual, _, _ = draw_reward(real, fake, torch.ones(1, 1, dtype=torch.bool), level='sentence')
        torch.testing.assert_close(actual, torch.tensor([.5]), atol=1e-6, rtol=1e-6)

    def test_multilayer_rewards_are_weighted_sums_not_averages(self):
        hidden = torch.cat([self.real, self.fake[:2].reshape(-1, 5, 3)])
        features = {0: hidden, 3: hidden * 1.7 + torch.tensor([.2, -.8, .4])}
        for level in ('token', 'sentence'):
            actual, active, per_layer = reward.summed_multi_draw_reward(
                features, 4, 2, 5, self.selected, [0, 3], [.3, 1.25],
                kernel='cosine', level=level)
            torch.testing.assert_close(actual, .3 * per_layer[0] + 1.25 * per_layer[3])
            self.assertGreater(float((per_layer[0] - per_layer[3]).abs().sum()), .01)
            self.assertEqual(active.tolist(), [True, True, level == 'sentence', False])
        pair_features = {index: layer[:8] for index, layer in features.items()}
        actual, _, per_layer = reward.summed_paired_reward(
            pair_features, 4, self.selected, [0, 3], [.3, 1.25],
            kernel='cosine', rows_of=2, level='sentence')
        torch.testing.assert_close(actual, .3 * per_layer[0] + 1.25 * per_layer[3])

    def test_last_token_pooling_and_sequence_padding_are_preserved(self):
        valid = torch.tensor([[True, True, True, False, False],
                              [False, True, True, True, False],
                              [True, False, True, False, True],
                              [True, True, True, True, True]])
        clean, active_ref = sentence_reference(self.real, self.selected, 'last_token', valid)
        for count in (1, 2, 4):
            fake = torch.stack([sentence_reference(draw, self.selected, 'last_token', valid)[0]
                                for draw in self.fake[:count]])
            actual, active, _ = draw_reward(
                self.real, self.fake[:count], self.selected, level='sentence',
                sentence_pooling='last_token', sequence_mask=valid)
            expected = [-fixed_reference(clean[row], fake[:, row]) if active_ref[row] else 0.
                        for row in range(len(clean))]
            torch.testing.assert_close(actual, torch.tensor(expected), atol=2e-6, rtol=2e-5)
            self.assertEqual(active.tolist(), active_ref)

    def test_frozen_native_and_generic_wrappers_keep_contextual_endpoints(self):
        clean = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        fake = torch.tensor([[[1, 2, 2, 3], [4, 6, 6, 7]],
                              [[2, 3, 2, 3], [4, 7, 6, 7]]])
        selected = torch.tensor([[True, True, False, False], [False, True, False, False]])
        valid = torch.tensor([[True, True, True, False], [False, True, True, True]])
        for generic in (False, True):
            critic, model = pooling_fixture._critic(generic)
            critic.kernel = 'cosine'
            critic.kernel_batch_size = 2
            for count in (1, 2):
                actual, active, layers = critic.multi_draw_reward(
                    clean, fake[:count], selected, eos_id=0, sequence_mask=valid,
                    return_per_layer=True)
                hidden = model.embedding(torch.cat([clean, fake[:count].reshape(-1, 4)])).cumsum(1)
                vectors, _ = sentence_reference(hidden, selected.repeat(count + 1, 1),
                                                  'last_token', valid.repeat(count + 1, 1))
                expected = [-fixed_reference(vectors[row], vectors[2:].reshape(count, 2, 2)[:, row])
                            for row in range(2)]
                for layer in layers.values():
                    torch.testing.assert_close(layer, torch.tensor(expected), atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual, .3 * layers[0] + 1.25 * layers[1])
                self.assertTrue(active.all())
                self.assertFalse(actual.requires_grad)
            pair, active, _ = critic.paired_reward(
                clean, fake[0], selected, eos_id=0, sequence_mask=valid, return_per_layer=True)
            self.assertTrue(active.all())
            self.assertEqual(pair.shape, (2,))
            torch.testing.assert_close(pair[0], pair[1])


class TokenTableCritic:
    kernel = 'cosine'
    kernel_level = 'sentence'
    resolved_layers = [0]

    def __init__(self):
        self.table = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        self.calls = []

    def multi_draw_reward(self, clean, draws, selected, eos):
        del eos
        self.calls.append(tuple(draws.shape))
        return draw_reward(self.table[clean], self.table[draws], selected,
                           level='sentence')[:2]


class CosinePolicyAndConfigurationTests(unittest.TestCase):
    def test_cosine_preserves_g_k_denominator_and_written_gradient_mask(self):
        gradients = []
        for count in (1, 2, 4):
            batch, length, vocab, group = 2, 4, 4, 3
            written = torch.tensor([[True, True, False, False]]).repeat(batch, 1)
            probabilities = torch.full((batch, length, vocab), 1 / vocab, requires_grad=True)
            critic = TokenTableCritic()

            def sample(flat, num_samples, replacement=False, **kwargs):
                self.assertTrue(replacement)
                self.assertEqual(num_samples, group * count)
                return torch.arange(group).repeat_interleave(count).expand(flat.shape[0], -1)

            with mock.patch.object(torch, 'multinomial', sample):
                loss, _ = reward.mmd_reinforce_loss(
                    probabilities, torch.zeros(batch, length, dtype=torch.long), written,
                    critic, group_size=group, eos_id=None, baseline='group',
                    kernel_source='draws', kernel_draws=count)
            loss.backward()
            expected = torch.zeros_like(probabilities)
            # Cosine rewards [0,-2,-4] give LOO advantages [3,0,-3].
            # 3 / (4 written positions * G=3 * p=.25) = 1, after K cancels.
            expected[..., 0][written] = -1.
            expected[..., 2][written] = 1.
            torch.testing.assert_close(probabilities.grad, expected, rtol=0, atol=0)
            self.assertEqual(critic.calls, [(count, batch, length)] * group)
            gradients.append(probabilities.grad)
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradient, gradients[0], rtol=0, atol=0)

    def test_all_position_scoring_never_expands_policy_gradient_support(self):
        clean = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        written = torch.tensor([[True, True, False, False], [False, True, False, False]])
        scored = torch.ones_like(written)
        valid = torch.tensor([[True, True, True, False], [False, True, True, True]])
        for source in ('rows', 'draws'):
            critic, _ = pooling_fixture._critic()
            critic.kernel = 'cosine'
            critic.kernel_batch_size = 2
            probabilities = torch.full((2, 4, 16), 1 / 16, requires_grad=True)
            method = 'paired_reward' if source == 'rows' else 'multi_draw_reward'
            torch.manual_seed(32)
            with mock.patch.object(critic, method, wraps=getattr(critic, method)) as score:
                loss, _ = reward.mmd_reinforce_loss(
                    probabilities, clean, written, critic, group_size=2, eos_id=0,
                    scored=scored, baseline='none', kernel_source=source, kernel_draws=2,
                    sequence_mask=valid)
            for call in score.call_args_list:
                torch.testing.assert_close(call.args[2], scored)
                torch.testing.assert_close(call.kwargs['sequence_mask'], valid)
            loss.backward()
            self.assertTrue(torch.isfinite(probabilities.grad).all())
            self.assertTrue((probabilities.grad[~written] == 0).all())
            self.assertGreater(float(probabilities.grad[written].abs().sum()), 0.)

    def test_algo_accepts_cosine_feature_routes_without_loading_real_models(self):
        getter = pooling_fixture._critic_getter()
        for source in ('rows', 'draws'):
            for level, pooling in (('token', 'mean'), ('sentence', 'mean'),
                                   ('sentence', 'last_token')):
                model = pooling_fixture._configured_model(
                    mmd_kernel='cosine', mmd_kernel_source=source, mmd_kernel_level=level,
                    mmd_sentence_pooling=pooling, mmd_kernel_batch_size=2)
                critic = SimpleNamespace(layer_description='tiny local fixture')
                with mock.patch.object(reward, 'load_owt_ar_backbone'), \
                        mock.patch.object(reward, 'FrozenBackboneFeatures', return_value=critic):
                    self.assertIs(getter(model), critic)
                self.assertEqual(critic.kernel, 'cosine')
                self.assertEqual(critic.kernel_level, level)
                self.assertEqual(critic.sentence_pooling, pooling)
        with mock.patch.object(reward, 'load_owt_ar_backbone') as loader:
            with self.assertRaises(ValueError):
                getter(pooling_fixture._configured_model(
                    mmd_kernel='cosine', mmd_reward_type='idlm', mmd_sentence_pooling='mean'))
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
