"""CPU routing and score-gradient references for omitting the RBF fake term.

Run: OMP_NUM_THREADS=2 python -B scripts/test_rbf_omit_fake_fake_routing.py
"""
import inspect
import itertools
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mmd_reward


def dense_kernel(real, fake, alpha, distance, delta):
    diff = real.double()[:, None] - fake.double()[None]
    if distance == 'squared_euclidean':
        dist = diff.square().sum(-1)
    else:
        dist = (torch.sqrt(diff.square() + delta ** 2) - delta).sum(-1)
    return torch.exp(-alpha * dist)


def dense_reward(real, fake, selected, *, source, level, alpha, distance,
                 delta, rows=2, omit_rr=False, omit_ff=False, cross=False):
    """Independent FP64 estimator over explicitly selected feature sets."""
    batch, length, width = real.shape
    count = fake.shape[0]
    values, actives = [], []
    step = rows if source == 'rows' else 1
    for start in range(0, batch, step):
        real_points, fake_points, labels = [], [], []
        for b in range(start, start + step):
            mask = selected[b]
            if not bool(mask.any()):
                continue
            x = real[b, mask]
            y = fake[:, b, mask]
            if level == 'sentence':
                real_points.append(x.mean(0, keepdim=True))
                fake_points.append(y.mean(1))
            else:
                real_points.append(x)
                fake_points.append(y.reshape(-1, width))
                labels.append(torch.arange(count).repeat_interleave(len(x)))
        n_real = sum(len(x) for x in real_points)
        fixed = source == 'draws' and level == 'sentence'
        active = n_real >= (1 if fixed or cross else 2)
        if not active:
            value = torch.tensor(0., dtype=torch.double)
        else:
            x, y = torch.cat(real_points), torch.cat(fake_points)
            value = 2 * dense_kernel(x, y, alpha, distance, delta).mean()
            if not omit_rr and not cross:
                if fixed:
                    value -= 1.
                else:
                    xx = dense_kernel(x, x, alpha, distance, delta)
                    value -= xx[~torch.eye(len(x), dtype=torch.bool)].mean()
            if not omit_ff:
                if fixed and count == 1:
                    value -= 1.
                else:
                    yy = dense_kernel(y, y, alpha, distance, delta)
                    keep = ~torch.eye(len(y), dtype=torch.bool)
                    if cross:
                        labels = torch.cat(labels)
                        keep = labels[:, None] != labels[None]
                    value -= yy[keep].mean()
        values.extend([value] * step)
        actives.extend([active] * step)
    return torch.stack(values).float(), torch.tensor(actives)


def token_features(tokens):
    values = tokens.float()
    return torch.stack([values.sin(), values.cos(), values / 5.], -1)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.calls = []

    def forward(self, input_ids, sigma=None, attention_mask=None):
        self.calls.append(input_ids.shape[0])
        hidden = token_features(input_ids)
        for i, layer in enumerate(self.blocks):
            hidden = layer(hidden * (i + 1))
        return hidden


def make_critic(adapter, level='token', rows=2):
    backbone = Backbone()
    if adapter == 'native':
        critic = mmd_reward.FrozenBackboneFeatures(
            backbone, mask_index=8, layer_index=[0, 1],
            layer_weights=[.4, 1.3], noise_level=0.)
    else:
        critic = mmd_reward.FrozenARFeatures.__new__(mmd_reward.FrozenARFeatures)
        critic.model, critic.vocab_size = backbone, 8
        critic._resolve_layers(backbone, [0, 1], [.4, 1.3])
    critic.kernel = 'rbf'
    critic.kernel_level, critic.kernel_batch_size = level, rows
    critic.forward_batch_size = 2
    return critic, backbone


class OmitFakeFakeRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        self.real = torch.randn(4, 5, 3)
        self.fake = torch.randn(3, 4, 5, 3)
        self.selected = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 0, 0, 0],
                                      [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                                     dtype=torch.bool)
        self.alpha, self.delta = .13, .3

    def expected(self, real, fake, **kwargs):
        return dense_reward(real, fake, self.selected, alpha=self.alpha,
                            delta=self.delta, **kwargs)

    def test_flag_is_last_optional_argument_with_false_default(self):
        for helper in (mmd_reward.rbf_sequence_reward,
                       mmd_reward.paired_sequence_reward,
                       mmd_reward.summed_paired_reward,
                       mmd_reward.summed_multi_draw_reward):
            params = list(inspect.signature(helper).parameters.values())
            self.assertEqual(params[-1].name, 'omit_fake_fake')
            self.assertIs(params[-1].default, False)
        self.assertIs(mmd_reward._LayeredFeatures.rbf_omit_fake_fake, False)

    def test_rows_token_sentence_distances_both_rr_states_and_weighted_layers(self):
        for level, rows, distance, omit_rr, block in itertools.product(
                ('token', 'sentence'), (1, 2, 4),
                ('squared_euclidean', 'pseudo_huber'), (False, True), (0, 2)):
            if level == 'sentence' and rows == 1:
                continue
            with self.subTest(level=level, rows=rows, distance=distance,
                              omit_rr=omit_rr, block=block):
                hidden = torch.cat([self.real, self.fake[0]])
                args = ({0: hidden, 2: hidden * 1.7}, 4,
                        self.selected, [0, 2], [.4, 1.3])
                kwargs = dict(kernel='rbf', level=level, rows_of=rows,
                              alpha=self.alpha, distance=distance,
                              huber_delta=self.delta, omit_real_real=omit_rr,
                              block_size=block)
                default = mmd_reward.summed_paired_reward(*args, **kwargs)
                explicit = mmd_reward.summed_paired_reward(
                    *args, omit_fake_fake=False, **kwargs)
                actual = mmd_reward.summed_paired_reward(
                    *args, omit_fake_fake=True, **kwargs)
                torch.testing.assert_close(default[0], explicit[0], atol=0, rtol=0)
                for index, scale in ((0, 1.), (2, 1.7)):
                    for omit_ff, output in ((False, default), (True, actual)):
                        expected, active = self.expected(
                            self.real * scale, self.fake[:1] * scale,
                            source='rows', level=level, rows=rows, distance=distance,
                            omit_rr=omit_rr, omit_ff=omit_ff)
                        torch.testing.assert_close(output[2][index], expected,
                                                   atol=2e-6, rtol=2e-5)
                        torch.testing.assert_close(output[1], active)
                torch.testing.assert_close(actual[0], .4 * actual[2][0] + 1.3 * actual[2][2])

    def test_draws_levels_k1_k3_cross_sentence_both_rr_states(self):
        modes = [('token', 1, False), ('token', 3, False),
                 ('sentence', 1, False), ('sentence', 3, False), ('token', 3, True)]
        for (level, count, cross), distance, omit_rr, block in itertools.product(
                modes, ('squared_euclidean', 'pseudo_huber'), (False, True), (0, 2)):
            with self.subTest(level=level, count=count, cross=cross,
                              distance=distance, omit_rr=omit_rr, block=block):
                hidden = torch.cat([self.real, self.fake[:count].flatten(0, 1)])
                args = ({0: hidden, 2: hidden * 1.7}, 4, count, 5,
                        self.selected, [0, 2], [.4, 1.3])
                kwargs = dict(kernel='rbf', level=level, alpha=self.alpha,
                              distance=distance, huber_delta=self.delta, block_size=block,
                              omit_real_real=omit_rr, exclude_same_sentence=cross)
                default = mmd_reward.summed_multi_draw_reward(*args, **kwargs)
                explicit = mmd_reward.summed_multi_draw_reward(
                    *args, omit_fake_fake=False, **kwargs)
                actual = mmd_reward.summed_multi_draw_reward(
                    *args, omit_fake_fake=True, **kwargs)
                torch.testing.assert_close(default[0], explicit[0], atol=0, rtol=0)
                for index, scale in ((0, 1.), (2, 1.7)):
                    for omit_ff, output in ((False, default), (True, actual)):
                        expected, active = self.expected(
                            self.real * scale, self.fake[:count] * scale,
                            source='draws', level=level, distance=distance,
                            omit_rr=omit_rr, omit_ff=omit_ff, cross=cross)
                        torch.testing.assert_close(output[2][index], expected,
                                                   atol=2e-6, rtol=2e-5)
                        torch.testing.assert_close(output[1], active)
                torch.testing.assert_close(actual[0], .4 * actual[2][0] + 1.3 * actual[2][2])

    def test_hf_native_all_four_routes_and_auto_alpha_unchanged(self):
        clean = torch.randint(0, 7, (4, 5))
        draws = torch.randint(0, 7, (2, 4, 5))
        for adapter, source, level, distance, omit_rr in itertools.product(
                ('native', 'hf'), ('rows', 'draws'), ('token', 'sentence'),
                ('squared_euclidean', 'pseudo_huber'), (False, True)):
            with self.subTest(adapter=adapter, source=source, level=level,
                              distance=distance, omit_rr=omit_rr):
                critic, backbone = make_critic(adapter, level=level)
                critic.rbf_distance, critic.rbf_huber_delta = distance, self.delta
                critic.rbf_omit_real_real = omit_rr
                name = 'summed_paired_reward' if source == 'rows' else 'summed_multi_draw_reward'
                helper = getattr(mmd_reward, name)
                def score():
                    if source == 'rows':
                        return critic.paired_reward(clean, draws[0], self.selected, 0,
                                                    return_per_layer=True)
                    return critic.multi_draw_reward(clean, draws, self.selected, 0,
                                                    return_per_layer=True)
                with mock.patch.object(mmd_reward, 'median_alpha', wraps=mmd_reward.median_alpha) as alpha:
                    with mock.patch.object(mmd_reward, name, wraps=helper) as routed:
                        default = score()
                        self.assertIs(routed.call_args.kwargs['omit_fake_fake'], False)
                        before = [(c.args[0].clone(), c.args[1].clone()) for c in alpha.call_args_list]
                        alpha.reset_mock()
                        critic.rbf_omit_fake_fake = True
                        actual = score()
                        self.assertIs(routed.call_args.kwargs['omit_fake_fake'], True)
                        self.assertIs(routed.call_args.kwargs['omit_real_real'], omit_rr)
                        after = [(c.args[0], c.args[1]) for c in alpha.call_args_list]
                self.assertEqual(len(before), 2)
                self.assertEqual(len(before), len(after))
                for old, new in zip(before, after):
                    torch.testing.assert_close(old[0], new[0], atol=0, rtol=0)
                    torch.testing.assert_close(old[1], new[1], atol=0, rtol=0)
                torch.testing.assert_close(default[1], actual[1])
                self.assertTrue(bool((actual[0][actual[1]] > default[0][actual[1]]).all()))
                critic.rbf_alpha = self.alpha
                actual = score()
                for index, scale in ((0, 1.), (1, 2.)):
                    expected, active = self.expected(
                        token_features(clean) * scale,
                        token_features(draws[:1] if source == 'rows' else draws) * scale,
                        source=source, level=level, distance=distance,
                        omit_rr=omit_rr, omit_ff=True)
                    torch.testing.assert_close(actual[2][index], expected, atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(actual[1], active)
                self.assertLessEqual(max(backbone.calls), 2)
                self.assertTrue(all(not block._forward_hooks for block in backbone.blocks))

    def test_feature_mse_ignores_flags_non_rbf_rejects_and_constraints_remain(self):
        paired = ({0: torch.cat([self.real, self.fake[0]])}, 4, self.selected, [0], [1.])
        draws = ({0: torch.cat([self.real, self.fake.flatten(0, 1)])},
                 4, 3, 5, self.selected, [0], [1.])
        for helper, args in ((mmd_reward.summed_paired_reward, paired),
                             (mmd_reward.summed_multi_draw_reward, draws)):
            for omit_rr in (False, True):
                default = helper(*args, reward_type='feature_mse')
                actual = helper(*args, reward_type='feature_mse', omit_real_real=omit_rr,
                                omit_fake_fake=True, exclude_same_sentence=True)
                torch.testing.assert_close(default[0], actual[0], atol=0, rtol=0)
                torch.testing.assert_close(default[1], actual[1])
            for kernel in ('linear', 'pseudo_huber', 'cosine'):
                with self.assertRaisesRegex(ValueError, 'mmd_rbf_omit_fake_fake requires'):
                    helper(*args, kernel=kernel, omit_fake_fake=True)
        with self.assertRaisesRegex(ValueError, 'batch_size >= 2'):
            mmd_reward.paired_sequence_reward(
                self.real, self.fake[0], self.selected, kernel='rbf', level='sentence',
                omit_real_real=True, omit_fake_fake=True)
        k1 = ({0: torch.cat([self.real, self.fake[0]])}, 4, 1, 5, self.selected, [0], [1.])
        with self.assertRaisesRegex(ValueError, 'batch_size >= 2'):
            mmd_reward.summed_multi_draw_reward(
                *k1, kernel='rbf', exclude_same_sentence=True,
                omit_real_real=True, omit_fake_fake=True)
        with self.assertRaisesRegex(ValueError, 'requires.*draws'):
            mmd_reward.summed_paired_reward(
                *paired, kernel='rbf', exclude_same_sentence=True,
                omit_real_real=True, omit_fake_fake=True)
        zero = torch.zeros_like(self.selected)
        reward, active = mmd_reward.rbf_sequence_reward(
            self.real, self.fake[0], zero, omit_real_real=True, omit_fake_fake=True)
        torch.testing.assert_close(reward, torch.zeros(4))
        self.assertFalse(bool(active.any()))

    def test_reinforce_rf_only_values_loss_and_student_gradients(self):
        clean = torch.randint(0, 7, (4, 5))
        modes = [('rows', 'token', 1, False), ('rows', 'sentence', 1, False),
                 ('draws', 'token', 2, False), ('draws', 'sentence', 2, False),
                 ('draws', 'sentence', 1, False), ('draws', 'token', 2, True)]
        for adapter, (source, level, count, cross), distance, baseline in itertools.product(
                ('native', 'hf'), modes, ('squared_euclidean', 'pseudo_huber'), ('none', 'group')):
            with self.subTest(adapter=adapter, source=source, level=level, count=count,
                              cross=cross, distance=distance, baseline=baseline):
                group = 3
                samples = torch.randint(0, 7, (group * count, 4, 5))
                samples = torch.where(self.selected[None], samples, clean[None])
                critic, _ = make_critic(adapter, level=level)
                critic.rbf_alpha, critic.rbf_huber_delta = self.alpha, self.delta
                critic.rbf_distance = distance
                critic.rbf_omit_real_real = critic.rbf_omit_fake_fake = True
                critic.rbf_exclude_same_sentence = cross
                expected_rewards = []
                for g in range(group):
                    layer_rewards = []
                    for scale, weight in ((1., .4), (2., 1.3)):
                        value, active = self.expected(
                            token_features(clean) * scale,
                            token_features(samples[g * count:(g + 1) * count]) * scale,
                            source=source, level=level, distance=distance, cross=cross,
                            omit_rr=True, omit_ff=True)
                        layer_rewards.append(weight * value)
                    expected_rewards.append(sum(layer_rewards))
                rewards = torch.stack(expected_rewards, -1)
                advantage = rewards
                if baseline == 'group':
                    advantage = rewards - (rewards.sum(-1, keepdim=True) - rewards) / (group - 1)
                initial = torch.randn(4, 5, 7)
                logits = initial.clone().requires_grad_()
                expected_logits = initial.clone().requires_grad_()
                def logprobs(values):
                    logp = values.log_softmax(-1)
                    return torch.stack([logp.gather(-1, sample[..., None]).squeeze(-1)
                                        for sample in samples])
                actual_loss, stats = mmd_reward.mmd_reinforce_loss_from_samples(
                    logprobs(logits), clean, self.selected, samples, critic,
                    group_size=group, eos_id=0, baseline=baseline,
                    kernel_source=source, kernel_draws=count)
                joint_logp = (logprobs(expected_logits) * self.selected[None]).sum(-1)
                joint_logp = joint_logp.reshape(group, count, 4).sum(1).T
                # TinyGSM deliberately retains its historical sum over groups.
                expected_loss = -(advantage * joint_logp).sum() / (self.selected.sum() * count)
                actual_loss.backward()
                expected_loss.backward()
                torch.testing.assert_close(stats['mmd/reward'], rewards[active].mean(), atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual_loss, expected_loss, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(logits.grad, expected_logits.grad, atol=2e-6, rtol=2e-5)
                self.assertGreater(float(logits.grad.abs().sum()), 0.)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
