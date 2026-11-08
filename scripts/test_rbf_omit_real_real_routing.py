"""CPU references for optional omission of only the RBF real-real term.

Run: OMP_NUM_THREADS=2 python -B scripts/test_rbf_omit_real_real_routing.py
"""
import itertools
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mmd_reward


def real_term(features, selected, alpha, distance, delta):
    """Independent FP64 real-real U-statistic, including inactive masks."""
    values = []
    for row, mask in zip(features, selected.bool()):
        points = row[mask].double()
        if len(points) < 2:
            values.append(0.)
            continue
        diff = points[:, None] - points[None]
        if distance == 'squared_euclidean':
            d = diff.square().sum(-1)
        else:
            d = (torch.sqrt(diff.square() + delta ** 2) - delta).sum(-1)
        k = torch.exp(-alpha * d)
        keep = ~torch.eye(len(points), dtype=torch.bool)
        values.append(float(k[keep].mean()))
    return torch.tensor(values)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.calls = []

    def forward(self, input_ids, sigma=None, attention_mask=None):
        self.calls.append(input_ids.shape[0])
        values = input_ids.float()
        hidden = torch.stack([values.sin(), values.cos(), values / 5.], -1)
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


class OmitRealRealRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.real = torch.randn(4, 5, 3)
        self.fake = torch.randn(3, 4, 5, 3)
        self.selected = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 0, 0, 0],
                                      [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                                     dtype=torch.bool)
        self.alpha, self.delta = .13, .3

    def test_rows_token_sentence_and_both_distances_match_removed_term(self):
        for level, rows, distance in itertools.product(
                ('token', 'sentence'), (2, 4), ('squared_euclidean', 'pseudo_huber')):
            with self.subTest(level=level, rows=rows, distance=distance):
                kwargs = dict(kernel='rbf', level=level, rows_of=rows,
                              alpha=self.alpha, distance=distance, huber_delta=self.delta)
                default, active = mmd_reward.paired_sequence_reward(
                    self.real, self.fake[0], self.selected, **kwargs)
                explicit, explicit_active = mmd_reward.paired_sequence_reward(
                    self.real, self.fake[0], self.selected, omit_real_real=False, **kwargs)
                actual, actual_active = mmd_reward.paired_sequence_reward(
                    self.real, self.fake[0], self.selected, omit_real_real=True, **kwargs)
                if level == 'sentence':
                    real, mask = mmd_reward.pool_over_selected(self.real, self.selected)
                    real = real.reshape(4 // rows, rows, -1)
                    mask = mask.reshape(4 // rows, rows)
                else:
                    real = self.real.reshape(4 // rows, rows * 5, -1)
                    mask = self.selected.reshape(4 // rows, rows * 5)
                removed = real_term(real, mask, self.alpha, distance, self.delta)
                removed = removed.repeat_interleave(rows)
                torch.testing.assert_close(default, explicit, atol=0, rtol=0)
                torch.testing.assert_close(actual, default + removed, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(active, actual_active)
                torch.testing.assert_close(active, explicit_active)

    def test_draws_token_sentence_k1_k3_and_weighted_layers(self):
        for level, count, distance in itertools.product(
                ('token', 'sentence'), (1, 3), ('squared_euclidean', 'pseudo_huber')):
            with self.subTest(level=level, count=count, distance=distance):
                hidden = torch.cat([self.real, self.fake[:count].flatten(0, 1)])
                features = {0: hidden, 2: hidden * 1.7}
                kwargs = dict(kernel='rbf', level=level, alpha=self.alpha,
                              distance=distance, huber_delta=self.delta, block_size=2)
                args = (features, 4, count, 5, self.selected, [0, 2], [.4, 1.3])
                default, active, default_layers = mmd_reward.summed_multi_draw_reward(*args, **kwargs)
                explicit = mmd_reward.summed_multi_draw_reward(*args, omit_real_real=False, **kwargs)
                actual, actual_active, actual_layers = mmd_reward.summed_multi_draw_reward(
                    *args, omit_real_real=True, **kwargs)
                torch.testing.assert_close(default, explicit[0], atol=0, rtol=0)
                for index, scale in ((0, 1.), (2, 1.7)):
                    if level == 'sentence':
                        removed = self.selected.any(-1).float()
                    else:
                        removed = real_term(self.real * scale, self.selected, self.alpha,
                                            distance, self.delta)
                    torch.testing.assert_close(actual_layers[index], default_layers[index] + removed,
                                               atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual, .4 * actual_layers[0] + 1.3 * actual_layers[2])
                torch.testing.assert_close(actual_active, active)
                self.assertEqual(active.tolist(), ([True, True, True, False] if level == 'sentence'
                                                  else [True, True, False, False]))

    def test_hf_and_native_adapters_route_to_real_helpers_with_hooks_and_auto_alpha(self):
        clean = torch.randint(0, 7, (4, 5))
        draws = torch.randint(0, 7, (2, 4, 5))
        for adapter, source, level in itertools.product(
                ('native', 'hf'), ('rows', 'draws'), ('token', 'sentence')):
            with self.subTest(adapter=adapter, source=source, level=level):
                critic, backbone = make_critic(adapter, level=level)
                helper_name = 'summed_paired_reward' if source == 'rows' else 'summed_multi_draw_reward'
                helper = getattr(mmd_reward, helper_name)
                def score():
                    if source == 'rows':
                        return critic.paired_reward(clean, draws[0], self.selected, 0,
                                                    return_per_layer=True)
                    return critic.multi_draw_reward(clean, draws, self.selected, 0,
                                                    return_per_layer=True)
                with mock.patch.object(mmd_reward, 'median_alpha', wraps=mmd_reward.median_alpha) as alpha:
                    with mock.patch.object(mmd_reward, helper_name, wraps=helper) as routed:
                        default = score()
                        self.assertFalse(routed.call_args.kwargs['omit_real_real'])
                        default_alphas = [(call.args[0].clone(), call.args[1].clone())
                                          for call in alpha.call_args_list]
                        alpha.reset_mock()
                        critic.rbf_omit_real_real = True
                        actual = score()
                        self.assertTrue(routed.call_args.kwargs['omit_real_real'])
                        omitted_alphas = [(call.args[0], call.args[1]) for call in alpha.call_args_list]
                self.assertEqual(len(default_alphas), 2)
                for expected, value in zip(default_alphas, omitted_alphas):
                    torch.testing.assert_close(expected[0], value[0], atol=0, rtol=0)
                    torch.testing.assert_close(expected[1], value[1], atol=0, rtol=0)
                torch.testing.assert_close(default[1], actual[1])
                self.assertTrue(bool((actual[0][actual[1]] > default[0][actual[1]]).all()))
                torch.testing.assert_close(actual[0], .4 * actual[2][0] + 1.3 * actual[2][1])
                self.assertLessEqual(max(backbone.calls), 2)
                self.assertTrue(all(not block._forward_hooks for block in backbone.blocks))

    def test_cross_sentence_result_is_unchanged_if_both_flags_are_true(self):
        hidden = torch.cat([self.real, self.fake.flatten(0, 1)])
        args = ({0: hidden}, 4, 3, 5, self.selected, [0], [1.])
        kwargs = dict(kernel='rbf', alpha=self.alpha, exclude_same_sentence=True)
        default = mmd_reward.summed_multi_draw_reward(*args, **kwargs)
        actual = mmd_reward.summed_multi_draw_reward(*args, omit_real_real=True, **kwargs)
        torch.testing.assert_close(actual[0], default[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], default[1])

    def test_feature_mse_ignores_flag_and_other_feature_kernels_reject(self):
        paired = ({0: torch.cat([self.real, self.fake[0]])}, 4, self.selected, [0], [1.])
        draws = ({0: torch.cat([self.real, self.fake.flatten(0, 1)])},
                 4, 3, 5, self.selected, [0], [1.])
        for helper, args in ((mmd_reward.summed_paired_reward, paired),
                             (mmd_reward.summed_multi_draw_reward, draws)):
            default = helper(*args, reward_type='feature_mse')
            actual = helper(*args, reward_type='feature_mse', omit_real_real=True)
            torch.testing.assert_close(default[0], actual[0], atol=0, rtol=0)
            torch.testing.assert_close(default[1], actual[1])
            with self.assertRaisesRegex(ValueError, 'mmd_rbf_omit_real_real requires'):
                helper(*args, kernel='linear', omit_real_real=True)

    def test_group_baseline_cancels_same_real_term_across_candidates(self):
        kwargs = dict(kernel='rbf', alpha=self.alpha)
        default, omitted = [], []
        for fake in self.fake:
            default.append(mmd_reward.paired_sequence_reward(
                self.real, fake, self.selected, **kwargs)[0])
            omitted.append(mmd_reward.paired_sequence_reward(
                self.real, fake, self.selected, omit_real_real=True, **kwargs)[0])
        first = mmd_reward.group_advantages(torch.stack(default, -1))
        second = mmd_reward.group_advantages(torch.stack(omitted, -1))
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)


    def test_group_reinforce_loss_and_student_gradient_are_unchanged_for_fixed_real_term(self):
        clean = torch.randint(0, 7, (4, 5))
        for source in ('rows', 'draws'):
            with self.subTest(source=source):
                group, draws = 3, 2
                count = group * (draws if source == 'draws' else 1)
                samples = torch.randint(0, 7, (count, 4, 5))
                samples = torch.where(self.selected.unsqueeze(0), samples, clean.unsqueeze(0))
                critic, _ = make_critic('native', rows=draws)
                critic.rbf_alpha = .2
                initial_logits = torch.randn(4, 5, 7)
                losses, gradients = [], []
                for omit in (False, True):
                    critic.rbf_omit_real_real = omit
                    logits = initial_logits.clone().requires_grad_()
                    logp = logits.log_softmax(-1)
                    sampled_logp = torch.stack([
                        logp.gather(-1, sample.unsqueeze(-1)).squeeze(-1)
                        for sample in samples])
                    loss, _ = mmd_reward.mmd_reinforce_loss_from_samples(
                        sampled_logp, clean, self.selected, samples, critic,
                        group_size=group, eos_id=0, baseline='group',
                        kernel_source=source, kernel_draws=draws)
                    loss.backward()
                    losses.append(loss.detach())
                    gradients.append(logits.grad)
                torch.testing.assert_close(losses[0], losses[1], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(gradients[0], gradients[1], atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
