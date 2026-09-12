"""CPU references for token RBF rewards using only distinct fake sentences.

Run: OMP_NUM_THREADS=2 python -B scripts/test_rbf_cross_sentence.py
The defining statistic omits real-real pairs and averages every token pair
between different generated sentences. No models, downloads, or CUDA needed.
"""

import ast
from contextlib import nullcontext, redirect_stdout
import io
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mmd_rbf
import mmd_reward


def kernel(a, b, alpha, distance, delta):
    diff = a.double()[:, None] - b.double()[None]
    if distance == 'squared_euclidean':
        d = diff.square().sum(-1)
    else:
        d = (torch.sqrt(diff.square() + delta ** 2) - delta).sum(-1)
    return torch.exp(-alpha * d)


def reference(real, fake, selected, alpha, distance, delta):
    """Enumerate sentences first, then every selected token pair, in FP64."""
    count = fake.shape[1]
    values = []
    for x, draws, mask in zip(real, fake, selected):
        x, draws = x[mask], draws[:, mask]
        if not len(x):
            values.append(0.)
            continue
        within_fake = sum(kernel(draws[r], draws[s], alpha, distance, delta).mean()
                          for r in range(count) for s in range(count) if r != s)
        cross = sum(kernel(x, draw, alpha, distance, delta).mean() for draw in draws)
        values.append(float(within_fake / (count * (count - 1)) - 2 * cross / count))
    return torch.tensor(values)


def critic_getter():
    path = ROOT / 'algo.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == '_mmd_critic']
    assert len(nodes) == 1
    nodes[0].decorator_list = []
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['_mmd_critic']


class CrossSentenceRBFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        generator = torch.Generator().manual_seed(281)
        self.real = torch.randn(4, 5, 7, generator=generator)
        self.fake = torch.randn(4, 4, 5, 7, generator=generator)
        self.selected = torch.tensor([[1, 1, 0, 1, 1], [0, 1, 1, 0, 0],
                                      [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                                     dtype=torch.bool)
        self.alpha, self.delta = .13, .37

    def test_dense_reference_for_k2_k4_both_distances_and_ragged_tiles(self):
        for distance, count, tile in itertools.product(
                ('squared_euclidean', 'pseudo_huber'), (2, 4), (0, 1, 2, 3, 20)):
            with self.subTest(distance=distance, count=count, tile=tile):
                fake = self.fake[:, :count]
                expected = reference(self.real, fake, self.selected, self.alpha, distance, self.delta)
                value, active = mmd_rbf.rbf_mmd2_cross_sentence(
                    self.real, fake, self.selected, self.alpha, block_size=tile,
                    distance=distance, huber_delta=self.delta)
                torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-5)
                self.assertEqual(active.tolist(), [True, True, True, False])
                self.assertFalse(value.requires_grad)

    def test_one_position_is_active_and_real_real_term_is_absent(self):
        real = torch.zeros(2, 1, 3)
        fake = torch.zeros(2, 2, 1, 3)
        selected = torch.tensor([[True], [False]])
        for distance in ('squared_euclidean', 'pseudo_huber'):
            value, active = mmd_rbf.rbf_mmd2_cross_sentence(
                real, fake, selected, self.alpha, distance=distance, huber_delta=self.delta)
            # Identical empirical distributions score -1 because +k(real,real)
            # is intentionally omitted; this is not a full nonnegative MMD.
            torch.testing.assert_close(value, torch.tensor([-1., 0.]))
            self.assertEqual(active.tolist(), [True, False])

    def test_pair_identity_is_sentence_identity_not_token_position(self):
        real = torch.full((1, 2, 1), 1000.)
        selected = torch.ones(1, 2, dtype=torch.bool)
        # Only matching token positions agree across the two distinct draws.
        # Keeping i=j across draws is required to obtain 2/4, not zero.
        fake = torch.tensor([[[[0.], [100.]], [[0.], [100.]]]])
        value, _ = mmd_rbf.rbf_mmd2_cross_sentence(real, fake, selected, 1., block_size=1)
        torch.testing.assert_close(value, torch.tensor([.5]))
        # Different positions in the SAME draw are identical, but every pair
        # across draws is distant. Same-draw off-diagonals must also disappear.
        fake = torch.tensor([[[[0.], [0.]], [[100.], [100.]]]])
        value, _ = mmd_rbf.rbf_mmd2_cross_sentence(real, fake, selected, 1., block_size=1)
        torch.testing.assert_close(value, torch.zeros(1))

    def test_exact_enumeration_removes_correlated_token_fake_term_bias(self):
        # Each sentence has two perfectly correlated token features (Z,Z),
        # Z~Bernoulli(.5). The two generated sentences remain independent.
        assignments = list(itertools.product((0., 1.), repeat=2))
        fake = torch.tensor(assignments).reshape(4, 2, 1, 1).expand(-1, -1, 2, -1)
        real = torch.zeros(4, 2, 1)
        mask = torch.ones(4, 2, dtype=torch.bool)
        alpha = math.log(2.)  # k(0,1)=.5
        actual, _ = mmd_rbf.rbf_mmd2_cross_sentence(real, fake, mask, alpha)
        # Independent-sentence fake term=.75, real-fake term=.75.
        self.assertAlmostEqual(float(actual.mean()), -.75, places=6)
        legacy, _ = mmd_rbf.rbf_mmd2_unequal(
            real, fake.flatten(1, 2), mask, mask.repeat(1, 2), alpha)
        legacy_without_real = legacy - 1.
        self.assertAlmostEqual(float(legacy_without_real.mean()), -2 / 3, places=6)
        self.assertAlmostEqual(float((legacy_without_real - actual).mean()), 1 / 12, places=6)

    def test_no_real_real_kernel_is_evaluated_and_explicit_tiles_stay_small(self):
        real = self.real[:2] + 1000.
        fake, selected = self.fake[:2], self.selected[:2]
        for distance, helper_name in (('squared_euclidean', '_kernel_block'),
                                       ('pseudo_huber', '_pseudo_huber_kernel_block')):
            original = getattr(mmd_rbf, helper_name)
            calls = []

            def observe(a, b, *args, **kwargs):
                self.assertFalse(bool((a > 900.).all() and (b > 900.).all()),
                                 'real-real kernels must not be computed')
                self.assertLessEqual(a.shape[1], 2)
                self.assertLessEqual(b.shape[1], 2)
                calls.append((a.shape[1], b.shape[1]))
                return original(a, b, *args, **kwargs)

            with mock.patch.object(mmd_rbf, helper_name, side_effect=observe):
                value, _ = mmd_rbf.rbf_mmd2_cross_sentence(
                    real, fake, selected, self.alpha, block_size=2,
                    distance=distance, huber_delta=self.delta)
            self.assertGreater(len(calls), 1)
            torch.testing.assert_close(value, reference(real, fake, selected, self.alpha,
                                                        distance, self.delta), atol=2e-6, rtol=2e-5)

    def test_wrapper_and_layer_weights_use_cross_sentence_statistic(self):
        flattened = self.fake.permute(1, 0, 2, 3).flatten(0, 1)
        hidden = torch.cat([self.real, flattened])
        features = {0: hidden, 2: hidden * 1.7}
        for distance in ('squared_euclidean', 'pseudo_huber'):
            total, active, layers = mmd_reward.summed_multi_draw_reward(
                features, 4, 4, 5, self.selected, [0, 2], [.3, 1.2],
                kernel='rbf', level='token', alpha=self.alpha, block_size=2,
                distance=distance, huber_delta=self.delta, exclude_same_sentence=True)
            for index, scale in ((0, 1.), (2, 1.7)):
                expected = -reference(self.real * scale, self.fake * scale, self.selected,
                                      self.alpha, distance, self.delta)
                torch.testing.assert_close(layers[index], expected, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(total, .3 * layers[0] + 1.2 * layers[2])
            self.assertEqual(active.tolist(), [True, True, True, False])

    def test_default_wrapper_retains_legacy_token_pair_estimator(self):
        hidden = torch.cat([self.real, self.fake.permute(1, 0, 2, 3).flatten(0, 1)])
        for distance in ('squared_euclidean', 'pseudo_huber'):
            kwargs = dict(kernel='rbf', level='token', alpha=self.alpha,
                          distance=distance, huber_delta=self.delta)
            default = mmd_reward.summed_multi_draw_reward(
                {0: hidden}, 4, 4, 5, self.selected, [0], [1.], **kwargs)
            explicit = mmd_reward.summed_multi_draw_reward(
                {0: hidden}, 4, 4, 5, self.selected, [0], [1.],
                exclude_same_sentence=False, **kwargs)
            legacy, active = mmd_rbf.rbf_mmd2_unequal(
                self.real, self.fake.flatten(1, 2), self.selected, self.selected.repeat(1, 4),
                self.alpha, distance=distance, huber_delta=self.delta)
            torch.testing.assert_close(default[0], explicit[0], atol=0, rtol=0)
            torch.testing.assert_close(default[0], -legacy, atol=2e-6, rtol=2e-5)
            self.assertEqual(default[1].tolist(), active.tolist())
            self.assertEqual(active.tolist(), [True, True, False, False])

    def test_helper_and_wrapper_reject_invalid_draw_counts_and_combinations(self):
        for count in (0, 1):
            with self.assertRaises(ValueError):
                mmd_rbf.rbf_mmd2_cross_sentence(
                    self.real, self.fake[:, :count], self.selected, self.alpha)
        for changes in (dict(kernel='linear'), dict(level='sentence'), dict(draw_count=1)):
            settings = dict(kernel='rbf', level='token', draw_count=4)
            settings.update(changes)
            count = settings.pop('draw_count')
            hidden = torch.cat([self.real, self.fake[:, :count].permute(1, 0, 2, 3).flatten(0, 1)])
            with self.assertRaises(ValueError):
                mmd_reward.summed_multi_draw_reward(
                    {0: hidden}, 4, count, 5, self.selected, [0], [1.],
                    alpha=self.alpha, exclude_same_sentence=True, **settings)
        with self.assertRaises(ValueError):
            mmd_reward.paired_sequence_reward(
                self.real, self.fake[:, 0], self.selected, kernel='rbf',
                alpha=self.alpha, exclude_same_sentence=True)

    def test_native_and_external_critic_methods_forward_flag_and_reject_rows(self):
        clean = torch.tensor([[0, 1, 2], [2, 1, 0]])
        draws = torch.tensor([[[1, 1, 2], [2, 0, 0]],
                              [[2, 0, 2], [1, 2, 0]]])
        selected = torch.tensor([[True, True, False], [False, True, False]])
        table = torch.tensor([[0., 0.], [1., 2.], [-1., .5]])
        expected = -reference(table[clean], table[draws].permute(1, 0, 2, 3),
                              selected, self.alpha, 'pseudo_huber', self.delta)
        for critic_type in (mmd_reward.FrozenBackboneFeatures, mmd_reward.FrozenARFeatures):
            # Skip constructors which load checkpoints; execute the real scoring
            # methods on deterministic token embeddings supplied by a tiny stub.
            critic = critic_type.__new__(critic_type)
            critic.layer_index, critic.resolved_layers, critic.layer_weights = 0, [0], [1.]
            critic.kernel, critic.kernel_level = 'rbf', 'token'
            critic.rbf_alpha, critic.rbf_distance = self.alpha, 'pseudo_huber'
            critic.rbf_huber_delta, critic.rbf_exclude_same_sentence = self.delta, True
            critic.rbf_block_size, critic.kernel_batch_size = 1, 2
            critic.noise_level, critic.mask_index = 0., 3
            captured = {}

            def forward(tokens, sigma):
                captured[0] = table[tokens]

            critic.denoiser = forward
            critic.features = lambda tokens, eos_id, attention_mask=None: {0: table[tokens]}
            with mock.patch.object(mmd_reward, 'capture_layer_hidden',
                                   side_effect=lambda *args: nullcontext((captured, None))):
                actual, active = critic.multi_draw_reward(clean, draws, selected, eos_id=0)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                self.assertEqual(active.tolist(), [True, True])
                with self.assertRaises(ValueError):
                    critic.paired_reward(clean, draws[0], selected, eos_id=0)

    def test_hydra_config_reaches_critic_and_rejects_invalid_combinations_before_loading(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
            config = compose(config_name='config', overrides=[
                'algo=mdlm', 'adversarial_distill.mmd_kernel=rbf',
                'adversarial_distill.mmd_feature_source=diffusion',
                'adversarial_distill.mmd_kernel_source=draws',
                'adversarial_distill.mmd_kernel_level=token',
                'adversarial_distill.mmd_kernel_batch_size=2',
                'adversarial_distill.mmd_rbf_exclude_same_sentence=true'])
        getter = critic_getter()
        teacher = object()
        fake_critic = SimpleNamespace(layer_description='CPU fixture')
        with mock.patch.object(mmd_reward, 'FrozenBackboneFeatures', return_value=fake_critic) as constructor:
            with redirect_stdout(io.StringIO()):
                critic = getter(SimpleNamespace(config=config, teacher=teacher, mask_index=12))
        self.assertIs(constructor.call_args.args[0], teacher)
        self.assertTrue(critic.rbf_exclude_same_sentence)
        for overrides in (dict(mmd_kernel='linear'), dict(mmd_kernel_source='rows'),
                          dict(mmd_kernel_level='sentence'), dict(mmd_kernel_batch_size=1),
                          dict(mmd_reward_type='idlm')):
            altered = OmegaConf.merge(config, {'adversarial_distill': overrides})
            with mock.patch.object(mmd_reward, 'FrozenBackboneFeatures') as constructor:
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    getter(SimpleNamespace(config=altered, teacher=teacher, mask_index=12))
                constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
