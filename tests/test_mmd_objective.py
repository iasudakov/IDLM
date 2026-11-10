"""CPU checks against the historical TinyGSM paired-token MMD formula."""

import unittest
from unittest.mock import patch

import torch

from mmd_objective import (FeatureReward, paired_rbf_reward, reinforce_loss,
                           reinforce_loss_from_samples)


def dense_reference(clean, generated, selected, alpha):
    """Independent explicit token-pair U-statistic, including the real constant."""
    result = []
    for real, fake, mask in zip(clean.float(), generated.float(), selected.bool()):
        real, fake = real[mask], fake[mask]
        count = len(real)
        if count < 2:
            result.append(clean.new_tensor(0., dtype=torch.float32))
            continue
        within_real = within_fake = cross = real.new_zeros(())
        for i in range(count):
            for j in range(count):
                if i != j:
                    within_real = within_real + torch.exp(-alpha * ((real[i] - real[j]) ** 2).sum())
                    within_fake = within_fake + torch.exp(-alpha * ((fake[i] - fake[j]) ** 2).sum())
                cross = cross + torch.exp(-alpha * ((real[i] - fake[j]) ** 2).sum())
        result.append(2 * cross / (count * count)
                      - (within_real + within_fake) / (count * (count - 1)))
    return torch.stack(result)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        self.batch_sizes = []
        self.reached_output = 0

    def forward(self, tokens, sigma):
        self.batch_sizes.append(len(tokens))
        assert sigma.dtype == torch.float32 and not sigma.any()
        hidden = self.embedding(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        self.reached_output += 1
        return hidden


class PairedRBFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_dense_formula_and_tiling_with_masked_outliers(self):
        rng = torch.Generator().manual_seed(21)
        real = torch.randn(4, 7, 5, generator=rng, requires_grad=True)
        fake = torch.randn(4, 7, 5, generator=rng, requires_grad=True)
        selected = torch.tensor([[1, 1, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0]])
        expected = dense_reference(real.detach(), fake.detach(), selected, .31)
        for size in (0, 1, 2, 3, 20):
            reward, active = paired_rbf_reward(real, fake, selected, alpha=.31, block_size=size)
            torch.testing.assert_close(reward, expected, atol=3e-7, rtol=2e-6)
            self.assertEqual(active.tolist(), [True, False, False, True])
            self.assertFalse(reward.requires_grad)
        mutated = real.detach().clone()
        mutated[~selected.bool()] = 9000
        reward, _ = paired_rbf_reward(mutated, fake, selected, alpha=.31)
        torch.testing.assert_close(reward, expected, atol=3e-7, rtol=2e-6)

    def test_negative_mmd_is_preserved_and_same_sequence_pairs_are_used(self):
        real = torch.tensor([[[0.], [2.]]])
        reward, active = paired_rbf_reward(real, real, torch.ones(1, 2, dtype=torch.bool), alpha=1.)
        # Exact real=fake U-statistic is negative; clipping would erase this reward.
        torch.testing.assert_close(reward, 1 - torch.exp(torch.tensor([-4.])))
        self.assertTrue(active.item())
        self.assertGreater(reward.item(), 0)

    def test_fp32_reward_inside_bf16_autocast(self):
        torch.manual_seed(1)
        real, fake = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
        selected = torch.ones(2, 4, dtype=torch.bool)
        expected, _ = paired_rbf_reward(real, fake, selected, alpha=.2)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            actual, _ = paired_rbf_reward(real, fake, selected, alpha=.2)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_invalid_kernel_arguments_fail(self):
        features = torch.zeros(1, 2, 1)
        selected = torch.ones(1, 2, dtype=torch.bool)
        for kwargs in ({'alpha': 0}, {'alpha': float('nan')}, {'block_size': -1}, {'block_size': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                paired_rbf_reward(features, features, selected, **kwargs)


class ReinforceTests(unittest.TestCase):
    def _fixture(self, dtype=torch.float32):
        logits = torch.tensor([[[.5, -.2, .1], [.2, .4, -.3], [.3, .1, -.2]],
                               [[.2, .5, -.4], [-.2, .4, .1], [.2, -.5, .3]]],
                              dtype=dtype, requires_grad=True)
        target = torch.tensor([[0, 1, 2], [2, 1, 0]])
        selected = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        draws = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1], [1, 0, 2, 2, 1, 0, 0, 2],
                              [0, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 1, 2, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2]])
        lookup = torch.tensor([[0., 0.], [1., 2.], [-2., .5]])
        return logits, target, selected, draws, lookup

    def test_loss_and_gradient_match_explicit_historical_estimator(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits, x0, selected, fixed, lookup = self._fixture(dtype)
                probabilities = logits.softmax(-1)
                seen = []

                def reward_fn(target, draws, mask):
                    seen.append(draws.clone())
                    return paired_rbf_reward(lookup[target], lookup[draws[0]], mask, alpha=.3)

                with patch('torch.multinomial', return_value=fixed) as sample:
                    actual, stats = reinforce_loss(probabilities, x0, selected, reward_fn)
                actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
                sample.assert_called_once()
                self.assertEqual(len(seen), 8)
                samples = torch.where(selected[None], fixed.view(2, 3, 8).permute(2, 0, 1), x0[None])
                self.assertTrue(torch.equal(torch.cat(seen), samples))
                rewards = torch.stack([dense_reference(lookup[x0], lookup[s], selected, .3)
                                       for s in samples], -1)
                advantages = rewards - (rewards.sum(-1, keepdim=True) - rewards) / 7
                advantages[1] = 0  # only one selected token: no pair estimator
                expected = probabilities.new_zeros(())
                logp = probabilities.clamp_min(1e-12).log()
                for g in range(8):
                    rows = logp.gather(-1, samples[g, ..., None]).squeeze(-1)
                    expected = expected + (advantages[:, g] * (rows * selected.to(dtype)).sum(1)).sum()
                expected = -expected / selected.sum().to(expected.dtype)
                expected_gradient = torch.autograd.grad(expected, logits)[0]
                tolerance = 3e-3 if dtype == torch.bfloat16 else 4e-7
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance)
                self.assertEqual(stats['reward_count'].item(), 1)
                self.assertTrue(torch.equal(actual_gradient[1], torch.zeros_like(actual_gradient[1])))
                self.assertTrue(torch.equal(actual_gradient[0, 2], torch.zeros_like(actual_gradient[0, 2])))

    def test_historical_loss_sums_eight_candidates(self):
        logits, x0, selected, fixed, lookup = self._fixture()
        reward = lambda real, fake, mask: paired_rbf_reward(lookup[real], lookup[fake[0]], mask, alpha=.3)
        with patch('torch.multinomial', return_value=fixed):
            summed, _ = reinforce_loss(logits.softmax(-1), x0, selected, reward)
            averaged, _ = reinforce_loss(logits.softmax(-1), x0, selected, reward, average_candidates=True)
        torch.testing.assert_close(summed, averaged * 8, atol=0, rtol=0)

    def test_empty_mask_returns_differentiable_zero_without_sampling(self):
        logits, x0, selected, _, _ = self._fixture()
        with patch('torch.multinomial') as sample:
            loss, stats = reinforce_loss(logits.softmax(-1), x0, torch.zeros_like(selected), None)
        sample.assert_not_called()
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        self.assertEqual(stats['reward_count'].item(), 0)

    def test_two_draw_recipe_is_rejected(self):
        logits, x0, selected, _, _ = self._fixture()
        with self.assertRaisesRegex(ValueError, 'different estimator'):
            reinforce_loss(logits.softmax(-1), x0, selected, None, draws_per_candidate=2)

    def test_checkpointed_eight_row_chunks_keep_full_batch_loss_and_gradients(self):
        from torch.utils.checkpoint import checkpoint
        torch.manual_seed(194)
        rows, length, vocabulary, groups = 19, 4, 5, 8
        inputs = torch.randn(rows, length, 3)
        targets = torch.randint(vocabulary, (rows, length))
        selected = torch.arange(length)[None] < (torch.arange(rows) % 5)[:, None]
        # A full physical chunk is inactive, so averaging chunk-level losses
        # would change both normalization and the contribution of active rows.
        selected[:8] = False
        fixed = torch.randint(vocabulary, (rows, length, groups))
        lookup = torch.randn(vocabulary, 6)
        base = torch.nn.Linear(3, vocabulary)
        chunked = torch.nn.Linear(3, vocabulary)
        chunked.load_state_dict(base.state_dict())
        rewards_called = []

        def reward_fn(real, fake, mask):
            rewards_called.append(real.shape[0])
            return paired_rbf_reward(lookup[real], lookup[fake[0]], mask, alpha=.13)

        with patch('torch.multinomial', return_value=fixed.reshape(-1, groups)):
            expected, expected_stats = reinforce_loss(base(inputs).softmax(-1), targets, selected, reward_fn)
        expected.backward()
        rewards_called.clear()
        chunks_called, score_chunks, sample_chunks = [], [], []

        def sample_chunk(features, original, mask, draws):
            chunks_called.append(len(features))
            probabilities = chunked(features).softmax(-1)
            samples = torch.where(mask[None], draws.permute(2, 0, 1), original[None])
            sampled_logp = probabilities.clamp_min(1e-12).log().gather(
                2, samples.permute(1, 2, 0)).permute(2, 0, 1)
            return sampled_logp, samples

        for start in range(0, rows, 8):
            scores, samples = checkpoint(sample_chunk, inputs[start:start + 8],
                targets[start:start + 8], selected[start:start + 8], fixed[start:start + 8],
                use_reentrant=False, preserve_rng_state=True)
            score_chunks.append(scores)
            sample_chunks.append(samples)
        actual, stats = reinforce_loss_from_samples(torch.cat(score_chunks, dim=1), targets,
            selected, torch.cat(sample_chunks, dim=1), reward_fn)
        self.assertEqual(chunks_called, [8, 8, 3])
        self.assertEqual(rewards_called, [19] * groups)
        actual.backward()
        self.assertEqual(chunks_called, [8, 8, 3, 3, 8, 8])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-6)
        for left, right in zip(chunked.parameters(), base.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=3e-7, rtol=2e-6)
        for key in stats:
            torch.testing.assert_close(stats[key], expected_stats[key], atol=0, rtol=0)

    def test_precomputed_samples_cannot_modify_unselected_context(self):
        logits, targets, selected, _, _ = self._fixture()
        samples = targets[None].expand(8, -1, -1).clone()
        samples[0, 0, 2] = (samples[0, 0, 2] + 1) % 3
        scores = logits.new_zeros(samples.shape, requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'Unselected'):
            reinforce_loss_from_samples(scores, targets, selected, samples, None)

    def test_precomputed_empty_mask_remains_differentiable(self):
        logits, targets, selected, _, _ = self._fixture()
        samples = targets[None].expand(8, -1, -1).clone()
        scores = logits.new_zeros(samples.shape, requires_grad=True)
        loss, stats = reinforce_loss_from_samples(scores, targets, torch.zeros_like(selected), samples, None)
        loss.backward()
        self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))
        self.assertEqual(stats['reward_count'].item(), 0)


class FrozenFeatureTests(unittest.TestCase):
    def test_feature_chunks_are_frozen_and_stop_at_selected_layer(self):
        torch.manual_seed(31)
        backbone = TinyBackbone()
        reward = FeatureReward(backbone, layer=1, alpha=.2, feature_batch_size=3)
        x0 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        fake = torch.tensor([[[6, 5, 4], [3, 2, 1]]])
        selected = torch.ones_like(x0, dtype=torch.bool)
        with torch.no_grad():
            hidden = backbone.blocks[1](backbone.blocks[0](backbone.embedding(torch.cat([x0, fake[0]]))))
            expected = dense_reference(hidden[:2], hidden[2:], selected, .2)
        torch.manual_seed(71)
        torch.rand(x0.shape)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(71)
        actual, _ = reward(x0, fake, selected)
        torch.testing.assert_close(actual, expected, atol=4e-7, rtol=2e-5)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(backbone.batch_sizes, [3, 1])
        self.assertEqual(backbone.reached_output, 0)
        self.assertFalse(backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))
        self.assertEqual(len(backbone.blocks[1]._forward_hooks), 0)

    def test_rejects_other_estimator_or_invalid_feature_layer(self):
        for kwargs in ({'kernel': 'token_rbf'}, {'layer': -1}, {'feature_batch_size': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FeatureReward(TinyBackbone(), **kwargs)


if __name__ == '__main__':
    unittest.main()
