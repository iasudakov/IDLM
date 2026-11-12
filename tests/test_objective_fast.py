"""CPU checks that mmd.objective_fast reproduces mmd.objective; no checkpoints or datasets required.

    CUDA_VISIBLE_DEVICES='' python -m tests.test_objective_fast
"""

import itertools
import unittest
from unittest.mock import patch

import torch

from mmd.objective import FeatureReward, reinforce_loss
from mmd.objective_fast import FastFeatureReward, ScoredHead, reinforce_loss_fast

VOCAB, LENGTH = 11, 7
MASKS = {
  'mixed': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]],
  'singletons': [[0, 0, 0, 1, 0, 0, 0]] * 6,
  'full': [[1] * LENGTH] * 6,
}


class TinyBackbone(torch.nn.Module):
  """Two mixing blocks, so a feature depends on every token of its sequence."""

  def __init__(self):
    super().__init__()
    self.embedding = torch.nn.Embedding(VOCAB, 5)
    self.position = torch.nn.Parameter(torch.randn(LENGTH, 5))
    self.blocks = torch.nn.ModuleList([torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)])
    self.sequences = 0

  def forward(self, tokens, sigma):
    self.sequences += tokens.shape[0]
    hidden = self.embedding(tokens) + self.position
    for block in self.blocks:
      hidden = block(hidden + hidden.mean(1, keepdim=True).tanh())
    return hidden


def scorers(exclude, **fast):
  torch.manual_seed(3)
  backbone = TinyBackbone()
  kwargs = dict(layer=1, alpha=.3, exclude_same_position=exclude)
  return (backbone, FeatureReward(backbone, block_size=2, feature_batch_size=4, **kwargs),
          FastFeatureReward(backbone, **kwargs, **fast))


class FastObjectiveTest(unittest.TestCase):
  def test_rewards_match_reference_per_candidate(self):
    for (name, mask), exclude, group, draws in itertools.product(MASKS.items(), (True, False), (1, 2, 4), (2, 3)):
      for fast in ({}, {'fast_feature_batch_size': 3, 'kernel_elements': 1}, {'fast_feature_batch_size': None},
                   {'share_repeated': False}):
        with self.subTest(mask=name, exclude=exclude, group=group, draws=draws, fast=fast):
          backbone, reference, scorer = scorers(exclude, **fast)
          selected = torch.tensor(mask, dtype=torch.bool)
          torch.manual_seed(11)
          x0 = torch.randint(VOCAB, selected.shape)
          sampled = torch.randint(VOCAB, (group * draws, *selected.shape))
          samples = torch.where(selected[None], sampled, x0[None])
          expected = [reference(x0, samples[g * draws:(g + 1) * draws], selected, return_paper=True)
                      for g in range(group)]
          backbone.sequences = 0
          reward, active, paper = scorer.score_candidates(x0, samples, selected, group, return_paper=True)
          # Each distinct sequence of a row once, and no row without a scored position.
          rows = selected.any(-1)
          count = lambda which: sum(  # noqa: E731
            len(torch.unique(torch.cat([x0[i, None], samples[:, i]]), dim=0)) if scorer.share_repeated
            else 1 + group * draws for i in which.nonzero().flatten())
          distinct = count(rows)
          self.assertEqual(backbone.sequences, distinct)
          self.assertLessEqual(distinct, int(rows.sum()) * (1 + group * draws))
          self.assertEqual((scorer.sequences_scored, scorer.sequences_encoded),
                           (int(rows.sum()) * (1 + group * draws), distinct))
          torch.testing.assert_close(reward, torch.stack([e[0] for e in expected], -1), atol=2e-6, rtol=1e-5)
          torch.testing.assert_close(paper, torch.stack([e[2] for e in expected], -1), atol=2e-6, rtol=1e-5)
          self.assertTrue(torch.equal(active, expected[0][1]))
          backbone.sequences = 0
          plain = scorer.score_candidates(x0, samples, selected, group)
          # Without Eq. 6 logging only rows with a reward are encoded.
          self.assertEqual(backbone.sequences, count(active))
          torch.testing.assert_close(plain[0], reward, atol=2e-6, rtol=1e-5)
          self.assertTrue(torch.equal(plain[1], active))

  def test_repeated_sequences_are_encoded_once(self):
    for exclude in (True, False):
      backbone, reference, scorer = scorers(exclude)
      selected = torch.tensor(MASKS['mixed'], dtype=torch.bool)
      torch.manual_seed(17)
      x0 = torch.randint(VOCAB, selected.shape)
      # Two possible tokens per position, one of them the real one: many repeats, some equal to x0.
      sampled = torch.where(torch.rand(8, *selected.shape) < .7, x0[None], (x0[None] + 1) % VOCAB)
      samples = torch.where(selected[None], sampled, x0[None])
      samples[:, 5] = samples[0, 5]  # a row whose samples are all identical
      samples[:2, 4] = x0[4]         # a candidate that reproduces the real sequence
      expected = [reference(x0, samples[g * 2:(g + 1) * 2], selected, return_paper=True) for g in range(4)]
      backbone.sequences = 0
      reward, active, paper = scorer.score_candidates(x0, samples, selected, 4, return_paper=True)
      self.assertLess(backbone.sequences, int(selected.any(-1).sum()) * 9 // 2)
      torch.testing.assert_close(reward, torch.stack([e[0] for e in expected], -1), atol=2e-6, rtol=1e-5)
      torch.testing.assert_close(paper, torch.stack([e[2] for e in expected], -1), atol=2e-6, rtol=1e-5)

  def test_hash_collision_disables_sharing(self):
    backbone, reference, scorer = scorers(True)
    selected = torch.tensor(MASKS['mixed'], dtype=torch.bool)
    torch.manual_seed(19)
    x0 = torch.randint(VOCAB, selected.shape)
    samples = torch.where(selected[None], torch.randint(VOCAB, (8, *selected.shape)), x0[None])
    expected = [reference(x0, samples[g * 2:(g + 1) * 2], selected, return_paper=True) for g in range(4)]
    original = torch.unique

    def colliding(key, return_inverse=False, **kwargs):  # every sequence falls into one hash group
      return original(torch.zeros_like(key), return_inverse=return_inverse, **kwargs)

    backbone.sequences = 0
    with patch('torch.unique', colliding):
      reward, _, paper = scorer.score_candidates(x0, samples, selected, 4, return_paper=True)
    self.assertEqual(backbone.sequences, int(selected.any(-1).sum()) * 9)
    torch.testing.assert_close(reward, torch.stack([e[0] for e in expected], -1), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(paper, torch.stack([e[2] for e in expected], -1), atol=2e-6, rtol=1e-5)

  def test_loss_gradient_and_statistics_match_reference(self):
    for (name, mask), exclude, (group, draws, average), log_paper in itertools.product(
        MASKS.items(), (True, False), ((1, 2, True), (4, 3, True), (4, 2, False)), (False, True)):
      with self.subTest(mask=name, exclude=exclude, group=group, draws=draws, log_paper=log_paper):
        _, reference, scorer = scorers(exclude)
        selected = torch.tensor(mask, dtype=torch.bool)
        torch.manual_seed(7)
        logits = torch.randn(*selected.shape, VOCAB)
        x0 = torch.randint(VOCAB, selected.shape)
        fixed = torch.randint(VOCAB, (selected.numel(), group * draws))
        kwargs = dict(group_size=group, draws_per_candidate=draws, average_candidates=average, log_paper=log_paper)
        a = logits.clone().requires_grad_()
        with patch('torch.multinomial', return_value=fixed):
          loss, stats = reinforce_loss(a.log_softmax(-1).exp(), x0, selected, reference, **kwargs)
        loss.backward()
        b = logits.clone().requires_grad_()
        fast_loss, fast_stats = reinforce_loss_fast(b.log_softmax(-1), x0, selected, scorer,
                                                    draws=fixed.view(*selected.shape, -1), **kwargs)
        fast_loss.backward()
        # Advantages are differences of float32 rewards, so agreement is absolute (rewards are O(1)).
        torch.testing.assert_close(fast_loss, loss, atol=1e-6, rtol=0)
        torch.testing.assert_close(b.grad, a.grad, atol=1e-6, rtol=0)
        self.assertEqual(stats.keys(), fast_stats.keys())
        for key in stats:
          torch.testing.assert_close(fast_stats[key].float(), stats[key].float(), atol=2e-6, rtol=1e-5)
          self.assertFalse(fast_stats[key].requires_grad)

  def test_loss_assembly_is_exact_given_rewards(self):
    """With one reward function for both paths, loss and gradient agree to rounding of the log-probabilities."""
    def reward_of(x0, samples, selected):  # (B,N,L) -> (N,), any deterministic function of the candidate
      return (samples.double().mul(torch.arange(1, LENGTH + 1)).sum((0, 2)) * .37).sin().float()

    class Reference:
      def __call__(self, x0, candidate, selected, return_paper=False):
        return reward_of(x0, candidate, selected), selected.sum(-1) >= 2

    class Fast:
      def score_candidates(self, x0, samples, selected, group_size, return_paper=False):
        candidates = samples.reshape(group_size, -1, *x0.shape)
        return torch.stack([reward_of(x0, c, selected) for c in candidates], -1), selected.sum(-1) >= 2

    for (name, mask), (group, draws, average) in itertools.product(
        MASKS.items(), ((1, 2, True), (4, 3, True), (8, 2, False))):
      with self.subTest(mask=name, group=group, draws=draws):
        selected = torch.tensor(mask, dtype=torch.bool)
        torch.manual_seed(13)
        logits = torch.randn(*selected.shape, VOCAB, dtype=torch.double)
        x0 = torch.randint(VOCAB, selected.shape)
        fixed = torch.randint(VOCAB, (selected.numel(), group * draws))
        kwargs = dict(group_size=group, draws_per_candidate=draws, average_candidates=average)
        a = logits.clone().requires_grad_()
        with patch('torch.multinomial', return_value=fixed):
          loss, stats = reinforce_loss(a.log_softmax(-1).exp(), x0, selected, Reference(), **kwargs)
        loss.backward()
        b = logits.clone().requires_grad_()
        fast_loss, fast_stats = reinforce_loss_fast(b.log_softmax(-1), x0, selected, Fast(),
                                                    draws=fixed.view(*selected.shape, -1), **kwargs)
        fast_loss.backward()
        torch.testing.assert_close(fast_loss, loss, atol=1e-13, rtol=1e-10)
        torch.testing.assert_close(b.grad, a.grad, atol=1e-13, rtol=1e-10)
        for key in stats:
          torch.testing.assert_close(fast_stats[key].float(), stats[key].float(), atol=0, rtol=1e-6)

  def test_scored_rows_give_the_same_loss_and_gradient(self):
    for (name, mask), (group, draws, average) in itertools.product(
        MASKS.items(), ((1, 2, True), (4, 3, True), (4, 2, False))):
      with self.subTest(mask=name, group=group, draws=draws):
        _, _, scorer = scorers(True)
        selected = torch.tensor(mask, dtype=torch.bool)
        torch.manual_seed(23)
        logits = torch.randn(*selected.shape, VOCAB, dtype=torch.double)
        x0 = torch.randint(VOCAB, selected.shape)
        fixed = torch.randint(VOCAB, (*selected.shape, group * draws))
        kwargs = dict(group_size=group, draws_per_candidate=draws, average_candidates=average, log_paper=True,
                      draws=fixed)
        a = logits.clone().requires_grad_()
        loss, stats = reinforce_loss_fast(a.log_softmax(-1), x0, selected, scorer, **kwargs)
        loss.backward()
        b = logits.clone().requires_grad_()
        scored_loss, scored_stats = reinforce_loss_fast(b.log_softmax(-1)[selected], x0, selected, scorer,
                                                        scored=True, **kwargs)
        scored_loss.backward()
        torch.testing.assert_close(scored_loss, loss, atol=1e-14, rtol=1e-10)
        torch.testing.assert_close(b.grad, a.grad, atol=1e-14, rtol=1e-10)
        for key in stats:
          torch.testing.assert_close(scored_stats[key], stats[key], atol=0, rtol=0)
    with self.assertRaises(ValueError):
      reinforce_loss_fast(torch.zeros(3, VOCAB), x0, selected, scorer, scored=True)

  def test_scored_head_returns_the_rows_of_the_full_forward(self):
    class Final(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.modulation, self.linear = torch.nn.Linear(3, 10), torch.nn.Linear(5, VOCAB)

      def forward(self, x, c):
        shift, scale = self.modulation(c)[:, None].chunk(2, dim=2)
        return self.linear(torch.nn.functional.layer_norm(x, [5]) * (1 + scale) + shift)

    class Backbone(torch.nn.Module):
      def __init__(self, keyword):
        super().__init__()
        self.embedding, self.condition = torch.nn.Embedding(VOCAB, 5), torch.nn.Linear(1, 3)
        self.output_layer, self.keyword = Final(), keyword

      def forward(self, tokens, sigma):
        hidden = self.embedding(tokens)
        hidden = hidden + hidden.mean(1, keepdim=True)
        c = self.condition(sigma[:, None])
        return self.output_layer(hidden, c=c) if self.keyword else self.output_layer(hidden, c)

    class Wrapper(torch.nn.Module):  # the Hugging Face model keeps the transformer in .backbone
      def __init__(self):
        super().__init__()
        self.backbone = Backbone(keyword=False)

      def forward(self, tokens, sigma):
        return self.backbone(tokens, sigma)

    torch.manual_seed(29)
    for backbone in (Backbone(keyword=True), Wrapper()):
      head = ScoredHead(backbone)
      tokens, sigma = torch.randint(VOCAB, (6, LENGTH)), torch.rand(6)
      for mask in (*MASKS.values(), [[0] * LENGTH] * 6):
        selected = torch.tensor(mask, dtype=torch.bool)
        full = backbone(tokens, sigma)
        rows = head.logits(tokens, sigma, selected.nonzero(as_tuple=True))
        self.assertEqual(rows.shape, (int(selected.sum()), VOCAB))
        torch.testing.assert_close(rows, full[selected], atol=1e-6, rtol=1e-6)
        # The restriction is off again afterwards, and gradients agree.
        self.assertEqual(backbone(tokens, sigma).shape, full.shape)
        weights = torch.randn(int(selected.sum()), VOCAB)
        expected = torch.autograd.grad((full[selected] * weights).sum(), list(backbone.parameters()), allow_unused=True)
        actual = torch.autograd.grad((rows * weights).sum(), list(backbone.parameters()), allow_unused=True)
        for e, g in zip(expected, actual):
          if e is not None:
            torch.testing.assert_close(g, e, atol=1e-5, rtol=1e-5)

  def test_empty_mask(self):
    _, _, scorer = scorers(True)
    logits = torch.randn(4, LENGTH, VOCAB, requires_grad=True)
    loss, stats = reinforce_loss_fast(logits.log_softmax(-1), torch.zeros(4, LENGTH, dtype=torch.long),
                                      torch.zeros(4, LENGTH, dtype=torch.bool), scorer, log_paper=True)
    loss.backward()
    self.assertEqual(float(loss), 0.)
    self.assertEqual(float(logits.grad.abs().sum()), 0.)
    self.assertEqual({k: float(v) for k, v in stats.items()},
                     dict.fromkeys(('mmd/reward', 'mmd/advantage_absmean', 'reward_count',
                                    'mmd/paper_surrogate', 'paper_count'), 0.))

  def test_internal_sampling_follows_the_model_at_scored_positions_only(self):
    torch.manual_seed(5)
    selected = torch.tensor([[1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1]], dtype=torch.bool)
    x0 = torch.randint(VOCAB, selected.shape)
    log_probs = torch.randn(*selected.shape, VOCAB).log_softmax(-1)
    seen = []

    class Recorder:
      def score_candidates(self, x0, samples, selected, group_size, return_paper=False):
        seen.append(samples.clone())
        return torch.zeros(x0.shape[0], group_size), selected.sum(-1) >= 2

    for _ in range(40):
      reinforce_loss_fast(log_probs, x0, selected, Recorder(), group_size=50, draws_per_candidate=2)
    samples = torch.cat(seen)  # (4000, N, L)
    self.assertTrue(torch.equal(samples[:, ~selected], x0[~selected].expand(len(samples), -1)))
    frequency = torch.stack([(samples == v).float().mean(0) for v in range(VOCAB)], -1)
    self.assertLess(float((frequency - log_probs.exp())[selected].abs().max()), .03)


if __name__ == '__main__':
  torch.set_num_threads(1)
  unittest.main()
