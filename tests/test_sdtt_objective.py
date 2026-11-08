"""Numerical SDTT contracts, independent of Lightning and the GPU backbone."""

import unittest

import torch

from sdtt_objective import collect_teacher_targets, masked_reverse_kl, round_schedule


class SDTTObjectiveTests(unittest.TestCase):
    def test_reverse_kl_direction_and_gradient(self):
        logits = torch.tensor([[[0.2, -0.5, -3.0]]], requires_grad=True)
        teacher = torch.tensor([[[.05, .85, .10]]]).log()
        log_student = logits.log_softmax(-1)
        actual = masked_reverse_kl(log_student, teacher, torch.tensor([[True]]))
        expected = (log_student.exp() * (log_student - teacher)).sum()
        self.assertTrue(torch.allclose(actual, expected))
        forward_kl = (teacher.exp() * (teacher - log_student)).sum()
        self.assertFalse(torch.allclose(actual, forward_kl))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_only_masked_answer_positions_contribute(self):
        logits = torch.randn(2, 4, 5, requires_grad=True)
        teacher = torch.randn(2, 4, 5).log_softmax(-1)
        mask = torch.tensor([[False, True, False, False], [False, False, True, True]])
        loss = masked_reverse_kl(logits.log_softmax(-1), teacher, mask)
        loss.backward()
        self.assertEqual(float(logits.grad[~mask].abs().sum()), 0)
        self.assertGreater(float(logits.grad[mask].abs().sum()), 0)

    def test_empty_rank_still_has_differentiable_zero(self):
        logits = torch.randn(1, 2, 3, requires_grad=True)
        loss = masked_reverse_kl(logits.log_softmax(-1), torch.zeros_like(logits),
                                 torch.zeros(1, 2, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertIsNotNone(logits.grad)

    def test_targets_capture_first_reveal_and_last_remaining_prediction(self):
        # Position 0 is a fixed prompt, 1 first reveals at step 0, 2 at step 1,
        # and 3 remains masked. Their targets must be [unused, A, B, B].
        mask_id = 3
        xt = torch.tensor([[0, 3, 3, 3]])
        first = torch.tensor([.6, .3, .1, 1e-30]).log().expand(1, 4, 4).clone()
        last = torch.tensor([.1, .2, .7, 1e-30]).log().expand(1, 4, 4).clone()
        predictions, proposals, times = iter([first, last]), iter([
            torch.tensor([[2, 1, 3, 3]]), torch.tensor([[2, 2, 2, 3]])]), []

        def forward(x, sigma):
            times.append((x.clone(), sigma.clone()))
            return next(predictions)

        def noise(t):
            return -torch.ones_like(t), 1 - t * .999

        target = collect_teacher_targets(
            xt, torch.tensor([.8]), forward=forward, noise=noise, mask_index=mask_id,
            vocab_size=4, dt=.1, num_steps=2, sampling_eps=.001,
            sample_categorical=lambda p: next(proposals))
        self.assertTrue(torch.equal(target[0, 1], first[0, 1]))
        self.assertTrue(torch.equal(target[0, 2:], last[0, 2:]))
        self.assertEqual(int(times[1][0][0, 0]), 0)
        self.assertEqual(int(times[1][0][0, 1]), 1)
        # Released SDTT uses forward times .8 and .6 for dt=.1, two steps.
        self.assertAlmostEqual(float((1 - times[1][1].exp().reciprocal()) / .999), .6, places=5)
        self.assertFalse(target.requires_grad)

    def test_all_seven_rounds_and_budget_saturation(self):
        expected = [512, 256, 128, 64, 32, 16, 8]
        for stage, steps in enumerate(expected):
            self.assertEqual(round_schedule(1024, 2, 8, 2500, stage * 2500),
                             (stage, steps * 2, steps))
        self.assertEqual(round_schedule(1024, 2, 8, 2500, 20000), (6, 16, 8))
        self.assertEqual(round_schedule(1024, 2, 8, 10000, 69999), (6, 16, 8))
        with self.assertRaises(ValueError):
            round_schedule(1000, 2, 8, 2500, 0)


if __name__ == '__main__':
    unittest.main()
