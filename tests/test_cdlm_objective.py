import unittest

import torch

from cdlm_objective import (corrupt_ground_truth, denoising_loss,
                            distribution_loss, trajectory_pair)


class ObjectiveTest(unittest.TestCase):
    def test_forward_kl_is_author_soft_cross_entropy(self):
        logits = torch.tensor([[.2, -.4]], requires_grad=True)
        teacher = torch.tensor([[.75, .25]]).log().requires_grad_()
        predictions = logits.log_softmax(-1)
        loss = distribution_loss(predictions, teacher)
        expected = -(torch.tensor([[.75, .25]]) * predictions).sum()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_consistency_target_is_current_student_stop_gradient(self):
        earlier = torch.tensor([[1., 0., -1.]], requires_grad=True)
        later = torch.tensor([[-1., 0., 1.]], requires_grad=True)
        loss = distribution_loss(earlier.log_softmax(-1), later.log_softmax(-1),
                                 target_is_log_probs=True)
        loss.backward()
        self.assertIsNone(later.grad)
        self.assertGreater(float(earlier.grad.abs().sum()), 0)

    def test_reverse_kl_ablation_has_the_opposite_direction(self):
        student = torch.tensor([[.2, .8]]).log().requires_grad_()
        teacher = torch.tensor([[.7, .3]]).log()
        loss = distribution_loss(student, teacher, kind='reverse_kl')
        expected = (student.exp() * (student - teacher)).sum()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_empty_consistency_positions_are_safe_for_backward(self):
        logits = torch.empty((0, 4), requires_grad=True)
        loss = distribution_loss(logits, torch.empty_like(logits))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertIsNotNone(logits.grad)

    def test_pairs_preserve_prompts_and_keep_a_nonempty_consistency_span(self):
        final = torch.tensor([[0, 1, 2, 3, 4, 5]])
        order = torch.tensor([[-1, -1, 2, 0, 3, 1]])
        valid = order >= 0
        earlier, later, new, still, start, end = trajectory_pair(
            final, order, valid, 6, 2, starts=torch.tensor([1]))
        self.assertTrue(torch.equal(earlier, torch.tensor([[0, 1, 6, 3, 6, 6]])))
        self.assertTrue(torch.equal(later, torch.tensor([[0, 1, 6, 3, 6, 5]])))
        self.assertEqual(int(new.sum()), 1)
        self.assertEqual(int(still.sum()), 2)
        _, _, _, full_end_still, _, _ = trajectory_pair(
            final, order, valid, 6, 32, starts=torch.tensor([1]))
        self.assertFalse(bool(full_end_still.any()))

    def test_ground_truth_corruption_never_masks_the_prompt(self):
        original = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        valid = torch.tensor([[False, False, True, True], [False, False, False, True]])
        for seed in range(10):
            torch.manual_seed(seed)
            noisy, selected, probabilities = corrupt_ground_truth(original, valid, 4)
            self.assertTrue(torch.equal(noisy[~valid], original[~valid]))
            self.assertTrue(bool(selected.any(-1).all()))
            self.assertTrue(bool(((probabilities >= .001) & (probabilities <= 1)).all()))

    def test_denoising_uses_inverse_mask_probability_and_answer_length(self):
        logs = torch.tensor([[[.2, .8], [.5, .5], [.75, .25], [.8, .2]]]).log()
        original = torch.tensor([[0, 0, 0, 1]])
        selected = torch.tensor([[False, False, True, False]])
        valid = torch.tensor([[False, False, True, True]])
        loss = denoising_loss(logs, original, selected, torch.tensor([.25]), valid)
        self.assertAlmostEqual(float(loss), float(-logs[0, 2, 0] / .25 / 2), places=6)


if __name__ == '__main__':
    unittest.main()
