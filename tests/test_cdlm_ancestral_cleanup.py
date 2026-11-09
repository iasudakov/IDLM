"""Independent CPU checks for optional ancestral cleanup after S transitions."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import sdtt_eval as validation


class LinearNoise:
    eps = .001

    def __call__(self, time):
        return -(1 - self.eps), 1 - (1 - self.eps) * time


class Denoiser:
    device = torch.device('cpu')
    mask_index, num_tokens = 6, 9
    noise = LinearNoise()
    time_conditioning = False

    def __init__(self, predictor='ancestral_cache', conditioning=False):
        self.config = OmegaConf.create({'sampling': {'predictor': predictor, 'steps': 4},
                                       'eval': {'gsm8k_steps': None}})
        self.time_conditioning = conditioning
        self.seen = []
        self.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens))

    def prior_sample(self, *shape):
        return torch.full(shape, self.mask_index, dtype=torch.long)

    def _sigma_from_alphat(self, alpha):
        return -alpha.log()

    def forward(self, state, sigma):
        self.seen.append((state.clone(), sigma.clone()))
        positions = torch.arange(state.shape[1]).expand_as(state)
        context = state.clamp_max(self.mask_index - 1).sum(-1, keepdim=True)
        winner = (context + positions + (3 * sigma.reshape(-1, 1)).long()) % 5 + 1
        probabilities = torch.full((*state.shape, self.mask_index + 1), .02)
        probabilities.scatter_(-1, winner.unsqueeze(-1), .8)
        probabilities[..., self.mask_index] = 0.
        return (probabilities / probabilities.sum(-1, keepdim=True)).log()


@torch.no_grad()
def oracle(model, prompts, steps, *, final_denoise, epsilon=1e-5):
    """Write transitions directly, without calling any production eval helper."""
    state = torch.full((len(prompts), model.num_tokens), model.mask_index, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        state[row, :len(prompt)] = torch.tensor(prompt)
    grid = torch.linspace(1., epsilon, steps + 1)
    width = (1 - epsilon) / steps
    distribution, forwards = None, 0
    for index in range(steps):
        time = grid[index] * torch.ones(len(prompts), 1)
        alpha = model.noise(time)[1]
        next_alpha = (torch.ones_like(alpha) if not final_denoise and index + 1 == steps
                      else model.noise(time - width)[1])
        if distribution is None:
            distribution = model.forward(state, -alpha.log()).exp()
            forwards += 1
        probability = ((next_alpha - alpha) / (1 - alpha).clamp_min(1e-12)).clamp(0, 1)
        selected = (torch.rand(state.shape) < probability) & (state == model.mask_index)
        next_state = state.clone()
        next_state[selected] = distribution.argmax(-1)[selected]
        if (model.config.sampling.predictor == 'ancestral' or model.time_conditioning
                or not torch.equal(next_state, state)):
            distribution = None
        state = next_state
        for row, prompt in enumerate(prompts):
            state[row, :len(prompt)] = torch.tensor(prompt)
    if final_denoise:
        endpoint = torch.full((len(prompts), 1), epsilon)
        clean_tokens = model.forward(state, -model.noise(endpoint)[1].log()).argmax(-1)
        forwards += 1
        remaining = state == model.mask_index
        state[remaining] = clean_tokens[remaining]
    return state, forwards


class AncestralCleanupTest(unittest.TestCase):
    def test_independent_oracle_across_budgets_seeds_cache_and_conditioning(self):
        prompts = [[0, 2], [0, 3, 1]]
        for cleanup in (False, True):
            for seed in (1, 5, 37):
                for steps in (1, 2, 4, 8):
                    for predictor, conditioning in (('ancestral_cache', False),
                                                     ('ancestral', False),
                                                     ('ancestral_cache', True)):
                        with self.subTest(cleanup=cleanup, seed=seed, steps=steps,
                                          predictor=predictor, conditioning=conditioning):
                            expected_model = Denoiser(predictor, conditioning)
                            actual_model = Denoiser(predictor, conditioning)
                            torch.manual_seed(seed)
                            expected, expected_nfe = oracle(
                                expected_model, prompts, steps, final_denoise=cleanup)
                            expected_rng = torch.get_rng_state().clone()
                            torch.manual_seed(seed)
                            actual, info = validation.generate_greedy_samples(
                                actual_model, prompts, steps, final_denoise=cleanup)
                            self.assertTrue(torch.equal(actual, expected))
                            self.assertEqual(info['nfe'], expected_nfe)
                            self.assertEqual(info['nfe_per_row'].tolist(), [expected_nfe] * 2)
                            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                            self.assertEqual(len(actual_model.seen), expected_nfe)
                            for (actual_x, actual_sigma), (expected_x, expected_sigma) in zip(
                                    actual_model.seen, expected_model.seen):
                                self.assertTrue(torch.equal(actual_x, expected_x))
                                self.assertTrue(torch.equal(actual_sigma, expected_sigma))
                            self.assertFalse(bool((actual == actual_model.mask_index).any()))
                            for state in [x for x, _ in actual_model.seen] + [actual]:
                                for row, prompt in enumerate(prompts):
                                    self.assertEqual(state[row, :len(prompt)].tolist(), prompt)
                            if predictor == 'ancestral' or conditioning:
                                self.assertEqual(expected_nfe, steps + int(cleanup))

    def test_cleanup_predicts_remaining_masks_at_epsilon_and_is_counted_after_cache_skips(self):
        for predictor, conditioning, expected in (('ancestral_cache', False, 2),
                                                   ('ancestral', False, 5),
                                                   ('ancestral_cache', True, 5)):
            with self.subTest(predictor=predictor, conditioning=conditioning):
                model = Denoiser(predictor, conditioning)
                with patch.object(torch, 'rand', side_effect=lambda shape, **kwargs:
                                  torch.full(shape, .999999)) as random:
                    result, info = validation.generate_greedy_samples(
                        model, [[0, 2]], 4, final_denoise=True)
                self.assertEqual(random.call_count, 4)
                self.assertEqual(info['nfe'], expected)
                self.assertEqual(len(model.seen), expected)
                self.assertTrue(bool((model.seen[-1][0][0, 2:] == model.mask_index).all()))
                self.assertFalse(bool((result == model.mask_index).any()))
                endpoint_sigma = -model.noise(torch.full((1, 1), 1e-5))[1].log()
                self.assertTrue(torch.equal(model.seen[-1][1], endpoint_sigma))

    def test_cleanup_forward_is_unconditional_but_never_rewrites_already_revealed_tokens(self):
        model = Denoiser()
        with patch.object(torch, 'rand', side_effect=lambda shape, **kwargs: torch.zeros(shape)):
            result, info = validation.generate_greedy_samples(
                model, [[0, 2]], 4, final_denoise=True)
        # Every answer token was committed in the first transition. A separate
        # cleanup call still occurs, without overwriting that completed state.
        # The first reveal invalidates the cache; one evaluation of the fully
        # revealed state is then cached for the remaining scheduled transitions.
        self.assertEqual(info['nfe'], 3)
        self.assertFalse(bool((model.seen[-1][0] == model.mask_index).any()))
        self.assertTrue(torch.equal(result, model.seen[-1][0]))
        self.assertEqual(result[0, :2].tolist(), [0, 2])

    def test_both_modes_consume_exactly_s_position_draws_and_default_is_legacy(self):
        prompts, steps, seed = [[0, 2], [0, 1, 2]], 4, 1234
        torch.manual_seed(seed)
        for _ in range(steps):
            torch.rand((2, Denoiser.num_tokens))
        expected_rng = torch.get_rng_state().clone()
        for cleanup in (False, True):
            with self.subTest(cleanup=cleanup):
                torch.manual_seed(seed)
                validation.generate_greedy_samples(
                    Denoiser(), prompts, steps, final_denoise=cleanup)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        torch.manual_seed(seed)
        explicit, explicit_metadata = validation.generate_greedy_samples(
            Denoiser(), prompts, steps, final_denoise=False)
        torch.manual_seed(seed)
        default, default_metadata = validation.generate_greedy_samples(Denoiser(), prompts, steps)
        self.assertTrue(torch.equal(default, explicit))
        self.assertEqual(default_metadata['nfe'], explicit_metadata['nfe'])

    def test_evaluation_forwards_cleanup_flag_and_pools_measured_calls_per_example(self):
        model = Denoiser('ancestral')
        data = [{'input_ids': [0, 2], 'answer': 'correct'} for _ in range(3)]
        with patch.object(validation, 'generate_greedy_samples',
                          wraps=validation.generate_greedy_samples) as generate, \
                patch.object(validation, 'grade_sample', return_value=True):
            totals = validation._evaluate_greedy(
                model, data, SimpleNamespace(map=map), 2, 2, 5., final_denoise=True)
        self.assertEqual(totals, (3, 3, 9, 6))  # correct, count, row NFE, batch forwards
        self.assertEqual(len(generate.call_args_list), 2)
        self.assertTrue(all(call.kwargs['final_denoise'] for call in generate.call_args_list))


if __name__ == '__main__':
    unittest.main()
