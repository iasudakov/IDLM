"""CPU oracle, statistics, seed isolation and pooled ancestral sweep checks."""

from contextlib import ExitStack
import math
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import cdlm_ancestral_sweep as ancestral
import sdtt_eval
from test_cdlm_ancestral import Model, reference_sflm


def examples():
    return [{'input_ids': [0, 2] if index % 2 == 0 else [0, 3, 1],
             'answer': str(index % 3), 'question': str(index)} for index in range(5)]


def model():
    value = Model()
    value.tokenizer.decode = lambda tokens, **kwargs: str(sum(tokens) % 3)
    value.config.eval.gsm8k_ancestral_steps = [1, 2, 4]
    value.config.eval.gsm8k_ancestral_seeds = [0, 5, 19]
    return value


def oracle(budgets, seeds, data):
    result = []
    with torch.random.fork_rng(devices=[]):
        for budget in budgets:
            runs = []
            for seed in seeds:
                teacher = model()
                torch.manual_seed(seed)
                correct = total_nfe = forwards = 0
                for start in range(0, len(data), 2):
                    chunk = data[start:start + 2]
                    samples, nfe = reference_sflm(teacher, [row['input_ids'] for row in chunk], budget)
                    for row, sample in zip(chunk, samples):
                        answer = str(sum(sample[len(row['input_ids']):].tolist()) % 3)
                        correct += answer == row['answer']
                    total_nfe += nfe * len(chunk)
                    forwards += nfe
                runs.append({'seed': seed, 'accuracy': correct / len(data), 'correct': correct,
                             'count': len(data), 'nfe': total_nfe / len(data),
                             'total_batch_forwards': forwards})
            result.append(runs)
    return result


class AncestralSweepTest(unittest.TestCase):
    def test_settings_fallback_and_ordered_custom_grid(self):
        value = model()
        value.config.eval.gsm8k_ancestral_steps = None
        value.config.eval.gsm8k_ancestral_seeds = None
        self.assertEqual(ancestral.settings(value), ([4], [1]))
        value.config.eval.gsm8k_steps = 7
        self.assertEqual(ancestral.settings(value), ([7], [1]))
        cfg = OmegaConf.create({'gsm8k_ancestral_steps': [4, 1, 2],
                                'gsm8k_ancestral_seeds': [0, 9]})
        self.assertEqual(ancestral.settings(value, cfg), ([4, 1, 2], [0, 9]))

    def test_invalid_steps_seeds_and_predictor_fail_before_forward(self):
        for field, invalid in [('gsm8k_ancestral_steps', [[], [1, 1], [0], [-1], [True], [1.5], 4]),
                               ('gsm8k_ancestral_seeds', [[], [1, 1], [-1], [True], [1.5], [2**64], 4])]:
            for bad in invalid:
                with self.subTest(field=field, bad=bad):
                    value = model()
                    value.config.eval[field] = bad
                    with self.assertRaises(ValueError):
                        ancestral.sweep(value, examples(), distributed=False)
                    self.assertEqual(value.seen, [])
        value = model()
        value.config.sampling.predictor = 'analytic'
        with self.assertRaisesRegex(ValueError, 'ancestral predictor'):
            ancestral.protocol(value)

    def test_actual_samples_counts_nfe_and_sample_sd_match_independent_oracle(self):
        value, data = model(), examples()
        expected = oracle([1, 2, 4], [0, 5, 19], data)
        observed = []
        torch.manual_seed(5551)
        before = torch.get_rng_state().clone()
        with patch.object(sdtt_eval, 'grade_sample',
                          side_effect=lambda text, answer, **kwargs: text == answer), \
                patch.object(torch.cuda, 'get_rng_state', side_effect=AssertionError('CPU touched CUDA')):
            result = ancestral.sweep(value, data, distributed=False, on_point=observed.append)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(result['points'], observed)
        self.assertEqual(result['protocol']['examples'], 5)
        self.assertEqual(result['protocol']['batch_size'], 2)
        self.assertEqual(result['protocol']['world_size'], 1)
        self.assertEqual(result['protocol']['std_ddof'], 1)
        self.assertEqual(result['protocol']['temperature'], 0.)
        self.assertFalse(result['protocol']['final_denoise'])
        for point, runs in zip(result['points'], expected):
            self.assertEqual(point['seeds'], runs)
            self.assertEqual(point['num_seeds'], 3)
            for source, mean_key, std_key in [('accuracy', 'accuracy_mean', 'accuracy_std'),
                                              ('nfe', 'nfe_mean', 'nfe_std')]:
                measurements = [row[source] for row in runs]
                mean = sum(measurements) / len(measurements)
                std = math.sqrt(sum((x - mean)**2 for x in measurements) / (len(measurements) - 1))
                self.assertAlmostEqual(point[mean_key], mean)
                self.assertAlmostEqual(point[std_key], std)

    def test_standalone_cfg_override_single_seed_null_sd_and_final_cleanup_forwarding(self):
        value = model()
        del value.trainer
        cfg = OmegaConf.create({'gsm8k_ancestral_steps': [8, 4], 'gsm8k_ancestral_seeds': [2],
                                'gsm8k_ancestral_final_denoise': True, 'gsm8k_batch_size': 3,
                                'gsm8k_grading_workers': 1})
        with patch.object(sdtt_eval, '_evaluate_greedy', return_value=(3, 5, 25, 7)) as evaluate, \
                patch.object(ancestral.dist, 'is_initialized', return_value=True), \
                patch.object(ancestral.dist, 'all_reduce', side_effect=AssertionError('standalone collective')), \
                patch.object(sdtt_eval, '_raise_if_failed', side_effect=AssertionError('standalone collective')):
            result = ancestral.sweep(value, examples(), distributed=False, cfg=cfg)
        self.assertEqual([p['steps'] for p in result['points']], [8, 4])
        self.assertTrue(all(call.kwargs == {'final_denoise': True} for call in evaluate.call_args_list))
        self.assertTrue(result['protocol']['final_denoise'])
        self.assertIn('separate cleanup', result['protocol']['final_removal'])
        for point in result['points']:
            self.assertIsNone(point['accuracy_std'])
            self.assertIsNone(point['nfe_std'])
            self.assertEqual(point['nfe_mean'], 5.)  # Never replace measured NFE with S+1.
            self.assertEqual(point['seeds'][0]['total_batch_forwards'], 7)

    def test_pooled_ddp_counts_include_uneven_and_empty_rank_shards(self):
        data = examples()
        for world in (3, 7):
            for rank in range(world):
                with self.subTest(world=world, rank=rank):
                    value = model()
                    value.config.sampling.predictor = 'ancestral'
                    value.config.eval.gsm8k_ancestral_steps = [2]
                    value.config.eval.gsm8k_ancestral_seeds = [3, 7]
                    local = data[rank::world]
                    expected_local = [sum(row['answer'] == '0' for row in local), len(local),
                                      2 * len(local), 2 if local else 0]
                    reductions = []

                    def reduce(totals, op):
                        reductions.append(totals.tolist())
                        totals.copy_(torch.tensor([2, 5, 10, 2 * min(world, 5)], dtype=totals.dtype))

                    def gather(target, error):
                        target[:] = [error] * world

                    with ExitStack() as stack:
                        for name, result in [('is_available', True), ('is_initialized', True),
                                             ('get_rank', rank), ('get_world_size', world)]:
                            stack.enter_context(patch.object(ancestral.dist, name, return_value=result))
                        stack.enter_context(patch.object(ancestral.dist, 'all_gather_object', side_effect=gather))
                        stack.enter_context(patch.object(ancestral.dist, 'all_reduce', side_effect=reduce))
                        stack.enter_context(patch.object(sdtt_eval, 'grade_sample',
                            side_effect=lambda text, answer, **kwargs: answer == '0'))
                        seed_rng = stack.enter_context(patch.object(sdtt_eval, '_seed_model_rng',
                                                                  wraps=sdtt_eval._seed_model_rng))
                        result = ancestral.sweep(value, data)
                    self.assertEqual(reductions, [expected_local, expected_local])
                    self.assertEqual([call.args[1] for call in seed_rng.call_args_list],
                                     [3 + 104729 * rank, 7 + 104729 * rank])
                    point = result['points'][0]
                    self.assertEqual(point['accuracy_mean'], .4)
                    self.assertEqual(point['accuracy_std'], 0.)
                    self.assertEqual(point['nfe_mean'], 2.)
                    self.assertTrue(all(row['count'] == 5 for row in point['seeds']))
                    self.assertTrue(all(row['total_batch_forwards'] == 2 * min(world, 5)
                                        for row in point['seeds']))

    def test_evaluation_and_callback_failures_restore_rng_and_do_not_start_next_budget(self):
        value = model()
        torch.manual_seed(8844)
        before = torch.get_rng_state().clone()
        for phase in ('evaluate', 'callback'):
            with self.subTest(phase=phase), \
                    patch.object(sdtt_eval, '_evaluate_greedy',
                                 side_effect=RuntimeError('evaluation failed') if phase == 'evaluate' else None,
                                 return_value=(1, 5, 10, 2)) as evaluate:
                callback = (lambda point: (_ for _ in ()).throw(RuntimeError('callback failed'))
                            if phase == 'callback' else None)
                with self.assertRaisesRegex(RuntimeError, 'eval' if phase == 'evaluate' else 'callback'):
                    ancestral.sweep(value, examples(), distributed=False, on_point=callback)
                self.assertEqual(evaluate.call_count, 1 if phase == 'evaluate' else 3)
                self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_peer_failure_before_reduce_or_after_callback_is_shared_collectively(self):
        for failed_phase in (2, 3):
            with self.subTest(failed_phase=failed_phase):
                value = model()
                value.config.eval.gsm8k_ancestral_steps = [1, 2]
                value.config.eval.gsm8k_ancestral_seeds = [1]
                gathers = 0

                def gather(out, error):
                    nonlocal gathers
                    gathers += 1
                    out[:] = [error, 'peer failed' if gathers == failed_phase else None]

                with ExitStack() as stack:
                    for name, result in [('is_available', True), ('is_initialized', True),
                                         ('get_rank', 0), ('get_world_size', 2)]:
                        stack.enter_context(patch.object(ancestral.dist, name, return_value=result))
                    stack.enter_context(patch.object(ancestral.dist, 'all_gather_object', side_effect=gather))
                    reduce = stack.enter_context(patch.object(ancestral.dist, 'all_reduce'))
                    evaluate = stack.enter_context(patch.object(sdtt_eval, '_evaluate_greedy',
                                                                return_value=(1, 3, 3, 1)))
                    with self.assertRaisesRegex(RuntimeError, 'peer failed'):
                        ancestral.sweep(value, examples(), on_point=lambda point: None)
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(reduce.call_count, int(failed_phase == 3))


if __name__ == '__main__':
    unittest.main()
