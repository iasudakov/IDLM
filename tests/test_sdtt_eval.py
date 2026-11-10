"""CPU checks for the paper GSM8K decoder, accounting, prompts, and grader."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import sdtt_eval
import sdtt_sandbox_gsm8k as grader


class Tokenizer:
    bos_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        return SimpleNamespace(input_ids=[1] * len(text))

    def decode(self, tokens, skip_special_tokens=False):
        return 'def simple_math_problem():\n    return 42'


class FixedModel:
    """A denoiser that emits EOS at every generated position with probability .5."""

    device = torch.device('cpu')
    mask_index = 4
    num_tokens = 5
    noise = SimpleNamespace(eps=.001)

    def __init__(self):
        self.seen = []
        self.tokenizer = Tokenizer()
        self.config = OmegaConf.create({'data': {'separator': '\\n'}})

    def prior_sample(self, *shape):
        return torch.full(shape, self.mask_index, dtype=torch.long)

    def _sigma_from_alphat(self, alpha):
        return -alpha.log()

    def forward(self, x, sigma):
        self.seen.append(x.clone())
        probs = torch.zeros(*x.shape, 5)
        probs[..., 1] = .5
        probs[..., 2] = .5
        return probs.log()


# Make the original linear schedule callable while keeping its epsilon visible.
class Noise:
    eps = .001

    def __call__(self, t):
        return -(1 - self.eps), 1 - (1 - self.eps) * t


FixedModel.noise = Noise()


class DecoderTest(unittest.TestCase):
    def test_confidence_is_strict_and_finished_rows_do_not_accrue_nfe(self):
        model = FixedModel()
        prompts = [[0, 2], [0, 2, 2]]
        samples, metadata = sdtt_eval.generate_samples(model, prompts, threshold=.5)
        self.assertEqual(metadata['nfe'], 3)
        self.assertEqual(metadata['nfe_per_row'].tolist(), [3, 2])
        self.assertEqual(samples.tolist(), [[0, 2, 1, 1, 1], [0, 2, 2, 1, 1]])
        # EOS does not terminate generation early: the complete 5-token row is filled.
        self.assertFalse(bool((samples == model.mask_index).any()))
        for state in model.seen:
            self.assertEqual(state[0, :2].tolist(), prompts[0])
            self.assertEqual(state[1, :3].tolist(), prompts[1])

    def test_low_threshold_reveals_all_positions_in_one_forward(self):
        samples, metadata = sdtt_eval.generate_samples(FixedModel(), [[0]], .49)
        self.assertEqual(metadata['nfe_per_row'].tolist(), [1])
        self.assertEqual(samples.tolist(), [[0, 1, 1, 1, 1]])

    def test_cap_fallback_is_counted_as_an_additional_forward(self):
        samples, metadata = sdtt_eval.generate_samples(FixedModel(), [[0]], 1., max_steps=1)
        self.assertEqual(metadata['nfe'], 2)
        self.assertEqual(metadata['nfe_per_row'].tolist(), [2])
        self.assertEqual(samples.tolist(), [[0, 1, 1, 1, 1]])

    def test_invalid_inputs_are_rejected(self):
        for prompts, threshold, temperature, cap in [
            ([], .5, 0., None), ([[0] * 5], .5, 0., None),
            ([[4]], .5, 0., None), ([[0]], float('nan'), 0., None),
            ([[0]], .5, -1., None), ([[0]], .5, 0., 0),
        ]:
            with self.subTest(prompts=prompts, threshold=threshold, temperature=temperature, cap=cap):
                with self.assertRaises(ValueError):
                    sdtt_eval.generate_samples(FixedModel(), prompts, threshold, temperature, cap)


class PromptAndReportTest(unittest.TestCase):
    def test_extended_reference_grid_and_literal_separator(self):
        self.assertEqual(len(sdtt_eval.THRESHOLDS), 29)
        self.assertEqual(sdtt_eval.THRESHOLDS[:3], [.325, .35, .375])
        self.assertEqual(sdtt_eval.THRESHOLDS[-4:], [.95, .98, .99, .995])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'examples.json'
            path.write_text(json.dumps([{'prompt': ' Q ', 'response_ground_truth': '#### 42'}]))
            tokenizer = Tokenizer()
            examples = sdtt_eval.load_examples(path, tokenizer, r'\n', 8)
            self.assertEqual(tokenizer.calls, [(r'\n', False), ('Q', False)])
            self.assertEqual(examples[0]['input_ids'], [0, 1, 1, 1])
            with self.assertRaises(ValueError):
                sdtt_eval.load_examples(path, tokenizer, r'\n', 4)

    def test_explicit_missing_data_does_not_download(self):
        config = OmegaConf.create({'eval': {'gsm8k_data_path': '/no/such/gsm8k-test.json'}})
        with self.assertRaises(FileNotFoundError):
            sdtt_eval.resolve_data_path(config)

    def test_report_contains_accuracy_and_mean_active_nfe(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data.json'
            output = Path(directory) / 'report.json'
            data.write_text(json.dumps([
                {'prompt': 'Q', 'response_ground_truth': '#### 42'},
                {'prompt': 'QQ', 'response_ground_truth': '#### 0'},
            ]))
            config = OmegaConf.create({
                'seed': 1,
                'eval': {
                    'checkpoint_path': 'student.ckpt', 'disable_ema': False,
                    'generated_samples_path': str(output), 'gsm8k_data_path': str(data),
                    'gsm8k_pareto_thresholds': [.49, .5], 'gsm8k_batch_size': 2,
                    'gsm8k_grading_workers': 1,
                },
            })
            # Single-character separator leaves 2 and 1 answer positions respectively.
            model = FixedModel()
            model.config.data.separator = '\n'
            with patch.object(sdtt_eval, 'grade_sample', side_effect=lambda sample, answer, timeout_s: answer == '#### 42'):
                results = sdtt_eval.evaluate(model, config)
            self.assertEqual([point['count'] for point in results], [2, 2])
            self.assertEqual([point['accuracy'] for point in results], [.5, .5])
            self.assertEqual([point['nfe'] for point in results], [1., 1.5])
            self.assertEqual([point['total_batch_forwards'] for point in results], [1, 2])
            report = json.loads(output.read_text())
            self.assertEqual(report['weights'], 'live')
            self.assertEqual(report['sequence_length'], 5)
            self.assertEqual(report['results'], results)
            self.assertEqual(len(report['data_sha256']), 64)
            # Checkpoint loading may apply EMA eagerly and discard shadows.
            model.ema = None
            model._evaluation_weights = 'ema'
            with patch.object(sdtt_eval, 'grade_sample', return_value=True):
                sdtt_eval.evaluate(model, config)
            self.assertEqual(json.loads(output.read_text())['weights'], 'ema')


class ExecutionGraderTest(unittest.TestCase):
    def test_execution_and_stdout_fallback_match_reference(self):
        self.assertTrue(grader.evaluate_samples('```python\ndef simple_math_problem():\n    return 42\n```', '#### 42', 1.))
        self.assertTrue(grader.evaluate_samples('def simple_math_problem():\n    print("Answer: 42")', '#### 42', 1.))
        self.assertTrue(grader.evaluate_samples('def simple_math_problem():\n    return 42.0005', '#### 42', 1.))
        self.assertFalse(grader.evaluate_samples('The answer is 42', '#### 42', 1.))
        self.assertFalse(grader.evaluate_samples('def simple_math_problem():\n    return 41', '#### 42', 1.))

    def test_subprocess_worker_and_timeout(self):
        self.assertTrue(sdtt_eval.grade_sample('def simple_math_problem():\n    return 42', '#### 42', .2))
        self.assertFalse(sdtt_eval.grade_sample('def simple_math_problem():\n    while True: pass', '#### 42', .05))


if __name__ == '__main__':
    unittest.main()
