"""Validation/standalone persistence and publication of the ancestral sweep."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import sdtt_eval as validation
from test_cdlm_ancestral import Model, examples


def enable_grid(config):
    config.eval.gsm8k_ancestral_steps = [1, 2]
    config.eval.gsm8k_ancestral_seeds = [1, 2]
    config.eval.gsm8k_ancestral_final_denoise = True


class AncestralIntegrationTest(unittest.TestCase):
    def model(self, directory):
        model = Model(directory)
        enable_grid(model.config)
        model._gsm8k_examples = examples()
        model._gsm8k_protocol = {'data_sha256': 'data-digest', 'weights': 'ema',
                                 'sequence_length': model.num_tokens}
        return model

    def test_complete_sweep_is_persisted_and_logged_before_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.model(directory)
            curve = [{'threshold': .49, 'accuracy': .5, 'nfe': 3.}]
            path = Path(directory) / 'ancestral_history.json'
            torch.manual_seed(991)
            before = torch.get_rng_state().clone()

            def confidence(*args):
                report = json.loads(path.read_text())['evaluations'][str(model.global_step)]
                self.assertTrue(report['complete'])
                self.assertEqual([point['steps'] for point in report['points']], [1, 2])
                self.assertEqual(report['protocol']['world_size'], 1)
                self.assertEqual(report['protocol']['examples'], 5)
                self.assertEqual(report['protocol']['data_sha256'], 'data-digest')
                logged = dict(item for values, step in model.trainer.logger.metrics
                              if step == model.global_step for item in values.items())
                self.assertEqual(logged['val/gsm8k_acc_ancestral_steps1'], 1.)
                self.assertEqual(logged['val/gsm8k_acc_ancestral_std_steps2'], 0.)
                self.assertIn('val/gsm8k_acc_ancestral_steps2_seed2', logged)
                self.assertNotIn('val/gsm8k_acc_greedy', logged)
                self.assertTrue(torch.equal(torch.get_rng_state(), before))
                return curve

            with patch.object(validation, 'grade_sample', return_value=True), \
                    patch.object(validation, 'sweep', side_effect=confidence):
                for step, options in [(0, {'initial': True}), (500, {}), (501, {'final': True})]:
                    model.global_step = step
                    validation.log_validation_gsm8k(model, **options)
                with patch.object(validation, '_validation_ancestral_sweep') as repeated:
                    validation.log_validation_gsm8k(model, final=True)
                repeated.assert_not_called()
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            history = json.loads(path.read_text())['evaluations']
            self.assertEqual(set(history), {'0', '500', '501'})
            report = json.loads((Path(directory) / 'pareto_history.json').read_text())
            self.assertEqual(report['curves']['501'], curve)
            self.assertIsNone(report['greedy'])
            self.assertTrue(report['ancestral']['protocol']['final_denoise'])

    def test_later_budget_failure_keeps_completed_budget_and_skips_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.model(directory)

            def evaluate(model, examples, pool, batch_size, steps, timeout, **kwargs):
                if steps == 2:
                    raise RuntimeError('second budget failed')
                return 3, 5, 10, 6

            with patch.object(validation, '_evaluate_greedy', side_effect=evaluate), \
                    patch.object(validation, 'sweep') as confidence:
                with self.assertRaisesRegex(RuntimeError, 'second budget failed'):
                    validation.log_validation_gsm8k(model)
            confidence.assert_not_called()
            report = json.loads((Path(directory) / 'ancestral_history.json').read_text())
            result = report['evaluations']['500']
            self.assertFalse(result['complete'])
            self.assertEqual([point['steps'] for point in result['points']], [1])
            self.assertEqual(result['points'][0]['accuracy_mean'], .6)
            self.assertTrue(any('val/gsm8k_acc_ancestral_steps1' in values
                                for values, step in model.trainer.logger.metrics))
            self.assertFalse(hasattr(model, '_gsm8k_last_logged_step'))

    def test_standalone_cfg_exports_all_seed_counts_and_preserves_confidence_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            # Only the supplied standalone config has the new grid.
            model = Model(directory)
            config = OmegaConf.create(OmegaConf.to_container(model.config, resolve=True))
            enable_grid(config)
            data, output = Path(directory) / 'data.json', Path(directory) / 'result.json'
            data.write_text(json.dumps([{'prompt': 'Q', 'response_ground_truth': 'yes'}]))
            config.eval.update({'gsm8k_data_path': str(data), 'checkpoint_path': 'student.ckpt',
                                'disable_ema': False, 'generated_samples_path': str(output)})
            streams = []

            def confidence(*args, **kwargs):
                streams.append(torch.rand(5))
                return 1, 1, 3, 3

            with patch.object(validation, '_evaluate_threshold', side_effect=confidence), \
                    patch.object(validation, 'grade_sample', return_value=True):
                config.eval.gsm8k_greedy_token = False
                baseline = validation.evaluate(model, config)
                config.eval.gsm8k_greedy_token = True
                result = validation.evaluate(model, config)
            self.assertEqual(baseline, result)
            self.assertTrue(torch.equal(*streams))
            report = json.loads(output.read_text())
            self.assertTrue(report['ancestral']['complete'])
            self.assertEqual(report['ancestral']['protocol']['checkpoint'], 'student.ckpt')
            self.assertEqual(report['ancestral']['protocol']['seeds'], [1, 2])
            self.assertEqual(len(report['ancestral']['points']), 2)
            self.assertTrue(all(run['count'] == 1 for point in report['ancestral']['points']
                                for run in point['seeds']))
            self.assertTrue(output.with_name('result_ancestral.png').is_file())

    def test_resume_keeps_prior_history_and_discards_future_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.model(directory)
            model.trainer.ckpt_path = 'resumed.ckpt'
            path = Path(directory) / 'ancestral_history.json'
            path.write_text(json.dumps({'evaluations': {'100': {'complete': True},
                                                        '1000': {'complete': True}}}))
            with patch.object(validation, 'grade_sample', return_value=True), \
                    patch.object(validation, 'sweep', return_value=[]):
                validation.log_validation_gsm8k(model)
            history = json.loads(path.read_text())['evaluations']
            self.assertEqual(set(history), {'100', '500'})
            self.assertTrue(history['500']['complete'])


if __name__ == '__main__':
    unittest.main()
