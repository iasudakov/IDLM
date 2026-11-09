"""Resolve the runnable TinyGSM recipe against the external MMD protocol.

These tests exercise Hydra composition without importing the training entry point
or allocating a model. The reference requirements come from Underdeep run
3e0ac6b1-7a8f-43c3-af58-d9e21b886da7, rather than SDTT's implementation.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_THRESHOLDS = [i / 1000 for i in range(325, 951, 25)] + [.98]
VALIDATION_THRESHOLDS = [.325, .4, .475, .55, .625, .7, .775, .85, .925, .98]


def compose_cli(*overrides):
    # Register the same generic config conveniences without importing main,
    # whose model imports require optional GPU/development dependencies.
    for name, resolver in {
        'cwd': lambda: str(ROOT),
        'device_count': lambda: 8,
        'eval': eval,
        'div_up': lambda x, y: (x + y - 1) // y,
    }.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        return compose(config_name='config', overrides=list(overrides))


def recipe(*overrides):
    return compose_cli('+experiment=sdtt_tinygsm', *overrides)


class TinyGSMProtocolTest(unittest.TestCase):
    def test_recipe_preserves_teacher_task_and_prompt_conventions(self):
        config = recipe()
        self.assertEqual(config.data.train, 'tiny_gsm')
        self.assertEqual(config.data.valid, 'tiny_gsm')
        self.assertEqual(config.data.tokenizer_name_or_path, 'HuggingFaceTB/SmolLM-135M')
        self.assertEqual(config.model.length, 512)
        self.assertFalse(config.data.wrap)
        self.assertFalse(config.data.train_on_prompt)
        self.assertTrue(config.data.train_on_pad)
        self.assertTrue(config.data.filter_too_long)
        self.assertEqual(config.data.separator, r'\n')
        self.assertEqual(config.algo.name, 'sdtt')
        self.assertEqual(config.algo.parameterization, 'subs')
        self.assertFalse(config.algo.time_conditioning)
        self.assertFalse(config.adversarial_distill.is_distill)
        self.assertEqual(config.logger.experiment, 'generative-models/tinygsm')

    def test_training_reference_uses_the_same_effective_global_batch(self):
        config = recipe()
        self.assertEqual(config.optim.lr, 1e-5)
        self.assertEqual(config.loader.batch_size, 32)
        self.assertEqual(config.loader.global_batch_size, 512)
        self.assertEqual(config.trainer.devices, 8)
        actual_batch = (config.loader.batch_size * config.trainer.devices
                        * config.trainer.num_nodes * config.trainer.accumulate_grad_batches)
        self.assertEqual(actual_batch, 512)
        self.assertIsNone(config.trainer.gradient_clip_val)

    def test_gpu_count_override_keeps_effective_batch_for_comparison(self):
        for devices in [1, 2, 4, 8]:
            with self.subTest(devices=devices):
                config = recipe(f'trainer.devices={devices}')
                actual_batch = (config.loader.batch_size * config.trainer.devices
                                * config.trainer.num_nodes * config.trainer.accumulate_grad_batches)
                self.assertEqual(actual_batch, 512)

    def test_validation_uses_reduced_grid_and_exec_accuracy(self):
        config = recipe()
        self.assertTrue(config.eval.gsm8k_at_validation)
        self.assertTrue(config.eval.validate_before_train)
        self.assertEqual(list(config.eval.gsm8k_pareto_thresholds), VALIDATION_THRESHOLDS)
        self.assertEqual(config.eval.gsm8k_pareto_temperature, 0.)
        self.assertIsNone(config.eval.gsm8k_num_examples)
        self.assertFalse(config.eval.compute_generative_perplexity)
        self.assertFalse(config.eval.generate_samples)
        self.assertEqual(config.sampling.predictor, 'ancestral_cache')


class LauncherTest(unittest.TestCase):
    def capture(self, script, args, *, preset=None):
        with tempfile.TemporaryDirectory(prefix='sdtt launcher ') as directory:
            temporary = Path(directory)
            capture = temporary / 'command.json'
            python = temporary / 'mock python'
            python.write_text(
                '#!/usr/bin/env python3\n'
                'import json, os, pathlib, sys\n'
                'pathlib.Path(os.environ["SDTT_CAPTURE"]).write_text(json.dumps('
                '{"args": sys.argv[1:], "cwd": os.getcwd(), '
                '"tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM")}))\n')
            python.chmod(0o755)
            env = dict(os.environ, PYTHON=str(python), SDTT_CAPTURE=str(capture))
            env.pop('SDTT_PRESET', None)
            if preset is not None:
                env['SDTT_PRESET'] = preset
            result = subprocess.run(['bash', str(ROOT / 'scripts' / script), *args],
                                    cwd=temporary, env=env, text=True, capture_output=True)
            command = json.loads(capture.read_text()) if capture.exists() else None
            return result, command

    def test_training_launcher_resolves_overrides_and_runs_from_repo(self):
        result, command = self.capture('train_sdtt_tinygsm.sh',
                                       ['trainer.devices=4', 'optim.lr=2e-5'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(command['cwd'], str(ROOT))
        self.assertEqual(command['args'][:3], ['-u', '-m', 'main'])
        config = compose_cli(*command['args'][3:])
        self.assertEqual(config.trainer.devices, 4)
        self.assertEqual(config.optim.lr, 2e-5)
        self.assertEqual(config.algo.name, 'sdtt')
        self.assertEqual(config.loader.global_batch_size, 512)
        self.assertEqual(config.trainer.max_steps, 70000)
        self.assertEqual(config.algo.grow_dt_every, 10000)

    def test_official_preset_can_be_selected_without_losing_user_overrides(self):
        result, command = self.capture('train_sdtt_tinygsm.sh', ['optim.lr=2e-5'],
                                       preset='sdtt_tinygsm_official')
        self.assertEqual(result.returncode, 0, result.stderr)
        config = compose_cli(*command['args'][3:])
        self.assertEqual(config.algo.grow_dt_every, 10000)
        self.assertEqual(config.trainer.max_steps, 70000)
        self.assertEqual(config.optim.lr, 2e-5)

    def test_generation_launcher_preserves_checkpoint_spaces_and_user_overrides(self):
        with tempfile.TemporaryDirectory(prefix='sdtt checkpoint ') as directory:
            checkpoint = Path(directory) / 'student model.ckpt'
            checkpoint.touch()
            result, command = self.capture('generation_sdtt_tinygsm.sh',
                                           [str(checkpoint), 'eval.gsm8k_batch_size=7'])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(command['cwd'], str(ROOT))
            self.assertEqual(command['tokenizers_parallelism'], 'false')
            config = compose_cli(*command['args'][3:])
            self.assertEqual(config.eval.checkpoint_path, str(checkpoint.resolve()))
            self.assertEqual(config.mode, 'sample_eval')
            self.assertEqual(config.trainer.devices, 1)
            self.assertEqual(config.eval.gsm8k_batch_size, 7)
            self.assertIsNone(config.eval.gsm8k_pareto_thresholds)
            self.assertIsNone(config.eval.gsm8k_num_examples)

    def test_generation_threshold_override_wins_over_full_sweep_default(self):
        result, command = self.capture('generation_sdtt_tinygsm.sh',
                                       ['hf://owner/repository/model.ckpt',
                                        'eval.gsm8k_pareto_thresholds=[0.4,0.7,0.98]'])
        self.assertEqual(result.returncode, 0, result.stderr)
        config = compose_cli(*command['args'][3:])
        self.assertEqual(list(config.eval.gsm8k_pareto_thresholds), [.4, .7, .98])

    def test_generation_launcher_requires_checkpoint_before_python(self):
        result, command = self.capture('generation_sdtt_tinygsm.sh', [])
        self.assertEqual(result.returncode, 2)
        self.assertIn('Usage:', result.stderr)
        self.assertIsNone(command)

    def test_generation_launcher_accepts_hub_checkpoint(self):
        path = 'hf://owner/repository/model.ckpt'
        result, command = self.capture('generation_sdtt_tinygsm.sh', [path])
        self.assertEqual(result.returncode, 0, result.stderr)
        config = compose_cli(*command['args'][3:])
        self.assertEqual(config.eval.checkpoint_path, path)


if __name__ == '__main__':
    unittest.main()
