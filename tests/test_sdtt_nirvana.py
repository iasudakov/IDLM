"""Check the cluster launch contract without assets, training, or network I/O."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_sdtt_protocol import compose_cli, REFERENCE_THRESHOLDS


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/train_sdtt_tinygsm_nirvana.sh'


class NirvanaLaunchTest(unittest.TestCase):
    def capture(self, *, code_override=False, no_bundle=False, arguments=()):
        with tempfile.TemporaryDirectory(prefix='sdtt nirvana ') as directory:
            temporary = Path(directory)
            bundle = temporary / 'source bundle'
            bundle_code = bundle / 'code'
            bundle_code.mkdir(parents=True)
            explicit_code = temporary / 'explicit code'
            explicit_code.mkdir()
            capture = temporary / 'command.json'
            python = temporary / 'mock python'
            python.write_text(
                '#!' + sys.executable + '\n'
                'import json, os, pathlib, sys\n'
                'pathlib.Path(os.environ["SDTT_CAPTURE"]).write_text(json.dumps('
                '{"arguments": sys.argv[1:], "cwd": os.getcwd()}))\n')
            python.chmod(0o755)
            env = dict(os.environ, PYTHON=str(python), SDTT_CAPTURE=str(capture))
            env.pop('SOURCE_CODE_PATH', None)
            env.pop('SDTT_CODE_DIR', None)
            if not no_bundle:
                env['SOURCE_CODE_PATH'] = str(bundle)
            if code_override:
                env['SDTT_CODE_DIR'] = str(explicit_code)
            completed = subprocess.run(['bash', str(SCRIPT), *arguments],
                                       cwd=temporary, env=env, text=True,
                                       capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            command = json.loads(capture.read_text())
            expected = explicit_code if code_override else ROOT if no_bundle else bundle_code
            self.assertEqual(command['cwd'], str(expected.resolve()))
            self.assertEqual(command['arguments'][:3], ['-u', '-m', 'main'])
            return command['arguments'][3:], expected

    def test_source_bundle_matches_mmd_assets_and_logging(self):
        arguments, code = self.capture()
        config = compose_cli(*arguments)
        self.assertEqual(config.training.finetune_path,
                         str(code / 'checkpoints/tinygsm/mdlm_notok.ckpt'))
        self.assertEqual(config.data.cache_dir, str(code / 'cache_dir'))
        self.assertEqual(config.model_weights_dir,
                         str(code / 'cache_dir/tinygsm_mmd_offline_bundle'))
        self.assertEqual(config.checkpointing.save_dir, str(code / 'sdtt_tinygsm'))
        self.assertIn('hydra.run.dir=' + str(code / 'sdtt_tinygsm'), arguments)
        self.assertEqual(config.logger._target_, 'underdeep_logger.UnderdeepLogger')
        self.assertEqual(config.logger.experiment, 'generative-models/tinygsm')
        self.assertEqual(config.logger.name, 'sdtt tinygsm lr1e-5 official 70k')

    def test_official_sdtt_uses_reference_batch_and_gsm8k_protocol(self):
        arguments, _ = self.capture()
        config = compose_cli(*arguments)
        self.assertEqual(config.algo.name, 'sdtt')
        self.assertFalse(config.adversarial_distill.is_distill)
        self.assertEqual(config.trainer.max_steps, 70000)
        self.assertEqual(config.algo.grow_dt_every, 10000)
        self.assertEqual(config.trainer.devices * config.loader.batch_size
                         * config.trainer.accumulate_grad_batches, 512)
        self.assertEqual(config.trainer.val_check_interval
                         // config.trainer.accumulate_grad_batches, 500)
        self.assertEqual(config.model.length, 512)
        self.assertEqual(config.loader.eval_batch_size, 64)
        self.assertEqual(config.eval.gsm8k_batch_size, 16)
        self.assertTrue(config.eval.validate_before_train)
        self.assertTrue(config.eval.gsm8k_at_validation)
        self.assertFalse(config.eval.generate_samples)
        self.assertEqual(list(config.eval.gsm8k_pareto_thresholds), REFERENCE_THRESHOLDS)

    def test_explicit_code_dir_and_trailing_overrides_win(self):
        arguments, _ = self.capture(code_override=True, arguments=[
            'trainer.devices=4', 'trainer.accumulate_grad_batches=4',
            'trainer.val_check_interval=2000', 'logger.name=sdtt four gpu',
            'training.finetune_path=/custom assets/mdlm.ckpt'])
        config = compose_cli(*arguments)
        self.assertEqual(config.trainer.devices, 4)
        self.assertEqual(config.trainer.accumulate_grad_batches, 4)
        self.assertEqual(config.trainer.val_check_interval, 2000)
        self.assertEqual(config.logger.name, 'sdtt four gpu')
        self.assertEqual(config.training.finetune_path, '/custom assets/mdlm.ckpt')

    def test_local_checkout_is_used_when_bundle_is_unset(self):
        arguments, code = self.capture(no_bundle=True)
        self.assertEqual(code, ROOT)
        self.assertIn('data.cache_dir=' + str(ROOT / 'cache_dir'), arguments)


if __name__ == '__main__':
    unittest.main()
