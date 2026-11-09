"""Compose CDLM recipes and capture shell commands without launching models."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_THRESHOLDS = [i / 1000 for i in range(325, 951, 25)] + [.98]
VALIDATION_THRESHOLDS = [.325, .4, .475, .55, .625, .7, .775, .85, .925, .98]


def compose_cli(*overrides):
    for name, resolver in {
        'cwd': lambda: str(ROOT), 'device_count': lambda: 4,
        'eval': eval, 'div_up': lambda x, y: (x + y - 1) // y,
    }.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        return compose(config_name='config', overrides=list(overrides))


def recipe(*overrides):
    return compose_cli('+experiment=cdlm_tinygsm', *overrides)


class CDLMRecipeTest(unittest.TestCase):
    def test_same_mdlm_backbone_and_conditional_tinygsm_format(self):
        cfg = recipe()
        self.assertEqual(cfg.algo.name, 'cdlm')
        self.assertEqual(cfg.algo.attention_mode, 'full')
        self.assertEqual(cfg.algo.parameterization, 'subs')
        self.assertFalse(cfg.algo.time_conditioning)
        self.assertFalse(cfg.algo.causal_attention)
        self.assertFalse(cfg.adversarial_distill.is_distill)
        self.assertEqual(cfg.model.name, 'small')
        self.assertEqual(cfg.model.length, 512)
        self.assertEqual(cfg.data.train, 'tiny_gsm')
        self.assertEqual(cfg.data.valid, 'tiny_gsm')
        self.assertEqual(cfg.data.tokenizer_name_or_path, 'HuggingFaceTB/SmolLM-135M')
        self.assertEqual(cfg.data.separator, r'\n')
        self.assertFalse(cfg.data.train_on_prompt)
        self.assertTrue(cfg.data.train_on_pad)
        self.assertFalse(cfg.data.wrap)
        self.assertTrue(cfg.data.filter_too_long)
        self.assertEqual(cfg.training.finetune_path, 'hf://jdeschena/s-flm/tinygsm/mdlm.ckpt')
        self.assertIsNone(cfg.training.student_init_path)
        self.assertFalse(cfg.training.student_init_use_ema)

    def test_4_gpu_batch_and_validation_match_reference_student_updates(self):
        cfg = recipe()
        self.assertEqual(cfg.trainer.devices, 4)
        self.assertEqual(cfg.loader.batch_size, 32)
        self.assertEqual(cfg.trainer.accumulate_grad_batches, 4)
        self.assertEqual(cfg.loader.global_batch_size, 512)
        self.assertEqual(cfg.trainer.devices * cfg.loader.batch_size
                         * cfg.trainer.accumulate_grad_batches, 512)
        self.assertEqual(cfg.trainer.val_check_interval // cfg.trainer.accumulate_grad_batches, 500)
        self.assertIsNone(cfg.trainer.check_val_every_n_epoch)
        self.assertEqual(cfg.trainer.limit_val_batches, 1000)
        self.assertEqual(cfg.callbacks.checkpoint_every_n_steps.every_n_train_steps, 1000)
        self.assertTrue(cfg.callbacks.checkpoint_every_n_steps.save_last)
        self.assertTrue(cfg.callbacks.checkpoint_every_n_steps.save_on_train_epoch_end)
        self.assertEqual(cfg.trainer.precision, 'bf16-mixed')
        self.assertEqual(cfg.training.loss_precision, 'float32')
        self.assertIsNone(cfg.trainer.gradient_clip_val)
        self.assertEqual(cfg.optim.lr, 1e-5)

    def test_default_16_epochs_and_optional_matched_update_budget(self):
        cfg = recipe()
        self.assertEqual(cfg.trainer.max_steps, -1)
        self.assertEqual(cfg.trainer.max_epochs, 16)
        self.assertEqual(cfg.algo.warmup_ratio, .05)
        self.assertEqual(cfg.lr_scheduler.num_warmup_steps, 0)
        matched = compose_cli('+experiment=cdlm_tinygsm_matched')
        self.assertEqual(matched.trainer.max_steps, 20000)
        self.assertEqual(matched.trainer.max_epochs, -1)
        self.assertEqual(matched.algo.kld_type, 'forward_kl')
        self.assertIsNone(matched.algo.warmup_ratio)
        self.assertEqual(matched.lr_scheduler.num_warmup_steps, 2500)

    def test_cdlm_losses_and_explicit_preprocessing_contract(self):
        cfg = recipe()
        self.assertEqual(cfg.algo.distill_weight, 1.)
        self.assertEqual(cfg.algo.consistency_weight, .5)
        self.assertEqual(cfg.algo.denoise_weight, .1)
        self.assertEqual(cfg.algo.kld_type, 'forward_kl')
        self.assertEqual(cfg.algo.denoise_mask_floor, .001)
        self.assertEqual(cfg.algo.segment_length, 32)
        self.assertEqual(list(cfg.algo.trajectory_temperatures), [0., .5])
        self.assertEqual(cfg.algo.trajectory_num_examples, 20000)
        self.assertEqual(cfg.algo.trajectory_batch_size, 8)
        self.assertEqual(cfg.algo.trajectory_shard_size, 8)
        self.assertEqual(cfg.algo.teacher_hidden_dtype, 'bfloat16')
        raw = OmegaConf.to_container(cfg.algo, resolve=False)
        self.assertEqual(raw['trajectory_cache_dir'],
                         '${hydra:runtime.cwd}/cache_dir/cdlm_tinygsm_trajectories')

    def test_shared_underdeep_gsm8k_comparison_protocol(self):
        cfg = recipe()
        self.assertEqual(cfg.logger.experiment, 'generative-models/tinygsm')
        self.assertTrue(cfg.eval.validate_before_train)
        self.assertTrue(cfg.eval.gsm8k_at_validation)
        self.assertTrue(cfg.eval.gsm8k_greedy_token)
        self.assertIsNone(cfg.eval.gsm8k_steps)
        self.assertEqual(cfg.sampling.steps, 64)
        self.assertEqual(list(cfg.eval.gsm8k_pareto_thresholds), VALIDATION_THRESHOLDS)
        self.assertEqual(cfg.eval.gsm8k_pareto_temperature, 0.)
        self.assertIsNone(cfg.eval.gsm8k_num_examples)
        self.assertEqual(cfg.eval.gsm8k_batch_size, 16)
        self.assertFalse(cfg.eval.generate_samples)
        self.assertFalse(cfg.eval.compute_generative_perplexity)
        self.assertEqual(cfg.sampling.num_sample_log, 8)


class CDLMLauncherTest(unittest.TestCase):
    def capture(self, script, args=(), *, preset=None, prepare_gpus=None,
                bundle=False, code_override=False, fail_prepare=False):
        with tempfile.TemporaryDirectory(prefix='cdlm launcher ') as directory:
            temporary = Path(directory)
            bundle_code = temporary / 'source bundle' / 'code'
            bundle_code.mkdir(parents=True)
            explicit_code = temporary / 'explicit code'
            explicit_code.mkdir()
            capture = temporary / 'commands.jsonl'
            python = temporary / 'mock python'
            python.write_text(
                '#!' + sys.executable + '\n'
                'import json, os, pathlib, sys\n'
                'record = {"args": sys.argv[1:], "cwd": os.getcwd()}\n'
                'with pathlib.Path(os.environ["CDLM_CAPTURE"]).open("a") as stream:\n'
                '    stream.write(json.dumps(record) + "\\n")\n'
                'if os.environ.get("CDLM_MOCK_FAIL_PREPARE") and "mode=prepare_cdlm" in sys.argv:\n'
                '    sys.exit(17)\n')
            python.chmod(0o755)
            env = dict(os.environ, PYTHON=str(python), CDLM_CAPTURE=str(capture))
            for key in ['CDLM_PRESET', 'CDLM_PREPARE_GPUS', 'SOURCE_CODE_PATH', 'CDLM_CODE_DIR']:
                env.pop(key, None)
            if preset:
                env['CDLM_PRESET'] = preset
            if prepare_gpus:
                env['CDLM_PREPARE_GPUS'] = str(prepare_gpus)
            if bundle:
                env['SOURCE_CODE_PATH'] = str(bundle_code.parent)
            if code_override:
                env['CDLM_CODE_DIR'] = str(explicit_code)
            if fail_prepare:
                env['CDLM_MOCK_FAIL_PREPARE'] = '1'
            result = subprocess.run(['bash', str(ROOT / 'scripts' / script), *args],
                                    cwd=temporary, env=env, text=True, capture_output=True)
            commands = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
            return result, commands, bundle_code, explicit_code

    def main_config(self, command):
        arguments = command['args']
        main = arguments.index('main')
        return compose_cli(*arguments[main + 1:])

    def test_training_only_wrapper_and_trailing_overrides(self):
        result, commands, _, _ = self.capture('train_cdlm_tinygsm.sh',
                                             ['optim.lr=2e-5', 'logger.name=custom cdlm'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]['cwd'], str(ROOT))
        self.assertEqual(commands[0]['args'][:3], ['-u', '-m', 'main'])
        cfg = self.main_config(commands[0])
        self.assertEqual(cfg.algo.name, 'cdlm')
        self.assertEqual(cfg.optim.lr, 2e-5)
        self.assertEqual(cfg.logger.name, 'custom cdlm')
        self.assertEqual(cfg.mode, 'train')

    def test_matched_budget_can_be_selected(self):
        result, commands, _, _ = self.capture('train_cdlm_tinygsm.sh',
                                             preset='cdlm_tinygsm_matched')
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = self.main_config(commands[0])
        self.assertEqual(cfg.trainer.max_steps, 20000)
        self.assertEqual(cfg.trainer.max_epochs, -1)

    def test_prepare_wrapper_uses_distributed_collection_and_forces_prepare_mode(self):
        result, commands, _, _ = self.capture('prepare_cdlm_tinygsm.sh',
                                             ['mode=train', 'algo.trajectory_num_examples=8'],
                                             prepare_gpus=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 1)
        self.assertIn('torch.distributed.run', commands[0]['args'])
        self.assertIn('--nproc_per_node=2', commands[0]['args'])
        cfg = self.main_config(commands[0])
        self.assertEqual(cfg.mode, 'prepare_cdlm')
        self.assertEqual(cfg.algo.trajectory_num_examples, 8)

    def test_cluster_job_prepares_then_trains_same_offline_assets(self):
        result, commands, code, _ = self.capture('train_cdlm_tinygsm_nirvana.sh',
                                                 ['logger.name=cluster cdlm'], bundle=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 2)
        self.assertIn('torch.distributed.run', commands[0]['args'])
        self.assertIn('--nproc_per_node=4', commands[0]['args'])
        for command, mode in zip(commands, ['prepare_cdlm', 'train']):
            self.assertEqual(command['cwd'], str(code.resolve()))
            cfg = self.main_config(command)
            self.assertEqual(cfg.mode, mode)
            self.assertEqual(cfg.algo.name, 'cdlm')
            self.assertEqual(cfg.trainer.max_epochs, 16)
            self.assertEqual(cfg.training.finetune_path, str(code / 'checkpoints/tinygsm/mdlm_notok.ckpt'))
            self.assertEqual(cfg.model_weights_dir, str(code / 'cache_dir/tinygsm_mmd_offline_bundle'))
            self.assertEqual(cfg.data.cache_dir, str(code / 'cache_dir'))
            self.assertEqual(cfg.algo.trajectory_cache_dir, str(code / 'cache_dir/cdlm_tinygsm_trajectories'))
            self.assertEqual(cfg.checkpointing.save_dir, str(code / 'cdlm_tinygsm'))
            self.assertEqual(cfg.logger.experiment, 'generative-models/tinygsm')
            self.assertEqual(cfg.logger.name, 'cluster cdlm')
            self.assertEqual(cfg.trainer.accumulate_grad_batches, 4)
            self.assertEqual(list(cfg.eval.gsm8k_pareto_thresholds), VALIDATION_THRESHOLDS)

    def test_cluster_code_override_and_failed_prepare_prevent_training(self):
        result, commands, _, code = self.capture('train_cdlm_tinygsm_nirvana.sh',
                                                code_override=True, fail_prepare=True)
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]['cwd'], str(code.resolve()))
        self.assertEqual(self.main_config(commands[0]).mode, 'prepare_cdlm')

    def test_cluster_warmstart_overrides_leave_original_teacher_and_default_eval(self):
        result, commands, code, _ = self.capture('train_cdlm_tinygsm_nirvana.sh',
            ['training.student_init_path=/trained/student mmd.ckpt',
             'training.student_init_use_ema=True', 'eval.gsm8k_steps=32'], bundle=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 2)
        for command in commands:
            cfg = self.main_config(command)
            self.assertEqual(cfg.training.finetune_path,
                             str(code / 'checkpoints/tinygsm/mdlm_notok.ckpt'))
            self.assertEqual(cfg.training.student_init_path, '/trained/student mmd.ckpt')
            self.assertTrue(cfg.training.student_init_use_ema)
            self.assertTrue(cfg.eval.gsm8k_greedy_token)
            self.assertEqual(cfg.eval.gsm8k_steps, 32)

    def test_generation_threshold_override_wins_over_full_sweep_default(self):
        result, commands, _, _ = self.capture('generation_cdlm_tinygsm.sh',
                                             ['hf://owner/repository/model.ckpt',
                                              'eval.gsm8k_pareto_thresholds=[0.4,0.7,0.98]'])
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = self.main_config(commands[0])
        self.assertEqual(list(cfg.eval.gsm8k_pareto_thresholds), [.4, .7, .98])

    def test_generation_requires_checkpoint_and_preserves_spaces(self):
        result, commands, _, _ = self.capture('generation_cdlm_tinygsm.sh')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Usage:', result.stderr)
        self.assertEqual(commands, [])
        with tempfile.TemporaryDirectory(prefix='cdlm checkpoint ') as directory:
            checkpoint = Path(directory) / 'student weights.ckpt'
            checkpoint.touch()
            result, commands, _, _ = self.capture('generation_cdlm_tinygsm.sh',
                                                 [str(checkpoint), 'eval.gsm8k_batch_size=7'])
            self.assertEqual(result.returncode, 0, result.stderr)
            cfg = self.main_config(commands[0])
            self.assertEqual(cfg.mode, 'sample_eval')
            self.assertEqual(cfg.trainer.devices, 1)
            self.assertEqual(cfg.eval.checkpoint_path, str(checkpoint.resolve()))
            self.assertEqual(cfg.eval.gsm8k_batch_size, 7)
            self.assertIsNone(cfg.eval.gsm8k_pareto_thresholds)
            self.assertIsNone(cfg.eval.gsm8k_num_examples)


if __name__ == '__main__':
    unittest.main()
