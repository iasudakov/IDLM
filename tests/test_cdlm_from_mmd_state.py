import tempfile
from pathlib import Path
import unittest

import torch
from omegaconf import OmegaConf

import cdlm_from_mmd_state as launch


class SourceTest(unittest.TestCase):
    def test_selects_saved_optimizer_step_and_requires_ema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'mmd'
            (source / '.hydra').mkdir(parents=True)
            (source / 'checkpoints').mkdir()
            torch.save({'run_uid': 'source-uid'}, source / 'underdeep_metadata.pt')
            OmegaConf.save(OmegaConf.create({
                'algo': {'name': 'mdlm'}, 'model': {'length': 512},
                'data': {'separator': r'\n'},
                'adversarial_distill': {'is_distill': True, 'mmd_reward_weight': 1.},
            }), source / '.hydra/config.yaml')
            chosen = source / 'checkpoints/train-batch=64000.ckpt'
            torch.save({'global_step': 8000, 'ema': {'shadow_params': [torch.ones(2)]}}, chosen)
            torch.save({'global_step': 7500, 'ema': {'shadow_params': [torch.zeros(2)]}},
                       source / 'checkpoints/step8000-looking-name.ckpt')
            self.assertEqual(launch.select_mmd(root, 'source-uid', 8000), chosen.resolve())
            with self.assertRaises(ValueError):
                launch.select_mmd(root, 'wrong-run', 8000)
            with self.assertRaises(ValueError):
                launch.select_mmd(root, 'source-uid', 8500)
            torch.save({'global_step': 8000}, chosen)
            with self.assertRaisesRegex(ValueError, 'no EMA'):
                launch.select_mmd(root, 'source-uid', 8000)

    def test_two_gpu_config_keeps_global_batch_and_new_teacher(self):
        from cdlm_preflight import compose_config
        repo = Path(launch.__file__).resolve().parent
        assets = {key: Path('/tmp') / key for key in ('cache', 'bundle', 'data')}
        cmd = launch.command(repo, Path('/tmp/8000.ckpt'), 2, assets, Path('/tmp/new-cdlm'), 8000, [])
        cfg = compose_config(['+experiment=cdlm_tinygsm_mmd', *cmd[2:]], code_dir=repo)
        self.assertEqual(cfg.training.finetune_path, '/tmp/8000.ckpt')
        self.assertTrue(cfg.training.teacher_use_ema)
        self.assertIsNone(cfg.training.student_init_path)
        self.assertFalse(cfg.checkpointing.resume_from_ckpt)
        self.assertEqual(cfg.trainer.max_epochs, 16)
        self.assertEqual(cfg.loader.batch_size * cfg.trainer.devices * cfg.trainer.accumulate_grad_batches, 512)
        self.assertEqual(cfg.trainer.val_check_interval, 4000)
        self.assertEqual(cfg.callbacks.checkpoint_every_n_steps.every_n_train_steps, 500)
        self.assertEqual(cfg.eval.gsm8k_pareto_thresholds[-3:], [.995, .998, .999])
        self.assertFalse(cfg.eval.gsm8k_greedy_token)
        self.assertEqual(cfg.data.separator, r'\n')


if __name__ == '__main__':
    unittest.main()
