"""Preparation and training select the same isolated teacher trajectory cache."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from cdlm import CDLM
from cdlm_trajectories import prepare_trajectories, read_cache_manifest
import main
from test_cdlm_training import CPUDenoiser, Original, Tokenizer, config as make_config


class TeacherCacheEntrypointTest(unittest.TestCase):
    def test_mdlm_raw_and_mmd_ema_collect_and_load_separate_model_folders(self):
        with patch('models.dit.DIT', CPUDenoiser), tempfile.TemporaryDirectory() as temporary:
            cfg = make_config(temporary)
            cfg.algo.trajectory_cache_per_model = True
            cfg.checkpointing.snapshot = False
            tokenizer = Tokenizer()
            original = CDLM(cfg, tokenizer, inference_only=True)
            state = {'backbone.' + name: tensor.clone()
                     for name, tensor in original.backbone.state_dict().items()}
            torch.save({'state_dict': state}, cfg.training.finetune_path)
            original.load_teacher_weights(cfg.training.finetune_path)
            mdlm_cache = prepare_trajectories(original, cfg, tokenizer, Original())
            mdlm_manifest = read_cache_manifest(mdlm_cache)
            self.assertEqual(mdlm_manifest['identity']['parameters']['teacher_weights'], 'raw')

            mmd_checkpoint = Path(temporary) / 'published-mmd.ckpt'
            torch.save({'state_dict': {name: tensor + .125 for name, tensor in state.items()},
                        'checkpoint_weights': 'ema'}, mmd_checkpoint)
            cfg.training.finetune_path = str(mmd_checkpoint)
            cfg.training.teacher_use_ema = True
            mmd = CDLM(cfg, tokenizer, inference_only=True)
            mmd.load_teacher_weights(cfg.training.finetune_path)
            mmd_cache = prepare_trajectories(mmd, cfg, tokenizer, Original())
            mmd_manifest = read_cache_manifest(mmd_cache)
            self.assertNotEqual(mdlm_cache, mmd_cache)
            self.assertEqual(mmd_manifest['identity']['parameters']['teacher_weights'], 'ema')
            self.assertEqual(Path(mmd_cache).parent.name, 'ema')
            self.assertEqual(read_cache_manifest(mdlm_cache), mdlm_manifest)

            loader, fingerprint = main._cdlm_train_loader(cfg, tokenizer, DataLoader(Original()))
            self.assertEqual(fingerprint, mmd_manifest['identity']['fingerprint'])
            self.assertEqual(loader.dataset.dataset.directory, Path(mmd_cache))
            self.assertEqual(len(loader.dataset), 8)
            self.assertEqual(len(next(iter(loader))['teacher_hidden']), 2)


if __name__ == '__main__':
    unittest.main()
