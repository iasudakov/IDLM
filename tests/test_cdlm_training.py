"""Real CDLM/Lightning/cache lifecycle on CPU, replacing only the DiT backbone."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import lightning as L
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset

from cdlm import CDLM
from cdlm_trajectories import (CDLMTrajectoryDataset, build_cache_identity,
                               prepare_trajectories)
import main

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    vocab_size = 7
    mask_token = None
    mask_token_id = None
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def get_vocab(self):
        return {str(index): index for index in range(self.vocab_size)}


class Output(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.linear = torch.nn.Linear(8, vocab)

    def forward(self, hidden):
        return self.linear(hidden)


class CPUDenoiser(torch.nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.output_layer = Output(vocab_size)

    def forward(self, x, sigma):
        hidden = self.embedding(x)
        return self.output_layer(hidden + hidden.mean(1, keepdim=True))


class Original(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        return {'input_ids': torch.tensor([0, 4, 5, 6, 1, 2]),
                'attention_mask': torch.tensor([0, 0, 1, 1, 1, 1])}


def config(directory):
    for name, fn in {'cwd': lambda: str(ROOT), 'device_count': lambda: 1,
                     'eval': eval, 'div_up': lambda x, y: (x + y - 1) // y}.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
        cfg = compose(config_name='config', overrides=[
            '+experiment=cdlm_tinygsm', 'trainer.accelerator=cpu', 'trainer.devices=1',
            'loader.batch_size=2', 'loader.global_batch_size=4', 'loader.num_workers=0',
            'eval.gsm8k_at_validation=false', 'eval.validate_before_train=false',
            'model.length=6', 'model.hidden_size=8', 'model.n_blocks=1', 'model.n_heads=2',
            'algo.segment_length=2', 'algo.trajectory_num_examples=4',
            'algo.trajectory_batch_size=2', 'algo.trajectory_shard_size=2', 'optim.lr=.01'])
    cfg.training.finetune_path = str(Path(directory) / 'original.ckpt')
    cfg.data.cache_dir = str(Path(directory) / 'data-cache')
    cfg.algo.trajectory_cache_dir = str(Path(directory) / 'trajectories')
    OmegaConf.update(cfg, 'algo.trajectory_cache_per_model', False, force_add=True)
    OmegaConf.update(cfg, 'checkpointing.snapshot_trajectories', False, force_add=True)
    cfg.checkpointing.save_dir = str(directory)
    return cfg


def trainer(steps):
    return L.Trainer(accelerator='cpu', devices=1, max_steps=steps, max_epochs=-1,
                     accumulate_grad_batches=2, logger=False, enable_checkpointing=False,
                     enable_progress_bar=False, enable_model_summary=False,
                     limit_val_batches=0, num_sanity_val_steps=0)


class TrainingTest(unittest.TestCase):
    def test_prepare_train_frozen_head_full_resume_and_ema_inference(self):
        with patch('models.dit.DIT', CPUDenoiser), tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(12)
            cfg = config(directory)
            tokenizer = Tokenizer()
            model = CDLM(cfg, tokenizer)
            torch.save({'state_dict': {'backbone.' + name: tensor.clone()
                                      for name, tensor in model.backbone.state_dict().items()}},
                       cfg.training.finetune_path)
            model.initialize_from_checkpoint(cfg.training.finetune_path)
            collector = CDLM(cfg, tokenizer, inference_only=True)
            collector.load_evaluation_weights(cfg.training.finetune_path, use_ema=False)
            prepare_trajectories(collector, cfg, tokenizer, Original())
            identity = build_cache_identity(cfg, tokenizer, Original())
            dataset = CDLMTrajectoryDataset(cfg.algo.trajectory_cache_dir, identity)
            loader = DataLoader(dataset, batch_size=2)
            model._cache_fingerprint = identity['fingerprint']
            head = {name: tensor.clone() for name, tensor in model.teacher_head.state_dict().items()}
            original_parameters = [p.clone() for p in model.backbone.parameters()]
            fit = trainer(2)
            fit.fit(model, loader)
            self.assertEqual(fit.global_step, 2)
            self.assertEqual(model.ema.num_updates, 2)
            self.assertTrue(any(not torch.equal(before, after)
                                for before, after in zip(original_parameters, model.backbone.parameters())))
            self.assertTrue(all(torch.equal(tensor, head[name])
                                for name, tensor in model.teacher_head.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.teacher_head.parameters()))
            last = Path(directory) / 'checkpoints/last.ckpt'
            saved = torch.load(last, weights_only=False)
            self.assertEqual(saved['global_step'], 2)
            self.assertEqual(saved['cdlm_cache_fingerprint'], identity['fingerprint'])
            self.assertTrue(saved['optimizer_states'])
            restored = CDLM(cfg, tokenizer)
            restored._cache_fingerprint = identity['fingerprint']
            resumed = trainer(3)
            resumed.fit(restored, loader, ckpt_path=str(last))
            self.assertEqual(restored.ema.num_updates, 3)
            self.assertEqual(torch.load(last, weights_only=False)['global_step'], 3)
            inference = CDLM(cfg, tokenizer, inference_only=True)
            inference.load_evaluation_weights(str(last))
            latest = torch.load(last, weights_only=False)
            for actual, shadow in zip(inference._get_parameters(), latest['ema']['shadow_params']):
                self.assertTrue(torch.equal(actual, shadow))
            self.assertFalse(hasattr(inference, 'teacher_head'))
            self.assertEqual(inference._evaluation_weights, 'ema')
            restored._cache_fingerprint = 'wrong-cache'
            with self.assertRaisesRegex(ValueError, 'different trajectory cache'):
                restored.on_load_checkpoint(latest)

    def test_author_warmup_ratio_uses_finite_total_updates(self):
        with patch('models.dit.DIT', CPUDenoiser), tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            model = CDLM(cfg, Tokenizer())
            model._trainer = SimpleNamespace(estimated_stepping_batches=1248)
            model.configure_optimizers()
            self.assertEqual(cfg.lr_scheduler.num_warmup_steps, 62)

    def test_main_cache_loader_validates_identity_and_keeps_full_global_batches(self):
        with patch('models.dit.DIT', CPUDenoiser), tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            tokenizer = Tokenizer()
            teacher = CDLM(cfg, tokenizer, inference_only=True)
            torch.save({'state_dict': {'backbone.' + name: tensor.clone()
                                      for name, tensor in teacher.backbone.state_dict().items()}},
                       cfg.training.finetune_path)
            teacher._evaluation_weights = 'live'
            prepare_trajectories(teacher, cfg, tokenizer, Original())
            loader, fingerprint = main._cdlm_train_loader(cfg, tokenizer, DataLoader(Original()))
            self.assertEqual(len(loader.dataset), 8)
            self.assertEqual(len(loader), 4)
            self.assertEqual(len(next(iter(loader))['input_ids']), 2)
            self.assertTrue(fingerprint)
            cfg.seed = 999
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                main._cdlm_train_loader(cfg, tokenizer, DataLoader(Original()))

    def test_finite_epochs_validate_across_boundaries_and_save_final_update(self):
        with patch('models.dit.DIT', CPUDenoiser), tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            tokenizer = Tokenizer()
            teacher = CDLM(cfg, tokenizer, inference_only=True)
            torch.save({'state_dict': {'backbone.' + name: tensor.clone()
                                      for name, tensor in teacher.backbone.state_dict().items()}},
                       cfg.training.finetune_path)
            teacher._evaluation_weights = 'live'
            prepare_trajectories(teacher, cfg, tokenizer, Original())
            identity = build_cache_identity(cfg, tokenizer, Original())
            loader = DataLoader(CDLMTrajectoryDataset(cfg.algo.trajectory_cache_dir, identity), batch_size=2)
            model = CDLM(cfg, tokenizer)
            model.initialize_from_checkpoint(cfg.training.finetune_path)
            model._cache_fingerprint = identity['fingerprint']
            periodic = L.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(Path(directory) / 'checkpoints'), every_n_train_steps=1000,
                save_top_k=-1, save_last=True, save_on_train_epoch_end=True)
            fit = L.Trainer(accelerator='cpu', devices=1, max_steps=-1, max_epochs=2,
                            accumulate_grad_batches=2, check_val_every_n_epoch=None,
                            val_check_interval=6, limit_val_batches=1, num_sanity_val_steps=0,
                            logger=False, callbacks=[periodic], enable_progress_bar=False,
                            enable_model_summary=False)
            calls = []
            with patch('sdtt_eval.log_validation_gsm8k',
                       side_effect=lambda module, **kwargs: calls.append((module.global_step, kwargs))):
                fit.fit(model, loader, DataLoader(Original(), batch_size=2))
            self.assertEqual(fit.global_step, 4)
            self.assertIn((3, {}), calls)
            self.assertIn((4, {'final': True}), calls)
            last = torch.load(Path(directory) / 'checkpoints/last.ckpt', weights_only=False)
            self.assertEqual(last['global_step'], 4)
            self.assertEqual(model.ema.num_updates, 4)


if __name__ == '__main__':
    unittest.main()
