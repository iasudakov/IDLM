"""Exercise the real Lightning/SDTT lifecycle using a small CPU denoiser.

The production DIT uses CUDA flash attention. Replacing only that backbone
lets these tests exercise loss/backward, accumulation, EMA and full resume.
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import lightning as L
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from sdtt import SDTT

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    vocab_size = 7
    mask_token = None
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2


class CPUDenoiser(torch.nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.output = torch.nn.Linear(8, vocab_size)

    def forward(self, x, sigma):
        hidden = self.embedding(x)
        return self.output(hidden + hidden.mean(dim=1, keepdim=True))


def config():
    resolvers = {'cwd': lambda: str(ROOT), 'device_count': lambda: 1,
                 'eval': eval, 'div_up': lambda x, y: (x + y - 1) // y}
    for name, fn in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
        return compose(config_name='config', overrides=[
            '+experiment=sdtt_tinygsm', 'trainer.accelerator=cpu', 'trainer.devices=1',
            'loader.batch_size=2', 'loader.global_batch_size=4', 'loader.num_workers=0',
            'eval.gsm8k_at_validation=false', 'eval.validate_before_train=false',
            'algo.orig_num_sampling_steps=8', 'algo.min_num_sampling_steps=1',
            'algo.grow_dt_every=1', 'lr_scheduler.num_warmup_steps=0', 'optim.lr=.01',
            'model.length=6'])


def loader():
    return DataLoader([
        {'input_ids': torch.tensor([0, 4, 5, 6, 1, 2]),
         'attention_mask': torch.tensor([0, 0, 1, 1, 1, 1])}
        for _ in range(24)], batch_size=2)


def trainer(max_steps, **kwargs):
    return L.Trainer(accelerator='cpu', devices=1, max_steps=max_steps,
                     accumulate_grad_batches=2, logger=False, enable_checkpointing=False,
                     enable_progress_bar=False, enable_model_summary=False,
                     limit_val_batches=0, num_sanity_val_steps=0, **kwargs)


class SDTTTrainingTests(unittest.TestCase):
    def test_warm_start_training_teacher_and_resume(self):
        with patch('models.dit.DIT', CPUDenoiser), tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(11)
            cfg = config()
            model = SDTT(cfg, Tokenizer())
            # Warm start loads raw backbone weights, not a serialized tokenizer,
            # teacher, optimizer, or a stale pretrained EMA.
            pretrained = Path(tmp) / 'mdlm.ckpt'
            torch.save({'state_dict': {'backbone.' + k: v.clone()
                                      for k, v in model.backbone.state_dict().items()}}, pretrained)
            model.initialize_from_checkpoint(str(pretrained))
            for live, shadow in zip(model._get_parameters(), model.ema.shadow_params):
                self.assertTrue(torch.equal(live, shadow))
            initial_teacher = {k: v.clone() for k, v in model.teacher.state_dict().items()}
            with torch.no_grad():
                model.backbone.output.bias[0].add_(.7)
            fit = trainer(2)
            fit.fit(model, loader())
            self.assertEqual(fit.global_step, 2)
            self.assertEqual(model.ema.num_updates, 2)
            self.assertEqual(int(model._sdtt_round), 1)
            self.assertTrue(any(not torch.equal(v, initial_teacher[k])
                                for k, v in model.teacher.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
            self.assertFalse(model.teacher.training)
            checkpoint = Path(tmp) / 'resume.ckpt'
            fit.save_checkpoint(checkpoint)
            saved = torch.load(checkpoint, weights_only=False)
            self.assertTrue(any(k.startswith('teacher.') for k in saved['state_dict']))
            self.assertTrue(saved['optimizer_states'])
            restored = SDTT(cfg, Tokenizer())
            resumed = trainer(4)
            resumed.fit(restored, loader(), ckpt_path=str(checkpoint))
            self.assertEqual(resumed.global_step, 4)
            self.assertEqual(restored.ema.num_updates, 4)
            self.assertEqual(int(restored._sdtt_round), 2)
            # Inference uses the saved EMA exactly once and needs no teacher.
            inference = SDTT(cfg, Tokenizer(), inference_only=True)
            inference.load_evaluation_weights(str(checkpoint), use_ema=True)
            self.assertFalse(hasattr(inference, 'teacher'))
            self.assertIsNone(inference.ema)
            for actual, shadow in zip(inference._get_parameters(), saved['ema']['shadow_params']):
                self.assertTrue(torch.equal(actual, shadow))
            self.assertEqual(inference._evaluation_weights, 'ema')

    def test_validation_restores_live_student_and_keeps_teacher_frozen(self):
        with patch('models.dit.DIT', CPUDenoiser):
            model = SDTT(config(), Tokenizer())
            model._trainer = SimpleNamespace(sanity_checking=False)
            model.log = lambda *args, **kwargs: None
            before = [p.detach().clone() for p in model._get_parameters()]
            for shadow in model.ema.shadow_params:
                shadow.add_(2)
            with patch('sdtt_eval.log_validation_gsm8k') as evaluate:
                model.on_validation_epoch_start()
                model.metrics.update_valid(torch.tensor(1.), torch.tensor(0.), torch.tensor(1.))
                model.on_validation_epoch_end()
                evaluate.assert_called_once_with(model)
            for actual, expected in zip(model._get_parameters(), before):
                self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(model.teacher.training)


if __name__ == '__main__':
    unittest.main()
