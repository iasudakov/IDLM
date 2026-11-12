"""CPU validation scheduling checks; no datasets, downloads, GPUs or tracker."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluated_checkpoint import EvaluatedCheckpoint
from validation_schedule import ValidationStart


class ToyModel(L.LightningModule):
  def __init__(self, accumulation=2, manual=True, fail=False):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.tensor(1.0))
    self.automatic_optimization = not manual
    self.accumulation = accumulation
    self.fail = fail
    self.evaluations = []
    self.validation_batches = []

  def training_step(self, batch, batch_idx):
    if self.fail:
      raise RuntimeError('synthetic training failure')
    loss = (self.weight - batch[0]).square().mean()
    if self.automatic_optimization:
      return loss
    optimizer = self.optimizers()
    self.manual_backward(loss / self.accumulation)
    if (batch_idx + 1) % self.accumulation == 0:
      optimizer.step()
      optimizer.zero_grad()

  def on_validation_epoch_start(self):
    self.evaluations.append((self.global_step, self.trainer.sanity_checking))

  def validation_step(self, batch, batch_idx):
    self.validation_batches.append(self.global_step)
    self.log('val/loss', self.weight.square(), batch_size=1)

  def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=.05)


def loader():
  return DataLoader(TensorDataset(torch.zeros(128)), batch_size=1)


def trainer(directory, start, steps=4, interval=2, **kwargs):
  return L.Trainer(accelerator='cpu', devices=1, max_steps=steps,
      logger=False, enable_progress_bar=False, enable_model_summary=False,
      enable_checkpointing=False, default_root_dir=directory,
      limit_val_batches=2, num_sanity_val_steps=2,
      val_check_interval=interval,
      callbacks=[ValidationStart(start), EvaluatedCheckpoint(directory, start_step=1)],
      **kwargs)


class ValidationScheduleTests(unittest.TestCase):
  def test_manual_accumulation_skips_whole_loops_and_saves_only_evaluated_steps(self):
    with tempfile.TemporaryDirectory() as directory:
      model = ToyModel()
      t = trainer(directory, start=3)
      t.fit(model, loader(), loader())
      self.assertEqual(model.evaluations, [(3, False), (4, False)])
      self.assertEqual(model.validation_batches, [3, 3, 4, 4])
      self.assertEqual(sorted(p.name for p in Path(directory).glob('*.ckpt')),
          ['eval-step=00000003.ckpt', 'eval-step=00000004.ckpt'])
      self.assertEqual(t.limit_val_batches, 2)

  def test_non_aligned_threshold_keeps_original_cadence(self):
    with tempfile.TemporaryDirectory() as directory:
      model = ToyModel()
      trainer(directory, start=3, steps=6, interval=4).fit(model, loader(), loader())
      self.assertEqual(model.evaluations, [(4, False), (6, False)])

  def test_zero_preserves_sanity_and_periodic_evaluation(self):
    with tempfile.TemporaryDirectory() as directory:
      model = ToyModel()
      trainer(directory, start=0, steps=2).fit(model, loader(), loader())
      self.assertEqual(model.evaluations, [(0, True), (1, False), (2, False)])

  def test_threshold_above_budget_and_standalone_validation(self):
    with tempfile.TemporaryDirectory() as directory:
      model = ToyModel()
      t = trainer(directory, start=10, steps=2)
      t.fit(model, loader(), loader())
      self.assertEqual(model.evaluations, [])
      self.assertEqual(model.validation_batches, [])
      self.assertFalse(list(Path(directory).glob('*.ckpt')))
      t.validate(model, loader(), verbose=False)
      self.assertEqual(model.evaluations, [(2, False)])
      self.assertEqual(len(model.validation_batches), 2)

  def test_automatic_accumulation_uses_optimizer_steps(self):
    with tempfile.TemporaryDirectory() as directory:
      model = ToyModel(manual=False)
      trainer(directory, start=3, accumulate_grad_batches=2).fit(model, loader(), loader())
      self.assertEqual(model.evaluations, [(3, False), (4, False)])

  def test_resume_before_and_after_threshold(self):
    with tempfile.TemporaryDirectory() as directory:
      t = trainer(directory, start=3, steps=2)
      t.fit(ToyModel(), loader(), loader())
      early_path = str(Path(directory) / 'early.ckpt')
      t.save_checkpoint(early_path)
      before = ToyModel()
      trainer(directory, start=3, steps=4).fit(before, loader(), loader(), ckpt_path=early_path)
      self.assertEqual(before.evaluations, [(3, False), (4, False)])
      after = ToyModel()
      trainer(directory, start=3, steps=6).fit(after, loader(), loader(),
          ckpt_path=str(Path(directory) / 'eval-step=00000004.ckpt'))
      # Lightning replays the current validation when resuming a checkpoint
      # saved inside on_validation_end; EvaluatedCheckpoint suppresses a
      # duplicate save. This is existing Lightning behavior, not a new cadence.
      self.assertEqual(after.evaluations, [(4, False), (5, False), (6, False)])
      self.assertEqual(len(list(Path(directory).glob('eval-step=00000004*.ckpt'))), 1)
      delayed = ToyModel()
      trainer(directory, start=6, steps=6).fit(delayed, loader(), loader(),
          ckpt_path=str(Path(directory) / 'eval-step=00000004.ckpt'))
      self.assertEqual(delayed.evaluations, [(6, False)])

  def test_failure_restores_original_validation_limit(self):
    with tempfile.TemporaryDirectory() as directory:
      t = trainer(directory, start=3)
      with self.assertRaisesRegex(RuntimeError, 'synthetic training failure'):
        t.fit(ToyModel(fail=True), loader(), loader())
      self.assertEqual(t.limit_val_batches, 2)

  def test_invalid_thresholds(self):
    for invalid in (-1, True, 1.5, '5500', None):
      with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'eval.start_step'):
        ValidationStart(invalid)

  def test_main_wires_callback_and_skips_explicit_baseline(self):
    # Import the entry point without downloading anything. ipdb is optional
    # in the CPU test environment; this only substitutes its unused debugger.
    try:
      import ipdb
    except ImportError:
      import pdb
      sys.modules['ipdb'] = pdb
    import main
    for start in (0, 5500):
      with self.subTest(start=start):
        cfg = OmegaConf.create({
            'training': {'finetune_path': ''},
            'checkpointing': {'save_dir': '/unused', 'resume_from_ckpt': False, 'snapshot': False},
            'eval': {'validate_before_train': True, 'start_step': start},
            'callbacks': {}, 'logger': {}, 'strategy': {}, 'trainer': {}, 'debug': True})
        t = mock.Mock()
        valid = SimpleNamespace(tokenizer=object())
        with mock.patch.object(main.hydra.utils, 'instantiate',
                side_effect=[mock.Mock(), mock.Mock(), t]) as instantiate, \
             mock.patch.object(main.dataloader, 'get_dataloaders', return_value=(object(), valid)), \
             mock.patch.object(main, '_print_batch'):
          main._train(mock.Mock(), cfg, mock.Mock(), object())
        self.assertEqual(t.validate.call_count, int(start == 0))
        t.fit.assert_called_once()
        callbacks = instantiate.call_args.kwargs['callbacks']
        gates = [c for c in callbacks if isinstance(c, ValidationStart)]
        self.assertEqual([c.start_step for c in gates], [5500] if start else [])


if __name__ == '__main__':
  torch.set_num_threads(1)
  unittest.main(verbosity=2)
