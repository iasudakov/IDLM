"""CPU checks for thresholded validation checkpoint saving; no model downloads."""
import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L
from lightning.fabric.utilities.cloud_io import get_filesystem
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluated_checkpoint import EvaluatedCheckpoint


class ToyModel(L.LightningModule):
  def __init__(self):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.tensor(1.0))
    self.register_buffer('ema_shadow', torch.tensor(7.0))

  def training_step(self, batch, batch_idx):
    return (self.weight - batch[0]).square().mean()

  def validation_step(self, batch, batch_idx):
    self.log('val/loss', (self.weight - batch[0]).square().mean(), sync_dist=True)

  def on_validation_epoch_start(self):
    self._live = self.weight.detach().clone()
    self.weight.data.copy_(self.ema_shadow)

  def on_validation_epoch_end(self):
    self.weight.data.copy_(self._live)

  def on_save_checkpoint(self, checkpoint):
    # Saving must occur after the evaluation hook restores live weights.
    assert torch.equal(self.weight.detach(), self._live)
    assert not torch.equal(self.weight.detach(), self.ema_shadow)
    checkpoint['ema'] = {'shadow_params': [self.ema_shadow.detach().clone()]}

  def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=.05)


def loaders():
  data = TensorDataset(torch.zeros(16))
  return DataLoader(data, batch_size=1), DataLoader(data, batch_size=8)


def trainer(callback, max_steps, **kwargs):
  return L.Trainer(accelerator='cpu', devices=1, max_steps=max_steps,
      val_check_interval=2, num_sanity_val_steps=2, logger=False,
      enable_progress_bar=False, enable_model_summary=False,
      enable_checkpointing=False, callbacks=[callback], **kwargs)


def fake_trainer(step=0, rank_zero=True, broadcast_result=None):
  return SimpleNamespace(global_step=step, sanity_checking=False,
      state=SimpleNamespace(fn='fit'), is_global_zero=rank_zero,
      strategy=SimpleNamespace(broadcast=Mock(side_effect=lambda p: p or broadcast_result)),
      save_checkpoint=Mock())


class CheckpointTests(unittest.TestCase):
  def test_inclusive_threshold_context_and_duplicate_suppression(self):
    with tempfile.TemporaryDirectory() as directory:
      callback = EvaluatedCheckpoint(directory, start_step=1000)
      t = fake_trainer(999)
      callback.on_validation_end(t, None)
      t.global_step = 1000
      t.sanity_checking = True
      callback.on_validation_end(t, None)
      t.sanity_checking = False
      t.state.fn = 'validate'
      callback.on_validation_end(t, None)
      t.save_checkpoint.assert_not_called()
      t.state.fn = 'fit'
      callback.on_validation_end(t, None)
      t.save_checkpoint.assert_called_once_with(
          str(Path(directory) / 'eval-step=00001000.ckpt'), weights_only=False)
      callback.on_validation_end(t, None)
      self.assertEqual(t.save_checkpoint.call_count, 1)
      restored = EvaluatedCheckpoint(directory, start_step=1000)
      restored.load_state_dict(callback.state_dict())
      restored.on_validation_end(t, None)
      self.assertEqual(t.save_checkpoint.call_count, 1)
      t.global_step = 1125
      restored.on_validation_end(t, None)
      self.assertEqual(t.save_checkpoint.call_count, 2)

  def test_nonzero_rank_enters_save_and_errors_do_not_mark_saved(self):
    callback = EvaluatedCheckpoint('/unused', start_step=5)
    t = fake_trainer(5, rank_zero=False,
        broadcast_result=('/rank-zero/eval-step=00000005.ckpt', None))
    callback.on_validation_end(t, None)
    t.strategy.broadcast.assert_called_once_with(None)
    t.save_checkpoint.assert_called_once_with('/rank-zero/eval-step=00000005.ckpt', weights_only=False)
    t.global_step = 6
    t.save_checkpoint.side_effect = OSError('simulated disk error')
    with self.assertRaises(OSError):
      callback.on_validation_end(t, None)
    self.assertEqual(callback.state_dict()['last_saved_step'], 5)

  def test_path_errors_reach_all_ranks_without_marking_saved(self):
    callback = EvaluatedCheckpoint('/unused')
    callback._available_path = Mock(side_effect=OSError('simulated path error'))
    leader = fake_trainer(5)
    with self.assertRaisesRegex(RuntimeError, 'simulated path error'):
      callback.on_validation_end(leader, None)
    leader.save_checkpoint.assert_not_called()
    result = leader.strategy.broadcast.call_args.args[0]
    follower = fake_trainer(5, rank_zero=False, broadcast_result=result)
    with self.assertRaisesRegex(RuntimeError, 'simulated path error'):
      callback.on_validation_end(follower, None)
    follower.save_checkpoint.assert_not_called()
    self.assertEqual(callback.state_dict()['last_saved_step'], -1)

  def test_remote_files_are_preserved(self):
    directory = f'memory://evaluated-checkpoint-test-{uuid4().hex}'
    fs = get_filesystem(directory)
    existing = f'{directory}/eval-step=00000005.ckpt'
    try:
      fs.makedirs(directory, exist_ok=True)
      with fs.open(existing, 'wb') as handle:
        handle.write(b'keep me')
      callback = EvaluatedCheckpoint(directory, start_step=5)
      t = fake_trainer(5)
      callback.on_validation_end(t, None)
      t.save_checkpoint.assert_called_once_with(
          f'{directory}/eval-step=00000005-v1.ckpt', weights_only=False)
      with fs.open(existing, 'rb') as handle:
        self.assertEqual(handle.read(), b'keep me')
    finally:
      fs.rm(directory, recursive=True)

  def test_existing_file_is_preserved_and_invalid_thresholds_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      existing = Path(directory) / 'eval-step=00000005.ckpt'
      existing.write_text('keep me')
      callback = EvaluatedCheckpoint(directory, start_step=5)
      t = fake_trainer(5)
      callback.on_validation_end(t, None)
      self.assertEqual(existing.read_text(), 'keep me')
      t.save_checkpoint.assert_called_once_with(
          str(Path(directory) / 'eval-step=00000005-v1.ckpt'), weights_only=False)
      for invalid in (-1, 1.5, True, '1000'):
        with self.assertRaises(ValueError):
          EvaluatedCheckpoint(directory, start_step=invalid)

  def test_real_validation_saves_full_state_after_ema_restore_and_resumes(self):
    with tempfile.TemporaryDirectory() as directory:
      callback = EvaluatedCheckpoint(directory, start_step=3)
      t = trainer(callback, 6)
      t.fit(ToyModel(), *loaders())
      files = sorted(Path(directory).glob('*.ckpt'))
      self.assertEqual([p.name for p in files], ['eval-step=00000004.ckpt', 'eval-step=00000006.ckpt'])
      for path, step in zip(files, (4, 6)):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(ckpt['global_step'], step)
        self.assertTrue(ckpt['optimizer_states'])
        self.assertEqual(ckpt['callbacks'][callback.state_key]['last_saved_step'], step)
        self.assertEqual(ckpt['ema']['shadow_params'][0].item(), 7.)
        self.assertNotEqual(ckpt['state_dict']['weight'].item(), 7.)
      old_hash = hashlib.sha256(files[-1].read_bytes()).hexdigest()
      restored = EvaluatedCheckpoint(directory, start_step=3)
      trainer(restored, 8).fit(ToyModel(), *loaders(), ckpt_path=str(files[-1]))
      self.assertEqual(restored.state_dict()['last_saved_step'], 8)
      self.assertEqual(hashlib.sha256(files[-1].read_bytes()).hexdigest(), old_hash)
      self.assertEqual(len(list(Path(directory).glob('*.ckpt'))), 3)


def ddp_smoke():
  with tempfile.TemporaryDirectory() as directory:
    callback = EvaluatedCheckpoint(directory, start_step=2)
    t = L.Trainer(accelerator='cpu', devices=2, strategy='ddp_fork',
        max_steps=2, val_check_interval=2, num_sanity_val_steps=0,
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        enable_checkpointing=False, callbacks=[callback])
    t.fit(ToyModel(), *loaders())
    files = list(Path(directory).glob('*.ckpt'))
    assert len(files) == 1, files
    ckpt = torch.load(files[0], map_location='cpu', weights_only=False)
    assert ckpt['global_step'] == 2
    assert ckpt['callbacks'][callback.state_key]['last_saved_step'] == 2
    assert ckpt['ema']['shadow_params'][0].item() == 7.
    print('Two-rank CPU DDP checkpoint save passed.', flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ddp-smoke', action='store_true')
  args, rest = parser.parse_known_args()
  if args.ddp_smoke:
    ddp_smoke()
  else:
    unittest.main(argv=[sys.argv[0]]+rest)
