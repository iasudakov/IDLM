"""Retain evaluated training models, using the optimizer-step evaluation axis."""
import numbers
import os

from lightning.pytorch import Callback
from lightning.fabric.utilities.cloud_io import get_filesystem


class EvaluatedCheckpoint(Callback):
  """Save after each fit validation at or beyond ``start_step``.

  Runs after the module's validation-epoch-end hook has restored live weights.
  ``Trainer.save_checkpoint`` also invokes the module's EMA-saving hooks, so
  the file retains both live weights and the EMA weights used for evaluation.
  This policy includes NLL-only validations; it does not gate on a particular
  benchmark or Pareto metric. Existing checkpoint callbacks are independent.
  """

  def __init__(self, dirpath, start_step=0, snapshot_dir=None, snapshot=False):
    if isinstance(start_step, bool) or not isinstance(start_step, numbers.Integral):
      raise ValueError('checkpointing.save_evaluated_from_step must be a nonnegative integer')
    if start_step < 0:
      raise ValueError('checkpointing.save_evaluated_from_step must be a nonnegative integer')
    self.dirpath = os.fspath(dirpath)
    self.start_step = int(start_step)
    self.snapshot_dir = os.fspath(snapshot_dir or dirpath)
    self.snapshot = bool(snapshot)
    self._last_saved_step = -1

  def state_dict(self):
    return {'last_saved_step': self._last_saved_step}

  def load_state_dict(self, state_dict):
    self._last_saved_step = int(state_dict.get('last_saved_step', -1))

  def _available_path(self, step):
    """Preserve earlier files if a fresh callback uses an existing directory."""
    fs = get_filesystem(self.dirpath)
    fs.makedirs(self.dirpath, exist_ok=True)
    stem = os.path.join(self.dirpath, f'eval-step={step:08d}')
    path = f'{stem}.ckpt'
    version = 1
    while fs.exists(path):
      path = f'{stem}-v{version}.ckpt'
      version += 1
    return path

  def on_validation_end(self, trainer, pl_module):
    state_fn = getattr(getattr(trainer, 'state', None), 'fn', None)
    if getattr(state_fn, 'value', state_fn) != 'fit' or trainer.sanity_checking:
      return
    step = int(trainer.global_step)
    if step < self.start_step or step <= self._last_saved_step:
      return

    # Pick a single path on rank zero. Every rank must then enter checkpoint
    # saving: distributed strategies may synchronize inside save_checkpoint.
    result = None
    if trainer.is_global_zero:
      try:
        result = (self._available_path(step), None)
      except Exception as error:
        result = (None, f'{type(error).__name__}: {error}')
    path, error = trainer.strategy.broadcast(result)
    if error is not None:
      raise RuntimeError(f'Cannot prepare evaluated checkpoint path: {error}')
    previous = self._last_saved_step
    # Include this save in the serialized callback state, preventing duplicate
    # validation files when this checkpoint is resumed.
    self._last_saved_step = step
    try:
      trainer.save_checkpoint(path, weights_only=False)
    except Exception:
      self._last_saved_step = previous
      raise
    if trainer.is_global_zero:
      print(f'Saved evaluated checkpoint at optimizer_step={step}: {path}', flush=True)
      if self.snapshot:
        import nirvana_snapshot
        nirvana_snapshot.save(self.snapshot_dir, enabled=True,
                              reason=f'evaluated optimizer_step={step}')
