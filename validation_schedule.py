"""Delay training validation until an optimizer-step threshold is reached."""
import numbers

from lightning.pytorch import Callback


class ValidationStart(Callback):
  """Gate whole validation loops, leaving Lightning's cadence unchanged.

  ``global_step`` already accounts for manual or automatic accumulation and
  checkpoint resume. Positive thresholds also suppress the initial sanity
  check. Standalone ``Trainer.validate`` calls are unaffected; the explicit
  step-zero baseline in main.py is gated separately.
  """

  def __init__(self, start_step=0):
    if (isinstance(start_step, bool)
        or not isinstance(start_step, numbers.Integral) or start_step < 0):
      raise ValueError('eval.start_step must be a nonnegative integer')
    self.start_step = int(start_step)
    self._limit_val_batches = None

  def on_fit_start(self, trainer, pl_module):
    if self.start_step == 0:
      return
    self._limit_val_batches = trainer.limit_val_batches
    # Training state is restored after this hook. Disable sanity validation
    # now; on_train_start rechecks the restored optimizer step.
    trainer.limit_val_batches = 0

  def _update(self, trainer):
    if self._limit_val_batches is not None:
      trainer.limit_val_batches = (
        self._limit_val_batches
        if trainer.global_step >= self.start_step else 0)

  def on_train_start(self, trainer, pl_module):
    self._update(trainer)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    # This hook follows optimizer.step and precedes the validation decision.
    self._update(trainer)

  def _restore(self, trainer):
    if self._limit_val_batches is not None:
      trainer.limit_val_batches = self._limit_val_batches
      self._limit_val_batches = None

  def on_fit_end(self, trainer, pl_module):
    self._restore(trainer)

  def on_exception(self, trainer, pl_module, exception):
    self._restore(trainer)
