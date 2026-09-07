"""The existing Underdeep transport, isolated from training's random state."""

import functools

from underdeep_logger import UnderdeepLogger
from .monitor import preserve_rng_state


def isolated(method):
  @functools.wraps(method)
  def wrapped(*args, **kwargs):
    with preserve_rng_state():
      return method(*args, **kwargs)
  return wrapped


class DiDiUnderdeepLogger(UnderdeepLogger):
  log_metrics = isolated(UnderdeepLogger.log_metrics)
  log_hyperparams = isolated(UnderdeepLogger.log_hyperparams)
  log_table = isolated(UnderdeepLogger.log_table)
  log_image = isolated(UnderdeepLogger.log_image)
  finalize = isolated(UnderdeepLogger.finalize)

  @property
  def experiment(self):
    with preserve_rng_state():
      return super().experiment
