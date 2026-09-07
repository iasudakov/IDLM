"""Pure validation/accounting for the opt-in, iteration-based DiDi recipe."""

from numbers import Integral


def _integer(value, name, minimum=0):
  if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
    raise ValueError(f'{name} must be an integer >= {minimum}; got {value!r}.')
  return int(value)


def _validate_iterations(config, iterations):
  iterations = _integer(iterations, 'didi_training.iterations', minimum=1)
  algo = config.get('algo', {})
  for name, value in (
      ('algo.discriminator_warmup_steps', algo.get('discriminator_warmup_steps', 0)),
      ('algo.lr_scheduler.warmup_steps', algo.get('lr_scheduler', {}).get('warmup_steps', 0))):
    value = _integer(value, name)
    if value >= iterations:
      raise ValueError(f'{name} must be in [0, didi_training.iterations); '
                       f'got {value} with iterations={iterations}.')
  _integer(algo.get('student_update_every', 1), 'algo.student_update_every', minimum=1)
  return iterations


def training_iterations(config):
  """Return the explicitly requested iteration horizon; None preserves legacy."""
  iterations = config.get('didi_training', {}).get('iterations')
  return None if iterations is None else _validate_iterations(config, iterations)


def optimizer_step_budget(config, iterations=None):
  """Map DiDi iterations onto Lightning's individual optimizer-update counter."""
  iterations = (training_iterations(config) if iterations is None
                else _validate_iterations(config, iterations))
  if iterations is None:
    return int(config.get('trainer', {}).get('max_steps', 10000))
  algo = config.get('algo', {})
  warmup = int(algo.get('discriminator_warmup_steps', 0))
  every = int(algo.get('student_update_every', 1))
  first_student = ((warmup + every - 1) // every) * every
  student_updates = max(0, (iterations - 1 - first_student) // every + 1)
  return iterations + student_updates


def training_protocol(config):
  """Resolved schedule/EMA semantics for provenance and safe resume checks.

  Evaluation weight selection is deliberately separate: selecting another
  already-saved weight set must not be interpreted as changing training.
  """
  iterations = training_iterations(config)
  algo = config.get('algo', {})
  scheduler = algo.get('lr_scheduler', {})
  return {
    'iterations': iterations,
    'optimizer_step_budget': optimizer_step_budget(config),
    'lr_horizon': (iterations if iterations is not None
                   else int(config.get('trainer', {}).get('max_steps', 10000))),
    'lr_horizon_units': ('didi_iterations' if iterations is not None
                         else 'legacy_lightning_optimizer_steps'),
    'lr_index_units': 'didi_iterations',
    'lr_warmup_steps': scheduler.get('warmup_steps', 0),
    'lr_schedule_config_name': scheduler.get('name'),
    'lr_schedule_implementation': 'linear_warmup_then_linear_decay',
    'lr_floor': 1e-8,
    'student_peak_lr': config.get('optim', {}).get('lr'),
    'discriminator_peak_lr': algo.get('discriminator_lr', 1e-5),
    'ema_beta': algo.get('ema_beta', 0.999),
    'discriminator_warmup_steps': algo.get('discriminator_warmup_steps', 0),
    'student_update_every': algo.get('student_update_every', 1),
  }
