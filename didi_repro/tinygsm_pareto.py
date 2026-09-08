"""Reference-style TinyGSM accuracy-versus-measured-NFE publication.

The visual layout and metric naming follow tinygsm-upstream-repro's
``trainer_base._publish_pareto`` and ``pareto_plot.py``. The authoritative
table/history retain every threshold and exact floating-point mean NFE. The
tracker's integer-step scalar view is approximate (NFE * 100); colliding
buckets are omitted rather than silently overwriting another threshold.

This module performs no sampling, imports no GPU libraries, and imports
Matplotlib only when an optional figure is requested.
"""

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import tempfile
import uuid


REFERENCE_RENDERER_SHA256 = '1feb9b282c172ca0c7b736cf41b51b14f461a45a584e21ca305b1a5ffd0e4dc2'
NFE_X_SCALE = 100
HISTORY_FILENAME = 'didi_tinygsm_pareto_history.json'


def _finite(name, value, *, minimum=None, maximum=None):
  if isinstance(value, bool):
    raise ValueError(f'{name} must be a finite number, not a boolean.')
  try:
    number = float(value)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError(f'{name} must be a finite number.') from error
  if (not math.isfinite(number) or (minimum is not None and number < minimum)
      or (maximum is not None and number > maximum)):
    raise ValueError(f'{name} is outside its finite allowed range.')
  return number


def _integer(name, value, *, minimum=0):
  number = _finite(name, value, minimum=minimum)
  if not number.is_integer():
    raise ValueError(f'{name} must be an integer.')
  return int(number)


def threshold_tag(value):
  """Keep familiar ``0325`` tags without six-digit ``:g`` collisions."""
  value = _finite('threshold', value, minimum=0, maximum=1)
  text = str(int(value)) if value.is_integer() else repr(value)
  return text.replace('.', '').replace('-', 'm').replace('+', 'p')


def _curve(points):
  if not isinstance(points, (list, tuple)) or not points:
    raise ValueError('A threshold Pareto curve must contain at least one point.')
  result, seen = [], set()
  for point in points:
    if not isinstance(point, dict):
      raise ValueError('Every threshold Pareto point must be a dictionary.')
    # Require JSON-safe extras too, so no partial tracker publication precedes
    # a failing history serialization. All evaluation metrics are CPU scalars.
    try:
      row = json.loads(json.dumps(point, allow_nan=False))
    except (TypeError, ValueError) as error:
      raise ValueError('Pareto points and extra metrics must be finite JSON values.') from error
    for field in ('threshold', 'nfe', 'accuracy'):
      if field not in row:
        raise ValueError(f'Threshold Pareto point is missing {field}.')
    row['threshold'] = _finite('threshold', row['threshold'], minimum=0, maximum=1)
    row['nfe'] = _finite('nfe', row['nfe'], minimum=0)
    row['accuracy'] = _finite('accuracy', row['accuracy'], minimum=0, maximum=1)
    if row['threshold'] in seen:
      raise ValueError('A threshold may occur only once in one checkpoint curve.')
    seen.add(row['threshold'])
    if 'count' in row:
      row['count'] = _integer('count', row['count'], minimum=1)
    if 'correct' in row:
      row['correct'] = _integer('correct', row['correct'])
      if 'count' in row and row['correct'] > row['count']:
        raise ValueError('Pareto correct count cannot exceed the example count.')
    result.append(row)
  return sorted(result, key=lambda row: row['threshold'])


def _training_step(trainer, model):
  context = getattr(model, '_didi_evaluation_step', None)
  value = context['value'] if context is not None else trainer.global_step
  return _integer('training checkpoint step', value)


def _loggers(trainer):
  candidates = list(getattr(trainer, 'loggers', []) or [])
  if not candidates and getattr(trainer, 'logger', None) is not None:
    candidates = [trainer.logger]
  return list({id(logger): logger for logger in candidates}.values())


def _run_uid(trainer, model):
  primary = getattr(trainer, 'logger', None)
  if primary is None:
    loggers = _loggers(trainer)
    primary = loggers[0] if loggers else None
  version = getattr(primary, 'version', None)
  if version is not None and str(version):
    return str(version)
  # An empty tracker UID is not evidence that this process is a resume of an
  # earlier offline run. Keep an identity for this model's lifetime only.
  if not hasattr(model, '_didi_tinygsm_pareto_session'):
    model._didi_tinygsm_pareto_session = 'untracked-' + uuid.uuid4().hex
  return model._didi_tinygsm_pareto_session


def _read_history(path):
  if not path.exists():
    return {'schema_version': 1, 'runs': {}}
  try:
    stored = json.loads(path.read_text())
    if stored.get('schema_version') != 1 or not isinstance(stored.get('runs'), dict):
      raise ValueError('unsupported history schema')
    for run_uid, run in stored['runs'].items():
      if not isinstance(run_uid, str) or not isinstance(run.get('curves'), dict):
        raise ValueError('invalid per-run history')
      for step, curve in run['curves'].items():
        if str(_integer('saved checkpoint step', step)) != step:
          raise ValueError('noncanonical saved checkpoint step')
        _curve(curve)
  except (AttributeError, TypeError, ValueError, OSError) as error:
    raise RuntimeError(f'Cannot read Pareto history; existing file preserved: {path}') from error
  return stored


def _write_history(path, payload):
  """Atomic replace: a serialization/write failure cannot truncate history."""
  serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix='.didi-tinygsm-pareto-',
                                          suffix='.tmp', dir=path.parent)
  try:
    with os.fdopen(descriptor, 'w') as stream:
      stream.write(serialized)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def load_reference(reference):
  """Optional measured teacher frontier; a missing file does not stop training."""
  if reference is None:
    return None
  try:
    if isinstance(reference, (str, os.PathLike)):
      path = Path(reference)
      if not path.is_file():
        return None
      reference = json.loads(path.read_text())
    if isinstance(reference, dict):
      reference = reference['points']
    if not isinstance(reference, (list, tuple)):
      raise ValueError('reference must be a list of points or a points object')
    points = []
    for point in reference:
      row = {'nfe': _finite('reference nfe', point['nfe'], minimum=0),
             'accuracy': _finite('reference accuracy', point['accuracy'], minimum=0, maximum=1)}
      if 'threshold' in point:
        row['threshold'] = _finite('reference threshold', point['threshold'], minimum=0, maximum=1)
      points.append(row)
    return points or None
  except (KeyError, TypeError, ValueError, OSError) as error:
    print(f'[didi-tinygsm-pareto] optional reference ignored: {error}', flush=True)
    return None


def render(history, reference=None, reference_label='Released MDLM teacher',
           title='Confidence threshold frontiers'):
  """Port of the reference figure: viridis checkpoint overlays in true NFE."""
  import matplotlib
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure
  import numpy as np

  if not history:
    raise ValueError('Cannot plot an empty threshold history.')
  normalized = {int(step): points for step, points in history.items()}
  steps = sorted(normalized)
  figure = Figure(figsize=(8, 6), dpi=110)
  canvas = FigureCanvasAgg(figure)
  axis = figure.subplots()
  cmap = matplotlib.colormaps['viridis']
  low, high = steps[0], steps[-1]
  norm = matplotlib.colors.Normalize(vmin=low / 1000, vmax=max(high, low + 1) / 1000)
  for step in steps:
    points = sorted(normalized[step], key=lambda point: point['nfe'])
    axis.plot([point['nfe'] for point in points], [point['accuracy'] * 100 for point in points],
              marker='.', linewidth=1.2, color=cmap(norm(step / 1000)), zorder=2)
  if reference:
    annotations = sorted(reference, key=lambda point: point['nfe'])
    axis.plot([point['nfe'] for point in annotations],
              [point['accuracy'] * 100 for point in annotations],
              marker='D', markersize=5, linestyle='--', color='black', linewidth=1.6,
              label=reference_label, zorder=3)
    axis.legend(loc='lower right', fontsize=9)
  else:
    annotations = sorted(normalized[steps[-1]], key=lambda point: point['nfe'])
  for point in annotations:
    if 'threshold' in point:
      axis.annotate(f"thr {point['threshold']:g}", (point['nfe'], point['accuracy'] * 100),
                    textcoords='offset points', xytext=(6, -3), fontsize=8, color='0.35')
  axis.set_xlabel('Average NFE per sequence')
  axis.set_ylabel('GSM8K accuracy (%)')
  axis.set_title(title, fontweight='bold')
  axis.grid(alpha=.25, zorder=0)
  axis.set_xlim(left=0)
  axis.set_ylim(bottom=0)
  bar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis)
  bar.set_label('Training checkpoint (k steps)')
  figure.tight_layout()
  canvas.draw()
  return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def publish_threshold_pareto(trainer, model, curve, *, reference=None, plot=False):
  """Publish one completed threshold sweep, called on rank zero only.

  Required point fields are threshold, nfe, accuracy. ``nfe`` must already be
  the example-weighted mean of actual per-sequence model forwards, not total
  batch forwards or a requested denoising budget. All extras are preserved.
  """
  if not getattr(trainer, 'is_global_zero', getattr(trainer, 'global_rank', 0) == 0):
    raise RuntimeError('Threshold Pareto publication must run on rank zero only.')
  points = _curve(curve)
  step, run_uid = _training_step(trainer, model), _run_uid(trainer, model)
  path = Path(model.config.checkpointing.save_dir) / HISTORY_FILENAME
  payload = _read_history(path)
  current = payload['runs'].setdefault(run_uid, {'curves': {}})
  current['curves'][str(step)] = points
  current['scalar_x_scale'] = NFE_X_SCALE
  current['renderer_reference_sha256'] = REFERENCE_RENDERER_SHA256
  payload['last_run_uid'] = run_uid
  _write_history(path, payload)

  buckets = defaultdict(list)
  for point in points:
    buckets[int(round(point['nfe'] * NFE_X_SCALE))].append(point)
  collisions = sorted(bucket for bucket, entries in buckets.items() if len(entries) > 1)
  protocol = {
    'training_step': step, 'run_uid': run_uid,
    'x_axis': 'example-weighted mean measured NFE per sequence',
    'batch_forward_totals': 'separate extra metric; never used as the frontier x axis',
    'scalar_x_scale': NFE_X_SCALE,
    'scalar_axis_label': 'integer step / 100 = approximate NFE per sequence',
    'scalar_collision_policy': 'omit every colliding bucket; table, history and image retain all points',
    'scalar_collision_buckets': collisions,
    'authoritative_values': 'pareto/gsm8k_thresholds table and per-run JSON history',
    'history_path': str(path), 'renderer_reference_sha256': REFERENCE_RENDERER_SHA256,
  }
  trends = {}
  for point in points:
    tag = threshold_tag(point['threshold'])
    trends[f'val/gsm8k_acc_thr{tag}'] = point['accuracy']
    trends[f'val/gsm8k_nfe_thr{tag}'] = point['nfe']
  trends['didi_tinygsm_pareto/scalar_collision_points'] = sum(len(buckets[bucket]) for bucket in collisions)
  if collisions:
    print('[didi-tinygsm-pareto] integer NFE*100 buckets collide; omitted from '
          f'approximate scalar view: {collisions}. All threshold points remain '
          'in the table/history/figure.', flush=True)
  rows = []
  for point in sorted(points, key=lambda point: (point['nfe'], point['threshold'])):
    bucket = int(round(point['nfe'] * NFE_X_SCALE))
    rows.append([point['threshold'], point['nfe'], point['accuracy'], point.get('correct'),
                 point.get('count'), bucket, len(buckets[bucket]) == 1,
                 json.dumps(point, sort_keys=True, allow_nan=False)])
  loggers = _loggers(trainer)
  for logger in loggers:
    logger.log_metrics(trends, step=step)
    for bucket, entries in sorted(buckets.items()):
      if len(entries) == 1:
        logger.log_metrics({f'pareto/gsm8k_step_{step}': entries[0]['accuracy']}, step=bucket)
    if hasattr(logger, 'log_table'):
      logger.log_table(key='pareto/gsm8k_thresholds',
        columns=['Threshold', 'Mean NFE per sequence', 'Accuracy', 'Correct', 'Examples',
                 'Integer scalar x (NFE * 100)', 'Included in approximate scalar view', 'Point (JSON)'],
        data=rows, step=step)
      logger.log_table(key='didi/tinygsm_pareto_protocol', columns=['Property', 'JSON value'],
        data=[[key, json.dumps(value)] for key, value in protocol.items()], step=step)
  if plot and any(hasattr(logger, 'log_image') for logger in loggers):
    try:
      image = render(current['curves'], reference=load_reference(reference))
      for logger in loggers:
        if hasattr(logger, 'log_image'):
          logger.log_image(key='pareto/frontier', image=image, step=step)
    except Exception as error:
      # Plotting is optional. Missing Matplotlib must not discard the completed
      # sweep or prevent training when exact tables/history already succeeded.
      print(f'[didi-tinygsm-pareto] optional figure skipped: {error}', flush=True)
  return {'run_uid': run_uid, 'training_step': step, 'curve': points,
          'history': current['curves'], 'protocol': protocol}
