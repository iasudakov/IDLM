"""Render the confidence sampler's accuracy-vs-NFE frontier as one figure.

`low_confidence_dynamic` has no fixed step budget -- its threshold is the budget
and the forward count is an outcome -- so a checkpoint's quality is a *curve*,
not a point. `trainer_base._log_gsm8k_pareto` scores every threshold at each
validation and hands the accumulated history here.

underdeep's native scalar graphs hold one series each, which is why the training
loop also logs a per-checkpoint series; this module exists for the view those
cannot give: every checkpoint's frontier overlaid, coloured by training step, so
whether the frontier is *moving* is visible at a glance.

Kept out of `trainer_base` so that module needs no matplotlib import, and pure
(history in, pixels out) so it can be tested without a GPU or a tracker.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')  # headless: rendered to a buffer, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_reference(path):
  """Read a reference frontier, or None if there is not one to draw.

  A missing file is not an error -- the reference is optional garnish, and a run
  configured to point at one that has not been measured yet should still plot.
  """
  if not path or not os.path.exists(path):
    return None
  try:
    with open(path) as f:
      data = json.load(f)
  except Exception as e:  # pragma: no cover - best-effort garnish
    print(f'[pareto] reference unreadable ({e})', flush=True)
    return None
  return data.get('points', data) if isinstance(data, dict) else data


def render(history, reference=None, reference_label='Released MDLM teacher',
           title='Confidence threshold frontiers'):
  """Draw every checkpoint's frontier on one axes and return it as pixels.

  Args:
    history: {training_step: [{'threshold', 'nfe', 'accuracy'}, ...]}. Accuracy
      is a fraction; the plot shows percent.
    reference: an optional frontier in the same point format, drawn as a black
      dashed line -- a fixed baseline the run is trying to beat.
    reference_label: legend entry for it.
    title: figure title.

  Returns:
    (H, W, 3) uint8 array, what `UnderdeepLogger.log_image` takes.
  """
  if not history:
    raise ValueError('nothing to plot: the history is empty')

  steps = sorted(int(s) for s in history)
  fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
  cmap = plt.get_cmap('viridis')
  # Colour by training step so the reader can tell which frontier is newest
  # without a legend entry per checkpoint -- there may eventually be dozens.
  lo, hi = steps[0], steps[-1]
  norm = matplotlib.colors.Normalize(vmin=lo / 1000, vmax=max(hi, lo + 1) / 1000)

  for step in steps:
    points = sorted(history[step] if step in history else history[str(step)],
                    key=lambda p: p['nfe'])
    ax.plot([p['nfe'] for p in points],
            [p['accuracy'] * 100 for p in points],
            marker='.', linewidth=1.2, color=cmap(norm(step / 1000)), zorder=2)

  if reference:
    points = sorted(reference, key=lambda p: p['nfe'])
    ax.plot([p['nfe'] for p in points],
            [p['accuracy'] * 100 for p in points],
            marker='D', markersize=5, linestyle='--', color='black',
            linewidth=1.6, label=reference_label, zorder=3)
    # Annotate thresholds once, on the reference, rather than on every curve --
    # the thresholds are the same for all of them and the plot gets unreadable.
    _annotate_thresholds(ax, points)
    ax.legend(loc='lower right', fontsize=9)
  else:
    _annotate_thresholds(
      ax, sorted(history[steps[-1]], key=lambda p: p['nfe']))

  ax.set_xlabel('Average NFE per sequence')
  ax.set_ylabel('GSM8K accuracy (%)')
  ax.set_title(title, fontweight='bold')
  ax.grid(alpha=0.25, zorder=0)
  ax.set_xlim(left=0)
  ax.set_ylim(bottom=0)

  bar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
  bar.set_label('Training checkpoint (k steps)')

  fig.tight_layout()
  fig.canvas.draw()
  image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  plt.close(fig)
  return image


def _annotate_thresholds(ax, points):
  for point in points:
    ax.annotate(f"thr {point['threshold']:g}",
                (point['nfe'], point['accuracy'] * 100),
                textcoords='offset points', xytext=(6, -3),
                fontsize=8, color='0.35')
