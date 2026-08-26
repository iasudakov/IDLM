"""Render the sampling-steps quality/diversity frontier as one figure.

A diffusion LM's unconditional quality is not a point but a *curve*: fewer
denoising steps buy speed and cost fluency, and the price is not linear. So a
checkpoint is summarised by sampling it at several step budgets and plotting
generative perplexity (how fluent the samples look to a strong AR) against
sample entropy (how varied they are).

Both axes are needed because either alone is trivially gamed. Perplexity falls
if the model collapses onto generic text -- and entropy falls with it, which is
the tell. A frontier that moves *down and to the right* over training is real
improvement; one that moves down and to the *left* is collapse wearing the same
costume.

`trainer_base._log_owt_pareto` measures one curve per validation and hands the
accumulated history here.

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
    print(f'[owt-pareto] reference unreadable ({e})', flush=True)
    return None
  return data.get('points', data) if isinstance(data, dict) else data


def render(history, reference=None, reference_label='Teacher (mdlm-owt)',
           title='Sampling-steps frontier: fluency vs diversity'):
  """Draw every checkpoint's frontier on one axes and return it as pixels.

  Args:
    history: {training_step: [{'steps', 'entropy', 'gen_ppl'}, ...]}.
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
    points = history[step] if step in history else history[str(step)]
    # Ordered by sampling budget, so the line traces the frontier rather than
    # zig-zagging in whatever order the sweep happened to finish.
    points = sorted(points, key=lambda p: p['steps'])
    x = [p['entropy'] for p in points]
    y = [p['gen_ppl'] for p in points]
    # Error bars where the sweep averaged several seeds. Drawn rather than
    # omitted because they are what says whether two frontiers actually
    # differ -- without them a gap of one noise width reads as progress.
    xerr = [p.get('entropy_std', 0.0) for p in points]
    yerr = [p.get('gen_ppl_std', 0.0) for p in points]
    colour = cmap(norm(step / 1000))
    if any(xerr) or any(yerr):
      ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker='.', linewidth=1.2,
                  color=colour, ecolor=colour, elinewidth=0.9, capsize=2,
                  alpha=0.95, zorder=2)
    else:
      ax.plot(x, y, marker='.', linewidth=1.2, color=colour, zorder=2)

  if reference:
    points = sorted(reference, key=lambda p: p['steps'])
    ax.plot([p['entropy'] for p in points],
            [p['gen_ppl'] for p in points],
            marker='D', markersize=5, linestyle='--', color='black',
            linewidth=1.6, label=reference_label, zorder=3)
    _annotate_steps(ax, points)
    ax.legend(loc='upper right', fontsize=9)
  else:
    _annotate_steps(
      ax, sorted(history[steps[-1]], key=lambda p: p['steps']))

  ax.set_xlabel('Sample entropy  (higher = more varied)')
  ax.set_ylabel('Generative perplexity  (lower = more fluent)')
  ax.set_title(title, fontweight='bold')
  # Perplexity at 2 steps can be orders of magnitude above the 64-step value,
  # which on a linear axis flattens the whole interesting end of the curve.
  ax.set_yscale('log')
  ax.grid(alpha=0.25, which='both', zorder=0)

  bar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
  bar.set_label('Training checkpoint (k steps)')

  fig.tight_layout()
  fig.canvas.draw()
  image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  plt.close(fig)
  return image


def _annotate_steps(ax, points):
  """Label each marker with the sampling budget that produced it."""
  for point in points:
    ax.annotate(f"{point['steps']:g}",
                (point['entropy'], point['gen_ppl']),
                textcoords='offset points', xytext=(6, -3),
                fontsize=8, color='0.35')
