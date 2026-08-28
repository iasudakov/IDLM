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

import collections
import json
import os

import matplotlib
matplotlib.use('Agg')  # headless: rendered to a buffer, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_reference(path, reference_temperature=1.0):
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
  if isinstance(data, dict) and 'curves' in data:
    return _curve_at_temperature(data['curves'], reference_temperature)
  return data.get('points', data) if isinstance(data, dict) else data


def _curve_at_temperature(curves, temperature):
  """Pick one temperature's frontier out of a `pareto_temperature.json`.

  A temperature sweep has no single frontier, so drawing one underneath another
  run means choosing which sampler to compare against. T=1 -- the untouched
  sampler -- is the honest default: it is what the checkpoint would be reported
  at, and comparing a student's whole temperature fan against a teacher already
  sharpened to T=0.7 would flatter the teacher for free.
  """
  keyed = {}
  for key, points in (curves or {}).items():
    try:
      keyed[float(key)] = points
    except (TypeError, ValueError):
      continue
  if not keyed:
    return None
  return keyed[min(keyed, key=lambda t: abs(t - temperature))]


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


def render_temperature(curves, reference=None,
                       reference_label='Teacher (mdlm-owt)',
                       title='Fluency vs diversity: temperature at each step budget',
                       subtitle=None):
  """Draw one frontier per step budget, with temperature walking along it.

  Grouped by BUDGET, not by temperature, because the budget is the thing you
  are actually choosing between: given 8 steps, temperature is the dial you
  still have, and the question is what the whole reachable frontier looks like
  at that price. A curve per temperature would answer the mirror question and
  put the interesting comparison -- 8 steps against 32 -- between the lines
  rather than along them.

  Each marker is labelled with the temperature that produced it, so a vertical
  read at fixed entropy gives what the extra compute buys, and a horizontal
  read at fixed perplexity gives how much sharpening substitutes for it.

  Axes are deliberately the same as `render` -- entropy right, perplexity down,
  log y -- so a temperature figure and a training figure can be laid side by
  side without re-reading the axes.

  Args:
    curves: {temperature: [{'steps', 'entropy', 'gen_ppl', ...}, ...]}, i.e.
      exactly what `pareto_temperature.sweep` returns. Regrouped internally.
    reference: optional frontier in the same point format, drawn dashed black.
    reference_label: legend entry for it.
    title: figure title.
    subtitle: small line under the title, e.g. which checkpoint this is.

  Returns:
    (H, W, 3) uint8 array, what `UnderdeepLogger.log_image` takes.
  """
  if not curves:
    raise ValueError('nothing to plot: no temperature curves')

  # A sweep straight out of `sweep()` is float-keyed; one read back from its
  # own JSON is string-keyed, and `'1'` is not `str(1.0)`.
  by_budget = collections.defaultdict(list)
  for key, points in curves.items():
    for point in points:
      by_budget[int(point['steps'])].append(dict(point, temperature=float(key)))
  budgets = sorted(by_budget)

  fig, ax = plt.subplots(figsize=(8.5, 6), dpi=110)
  # Sequential, dark-to-light with the budget: more compute reads as further
  # along one axis, which is what a budget is. (`render` uses the same family
  # for the training-step colourbar.)
  cmap = plt.get_cmap('viridis')
  positions = ({b: i / max(len(budgets) - 1, 1)
                for i, b in enumerate(budgets)})

  for budget in budgets:
    points = sorted(by_budget[budget], key=lambda p: p['temperature'])
    x = [p['entropy'] for p in points]
    y = [p['gen_ppl'] for p in points]
    xerr = [p.get('entropy_std', 0.0) or 0.0 for p in points]
    yerr = [p.get('gen_ppl_std', 0.0) or 0.0 for p in points]
    colour = cmap(0.15 + 0.7 * positions[budget])
    style = dict(marker='o', markersize=4.5, linewidth=1.8, color=colour,
                 label=f'{budget} steps', zorder=3)
    if any(xerr) or any(yerr):
      ax.errorbar(x, y, xerr=xerr, yerr=yerr, ecolor=colour, elinewidth=0.9,
                  capsize=2, alpha=0.95, **style)
    else:
      ax.plot(x, y, **style)
    _annotate_temperatures(ax, points)

  if reference:
    points = sorted(reference, key=lambda p: p['steps'])
    ax.plot([p['entropy'] for p in points],
            [p['gen_ppl'] for p in points],
            marker='D', markersize=5, linestyle='--', color='black',
            linewidth=1.6, label=reference_label, zorder=4)
    _annotate_steps(ax, points)

  ax.set_xlabel('Sample entropy  (higher = more varied)')
  ax.set_ylabel('Generative perplexity  (lower = more fluent)')
  # `pad` leaves the subtitle a line of its own; without it the two overlap.
  ax.set_title(title, fontweight='bold', pad=20 if subtitle else 6)
  if subtitle:
    # Under the title rather than in the legend: it names the checkpoint, which
    # is context for the whole figure, not one more series to compare.
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha='center',
            va='bottom', fontsize=8.5, color='0.35')
  ax.set_yscale('log')
  ax.grid(alpha=0.25, which='both', zorder=0)
  ax.legend(loc='best', fontsize=9, title='Denoising budget',
            title_fontsize=9)

  fig.tight_layout()
  fig.canvas.draw()
  image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  plt.close(fig)
  return image


def _annotate_temperatures(ax, points):
  """Label each marker with the sampling temperature that produced it."""
  for point in points:
    ax.annotate(f"T={point['temperature']:g}",
                (point['entropy'], point['gen_ppl']),
                textcoords='offset points', xytext=(6, -3),
                fontsize=7.5, color='0.35')
