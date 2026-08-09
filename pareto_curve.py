"""Quality/diversity Pareto curve over training.

`val/gen_ppl` and `val/sample_entropy` are logged as two separate scalar-vs-step
curves, but the object worth looking at is the *trade-off* between them: sample
entropy is the diversity of the generated text and generative perplexity is its
quality under the judge model, and a run buys one with the other as it trains.
Plotted against each other, one point per validation, the run traces a curve
through the (diversity, quality) plane, and the non-dominated subset of those
points is the best trade-off the run has reached so far.

This module keeps that curve on disk as a JSONL (one point per validation,
appended by `trainer_base.on_validation_epoch_end`) and renders it to an image
that goes into the experiment tracker alongside the scalars. Nothing here costs
a forward pass: both numbers are already computed by the time the point is
appended.

The same file is the input to the offline comparison plot, which is where the
curve actually earns its keep -- a single run's trajectory says how it moved,
several runs overlaid say which hyper-parameter buys a better trade-off:

    python -m pareto_curve run_a.jsonl run_b.jsonl -o pareto.png \
        --labels 'entropy 0.0' 'entropy 0.01'

Rendering is optional: with no matplotlib installed `render` returns None and
the caller logs nothing, exactly as `underdeep_logger` degrades when the
tracker is unavailable.
"""

import json
import os
import typing


# Blue ordinal ramp, light -> dark. Discrete ordered marks, so the light end
# starts at the step that still clears 2:1 against a light surface rather than
# at the near-white end of the sequential ramp.
_RAMP = ('#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
         '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
# Validated 4-run categorical order (a subset of `_RAMP`, monotone in
# lightness with visible gaps).
_RAMP4 = ('#86b6ef', '#3987e5', '#256abf', '#104281')
_SURFACE, _INK, _INK2, _INK3 = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984'


def append(path, step, sample_entropy, gen_ppl):
  """Append one validation point to the JSONL history at `path`.

  Caller is responsible for rank-zero gating. Creates the parent directory if
  needed. Returns True when the point was written.
  """
  if not path:
    return False
  point = {'step': int(step),
           'sample_entropy': float(sample_entropy),
           'gen_ppl': float(gen_ppl)}
  parent = os.path.dirname(os.fspath(path))
  if parent:
    os.makedirs(parent, exist_ok=True)
  with open(path, 'a') as f:
    f.write(json.dumps(point) + '\n')
  return True


def load(path):
  """Read a JSONL history back as a step-sorted list of points.

  Later lines win on a repeated `step`, so resuming a run and re-validating a
  step it had already reached replaces that point instead of duplicating it.
  Unparseable lines and non-finite values are dropped rather than raising: this
  file is a log, and half a curve is more useful than a crash in a training
  loop.
  """
  if not path or not os.path.exists(path):
    return []
  by_step = {}
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        point = json.loads(line)
        step = int(point['step'])
        entropy = float(point['sample_entropy'])
        gen_ppl = float(point['gen_ppl'])
      except (ValueError, TypeError, KeyError):
        continue
      if not (entropy == entropy and gen_ppl == gen_ppl):  # NaN
        continue
      if gen_ppl <= 0:                                     # log-scale axis
        continue
      by_step[step] = {'step': step, 'sample_entropy': entropy,
                       'gen_ppl': gen_ppl}
  return [by_step[k] for k in sorted(by_step)]


def frontier(points):
  """The non-dominated points: maximal `sample_entropy`, minimal `gen_ppl`.

  A point is dominated when some other point is at least as diverse and at
  least as good, and strictly better on one of the two. Returned sorted by
  entropy, so plotting it as a line draws the frontier left to right.
  """
  keep = []
  for p in points:
    if not any(q['sample_entropy'] >= p['sample_entropy']
               and q['gen_ppl'] <= p['gen_ppl']
               and (q['sample_entropy'] > p['sample_entropy']
                    or q['gen_ppl'] < p['gen_ppl'])
               for q in points):
      keep.append(p)
  return sorted(keep, key=lambda p: (p['sample_entropy'], -p['gen_ppl']))


def _colors(n):
  """`n` colours from the ordinal ramp, light (first run) to dark (last)."""
  if n <= len(_RAMP4):
    return list(_RAMP4[:n])
  step = (len(_RAMP) - 1) / float(n - 1)
  return [_RAMP[int(round(i * step))] for i in range(n)]


def render(runs, title=None, dpi=150, figsize=(7.6, 5.2)):
  """Render the Pareto plot as an RGB `uint8` HWC array, or None.

  Args:
    runs: list of `(label, points)` pairs, `points` as returned by `load`. A
      single pair is drawn as one trajectory coloured by training step (early
      light, late dark) with a step colourbar; several pairs are drawn one
      colour per run with a legend. The dashed frontier is always taken over
      the union of every run's points.
    title: figure title. Defaults to a description of what is plotted.
    dpi, figsize: passed to matplotlib.

  Returns None when matplotlib is unavailable or nothing has enough points to
  plot, so a caller can skip logging without a special case.
  """
  runs = [(label, pts) for label, pts in runs if pts]
  if not runs:
    return None
  try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
  except Exception as e:  # pragma: no cover - rendering is best-effort
    print('[pareto] render skipped (matplotlib unavailable: {})'.format(e),
          flush=True)
    return None

  fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=_SURFACE)
  ax.set_facecolor(_SURFACE)

  every = [p for _, pts in runs for p in pts]
  front = frontier(every)
  if len(front) > 1:
    ax.plot([p['sample_entropy'] for p in front],
            [p['gen_ppl'] for p in front],
            lw=1.3, ls=(0, (5, 3)), color=_INK3, zorder=1)

  if len(runs) == 1:
    label, pts = runs[0]
    x = [p['sample_entropy'] for p in pts]
    y = [p['gen_ppl'] for p in pts]
    steps = [p['step'] for p in pts]
    ax.plot(x, y, '-', lw=1.4, color='#c9d6e6', zorder=2,
            solid_capstyle='round')
    cmap = LinearSegmentedColormap.from_list('idlm_steps', _RAMP)
    sc = ax.scatter(x, y, c=steps, cmap=cmap, s=48, zorder=4,
                    edgecolors=_SURFACE, linewidths=1.4)
    if len(set(steps)) > 1:
      bar = fig.colorbar(sc, ax=ax, pad=0.015)
      bar.set_label('training step', color=_INK2, fontsize=9.5)
      bar.ax.tick_params(colors=_INK2, labelsize=8.5)
      bar.outline.set_visible(False)
    # "You are here": the newest checkpoint, the one a reader is looking for.
    ax.annotate('step {:,}'.format(steps[-1]), (x[-1], y[-1]),
                xytext=(11, -4), textcoords='offset points',
                color=_INK, fontsize=9.5, fontweight='bold')
    subtitle = '{} · {} validation{}'.format(
      label, len(pts), '' if len(pts) == 1 else 's')
    legend = None
  else:
    palette = _colors(len(runs))
    for (label, pts), color in zip(runs, palette):
      x = [p['sample_entropy'] for p in pts]
      y = [p['gen_ppl'] for p in pts]
      ax.plot(x, y, '-', lw=2.0, color=color, zorder=3,
              solid_capstyle='round')
      ax.plot(x, y, 'o', ms=6.0, color=color, mec=_SURFACE, mew=1.4, zorder=4)
      ax.plot(x[-1], y[-1], 'o', ms=10.0, color=color, mec=_SURFACE, mew=1.8,
              zorder=5)
    legend = [Line2D([], [], color=c, lw=2.4, marker='o', ms=6, mec=_SURFACE,
                     mew=1.2, label=label)
              for (label, _), c in zip(runs, palette)]
    subtitle = '{} runs · one point per validation'.format(len(runs))

  handles = list(legend or [])
  if len(front) > 1:
    handles.append(Line2D([], [], color=_INK3, lw=1.3, ls=(0, (5, 3)),
                          label='Pareto frontier'))
  if handles:
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=9,
              labelcolor=_INK2, handlelength=2.2, labelspacing=0.5)

  # Log perplexity axis, but ticked 5/10/20/50/... and labelled as plain
  # numbers -- `10^1` is a worse way to say `10` to someone reading a ppl.
  ax.set_yscale('log')
  ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0),
                                        numticks=16))
  ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: '%g' % v))
  ax.yaxis.set_minor_formatter(NullFormatter())
  ax.set_xlabel('sample entropy  (diversity  →)', color=_INK2)
  ax.set_ylabel('gen_ppl  (←  quality)', color=_INK2)
  ax.set_title(title or 'Quality–diversity trade-off over training',
               fontsize=13, fontweight='bold', loc='left', pad=24, color=_INK)
  ax.text(0, 1.018, subtitle, transform=ax.transAxes, color=_INK2,
          fontsize=9.5, va='bottom')
  ax.grid(True, which='major', color='#eceae5', lw=0.8)
  ax.set_axisbelow(True)
  ax.tick_params(colors=_INK2, labelsize=9)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  for side in ('left', 'bottom'):
    ax.spines[side].set_color('#d8d7d2')

  fig.tight_layout()
  fig.canvas.draw()
  image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  plt.close(fig)
  return image


def render_history(path, label=None, title=None):
  """`render` straight off a JSONL history. Returns None when it cannot."""
  points = load(path)
  if len(points) < 1:
    return None
  return render([(label or 'this run', points)], title=title)


def _main(argv: typing.Optional[typing.List[str]] = None):
  import argparse
  parser = argparse.ArgumentParser(
    description='Plot quality/diversity Pareto curves from JSONL histories '
                'written during training.')
  parser.add_argument('paths', nargs='+',
                      help='one pareto_curve.jsonl per run, light -> dark in '
                           'the order given')
  parser.add_argument('-o', '--out', default='pareto.png',
                      help='output image path (default: pareto.png)')
  parser.add_argument('--labels', nargs='*', default=None,
                      help='legend label per run (default: file stem)')
  parser.add_argument('--title', default=None)
  args = parser.parse_args(argv)

  labels = args.labels or [
    os.path.basename(os.path.dirname(os.path.abspath(p))) or
    os.path.splitext(os.path.basename(p))[0] for p in args.paths]
  if len(labels) != len(args.paths):
    parser.error('got {} --labels for {} runs'.format(
      len(labels), len(args.paths)))

  runs = [(label, load(path)) for label, path in zip(labels, args.paths)]
  for label, points in runs:
    print('{:>28}: {} point(s)'.format(label, len(points)))
  image = render(runs, title=args.title)
  if image is None:
    raise SystemExit('nothing to plot')

  from PIL import Image
  Image.fromarray(image).save(args.out)
  print('wrote {}'.format(args.out))


if __name__ == '__main__':
  _main()
