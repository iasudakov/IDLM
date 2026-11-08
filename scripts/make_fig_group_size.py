"""Temperature Pareto by MMD group size: one panel per sampling-step budget.

Three runs differing only in `adversarial_distill.mmd_group_size` (2, 4, 8),
each shown at its OWN best checkpoint. Selection is the paper's rule: among the
checkpoints whose temperature curve reaches the target entropy, take the one
minimising the geometric mean of generative perplexity over the three budgets,
read at that entropy by linear interpolation between bracketing temperatures.
The geometric mean because perplexity spans ~17 to ~50 across budgets and an
arithmetic mean would be decided almost entirely by the 8-step column.

Reading at matched entropy is the point: generative perplexity falls as a model
sharpens, so a raw best-PPL pick rewards collapse. Entropy is the axis that
exposes it, which is why the curves are drawn against it rather than against
temperature.

Usage:
  python scripts/make_fig_group_size.py                 # fetch, select, draw
  python scripts/make_fig_group_size.py --curves <json> # redraw, no network

The fetch writes its selected curves next to the figure, so the drawing can be
changed later without paying for the tracker round-trip again.
"""
import argparse
import json
import os

RUNS = [('$G$=2', '235a9a4e-62e7-4a49-85e9-443ebaf86ca6',
         dict(c='tab:red', m='s')),
        ('$G$=4', '5d1607a1-9288-4e1b-b9ca-d60814b8a562',
         dict(c='tab:orange', m='^')),
        ('$G$=8', 'daf2f504-a62c-4b08-8f70-dd0c3d9f23af',
         dict(c='tab:blue', m='o'))]
BUDGETS = (8, 16, 32)
EXPERIMENT = 'generative-models/owt'
# The teacher's regime. `_publish_owt_pareto_temperature` logs the sweep as
# `val/{gen_ppl,entropy}_s{budget}_t{temperature}`.
SELECT_H = 5.2
MARK_H = 5.43
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ppl_at(curve, h):
    """gen_ppl where the curve crosses entropy `h`, or None if it never does.

    `curve` is [(temperature, gen_ppl, entropy), ...]. Interpolates linearly
    between the two bracketing temperatures, which is what reading a Pareto
    curve at a fixed entropy means when the sweep is discrete.
    """
    pts = sorted((e, p) for _, p, e in curve)
    for (e0, p0), (e1, p1) in zip(pts, pts[1:]):
        if e0 <= h <= e1:
            return p0 if e1 == e0 else p0 + (p1 - p0) * (h - e0) / (e1 - e0)
    return None


def geo_mean_at(curves, h):
    vals = [ppl_at(curves[str(b)], h) for b in BUDGETS]
    if any(v is None for v in vals):
        return None
    return (vals[0] * vals[1] * vals[2]) ** (1 / 3)


def fetch():
    """Pull every checkpoint's sweep and keep each run's best."""
    import underdeep as U

    def series(metric):
        d = metric.get_values(sampling_level=10000)
        return dict(zip(d['steps'], d['y']['data']))

    client = U.Client(experiment=EXPERIMENT)
    selected = {}
    for label, uid, _ in RUNS:
        by = {m.name: m for m in client.runs.get(uid).list_metrics()}
        temps = sorted({n.split('_t')[1] for n in by
                        if n.startswith('val/gen_ppl_s8_t')}, key=float)
        S = {n: series(m) for n, m in by.items()
             if n.startswith('val/') and '_t' in n}
        wanted = [f'val/{k}_s{b}_t{t}' for b in BUDGETS for t in temps
                  for k in ('gen_ppl', 'entropy')]
        steps = sorted(set.intersection(*(set(S[k]) for k in wanted if k in S)))
        best, best_gm = None, None
        for step in steps:
            curves = {str(b): [(float(t), S[f'val/gen_ppl_s{b}_t{t}'][step],
                                S[f'val/entropy_s{b}_t{t}'][step])
                               for t in temps] for b in BUDGETS}
            gm = geo_mean_at(curves, SELECT_H)
            if gm is not None and (best_gm is None or gm < best_gm):
                best, best_gm = (step, curves), gm
        if best is None:
            raise SystemExit(f'{label}: no checkpoint reaches H={SELECT_H}')
        step, curves = best
        selected[label] = {
            'uid': uid, 'step': step, 'geo_mean': round(best_gm, 3),
            'curves': {b: [(t, round(p, 2), round(e, 3)) for t, p, e in c]
                       for b, c in curves.items()}}
        print(f'{label}: step {step}, geo-mean PPL {best_gm:.2f} '
              f'at H={SELECT_H} (of {len(steps)} checkpoints)')
    return selected


def draw(selected, out_stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.6,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.titlesize': 8.5, 'font.family': 'serif'})
    xlo, xhi = 5.1, 5.55
    ticks = [5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), dpi=200)
    for ax, budget in zip(axes, BUDGETS):
        visible = []
        for label, _, style in RUNS:
            arm = selected[label]
            pts = arm['curves'][str(budget)]
            xs = [e for _, _, e in pts]
            ys = [p for _, p, _ in pts]
            ax.plot(xs, ys, marker=style['m'], ms=3.2, ls='-', lw=1.2,
                    color=style['c'], alpha=0.95,
                    label=f"{label} (step {arm['step']})")
            visible += [y for x, y in zip(xs, ys) if xlo <= x <= xhi]
        ax.axvline(MARK_H, color='0.45', ls=':', lw=0.9, zorder=1)
        ax.set_yscale('log')
        ax.set_xlim(xlo, xhi)
        ax.set_xticks([5.1, 5.2, 5.3, 5.4, 5.5])
        lo, hi = min(visible), max(visible)
        ax.set_ylim(lo / 1.15, hi * 1.15)
        keep = [t for t in ticks if lo * 1.02 <= t <= hi / 1.02]
        while len(keep) > 4:
            keep = keep[::2]
        ax.set_yticks(keep)
        ax.set_yticklabels([str(t) for t in keep])
        ax.set_title(f'{budget} sampling steps')
        ax.set_xlabel('sample entropy $\\rightarrow$', labelpad=1.5)
        ax.set_ylabel('gen. perplexity', labelpad=1.5)
        ax.grid(alpha=0.25, lw=0.5, zorder=0)
        ax.minorticks_off()
    axes[0].annotate(f'$H{{=}}{MARK_H}$', (MARK_H, 0.97),
                     xycoords=('data', 'axes fraction'),
                     textcoords='offset points', xytext=(-3, -7),
                     fontsize=6, color='0.45', ha='right')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               columnspacing=1.6, handlelength=1.6, handletextpad=0.45,
               bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        path = f'{out_stem}.{ext}'
        fig.savefig(path, bbox_inches='tight',
                    **({'dpi': 190} if ext == 'png' else {}))
        print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curves', default=None,
                    help='redraw from a saved JSON instead of the tracker')
    ap.add_argument('--out', default=os.path.join(HERE, 'assets',
                                                  'fig_group_size'))
    args = ap.parse_args()

    if args.curves:
        selected = json.load(open(args.curves))
    else:
        selected = fetch()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        path = f'{args.out}_curves.json'
        with open(path, 'w') as f:
            json.dump(selected, f, indent=1)
        print(f'wrote {path}')
    draw(selected, args.out)


if __name__ == '__main__':
    main()
