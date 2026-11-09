"""Publish ancestral sampling accuracy across configured budgets and seeds."""

import warnings


def point_metrics(point):
    """Return accuracy fractions and actual NFE scalars for one step budget."""
    budget = int(point['steps'])
    metrics = {}
    for metric, field in (('acc', 'accuracy'), ('nfe', 'nfe')):
        prefix = f'val/gsm8k_{metric}_ancestral'
        metrics[f'{prefix}_steps{budget}'] = float(point[field + '_mean'])
        deviation = point.get(field + '_std')
        if deviation is not None:
            metrics[f'{prefix}_std_steps{budget}'] = float(deviation)
        for result in point.get('seeds', []):
            metrics[f'{prefix}_steps{budget}_seed{int(result["seed"])}'] = float(result[field])
    return metrics


def _destinations(model):
    trainer = getattr(model, 'trainer', None)
    if trainer is None or getattr(trainer, 'global_rank', 0) != 0:
        return ()
    return tuple(logger for logger in (getattr(trainer, 'loggers', None) or ())
                 if logger is not None)


def log_point(model, point, protocol):
    """Publish immediately at the exact optimizer step and at budget S on curves.

    Protocol metadata is retained in the caller's sweep result. The curve's x
    coordinate is the configured sampler budget, independently of actual NFE.
    """
    destinations = _destinations(model)
    if not destinations:
        return
    step = int(model.global_step)
    metrics = point_metrics(point)
    curve = {f'ancestral/gsm8k_step_{step}': float(point['accuracy_mean'])}
    if point.get('accuracy_std') is not None:
        curve[f'ancestral/gsm8k_std_step_{step}'] = float(point['accuracy_std'])
    for logger in destinations:
        logger.log_metrics(metrics, step=step)
        logger.log_metrics(curve, step=int(point['steps']))


def render_ancestral(points, step=None):
    """Render mean accuracy with sample SD against configured sampler steps.

    Error bars use the supplied across-seed sample standard deviation (ddof=1),
    and are omitted where it is undefined for one seed. Actual measured NFE is
    reported separately; it does not replace the configured-budget x axis.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ordered = sorted(points, key=lambda point: point['steps'])
    budgets = [int(point['steps']) for point in ordered]
    means = [100 * float(point['accuracy_mean']) for point in ordered]
    with_nfe = bool(ordered) and all(point.get('nfe_mean') is not None for point in ordered)
    figure = Figure(figsize=(8.5, 6.2 if with_nfe else 5), dpi=120,
                    constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 1, height_ratios=(4, 1)) if with_nfe else None
    axes = figure.add_subplot(grid[0] if with_nfe else 111)
    axes.plot(budgets, means, color='#1763a6', marker='o', linewidth=2,
              label='Mean accuracy')
    defined = [point for point in ordered if point.get('accuracy_std') is not None]
    if defined:
        axes.errorbar([int(point['steps']) for point in defined],
                      [100 * float(point['accuracy_mean']) for point in defined],
                      yerr=[100 * float(point['accuracy_std']) for point in defined],
                      fmt='none', ecolor='#1763a6', capsize=4,
                      label='Sample SD (ddof=1)')
    lower = [100 * (float(point['accuracy_mean']) - float(point.get('accuracy_std') or 0))
             for point in ordered]
    upper = [100 * (float(point['accuracy_mean']) + float(point.get('accuracy_std') or 0))
             for point in ordered]
    title = 'TinyGSM ancestral accuracy'
    if step is not None:
        title += f' (training step {int(step)})'
    axes.set_xscale('log', base=2)
    axes.set_xticks(budgets, labels=[str(budget) for budget in budgets])
    maximum = max(upper, default=0.)
    margin = max(.1, maximum * .1)
    minimum = min(lower, default=0.)
    axes.set(xlabel='Configured sampler steps', ylabel='GSM8K execution accuracy (%)',
             title=title, ylim=(minimum - margin if minimum < 0 else 0.,
                                max(1., maximum + margin)))
    axes.grid(alpha=.2)
    if ordered:
        axes.legend(loc='best')
    if with_nfe:
        table_axes = figure.add_subplot(grid[1])
        table_axes.set_axis_off()
        table_axes.table(
            cellText=[[f'{float(point["nfe_mean"]):.2f}' for point in ordered],
                      [f'{float(point["nfe_std"]):.2f}'
                       if point.get('nfe_std') is not None else 'n/a' for point in ordered]],
            colLabels=[str(budget) for budget in budgets],
            rowLabels=['Mean actual NFE', 'Sample SD (ddof=1)'],
            cellLoc='center', loc='center')
    seed_counts = {point['num_seeds'] for point in ordered if 'num_seeds' in point}
    population = 'seeds'
    if len(seed_counts) == 1:
        count = int(next(iter(seed_counts)))
        population = f'{count} seed' + ('s' if count != 1 else '')
    figure.suptitle(f'Error bars: sample SD across {population} (ddof=1); '
                    'omitted for a single seed.', fontsize=9)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def publish_result(model, result):
    """Publish discoverable per-seed results and an accuracy/budget image."""
    destinations = _destinations(model)
    if not destinations:
        return
    step = int(model.global_step)
    points = sorted(result['points'], key=lambda point: point['steps'])
    rows = [[int(point['steps']), int(seed['seed']), int(seed['correct']),
             int(seed['count']), float(seed['accuracy']), float(seed['nfe'])]
            for point in points for seed in point.get('seeds', [])]
    image = None
    render_failed = False
    for logger in destinations:
        if rows and callable(getattr(logger, 'log_table', None)):
            try:
                logger.log_table('ancestral/results',
                                 ['steps', 'seed', 'correct', 'count', 'accuracy', 'NFE'],
                                 rows, step=step)
            except Exception as error:
                warnings.warn(f'Could not publish ancestral GSM8K results table: {error}')
        if callable(getattr(logger, 'log_image', None)):
            if image is None and not render_failed:
                try:
                    image = render_ancestral(points, step=step)
                except Exception as error:
                    render_failed = True
                    warnings.warn(f'Could not render ancestral GSM8K accuracy plot: {error}')
            if image is not None:
                try:
                    logger.log_image('ancestral/accuracy_vs_steps', image, step=step)
                except Exception as error:
                    warnings.warn(f'Could not publish ancestral GSM8K accuracy image: {error}')
