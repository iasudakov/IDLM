"""Sweep one finished checkpoint over sampling temperatures and step budgets.

`trainer_base._log_owt_pareto` answers "is this run's frontier moving the right
way" -- it holds the sampler fixed and varies the checkpoint. This answers the
question that only makes sense once training is over: with the model fixed,
what does trading the sampler's sharpness actually buy at each step budget.

The two axes are the same as the training sweep's, deliberately. Generative
perplexity alone is trivially gamed -- a sampler that collapses onto generic
text scores *better* on it -- and entropy is what exposes that. Temperature is
the cleanest way to walk that trade-off: below 1 sharpens `p(x0 | x_t)` and
moves the curve down and to the left, above 1 flattens it and moves the curve up
and to the right. So a student that looks like it beat its teacher may just be
sampling sharper, and the tell is whether its curve sits BELOW the teacher's
temperature curve or merely somewhere along it.

Driven by `main.py`'s `mode=pareto_eval`. Kept out of `trainer_base` because it
needs no trainer: one process, one GPU, no dataloader.
"""

import collections
import json
import os

import torch

import owt_pareto_plot

# Entropy is a float around 5 and a tracker indexes points by an integer step,
# so the frontier series carries it at fixed scale: an x of 5062 means entropy
# 5.062. Shared with `trainer_base` so both frontiers read the same way.
_ENTROPY_X_SCALE = 1000
# Prime, so a temperature index and a repeat index cannot collide onto a shared
# seed the way `base + temp + repeat` would.
_SEED_STRIDE = 9973
# Same trick for temperature: a tracker step is an integer, so T=0.7 is logged
# at x=70.
_TEMP_X_SCALE = 100
# Offsets each rank's seed. Prime and far larger than the (repeat, budget) grid
# above, so eight GPUs cannot land on a seed the single-GPU sweep already used.
_RANK_STRIDE = 104729


def _mean(values):
  return float(sum(values) / len(values)) if values else 0.0


def _std(values):
  """Sample standard deviation; 0 for fewer than two values."""
  if len(values) < 2:
    return 0.0
  mean = _mean(values)
  return float((sum((v - mean) ** 2 for v in values)
                / (len(values) - 1)) ** 0.5)


def _distributed():
  """`(rank, world_size)`; `(0, 1)` unless torchrun set up a process group."""
  if (torch.distributed.is_available()
      and torch.distributed.is_initialized()):
    return torch.distributed.get_rank(), torch.distributed.get_world_size()
  return 0, 1


def _shard(total, rank, world_size):
  """This rank's share of `total` items; shares differ by at most one."""
  base, extra = divmod(total, world_size)
  return base + (1 if rank < extra else 0)


@torch.no_grad()
def sweep(model, temperatures, budgets, seeds=1, num_batches=1,
          batch_size=8, max_length=1024, keep_samples=2):
  """Measure (entropy, gen_ppl) for every (temperature, step budget) pair.

  Every temperature is driven through the SAME seeds, so the curves are a
  paired comparison: a gap between two of them is the sampler changing, not one
  of them drawing luckier noise. This matters more here than in training,
  because the differences between adjacent temperatures are small enough to be
  buried by unpaired sampling noise.

  Args:
    model: a diffusion model already in eval mode and on its device.
    temperatures: sampler temperatures to sweep.
    budgets: denoising-step counts; each is one point on each curve.
    seeds: seeded repeats per point, averaged, spread reported as the std.
    num_batches: sample batches per repeat. `num_batches * batch_size` is the
      total per point and is split across ranks under torchrun, not multiplied
      by them, so the protocol is the same on 1 GPU and on 8.
    batch_size: sequences per generate call -- the memory knob, and the cap on
      how much `record_generative_perplexity` is handed at once.
    max_length: sequence length handed to the reference LM.
    keep_samples: text samples per temperature kept for logging, 0 to keep none.

  Returns:
    (curves, samples): `{temperature: [point, ...]}` where a point is
    `{steps, entropy, gen_ppl, entropy_std, gen_ppl_std, seeds}`, and
    `{temperature: [text, ...]}`.
  """
  temperatures = [float(t) for t in temperatures]
  budgets = sorted({int(b) for b in budgets})
  seeds = max(1, int(seeds))
  rank, world_size = _distributed()
  # `num_batches * batch_size` is the protocol's sample count per point, and it
  # stays that whatever the GPU count: the ranks SPLIT those samples, they do
  # not each draw a full set. Pooling is exact -- both metrics hold a sum and a
  # weight, torchmetrics all-reduces both with `sum`, and only then divides --
  # so N GPUs compute the number one GPU would, over a different random draw.
  total_samples = num_batches * batch_size
  assert total_samples >= world_size, (
    f'{total_samples} samples cannot be split across {world_size} ranks; '
    'raise eval.owt_pareto_num_batches or launch on fewer GPUs')
  local_samples = _shard(total_samples, rank, world_size)
  if rank == 0:
    print(f'[pareto-temp] {total_samples} samples per point over {world_size} '
          f'rank(s); rank 0 draws {local_samples} in chunks of <= {batch_size}',
          flush=True)
  original_temperature = model.sampling_temperature

  curves, samples = {}, {}
  try:
    for temperature in temperatures:
      model.sampling_temperature = temperature
      curve = []
      for budget in budgets:
        entropies, perplexities = [], []
        for repeat in range(seeds):
          # Seeded on (repeat, budget, rank) only -- NOT on the temperature --
          # which is what makes the comparison paired across curves. The rank
          # term is what keeps eight GPUs from drawing the same eight samples.
          torch.manual_seed(_SEED_STRIDE * (repeat + 1) + budget
                            + _RANK_STRIDE * rank)
          # torchmetrics aggregate until reset, so without this each point
          # would carry the previous points' samples too and every curve would
          # flatten into a running average.
          model.metrics.reset()
          remaining, first = local_samples, True
          while remaining > 0:
            # Never more than `batch_size` per call: `record_generative_
            # perplexity` floor-divides its input by `eval_ppl_batch_size`, so
            # a larger chunk would silently drop its remainder from gen_ppl
            # while entropy still counted those sequences.
            chunk = min(batch_size, remaining)
            remaining -= chunk
            tokens = model.generate_samples(num_samples=chunk,
                                            num_steps=budget)
            model.metrics.record_entropy(tokens)
            text = model.tokenizer.batch_decode(tokens)
            model.metrics.record_generative_perplexity(
              text, max_length, device=model.device)
            if (keep_samples and budget == budgets[-1] and repeat == 0
                and first and rank == 0):
              samples[temperature] = list(text[:keep_samples])
            first = False
          # Collective when distributed: every rank must reach both computes,
          # and each returns the pooled value, so the curves agree everywhere.
          entropies.append(float(model.metrics.sample_entropy.compute()))
          perplexities.append(float(model.metrics.gen_ppl.compute()))
        point = {'steps': budget,
                 'entropy': _mean(entropies),
                 'gen_ppl': _mean(perplexities),
                 'entropy_std': _std(entropies),
                 'gen_ppl_std': _std(perplexities),
                 'seeds': seeds}
        curve.append(point)
        spread = ('' if seeds < 2 else
                  f" +/- {point['gen_ppl_std']:.2f} / "
                  f"{point['entropy_std']:.4f} over {seeds} seeds")
        if rank == 0:
          print(f"[pareto-temp] T={temperature:g} budget={budget}: "
                f"gen_ppl {point['gen_ppl']:.2f} "
                f"entropy {point['entropy']:.4f}{spread}", flush=True)
      curves[temperature] = curve
  finally:
    model.sampling_temperature = original_temperature
  return curves, samples


def publish(curves, samples, logger, save_dir, checkpoint_label,
            reference=None, json_name='pareto_temperature.json'):
  """Write the sweep to disk and to the tracker.

  Three views, because no single one answers everything:
    * the figure -- all curves on one axes, which is how the trade-off is read;
    * `gen_ppl`/`entropy` against the step budget, one series per temperature,
      which is how you read "what does temperature cost me at 8 steps";
    * the frontier itself, x-axis entropy, which is the figure as scrubbable
      scalars.
  """
  path = os.path.join(save_dir, json_name)
  payload = {'checkpoint': checkpoint_label,
             'curves': {f'{t:g}': curve for t, curve in curves.items()},
             # Kept so `python -m pareto_temperature <json>` can redraw and
             # re-publish everything without paying for the sampling again.
             'samples': {f'{t:g}': texts for t, texts in (samples or {}).items()}}
  try:
    os.makedirs(save_dir, exist_ok=True)
    with open(path, 'w') as f:
      json.dump(payload, f, indent=2)
    print(f'[pareto-temp] wrote {path}', flush=True)
  except Exception as e:  # pragma: no cover - best-effort
    print(f'[pareto-temp] json write failed: {e}', flush=True)

  if logger is None:
    return path

  # Grouped by budget, like the figure: a series is one step count, and its
  # x-axis is the temperature walking along that budget's frontier.
  by_budget = collections.defaultdict(list)
  for temperature, curve in curves.items():
    for point in curve:
      by_budget[int(point['steps'])].append(
        dict(point, temperature=float(temperature)))

  for budget, points in sorted(by_budget.items()):
    points = sorted(points, key=lambda p: p['temperature'])
    tag = f'S{budget}'
    for point in points:
      # x-axis is temperature (scaled to an integer): what sharpening costs at
      # this budget, read one budget at a time.
      _log(logger, {f'pareto_temp/gen_ppl_{tag}': point['gen_ppl'],
                    f'pareto_temp/entropy_{tag}': point['entropy']},
           step=int(round(point['temperature'] * _TEMP_X_SCALE)))
    for point in points:
      # x-axis is entropy: the frontier, as the figure draws it. Separate loop
      # because it is a different step axis, and interleaving the two would
      # leave the tracker's series non-monotone in their own x.
      _log(logger, {f'pareto_temp/frontier_{tag}': point['gen_ppl']},
           step=int(round(point['entropy'] * _ENTROPY_X_SCALE)))

  if samples and hasattr(logger, 'log_table'):
    rows = [[f'{t:g}', text]
            for t, texts in sorted(samples.items()) for text in texts]
    try:
      logger.log_table(key='pareto_temp/samples',
                       columns=['temperature', 'sample'], data=rows, step=0)
    except Exception as e:  # pragma: no cover - best-effort
      print(f'[pareto-temp] sample table failed: {e}', flush=True)

  if hasattr(logger, 'log_image'):
    try:
      image = owt_pareto_plot.render_temperature(
        curves, reference=reference, subtitle=checkpoint_label)
      logger.log_image('pareto/temperature_frontier', image, step=0)
      print('[pareto-temp] figure logged', flush=True)
    except Exception as e:  # pragma: no cover - plotting is best-effort
      print(f'[pareto-temp] figure failed: {e}', flush=True)
  return path


def _log(logger, metrics, step):
  try:
    logger.log_metrics(metrics, step=step)
  except Exception as e:  # pragma: no cover - tracker best-effort
    print(f'[pareto-temp] scalar log failed: {e}', flush=True)


def load(path):
  """Read a sweep back from its JSON, keyed the way `sweep` returns it."""
  with open(path) as f:
    data = json.load(f)
  curves = {float(t): points for t, points in data.get('curves', {}).items()}
  samples = {float(t): texts for t, texts in (data.get('samples') or {}).items()}
  return curves, samples, data.get('checkpoint', os.path.basename(path))


def _cli(argv=None):
  """Redraw and re-publish a finished sweep: `python -m pareto_temperature X`.

  The sampling is the expensive part and it is already paid for once the JSON
  exists, so changing how the frontier is drawn or which reference sits under
  it should not cost another GPU-hour.
  """
  import argparse

  parser = argparse.ArgumentParser(description=_cli.__doc__)
  parser.add_argument('json_path', help='a pareto_temperature.json')
  parser.add_argument('--name', default=None,
                      help='tracker run name (default: from the checkpoint)')
  parser.add_argument('--experiment', default='generative-models/owt')
  parser.add_argument('--reference', default=None,
                      help='another sweep or frontier JSON, drawn dashed')
  parser.add_argument('--png', default=None,
                      help='also write the figure here')
  parser.add_argument('--no-tracker', action='store_true',
                      help='write the figure only, publish nothing')
  args = parser.parse_args(argv)

  curves, samples, checkpoint = load(args.json_path)
  save_dir = os.path.dirname(os.path.abspath(args.json_path))
  reference = owt_pareto_plot.load_reference(args.reference)

  if args.png:
    import matplotlib.pyplot as plt
    plt.imsave(args.png, owt_pareto_plot.render_temperature(
      curves, reference=reference, subtitle=checkpoint))
    print(f'[pareto-temp] wrote {args.png}', flush=True)

  if args.no_tracker:
    return

  import underdeep_logger
  logger = underdeep_logger.UnderdeepLogger(
    experiment=args.experiment,
    name=args.name or f'pareto-temp {checkpoint}',
    save_dir=save_dir, resume=False, force_new_run=True)
  publish(curves, samples, logger, save_dir=save_dir,
          checkpoint_label=checkpoint, reference=reference,
          json_name=os.path.basename(args.json_path))
  logger.finalize('success')


if __name__ == '__main__':
  _cli()
