"""Seed-averaged fixed-budget ancestral evaluation for TinyGSM."""

from concurrent.futures import ThreadPoolExecutor
import statistics
import math

import torch
import torch.distributed as dist


RANK_SEED_OFFSET = 104729


def settings(model, cfg=None):
    """Return ordered distinct budgets/seeds, retaining the legacy one-run fallback."""
    from sdtt_eval import _greedy_steps
    cfg = model.config.eval if cfg is None else cfg
    budgets = cfg.get('gsm8k_ancestral_steps')
    seeds = cfg.get('gsm8k_ancestral_seeds')
    try:
        budgets = [_greedy_steps(model, cfg)] if budgets is None else list(budgets)
        seeds = [model.config.get('seed', 1)] if seeds is None else list(seeds)
    except TypeError as error:
        raise ValueError('GSM8K ancestral steps and seeds must be lists or null.') from error
    if (not budgets or any(type(value) is not int or value <= 0 for value in budgets)
            or len(set(budgets)) != len(budgets)):
        raise ValueError('GSM8K ancestral steps must be distinct positive integers.')
    if (not seeds or any(type(value) is not int or not 0 <= value < 2**64 for value in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('GSM8K ancestral seeds must be distinct nonnegative 64-bit integers.')
    return budgets, seeds


def protocol(model, cfg=None):
    """Describe the requested grid and actual cache-aware NFE accounting."""
    cfg = model.config.eval if cfg is None else cfg
    budgets, seeds = settings(model, cfg)
    predictor = model.config.sampling.get('predictor', 'ancestral_cache')
    if predictor not in ('ancestral', 'ancestral_cache'):
        raise ValueError('GSM8K ancestral evaluation requires an ancestral predictor.')
    final_denoise = bool(cfg.get('gsm8k_ancestral_final_denoise', False))
    temperature = float(cfg.get('gsm8k_ancestral_temperature', 0.))
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Ancestral temperature must be finite and nonnegative.')
    return {
        'sampler': 'random', 'predictor': predictor, 'temperature': temperature,
        'steps': budgets, 'seeds': seeds, 'epsilon': 1e-5, 'std_ddof': 1,
        'single_seed_std': None, 'rank_seed_offset': RANK_SEED_OFFSET,
        'seed_policy': 'seed + rank_seed_offset * rank',
        'final_denoise': final_denoise,
        'position_selection': 'ancestral Bernoulli unmasking',
        'final_removal': ('one separate cleanup forward after the requested transitions'
                          if final_denoise else 'inside the last scheduled step; no extra forward'),
        'nfe_definition': 'actual cached batch forwards, shared by all rows in each batch; '
                          'mean per example, including separate cleanup when enabled',
    }


def _check_error(error, distributed, raise_if_failed):
    # Standalone evaluation must never join an existing process group's collectives.
    if distributed:
        raise_if_failed(error)
    elif error is not None:
        raise error


@torch.no_grad()
def sweep(model, examples, *, distributed=True, on_point=None, cfg=None):
    """Pool seed-level counts across ranks, then report mean/sample SD per budget.

    The callback receives each completed budget on every participating rank.
    Callback failures reach the same collective boundary before the next budget.
    """
    from sdtt_eval import (_evaluate_greedy, _raise_if_failed, _rng_devices,
                           _seed_model_rng, _settings)
    cfg = model.config.eval if cfg is None else cfg
    initialized = bool(distributed and dist.is_available() and dist.is_initialized())
    rank = dist.get_rank() if initialized else 0
    world_size = dist.get_world_size() if initialized else 1
    error = None
    try:
        if not examples:
            raise ValueError('GSM8K examples must be nonempty.')
        metadata = protocol(model, cfg)
        budgets, seeds = metadata['steps'], metadata['seeds']
        if any(seed + RANK_SEED_OFFSET * rank >= 2**64 for seed in seeds):
            raise ValueError('GSM8K ancestral rank-offset seeds exceed the 64-bit RNG range.')
        _, batch_size, workers, _, _, timeout = _settings(cfg, [0.])
        metadata.update(batch_size=batch_size, world_size=world_size, examples=len(examples))
    except Exception as caught:
        error = caught
    _check_error(error, initialized, _raise_if_failed)
    local_examples = examples[rank::world_size]
    points = []
    with torch.random.fork_rng(devices=_rng_devices(model)), ThreadPoolExecutor(max_workers=workers) as pool:
        for budget in budgets:
            runs = []
            for seed in seeds:
                error = None
                try:
                    _seed_model_rng(model, seed + RANK_SEED_OFFSET * rank)
                    options = {'final_denoise': True} if metadata['final_denoise'] else {}
                    if metadata['temperature'] != 0:
                        options['temperature'] = metadata['temperature']
                    correct, count, total_nfe, batch_forwards = _evaluate_greedy(
                        model, local_examples, pool, batch_size, budget, timeout, **options)
                except Exception as caught:
                    error = caught
                _check_error(error, initialized, _raise_if_failed)
                totals = torch.tensor([correct, count, total_nfe, batch_forwards],
                                      dtype=torch.float64, device=model.device)
                if initialized:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                correct, count, total_nfe, batch_forwards = [int(value) for value in totals.tolist()]
                run = {'seed': seed, 'accuracy': correct / count, 'correct': correct, 'count': count,
                       'nfe': total_nfe / count, 'total_batch_forwards': batch_forwards}
                runs.append(run)
                if rank == 0:
                    print(f'[gsm8k-ancestral] steps {budget} seed {seed}: '
                          f'{100 * run["accuracy"]:.2f}% ({correct}/{count}) '
                          f'at {run["nfe"]:.2f} NFE/example [T={metadata["temperature"]:g}]', flush=True)
            accuracy = [run['accuracy'] for run in runs]
            nfe = [run['nfe'] for run in runs]
            point = {'steps': budget, 'accuracy_mean': statistics.mean(accuracy),
                     'accuracy_std': statistics.stdev(accuracy) if len(runs) > 1 else None,
                     'nfe_mean': statistics.mean(nfe),
                     'nfe_std': statistics.stdev(nfe) if len(runs) > 1 else None,
                     'num_seeds': len(runs), 'seeds': runs}
            points.append(point)
            error = None
            try:
                if on_point is not None:
                    on_point(point)
            except Exception as caught:
                error = caught
            _check_error(error, initialized, _raise_if_failed)
    return {'points': points, 'protocol': metadata}
