"""Paired raw/EMA GSM8K validation, including ancestral pass@k.

FP32 prediction normalization with the configured backbone forward precision.
The pass@k protocol matches scripts/paper_recompute/paper_eval.py: S-FLM NFE,
random reveals, T=0 by default, independent draws, unbiased pass@k estimator.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from eval_precision import capture_rng_state, replay_rng_state, fp32_prediction_normalization


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('Require 0 <= correct <= samples and 1 <= k <= samples')
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


@contextmanager
def validation_weights(model, weights):
    """Select weights inside an active validation; restore exactly on exit.

    _eval_mode has already stored live parameters and installed EMA. Never use
    requires_grad to select parameters: the student can be entirely frozen.
    Do not overwrite EMA's saved live parameters or its update counter.
    """
    params = list(model._get_parameters())
    previous = [p.detach().clone() for p in params]
    if weights not in ('raw', 'ema'):
        raise ValueError(f'Unknown evaluation weights: {weights}')
    ema = model.ema
    if weights == 'ema' and ema is None:
        raise ValueError('EMA evaluation requested but training.ema is disabled')
    try:
        with torch.no_grad():
            if weights == 'ema':
                ema.copy_to(params)
            elif ema is not None:
                raw = ema.collected_params
                if len(raw) != len(params) or any(p.shape != q.shape for p, q in zip(params, raw)):
                    raise ValueError('Missing live-weight snapshot from _eval_mode')
                for p, q in zip(params, raw):
                    p.copy_(q)
        yield
    finally:
        with torch.no_grad():
            for p, q in zip(params, previous):
                p.copy_(q)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rank_zero(model, action):
    """Propagate grading/IO failures before another rank starts the next draw."""
    failure = None
    result = None
    if model.trainer.global_rank == 0:
        try:
            result = action()
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
    if model.trainer.world_size > 1:
        status = [failure]
        torch.distributed.broadcast_object_list(status, src=0)
        failure = status[0]
    if failure:
        raise RuntimeError(f'GSM8K evaluation failed: {failure}')
    return result


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def _log(model, metrics, step):
    if model.trainer.logger is not None:
        model.trainer.logger.log_metrics(metrics, step=step)


def _passk(model, dataset, indices, batch_size, weights):
    cfg = model.config.eval
    n = int(cfg.get('gsm8k_passk_samples', 32))
    budgets = [int(v) for v in cfg.get('gsm8k_passk_budgets', [4, 8, 16, 32])]
    ks = [int(v) for v in cfg.get('gsm8k_passk_ks', [1, 2, 4, 8, 16, 32])]
    if n < 1 or not ks or any(k < 1 or k > n for k in ks) or not budgets or min(budgets) < 1:
        raise ValueError('Invalid pass@k samples, ks, or NFE budgets')
    total = len(dataset)
    if total != 1319:
        raise ValueError(f'Expected full 1319-example GSM8K test set, got {total}')
    rank, world = model.trainer.global_rank, max(1, model.trainer.world_size)
    # _gsm8k_generate gathers shards in rank order, not original example order.
    order = [i for r in range(world) for i in range(r, total, world)]
    step = int(model.global_step)
    seed = int(cfg.get('gsm8k_passk_seed', 1))
    model.remasking_strategy = 'random'
    model.sampling_temperature = float(cfg.get('gsm8k_passk_temperature', 0.0))
    out = Path(model.config.checkpointing.save_dir) / 'passk' / f'step_{step:08d}' / weights
    report = dict(step=step, weights=weights, complete=False, total=total, samples=n,
                  seed=seed, budgets=budgets, ks=ks, temperature=model.sampling_temperature,
                  sampler='ancestral_cache', normalization='fp32',
                  forward_precision=str(model.config.trainer.precision), results=[])
    for budget in budgets:
        counts, nfes = [0] * total, []
        for draw in range(n):
            _seed(seed + 100003 * draw + rank)
            rows, _, per_seq = model._gsm8k_generate(dataset, indices, batch_size, budget, 'sflm')
            def grade_and_save():
                import sandbox_gsm8k
                if len(rows) != total or len(per_seq) != total:
                    raise ValueError('Incomplete gathered GSM8K predictions')
                timeout = float(cfg.get('tinygsm_exec_timeout', 5.0))
                for i, (response, gold) in zip(order, rows):
                    counts[i] += int(sandbox_gsm8k.evaluate_samples(response, gold, timeout_s=timeout))
                nfes.append(sum(per_seq) / total)
                _write(out / f'progress_nfe{budget}.json',
                       dict(draws=draw + 1, correct_per_question=counts, nfes=nfes))
                print(f'[gsm8k-passk-{weights}] step={step} NFE={budget} draw={draw+1}/{n}', flush=True)
            _rank_zero(model, grade_and_save)
        def publish():
            values = {str(k): sum(pass_at_k(n, c, k) for c in counts) / total for k in ks}
            nfe = sum(nfes) / len(nfes)
            report['results'].append(dict(steps=budget, nfe=nfe, pass_at_k=values,
                                          correct_per_question=counts.copy()))
            for k, value in values.items():
                _log(model, {f'passk/step_{step}/nfe_{budget}_{weights}_fp32norm': value}, int(k))
            _log(model, {f'val/pass_at_{k}_nfe{budget}_{weights}_fp32norm': value
                         for k, value in values.items()}, step)
            _write(out / 'results.json', report)
        _rank_zero(model, publish)
    report['complete'] = True
    _rank_zero(model, lambda: _write(out / 'results.json', report))


@torch.no_grad()
def run_weighted_benchmark(model, dataset, indices, batch_size, steps):
    cfg = model.config.eval
    weights = list(cfg.gsm8k_weights)
    if not weights or len(set(weights)) != len(weights) or any(w not in ('raw', 'ema') for w in weights):
        raise ValueError('eval.gsm8k_weights must contain distinct raw/ema entries')
    if cfg.get('gsm8k_num_examples') or len(dataset) != 1319:
        raise ValueError('Paired evaluation requires all 1319 GSM8K examples')
    if model.config.sampling.predictor != 'ancestral_cache':
        raise ValueError('Paired evaluation currently requires sampling.predictor=ancestral_cache')
    original = (model.remasking_strategy, model.dynamic_threshold, model.sampling_temperature)
    rng = capture_rng_state()
    try:
        for weight in weights:
            suffix = f'_{weight}_fp32norm'
            with replay_rng_state(rng), validation_weights(model, weight), fp32_prediction_normalization(model):
                model.remasking_strategy, model.dynamic_threshold, model.sampling_temperature = original
                temperatures = [('sampled', original[2])]
                if cfg.get('gsm8k_greedy_token', True) and original[2] != 0.0:
                    temperatures.append(('greedy', 0.0))
                for name, temperature in temperatures:
                    model.remasking_strategy = 'random'
                    model.sampling_temperature = temperature
                    rows, _, per_seq = model._gsm8k_generate(dataset, indices, batch_size, steps, 'sflm')
                    def publish():
                        correct, accuracy = model._gsm8k_grade(rows)
                        if len(rows) != 1319:
                            raise ValueError('Incomplete gathered GSM8K predictions')
                        _log(model, {f'val/gsm8k_acc_{name}{suffix}': accuracy,
                                     f'val/gsm8k_nfe_{name}{suffix}': sum(per_seq)/len(per_seq)},
                             int(model.global_step))
                        print(f'[gsm8k-{name}{suffix}] step {model.global_step}: {correct}/1319', flush=True)
                    _rank_zero(model, publish)
                model._run_gsm8k_pareto(dataset, indices, batch_size, suffix=suffix)
                if cfg.get('gsm8k_passk', False):
                    _passk(model, dataset, indices, batch_size, weight)
    finally:
        model.remasking_strategy, model.dynamic_threshold, model.sampling_temperature = original
