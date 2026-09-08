"""Optional temperature/budget sweeps using the original DiDi sampler.

Extra generation is opt-in. Every rank generates its configured per-rank
sample count; rank zero evaluates the pooled tokens with a separate reference
LM. Training's metrics, random state, gradients and mixed module modes are not
used as scratch state. All tracker axes remain actual training steps.
"""

from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
from pathlib import Path

import lightning as L
import torch

from . import UPSTREAM_REVISION
from .monitor import _entropy, _root_metrics_class, config_rows, preserve_rng_state
from .temperature import sampling_temperature
from .sampling_protocol import sampling_protocol


def _distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _synchronized_error(error):
    """Synchronize Python-stage failures, not failed workers/process groups."""
    rank, world = _distributed()
    errors = [error]
    if world > 1:
        errors = [None] * world
        torch.distributed.all_gather_object(errors, error)
    if errors and all(value == errors[0] for value in errors) and errors[0] and errors[0].startswith('rank '):
        return errors[0]
    return '; '.join(f'rank {i}: {value}' for i, value in enumerate(errors) if value) or None


def _gather_tokens(tokens):
    rank, world = _distributed()
    if world == 1:
        return tokens
    gathered = [None] * world if rank == 0 else None
    torch.distributed.gather_object(tokens, gathered, dst=0)
    return torch.cat(gathered) if rank == 0 else None


@contextmanager
def _sampling_state(model, sampler):
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.grad, parameter.requires_grad)
                 for parameter in model.parameters()]
    original_sampler = model.sampler
    try:
        # Keep original grad tensors out of reach of any sampling backward().
        # Restoring their references avoids cloning a whole training model's
        # gradient buffers just to run a diagnostic.
        for parameter, _, _ in gradients:
            parameter.grad = None
        model.sampler = sampler
        model.eval()
        yield
    finally:
        model.sampler = original_sampler
        for parameter, gradient, requires_grad in gradients:
            parameter.grad = gradient
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training


def _mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return float(mean), float(math.sqrt(variance))


def _integer(name, value, minimum=1):
    try:
        result = int(value)
        valid = not isinstance(value, bool) and float(value) == result and result >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return result


def _temperature_key(value):
    """Round-trip float labels without losing the established ``t1`` spelling."""
    result = str(float(value))
    return result[:-2] if result.endswith('.0') else result


class DiDiPareto(L.Callback):
    def __init__(self, config, metrics_factory=None):
        self.config = config
        self.options = config.get('didi_pareto', {})
        self.every = _integer('didi_pareto.every_n_validations', self.options.get('every_n_validations', 0), 0)
        self.temperatures = sorted({float(t) for t in self.options.get(
            'temperatures', [.75, .85, .95, 1.0, 1.05, 1.2])})
        self.budgets = sorted({_integer('didi_pareto.steps', b) for b in self.options.get('steps', [8, 16, 32])})
        self.seeds = _integer('didi_pareto.seeds', self.options.get('seeds', 1))
        self.num_batches = _integer('didi_pareto.num_batches', self.options.get('num_batches', 2))
        batch_size = self.options.get('batch_size', None)
        self.batch_size = _integer('didi_pareto.batch_size', config.loader.eval_batch_size if batch_size is None else batch_size)
        self.keep_samples = _integer('didi_pareto.keep_samples', self.options.get('keep_samples', 2), 0)
        if self.every < 0:
            raise ValueError('didi_pareto.every_n_validations must be nonnegative')
        if not self.temperatures or any(not math.isfinite(t) or t <= 0 for t in self.temperatures):
            raise ValueError('didi_pareto.temperatures must contain positive finite values')
        if not self.budgets or min(self.budgets) < 1:
            raise ValueError('didi_pareto.steps must contain positive budgets')
        if min(self.seeds, self.num_batches, self.batch_size) < 1 or self.keep_samples < 0:
            raise ValueError('didi_pareto seed/batch counts must be positive; keep_samples nonnegative')
        if int(config.eval.perplexity_batch_size) < 1:
            raise ValueError('eval.perplexity_batch_size must be positive')
        self.metrics_factory = metrics_factory
        self.evaluator = None
        self.validations_seen = 0
        self._history = None
        self._history_identity = None

    def state_dict(self):
        return {'validations_seen': self.validations_seen}

    def load_state_dict(self, state_dict):
        self.validations_seen = int(state_dict.get('validations_seen', 0))

    def _protocol(self, model, sampler, world):
        if sampler not in ('ancestral', 'ancestral_cache', 'guided'):
            raise ValueError('DiDi Pareto supports ancestral, ancestral_cache, or guided sampling')
        if sampler == 'guided' and min(self.budgets) < 2:
            raise ValueError('The original guided schedule needs at least two denoising steps')
        source_protocol = getattr(model, '_didi_sampling_protocol', sampling_protocol(self.config))
        if source_protocol['sampler_source'] == 'inference' and sampler != 'guided':
            raise ValueError('didi_eval.sampler_source=inference requires the guided Pareto sampler')
        return {
            'upstream_revision': UPSTREAM_REVISION,
            **source_protocol,
            'sampler': sampler,
            'weights': getattr(model, '_didi_evaluation_weights', 'backbone'),
            'temperatures': self.temperatures,
            'steps': self.budgets,
            'seeds': self.seeds,
            'seed_formula': '9973*(repeat+1)+steps+104729*rank',
            'batch_size_per_rank': self.batch_size,
            'batches_per_rank': self.num_batches,
            'world_size': world,
            'samples_per_seed': self.batch_size * self.num_batches * world,
            'sequence_length': int(model.num_tokens),
            'sampler_eps': 1e-5,
            'trainer_precision': str(self.config.get('trainer', {}).get('precision', 'unknown')),
            'noise_eps': float(getattr(getattr(model, 'noise', None), 'eps', 1e-3)),
            'eval_model': str(self.config.eval.gen_ppl_eval_model_name_or_path),
            'eval_batch_size': int(self.config.eval.perplexity_batch_size),
            'noise_removal': str(self.config.sampling.noise_removal),
            'use_float64': bool(self.config.sampling.use_float64),
            'p_nucleus': float(self.config.sampling.p_nucleus),
            'guidance': {name: getattr(model, name, None) for name in (
                'guidance_scale_start', 'guidance_scale_end', 'rerank_steps_ratio', 'num_candidates')},
        }

    def _get_evaluator(self, model):
        if self.evaluator is None:
            factory = self.metrics_factory or _root_metrics_class()
            self.evaluator = factory(
                gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
                eval_ppl_batch_size=int(self.config.eval.perplexity_batch_size))
            for value in vars(self.evaluator).values():
                if isinstance(value, torch.nn.Module):
                    for module in value.modules():
                        if hasattr(module, 'sync_on_compute'):
                            module.sync_on_compute = False
                            # Older TorchMetrics caches this flag in __init__;
                            # changing only the public flag still all-reduces.
                            module._to_sync = False
        self.evaluator.to(model.device)
        reference = getattr(self.evaluator, '_eval_model', None)
        if reference is not None:
            reference.to(model.device)
        return self.evaluator

    def _offload(self):
        reference = getattr(self.evaluator, '_eval_model', None)
        if reference is not None:
            reference.to('cpu')

    def _evaluate_tokens(self, model, tokens):
        try:
            evaluator = self._get_evaluator(model)
            texts = model.tokenizer.batch_decode(tokens)
            evaluator.gen_ppl.reset()
            chunk = int(self.config.eval.perplexity_batch_size)
            with torch.no_grad():
                for start in range(0, len(texts), chunk):
                    evaluator.record_generative_perplexity(
                        texts[start:start + chunk], max_length=model.num_tokens, device=model.device)
                perplexity = float(evaluator.gen_ppl.compute())
            entropy = _entropy(tokens)
            if not math.isfinite(perplexity) or not math.isfinite(entropy):
                raise ValueError(f'nonfinite sweep metrics: entropy={entropy}, gen_ppl={perplexity}')
            return entropy, perplexity, texts[:self.keep_samples]
        finally:
            # Do not leave the reference LM occupying sampling GPU memory.
            self._offload()

    def _sweep(self, model, protocol):
        rank, world = _distributed()
        curves, samples = {}, {}
        for temperature in self.temperatures:
            curve = []
            for budget in self.budgets:
                entropies, perplexities = [], []
                for repeat in range(self.seeds):
                    error, batches = None, []
                    try:
                        torch.manual_seed(9973 * (repeat + 1) + budget + 104729 * rank)
                        with sampling_temperature(model, temperature):
                            for _ in range(self.num_batches):
                                # The original sampler exits in train mode.
                                # Restore eval mode before each independent call.
                                model.eval()
                                tokens = model.generate_samples(num_samples=self.batch_size, num_steps=budget)
                                batches.append(tokens.detach().cpu().clone())
                        tokens = torch.cat(batches)
                        if tokens.shape != (self.batch_size * self.num_batches, model.num_tokens):
                            raise ValueError(f'unexpected sample shape {tuple(tokens.shape)}')
                    except Exception as exception:
                        error = f'generation T={temperature:g}, steps={budget}, seed={repeat}: {exception}'
                    error = _synchronized_error(error)
                    if error:
                        raise RuntimeError(error)
                    error = None
                    try:
                        tokens = _gather_tokens(tokens)
                    except Exception as exception:
                        error = f'gathering T={temperature:g}, steps={budget}, seed={repeat}: {exception}'
                    error = _synchronized_error(error)
                    if error:
                        raise RuntimeError(error)
                    error = None
                    if rank == 0:
                        try:
                            entropy, perplexity, kept = self._evaluate_tokens(model, tokens)
                            entropies.append(entropy)
                            perplexities.append(perplexity)
                            if repeat == 0 and kept:
                                samples[f'{_temperature_key(temperature)}/s{budget}'] = kept
                        except Exception as exception:
                            error = f'evaluation T={temperature:g}, steps={budget}, seed={repeat}: {exception}'
                    error = _synchronized_error(error)
                    if error:
                        raise RuntimeError(error)
                error = None
                if rank == 0:
                    try:
                        entropy, entropy_std = _mean_std(entropies)
                        perplexity, perplexity_std = _mean_std(perplexities)
                        curve.append({'steps': budget, 'entropy': entropy, 'gen_ppl': perplexity,
                                      'entropy_std': entropy_std, 'gen_ppl_std': perplexity_std,
                                      'seeds': self.seeds,
                                      'samples_per_seed': protocol['samples_per_seed'],
                                      'total_samples': protocol['samples_per_seed'] * self.seeds})
                        curves[_temperature_key(temperature)] = curve
                        print(f'[didi-pareto] T={temperature:g}, steps={budget}: '
                              f'gen_ppl={perplexity:.4f}, entropy={entropy:.4f}', flush=True)
                    except Exception as exception:
                        error = f'aggregating T={temperature:g}, steps={budget}: {exception}'
                error = _synchronized_error(error)
                if error:
                    raise RuntimeError(error)
        return curves, samples

    def _load_history(self, run_uid, protocol):
        identity = (run_uid, json.dumps(protocol, sort_keys=True))
        if self._history_identity == identity:
            return
        self._history, self._history_identity = {}, identity
        path = Path(str(self.config.checkpointing.save_dir)) / 'didi_pareto_temp_history.json'
        if not path.is_file():
            return
        original = path.read_bytes()
        try:
            stored = json.loads(original)
            if run_uid and stored.get('run_uid') == run_uid and stored.get('protocol') == protocol:
                self._history = stored['curves']
                return
        except (ValueError, KeyError, TypeError):
            pass
        # Preserve old data when a directory is reused for another run/protocol.
        archive = path.with_name(f'{path.stem}.previous.{hashlib.sha256(original).hexdigest()[:12]}.json')
        if not archive.exists():
            archive.write_bytes(original)
        print(f'[didi-pareto] incompatible previous history preserved as {archive.name}', flush=True)

    def _publish(self, trainer, model, step, protocol, curves, samples):
        logger = trainer.logger
        run_uid = str(getattr(logger, 'version', '') or '') if logger else ''
        self._load_history(run_uid, protocol)
        self._history[str(step)] = curves
        directory = Path(str(self.config.checkpointing.save_dir))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'didi_pareto_temp_history.json'
        path.write_text(json.dumps({'run_uid': run_uid, 'protocol': protocol,
                                   'curves': self._history, 'latest_samples': samples}, indent=2) + '\n')
        if not logger:
            return
        metrics, rows = {}, []
        for temperature, curve in curves.items():
            for point in curve:
                tag = f"s{point['steps']}_t{_temperature_key(temperature)}"
                for key, stem in (('gen_ppl', 'gen_ppl'), ('entropy', 'entropy'),
                                  ('gen_ppl_std', 'gen_ppl_std'), ('entropy_std', 'entropy_std')):
                    metrics[f'val/{stem}_{tag}'] = point[key]
                rows.append([step, float(temperature), point['steps'], point['entropy'], point['gen_ppl'],
                             point['entropy_std'], point['gen_ppl_std'], self.seeds,
                             point['samples_per_seed'], point['total_samples']])
        metrics['didi_pareto/samples_per_seed'] = protocol['samples_per_seed']
        logger.log_metrics(metrics, step=step)
        if hasattr(logger, 'log_table'):
            logger.log_table(
                key='pareto/temperature_values',
                columns=['train_step', 'temperature', 'steps', 'entropy', 'gen_ppl',
                         'entropy_std', 'gen_ppl_std', 'seeds', 'samples_per_seed', 'total_samples'],
                data=rows, step=step)
            logger.log_table(key='pareto/protocol', columns=['Parameter', 'JSON value'],
                             data=config_rows(protocol), step=step)
            if samples:
                logger.log_table(key='pareto/temperature_samples', columns=['temperature/budget', 'sample'],
                                 data=[[key, text] for key, texts in samples.items() for text in texts], step=step)
        if hasattr(logger, 'log_image'):
            import owt_pareto_plot
            reference = owt_pareto_plot.load_reference(self.options.get('reference', None))
            logger.log_image('pareto/temperature_frontier',
                             owt_pareto_plot.render_temperature(
                                 curves, reference=reference, subtitle=f'training step {step}; {protocol["sampler"]}'),
                             step=step)
            if len(self._history) > 1:
                sweeps = [{'short': f'{int(saved_step) / 1000:g}k',
                           'label': f'training step {saved_step}', 'curves': saved}
                          for saved_step, saved in sorted(self._history.items(), key=lambda item: int(item[0]))]
                logger.log_image('pareto/temperature_by_checkpoint',
                                 owt_pareto_plot.render_checkpoints(sweeps, reference=reference), step=step)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.every <= 0:
            return
        baseline = getattr(pl_module, '_didi_evaluation_step', None)
        if baseline is None:
            self.validations_seen += 1
            if (self.validations_seen - 1) % self.every:
                return
        step = int(baseline['value']) if baseline is not None else int(trainer.global_step)
        rank, world = _distributed()
        sampler = self.options.get('sampler', None) or pl_module.sampler
        error, protocol = None, None
        with preserve_rng_state(), torch.inference_mode(False):
            try:
                protocol = self._protocol(pl_module, sampler, world)
            except Exception as exception:
                error = str(exception)
            error = _synchronized_error(error)
            if not error:
                if rank == 0:
                    points = len(self.temperatures) * len(self.budgets)
                    total = protocol['samples_per_seed'] * self.seeds * points
                    print(f'[didi-pareto] starting step={step}, sampler={sampler}, '
                          f'{points} points x {self.seeds} seed(s), '
                          f'{protocol["samples_per_seed"]} global samples/seed/point '
                          f'({total} samples total)', flush=True)
                state = ExitStack()
                try:
                    try:
                        state.enter_context(_sampling_state(pl_module, sampler))
                    except Exception as exception:
                        error = f'preparing sampler: {exception}'
                    error = _synchronized_error(error)
                    if not error:
                        try:
                            curves, samples = self._sweep(pl_module, protocol)
                        except Exception as exception:
                            error = str(exception)
                    error = _synchronized_error(error)
                    if not error and rank == 0:
                        try:
                            self._publish(trainer, pl_module, step, protocol, curves, samples)
                        except Exception as exception:
                            error = f'publishing: {exception}'
                    error = _synchronized_error(error)
                finally:
                    cleanup_error = None
                    try:
                        state.close()
                    except Exception as exception:
                        cleanup_error = f'restoring sampler state: {exception}'
                    cleanup_error = _synchronized_error(cleanup_error)
                    error = error or cleanup_error
            if rank == 0:
                if error:
                    print(f'[didi-pareto] sweep failed; training continues: {error}', flush=True)
                else:
                    print(f'[didi-pareto] completed step={step}', flush=True)
                if trainer.logger:
                    try:
                        trainer.logger.log_metrics({'didi_pareto/failed': int(error is not None)}, step=step)
                        if error and hasattr(trainer.logger, 'log_table'):
                            trainer.logger.log_table(key='pareto/errors', columns=['Error'], data=[[error]], step=step)
                    except Exception as exception:
                        print(f'[didi-pareto] status logging failed: {exception}', flush=True)


def attach_pareto(model, config):
    """Return a callback; append it after the original-sample monitor callback."""
    every = _integer('didi_pareto.every_n_validations', config.get('didi_pareto', {}).get('every_n_validations', 0), 0)
    if every == 0:
        return None
    validate_options(config)
    return DiDiPareto(config)


def validate_options(config):
    """Pure config preflight, before teacher/data/tracker initialization."""
    if _integer('didi_pareto.every_n_validations', config.get('didi_pareto', {}).get('every_n_validations', 0), 0) == 0:
        return
    callback = DiDiPareto(config)
    if config.get('trainer', {}).get('limit_val_batches', 1) == 0:
        raise ValueError('didi_pareto requires trainer.limit_val_batches > 0')
    sampler = callback.options.get('sampler', None) or config.sampling.predictor
    if sampling_protocol(config)['sampler_source'] == 'inference' and sampler != 'guided':
        raise ValueError('didi_eval.sampler_source=inference requires the guided Pareto sampler')
    if sampler not in ('ancestral', 'ancestral_cache', 'guided'):
        raise ValueError('DiDi Pareto supports ancestral, ancestral_cache, or guided sampling')
    if sampler == 'guided' and min(callback.budgets) < 2:
        raise ValueError('The original guided schedule needs at least two denoising steps')
