"""Conditional TinyGSM validation and fixed/adaptive GSM8K execution accuracy.

This is an opt-in task adaptation, not DiDi's published OWT evaluation. The
reference grader executes generated programs with restricted Python builtins;
it is NOT an operating-system security sandbox. Only enable execution grading
for trusted experiments. No grader heuristics are changed here.
"""

from contextlib import contextmanager, nullcontext
import functools
import hashlib
import importlib.util
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys

import lightning as L
from omegaconf import OmegaConf
import torch

from .monitor import preserve_rng_state
from .eval_precision import capture_rng_state, fp32_prediction_normalization, replay_rng_state


def _integer(name, value, minimum=1):
    try:
        result = int(value)
        valid = not isinstance(value, bool) and float(value) == result and result >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return result


def threshold_options(config):
    """Validate the opt-in sweep without data, models, or tracker access."""
    options = config.get('eval', {})
    if not isinstance(options.get('gsm8k_pareto_fp32', True), bool):
        raise ValueError('eval.gsm8k_pareto_fp32 must be a boolean')
    raw = options.get('gsm8k_pareto_thresholds', None)
    thresholds = []
    if raw is not None:
        if not (isinstance(raw, (list, tuple)) or OmegaConf.is_list(raw)):
            raise ValueError('eval.gsm8k_pareto_thresholds must be a list, or null to disable')
        for value in raw:
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not 0 <= value <= 1 or not math.isfinite(value)):
                raise ValueError('eval.gsm8k_pareto_thresholds must contain finite numbers in [0, 1], not booleans')
            thresholds.append(float(value))
        if len(set(thresholds)) != len(thresholds):
            raise ValueError('eval.gsm8k_pareto_thresholds contains duplicate thresholds')
        thresholds.sort()
        if thresholds and not options.get('gsm8k_at_validation', False):
            raise ValueError('eval.gsm8k_pareto_thresholds requires eval.gsm8k_at_validation=true')
    temperature = options.get('gsm8k_pareto_temperature', 0.0)
    try:
        valid_temperature = not isinstance(temperature, bool) and isinstance(temperature, Real)
        temperature = float(temperature)
        valid_temperature = valid_temperature and math.isfinite(temperature) and temperature >= 0
    except (TypeError, ValueError, OverflowError):
        valid_temperature = False
    if not valid_temperature:
        raise ValueError('eval.gsm8k_pareto_temperature must be a finite nonnegative number')
    cap = options.get('gsm8k_pareto_max_steps', None)
    if cap is not None:
        if isinstance(cap, bool) or not isinstance(cap, Integral):
            raise ValueError('eval.gsm8k_pareto_max_steps must be a positive integer or null')
        cap = _integer('eval.gsm8k_pareto_max_steps', cap)
    return thresholds, float(temperature), cap


def _distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _stage(label, operation):
    """All ranks finish a Python stage before anyone advances or raises.

    This handles ordinary generation, grading, and logger exceptions. It
    cannot recover a killed worker or a broken process group.
    """
    result, error = None, None
    try:
        result = operation()
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
    _, world = _distributed()
    errors = [error]
    if world > 1:
        errors = [None] * world
        torch.distributed.all_gather_object(errors, error)
    failures = '; '.join(f'rank {rank}: {value}' for rank, value in enumerate(errors) if value)
    if failures:
        raise RuntimeError(f'TinyGSM evaluation {label} failed: {failures}')
    return result


def _gather_rows(rows):
    rank, world = _distributed()
    if world == 1:
        return rows
    shards = [None] * world if rank == 0 else None
    # Empty ranks still participate, without needing a tensor shape inferred
    # from a nonexistent first example.
    torch.distributed.gather_object(rows, shards, dst=0)
    return [row for shard in shards for row in shard] if rank == 0 else None


@contextmanager
def _evaluation_state(model):
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.grad, parameter.requires_grad)
                 for parameter in model.parameters()]
    try:
        for parameter, _, _ in gradients:
            parameter.grad = None
        model.eval()
        yield
    finally:
        for parameter, gradient, requires_grad in gradients:
            parameter.grad = gradient
            parameter.requires_grad_(requires_grad)
        for module, mode in modes:
            module.training = mode


def _grader():
    """Use the root reference grader by path, not an upstream import alias."""
    name = '_idlm_didi_tinygsm_grader'
    path = Path(__file__).resolve().parents[1] / 'sandbox_gsm8k.py'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _benchmark_dataset(config, tokenizer):
    from .tinygsm_data import get_gsm8k_test_dataset
    return get_gsm8k_test_dataset(config, tokenizer)


def _adaptive_samples(*args, **kwargs):
    from .tinygsm_adaptive import generate_adaptive
    return generate_adaptive(*args, **kwargs)


def _publish_thresholds(*args, **kwargs):
    from .tinygsm_pareto import publish_threshold_pareto
    return publish_threshold_pareto(*args, **kwargs)


def _answer_entropy(tokens):
    if tokens.numel() == 0:
        raise ValueError('A generated answer contains no token positions')
    _, counts = torch.unique(tokens, return_counts=True, sorted=False)
    return float(torch.special.entr(counts.float() / counts.sum()).sum())


def _step(trainer, model):
    context = getattr(model, '_didi_evaluation_step', None)
    return int(context['value'] if context is not None else trainer.global_step)


def _loggers(trainer):
    values = list(getattr(trainer, 'loggers', []) or [])
    if not values and getattr(trainer, 'logger', None):
        values = [trainer.logger]
    return list({id(logger): logger for logger in values}.values())


class DiDiTinyGSMEvaluation(L.Callback):
    def __init__(self, config, dataset_factory=None, grader_factory=None,
                 adaptive_generator=None, threshold_publisher=None):
        self.config = config
        self.options = config.get('eval', {})
        self.conditional = bool(self.options.get('generate_samples', False))
        self.benchmark = bool(self.options.get('gsm8k_at_validation', False))
        self.execute = bool(self.options.get('tinygsm_exec', False))
        self.thresholds, self.threshold_temperature, self.threshold_cap = threshold_options(config)
        self.every = _integer('eval.gsm8k_every_n_validations', self.options.get('gsm8k_every_n_validations', 1))
        self.sample_batches = _integer('sampling.num_sample_batches', config.sampling.get('num_sample_batches', 1), 0)
        self.keep_samples = _integer('sampling.num_sample_log', config.sampling.get('num_sample_log', 8), 0)
        self.primary_steps = _integer('sampling.steps', config.sampling.steps)
        requested = self.options.get('gsm8k_steps', None)
        if requested is None:
            requested = []
        elif isinstance(requested, (int, float, bool)):
            requested = [requested]
        if isinstance(requested, str):
            raise ValueError('eval.gsm8k_steps must be an integer or a list of integers')
        self.budgets = [self.primary_steps] + sorted({
            _integer('eval.gsm8k_steps', value) for value in requested} - {self.primary_steps})
        configured_batch = self.options.get('gsm8k_batch_size', None)
        self.batch_size = _integer('eval.gsm8k_batch_size', config.loader.eval_batch_size
                                   if configured_batch is None else configured_batch)
        limit = self.options.get('gsm8k_num_examples', None)
        self.limit = None if limit is None else _integer('eval.gsm8k_num_examples', limit)
        self.timeout = float(self.options.get('tinygsm_exec_timeout', 5.0))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('eval.tinygsm_exec_timeout must be positive and finite')
        if self.conditional and self.sample_batches == 0:
            raise ValueError('eval.generate_samples requires sampling.num_sample_batches > 0')
        self.dataset_factory = dataset_factory or _benchmark_dataset
        self.grader_factory = grader_factory or _grader
        self.adaptive_generator = adaptive_generator or _adaptive_samples
        self.threshold_publisher = threshold_publisher or _publish_thresholds
        self.dataset = None
        self.validations_seen = 0
        self.batches = []
        self.buffer_error = None

    def state_dict(self):
        return {'validations_seen': self.validations_seen}

    def load_state_dict(self, state_dict):
        self.validations_seen = _integer('validations_seen', state_dict.get('validations_seen', 0), 0)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.batches = []
        self.buffer_error = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking or not self.conditional or len(self.batches) >= self.sample_batches:
            return
        # Do not raise here: another rank may have fewer/no validation batches.
        # Synchronize a buffering error at the common epoch-end boundary.
        try:
            if dataloader_idx != 0:
                raise ValueError('TinyGSM conditional evaluation supports one validation dataloader')
            self.batches.append({key: batch[key].detach().cpu().clone()
                                 for key in ('input_ids', 'attention_mask', 'prompt_len') if key in batch})
        except Exception as exception:
            self.buffer_error = f'{type(exception).__name__}: {exception}'

    def _generate(self, model, inputs, lengths, steps):
        if inputs.ndim != 2 or lengths.ndim != 1 or len(inputs) != len(lengths):
            raise ValueError('Conditional generation requires 2D input_ids and one prompt_len per row')
        if len(inputs) == 0:
            return inputs
        if (lengths <= 0).any() or (lengths >= model.num_tokens).any() or (lengths > inputs.shape[1]).any():
            raise ValueError('Each prompt must be nonempty and shorter than model.length, with tokens present')
        inputs, lengths = inputs.to(model.device), lengths.to(model.device)
        model.eval()  # Original samplers may exit in train mode.
        samples = model.generate_samples(
            num_samples=len(inputs), num_steps=steps,
            prefix_tokens=inputs[:, :int(lengths.max())], prefix_lengths=lengths)
        return self._validate_samples(model, samples, inputs, lengths)

    @staticmethod
    def _validate_samples(model, samples, inputs, lengths):
        if not isinstance(samples, torch.Tensor) or samples.shape != (len(inputs), model.num_tokens):
            raise ValueError('Unexpected conditional sample shape')
        for index, length in enumerate(lengths.tolist()):
            if not torch.equal(samples[index, :length], inputs[index, :length]):
                raise ValueError('Conditional sampler changed a prompt token')
        return samples.detach().cpu()

    def _conditional_rows(self, model):
        if self.buffer_error:
            raise ValueError(self.buffer_error)
        rows = []
        for batch in self.batches:
            inputs = batch['input_ids']
            if 'prompt_len' not in batch:
                raise ValueError('TinyGSM evaluation requires explicit prompt_len; loss masks are not prompt boundaries')
            lengths = batch['prompt_len'].long()
            samples = self._generate(model, inputs, lengths, self.primary_steps)
            for index, length in enumerate(lengths.tolist()):
                answer = samples[index, length:]
                decode = lambda ids: model.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                rows.append({'question': decode(inputs[index, :length]), 'generated': decode(answer),
                             'gold': decode(inputs[index, length:]), 'entropy': _answer_entropy(answer)})
        return rows

    def _conditional_metrics(self, rows):
        if not rows:
            raise ValueError('No conditional validation rows were generated on any rank')
        count = len(rows)
        metrics = {'val/answer_count': count,
                   'val/answer_exact_match': sum(row['generated'].strip() == row['gold'].strip()
                                                 for row in rows) / count,
                   'val/sample_entropy': sum(row['entropy'] for row in rows) / count}
        if self.execute:
            grader = self.grader_factory()
            correct = gradable = 0
            for row in rows:
                ok, can = grader.evaluate_tinygsm_sample(row['generated'], row['gold'], timeout_s=self.timeout)
                if ok and not can:
                    raise ValueError('Grader returned correct=True for an ungradable reference')
                correct += int(ok)
                gradable += int(can)
            metrics.update({'val/answer_correct': correct, 'val/answer_gradable_count': gradable,
                            'val/answer_gradable': gradable / count})
            # No gradable references is an undefined accuracy, not zero.
            if gradable:
                metrics['val/answer_acc'] = correct / gradable
        return metrics

    def _load_dataset(self, model):
        if self.dataset is None:
            self.dataset = self.dataset_factory(self.config, model.tokenizer)
        total = len(self.dataset) if self.limit is None else min(self.limit, len(self.dataset))
        if total == 0:
            raise ValueError('GSM8K benchmark dataset is empty')
        return total

    def _benchmark_batches(self, total):
        rank, world = _distributed()
        indices = list(range(rank, total, world))
        for start in range(0, len(indices), self.batch_size):
            chunk = indices[start:start + self.batch_size]
            prompts = [torch.as_tensor(self.dataset[index]['input_ids'], dtype=torch.long) for index in chunk]
            if any(prompt.ndim != 1 for prompt in prompts):
                raise ValueError('GSM8K prompt input_ids must be one-dimensional')
            lengths = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long)
            padded = torch.zeros(len(prompts), int(lengths.max()), dtype=torch.long)
            for index, prompt in enumerate(prompts):
                padded[index, :len(prompt)] = prompt
            yield chunk, padded, lengths

    def _decoded_rows(self, model, chunk, samples, lengths):
        rows = []
        for offset, index in enumerate(chunk):
            row = self.dataset[index]
            gold = row['response_ground_truth']
            if not isinstance(gold, str) or not gold.strip():
                raise ValueError('GSM8K response_ground_truth must be a nonempty answer string')
            answer = samples[offset, int(lengths[offset]):]
            rows.append({'index': index, 'generated': model.tokenizer.decode(answer.tolist(), skip_special_tokens=True),
                         'gold': gold, 'entropy': _answer_entropy(answer)})
        return rows

    def _benchmark_rows(self, model, total, steps):
        rows = []
        for chunk, padded, lengths in self._benchmark_batches(total):
            samples = self._generate(model, padded, lengths, steps)
            rows.extend(self._decoded_rows(model, chunk, samples, lengths))
        return rows

    def _threshold_batches(self, model, total, threshold, *, fp32norm=False):
        batches = []
        for chunk, padded, lengths in self._benchmark_batches(total):
            model.eval()
            padded, lengths = padded.to(model.device), lengths.to(model.device)
            # Enter inside the synchronized generation stage, including errors
            # on a rank with data while another rank has an empty shard.
            with fp32_prediction_normalization(model) if fp32norm else nullcontext():
                samples, metadata = self.adaptive_generator(
                    model, prefix_tokens=padded, prefix_lengths=lengths, threshold=threshold,
                    temperature=self.threshold_temperature, max_steps=self.threshold_cap,
                    return_metadata=True)
            samples = self._validate_samples(model, samples, padded, lengths)
            if metadata.get('complete') is not True or bool((samples == model.mask_index).any()):
                raise ValueError('Adaptive sweep returned incomplete output; no partial accuracy will be published')
            nfe = _integer('adaptive metadata.nfe', metadata['nfe'])
            per_row = torch.as_tensor(metadata['nfe_per_row']).detach().cpu()
            if (per_row.shape != (len(chunk),) or per_row.dtype == torch.bool or per_row.is_complex()
                    or not bool(torch.isfinite(per_row).all()) or bool((per_row < 1).any())
                    or bool((per_row > nfe).any()) or not torch.equal(per_row, per_row.round())):
                raise ValueError('Adaptive nfe_per_row must contain one positive integral forward count per example, <= batch nfe')
            fallback = _integer('adaptive metadata.fallback_nfe', metadata.get('fallback_nfe', 0), 0)
            if fallback > 1 or fallback > nfe:
                raise ValueError('Adaptive fallback_nfe must be 0 or 1, included in batch nfe')
            cap = _integer('adaptive metadata.max_steps', metadata['max_steps'])
            if metadata.get('remasking_strategy') != 'low_confidence_dynamic':
                raise ValueError('Threshold sweep must use low_confidence_dynamic, not the fixed/guided sampler')
            weights = metadata.get('weights')
            if weights != getattr(model, '_didi_evaluation_weights', 'backbone'):
                raise ValueError('Adaptive sampler evaluated a different weight set than configured')
            physical_forwards = _integer('adaptive metadata.physical_row_forwards',
                                         metadata.get('physical_row_forwards', nfe * len(chunk)))
            if physical_forwards != nfe * len(chunk):
                raise ValueError('Adaptive physical_row_forwards disagrees with full-batch forward accounting')
            rows = self._decoded_rows(model, chunk, samples, lengths.cpu())
            for row, forwards in zip(rows, per_row.tolist()):
                row['nfe'] = int(forwards)
            batches.append({'rows': rows, 'nfe': nfe, 'fallback_nfe': fallback, 'max_steps': cap,
                            'weights': weights, 'physical_row_forwards': physical_forwards})
        return batches

    def _threshold_sweep(self, model, total, rank, *, fp32norm=False):
        curve = []
        suffix = '_fp32norm' if fp32norm else ''
        for threshold in self.thresholds:
            local = _stage(f'threshold generation{suffix} {threshold:g}',
                           lambda: self._threshold_batches(model, total, threshold, fp32norm=fp32norm))
            batches = _stage(f'threshold gather{suffix} {threshold:g}', lambda: _gather_rows(local))
            point = _stage(f'threshold grading{suffix} {threshold:g}',
                           lambda: self._threshold_metrics(batches, total, threshold) if rank == 0 else None)
            if rank == 0:
                curve.append(point)
        return curve

    def _threshold_metrics(self, batches, total, threshold):
        rows = [row for batch in batches for row in batch['rows']]
        point = self._benchmark_metrics(rows, total, None)
        del point['steps']
        caps = {batch['max_steps'] for batch in batches}
        if len(caps) != 1:
            raise ValueError('Adaptive effective max_steps differs between generation batches')
        weights = {batch['weights'] for batch in batches}
        if len(weights) != 1:
            raise ValueError('Adaptive evaluated weight set differs between generation batches')
        # Weight by examples, not ranks or batches: the final batches and some
        # distributed shards may be smaller or empty.
        total_example_forwards = sum(row['nfe'] for row in rows)
        return {'threshold': threshold, 'nfe': total_example_forwards / total, **point,
                'temperature': self.threshold_temperature, 'weights': weights.pop(),
                'total_example_forwards': total_example_forwards,
                'total_batch_forwards': sum(batch['nfe'] for batch in batches),
                'physical_row_forwards': sum(batch['physical_row_forwards'] for batch in batches),
                'generation_batches': len(batches),
                'fallback_batches': sum(batch['fallback_nfe'] for batch in batches),
                'max_steps': caps.pop()}

    def _benchmark_metrics(self, rows, total, steps):
        if len(rows) != total or sorted(row['index'] for row in rows) != list(range(total)):
            raise ValueError('GSM8K gathered answers have missing or duplicated dataset indices')
        grader = self.grader_factory()
        correct = sum(bool(grader.evaluate_samples(row['generated'], row['gold'], timeout_s=self.timeout))
                      for row in rows)
        return {'steps': steps, 'accuracy': correct / total, 'correct': correct, 'count': total,
                'answer_entropy': sum(row['entropy'] for row in rows) / total}

    def _protocol(self, model, world, total):
        grader_path = Path(__file__).resolve().parents[1] / 'sandbox_gsm8k.py'
        return {'task': 'TinyGSM adaptation; not a claim of DiDi paper reproduction',
                'sampler': str(self.config.sampling.predictor),
                'weights': getattr(model, '_didi_evaluation_weights', 'backbone'),
                'sampler_source': getattr(model, '_didi_sampler_source', 'main'),
                'sequence_length': int(model.num_tokens),
                'noise_removal': self.config.sampling.get('noise_removal', None),
                'use_float64': self.config.sampling.get('use_float64', None),
                'guidance': {name: getattr(model, name, None) for name in (
                    'guidance_scale_start', 'guidance_scale_end', 'rerank_steps_ratio', 'num_candidates')},
                'primary_steps': self.primary_steps, 'gsm8k_steps': self.budgets,
                'threshold_sweep': None if not self.thresholds else {
                    'thresholds': self.thresholds, 'temperature': self.threshold_temperature,
                    'prediction_normalizations': ['original', 'fp32']
                        if self.options.get('gsm8k_pareto_fp32', True) else ['original'],
                    'max_steps': self.threshold_cap, 'sampler': 'low_confidence_dynamic',
                    'guidance': 'none; discriminator is not called',
                    'weights': getattr(model, '_didi_evaluation_weights', 'backbone'),
                    'nfe': 'mean actual active forward calls per example, weighted by global example count',
                    'batch_cost': 'sum of actual model forward calls over all generation batches, reported separately',
                    'fixed_sampling_steps': 'ignored by adaptive sampler; primary/fixed evaluation remains unchanged'},
                'step_axis': 'denoising steps, not measured NFE; cleanup/guidance can add model calls',
                'world_size': world, 'gsm8k_batch_size_per_rank': self.batch_size,
                'gsm8k_examples': total, 'gsm8k_sharding': 'dataset index modulo world_size; no duplicates',
                'conditional_rows': 'buffered validation rows; distributed sampler padding may repeat rows',
                'entropy': 'mean per-answer token-histogram entropy; prompt excluded, generated EOS/padding included',
                'tinygsm_execution': self.execute, 'gsm8k_execution': self.benchmark,
                'execution_timeout_seconds': self.timeout,
                'grader_security': 'restricted in-process Python execution, not an OS security boundary',
                'grader_sha256': hashlib.sha256(grader_path.read_bytes()).hexdigest()
                if self.execute or self.benchmark else None}

    def _publish(self, trainer, model, conditional, conditional_metrics, curves, total):
        _, world = _distributed()
        step = _step(trainer, model)
        metrics = dict(conditional_metrics)
        for point in curves:
            suffix = f'_s{point["steps"]}'
            for name, value in (('acc', point['accuracy']), ('correct', point['correct']), ('count', point['count'])):
                metrics[f'val/gsm8k_{name}{suffix}'] = value
                if point['steps'] == self.primary_steps:
                    metrics[f'val/gsm8k_{name}'] = value
        protocol = self._protocol(model, world, total)
        for logger in _loggers(trainer):
            logger.log_metrics(metrics, step=step)
            if hasattr(logger, 'log_table'):
                logger.log_table(key='didi/tinygsm_protocol', columns=['Property', 'Value'],
                                 data=[[name, json.dumps(value)] for name, value in protocol.items()], step=step)
                if conditional and self.keep_samples:
                    logger.log_table(key='val/samples', columns=['Question', 'Generated', 'Ground truth'],
                                     data=[[row['question'], row['generated'], row['gold']]
                                           for row in conditional[:self.keep_samples]], step=step)
                if curves:
                    logger.log_table(key='pareto/gsm8k_fixed_steps',
                                     columns=['Denoising steps', 'Accuracy', 'Correct', 'Examples', 'Answer entropy'],
                                     data=[[point['steps'], point['accuracy'], point['correct'], point['count'],
                                            point['answer_entropy']] for point in sorted(curves, key=lambda p: p['steps'])],
                                     step=step)
            if len(curves) > 1 and hasattr(logger, 'log_image') and self.options.get('gsm8k_plot', True):
                logger.log_image(key=f'pareto/gsm8k_fixed_steps_{step}', image=self._plot(curves), step=step)

    @staticmethod
    def _plot(curves):
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        figure = Figure(figsize=(7, 4))
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()
        ordered = sorted(curves, key=lambda point: point['steps'])
        axis.plot([point['steps'] for point in ordered], [100 * point['accuracy'] for point in ordered], 'o-')
        axis.set(xlabel='Denoising steps (not total NFE)', ylabel='GSM8K execution accuracy (%)',
                 title='DiDi TinyGSM: fixed-budget conditional sampling')
        axis.grid(alpha=.25)
        figure.tight_layout()
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            self.batches = []
            return
        rank, world = _distributed()
        if int(getattr(trainer, 'world_size', world)) != world:
            raise RuntimeError('TinyGSM evaluation requires an initialized distributed process group on every rank')
        benchmark_now = self.benchmark and self.validations_seen % self.every == 0
        self.validations_seen += 1
        if not self.conditional and not benchmark_now:
            return
        try:
            # Lightning validate() may use inference_mode; guidance needs input
            # autograd but no persistent training parameter gradients.
            with torch.inference_mode(False), torch.no_grad(), preserve_rng_state(), _evaluation_state(pl_module):
                conditional, metrics, curves, threshold_curve, total = None, {}, [], [], None
                if self.conditional:
                    local = _stage('conditional generation', lambda: self._conditional_rows(pl_module))
                    conditional = _stage('conditional gather', lambda: _gather_rows(local))
                    metrics = _stage('conditional grading', lambda: self._conditional_metrics(conditional) if rank == 0 else {})
                if benchmark_now:
                    total = _stage('dataset loading', lambda: self._load_dataset(pl_module))
                    # Counts must agree before ranks generate different shards.
                    counts = [total]
                    if world > 1:
                        counts = [None] * world
                        torch.distributed.all_gather_object(counts, total)
                    if len(set(counts)) != 1:
                        raise RuntimeError(f'TinyGSM benchmark dataset lengths differ across ranks: {counts}')
                    for steps in self.budgets:
                        local = _stage(f'GSM8K generation s{steps}', lambda: self._benchmark_rows(pl_module, total, steps))
                        rows = _stage(f'GSM8K gather s{steps}', lambda: _gather_rows(local))
                        point = _stage(f'GSM8K grading s{steps}', lambda: self._benchmark_metrics(rows, total, steps)
                                       if rank == 0 else None)
                        if rank == 0:
                            curves.append(point)
                    fp32 = bool(self.thresholds) and self.options.get('gsm8k_pareto_fp32', True)
                    initial_rng = capture_rng_state() if fp32 else None
                    threshold_curve = self._threshold_sweep(pl_module, total, rank)
                    if fp32:
                        with replay_rng_state(initial_rng):
                            fp32_curve = self._threshold_sweep(pl_module, total, rank, fp32norm=True)
                _stage('publishing', lambda: self._publish(trainer, pl_module, conditional, metrics, curves, total)
                       if rank == 0 else None)
                if self.thresholds and benchmark_now:
                    _stage('threshold publishing', lambda: self.threshold_publisher(
                        trainer, pl_module, threshold_curve,
                        reference=self.options.get('gsm8k_pareto_reference', None),
                        plot=self.options.get('gsm8k_pareto_plot', False)) if rank == 0 else None)
                    if fp32:
                        _stage('threshold publishing_fp32norm', lambda: self.threshold_publisher(
                            trainer, pl_module, fp32_curve, suffix='_fp32norm',
                            reference=self.options.get('gsm8k_pareto_reference', None),
                            plot=self.options.get('gsm8k_pareto_plot', False)) if rank == 0 else None)
        finally:
            self.batches = []


def attach_tinygsm_evaluation(model, config):
    """Return the task callback and suppress upstream unconditional generation.

    NLL validation still runs unchanged. The public config flag controls this
    callback; it is temporarily disabled only inside the original epoch hook.
    """
    if getattr(model, '_didi_tinygsm_evaluation_attached', False):
        raise RuntimeError('TinyGSM evaluation is already attached')
    callback = DiDiTinyGSMEvaluation(config)
    original = model.on_validation_epoch_end

    @functools.wraps(original)
    def nll_only(*args, **kwargs):
        previous = config.eval.generate_samples
        try:
            config.eval.generate_samples = False
            return original(*args, **kwargs)
        finally:
            config.eval.generate_samples = previous

    model.on_validation_epoch_end = nll_only
    model._didi_tinygsm_evaluation_attached = True
    return callback
