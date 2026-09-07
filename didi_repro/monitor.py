"""Read-only diagnostics around the unmodified upstream validation sampler.

The callback captures samples the upstream code already generated; it does not
change its sampler, EMA choice, cadence, model modes, or training randomness.
The root evaluation implementation is imported by filename, never as
``metrics`` (that name belongs to the isolated upstream package).
"""

from contextlib import contextmanager
import functools
import importlib.util
import json
from pathlib import Path
import random
import sys

import lightning as L
import numpy as np
from omegaconf import OmegaConf
import torch


@contextmanager
def preserve_rng_state():
    """Diagnostics must not advance Python, NumPy, CPU, or active CUDA RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _resolved_config(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


def config_rows(config, upstream_revision=None):
    """Lossless JSON-valued rows, including strings, nulls, and whole lists."""
    rows = []

    def walk(value, prefix):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                walk(child, f'{prefix}.{key}' if prefix else str(key))
        else:
            rows.append([prefix, json.dumps(value, ensure_ascii=False)])

    walk(_resolved_config(config), '')
    if upstream_revision is not None:
        rows.append(['didi_upstream_revision', json.dumps(upstream_revision)])
    return rows


def _root_metrics_class():
    name = '_idlm_didi_monitor_metrics'
    if name not in sys.modules:
        filename = Path(__file__).resolve().parents[1] / 'metrics.py'
        spec = importlib.util.spec_from_file_location(name, filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name].Metrics


def _entropy(tokens):
    values = []
    for row in tokens:
        _, counts = torch.unique(row, return_counts=True, sorted=False)
        values.append(torch.special.entr(counts.float() / counts.sum()).sum())
    return float(torch.stack(values).mean())


def _gradient_moment(evaluator, samples, references, max_length, batch_size, device):
    """Same paired GM estimator as OWT, retaining its auxiliary PPL metrics."""
    from gradient_moment import evaluate, make_generative_llm_metrics
    from gradient_moment.huggingface import text_causal_lm_forward

    pair_rows = 2 * batch_size
    count = min(len(samples), len(references)) // pair_rows * pair_rows
    if count == 0:
        return None
    reference_model = evaluator._ensure_eval_model(device)

    def tokenize(texts):
        encoded = evaluator.tokenizer(
            texts, return_tensors='pt', return_token_type_ids=False,
            truncation=True, padding=True, max_length=max_length)
        return {'input_ids': encoded['input_ids'].to(device).long(),
                'attention_mask': encoded['attention_mask'].to(device).float()}

    pairs = [(tokenize(samples[i:i + pair_rows]),
              tokenize(references[i:i + pair_rows]))
             for i in range(0, count, pair_rows)]
    forward = text_causal_lm_forward(reference_model, evaluator.tokenizer.eos_token_id)
    metric = make_generative_llm_metrics(forward, reference_model)
    with torch.enable_grad():
        result = evaluate([metric], pairs)[metric.name]
    names = {
        'centered_gm': 'val/gm',
        'uncentered_gm_samples': 'val/gm_uncentered_samples',
        'uncentered_gm_data': 'val/gm_uncentered_data',
        'generative_ppl_samples': 'val/gen_ppl_gm',
        'generative_ppl_data': 'val/data_ppl_gm',
        'generative_entropy_samples': 'val/predictive_entropy_samples',
        'generative_entropy_data': 'val/predictive_entropy_data',
    }
    values = {name: float(result[key]['mean']) for key, name in names.items()}
    values['val/gm_pairs'] = int(result['centered_gm']['n'])
    return values


class DiDiMonitor(L.Callback):
    """Log full provenance and evaluate captured upstream samples on rank zero.

    ``attach_monitor`` installs the capture wrapper. Add the returned object to
    the Trainer's callbacks. No extra generation is performed, including when
    there are too few upstream samples to estimate GM.
    """

    def __init__(self, config, upstream_revision=None, metrics_factory=None):
        self.config = config
        self.options = config.get('didi_monitor', {})
        self.upstream_revision = upstream_revision
        self.metrics_factory = metrics_factory
        self.evaluator = None
        self.capturing = False
        self.samples = []
        self.references = []
        self._provenance_logged = False
        self.gm_batch_size = int(self.options.get('gm_batch_size', 4))
        self.max_references = int(self.options.get('max_reference_samples', 64))
        if self.gm_batch_size < 1 or self.max_references < 1:
            raise ValueError('didi_monitor GM batch size and reference cap must be positive')
        if int(config.eval.perplexity_batch_size) < 1:
            raise ValueError('eval.perplexity_batch_size must be positive')

    def _provenance(self, trainer, model=None):
        if self._provenance_logged or not trainer.is_global_zero:
            return
        directory = Path(str(self.config.checkpointing.save_dir))
        directory.mkdir(parents=True, exist_ok=True)
        extra = {
            'teacher_checkpoint': _resolved_config(
                getattr(model, '_didi_teacher_metadata', None)),
            'student_initialization': _resolved_config(
                getattr(model, '_didi_student_metadata', None)),
            'effective_hyperparameters': _resolved_config(
                getattr(model, '_didi_effective_hyperparameters', None)),
        }
        for key, filename in (('environment', 'didi_environment.json'),
                              ('entrypoint_provenance', 'didi_provenance.json')):
            path = directory / filename
            if path.is_file():
                extra[key] = json.loads(path.read_text())
        payload = {'upstream_revision': self.upstream_revision,
                   'config': _resolved_config(self.config), **extra}
        (directory / 'didi_monitor_config.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        logger = trainer.logger
        if logger and hasattr(logger, 'log_table'):
            rows = config_rows(self.config, self.upstream_revision)
            rows.extend(config_rows({'didi_provenance': extra}))
            logger.log_table(key='didi/config', columns=['Parameter', 'JSON value'],
                             data=rows, step=0)
        self._provenance_logged = True

    def on_fit_start(self, trainer, pl_module):
        with preserve_rng_state():
            self._provenance(trainer, pl_module)

    def on_validation_start(self, trainer, pl_module):
        self.samples = []
        self.references = []
        self.capturing = not trainer.sanity_checking

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        if not self.capturing or not self.options.get('gm', True):
            return
        room = self.max_references - sum(len(x) for x in self.references)
        if room > 0:
            self.references.append(batch['input_ids'][:room].detach().cpu().clone())

    def _get_evaluator(self, model):
        if self.evaluator is None:
            factory = self.metrics_factory or _root_metrics_class()
            self.evaluator = factory(
                gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
                eval_ppl_batch_size=int(self.config.eval.perplexity_batch_size))
            # Only rank zero evaluates the gathered samples. Torchmetrics must
            # not initiate distributed collectives from these local metrics.
            for value in vars(self.evaluator).values():
                if isinstance(value, torch.nn.Module):
                    for module in value.modules():
                        if hasattr(module, 'sync_on_compute'):
                            module.sync_on_compute = False
                            # TorchMetrics caches this flag at construction;
                            # update()/compute() need the cached flag disabled
                            # too when only rank zero owns the evaluator.
                            module._to_sync = False
        self.evaluator.to(model.device)
        cached_model = getattr(self.evaluator, '_eval_model', None)
        if cached_model is not None:
            cached_model.to(model.device)
        return self.evaluator

    def _evaluate(self, model, samples, references):
        values = {'val/sample_entropy': _entropy(samples),
                  'didi_monitor/sample_count': len(samples),
                  'didi_monitor/reference_count': len(references)}
        texts = model.tokenizer.batch_decode(samples)
        reference_texts = model.tokenizer.batch_decode(references) if len(references) else []
        if not (self.options.get('gen_ppl', True) or self.options.get('gm', True)):
            return values, texts
        evaluator = self._get_evaluator(model)
        if self.options.get('gen_ppl', True):
            evaluator.gen_ppl.reset()
            chunk = int(self.config.eval.perplexity_batch_size)
            with torch.no_grad():
                # Root evaluator uses full internal batches: explicit chunks
                # include the tail instead of silently discarding sample rows.
                for offset in range(0, len(texts), chunk):
                    evaluator.record_generative_perplexity(
                        texts[offset:offset + chunk], max_length=model.num_tokens,
                        device=model.device)
            values['val/gen_ppl'] = float(evaluator.gen_ppl.compute())
        if self.options.get('gm', True):
            gm = _gradient_moment(evaluator, texts, reference_texts,
                                  model.num_tokens, self.gm_batch_size, model.device)
            if gm is None:
                print(f'[didi-monitor] GM skipped: need {2 * self.gm_batch_size} '
                      'captured samples and reference rows; no extra samples generated.', flush=True)
            else:
                values.update(gm)
        return values, texts

    def on_validation_end(self, trainer, pl_module):
        self.capturing = False
        if trainer.sanity_checking:
            return
        # Gather the original samples (not new draws), so diagnostics describe
        # the same number of generated rows as the distributed upstream run.
        with preserve_rng_state(), torch.inference_mode(False):
            payload = (self.samples, self.references)
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            if distributed:
                gathered = [None] * torch.distributed.get_world_size() if trainer.is_global_zero else None
                torch.distributed.gather_object(payload, gathered, dst=0)
            else:
                gathered = [payload]
            if trainer.is_global_zero:
                try:
                    self._provenance(trainer, pl_module)
                    sample_batches = [x for samples, _ in gathered for x in samples]
                    reference_batches = [x for _, references in gathered for x in references]
                    if sample_batches:
                        samples = torch.cat(sample_batches)
                        references = (torch.cat(reference_batches)[:self.max_references]
                                      if reference_batches else samples[:0])
                        values, texts = self._evaluate(pl_module, samples, references)
                        values['didi_monitor/failed'] = 0
                        if trainer.logger:
                            trainer.logger.log_metrics(values, step=trainer.global_step)
                            if hasattr(trainer.logger, 'log_table'):
                                count = int(self.config.sampling.num_sample_log)
                                trainer.logger.log_table(
                                    key='val/samples', columns=['Generated Samples'],
                                    data=[[text] for text in texts[:count]], step=trainer.global_step)
                                trainer.logger.log_table(
                                    key='didi/diagnostics', columns=['Property', 'Value'],
                                    data=[['sampler', str(pl_module.sampler)],
                                          ['weights', 'backbone (upstream generate_samples)'],
                                          ['sampling_steps', str(self.config.sampling.steps)],
                                          ['sample_source', 'captured upstream validation; no extra draws']],
                                    step=trainer.global_step)
                    elif trainer.logger:
                        trainer.logger.log_metrics(
                            {'didi_monitor/sample_count': 0, 'didi_monitor/skipped': 1},
                            step=trainer.global_step)
                except Exception as error:
                    print(f'[didi-monitor] diagnostics failed; training continues: {error}', flush=True)
                    if trainer.logger:
                        trainer.logger.log_metrics({'didi_monitor/failed': 1}, step=trainer.global_step)
                finally:
                    reference_model = getattr(self.evaluator, '_eval_model', None)
                    if reference_model is not None:
                        # A reference LM must not consume training GPU capacity.
                        try:
                            reference_model.to('cpu')
                        except Exception as error:
                            print(f'[didi-monitor] evaluator offload failed: {error}', flush=True)
            if distributed:
                torch.distributed.barrier()
        self.samples = []
        self.references = []


def attach_monitor(model, config, upstream_revision=None):
    """Return a callback after attaching capture; call after runtime adapters."""
    if not config.get('didi_monitor', {}).get('enabled', True):
        return None
    monitor = DiDiMonitor(config, upstream_revision=upstream_revision)
    original = model.generate_samples

    @functools.wraps(original)
    def capture(*args, **kwargs):
        samples = original(*args, **kwargs)
        if monitor.capturing:
            monitor.samples.append(samples.detach().cpu().clone())
        return samples

    model.generate_samples = capture
    return monitor
