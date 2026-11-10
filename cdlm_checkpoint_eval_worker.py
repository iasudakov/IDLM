"""Distributed, read-only evaluation of a prepared TinyGSM checkpoint plan.

Launch through torchrun. The CPU coordinator supplies a validated plan and a
resolved OmegaConf configuration; no training or trajectory collection runs here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


NORMALIZATIONS = ('bf16', 'fp32')
ANCESTRAL_STEPS = [4, 8, 16, 32, 48, 64]
CONFIDENCE_THRESHOLDS = [.325, .4, .475, .55, .625, .7, .775, .85, .925, .98]
REPORT_SCHEMA = 1
_active_publisher = None


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'File changed while hashing: {path}')
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('schema_version') != 1:
        raise ValueError('Evaluation plan must have schema_version=1.')
    root = Path(plan.get('output_dir', '')).expanduser()
    if not root.is_absolute():
        raise ValueError('Plan output_dir must be absolute.')
    root = root.resolve()
    namespace = plan.get('snapshot_namespace')
    if (not isinstance(namespace, str) or not namespace or
            Path(namespace).is_absolute() or '..' in Path(namespace).parts):
        raise ValueError('Evaluation snapshot namespace must be a nonempty relative path.')
    entries = plan.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Evaluation plan contains no checkpoints.')
    seen_dirs, seen_steps = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Checkpoint plan entries must be mappings.')
        source, family, step = [entry.get(key) for key in ('source_run', 'family', 'step')]
        if not isinstance(source, str) or not source or not isinstance(family, str) or not family:
            raise ValueError('Every entry needs source_run and family strings.')
        if '/' in source or '\\' in source or source in ('.', '..'):
            raise ValueError('source_run must be a single safe directory component.')
        if type(step) is not int or step < 0:
            raise ValueError('Checkpoint step must be a nonnegative integer.')
        checkpoint = Path(entry.get('checkpoint', '')).expanduser()
        output = Path(entry.get('output_dir', '')).expanduser()
        if not checkpoint.is_absolute() or not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint must be an existing absolute path: {checkpoint}')
        if not output.is_absolute():
            raise ValueError('Checkpoint output directories must be absolute.')
        checkpoint, output = checkpoint.resolve(), output.resolve()
        if root not in output.parents:
            raise ValueError('Checkpoint output directory must be inside plan output_dir.')
        if root == checkpoint or root in checkpoint.parents:
            raise ValueError('Input checkpoint cannot be inside the evaluation output directory.')
        if output in seen_dirs or (source, step) in seen_steps:
            raise ValueError('Duplicate output directory or source/checkpoint step in plan.')
        seen_dirs.add(output)
        seen_steps.add((source, step))
    return root


def validate_config(config):
    if config.model.get('force_fp32', False) is not False:
        raise ValueError('This comparison requires BF16 forward: model.force_fp32=False.')
    if str(config.trainer.precision) not in ('bf16', 'bf16-mixed'):
        raise ValueError('Checkpoint evaluation requires trainer.precision=bf16-mixed.')
    if config.eval.disable_ema:
        raise ValueError('Checkpoint evaluation requires EMA weights.')
    if config.algo.name != 'cdlm' or config.algo.causal_attention or config.algo.time_conditioning:
        raise ValueError('Use the common full-attention, time-unconditional CDLM inference adapter.')
    if config.data.train != 'tiny_gsm' or config.model.length != 512:
        raise ValueError('This evaluation plan requires TinyGSM and sequence length512.')
    if config.data.separator != r'\n':
        raise ValueError('Use the historical literal backslash+n TinyGSM separator.')
    if config.sampling.predictor != 'ancestral_cache':
        raise ValueError('This evaluation uses sampling.predictor=ancestral_cache.')
    if config.eval.get('gsm8k_num_examples') is not None:
        raise ValueError('Evaluate all1319 GSM8K test examples, without a sample limit.')
    if float(config.eval.gsm8k_pareto_temperature) != 0.:
        raise ValueError('Confidence token temperature must be zero.')
    if list(config.eval.gsm8k_pareto_thresholds) != CONFIDENCE_THRESHOLDS:
        raise ValueError('Use the common ten confidence thresholds for both normalizations.')
    if (list(config.eval.gsm8k_ancestral_steps) != ANCESTRAL_STEPS or
            list(config.eval.gsm8k_ancestral_seeds) != [1] or
            config.eval.gsm8k_ancestral_final_denoise is not True):
        raise ValueError('Ancestral evaluation requires steps4/8/16/32/48/64, seed1, final denoise.')
    if config.eval.get('gsm8k_max_steps') not in (None, 512):
        raise ValueError('Confidence decoding must retain its full512-step cap.')
    path = Path(str(config.eval.get('gsm8k_data_path') or '')).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError('Supply an existing absolute eval.gsm8k_data_path; no dataset download is attempted.')
    bundle = Path(str(config.get('model_weights_dir') or '')).expanduser()
    if not bundle.is_absolute() or not bundle.is_dir():
        raise FileNotFoundError('Supply an existing absolute offline model_weights_dir for the tokenizer.')


def normalize_output(model_output, xt, mask_index, neg_infinity, normalization):
    """Keep the historical BF16 norm and CDLM FP32 norm explicitly separate."""
    import torch
    if normalization not in NORMALIZATIONS:
        raise ValueError(f'Unknown logit normalization: {normalization}')
    if model_output.dtype != torch.bfloat16:
        raise ValueError(f'Expected BF16 transformer logits, received {model_output.dtype}.')
    # Explicitly control normalization even when Lightning-style outer mixed
    # precision is enabled. Neither branch may be silently autocast to the other.
    with torch.autocast(device_type=model_output.device.type, enabled=False):
        logits = model_output.float() if normalization == 'fp32' else model_output
        logits[:, :, mask_index] += neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        visible = xt.ne(mask_index)
        logits[visible] = neg_infinity
        logits[visible, xt[visible]] = 0
    return logits


def _model_type():
    from cdlm import CDLM

    class EvaluationModel(CDLM):
        normalization = 'fp32'
        _checkpoint_step = 0

        @property
        def global_step(self):
            return self._checkpoint_step

        def _process_model_output(self, model_output, xt, sigma):
            return normalize_output(model_output, xt, self.mask_index,
                                    self.neg_infinity, self.normalization)

    return EvaluationModel


def load_weights(model, checkpoint_path, expected_step):
    """Read once; do not restore optimizer, loops, teachers or tracker metadata."""
    from sdtt_checkpoint import read_checkpoint, backbone_state, apply_ema
    checkpoint = read_checkpoint(str(checkpoint_path))
    if type(checkpoint.get('global_step')) is not int or checkpoint['global_step'] != expected_step:
        raise ValueError(f'Checkpoint global_step does not match planned step{expected_step}: {checkpoint_path}')
    model.backbone.load_state_dict(backbone_state(checkpoint), strict=True)
    apply_ema(model, checkpoint)
    model.ema = None
    model._evaluation_weights = 'ema'
    model._checkpoint_step = expected_step
    model.eval()
    return {'checkpoint_format': ('ema_weights_only' if checkpoint.get('checkpoint_weights') == 'ema'
                                  else 'full_checkpoint_with_ema'),
            'checkpoint_epoch': checkpoint.get('epoch')}


def report_identity(entry, checkpoint_sha256, protocol):
    return {'schema_version': REPORT_SCHEMA, 'source_run': entry['source_run'],
            'family': entry['family'], 'step': entry['step'],
            'checkpoint_sha256': checkpoint_sha256, 'protocol_sha256': _json_hash(protocol)}


def reusable_report(path, identity):
    """Reject stale results rather than overwriting or silently skipping them."""
    path = Path(path)
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get('identity') != identity:
        raise ValueError(f'Existing evaluation has a different checkpoint or protocol: {path}. '
                         'Choose a new evaluation output directory.')
    if report.get('complete') is not True:
        return None
    curves = report.get('confidence', {})
    for normalization in NORMALIZATIONS:
        points = curves.get(normalization, [])
        if ([point.get('threshold') for point in points] != CONFIDENCE_THRESHOLDS or
                any(point.get('count') != 1319 for point in points)):
            raise ValueError(f'Complete evaluation has an incomplete confidence curve: {path}')
    ancestral = report.get('ancestral', {})
    if (ancestral.get('complete') is not True or
            [point.get('steps') for point in ancestral.get('points', [])] != ANCESTRAL_STEPS or
            any([seed.get('seed') for seed in point.get('seeds', [])] != [1] or
                any(seed.get('count') != 1319 for seed in point.get('seeds', []))
                for point in ancestral.get('points', []))):
        raise ValueError(f'Complete evaluation has an incomplete ancestral curve: {path}')
    return report


def _sync_error(error, context):
    import torch.distributed as dist
    errors = [None] * dist.get_world_size()
    local = None if error is None else f'{type(error).__name__}: {error}'
    dist.all_gather_object(errors, local)
    if any(item is not None for item in errors):
        raise RuntimeError(f'{context} failed across ranks: {errors}') from error


def _rank_zero_result(function, context):
    import torch.distributed as dist
    value, error = None, None
    if dist.get_rank() == 0:
        try:
            value = function()
        except Exception as caught:
            error = caught
    _sync_error(error, context)
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _runtime_protocol(config, tokenizer, data_sha256, world_size):
    import torch
    from omegaconf import OmegaConf
    matmul = torch.backends.cuda.matmul
    matmul_precision = (str(matmul.fp32_precision) if hasattr(matmul, 'fp32_precision')
                        else {'allow_tf32': bool(matmul.allow_tf32)})
    source_root = Path(__file__).resolve().parent
    implementation = {name: file_sha256(source_root / name) for name in (
        'cdlm_checkpoint_eval_worker.py', 'sdtt_eval.py', 'cdlm_ancestral_sweep.py',
        'models/dit.py', 'trainer_base.py', 'cdlm.py', 'sdtt_checkpoint.py',
        'sdtt_sandbox_gsm8k.py')}
    return {
        'schema_version': REPORT_SCHEMA, 'adapter': 'CDLM inference_only',
        'forward_precision': 'bf16', 'weights': 'ema',
        'model': OmegaConf.to_container(config.model, resolve=True),
        'data_sha256': data_sha256, 'examples': 1319,
        'tokenizer_name': config.data.tokenizer_name_or_path,
        'tokenizer_vocabulary_sha256': _json_hash(tokenizer.get_vocab()),
        'separator': config.data.separator,
        'batch_size_per_rank': int(config.eval.gsm8k_batch_size),
        'world_size': world_size, 'sharding': 'examples[rank::world_size]',
        'seed': int(config.seed), 'rank_seed_offset': 104729,
        'confidence': {'normalizations': list(NORMALIZATIONS),
                       'thresholds': CONFIDENCE_THRESHOLDS, 'temperature': 0.,
                       'max_steps': 512,
                       'nfe_definition': 'mean active forwards per example, including capped final removal'},
        'ancestral': {'normalization': 'fp32', 'steps': ANCESTRAL_STEPS,
                      'seeds': [1], 'temperature': 0., 'predictor': 'ancestral_cache',
                      'final_denoise': True, 'epsilon': 1e-5,
                      'nfe_definition': 'actual cached batch forwards per example including separate cleanup'},
        'grading': 'S-FLM simple_math_problem execution',
        'timeout_seconds': float(config.eval.tinygsm_exec_timeout),
        'torch_version': str(torch.__version__), 'matmul_precision': matmul_precision,
        'device': torch.cuda.get_device_name(torch.cuda.current_device()),
        'implementation_sha256': implementation,
    }


class Publisher:
    """Only rank zero creates new evaluation runs; training run IDs stay read-only."""

    def __init__(self, plan, config):
        from underdeep_logger import UnderdeepLogger
        global _active_publisher
        self.root = Path(plan['output_dir'])
        self.loggers, self.histories = {}, {}
        self.source_counts = {}
        self.source_completed = {}
        _active_publisher = self
        for entry in plan['entries']:
            source = entry['source_run']
            self.source_counts[source] = self.source_counts.get(source, 0) + 1
        for entry in plan['entries']:
            source = entry['source_run']
            if source in self.loggers:
                continue
            prefix = str(config.logger.get('name') or 'CDLM checkpoint evaluation')
            logger = UnderdeepLogger(
                experiment=config.logger.get('experiment', 'generative-models/tinygsm'),
                name=f'{prefix} | {entry["family"]} | {source[:8]}',
                save_dir=str(self.root / 'trackers' / source), resume=True, force_new_run=False)
            if logger.experiment is None:
                raise RuntimeError('Underdeep could not initialize the evaluation run. '
                                   'Check the existing Underdeep credentials/package; no checkpoint evaluation started.')
            logger.log_metrics({'eval/checkpoints_total': self.source_counts[source],
                                'eval/world_size': int(os.environ['WORLD_SIZE'])}, step=0)
            self.loggers[source] = logger
            self.source_completed[source] = set()

    def provenance(self, entry, payload):
        logger = self.loggers[entry['source_run']]
        identity, protocol = payload['identity'], payload['protocol']
        logger.log_table('eval/provenance',
                         ['source_run', 'family', 'checkpoint', 'checkpoint_sha256',
                          'step', 'weights', 'protocol_sha256', 'data_sha256',
                          'world_size', 'forward_precision', 'confidence_normalizations',
                          'ancestral_normalization', 'ancestral_seeds'],
                         [[entry['source_run'], entry['family'], entry['checkpoint'],
                           identity['checkpoint_sha256'], entry['step'], 'ema',
                           identity['protocol_sha256'], protocol['data_sha256'],
                           protocol['world_size'], 'bf16', 'bf16,fp32', 'fp32', '1']],
                         step=entry['step'])

    def confidence(self, entry, normalization, points, samples):
        from sdtt_eval import render_frontier
        source, step = entry['source_run'], entry['step']
        logger = self.loggers[source]
        point = points[-1]
        tag = f'{point["threshold"]:g}'.replace('.', '')
        suffix = f'{normalization}norm'
        logger.log_metrics({f'val/gsm8k_acc_thr{tag}_{suffix}': point['accuracy'],
                            f'val/gsm8k_nfe_thr{tag}_{suffix}': point['nfe']}, step=step)
        logger.log_metrics({f'pareto/gsm8k_step_{step}_{suffix}': point['accuracy']},
                           step=int(round(point['nfe'] * 100)))
        history = self.histories.setdefault((source, normalization), {})
        history[step] = list(points)
        logger.log_image(f'pareto/frontier_{suffix}', render_frontier(history, step), step=step)
        if samples:
            logger.log_table(f'val/samples_{suffix}',
                             ['question', 'generated', 'reference', 'grade', 'NFE', 'threshold'],
                             [[row['question'], row['generated'], row['reference'], row['grade'],
                               row['nfe'], row['threshold']] for row in samples], step=step)

    def ancestral_point(self, entry, point):
        from cdlm_ancestral_logging import point_metrics
        logger = self.loggers[entry['source_run']]
        logger.log_metrics(point_metrics(point), step=entry['step'])
        logger.log_metrics({f'ancestral/gsm8k_step_{entry["step"]}': point['accuracy_mean']},
                           step=point['steps'])

    def ancestral_result(self, entry, result):
        from cdlm_ancestral_logging import render_ancestral
        logger = self.loggers[entry['source_run']]
        rows = [[point['steps'], sample['seed'], sample['correct'], sample['count'],
                 sample['accuracy'], sample['nfe']]
                for point in result['points'] for sample in point['seeds']]
        logger.log_table('ancestral/results', ['steps', 'seed', 'correct', 'count', 'accuracy', 'NFE'],
                         rows, step=entry['step'])
        logger.log_image('ancestral/accuracy_vs_steps',
                         render_ancestral(result['points'], step=entry['step']), step=entry['step'])

    def completed(self, entry):
        source = entry['source_run']
        self.source_completed[source].add(entry['step'])
        self.loggers[source].log_metrics({'eval/checkpoint_complete': 1,
                                         'eval/checkpoints_completed': len(self.source_completed[source])},
                                        step=entry['step'])

    def republish(self, entry, payload):
        self.provenance(entry, payload)
        for normalization in NORMALIZATIONS:
            points = payload['confidence'][normalization]
            for index in range(len(points)):
                self.confidence(entry, normalization, points[:index + 1], [])
        for point in payload['ancestral']['points']:
            self.ancestral_point(entry, point)
        self.ancestral_result(entry, payload['ancestral'])
        self.completed(entry)

    def finish(self):
        for logger in self.loggers.values():
            logger.finalize('success')

    def failed(self, error):
        # UnderdeepLogger.finalize currently always calls finish() without an
        # error. Use the SDK's explicit error argument for unfinished sources.
        for source, logger in self.loggers.items():
            try:
                if len(self.source_completed[source]) == self.source_counts[source]:
                    logger.finalize('success')
                else:
                    logger.log_metrics({'eval/failed': 1}, step=0)
                    logger.experiment.finish(error=f'{type(error).__name__}: {error}', timeout=30)
                    logger._finished = True
            except Exception as caught:
                print(f'[checkpoint-eval] Could not finalize failed evaluation tracker: {caught}', flush=True)


def _publish_snapshot(plan, config, reason):
    import nirvana_snapshot
    if not config.checkpointing.get('snapshot', True):
        print('[checkpoint-eval] Snapshot publication disabled by configuration.', flush=True)
        return False
    if not nirvana_snapshot.save(plan['output_dir'], enabled=True,
                                 namespace=plan['snapshot_namespace'], reason=reason):
        raise RuntimeError('Evaluation report was written locally but snapshot publication failed. '
                           'Reconnect writable Nirvana state before restarting; completed matching reports can be reused.')
    return True


def _run(plan, config):
    import torch
    import torch.distributed as dist
    import dataloader
    import model_assets
    import sdtt_eval
    from cdlm_ancestral_sweep import sweep as ancestral_sweep

    rank, world_size = dist.get_rank(), dist.get_world_size()
    error = None
    try:
        validate_plan(plan)
        validate_config(config)
        if world_size != int(config.trainer.devices):
            raise ValueError('torchrun world size must match configured trainer.devices.')
        model_assets.configure_model_weights(config, base_dir=os.getcwd())
    except Exception as caught:
        error = caught
    _sync_error(error, 'Evaluation configuration')
    publisher = None
    error = None
    if rank == 0:
        try:
            publisher = Publisher(plan, config)
        except Exception as caught:
            error = caught
    _sync_error(error, 'Underdeep initialization')

    error = None
    tokenizer = model = examples = protocol = None
    try:
        tokenizer = dataloader.get_tokenizer(config)
        model = _model_type()(config, tokenizer, inference_only=True)
        model.to(torch.device('cuda', torch.cuda.current_device()))
        model.eval()
        data_path = Path(config.eval.gsm8k_data_path)
        examples = sdtt_eval.load_examples(data_path, tokenizer, config.data.separator,
                                          config.model.length)
        if len(examples) != 1319:
            raise ValueError(f'Expected1319 GSM8K examples, received {len(examples)}.')
        protocol = _runtime_protocol(config, tokenizer, file_sha256(data_path), world_size)
    except Exception as caught:
        error = caught
    _sync_error(error, 'Evaluation model and data initialization')
    protocols = [None] * world_size
    dist.all_gather_object(protocols, protocol)
    if any(item != protocol for item in protocols):
        raise ValueError('Ranks disagree on the evaluation protocol or hardware.')
    _rank_zero_result(lambda: write_json(Path(plan['output_dir']) / 'protocol.json', protocol),
                      'Protocol publication')

    for entry in plan['entries']:
        checkpoint_hash = _rank_zero_result(lambda: file_sha256(entry['checkpoint']),
                                            'Checkpoint hashing')
        identity = report_identity(entry, checkpoint_hash, protocol)
        result_path = Path(entry['output_dir']) / 'result.json'
        previous = _rank_zero_result(lambda: reusable_report(result_path, identity),
                                     'Evaluation resume check')
        if previous is not None:
            def reuse():
                print(f'[checkpoint-eval] Reusing complete result: {result_path}', flush=True)
                publisher.republish(entry, previous)
                _publish_snapshot(plan, config, f'evaluation reuse {entry["source_run"]} step={entry["step"]}')
            _rank_zero_result(reuse, 'Completed evaluation reuse')
            continue

        error = None
        metadata = None
        try:
            metadata = load_weights(model, entry['checkpoint'], entry['step'])
        except Exception as caught:
            error = caught
        _sync_error(error, f'Checkpoint loading at step{entry["step"]}')
        payload = {'identity': identity, 'protocol': protocol, 'complete': False,
                   'checkpoint': entry['checkpoint'], **metadata,
                   'confidence': {norm: [] for norm in NORMALIZATIONS},
                   'samples': {norm: [] for norm in NORMALIZATIONS},
                   'ancestral': {'points': [], 'complete': False}}
        if rank == 0:
            print(f'[checkpoint-eval] {entry["source_run"]} step={entry["step"]} '
                  f'SHA256={checkpoint_hash}: starting dual-normalization confidence and ancestral.', flush=True)
        _rank_zero_result(lambda: (write_json(result_path, payload), publisher.provenance(entry, payload)),
                          'Checkpoint evaluation start')

        # One threshold per call permits immediate persistent reporting while
        # retaining the exact seeded rank-strided sweep used during training.
        sample_threshold = min(CONFIDENCE_THRESHOLDS, key=lambda value: abs(value - .9))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for normalization in NORMALIZATIONS:
                model.normalization = normalization
                try:
                    for threshold in CONFIDENCE_THRESHOLDS:
                        config.eval.gsm8k_pareto_thresholds = [threshold]
                        point = sdtt_eval.sweep(model, examples)[0]
                        if rank == 0:
                            payload['confidence'][normalization].append(point)
                            if threshold == sample_threshold:
                                payload['samples'][normalization] = model._gsm8k_sample_records

                        def publish_confidence():
                            write_json(result_path, payload)
                            publisher.confidence(entry, normalization, payload['confidence'][normalization],
                                                 payload['samples'][normalization]
                                                 if threshold == sample_threshold else [])
                        _rank_zero_result(publish_confidence, 'Confidence result publication')
                finally:
                    config.eval.gsm8k_pareto_thresholds = CONFIDENCE_THRESHOLDS

            model.normalization = 'fp32'

            def completed_point(point):
                if rank == 0:
                    payload['ancestral']['points'].append(point)
                    write_json(result_path, payload)
                    publisher.ancestral_point(entry, point)

            result = ancestral_sweep(model, examples, distributed=True,
                                      on_point=completed_point, cfg=config.eval)
            if rank == 0:
                payload['ancestral'] = {**result, 'complete': True, 'normalization': 'fp32'}

        def completed_checkpoint():
            publisher.ancestral_result(entry, payload['ancestral'])
            publisher.completed(entry)
            payload['complete'] = True
            write_json(result_path, payload)
            _publish_snapshot(plan, config, f'evaluation {entry["source_run"]} step={entry["step"]}')
        _rank_zero_result(completed_checkpoint, 'Checkpoint report and snapshot publication')

    def completed_plan():
        publisher.finish()
        _publish_snapshot(plan, config, 'all checkpoint evaluations complete')
        print(f'[checkpoint-eval] All{len(plan["entries"])} checkpoints completed.', flush=True)
    _rank_zero_result(completed_plan, 'Final evaluation publication')


def run(plan, config):
    global _active_publisher
    _active_publisher = None
    try:
        _run(plan, config)
    except BaseException as error:
        if _active_publisher is not None:
            _active_publisher.failed(error)
        raise
    finally:
        _active_publisher = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError('Checkpoint evaluation requires CUDA; this command does not run CPU model evaluation.')
    if any(name not in os.environ for name in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')):
        raise RuntimeError('Launch this worker through torch.distributed.run (torchrun).')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=6))
    try:
        plan = config = None
        error = None
        try:
            plan = json.loads(Path(args.plan).read_text())
            config = OmegaConf.load(args.config)
            # The coordinator promises resolved config; reject hidden runtime
            # resolvers here before any model or collective sampling begins.
            OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        except Exception as caught:
            error = caught
        _sync_error(error, 'Evaluation plan loading')
        run(plan, config)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
