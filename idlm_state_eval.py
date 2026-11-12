"""Extend an IDLM checkpoint's FP32 confidence curve from a Nirvana state.

Uses the saved global_step and EMA, never the train-batch filename as the step.
The incoming training state is read only; results use a new tracker and namespace.
"""
import argparse
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace
import uuid

SOURCE_RUN = '1be7c4ce-1533-4a28-92a0-ba4de568d9f5'
ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path):
    import torch
    # These are the user's trusted Lightning checkpoints. mmap avoids loading
    # Adam states just to inspect the saved step or copy the model parameters.
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def find_source_directory(snapshot, source_run):
    snapshot = Path(snapshot).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f'Expected an unpacked state directory: {snapshot}')
    import torch
    matches = []
    for metadata in sorted(snapshot.rglob('underdeep_metadata.pt')):
        if snapshot not in metadata.resolve().parents:
            raise ValueError('Snapshot metadata link escapes the input directory')
        value = torch.load(metadata, map_location='cpu', weights_only=False)
        if value.get('run_uid') == source_run:
            matches.append(metadata.parent)
    if len(matches) != 1:
        raise ValueError(f'Expected one state for {source_run}; found {len(matches)}. '
                         'Connect the training run output state to the snapshot/state input.')
    return matches[0]


def inventory_checkpoints(directory):
    directory = Path(directory).resolve()
    records = []
    for path in sorted((directory / 'checkpoints').glob('*.ckpt')):
        resolved = path.resolve()
        if directory not in resolved.parents:
            raise ValueError(f'Checkpoint link escapes the source state: {path.name}')
        if not path.is_file() or path.stat().st_size == 0:
            continue
        checkpoint = load_checkpoint(path)
        step = checkpoint.get('global_step')
        if type(step) is not int or step < 0:
            raise ValueError(f'Missing saved integer global_step: {path}')
        records.append({'path': str(resolved), 'global_step': step,
                        'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                        'has_ema': bool(checkpoint.get('ema'))})
        del checkpoint
        gc.collect()
    return records


def select_checkpoint(records, step):
    matches = [row for row in records if row['global_step'] == step]
    if len(matches) != 1:
        available = sorted({row['global_step'] for row in records})
        raise ValueError(f'Expected exactly one checkpoint at global_step={step}; '
                         f'found {len(matches)}. Available saved steps: {available}. '
                         'An evaluation metric does not imply that its weights were saved.')
    if not matches[0]['has_ema']:
        raise ValueError('Selected checkpoint has no EMA; refusing raw-weight substitution')
    return matches[0]


def reference_points(document, source_run, step):
    if document.get('source_run_uid') != source_run:
        raise ValueError('Reference curve belongs to another source run')
    if document.get('normalization') != 'fp32':
        raise ValueError('Reference must use FP32 prediction normalization')
    return document.get('curves', {}).get(str(step), [])


def merge_points(reference, measured):
    result = {float(row['threshold']): dict(row, origin='source_training_eval')
              for row in reference}
    for row in measured:
        result[float(row['threshold'])] = dict(row, origin='state_reevaluation')
    return sorted(result.values(), key=lambda row: row['nfe'])


def prepare(args):
    from omegaconf import OmegaConf
    import model_assets
    import utils
    utils.register_omegaconf_resolvers()
    if args.snapshot_root:
        snapshot = Path(args.snapshot_root).expanduser().resolve()
    else:
        from nirvana_dl import snapshot as sdk_snapshot
        value = sdk_snapshot.get_snapshot_path()
        if not value:
            raise ValueError('No connected Nirvana state; use --snapshot-root for a local state')
        snapshot = Path(value).resolve()
    source = find_source_directory(snapshot, args.source_run)
    records = inventory_checkpoints(source)
    print('Saved checkpoints:', [(Path(r['path']).name, r['global_step']) for r in records], flush=True)
    selected = select_checkpoint(records, args.step)
    cfg_path = source / '.hydra/config.yaml'
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Original training configuration missing: {cfg_path}')
    cfg = OmegaConf.load(cfg_path)
    if (cfg.algo.name != 'mdlm' or cfg.model.length != 512 or
            cfg.data.separator != r'\n' or cfg.eval.disable_ema or
            cfg.adversarial_distill.mmd_reward_type != 'idlm' or
            not cfg.adversarial_distill.is_distill):
        raise ValueError('Expected the TinyGSM IDLM-REINFORCE configuration, length512 and saved EMA')
    if args.devices <= 0 or any(not math.isfinite(t) or not 0 < t < 1 for t in args.thresholds):
        raise ValueError('Use positive devices and finite thresholds strictly between 0 and 1')
    if len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError('Thresholds must be distinct')
    output = Path(args.output_dir or ROOT / 'idlm_state_eval' / uuid.uuid4().hex).resolve()
    if output == snapshot or snapshot in output.parents or output in snapshot.parents:
        raise ValueError('Output and input state directories must not overlap')
    if (output / 'results.json').exists():
        raise FileExistsError('Results already exist; select a new --output-dir')
    input_root = Path(os.environ.get('INPUT_PATH', ROOT))
    bundle = Path(args.model_weights_dir or input_root / 'tinygsm_mmd_offline_bundle').resolve()
    data = Path(args.data_path or input_root / 'nirvana_checkpoints/data/gsm8k_test.json').resolve()
    if not data.is_file() or len(json.loads(data.read_text())) != 1319:
        raise ValueError(f'Expected the complete local 1319-example GSM8K JSON: {data}')
    cfg.model_weights_dir = str(bundle)
    for name in [cfg.data.tokenizer_name_or_path, cfg.eval.gen_ppl_eval_model_name_or_path]:
        model_assets.resolve_pretrained_path(name, str(bundle), kind='tokenizer')
    cfg.training.finetune_path = None  # Load only the selected full checkpoint.
    cfg.eval.checkpoint_path = selected['path']
    cfg.eval.gsm8k_data_path = str(data)
    cfg.eval.gsm8k_pareto_thresholds = sorted(args.thresholds)
    cfg.eval.gsm8k_pareto_fp32 = True
    cfg.eval.gsm8k_pareto_temperature = 0.0
    cfg.eval.gsm8k_num_examples = None
    cfg.data.cache_dir = str(output / 'cache')
    cfg.trainer.devices = args.devices
    cfg.checkpointing.save_dir = str(output)
    cfg.checkpointing.resume_from_ckpt = False
    OmegaConf.resolve(cfg)
    reference = []
    reference_path = Path(args.reference_json) if args.reference_json else ROOT / 'eval_references/idlm_reinforce_1be7c4ce_fp32_reference.json'
    if args.reference_json and not reference_path.is_file():
        raise FileNotFoundError(f'Reference file is missing: {reference_path}')
    if reference_path.is_file():
        reference = reference_points(json.loads(reference_path.read_text()), args.source_run, args.step)
    if args.devices != 2:
        print('WARNING: source used two GPUs; changing sharding can affect numerical results.', flush=True)
    plan = {'source_run_uid': args.source_run, 'checkpoint': selected,
            'inventory': records, 'step': args.step, 'devices': args.devices,
            'thresholds': sorted(args.thresholds), 'output_dir': str(output),
            'snapshot_root': str(snapshot), 'source_directory': str(source),
            'reference_points': reference, 'no_underdeep': args.no_underdeep,
            'no_snapshot': args.no_snapshot, 'name': args.name or f'IDLM-REINFORCE step{args.step} FP32 confidence extension',
            'data_sha256': sha256(data)}
    if not args.plan_only:
        plan['checkpoint_sha256'] = sha256(selected['path'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'plan.json', plan)
    OmegaConf.save(cfg, output / 'evaluation_config.yaml')
    print(f'Exact selected optimizer step: {args.step}; file: {selected["path"]}', flush=True)
    return plan


def evaluate_worker(plan_path):
    import datetime
    import lightning as L
    from omegaconf import OmegaConf
    import torch
    import algo
    import dataloader
    from eval_precision import fp32_prediction_normalization
    import utils
    utils.register_omegaconf_resolvers()
    plan = json.loads(Path(plan_path).read_text())
    output = Path(plan['output_dir'])
    cfg = OmegaConf.load(output / 'evaluation_config.yaml')
    rank = int(os.environ['RANK'])
    world = plan['devices']
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', init_method=os.environ['IDLM_EVAL_RENDEZVOUS'],
                                        rank=rank, world_size=world,
                                        timeout=datetime.timedelta(minutes=60))
    L.seed_everything(int(cfg.seed), workers=False)
    path = Path(plan['checkpoint']['path'])
    stat = path.stat()
    if stat.st_size != plan['checkpoint']['size'] or stat.st_mtime_ns != plan['checkpoint']['mtime_ns']:
        raise ValueError('Checkpoint changed after preflight')
    checkpoint = load_checkpoint(path)
    if checkpoint.get('global_step') != plan['step']:
        raise ValueError('Saved checkpoint step changed after preflight')
    tokenizer = dataloader.get_tokenizer(cfg)
    model = algo.MDLM(config=cfg, tokenizer=tokenizer)
    # Lightning's native loading order, without materializing Adam storage.
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(torch.device('cuda', rank)).eval()
    parameters = list(model._get_parameters())
    if model.ema is None or len(parameters) != len(model.ema.shadow_params):
        raise ValueError('Saved EMA does not match the model parameters')
    if any(p.shape != s.shape for p, s in zip(parameters, model.ema.shadow_params)):
        raise ValueError('Saved EMA tensor shapes do not match')
    model._eval_mode()
    if any(not torch.equal(p.detach().cpu(), s.detach().cpu()) for p, s in zip(parameters, model.ema.shadow_params)):
        raise ValueError('Model parameters differ from saved EMA after evaluation setup')
    del checkpoint
    gc.collect()
    logger = None
    if rank == 0 and not plan['no_underdeep']:
        from underdeep_logger import UnderdeepLogger
        logger = UnderdeepLogger(experiment='generative-models/tinygsm', name=plan['name'],
                                 save_dir=str(output), resume=False, force_new_run=True)
        if logger.experiment is None:
            raise RuntimeError('Could not create the separate evaluation Underdeep run')
        logger.log_hyperparams({'config': OmegaConf.to_container(cfg, resolve=True)})
        logger.log_table('eval/source', ['source_run', 'optimizer_step', 'checkpoint_sha256'],
                         [[plan['source_run_uid'], plan['step'], plan['checkpoint_sha256']]], step=plan['step'])
    model._trainer = SimpleNamespace(global_rank=rank, world_size=world,
                                     global_step=plan['step'], logger=logger)
    dataset = model._gsm8k_benchmark_dataset()
    if len(dataset) != 1319:
        raise ValueError('Benchmark must contain all 1319 questions')
    indices = list(range(rank, len(dataset), world))
    order = [i for r in range(world) for i in range(r, len(dataset), world)]
    report = {'source_run_uid': plan['source_run_uid'], 'checkpoint_step': plan['step'],
              'checkpoint_sha256': plan['checkpoint_sha256'], 'data_sha256': plan['data_sha256'],
              'normalization': 'fp32', 'backbone_precision': 'bf16', 'weights': 'saved EMA',
              'temperature': 0.0, 'seed': int(cfg.seed), 'devices': world,
              'batch_size_per_rank': int(cfg.eval.gsm8k_batch_size), 'num_examples': 1319,
              'complete': False, 'measured_points': [], 'reference_points': plan['reference_points'],
              'underdeep_run_uid': None if logger is None else logger.version}
    native_generate = model._gsm8k_generate
    native_grade = model._gsm8k_grade
    latest = {}

    def generate(self, *a, **kw):
        rows, nfes, per_seq = native_generate(*a, **kw)
        if rank == 0:
            if len(rows) != 1319 or len(per_seq) != 1319:
                raise ValueError('Incomplete gathered benchmark')
            latest['nfe'] = sum(per_seq) / len(per_seq)
            records = [dict(index=i, response=row[0], ground_truth=row[1], nfe=n)
                       for i, row, n in zip(order, rows, per_seq)]
            with gzip.open(output / f'predictions_thr{self.dynamic_threshold:g}.json.gz', 'wt') as stream:
                json.dump(records, stream)
        return rows, nfes, per_seq

    def grade(self, rows):
        correct, accuracy = native_grade(rows)
        report['measured_points'].append({'threshold': float(self.dynamic_threshold),
                                         'nfe': latest['nfe'], 'accuracy': accuracy,
                                         'correct': correct, 'total': len(rows)})
        write_json(output / 'results.json', report)
        return correct, accuracy

    def publish(self, curve, suffix=''):
        if suffix != '_fp32norm':
            raise ValueError('Only FP32 normalization is enabled')
        if len(curve) != len(plan['thresholds']):
            raise ValueError('Incomplete threshold sweep')
        report['curve'] = merge_points(plan['reference_points'], report['measured_points'])
        overlap = []
        for old in plan['reference_points']:
            for new in report['measured_points']:
                if old['threshold'] == new['threshold']:
                    overlap.append({'threshold': old['threshold'],
                                    'accuracy_delta_pp': 100*(new['accuracy']-old['accuracy']),
                                    'nfe_delta': new['nfe']-old['nfe']})
        report['overlap_check'] = overlap
        if overlap:
            print('Reference overlap:', json.dumps(overlap), flush=True)
        report['complete'] = True
        write_json(output / 'results.json', report)
        write_json(output / 'pareto_history_fp32norm.json',
                   {'run_uid': report['underdeep_run_uid'], 'curves': {str(plan['step']): report['curve']}})
        import pareto_plot
        from PIL import Image
        plot = pareto_plot.render({plan['step']: report['curve']})
        Image.fromarray(plot).save(output / 'confidence_fp32.png')
        if logger:
            for point in report['curve']:
                tag = f"{point['threshold']:g}".replace('.', '')
                logger.log_metrics({f'val/gsm8k_acc_thr{tag}_fp32norm': point['accuracy'],
                                    f'val/gsm8k_nfe_thr{tag}_fp32norm': point['nfe']}, step=plan['step'])
                logger.log_metrics({f'pareto/gsm8k_step_{plan["step"]}_fp32norm': point['accuracy']},
                                   step=round(point['nfe']*100))
            logger.log_image('pareto/frontier_fp32norm', str(output / 'confidence_fp32.png'), step=plan['step'])
            logger.finalize('success')

    model._gsm8k_generate = MethodType(generate, model)
    model._gsm8k_grade = MethodType(grade, model)
    model._publish_pareto = MethodType(publish, model)
    try:
        with torch.inference_mode(), fp32_prediction_normalization(model):
            model._run_gsm8k_pareto(dataset, indices, int(cfg.eval.gsm8k_batch_size), suffix='_fp32norm')
        torch.distributed.barrier()
    except BaseException:
        if logger:
            logger.finalize('failed')
        raise
    finally:
        torch.distributed.destroy_process_group()


def launch(plan):
    output = Path(plan['output_dir'])
    rendezvous = output / ('rendezvous_' + uuid.uuid4().hex)
    children = []
    try:
        for rank in range(plan['devices']):
            env = os.environ.copy()
            env.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(plan['devices']),
                       IDLM_EVAL_RENDEZVOUS='file://' + str(rendezvous))
            children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                              '--worker-plan', str(output / 'plan.json')], env=env))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('An evaluation worker failed; see its traceback above')
            time.sleep(0.5)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError('An evaluation worker failed')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
        rendezvous.unlink(missing_ok=True)
    if not json.loads((output / 'results.json').read_text())['complete']:
        raise RuntimeError('Evaluation report is incomplete')
    if not plan['no_snapshot']:
        from nirvana_dl import snapshot
        target = Path(snapshot.get_snapshot_path()).resolve() / 'idlm_confidence_eval' / output.name
        if target.exists():
            raise FileExistsError(f'Evaluation snapshot namespace already exists: {target}')
        shutil.copytree(output, target)
        snapshot.dump_snapshot(str(Path(snapshot.get_snapshot_path()).resolve()))
    print(f'Complete. Results: {output / "results.json"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-plan', help=argparse.SUPPRESS)
    parser.add_argument('--step', type=int)
    parser.add_argument('--source-run', default=SOURCE_RUN)
    parser.add_argument('--snapshot-root')
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.99, 0.995])
    parser.add_argument('--model-weights-dir')
    parser.add_argument('--data-path')
    parser.add_argument('--reference-json')
    parser.add_argument('--output-dir')
    parser.add_argument('--name')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--no-underdeep', action='store_true')
    parser.add_argument('--no-snapshot', action='store_true')
    args = parser.parse_args(argv)
    if args.worker_plan:
        evaluate_worker(args.worker_plan)
        return
    if args.step is None or args.step < 0:
        parser.error('--step must explicitly specify a nonnegative saved optimizer step')
    plan = prepare(args)
    if not args.plan_only:
        launch(plan)


if __name__ == '__main__':
    main()
