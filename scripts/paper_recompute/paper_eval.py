"""Evaluate native MDLM students; confidence FP32 and ancestral T=0 pass@k.

Run from the repository root. Only trusted own PyTorch checkpoints are accepted.
This script never trains or silently substitutes a different checkpoint step.
"""
import argparse
import datetime
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
import uuid
from types import SimpleNamespace

THRESHOLDS = [.325, .35, .375, .4, .425, .45, .475, .5, .525, .55,
              .575, .6, .625, .65, .675, .7, .725, .75, .775, .8,
              .825, .85, .875, .9, .925, .95, .98, .99, .995]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('Require 0 <= c <= n and 1 <= k <= n')
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n-c-i)/(n-i) for i in range(k))


def existing(paths, directory=False):
    for p in map(Path, paths):
        if p.is_dir() if directory else p.is_file():
            return p.resolve()
    raise FileNotFoundError('Missing input; checked: ' + ', '.join(map(str, paths)))


def source_directory(snapshot, uid):
    import torch
    hits = []
    for p in snapshot.rglob('underdeep_metadata.pt'):
        if snapshot not in p.resolve().parents:
            continue
        meta = torch.load(p, map_location='cpu', weights_only=False)
        if meta.get('run_uid') == uid:
            hits.append(p.parent)
    if len(hits) != 1:
        raise ValueError(f'Expected one state directory for {uid}; found {len(hits)}')
    return hits[0]


def choose_checkpoint(source, step):
    inventory = []
    for p in sorted((source / 'checkpoints').glob('*.ckpt')):
        ckpt = load_checkpoint(p)
        inventory.append({'path': str(p), 'global_step': ckpt.get('global_step'),
                          'has_ema': 'ema' in ckpt})
        del ckpt
    hits = [r for r in inventory if r['global_step'] == step]
    if len(hits) > 1:
        hashes = {sha256(r['path']) for r in hits}
        if len(hashes) != 1:
            raise ValueError('Several distinct checkpoints at this step; use --checkpoint explicitly')
    if not hits:
        raise ValueError(f'No saved global_step={step}; available: ' +
                         str(sorted({r['global_step'] for r in inventory if isinstance(r['global_step'], int)})))
    return Path(hits[0]['path']), inventory


def prepare(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import utils
    utils.register_omegaconf_resolvers()
    repo = Path.cwd().resolve()
    snapshot = None
    inventory = []
    if args.checkpoint:
        ckpt = Path(args.checkpoint).resolve()
        config_file = Path(args.config).resolve() if args.config else None
    else:
        if not args.source_run:
            raise ValueError('Supply --checkpoint or --source-run with its connected state')
        if args.snapshot_root:
            snapshot = Path(args.snapshot_root).resolve()
        else:
            from nirvana_dl.snapshot import get_snapshot_path
            snapshot = Path(get_snapshot_path()).resolve()
        source = source_directory(snapshot, args.source_run)
        ckpt, inventory = choose_checkpoint(source, args.step)
        config_file = source / '.hydra/config.yaml'
    if config_file:
        cfg = OmegaConf.load(config_file)
    else:
        with initialize_config_dir(config_dir=str(repo / 'configs'), version_base=None):
            cfg = compose(config_name='config', overrides=[
                'data=tiny-gsm', 'algo=mdlm', 'model=small', 'model.length=512',
                'sampling.predictor=ancestral_cache', 'loader.num_workers=0'])
    saved = load_checkpoint(ckpt)
    actual_step = saved.get('global_step')
    # A base teacher's pretraining step is not distillation iteration zero.
    if not args.base_teacher and actual_step != args.step:
        raise ValueError(f'Requested step {args.step}, checkpoint contains {actual_step}')
    if args.base_teacher and args.step != 0:
        raise ValueError('--base-teacher must be labelled --step 0')
    if args.base_teacher and any(k.startswith(('teacher.', 'fake_model.')) for k in saved['state_dict']):
        raise ValueError('--base-teacher requires the original mdlm_notok.ckpt, not a trained student')
    if args.weights == 'ema' and 'ema' not in saved:
        raise ValueError('EMA explicitly requested but missing from checkpoint')
    del saved
    inputs = Path(os.environ.get('INPUT_PATH', repo))
    bundle = existing([args.model_weights_dir] if args.model_weights_dir else [
        inputs / 'tinygsm_mmd_offline_bundle', repo / 'cache_dir/tinygsm_mmd_offline_bundle'], True)
    data = existing([args.data_path] if args.data_path else [
        inputs / 'nirvana_checkpoints/data/gsm8k_test.json', repo / 'data/gsm8k_test.json',
        bundle / 'data/gsm8k_test.json'])
    if len(json.loads(data.read_text())) != 1319:
        raise ValueError('Require the complete 1319-example GSM8K test set')
    output = Path(args.output_dir or repo / 'outputs/paper_eval' / uuid.uuid4().hex).resolve()
    if snapshot and (output == snapshot or snapshot in output.parents or output in snapshot.parents):
        raise ValueError('Output must be separate from the input state')
    output.mkdir(parents=True, exist_ok=False)
    source_cfg = OmegaConf.to_container(cfg, resolve=True)
    write_json(output / 'source_config.json', source_cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.model_weights_dir = str(bundle)
    cfg.training.finetune_path = ''
    cfg.training.continue_teacher = False
    cfg.training.ema = 0  # explicit weight copy below, no automatic EMA swapping
    cfg.adversarial_distill.is_distill = False
    cfg.adversarial_distill.mmd_implementation = 'legacy'  # inference-only; no objective is constructed
    cfg.eval.gsm8k_data_path = str(data)
    cfg.eval.checkpoint_path = str(ckpt)
    cfg.eval.disable_ema = True
    cfg.eval.compute_generative_perplexity = False
    cfg.data.cache_dir = str(output / 'tokenized_cache')
    cfg.sampling.predictor = 'ancestral_cache'
    cfg.sampling.noise_removal = 'ancestral'
    cfg.sampling.temperature = 0.0
    cfg.sampling.step_convention = 'sflm'
    cfg.trainer.devices = args.devices
    cfg.checkpointing.save_dir = str(output)
    cfg.checkpointing.resume_from_ckpt = False
    if cfg.algo.name != 'mdlm' or cfg.model.length != 512 or cfg.data.separator != r'\n':
        raise ValueError('Require native MDLM length512 and literal backslash-n separator')
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, output / 'evaluation_config.yaml')
    plan = dict(vars(args), checkpoint=str(ckpt), checkpoint_saved_global_step=actual_step,
                checkpoint_sha256=None if args.plan_only else sha256(ckpt),
                data_sha256=sha256(data), output_dir=str(output), inventory=inventory,
                batch_size=args.batch_size or int(source_cfg['eval'].get('gsm8k_batch_size') or 16),
                repo=str(repo), source_run=args.source_run, forward_precision='bf16',
                normalization_precision='fp32', separator_hex=cfg.data.separator.encode().hex())
    plan['git_commit'] = subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True).stdout.strip()
    write_json(output / 'plan.json', plan)
    print('PLAN:', str(output / 'plan.json'), flush=True)
    return plan


def load_student(cfg, checkpoint_path, weights):
    import torch
    import algo
    import dataloader
    tokenizer = dataloader.get_tokenizer(cfg)
    model = algo.MDLM(cfg, tokenizer)
    saved = load_checkpoint(checkpoint_path)
    target = model.state_dict()
    source = saved['state_dict']
    missing = set(target) - set(source)
    extras = set(source) - set(target)
    if missing or any(not k.startswith(('teacher.', 'fake_model.')) for k in extras):
        raise ValueError(f'Student state mismatch: missing={sorted(missing)}, unexpected={sorted(extras)[:8]}')
    model.load_state_dict({k: source[k] for k in target}, strict=True)
    if weights == 'ema':
        parameters = list(model._get_parameters())
        shadows = saved['ema']['shadow_params']
        if len(parameters) != len(shadows) or any(p.shape != s.shape for p,s in zip(parameters,shadows)):
            raise ValueError('EMA parameter count/order shapes do not match native student')
        with torch.no_grad():
            for p,s in zip(parameters,shadows):
                p.copy_(s)  # include frozen parameters
        if any(not torch.equal(p.detach(),s) for p,s in zip(parameters,shadows)):
            raise ValueError('EMA copy verification failed')
    model.eval()
    del saved
    gc.collect()
    return model


def worker(plan_file):
    import torch
    import lightning as L
    from omegaconf import OmegaConf
    from eval_precision import fp32_prediction_normalization
    import sandbox_gsm8k
    import utils
    utils.register_omegaconf_resolvers()
    p = json.loads(Path(plan_file).read_text())
    out = Path(p['output_dir'])
    rank, world = int(os.environ['RANK']), p['devices']
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', timeout=datetime.timedelta(hours=2))
    cfg = OmegaConf.load(out / 'evaluation_config.yaml')
    L.seed_everything(p['seed'], workers=False)
    model = load_student(cfg, p['checkpoint'], p['weights']).to(torch.device('cuda', rank))
    logger = None
    if rank == 0 and not p['no_underdeep']:
        from underdeep_logger import UnderdeepLogger
        logger = UnderdeepLogger(experiment='generative-models/tinygsm',
                 name=p['name'] or f"paper {p['suite']} step{p['step']} {p['weights']} FP32",
                 save_dir=str(out), resume=False, force_new_run=True)
        if logger.experiment is None:
            raise RuntimeError('Could not create the separate evaluation run')
        logger.log_hyperparams({'evaluation': p})
    model._trainer = SimpleNamespace(global_rank=rank, world_size=world,
                                     global_step=p['step'], logger=logger)
    dataset = model._gsm8k_benchmark_dataset()
    if len(dataset) != 1319:
        raise ValueError('Incomplete tokenized test set')
    indices = list(range(rank,1319,world))
    order = [i for r in range(world) for i in range(r,1319,world)]
    report = dict(plan=p, complete=False, results=[],
                  underdeep_run_uid=None if logger is None else logger.version)
    def sample(tag, steps):
        with torch.inference_mode(), fp32_prediction_normalization(model):
            rows, _, nfes = model._gsm8k_generate(dataset, indices, p['batch_size'], steps, 'sflm')
        if rank != 0:
            return None
        if len(rows) != 1319 or len(nfes) != 1319:
            raise ValueError('Incomplete gathered predictions')
        scores = [int(sandbox_gsm8k.evaluate_samples(r,g,timeout_s=5.0)) for r,g in rows]
        records = [dict(index=i,response=r,ground_truth=g,nfe=n,correct=c)
                   for i,(r,g),n,c in zip(order,rows,nfes,scores)]
        with gzip.open(out / (tag+'.json.gz'),'wt') as f:
            json.dump(records,f)
        return dict(zip(order,scores)),sum(nfes)/1319
    try:
        if p['suite'] == 'confidence':
            model.remasking_strategy = 'low_confidence_dynamic'
            for threshold in p['thresholds']:
                model.dynamic_threshold = threshold
                got = sample(f'threshold_{threshold:g}',64)
                if rank == 0:
                    scores,nfe = got
                    point = dict(threshold=threshold,nfe=nfe,correct=sum(scores.values()),
                                 total=1319,accuracy_percent=100*sum(scores.values())/1319)
                    report['results'].append(point)
                    if logger:
                        logger.log_metrics({f'pareto/gsm8k_step_{p["step"]}_fp32norm':point['accuracy_percent']/100},step=round(nfe*100))
                    write_json(out/'results.json',report)
                    print(point,flush=True)
                torch.distributed.barrier()
        else:
            model.remasking_strategy = 'random'
            for steps in p['budgets']:
                counts = [0]*1319
                nfes = []
                for draw in range(p['samples']):
                    # A different reproducible RNG stream for each draw and rank.
                    L.seed_everything(p['seed']+100003*draw+rank,workers=False)
                    got = sample(f'nfe{steps}_draw{draw:03d}',steps)
                    if rank == 0:
                        scores,nfe = got
                        counts = [a+scores[i] for i,a in enumerate(counts)]
                        nfes.append(nfe)
                        write_json(out/f'progress_nfe{steps}.json',dict(draws=draw+1,correct_per_question=counts,nfes=nfes))
                        print(f'pass@k budget={steps} draws={draw+1}/{p["samples"]}',flush=True)
                    torch.distributed.barrier()
                if rank == 0:
                    values = {str(k):100*sum(pass_at_k(p['samples'],c,k) for c in counts)/1319 for k in p['ks']}
                    report['results'].append(dict(steps=steps,nfe=sum(nfes)/len(nfes),samples=p['samples'],pass_at_k_percent=values))
                    if logger:
                        for k,value in values.items():
                            logger.log_metrics({f'passk/steps_{steps}_fp32norm':value/100},step=int(k))
                    write_json(out/'results.json',report)
        if rank == 0:
            report['complete']=True
            write_json(out/'results.json',report)
            if logger:logger.finalize('success')
        torch.distributed.barrier()
    except BaseException:
        if logger:logger.finalize('failed')
        raise
    finally:
        torch.distributed.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-plan',help=argparse.SUPPRESS)
    parser.add_argument('--suite',choices=['confidence','passk'],default='confidence')
    parser.add_argument('--source-run')
    parser.add_argument('--checkpoint')
    parser.add_argument('--config')
    parser.add_argument('--base-teacher',action='store_true')
    parser.add_argument('--step',type=int)
    parser.add_argument('--weights',choices=['raw','ema'])
    parser.add_argument('--snapshot-root')
    parser.add_argument('--devices',type=int,default=2)
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--model-weights-dir')
    parser.add_argument('--data-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--name')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--thresholds',nargs='+',type=float,default=THRESHOLDS)
    parser.add_argument('--budgets',nargs='+',type=int,default=[4,8,16,32])
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--ks',nargs='+',type=int,default=[1,2,4,8,16,32])
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--no-underdeep',action='store_true')
    parser.add_argument('--no-snapshot',action='store_true')
    args=parser.parse_args()
    if args.worker_plan:
        worker(args.worker_plan);return
    if args.step is None or args.step<0 or args.weights is None:
        parser.error('Explicit --step and --weights are required')
    if args.devices<1 or (args.batch_size is not None and args.batch_size<1):
        parser.error('Devices and batch size must be positive')
    if any(not 0<t<1 for t in args.thresholds) or len(set(args.thresholds))!=len(args.thresholds):
        parser.error('Thresholds must be unique, finite and inside (0,1)')
    args.thresholds=sorted(args.thresholds)
    if args.samples<1 or any(not 1<=k<=args.samples for k in args.ks) or any(b<1 for b in args.budgets):
        parser.error('Positive budgets and 1 <= k <= samples required')
    plan=prepare(args)
    if args.plan_only:return
    try:
        subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
                        f'--nproc_per_node={args.devices}',str(Path(__file__).resolve()),
                        '--worker-plan',str(Path(plan['output_dir'])/'plan.json')],check=True)
    finally:
        if not args.no_snapshot:
            from nirvana_dl import snapshot
            root=Path(snapshot.get_snapshot_path()).resolve()
            dest=root/'paper_evaluations'/Path(plan['output_dir']).name
            if dest.exists():raise FileExistsError(dest)
            shutil.copytree(plan['output_dir'],dest)
            snapshot.dump_snapshot(str(root))


if __name__=='__main__':
    main()
