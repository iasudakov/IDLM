"""Start a fresh CDLM experiment from an exact native MMD EMA checkpoint.

The incoming state is read only. Recollect teacher trajectories with the new
weights; do not restore the MMD optimizer, tracker, or training iteration.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def existing(paths, directory=False):
    for path in paths:
        if path.is_dir() if directory else path.is_file():
            return path.resolve()
    raise FileNotFoundError('Missing asset: ' + ', '.join(map(str, paths)))


def select_mmd(snapshot, uid, step):
    import torch
    root = Path(snapshot).resolve()
    matches = []
    for meta in root.rglob('underdeep_metadata.pt'):
        if root not in meta.resolve().parents:
            continue
        value = torch.load(meta, map_location='cpu', weights_only=False)
        if value.get('run_uid') == uid:
            matches.append(meta.parent)
    if len(matches) != 1:
        raise ValueError(f'Expected one source state for {uid}, found {len(matches)}')
    source = matches[0]
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(source / '.hydra/config.yaml')
    if (cfg.algo.name != 'mdlm' or cfg.model.length != 512
            or cfg.data.separator != r'\n'
            or not cfg.adversarial_distill.is_distill
            or cfg.adversarial_distill.mmd_reward_weight <= 0):
        raise ValueError('Expected a native TinyGSM MMD source with the historical separator')
    hits, available = [], []
    for path in sorted((source / 'checkpoints').glob('*.ckpt')):
        if source not in path.resolve().parents:
            raise ValueError('Checkpoint escapes source state')
        ckpt = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        saved_step = ckpt.get('global_step')
        available.append(saved_step)
        if saved_step == step:
            if not ckpt.get('ema', {}).get('shadow_params'):
                raise ValueError('Requested MMD checkpoint has no EMA')
            hits.append(path.resolve())
        del ckpt
        gc.collect()
    if len(hits) != 1:
        raise ValueError(f'Expected one checkpoint at global_step={step}; available={available}')
    return hits[0]


def command(repo, checkpoint, devices, assets, out, step, extra):
    if devices not in (1, 2, 4, 8):
        raise ValueError('Use 1, 2, 4, or 8 GPUs')
    accum = 512 // (devices * 32)
    overrides = [
        f'trainer.devices={devices}', f'trainer.accumulate_grad_batches={accum}',
        'loader.batch_size=32', 'loader.global_batch_size=512',
        f'trainer.val_check_interval={500 * accum}',
        'callbacks.checkpoint_every_n_steps.every_n_train_steps=500',
        'eval.gsm8k_greedy_token=False',
        'eval.gsm8k_pareto_thresholds=[0.325,0.4,0.475,0.55,0.625,0.7,0.775,0.85,0.925,0.95,0.98,0.99,0.995,0.998,0.999]',
        f'logger.name=MMD step{step} EMA to CDLM {devices}gpu',
        *extra,
        # These define this experiment and are identical for collection/train.
        f'training.finetune_path={json.dumps(str(checkpoint))}',
        'training.teacher_use_ema=True', 'training.teacher_allow_hub=False',
        'training.student_init_path=null', 'training.student_init_use_ema=True',
        'checkpointing.resume_from_ckpt=False',
        'logger.resume=False', 'logger.force_new_run=True',
        f'data.cache_dir={json.dumps(str(assets["cache"]))}',
        f'model_weights_dir={json.dumps(str(assets["bundle"]))}',
        f'eval.gsm8k_data_path={json.dumps(str(assets["data"]))}',
        f'checkpointing.save_dir={json.dumps(str(out))}',
        f'hydra.run.dir={json.dumps(str(out))}',
        f'checkpointing.snapshot_namespace=cdlm_mmd{step}',
        f'checkpointing.trajectory_snapshot_namespace=cdlm_mmd{step}_trajectories',
    ]
    return ['bash', str(repo / 'scripts/train_cdlm_tinygsm_nirvana.sh'), *overrides]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--step', type=int, default=8000)
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--snapshot-root')
    parser.add_argument('--output-dir')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.step < 1:
        parser.error('Step must be positive')
    repo = Path(__file__).resolve().parent
    os.chdir(repo)
    from cdlm_snapshot_init import resolve_snapshot_root
    root = resolve_snapshot_root(args.snapshot_root)
    checkpoint = select_mmd(root, args.source_run, args.step)
    inputs = Path(os.environ.get('INPUT_PATH', repo))
    assets = {
        'bundle': existing([inputs / 'tinygsm_mmd_offline_bundle', repo / 'cache_dir/tinygsm_mmd_offline_bundle'], True),
        'cache': existing([inputs / 'nirvana_checkpoints/cache_dir', repo / 'cache_dir'], True),
        'data': existing([inputs / 'nirvana_checkpoints/data/gsm8k_test.json', repo / 'data/gsm8k_test.json']),
    }
    if len(json.loads(assets['data'].read_text())) != 1319:
        raise ValueError('Require all 1319 GSM8K examples')
    cache = assets['cache'] / 'tiny_gsm_bs512_answer_only_filtered_train_on_pad_HuggingFaceTB__SmolLM-135M_sep5c6e'
    if not all((cache / split).is_dir() for split in ('train', 'validation')):
        raise FileNotFoundError(f'Missing separator-aware TinyGSM cache: {cache}')
    out = Path(args.output_dir or repo / f'cdlm_mmd{args.step}').resolve()
    if out == root or root in out.parents or out in root.parents:
        raise ValueError('Training output must not overlap source snapshot')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty --output-dir for the new CDLM experiment')
    cmd = command(repo, checkpoint, args.devices, assets, out, args.step, args.extra)
    print(shlex.join(cmd), flush=True)
    if args.plan_only:
        return
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    (out / 'mmd_initialization.json').write_text(json.dumps({
        'source_run': args.source_run, 'step': args.step, 'weights': 'ema',
        'checkpoint': str(checkpoint), 'checkpoint_sha256': digest.hexdigest(),
        'teacher_and_student': 'MMD EMA; fresh optimizer and fresh CDLM EMA history',
        'command': cmd,
    }, indent=2) + '\n')
    env = dict(os.environ, CDLM_PRESET='cdlm_tinygsm_mmd',
               CDLM_PREPARE_GPUS=str(args.devices), CDLM_CODE_DIR=str(repo),
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', TOKENIZERS_PARALLELISM='false')
    subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__':
    main()
