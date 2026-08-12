"""Reproduce the D-MMD Table 1 "Masked Teacher" row from a trained checkpoint.

The paper's protocol (arXiv:2603.20155 Sec. 6.1): 50K unconditional samples per
NFE, FID measured against the CIFAR-10 *train* split. Target:

    NFE   |   16     32     64    128    256    512   1024
    FID   |  122.9   47.1   20.0   11.1    7.8    6.7    6.4

All sampling uses the EMA weights. `--nfe` counts denoiser calls, matching how
`train_cifar10_mdlm.sample` defines it.

  torchrun --nproc_per_node=8 eval_cifar10_fid.py \
      --ckpt outputs/cifar10_teacher/checkpoints/latest.pt

  # a single cheap point while training is still running
  torchrun --nproc_per_node=8 eval_cifar10_fid.py \
      --ckpt outputs/.../ema_0100000.pt --nfe 128 --num-samples 10000
"""
import argparse
import json
import os
import time

import torch

from train_cifar10_mdlm import (
  FID, build_model, generate_for_fid, is_main, setup_distributed)


def load_weights(path, model, device):
  """Load EMA weights from either a `latest.pt` or a bare `ema_*.pt`.

  Returns the training `args` dict when the checkpoint carries one (so the
  architecture flags do not have to be repeated on the command line).
  """
  ckpt = torch.load(path, map_location=device, weights_only=False)
  shadow = ckpt['ema']['shadow_params'] if 'ema' in ckpt else ckpt['shadow_params']
  params = [p for p in model.parameters() if p.requires_grad]
  if len(shadow) != len(params):
    raise ValueError(
      f'checkpoint has {len(shadow)} EMA tensors but the model has '
      f'{len(params)} trainable parameters -- --ch / --time-cond probably do '
      f'not match how this checkpoint was trained')
  with torch.no_grad():
    for s, p in zip(shadow, params):
      p.data.copy_(s.to(device))
  return ckpt.get('args', {}), ckpt.get('step')


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--ckpt', required=True)
  p.add_argument('--out', default=None,
                 help='where to write fid_results.json (default: next to --ckpt)')
  p.add_argument('--data-dir', default='~/data/cifar10')
  p.add_argument('--nfe', default='16,32,64,128,256,512,1024')
  p.add_argument('--num-samples', type=int, default=50_000)
  p.add_argument('--batch-size', type=int, default=250, help='per GPU')
  p.add_argument('--ch', type=int, default=128)
  p.add_argument('--time-cond', action='store_true')
  p.add_argument('--seed', type=int, default=1)
  args = p.parse_args()

  rank, world_size, device = setup_distributed()
  torch.manual_seed(args.seed + rank)

  ckpt_path = os.path.abspath(args.ckpt)
  out_dir = os.path.abspath(args.out) if args.out else os.path.dirname(ckpt_path)
  os.makedirs(out_dir, exist_ok=True)

  # Architecture flags recorded at training time win over the CLI defaults, so
  # the same command works for any checkpoint this repo produced.
  peek = torch.load(ckpt_path, map_location='cpu', weights_only=False)
  train_args = peek.get('args', {}) if isinstance(peek, dict) else {}
  ch = train_args.get('ch', args.ch)
  time_cond = train_args.get('time_cond', args.time_cond)
  del peek

  model = build_model(ch=ch, dropout=0.0, time_conditioning=time_cond).to(device)
  _, step = load_weights(ckpt_path, model, device)
  model.eval()

  fid = FID(device, os.path.join(out_dir, 'fid_ref.pt'),
            args.data_dir) if is_main(rank) else None

  nfes = [int(v) for v in args.nfe.split(',')]
  results = {}
  for nfe in nfes:
    t0 = time.time()
    images = generate_for_fid(model, args.num_samples, nfe, args.batch_size,
                              device, time_cond, rank, world_size,
                              log_every=20)
    if is_main(rank):
      results[nfe] = fid.score(images)
      print(f'[fid] NFE {nfe:>5d}  FID {results[nfe]:7.2f}   '
            f'({args.num_samples} samples, {time.time() - t0:.0f}s)',
            flush=True)

  if is_main(rank):
    paper = {16: 122.9, 32: 47.1, 64: 20.0, 128: 11.1,
             256: 7.8, 512: 6.7, 1024: 6.4}
    print(f'\n  D-MMD Table 1, Masked Teacher -- checkpoint {ckpt_path}'
          + (f' (step {step})' if step else ''))
    print(f'  {"NFE":>6} | {"this run":>9} | {"paper":>7}')
    print(f'  {"-" * 6}-+-{"-" * 9}-+-{"-" * 7}')
    for nfe in nfes:
      ref = f'{paper[nfe]:7.1f}' if nfe in paper else f'{"-":>7}'
      print(f'  {nfe:>6} | {results[nfe]:9.2f} | {ref}')
    path = os.path.join(out_dir, 'fid_results.json')
    with open(path, 'w') as f:
      json.dump({'checkpoint': ckpt_path, 'step': step,
                 'num_samples': args.num_samples,
                 'fid': results, 'paper': paper}, f, indent=2)
    print(f'\n  written to {path}', flush=True)

  if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
  main()
