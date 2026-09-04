"""Did the generator escape the factorisation error on the 2-example set?

Trained on exactly two sequences A and B, a one-step factorised model can only
produce the per-position mixture of the two, so its samples are token-wise
chimeras: roughly half their positions match A and half match B, and neither
sequence is ever emitted cleanly. A generator that can condition on a
per-sample input noise vector is able to map noise -> A or noise -> B instead.

This reports, per sample, the fraction of positions matching A and matching B:

  purity   = max(match_A, match_B)      1.0 = a clean mode, ~0.5 = a chimera
  coverage = are BOTH modes produced    a generator that always emits A has
                                        escaped factorisation but collapsed

Usage:
  python scripts/check_factorization.py --ckpt <path> --data /tmp/overfit_cache
                                        [--steps 1 2 4] [--samples 32]
                                        [--direct-only]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def purity_report(samples, a, b, label):
  """`samples` (N, L) long; `a`, `b` (L,) long."""
  match_a = (samples == a).float().mean(-1)
  match_b = (samples == b).float().mean(-1)
  purity = torch.maximum(match_a, match_b)
  nearest_a = (match_a >= match_b)
  print(f'  {label}')
  print(f'     match_A  mean {match_a.mean():.3f}   match_B  mean {match_b.mean():.3f}')
  print(f'     purity   mean {purity.mean():.3f}  min {purity.min():.3f}  '
        f'max {purity.max():.3f}')
  print(f'     nearest mode: A x{int(nearest_a.sum())}, B x{int((~nearest_a).sum())}'
        f'   (both covered: {bool(nearest_a.any() and (~nearest_a).any())})')
  clean = (purity > 0.9).float().mean()
  print(f'     samples >90% one mode: {clean:.0%}'
        f'   <-- chimeras if this is 0% and purity ~= 0.5')
  return float(purity.mean())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--data', default='/tmp/overfit_cache')
  ap.add_argument('--steps', type=int, nargs='+', default=[1, 2, 4, 8])
  ap.add_argument('--samples', type=int, default=32)
  ap.add_argument('--direct-only', action='store_true',
                  help='score exactly one model forward from an all-MASK '
                       'input, without the sampler cleanup forward')
  ap.add_argument('--no-ema', action='store_true',
                  help='score the RAW weights. Essential for a short overfit: '
                       'at training.ema=0.9999 the shadow still carries '
                       '0.9999^steps of the initial checkpoint -- 92%% of it '
                       'after 800 steps -- so the EMA copy is mostly the model '
                       'you started from and shows no learning at all.')
  args = ap.parse_args()

  import omegaconf
  # Importing `main` registers the custom resolvers (`device_count`, `div_up`,
  # `eval`, `cwd`) that the saved config interpolates; without them the
  # checkpoint's config cannot be resolved at all.
  import main  # noqa: F401
  import algo
  import dataloader as dl
  import trainer_base

  ref = torch.load(os.path.join(args.data, 'examples.pt'), weights_only=False)
  ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
  cfg = omegaconf.OmegaConf.create(ckpt['hyper_parameters']['config'])
  print(f"config: noise_dim={cfg.adversarial_distill.get('noise_dim', 0)} "
        f"noise_scale={cfg.adversarial_distill.get('noise_scale', 0)} "
        f"noise_shared={cfg.adversarial_distill.get('noise_shared', False)} "
        f"dmd_weight={cfg.adversarial_distill.dmd_weight} "
        f"mmd_w={cfg.adversarial_distill.mmd_reward_weight}")

  # `TrainerBase.__init__` calls `save_hyperparameters()`, so the checkpoint
  # already carries the tokenizer the run used; rebuilding it is a fallback.
  tok = ckpt['hyper_parameters'].get('tokenizer') or dl.get_tokenizer(cfg)
  model = algo.MDLM(cfg, tokenizer=tok).to('cuda').eval()
  missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
  print(f'loaded: {len(missing)} missing, {len(unexpected)} unexpected keys')
  if args.no_ema:
    print('scoring RAW weights (EMA not applied)')
  elif model.ema is not None and 'ema' in ckpt:
    model.ema.load_state_dict(ckpt['ema'])
    model.ema.copy_to(model._get_parameters())
    decay = float(cfg.training.ema)
    print(f'EMA weights swapped in (decay={decay}); after N steps the shadow '
          f'still holds decay^N of the initial checkpoint')

  a = torch.tensor(ref['A'], device='cuda')
  b = torch.tensor(ref['B'], device='cuda')
  print(f'\nA/B agree on {(a == b).float().mean():.1%} of positions '
        f'(pure chimera purity ~= 0.5 + half that)')
  with torch.no_grad():
    torch.manual_seed(0)
    xt = torch.full(
      (args.samples, a.numel()), model.mask_index,
      dtype=torch.long, device='cuda')
    t = torch.ones(args.samples, device='cuda')
    _, alpha_t = model.noise(t)
    sigma = model._sigma_from_alphat(alpha_t.unsqueeze(-1))
    probs = model.forward(xt, sigma=sigma).exp()
    direct = trainer_base.sample_categorical(probs)
    purity_report(direct, a, b, '--- direct: exactly one forward ---')
    entropy = -(probs.clamp_min(1e-12).log() * probs).sum(-1).mean()
    confidence = probs.max(-1).values.mean()
    print(f'     soft token entropy {entropy:.3f}, mean max probability '
          f'{confidence:.3f}')
    greedy = probs.argmax(-1)
    purity_report(greedy, a, b,
                  '--- greedy per noise draw (no categorical randomness) ---')
    print(f'     unique greedy sequences: '
          f'{greedy.unique(dim=0).shape[0]}/{args.samples}')

    if args.direct_only:
      return
    for k in args.steps:
      torch.manual_seed(0)
      s = model.generate_samples(num_samples=args.samples, num_steps=k)
      purity_report(s, a, b, f'--- {k} step(s) ---')


if __name__ == '__main__':
  main()
