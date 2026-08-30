"""What `adversarial_distill.mmd_kernel_batch_size` costs, measured.

Run: `python scripts/probe_kernel_batch_size.py`  (~2 min on one GPU)

Pooling N sequences into one kernel row buys a statistic with N times the
tokens behind it, and costs credit assignment: `paired_sequence_reward`
computes ONE reward per row and `repeat_interleave`s it to all N sequences,
while REINFORCE still multiplies each sequence's advantage by ITS OWN
log-probability. This measures both sides against the real feature map
(mdlm-owt read at its middle block, the `mmd_feature_source=diffusion` path)
on real OpenWebText rows, with per-sequence corruption as a stand-in for
students of varying quality.

Batch order matters: `merge_kernel_rows` folds ADJACENT sequences, so quality
must be assigned i.i.d. across the batch or the probe flatters pooling by
grouping sequences that were already alike.
"""
import sys

import datasets
import torch
import transformers

sys.path.insert(0, __file__.rsplit('/', 2)[0])
import mmd_reward

DEV = 'cuda'
B, L = 32, 1024
ROWS = (1, 2, 4, 8, 16, 32)
OWT = ('/extra_disk_1/mnt_source/data/owt/'
       'openwebtext-valid_validation_bs1024_wrapped.dat')


def pearson(a, b):
  a, b = a - a.mean(), b - b.mean()
  return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
  ds = datasets.load_from_disk(OWT)
  x0 = torch.tensor([ds[i]['input_ids'] for i in range(B)], device=DEV)[:, :L]
  model = transformers.AutoModelForMaskedLM.from_pretrained(
    'kuleshov-group/mdlm-owt', trust_remote_code=True).to(DEV).eval()
  vocab = model.config.vocab_size
  critic = mmd_reward.FrozenBackboneFeatures(model, vocab - 1, layer_index=None,
                                             noise_level=0.1)
  critic.kernel = 'linear'
  g = torch.Generator(device=DEV).manual_seed(7)
  selected = torch.rand(B, L, device=DEV, generator=g) < 0.5

  def corrupt(frac, gen, sel=None):
    sel = selected if sel is None else sel
    hit = (torch.rand(x0.shape, device=DEV, generator=gen)
           < frac.unsqueeze(-1)) & sel
    rnd = torch.randint(0, vocab - 1, x0.shape, device=DEV, generator=gen)
    return torch.where(hit, rnd, x0)

  def reward(tokens, rows_of, sel=None, seed=11):
    critic.kernel_batch_size = rows_of
    torch.manual_seed(seed)                     # pin the critic's own masking
    r, _ = critic.paired_reward(x0, tokens, selected if sel is None else sel,
                                eos_id=None)
    return r.float().cpu()

  # A. How much of the reward a sequence receives is about that sequence?
  print('=== A. credit assignment: corr(reward_i, own quality_i) ===')
  print(f'{"rows_of":>8}{"corr":>10}{"1/sqrt(N)":>12}{"reward std":>13}')
  for n in ROWS:
    cs, sds = [], []
    for trial in range(8):
      gen = torch.Generator(device=DEV).manual_seed(100 + trial)
      frac = torch.rand(B, device=DEV, generator=gen) * 0.9 + 0.05
      r = reward(corrupt(frac, gen), n)
      cs.append(pearson(r, (-frac).cpu()))
      sds.append(float(r.std()))
    ref = (1.0 / n ** 0.5) if n < B else 0.0
    print(f'{n:>8}{sum(cs)/len(cs):>10.3f}{ref:>12.3f}'
          f'{sum(sds)/len(sds):>13.4f}')

  # B. Cross-talk: does a sequence's reward move when a ROW-MATE changes?
  print('\n=== B. cross-talk: rewrite sequence 0, watch sequences 1..3 ===')
  gen = torch.Generator(device=DEV).manual_seed(42)
  frac = torch.rand(B, device=DEV, generator=gen) * 0.9 + 0.05
  base = corrupt(frac, gen)
  moved = base.clone()
  gen2 = torch.Generator(device=DEV).manual_seed(43)
  rnd = torch.randint(0, vocab - 1, (L,), device=DEV, generator=gen2)
  moved[0] = torch.where(selected[0], rnd, x0[0])   # only sequence 0 changed
  for n in ROWS:
    d = (reward(moved, n) - reward(base, n)).abs()
    outside = d[n:]
    outside = f'{outside.max():.4f}' if outside.numel() else 'n/a (one row)'
    print(f'  rows_of={n:>2}: |d| own={d[0]:.4f}  row-mates 1..3='
          f'{d[1]:.4f},{d[2]:.4f},{d[3]:.4f}  outside-row max={outside}')

  # C. What actually drives learning: the advantage after the LOO baseline.
  print('\n=== C. REINFORCE advantage vs the advantage deserved (G=4) ===')
  G = 4
  for n in ROWS:
    cs, spreads = [], []
    for trial in range(6):
      gen = torch.Generator(device=DEV).manual_seed(200 + trial)
      centre = torch.rand(B, device=DEV, generator=gen) * 0.6 + 0.2
      rs, qs = [], []
      for _ in range(G):
        jit = (torch.rand(B, device=DEV, generator=gen) - 0.5) * 0.3
        f = (centre + jit).clamp(0.02, 0.98)
        rs.append(reward(corrupt(f, gen), n))
        qs.append((-f).cpu())
      R, Q = torch.stack(rs, -1), torch.stack(qs, -1)
      A = mmd_reward.group_advantages(R)
      A_true = Q - (Q.sum(-1, keepdim=True) - Q) / (G - 1)
      cs.append(pearson(A.flatten(), A_true.flatten()))
      spreads.append(float(R.std(-1).mean()))
    print(f'  rows_of={n:>2}: corr(advantage, deserved) = '
          f'{sum(cs)/len(cs):>6.3f}   within-group spread = '
          f'{sum(spreads)/len(spreads):.5f}')

  # D. Does pooling pay when the per-sequence statistic is starved?
  print('\n=== D. corr(reward, quality) vs how many tokens each sequence wrote ===')
  print(f'{"kernel":>7}{"tokens/seq":>12}' +
        ''.join(f'{f"rows_of={n}":>13}' for n in (1, 2, 4, 8)))
  for kernel in ('linear', 'rbf'):
    critic.kernel = kernel
    for mf in (0.005, 0.02, 0.05, 0.2, 0.5):
      g2 = torch.Generator(device=DEV).manual_seed(7)
      sel = torch.rand(B, L, device=DEV, generator=g2) < mf
      cols = []
      for n in (1, 2, 4, 8):
        cs = []
        for trial in range(6):
          gen = torch.Generator(device=DEV).manual_seed(300 + trial)
          frac = torch.rand(B, device=DEV, generator=gen) * 0.9 + 0.05
          r = reward(corrupt(frac, gen, sel), n, sel=sel)
          cs.append(pearson(r, (-frac).cpu()))
        cols.append(sum(cs) / len(cs))
      print(f'{kernel:>7}{int(mf*L):>12}' + ''.join(f'{c:>13.3f}' for c in cols))


if __name__ == '__main__':
  main()
