"""`mmd_kernel_level`: one sample is a token, or a mean-pooled sentence.

The linear kernel sees only means, so a mean of means equals the pooled mean
whenever the sequences in a row carry equal scored counts -- always at
rows_of=1. The rbf kernel compares distributions, so the two levels are
different statistics outright.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmd_reward import paired_sequence_reward as R

ok = []
def chk(n, c): ok.append((n, bool(c))); print(f"{'PASS' if c else 'FAIL'}  {n}")

B, L, D = 8, 40, 6
torch.manual_seed(0)
clean = torch.randn(B, L, D)
# A per-sequence constant shift: moves the SENTENCE vectors as much as the
# token ones. A small i.i.d. per-position jitter would average away under
# pooling and leave both MMDs at ~0, which measures nothing.
gen = clean + 0.8 * torch.randn(B, 1, D)

# --- equal scored counts per sequence ---
eq = torch.zeros(B, L, dtype=torch.bool); eq[:, 5:25] = True     # 20 each
for n in (1, 2, 4, 8):
    a, _ = R(clean, gen, eq, kernel='linear', rows_of=n, level='token')
    b, _ = R(clean, gen, eq, kernel='linear', rows_of=n, level='sentence')
    chk(f'linear, EQUAL counts, rows_of={n}: identical (max |d|={(a-b).abs().max():.2e})',
        torch.allclose(a, b, atol=1e-5))

# --- unequal scored counts ---
un = torch.zeros(B, L, dtype=torch.bool)
for i in range(B):
    un[i, : 4 + 4 * i] = True                                     # 4..32 positions
for n in (1, 2, 4):
    a, _ = R(clean, gen, un, kernel='linear', rows_of=n, level='token')
    b, _ = R(clean, gen, un, kernel='linear', rows_of=n, level='sentence')
    same = torch.allclose(a, b, atol=1e-5)
    chk(f'linear, UNEQUAL counts, rows_of={n}: '
        + ('identical' if n == 1 else f'differs (max |d|={(a-b).abs().max():.4f})'),
        same if n == 1 else not same)

# --- rbf must change ---
for n in (2, 4, 8):
    a, _ = R(clean, gen, eq, kernel='rbf', rows_of=n, level='token')
    b, _ = R(clean, gen, eq, kernel='rbf', rows_of=n, level='sentence')
    chk(f'rbf, rows_of={n}: sentence differs from token '
        f'(token {a[0]:.5f} vs sentence {b[0]:.5f})',
        abs(float(a[0]) - float(b[0])) > 1e-4)

# --- guard ---
try:
    R(clean, gen, eq, kernel='rbf', rows_of=1, level='sentence')
    chk('rbf + sentence + rows_of=1 raises', False)
except ValueError as e:
    chk('rbf + sentence + rows_of=1 raises', 'mmd_kernel_batch_size >= 2' in str(e))

# --- shape / sharing invariants ---
r, act = R(clean, gen, eq, kernel='rbf', rows_of=4, level='sentence')
chk('reward stays per sequence', r.shape == (B,))
chk(f'a row of 4 shares one value (row0={r[:4].tolist()})',
    len(set(r[:4].tolist())) == 1 and len(set(r.tolist())) == B // 4)
# identical inputs must score 0
z, _ = R(clean, clean, eq, kernel='rbf', rows_of=4, level='sentence')
chk(f'identical sets score ~0 ({z.abs().max():.2e})', float(z.abs().max()) < 1e-5)

print(f"\n{sum(v for _, v in ok)}/{len(ok)} passed")
sys.exit(0 if all(v for _, v in ok) else 1)
