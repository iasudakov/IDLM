"""Numerical checks for `mmd_reward.py`. CPU only, no data, no checkpoints.

Run:  python scripts/check_mmd_reward.py

Covers the parts of the port that are easy to get wrong and impossible to notice
from a training curve: the unbiased RBF estimator against a brute-force double
loop, the rank-4 (UNet) hook flatten, and the `active` bookkeeping -- which on
this branch is the ONLY thing standing between an empty row and a fabricated
reward, because the feature grid carries no mask.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmd_reward  # noqa: E402


FAILURES = []


def check(name, ok, detail=''):
  print(f'{"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}')
  if not ok:
    FAILURES.append(name)


def brute_force_mmd2(x, y, w, alpha):
  """The estimator `rbf_sequence_reward` computes, written out as loops."""
  idx = [i for i in range(x.shape[0]) if w[i] > 0]
  n = len(idx)
  if n < 2:
    return None

  def k(a, b):
    return float(torch.exp(-alpha * (a - b).pow(2).sum()))

  s_xx = sum(k(x[i], x[j]) for i in idx for j in idx if i != j)
  s_yy = sum(k(y[i], y[j]) for i in idx for j in idx if i != j)
  s_xy = sum(k(x[i], y[j]) for i in idx for j in idx)
  return s_xx / (n * (n - 1)) + s_yy / (n * (n - 1)) - 2.0 * s_xy / (n * n)


class _FakeBlock(torch.nn.Module):
  """Emits a rank-4 activation, the way every UNet module does."""

  def __init__(self, channels, grid):
    super().__init__()
    self.channels, self.grid = channels, grid

  def forward(self, x):
    batch = x.shape[0]
    # Deterministic in the input so clean and generated rows differ.
    base = x.float().mean(dim=1).reshape(batch, 1, 1, 1)
    ramp = torch.linspace(0, 1, self.channels).reshape(1, -1, 1, 1)
    cells = torch.linspace(0, 1, self.grid * self.grid).reshape(
      1, 1, self.grid, self.grid)
    return base + ramp + cells


class _FakeUNet(torch.nn.Module):
  """Enough of a UNet for the layer probe and the hook: `middle_modules`."""

  def __init__(self, channels=8, grid=4):
    super().__init__()
    self.middle_modules = torch.nn.ModuleList(
      [_FakeBlock(channels, grid) for _ in range(3)])

  def forward(self, x, sigma=None):
    h = x
    for module in self.middle_modules:
      h = module(h if h.ndim == 2 else h.flatten(1))
    return h


def main():
  torch.manual_seed(0)

  # -- 1. the unbiased RBF estimator against a brute-force double loop ------ #
  for n_sel in (16, 7):
    clean = torch.randn(2, 16, 5)
    generated = clean + 0.3 * torch.randn(2, 16, 5)
    selected = torch.zeros(2, 16, dtype=torch.bool)
    selected[:, :n_sel] = True
    alpha = 0.7
    reward, active = mmd_reward.rbf_sequence_reward(
      clean, generated, selected, alpha=alpha)
    worst = 0.0
    for row in range(2):
      expected = brute_force_mmd2(clean[row], generated[row],
                                  selected[row].float(), alpha)
      worst = max(worst, abs(float(-max(expected, 0.0)) - float(reward[row])))
    check(f'rbf estimator matches brute force (n={n_sel})', worst < 1e-5,
          f'max|diff| = {worst:.2e}')
    check(f'rbf active at n={n_sel}', bool(active.all()))

  # -- 2. identical rows score ~0, a different row scores worse ------------ #
  clean = torch.randn(3, 16, 8)
  selected = torch.ones(3, 16, dtype=torch.bool)
  same, _ = mmd_reward.paired_sequence_reward(clean, clean.clone(), selected)
  worse, _ = mmd_reward.paired_sequence_reward(
    clean, torch.randn(3, 16, 8), selected)
  check('linear kernel: identical rows reward ~0',
        float(same.abs().max()) < 1e-6, f'max|R| = {float(same.abs().max()):.2e}')
  check('linear kernel: different row is strictly worse',
        bool((worse < same - 1e-3).all()),
        f'{float(worse.mean()):.4f} vs {float(same.mean()):.4f}')

  # -- 3. rank-4 (UNet) hook flatten --------------------------------------- #
  net = _FakeUNet(channels=8, grid=4)
  modules, indices = mmd_reward.resolve_feature_layers(net, None)
  check('UNet default block is the bottleneck (last middle module)',
        indices == [2], f'indices = {indices}')
  with mmd_reward.capture_layer_hidden(net, None) as (feats, idx):
    net(torch.randint(0, 5, (6, 12)))
  hidden = feats[idx[0]]
  check('rank-4 activation is flattened to (batch, H*W, C)',
        tuple(hidden.shape) == (6, 16, 8), f'shape = {tuple(hidden.shape)}')

  # -- 4. merge_kernel_rows guards the fold -------------------------------- #
  ok = False
  try:
    mmd_reward.merge_kernel_rows(clean, clean, selected, rows_of=2)
  except ValueError:
    ok = True
  check('merge_kernel_rows rejects a rows_of that does not divide the batch', ok)
  folded = mmd_reward.merge_kernel_rows(
    torch.randn(4, 16, 8), torch.randn(4, 16, 8),
    torch.ones(4, 16, dtype=torch.bool), rows_of=2)
  check('merge_kernel_rows folds 4x16 into 2x32',
        tuple(folded[0].shape) == (2, 32, 8), f'{tuple(folded[0].shape)}')

  # -- 5. `active` comes from the token mask, not the feature grid ---------- #
  critic = mmd_reward.FrozenBackboneFeatures(net, mask_index=256,
                                             noise_level=0.0)
  tokens = torch.randint(0, 250, (4, 12))
  token_selected = torch.ones(4, 12, dtype=torch.bool)
  token_selected[1] = False          # this row wrote nothing
  reward, active = critic.paired_reward(
    tokens, torch.randint(0, 250, (4, 12)), token_selected)
  check('a row that wrote nothing is inactive',
        (not bool(active[1])) and bool(active[0]),
        f'active = {active.tolist()}')
  check('an inactive row scores exactly 0', float(reward[1]) == 0.0)

  # -- 6. fold_active ------------------------------------------------------ #
  folded = mmd_reward.fold_active(
    torch.tensor([True, False, False, False]), rows_of=2)
  check('fold_active: a row is active if any member is',
        folded.tolist() == [True, True, False, False], f'{folded.tolist()}')

  # -- 7. end-to-end loss: gradient flows, every stat key present ----------- #
  logits = torch.randn(4, 12, 257, requires_grad=True)
  probs = torch.softmax(logits, dim=-1)
  loss, stats = mmd_reward.mmd_reinforce_loss(
    probs, tokens, token_selected, critic, group_size=4, reward_weight=1.0)
  loss.backward()
  check('loss is differentiable w.r.t. the student', logits.grad is not None
        and bool(torch.isfinite(logits.grad).all()))
  check('every STAT_KEYS entry is logged',
        all(k in stats for k in mmd_reward.STAT_KEYS), f'{sorted(stats)}')
  check('zero_stats covers the same keys',
        set(mmd_reward.zero_stats()) == set(mmd_reward.STAT_KEYS))

  # -- 8. the silent no-gradient combination is rejected -------------------- #
  critic.kernel_batch_size = 4
  ok = False
  try:
    mmd_reward.mmd_reinforce_loss(probs, tokens, token_selected, critic,
                                  group_size=1, baseline='batch')
  except ValueError as e:
    ok = 'train on nothing' in str(e)
  check('baseline=batch at rows_of == batch is rejected', ok)
  critic.kernel_batch_size = 1

  print()
  if FAILURES:
    print(f'{len(FAILURES)} FAILED: {", ".join(FAILURES)}')
    return 1
  print('all checks passed')
  return 0


if __name__ == '__main__':
  sys.exit(main())
