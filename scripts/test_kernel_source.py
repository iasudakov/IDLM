"""`mmd_kernel_source=draws`: the kernel batch is K samples at ONE state.

Each REINFORCE candidate becomes a set of K draws pooled into one MMD estimate
instead of a single sample, so `mmd_group_size` still supplies the
leave-one-out spread while the estimate itself sharpens with K. The invariant
that matters is the same as everywhere else: widening what the critic sees must
not widen what the gradient reaches.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmd_reward
ok=[]
def chk(n,c): ok.append((n,bool(c))); print(f"{'PASS' if c else 'FAIL'}  {n}")

B, L, V, D = 4, 24, 30, 5
G, K = 3, 4
torch.manual_seed(0)
written = torch.zeros(B, L, dtype=torch.bool); written[:, 6:18] = True
x0 = torch.randint(0, V, (B, L))

class Spy:
    resolved_layers=[0]
    def __init__(self): self.calls=0; self.shapes=[]
    def multi_draw_reward(self, clean, draws, scored, eos):
        self.calls += 1; self.shapes.append(tuple(draws.shape))
        # vary with the draws so the LOO advantage is non-zero
        r = -(draws.float().mean(0) * scored.float()).sum(-1) * 0.01
        return r, torch.ones(B, dtype=torch.bool)

probs = torch.full((B, L, V), 1.0/V, requires_grad=True)
spy = Spy()
loss, stats = mmd_reward.mmd_reinforce_loss(
    probs, x0, written, spy, group_size=G, eos_id=None, reward_weight=1.0,
    baseline='group', kernel_source='draws', kernel_draws=K)
loss.backward()
g = probs.grad.abs().sum(-1)

chk(f'critic called once per candidate ({spy.calls} == G={G})', spy.calls == G)
chk(f'each call sees K draws {spy.shapes[0]} == {(K,B,L)}', spy.shapes[0] == (K,B,L))
chk('gradient exactly zero outside the written span',
    bool((g[:, ~written[0]] == 0).all()))
chk('written positions carry gradient', bool((g[:, written[0]] > 0).any()))
chk('reward_spread reported over the group', float(stats['mmd/reward_spread']) > 0)

# rows mode must be untouched
class Spy2:
    resolved_layers=[0]
    def __init__(self): self.calls=0
    def paired_reward(self, clean, gen, scored, eos, return_per_layer=False):
        self.calls += 1
        return -(gen.float()*scored.float()).sum(-1)*0.01, torch.ones(B, dtype=torch.bool)
p2 = torch.full((B, L, V), 1.0/V, requires_grad=True)
s2 = Spy2()
l2, _ = mmd_reward.mmd_reinforce_loss(p2, x0, written, s2, group_size=G,
                                      eos_id=None, baseline='group')
l2.backward()
chk(f'rows mode still calls paired_reward G times ({s2.calls})', s2.calls == G)

for bad, msg in ((0, 'must be >= 1'), (None, None)):
    pass
try:
    mmd_reward.mmd_reinforce_loss(probs, x0, written, Spy2(), group_size=G,
                                  eos_id=None, kernel_source='draws', kernel_draws=K)
    chk('draws mode rejects a critic without multi_draw_reward', False)
except ValueError as e:
    chk('draws mode rejects a critic without multi_draw_reward',
        'multi_draw_reward' in str(e))
try:
    mmd_reward.mmd_reinforce_loss(probs, x0, written, spy, group_size=G,
                                  eos_id=None, kernel_source='bogus')
    chk('bad kernel_source raises', False)
except ValueError as e:
    chk('bad kernel_source raises', 'rows|draws' in str(e))

print(f"\n{sum(v for _,v in ok)}/{len(ok)} passed")
sys.exit(0 if all(v for _,v in ok) else 1)
