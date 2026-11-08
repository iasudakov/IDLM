"""`mmd_kernel_source=draws`: the kernel batch is K samples at ONE state.

Each REINFORCE candidate becomes a set of K draws pooled into one MMD estimate
instead of a single sample, so `mmd_group_size` still supplies the
leave-one-out spread while the estimate itself sharpens with K. The invariant
that matters is the same as everywhere else: widening what the critic sees must
not widen what the gradient reaches.
"""
import math
import os
import sys
from unittest import mock

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
    kernel_level='token'
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

# Pin two invariants that are easy to lose in this dispatch: draws with K=1
# must still use multi_draw_reward (the sentence estimator's biased fallback),
# and the joint K-draw log probability must be normalized by K without changing
# OWT's historical 1/G normalization.
class IndexedMultiReward(Spy):
    kernel_level='sentence'
    def multi_draw_reward(self, clean, draws, scored, eos):
        del clean, scored, eos
        self.calls += 1; self.shapes.append(tuple(draws.shape))
        # The deterministic sampler below repeats one token within a candidate,
        # so this reward is independent of K.
        return draws[:, :, 0].float().mean(0), torch.ones(
            draws.shape[1], dtype=torch.bool)

def run_k(draw_count):
    group, batch, length, vocab = 3, 2, 4, 5
    candidate_ids = torch.arange(group).repeat_interleave(draw_count)
    def draw_candidates(flat_probs, num_samples, replacement=False, **kwargs):
        del kwargs
        assert replacement and num_samples == group * draw_count
        return candidate_ids.to(flat_probs.device).expand(flat_probs.shape[0], -1)
    candidate_probs = torch.full(
        (batch, length, vocab), 1.0 / vocab, requires_grad=True)
    clean = torch.zeros(batch, length, dtype=torch.long)
    scored = torch.ones(batch, length, dtype=torch.bool)
    critic = IndexedMultiReward()
    with mock.patch.object(torch, 'multinomial', draw_candidates):
        candidate_loss, _ = mmd_reward.mmd_reinforce_loss(
            candidate_probs, clean, scored, critic, group_size=group,
            eos_id=None, baseline='group', kernel_source='draws',
            kernel_draws=draw_count)
    candidate_loss.backward()
    return candidate_loss.detach(), candidate_probs.grad.detach(), critic

k1 = run_k(1)
k2 = run_k(2)
k4 = run_k(4)
for wider in (k2, k4):
    torch.testing.assert_close(wider[0], k1[0], rtol=0, atol=0)
    torch.testing.assert_close(wider[1], k1[1], rtol=0, atol=0)
chk('draws K=1 uses one multi-draw call per candidate',
    k1[2].calls == 3 and k1[2].shapes == [(1, 2, 4)] * 3)
chk('sentence-level draws accepts K=2',
    k2[2].calls == 3 and k2[2].shapes == [(2, 2, 4)] * 3)
chk('1/K normalization cancels duplicated draw log-probabilities', True)
chk('sentence draws preserve OWT 1/G normalization',
    torch.allclose(k2[1][..., 0], torch.full((2, 4), 0.3125))
    and torch.allclose(k2[1][..., 2], torch.full((2, 4), -0.3125)))

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

# Token-level RBF over K identical copies of {-1, 1} has the exact reward
# K * (1 - exp(-4)) / (2K - 1). It must remain positive, including at K=1.
points = torch.tensor([[[-1.0], [1.0]],
                       [[-1.0], [1.0]],
                       [[-1.0], [1.0]]])
mask = torch.tensor([[True, True], [False, False], [True, False]])
for draw_count in (1, 2, 4):
    expected = draw_count * (1.0 - math.exp(-4.0)) / (2 * draw_count - 1)
    for block_size in (0, 1):
        reward, active, per_layer = mmd_reward.summed_multi_draw_reward(
            {0: points.repeat(draw_count + 1, 1, 1)}, batch=3,
            draw_count=draw_count, length=2, selected=mask,
            indices=[0], weights=[1.0], kernel='rbf', alpha=1.0,
            block_size=block_size, level='token')
        torch.testing.assert_close(reward, torch.tensor([expected, 0.0, 0.0]))
        torch.testing.assert_close(per_layer[0], reward)
        chk(f'token draws preserve positive reward and inactive rows, '
            f'K={draw_count}, block={block_size}',
            active.tolist() == [True, False, False])

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
