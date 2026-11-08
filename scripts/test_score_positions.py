"""Checks that `mmd_score_positions` widens SCORING without widening the
gradient: the student only wrote the masked span, so the REINFORCE surrogate
must keep summing log-probabilities over exactly that span whatever the critic
is asked to look at."""
import sys, torch
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmd_reward

ok = []
def check(n, c): ok.append((n, bool(c))); print(f"{'PASS' if c else 'FAIL'}  {n}")

B, L, V = 4, 12, 20
torch.manual_seed(0)
written = torch.zeros(B, L, dtype=torch.bool); written[:, 4:7] = True
content = torch.ones(B, L, dtype=torch.bool)
x0 = torch.randint(0, V, (B, L))

class SpyCritic:
    resolved_layers = [0]
    def __init__(self): self.seen = None
    def paired_reward(self, clean, gen, selected, eos_id, return_per_layer=False):
        self.seen = selected.clone()
        # Must VARY across group members, or the leave-one-out advantage is
        # identically zero and nothing has a gradient at all.
        r = -(gen.float() * selected.float()).sum(-1) * 0.01
        return r, torch.ones(B, dtype=torch.bool)

def run(scored):
    probs = torch.full((B, L, V), 1.0 / V, requires_grad=True)
    critic = SpyCritic()
    loss, stats = mmd_reward.mmd_reinforce_loss(
        probs, x0, written, critic, group_size=2, eos_id=None,
        reward_weight=1.0, baseline='group', scored=scored)
    loss.backward()
    return critic.seen, probs.grad

seen, grad = run(None)
check('scored=None -> critic sees the written span', bool((seen == written).all()))
g_written = grad.abs().sum(-1)
# Outside the written span the gradient must be EXACTLY zero. Inside it, a
# position can still be zero when both group members drew the same token
# there: at K=2 the advantages are equal and opposite, so identical draws
# cancel. So require zero outside, and some signal inside.
check('gradient is exactly zero outside the written span',
      bool((g_written[:, ~written[0]] == 0).all()))
check('some written positions carry gradient',
      bool((g_written[:, written[0]] > 0).any()))

seen_all, grad_all = run(content)
check("scored='all' -> critic sees every content position",
      bool((seen_all == content).all()))
g_all = grad_all.abs().sum(-1)
check('widening `scored` leaves the gradient exactly zero outside written',
      bool((g_all[:, ~written[0]] == 0).all()))
check('and still carries gradient inside written',
      bool((g_all[:, written[0]] > 0).any()))
check('a wider scored span changes the reward the critic returns',
      int(seen.sum()) != int(seen_all.sum()))

try:
    run(torch.ones(B, L + 1, dtype=torch.bool))
    check('shape mismatch raises', False)
except ValueError:
    check('shape mismatch raises', True)

print(f"\n{sum(v for _, v in ok)}/{len(ok)} passed")
sys.exit(0 if all(v for _, v in ok) else 1)
