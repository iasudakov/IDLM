"""Checks that `mmd_score_positions` widens SCORING without widening the
gradient: the student only wrote the masked span, so the REINFORCE surrogate
must keep summing log-probabilities over exactly that span whatever the critic
is asked to look at.

Note the sibling knob `mmd_scoreD_positions`, which picks the CONTENT set
(answer vs non_prompt). This one operates inside that choice.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmd_reward

ok = []


def check(name, cond):
  ok.append((name, bool(cond)))
  print(f"{'PASS' if cond else 'FAIL'}  {name}")


B, L, V = 4, 12, 20
torch.manual_seed(0)
# content = everything but a 2-token "prompt"; written = the masked subset
content = torch.zeros(B, L, dtype=torch.bool); content[:, 2:] = True
written = torch.zeros(B, L, dtype=torch.bool); written[:, 4:7] = True
x0 = torch.randint(0, V, (B, L))


class SpyCritic:
  resolved_layers = [0]

  def __init__(self):
    self.seen = None

  def paired_reward(self, clean, gen, selected, eos_id, return_per_layer=False):
    self.seen = selected.clone()
    # Must vary across group members, or the leave-one-out advantage is
    # identically zero and nothing carries a gradient at all.
    r = -(gen.float() * selected.float()).sum(-1) * 0.01
    return r, torch.ones(B, dtype=torch.bool)


def run(scored):
  probs = torch.full((B, L, V), 1.0 / V, requires_grad=True)
  critic = SpyCritic()
  loss, _ = mmd_reward.mmd_reinforce_loss(
    probs, x0, written, critic, group_size=2, eos_id=None,
    reward_weight=1.0, scored=scored)
  loss.backward()
  return critic.seen, probs.grad.abs().sum(-1)


seen, grad = run(None)
check('scored=None -> critic sees the written span', bool((seen == written).all()))
check('gradient is exactly zero outside the written span',
      bool((grad[:, ~written[0]] == 0).all()))
check('some written positions carry gradient',
      bool((grad[:, written[0]] > 0).any()))

seen_all, grad_all = run(content)
check("scored=content -> critic sees the whole content set",
      bool((seen_all == content).all()))
check('widening `scored` leaves the gradient exactly zero outside written',
      bool((grad_all[:, ~written[0]] == 0).all()))
check('and still carries gradient inside written',
      bool((grad_all[:, written[0]] > 0).any()))
check('the prompt is never scored, under either setting',
      not bool(seen[:, :2].any()) and not bool(seen_all[:, :2].any()))
check('a wider scored span reaches more positions',
      int(seen_all.sum()) > int(seen.sum()))

try:
  run(torch.ones(B, L + 1, dtype=torch.bool))
  check('shape mismatch raises', False)
except ValueError:
  check('shape mismatch raises', True)

print(f"\n{sum(v for _, v in ok)}/{len(ok)} passed")
sys.exit(0 if all(v for _, v in ok) else 1)
