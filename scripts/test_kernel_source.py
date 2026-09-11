"""CPU regression tests for same-state MMD kernel batches.

Run from the repository root:

    python scripts/test_kernel_source.py
"""

import math
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmd_rbf
import mmd_reward


checks = []


def check(name, condition):
  checks.append((name, bool(condition)))
  print(f'{"PASS" if condition else "FAIL"}  {name}')


BATCH, LENGTH, VOCAB = 3, 12, 20
GROUP, DRAWS = 3, 4
torch.manual_seed(0)
x0 = torch.randint(0, VOCAB, (BATCH, LENGTH))
written = torch.zeros(BATCH, LENGTH, dtype=torch.bool)
written[:, 3:9] = True


class MultiDrawSpy:
  resolved_layers = [0]
  kernel_level = 'token'

  def __init__(self):
    self.calls = 0
    self.shapes = []
    self.scored = []

  def multi_draw_reward(self, clean, draws, scored, eos_id):
    del clean, eos_id
    self.calls += 1
    self.shapes.append(tuple(draws.shape))
    self.scored.append(scored.clone())
    reward = -(draws.float().mean(0) * scored.float()).sum(-1) * 0.01
    return reward, scored.any(dim=1)


probs = torch.full(
  (BATCH, LENGTH, VOCAB), 1.0 / VOCAB, requires_grad=True)
spy = MultiDrawSpy()
loss, stats = mmd_reward.mmd_reinforce_loss(
  probs, x0, written, spy, group_size=GROUP, eos_id=None,
  reward_weight=1.0, baseline='group', kernel_source='draws',
  kernel_draws=DRAWS)
loss.backward()
gradient = probs.grad.abs().sum(-1)

check(f'one critic call per candidate ({spy.calls} == {GROUP})',
      spy.calls == GROUP)
check(f'each candidate contains K draws {spy.shapes[0]}',
      spy.shapes == [(DRAWS, BATCH, LENGTH)] * GROUP)
check('K need not divide the device batch in draws mode', BATCH % DRAWS != 0)
check('critic receives the configured scored mask',
      all(torch.equal(mask, written) for mask in spy.scored))
check('gradient is zero outside the written span',
      bool((gradient[~written] == 0).all()))
check('written positions receive gradient', bool((gradient[written] > 0).any()))
check('outer group still supplies reward spread',
      float(stats['mmd/reward_spread']) > 0)


class PairedSpy:
  resolved_layers = [0]

  def paired_reward(self, clean, generated, selected, eos_id,
                    return_per_layer=False):
    del clean, eos_id, return_per_layer
    reward = -(generated.float() * selected.float()).sum(-1) * 0.01
    return reward, selected.any(dim=1)


def run_rows(explicit):
  row_probs = torch.full(
    (BATCH, LENGTH, VOCAB), 1.0 / VOCAB, requires_grad=True)
  kwargs = ({'kernel_source': 'rows', 'kernel_draws': DRAWS}
            if explicit else {})
  torch.manual_seed(17)
  row_loss, row_stats = mmd_reward.mmd_reinforce_loss(
    row_probs, x0, written, PairedSpy(), group_size=GROUP, eos_id=None,
    reward_weight=1.0, baseline='group', **kwargs)
  row_loss.backward()
  return row_loss.detach(), row_probs.grad.detach(), row_stats


implicit_rows = run_rows(False)
explicit_rows = run_rows(True)
torch.testing.assert_close(explicit_rows[0], implicit_rows[0], rtol=0, atol=0)
torch.testing.assert_close(explicit_rows[1], implicit_rows[1], rtol=0, atol=0)
check('explicit rows mode is numerically identical to the legacy default', True)


class IndexedMultiReward(MultiDrawSpy):
  kernel_level = 'sentence'

  def multi_draw_reward(self, clean, draws, scored, eos_id):
    del clean, scored, eos_id
    # The deterministic sampler below repeats one token for every member of a
    # candidate, so this reward is independent of K.
    reward = draws[:, :, 0].float().mean(0)
    return reward, torch.ones(draws.shape[1], dtype=torch.bool)


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
  selected = torch.ones(batch, length, dtype=torch.bool)
  with mock.patch.object(torch, 'multinomial', draw_candidates):
    candidate_loss, _ = mmd_reward.mmd_reinforce_loss(
      candidate_probs, clean, selected, IndexedMultiReward(),
      group_size=group, eos_id=None, baseline='group',
      kernel_source='draws', kernel_draws=draw_count)
  candidate_loss.backward()
  return candidate_loss.detach(), candidate_probs.grad.detach()


k1 = run_k(1)
k2 = run_k(2)
k4 = run_k(4)
torch.testing.assert_close(k2[0], k1[0], rtol=0, atol=0)
torch.testing.assert_close(k2[1], k1[1], rtol=0, atol=0)
torch.testing.assert_close(k4[0], k1[0], rtol=0, atol=0)
torch.testing.assert_close(k4[1], k1[1], rtol=0, atol=0)
check('1/K normalization cancels duplicated sentence-draw log-probabilities',
      True)
check('sentence draws preserve TinyGSM omission of 1/G',
      torch.allclose(k2[1][..., 0], torch.full((2, 4), 0.9375))
      and torch.allclose(k2[1][..., 2], torch.full((2, 4), -0.9375)))


def draw_group_indices(flat_probs, num_samples, replacement=False, **kwargs):
  del kwargs
  assert replacement and num_samples == 3
  return torch.arange(3, device=flat_probs.device).expand(flat_probs.shape[0], -1)


class IndexedPairReward:
  resolved_layers = [0]

  def paired_reward(self, clean, generated, selected, eos_id,
                    return_per_layer=False):
    del clean, eos_id, return_per_layer
    return generated[:, 0].float(), selected.any(dim=1)


legacy_probs = torch.full((2, 4, 5), 0.2, requires_grad=True)
with mock.patch.object(torch, 'multinomial', draw_group_indices):
  legacy_loss, _ = mmd_reward.mmd_reinforce_loss(
    legacy_probs, torch.zeros(2, 4, dtype=torch.long),
    torch.ones(2, 4, dtype=torch.bool), IndexedPairReward(), group_size=3,
    eos_id=None, baseline='group')
legacy_loss.backward()
check('legacy TinyGSM omission of 1/G is preserved',
      torch.allclose(legacy_probs.grad[..., 0],
                     torch.full((2, 4), 0.9375))
      and torch.allclose(legacy_probs.grad[..., 2],
                         torch.full((2, 4), -0.9375)))


# The unequal-size blocked RBF implementation must agree with a dense formula.
torch.manual_seed(5)
real = torch.randn(2, 5, 3)
fake = torch.randn(2, 9, 3)
w_real = torch.tensor([[1, 1, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)
w_fake = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
alpha = 0.3
actual, active = mmd_rbf.rbf_mmd2_unequal(
  real, fake, w_real, w_fake, alpha, block_size=3)


def dense_mmd(row):
  x = real[row][w_real[row].bool()]
  y = fake[row][w_fake[row].bool()]
  if len(x) < 2 or len(y) < 2:
    return 0.0
  kxx = torch.exp(-alpha * torch.cdist(x, x).square())
  kyy = torch.exp(-alpha * torch.cdist(y, y).square())
  kxy = torch.exp(-alpha * torch.cdist(x, y).square())
  return ((kxx.sum() - len(x)) / (len(x) * (len(x) - 1))
          + (kyy.sum() - len(y)) / (len(y) * (len(y) - 1))
          - 2 * kxy.mean())


expected = torch.tensor([dense_mmd(0), dense_mmd(1)])
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
check('unequal-size blocked RBF agrees with the dense estimator', True)
check('rows with fewer than two real atoms are inactive',
      active.tolist() == [True, False])


# End-to-end token draws must preserve negative MMD estimates in the reward.
# Real atoms {0,1}, duplicated into K=2 fake draws, have k(0,1)=1/2.
# The real term is 1/2, fake term 2/3, cross term 3/2, so reward=+1/3.
# The other rows exercise the one-atom and empty-mask inactive cases.
unclipped_real = torch.tensor([[[0.0], [1.0], [999.0]],
                               [[0.0], [999.0], [999.0]],
                               [[999.0], [999.0], [999.0]]])
unclipped_selected = torch.tensor([[True, True, False],
                                  [True, False, False],
                                  [False, False, False]])
unclipped_fake = torch.where(
  unclipped_selected.unsqueeze(-1), unclipped_real, -unclipped_real)
unclipped_features = {0: torch.cat(
  [unclipped_real, unclipped_fake, unclipped_fake], dim=0)}
for block in (0, 1):
  reward, active, per_layer = mmd_reward.summed_multi_draw_reward(
    unclipped_features, batch=3, draw_count=2, length=3,
    selected=unclipped_selected, indices=[0], weights=[1.0],
    kernel='rbf', alpha=math.log(2), block_size=block, level='token')
  torch.testing.assert_close(reward, torch.tensor([1 / 3, 0.0, 0.0]))
  torch.testing.assert_close(per_layer[0], reward)
  check(f'token draws keep negative estimates; mask inactive rows (block={block})',
        active.tolist() == [True, False, False])


class StubExternalAR(mmd_reward.FrozenARFeatures):
  def __init__(self):
    self.resolved_layers = [0]
    self.layer_weights = [1.0]
    self.kernel = 'linear'
    self.kernel_level = 'token'
    self.rbf_alpha = None
    self.rbf_block_size = 0
    self.seen_shape = None
    self.seen_mask = None

  def features(self, tokens, eos_id, attention_mask=None):
    del eos_id
    self.seen_shape = tuple(tokens.shape)
    self.seen_mask = attention_mask
    return {0: torch.nn.functional.one_hot(
      tokens.remainder(7), num_classes=7).float()}


external = StubExternalAR()
external_mask = torch.ones(BATCH, LENGTH, dtype=torch.long)
external_reward, external_active = external.multi_draw_reward(
  x0, x0.unsqueeze(0).repeat(DRAWS, 1, 1), written, eos_id=0,
  attention_mask=external_mask)
check('external HF feature critics support draws without loading a model',
      external_reward.shape == (BATCH,) and bool(external_active.all()))
check('external HF attention masks repeat across all K draws',
      external.seen_shape == ((DRAWS + 1) * BATCH, LENGTH)
      and external.seen_mask.shape == ((DRAWS + 1) * BATCH, LENGTH))


for kwargs, expected_message in (
    ({'kernel_source': 'invalid'}, 'rows|draws'),
    ({'kernel_source': 'draws', 'kernel_draws': 0}, 'must be >= 1'),
    ({'kernel_source': 'draws', 'kernel_draws': 2}, 'multi_draw_reward')):
  try:
    mmd_reward.mmd_reinforce_loss(
      probs, x0, written, PairedSpy(), group_size=GROUP, eos_id=None,
      **kwargs)
  except ValueError as error:
    check(f'invalid combination reports {expected_message}',
          expected_message in str(error))
  else:
    check(f'invalid combination reports {expected_message}', False)

sentence_spy = MultiDrawSpy()
sentence_spy.kernel_level = 'sentence'
sentence_loss, _ = mmd_reward.mmd_reinforce_loss(
  probs, x0, written, sentence_spy, group_size=GROUP, eos_id=None,
  kernel_source='draws', kernel_draws=2)
check('sentence + draws reaches one K-draw critic call per candidate',
      sentence_spy.calls == GROUP
      and sentence_spy.shapes == [(2, BATCH, LENGTH)] * GROUP
      and sentence_loss.requires_grad)

external.kernel_level = 'sentence'
external.kernel = 'rbf'
external.rbf_alpha = 0.2
sentence_external_reward, sentence_external_active = external.multi_draw_reward(
  x0, x0.unsqueeze(0).repeat(DRAWS, 1, 1), written, eos_id=0,
  attention_mask=external_mask)
check('external HF feature critics support sentence-level draws',
      sentence_external_reward.shape == (BATCH,)
      and bool(sentence_external_active.all())
      and torch.isfinite(sentence_external_reward).all())


print(f'\n{sum(value for _, value in checks)}/{len(checks)} passed')
sys.exit(0 if all(value for _, value in checks) else 1)
