"""Tests for the MMD feature map (`mmd_reward.FrozenBackboneFeatures`).

CPU-only and checkpoint-free. The backbone is stubbed, which is not merely a
convenience: this repo's real DiT calls flash-attn's rotary kernel, and that has
no CPU backend at all (`ValueError: Cannot find backend for cpu`), so the real
module cannot be exercised without a GPU. What is under test here is the reward
arithmetic and the row handling, both of which are pure tensor code.

    python scripts/test_mmd_ar_features.py

The properties, in order:
  1. identical rows score 0, and the reward falls monotonically as more of the
     generated answer is corrupted;
  2. only `selected` positions move the reward;
  3. `noise_level=0` -- the setting the AR feature map depends on -- passes both
     rows through byte-identical, while `noise_level>0` masks them *the same
     way* in both rows;
  4. the student's pad and mask ids reach the backbone unremapped, which is what
     makes S-FLM's 49154-row embedding table usable directly.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mmd_reward  # noqa: E402

HIDDEN = 16
MASK_INDEX = 49153
PAD_INDEX = 49152


class _ScaleBlock(torch.nn.Module):
  """A block that visibly transforms its input, so blocks differ from each other."""

  def __init__(self, scale):
    super().__init__()
    self.scale = scale

  def forward(self, hidden):
    return torch.tanh(hidden * self.scale)


class StubBackbone(torch.nn.Module):
  """A DiT-shaped stand-in: `.blocks`, called as `(tokens, sigma)`.

  Each token id maps to a fixed pseudo-random feature vector, so two rows agree
  exactly where their tokens agree -- which is the only property the reward
  arithmetic depends on. Records what it was called with, so the tests can
  assert on the rows the feature map actually built.
  """

  def __init__(self, n_blocks=4, vocab=49154):
    super().__init__()
    # Distinct per block: a shared Identity would make every layer's features
    # equal, which would let a broken multi-block sum pass unnoticed.
    self.blocks = torch.nn.ModuleList(
      _ScaleBlock(1.0 + index) for index in range(n_blocks))
    generator = torch.Generator().manual_seed(0)
    self.table = torch.randn(vocab, HIDDEN, generator=generator)
    self.seen = []

  def forward(self, tokens, sigma):
    self.seen.append(tokens.clone())
    hidden = self.table[tokens]
    for block in self.blocks:
      hidden = block(hidden)
    return hidden


def _run(name, fn):
  fn()
  print(f'PASS  {name}')


# --------------------------------------------------------------------------- #

def test_identical_rows_score_zero_and_corruption_hurts():
  backbone = StubBackbone()
  critic = mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX,
                                             noise_level=0.0)
  clean = torch.randint(0, 4000, (2, 12))
  selected = torch.zeros(2, 12, dtype=torch.bool)
  selected[:, 4:10] = True

  reward, active = critic.paired_reward(clean, clean.clone(), selected,
                                        eos_id=0)
  assert bool(active.all())
  assert torch.allclose(reward, torch.zeros_like(reward), atol=1e-6), reward

  # Corrupting more of the scored span must not make the reward go up.
  previous = reward
  for n in (1, 3, 6):
    generated = clean.clone()
    generated[:, 4:4 + n] = 4321
    worse, _ = critic.paired_reward(clean, generated, selected, eos_id=0)
    assert bool((worse <= previous + 1e-6).all()), (n, worse, previous)
    assert bool((worse < 0).all()), (n, worse)
    previous = worse


def test_only_selected_positions_count():
  backbone = StubBackbone()
  critic = mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX,
                                             noise_level=0.0)
  clean = torch.randint(0, 4000, (2, 12))
  selected = torch.zeros(2, 12, dtype=torch.bool)
  selected[:, 4:10] = True

  # A position the student did not write must not enter the reward at all.
  generated = clean.clone()
  generated[:, 0] = 4321
  reward, _ = critic.paired_reward(clean, generated, selected, eos_id=0)
  assert torch.allclose(reward, torch.zeros_like(reward), atol=1e-6), reward

  # A row selecting nothing is reported inactive rather than scored.
  empty = torch.zeros(2, 12, dtype=torch.bool)
  empty[0, 4:10] = True
  _, active = critic.paired_reward(clean, generated, empty, eos_id=0)
  assert active.tolist() == [True, False], active


def test_noise_level_zero_passes_rows_through_unchanged():
  """The property the AR feature map relies on: no masking at all."""
  backbone = StubBackbone()
  critic = mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX,
                                             noise_level=0.0)
  clean = torch.randint(0, 4000, (2, 12))
  generated = torch.randint(0, 4000, (2, 12))
  selected = torch.ones(2, 12, dtype=torch.bool)

  torch.manual_seed(0)
  critic.paired_reward(clean, generated, selected, eos_id=0)
  stacked = backbone.seen[-1]
  assert torch.equal(stacked[:2], clean), 'clean row was altered'
  assert torch.equal(stacked[2:], generated), 'generated row was altered'
  assert not bool((stacked == MASK_INDEX).any()), 'a mask was introduced'


def test_positive_noise_level_masks_both_rows_identically():
  """The denoiser path: masked, but paired -- the same positions in both rows."""
  backbone = StubBackbone()
  critic = mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX,
                                             noise_level=0.5)
  clean = torch.randint(0, 4000, (2, 12))
  generated = torch.randint(0, 4000, (2, 12))
  selected = torch.ones(2, 12, dtype=torch.bool)

  torch.manual_seed(0)
  critic.paired_reward(clean, generated, selected, eos_id=0)
  stacked = backbone.seen[-1]
  clean_row, gen_row = stacked[:2], stacked[2:]
  assert bool((clean_row == MASK_INDEX).any()), 'noise_level=0.5 masked nothing'
  assert torch.equal(clean_row == MASK_INDEX, gen_row == MASK_INDEX), (
    'the two rows were masked differently, which would add a difference that '
    'has nothing to do with what the student wrote')


def test_student_pad_and_mask_ids_are_not_remapped():
  """S-FLM's AR has 49154 embeddings, so both ids index it directly.

  The HF path has to remap them onto EOS because SmolLM-135M stops at 49152;
  this path must not, or the features would describe a different sequence.
  """
  backbone = StubBackbone()
  critic = mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX,
                                             noise_level=0.0)
  clean = torch.full((1, 6), PAD_INDEX, dtype=torch.long)
  clean[0, :3] = torch.tensor([7, 8, 9])
  generated = clean.clone()
  generated[0, 5] = MASK_INDEX
  selected = torch.zeros(1, 6, dtype=torch.bool)
  selected[0, :3] = True

  critic.paired_reward(clean, generated, selected, eos_id=0)
  stacked = backbone.seen[-1]
  assert int(stacked.max()) == MASK_INDEX, (
    f'ids were remapped; max reaching the backbone was {int(stacked.max())}')
  assert bool((stacked == PAD_INDEX).any())


def test_middle_block_is_the_default():
  backbone = StubBackbone(n_blocks=12)
  critic = mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX,
                                             noise_level=0.0)
  assert critic.resolved_layer == 6, critic.resolved_layer
  explicit = mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX,
                                               layer_index=2, noise_level=0.0)
  assert explicit.resolved_layer == 2
  try:
    mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX, layer_index=99)
  except ValueError:
    pass
  else:
    raise AssertionError('an out-of-range feature layer was accepted')


def test_summed_layers_equal_the_sum_of_single_layer_rewards():
  """A list of blocks must reproduce, exactly, the terms taken one at a time."""
  backbone = StubBackbone(n_blocks=12)
  clean = torch.tensor([[3, 4, 5, 6, 7, 8]])
  generated = torch.tensor([[3, 40, 50, 6, 7, 8]])
  selected = torch.zeros(1, 6, dtype=torch.bool)
  selected[0, 1:3] = True

  singles = []
  for index in (2, 9):
    critic = mmd_reward.FrozenBackboneFeatures(
      backbone, MASK_INDEX, layer_index=index, noise_level=0.0)
    singles.append(critic.paired_reward(clean, generated, selected, eos_id=0)[0])

  summed = mmd_reward.FrozenBackboneFeatures(
    backbone, MASK_INDEX, layer_index=[2, 9], noise_level=0.0)
  assert summed.resolved_layers == [2, 9], summed.resolved_layers
  reward, _, per_layer = summed.paired_reward(
    clean, generated, selected, eos_id=0, return_per_layer=True)
  torch.testing.assert_close(reward, singles[0] + singles[1])
  torch.testing.assert_close(per_layer[2], singles[0])
  torch.testing.assert_close(per_layer[9], singles[1])

  weighted = mmd_reward.FrozenBackboneFeatures(
    backbone, MASK_INDEX, layer_index=[2, 9], layer_weights=[3.0, 0.5],
    noise_level=0.0)
  torch.testing.assert_close(
    weighted.paired_reward(clean, generated, selected, eos_id=0)[0],
    3.0 * singles[0] + 0.5 * singles[1])


def test_one_forward_serves_every_requested_block():
  """Reading N blocks must not cost N forwards -- that is the whole point."""
  backbone = StubBackbone(n_blocks=12)
  critic = mmd_reward.FrozenBackboneFeatures(
    backbone, MASK_INDEX, layer_index=[0, 5, 11], noise_level=0.0)
  clean = torch.tensor([[3, 4, 5, 6]])
  selected = torch.ones(1, 4, dtype=torch.bool)
  before = len(backbone.seen)
  critic.paired_reward(clean, clean.clone(), selected, eos_id=0)
  assert len(backbone.seen) - before == 1, len(backbone.seen) - before


def test_bad_layer_lists_are_rejected():
  backbone = StubBackbone(n_blocks=12)
  for bad, why in (([], 'empty list'),
                   ([2, 2], 'duplicate block'),
                   ([2, 99], 'out-of-range entry'),
                   ('6', 'a string')):
    try:
      mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX, layer_index=bad)
    except ValueError:
      continue
    raise AssertionError(f'{why} was accepted: {bad!r}')

  try:
    mmd_reward.FrozenBackboneFeatures(backbone, MASK_INDEX, layer_index=[2, 9],
                                      layer_weights=[1.0])
  except ValueError:
    pass
  else:
    raise AssertionError('a weight list of the wrong length was accepted')


def test_summed_layers_log_a_per_block_breakdown():
  """A summed reward must say which block is driving it -- the terms differ ~400x."""
  backbone = StubBackbone(n_blocks=12)
  torch.manual_seed(0)
  batch, length, vocab = 2, 6, 32
  probs = torch.rand(batch, length, vocab).softmax(-1).requires_grad_(True)
  x0 = torch.randint(0, vocab, (batch, length))
  selected = torch.ones(batch, length, dtype=torch.bool)

  single = mmd_reward.FrozenBackboneFeatures(
    backbone, MASK_INDEX, layer_index=9, noise_level=0.0)
  _, stats = mmd_reward.mmd_reinforce_loss(
    probs, x0, selected, single, group_size=3, eos_id=0)
  assert not [k for k in stats if 'reward_l' in k], sorted(stats)

  summed = mmd_reward.FrozenBackboneFeatures(
    backbone, MASK_INDEX, layer_index=[2, 9], noise_level=0.0)
  loss, stats = mmd_reward.mmd_reinforce_loss(
    probs, x0, selected, summed, group_size=3, eos_id=0)
  for key in ('mmd/reward_l2', 'mmd/reward_l9',
              'mmd/reward_spread_l2', 'mmd/reward_spread_l9'):
    assert key in stats, f'{key} missing from {sorted(stats)}'
  assert loss.requires_grad, 'the surrogate lost its gradient'


def test_multi_draw_reward_uses_one_state_and_keeps_layer_metrics():
  backbone = StubBackbone(n_blocks=4)
  critic = mmd_reward.FrozenBackboneFeatures(
    backbone, MASK_INDEX, layer_index=[1, 3], noise_level=0.0)
  critic.kernel = 'rbf'
  critic.kernel_level = 'sentence'
  critic.rbf_alpha = 0.2

  torch.manual_seed(13)
  batch, length, draw_count, vocab = 3, 8, 4, 32
  clean = torch.randint(0, vocab, (batch, length))
  draws = clean.unsqueeze(0).repeat(draw_count, 1, 1)
  draws[:, :, 2:6] = torch.randint(
    0, vocab, (draw_count, batch, 4))
  selected = torch.zeros(batch, length, dtype=torch.bool)
  selected[:, 2:6] = True

  reward, active, per_layer = critic.multi_draw_reward(
    clean, draws, selected, eos_id=0, return_per_layer=True)
  assert reward.shape == (batch,)
  assert active.all()
  assert set(per_layer) == {1, 3}
  assert torch.isfinite(reward).all()
  assert backbone.seen[-1].shape == (
    (draw_count + 1) * batch, length)

  # Exercise the full G x K REINFORCE route and ensure the multi-layer
  # diagnostics survive it.
  critic.kernel = 'linear'
  probs = torch.rand(batch, length, vocab).softmax(-1).requires_grad_(True)
  loss, stats = mmd_reward.mmd_reinforce_loss(
    probs, clean, selected, critic, group_size=2, eos_id=0,
    kernel_source='draws', kernel_draws=draw_count)
  loss.backward()
  for key in ('mmd/reward_l1', 'mmd/reward_l3',
              'mmd/reward_spread_l1', 'mmd/reward_spread_l3'):
    assert key in stats, f'{key} missing from {sorted(stats)}'
  assert bool((probs.grad[~selected] == 0).all())


def test_multi_draw_reward_shares_diffusion_feature_mask():
  backbone = StubBackbone(n_blocks=2)
  critic = mmd_reward.FrozenBackboneFeatures(
    backbone, MASK_INDEX, layer_index=1, noise_level=0.5)
  torch.manual_seed(21)
  batch, length, draw_count, vocab = 2, 10, 3, 32
  clean = torch.randint(0, vocab, (batch, length))
  draws = clean.unsqueeze(0).repeat(draw_count, 1, 1)
  selected = torch.zeros(batch, length, dtype=torch.bool)
  selected[:, 2:8] = True

  critic.multi_draw_reward(clean, draws, selected, eos_id=0)
  seen = backbone.seen[-1]
  seen_clean = seen[:batch]
  seen_draws = seen[batch:].reshape(draw_count, batch, length)
  for draw in seen_draws:
    assert torch.equal(draw, seen_clean)


def test_one_linear_draw_matches_the_paired_reward():
  backbone = StubBackbone(n_blocks=4)
  critic = mmd_reward.FrozenBackboneFeatures(
    backbone, MASK_INDEX, layer_index=[1, 3], noise_level=0.0)
  critic.kernel = 'linear'
  torch.manual_seed(34)
  batch, length, vocab = 3, 9, 32
  clean = torch.randint(0, vocab, (batch, length))
  generated = torch.randint(0, vocab, (batch, length))
  selected = torch.zeros(batch, length, dtype=torch.bool)
  selected[0, 1:8] = True
  selected[1, 2:7] = True
  selected[2, 3:6] = True

  paired = critic.paired_reward(
    clean, generated, selected, eos_id=0, return_per_layer=True)
  drawn = critic.multi_draw_reward(
    clean, generated.unsqueeze(0), selected, eos_id=0,
    return_per_layer=True)
  torch.testing.assert_close(drawn[0], paired[0], rtol=0, atol=0)
  assert torch.equal(drawn[1], paired[1])
  for layer in paired[2]:
    torch.testing.assert_close(drawn[2][layer], paired[2][layer], rtol=0, atol=0)


def test_missing_checkpoint_points_at_the_fetch_script():
  try:
    mmd_reward.load_ar_backbone('/nonexistent/ar_notok.ckpt', None, None, 'cpu')
  except FileNotFoundError as e:
    assert 'fetch_sflm_ar.sh' in str(e), str(e)
  else:
    raise AssertionError('a missing checkpoint did not raise')


def main():
  torch.manual_seed(0)
  for name, fn in sorted(globals().items()):
    if name.startswith('test_'):
      _run(name, fn)
  print('\nall MMD feature-map tests passed')


if __name__ == '__main__':
  main()
