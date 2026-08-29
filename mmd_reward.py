"""Paired feature-discrepancy reward, ported from the `owt_mmd` branch.

The idea: a *frozen* model is used not as a critic that scores likelihood, but
as a fixed feature map. Run it over the clean data and over the student's
sample, take the hidden states of one block, mean-pool them, and reward the
student for closing the gap between the two pooled features:

    R = -mean_h( sqrt(delta_h^2 + c^2) - c ),   delta = pooled_gen - pooled_clean

That is a pseudo-Huber distance (c = 1e-3): quadratic for small deltas, linear
for large ones, so a single badly-placed coordinate cannot dominate the gradient
the way a squared distance would. The reward drives a REINFORCE step whose
baseline is the leave-one-out mean over a group of samples drawn at the same
state. `mmd_kernel: rbf` swaps the mean comparison for a distributional one.

WHAT CHANGED IN THE PORT (owt_mmd -> CIFAR-10)

* **No AR feature source.** `FrozenARFeatures`, `load_owt_ar_backbone` and the
  vendored `backbone_dit_legacy` existed only to load `kuleshov-group/owt-ar`, a
  GPT-2-vocabulary causal DiT over text. There is no pixel analogue, so the only
  feature map here is the frozen diffusion teacher.

* **The feature map is a UNet, so features are not per token.** On text there is
  one hidden vector per token and the `(batch, length)` mask of "positions the
  student wrote" indexes features directly. The CIFAR bottleneck is
  `(batch, 256, 4, 4)` -- 16 spatial cells against 3072 tokens. The mask is NOT
  downsampled onto those cells: the read point sits after both the 16x16 and the
  middle 4x4 self-attention blocks, so every cell is already a global function of
  all 3072 tokens, and a 4x4 mask would assert a locality the features do not
  have. (It is also degenerate -- with i.i.d. masking, a cell covers 192 tokens,
  so it is selected with probability `1-(1-p)^192`, which is ~1 at any mask rate
  that occurs in training.)

  So the reward compares **whole images**. `selected` keeps exactly the two jobs
  it can still do correctly: deciding `active`, and building the surrogate's
  summed log-probability. Anyone wanting genuine spatial locality must hook an
  *earlier*, pre-attention map, not downsample the mask at the bottleneck.

* **`active` is computed from the token mask**, not from the feature weights.
  With uniform cell weights the feature-level count is always 16, so the
  estimators' own `n >= 2` guard would happily score a row where the student
  wrote nothing.
"""

import contextlib

import torch

import mmd_rbf

# Kept at the `owt_mmd` value so rewards are comparable across the two branches.
PSEUDO_HUBER_C = 0.001


def feature_blocks(core):
  """The frozen model's ordered feature-producing modules, and its kind.

  Returns `(modules, kind)` where `kind` is 'unet' or 'transformer'. The kind
  decides what `mmd_feature_layer: null` means -- see `resolve_feature_layers`.

  Fails loudly rather than silently hooking an unrelated module.
  """
  # UNet first: it has none of the transformer attribute paths, and its
  # bottleneck (`middle_modules`) is the natural read point -- the lowest
  # resolution, where the representation is most about content and least about
  # pixels.
  middle = getattr(core, 'middle_modules', None)
  if middle is not None and len(middle) > 0:
    return middle, 'unet'

  candidates = (
    getattr(getattr(core, 'model', None), 'layers', None),
    getattr(getattr(getattr(core, 'model', None), 'decoder', None), 'layers', None),
    getattr(getattr(core, 'transformer', None), 'h', None),
    # This repo's own DiT, which is what `algo.backbone=dit` builds.
    getattr(core, 'blocks', None),
    getattr(getattr(core, 'backbone', None), 'blocks', None),
  )
  for layers in candidates:
    if layers is not None and len(layers) > 0:
      configured = getattr(getattr(core, 'config', None),
                           'num_hidden_layers', None)
      n = len(layers) if configured is None else int(configured)
      if n < 1 or n > len(layers):
        raise ValueError(
          f'config.num_hidden_layers={n} is incompatible with the '
          f'{len(layers)} discovered blocks')
      return layers[:n], 'transformer'
  raise ValueError(
    'MMD feature extraction could not find feature blocks at '
    'core.middle_modules, core.model.layers, core.transformer.h, or core.blocks')


def resolve_feature_layers(core, layer_index):
  """Resolve `mmd_feature_layer` to an ordered list of blocks and their indices.

  Accepts `None`, a single zero-based integer, or a sequence of them.

  `None` means the LAST block on a UNet and the MIDDLE one on a transformer, and
  the two defaults are picked for the same underlying reason. On a transformer
  the blocks form a ladder: late ones specialize toward the output head and early
  ones barely differ from the embedding, so neither end says much about whether
  two sequences *mean* the same thing. A UNet is U-shaped, not a ladder -- its
  "middle" in list order is the middle of `middle_modules`, which is an
  `AttnBlock`, not a semantic midpoint. Its actual bottleneck is the end of that
  list, which is where the representation is most compressed.

  A sequence asks for a *summed* reward -- one pseudo-Huber term per block. Be
  aware the terms are not on a common scale; pass `mmd_feature_layer_weights` to
  put them on comparable footing.
  """
  layers, kind = feature_blocks(core)
  if layer_index is None:
    indices = [len(layers) - 1] if kind == 'unet' else [len(layers) // 2]
  elif isinstance(layer_index, int) and not isinstance(layer_index, bool):
    indices = [layer_index]
  elif isinstance(layer_index, (str, bytes)):
    raise ValueError('mmd_feature_layer must be null, a zero-based integer, or '
                     f'a list of them, got {layer_index!r}')
  else:
    try:
      indices = [int(v) for v in layer_index]
    except TypeError:
      raise ValueError('mmd_feature_layer must be null, a zero-based integer, '
                       f'or a list of them, got {layer_index!r}')
    if not indices:
      raise ValueError('mmd_feature_layer is an empty list; the reward would '
                       'have no term at all')
    if len(set(indices)) != len(indices):
      raise ValueError(f'mmd_feature_layer={indices} repeats a block, which '
                       'just rescales that block\'s term -- use '
                       'mmd_feature_layer_weights instead')
  for index in indices:
    if index < 0 or index >= len(layers):
      raise ValueError(
        f'mmd_feature_layer={index} is outside [0,{len(layers) - 1}]')
  return [layers[i] for i in indices], indices


def resolve_layer_weights(indices, weights):
  """One weight per resolved block; `None` means an unweighted sum."""
  if weights is None:
    return [1.0] * len(indices)
  if isinstance(weights, (int, float)) and not isinstance(weights, bool):
    weights = [weights]
  values = [float(w) for w in weights]
  if len(values) != len(indices):
    raise ValueError(
      f'mmd_feature_layer_weights has {len(values)} entries but '
      f'mmd_feature_layer resolved to {len(indices)} blocks {indices}')
  if not any(values):
    raise ValueError('mmd_feature_layer_weights are all zero, which leaves the '
                     'reward identically zero')
  return values


@contextlib.contextmanager
def capture_layer_hidden(core, layer_index):
  """Capture the output of each requested block, in one forward.

  Yields `({index: (batch, positions, features)}, indices)`. Only the first call
  through a block is retained, so a second forward cannot quietly hold another
  full activation. Reading several blocks costs one forward and one activation
  per block, not one forward per block.

  A rank-4 `(batch, channels, H, W)` output -- what every UNet module emits -- is
  flattened to `(batch, H*W, channels)`: positions are spatial cells, features
  are channels. Everything downstream then sees the same rank-3 layout a
  transformer block would have produced.
  """
  modules, indices = resolve_feature_layers(core, layer_index)
  captured = {}

  def _make_hook(index):
    def _hook(_module, _inputs, output):
      if index in captured:
        return
      hidden = output[0] if isinstance(output, (tuple, list)) else output
      if hasattr(hidden, 'last_hidden_state'):
        hidden = hidden.last_hidden_state
      if isinstance(hidden, torch.Tensor) and hidden.ndim == 4:
        hidden = hidden.flatten(2).transpose(1, 2)
      if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise RuntimeError(
          'MMD feature hook expected a rank-3 hidden tensor (or a rank-4 '
          f'convolutional one), got {type(hidden).__name__} with shape '
          f'{getattr(hidden, "shape", None)}')
      captured[index] = hidden.detach()
    return _hook

  handles = [module.register_forward_hook(_make_hook(index))
             for module, index in zip(modules, indices)]
  completed = False
  try:
    yield captured, indices
    completed = True
  finally:
    for handle in handles:
      handle.remove()
  if completed and len(captured) != len(indices):
    missing = [i for i in indices if i not in captured]
    raise RuntimeError(
      f'MMD feature hook on block(s) {missing} did not run')


def pseudo_huber_reward(feature_delta):
  """`-mean(sqrt(delta^2 + c^2) - c)` over the final (feature) dimension, in fp32.

  Leading dimensions are preserved, so a `(batch, blocks, hidden)` delta gives a
  `(batch, blocks)` reward. fp32 regardless of the teacher's dtype: the deltas
  are small and a bf16 square would quantise most of them to the same value.

  NOTE the constant does not mean on a UNet what it means on a transformer.
  `c = 1e-3` was chosen against transformer hidden states; UNet ResBlock outputs
  after GroupNorm are O(1)-O(10), so essentially every delta lands in the LINEAR
  regime and `R ~= -mean|delta|`. The "quadratic for small deltas" story above
  describes the text setting, not this one. The reward is still a perfectly good
  distance -- just do not expect `mmd_reward_weight` to carry over from an OWT
  recipe.
  """
  delta = feature_delta.float()
  c = PSEUDO_HUBER_C
  return -(torch.sqrt(delta ** 2 + c ** 2) - c).mean(dim=-1)


def sequence_pooled_delta(clean, generated, selected):
  """Mean-pool both feature streams over `selected`, and subtract.

  Args:
    clean, generated: `(batch, positions, features)` over the same span.
    selected: `(batch, positions)` bool, at FEATURE resolution. A row selecting
      nothing pools to zero and is reported inactive.

  Returns `(feature_delta, active)`, shapes `(batch, features)` and `(batch,)`.
  The returned `active` reflects the feature mask only; callers that know which
  rows the student actually wrote should AND it with their own (see
  `summed_paired_reward`).
  """
  if clean.shape != generated.shape:
    raise ValueError('MMD clean/generated features must have identical shapes, '
                     f'got {tuple(clean.shape)} vs {tuple(generated.shape)}')
  if clean.ndim != 3:
    raise ValueError(f'MMD features must be rank-3, got {tuple(clean.shape)}')
  batch, length, _ = clean.shape
  if selected.shape != (batch, length):
    raise ValueError(f'MMD selected mask must be {(batch, length)}, got '
                     f'{tuple(selected.shape)}')

  selected = selected.bool()
  weight = selected.unsqueeze(-1).to(torch.float32)
  denom = selected.sum(dim=1).clamp_min(1).to(torch.float32).unsqueeze(-1)
  clean_mean = (clean.float() * weight).sum(dim=1) / denom
  gen_mean = (generated.float() * weight).sum(dim=1) / denom
  return gen_mean - clean_mean, selected.any(dim=1)


def median_alpha(feat, selected, max_samples=512, generator=None):
  """`alpha = 1 / (2 * sigma^2)` from the median pairwise squared distance.

  The standard bandwidth heuristic, measured on the REAL branch only, so it never
  needs a gradient and never moves with the student.

  CAVEAT on this branch: it pools every row's positions together, so the median
  is taken over CROSS-IMAGE pairs while `rbf_sequence_reward` runs the estimator
  WITHIN one image. Bottleneck features are largely image-identity features, so
  the cross-image median overestimates the within-row scale, the kernel saturates
  near 1, and the three sums cancel into noise. Pin `mmd_rbf_alpha` rather than
  relying on this when `mmd_kernel: rbf`. The mismatch closes as
  `mmd_kernel_batch_size` approaches the batch size.

  Falls back to `alpha = 1` when the distances degenerate.
  """
  valid = feat[selected.bool()]
  if valid.shape[0] < 2:
    return 1.0
  if valid.shape[0] > max_samples:
    idx = torch.randperm(valid.shape[0], device=valid.device,
                         generator=generator)[:max_samples]
    valid = valid[idx]
  d2 = torch.cdist(valid.float(), valid.float()).pow_(2)
  # Off-diagonal only: the diagonal is exactly 0 and would drag the median down
  # by however many samples were drawn.
  n = d2.shape[0]
  off = d2[~torch.eye(n, dtype=torch.bool, device=d2.device)]
  med = off.median()
  if not torch.isfinite(med) or med <= 0:
    return 1.0
  return float(1.0 / (2.0 * med))


def rbf_sequence_reward(clean, generated, selected, alpha=None, block_size=0):
  """`-MMD^2` per row under a Gaussian kernel, over the scored positions.

  The linear-kernel reward compares the two feature sets by their MEANS, which is
  all a linear kernel can see. An RBF kernel compares the distributions, so a
  sample that matches the clean row on average but not in shape is no longer
  free.

  The unbiased estimator, with the self-pairs removed:

      MMD^2 = (S_xx - n)/(n(n-1)) + (S_yy - n)/(n(n-1)) - 2 S_xy/n^2

  `- n` is exact rather than approximate: the weights are 0/1, so the diagonal
  contributes `sum_i w_i^2 k(x_i, x_i)` = `sum_i w_i` = `n`. **Keeping the
  weights 0/1 is a correctness requirement of this estimator**, which is why the
  feature mask here is all-ones rather than a fractional downsample of the token
  mask.

  ON CIFAR THIS PATH NEEDS CARE. The bottleneck gives n = 16 positions per row,
  where the estimator's null standard deviation is the same order as the entire
  signal, and `clamp_min(0)` below then fires on about half of all rows -- their
  advantage becomes `-mean(rest)`, i.e. REINFORCE pushes DOWN samples that were
  statistically indistinguishable from data. Raise `mmd_kernel_batch_size` (see
  `merge_kernel_rows`) to put more positions behind the statistic, and pin
  `mmd_rbf_alpha`. `mmd_kernel: linear` has neither problem: a 256-dimensional
  mean over 16 cells is a perfectly well-behaved image descriptor.

  Returns `(reward, active)`, matching the linear path.
  """
  weights = selected.to(torch.float32)
  n = weights.sum(dim=1)
  active = n >= 2
  if not bool(active.any()):
    zeros = clean.new_zeros(clean.shape[0], dtype=torch.float32)
    return zeros, active

  real = clean.detach().float()
  fake = generated.detach().float()
  if alpha is None:
    alpha = median_alpha(real, selected)

  s_xx, s_yy, s_xy = mmd_rbf.rbf_pair_sums(
    real, fake, weights, alpha, block_size=block_size)

  safe_n = n.clamp_min(2.0)
  pairs = safe_n * (safe_n - 1.0)
  square = safe_n * safe_n
  mmd2 = ((s_xx - safe_n) / pairs
          + (s_yy - safe_n) / pairs
          - 2.0 * s_xy / square)
  # A sampling estimator of a squared quantity can come out slightly negative;
  # clamping keeps the reward a true distance rather than letting noise read as
  # "better than identical".
  reward = -mmd2.clamp_min(0.0)
  return torch.where(active, reward, torch.zeros_like(reward)), active


def merge_kernel_rows(clean, generated, selected, rows_of):
  """Fold `rows_of` adjacent sequences into one kernel row.

  The ROW is the unit the statistic is computed over, for both kernels: the
  linear kernel pools it to one mean, and the rbf kernel takes its valid
  positions as the sample set. So merging N sequences means the reward stops
  asking "does this sample match its own clean row" and starts asking "does this
  GROUP of samples match that group of clean rows".

  That is a real trade, not a free widening. A row of N sequences can be
  satisfied by samples that swap content between them: two rows whose contents
  are exchanged pool to the same mean and hold the same multiset, so neither
  kernel can tell them apart. What it buys is a statistic with N times the
  positions behind it, which for rbf means N^2 times the pairs -- the reason to
  reach for it here, where one image contributes only 16.

  Returns the folded views; `rows_of=1` returns the inputs untouched.
  """
  if rows_of <= 1:
    return clean, generated, selected
  batch = clean.shape[0]
  if batch % rows_of:
    raise ValueError(
      f'mmd_kernel_batch_size={rows_of} does not divide the batch size '
      f'{batch}; a row would straddle the end of the batch')
  rows = batch // rows_of
  return (clean.reshape(rows, -1, clean.shape[-1]),
          generated.reshape(rows, -1, generated.shape[-1]),
          selected.reshape(rows, -1))


def paired_sequence_reward(clean, generated, selected, kernel='linear',
                           alpha=None, block_size=0, rows_of=1):
  """One reward per sequence; rows selecting nothing score zero.

  `kernel='linear'` pools both sides and scores the difference with
  pseudo-Huber, which is what a linear kernel's MMD reduces to. `kernel='rbf'`
  compares the two feature distributions instead.

  `rows_of > 1` computes the statistic over that many sequences at once (see
  `merge_kernel_rows`) and hands every sequence in a row its row's reward.
  REINFORCE multiplies each sequence's advantage by ITS OWN log-probability, so
  the return has to stay per sequence whatever the row size -- the row only
  changes what the number means, not its shape.
  """
  batch = clean.shape[0]
  clean, generated, selected = merge_kernel_rows(
    clean, generated, selected, rows_of)
  if kernel == 'rbf':
    reward, active = rbf_sequence_reward(clean, generated, selected,
                                         alpha=alpha, block_size=block_size)
  elif kernel == 'linear':
    delta, active = sequence_pooled_delta(clean, generated, selected)
    reward = pseudo_huber_reward(delta)
    reward = torch.where(active, reward, torch.zeros_like(reward))
  else:
    raise ValueError(f'mmd_kernel must be linear|rbf, got {kernel!r}')
  if rows_of > 1:
    # Every sequence in a row carries that row's reward. The group baseline then
    # subtracts a per-row quantity from a per-row quantity, so the advantage is
    # still well defined -- it is just shared.
    reward = reward.repeat_interleave(rows_of)
    active = active.repeat_interleave(rows_of)
    assert reward.shape[0] == batch
  return reward, active


def fold_active(active, rows_of):
  """A row is active when any of the sequences folded into it is.

  Kept next to `merge_kernel_rows`, which does the same fold to the features.
  Returns a per-sequence mask again, so the caller never has to track whether it
  is holding rows or sequences.
  """
  if rows_of <= 1:
    return active
  batch = active.shape[0]
  if batch % rows_of:
    raise ValueError(
      f'mmd_kernel_batch_size={rows_of} does not divide the batch size {batch}')
  folded = active.reshape(batch // rows_of, rows_of).any(dim=-1)
  return folded.repeat_interleave(rows_of)


def summed_paired_reward(features, batch, active, indices, weights,
                         kernel='linear', alpha=None, block_size=0,
                         rows_of=1):
  """`R = sum_l w_l * R_l` over the requested blocks.

  Args:
    features: `{index: (2 * batch, positions, hidden)}` as `capture_layer_hidden`
      yields it -- clean rows stacked above generated ones.
    batch: rows per half.
    active: `(batch,)` bool, computed by the caller from the TOKEN-level mask.
      See the module docstring: the feature grid carries no usable mask, so every
      feature position is pooled and this is the only thing that knows whether a
      row is worth scoring at all.
    indices, weights: parallel lists from `resolve_feature_layers` and
      `resolve_layer_weights`.

  Summing rather than averaging keeps a single-block config numerically identical
  to what it was before multi-block support existed.

  Returns `(reward, active, per_layer)`; `per_layer` is unweighted, for logging.
  """
  active = fold_active(active.bool(), rows_of)
  total, per_layer = None, {}
  for index, weight in zip(indices, weights):
    stacked = features[index]
    # Uniform over the feature grid -- the mask does not survive the read point.
    selected = torch.ones(stacked.shape[0] // 2, stacked.shape[1],
                          dtype=torch.bool, device=stacked.device)
    reward, _ = paired_sequence_reward(
      stacked[:batch], stacked[batch:], selected, kernel=kernel,
      alpha=alpha, block_size=block_size, rows_of=rows_of)
    reward = torch.where(active, reward, torch.zeros_like(reward))
    per_layer[index] = reward
    total = reward * weight if total is None else total + reward * weight
  return total, active, per_layer


def batch_advantages(reward, active):
  """Leave-one-out advantages over the *batch* axis, for a group of one.

  With `group_size=1` there is no second sample at the same state, so the group
  baseline does not exist. The other sequences in the batch are the next best
  comparison set: `A_i = r_i - mean_{j != i, active} r_j`, which is still
  independent of the sample it scores and so still unbiased.

  It is a noisier baseline than the group one, and not for a subtle reason: those
  rows carry different `x0` and different masking fractions, so their rewards are
  not measurements of the same quantity. It removes the reward's overall offset --
  which is what matters most here, since the pseudo-Huber reward is a negative
  distance and therefore always <= 0, so an unbaselined advantage is negative for
  every sample and pushes down every sequence the student wrote regardless of how
  good it was.

  DEGENERATE AT `mmd_kernel_batch_size == batch`: every row then carries the
  identical row reward, so the leave-one-out mean equals the reward and every
  advantage is exactly zero -- a silent no-gradient run. `mmd_reinforce_loss`
  rejects that combination rather than letting it train on nothing.

  Inactive rows are excluded from the baseline rather than counted as zero, which
  would drag it toward 0 by however many rows scored nothing.
  """
  if reward.shape[0] < 2:
    raise ValueError(
      'batch_advantages needs at least 2 sequences in the batch when '
      f'group_size=1, got {reward.shape[0]}; there is no baseline otherwise')
  weight = active.to(reward.dtype).unsqueeze(-1)      # (batch, 1)
  count = weight.sum(dim=0, keepdim=True)             # (1, 1)
  total = (reward * weight).sum(dim=0, keepdim=True)  # (1, group)
  loo = (total - reward * weight) / (count - weight).clamp_min(1.0)
  return reward - loo


def group_advantages(reward):
  """Leave-one-out advantages over the last (group) axis.

  `A_k = r_k - mean_{j != k} r_j`. The baseline excludes the candidate it scores,
  so it is independent of that sample and the estimator stays unbiased -- unlike
  centering on the group mean, which shrinks every advantage toward zero by 1/K.
  An all-equal group gets exactly zero, which is the desired no-op.
  """
  k = reward.shape[-1]
  if k < 2:
    raise ValueError('group_advantages needs a group of at least 2 samples, '
                     f'got {k}; leave-one-out has no baseline otherwise')
  loo = (reward.sum(-1, keepdim=True) - reward) / (k - 1)
  return reward - loo


class _LayeredFeatures:
  """Shared bookkeeping for a feature map: which blocks, at what weight."""

  # Set by the caller after construction; kept on the base so the plumbing lives
  # in one place.
  kernel = 'linear'
  rbf_alpha = None
  rbf_block_size = 0
  kernel_batch_size = 1

  def _resolve_layers(self, core, layer_index, layer_weights):
    self.layer_index = layer_index
    _, self.resolved_layers = resolve_feature_layers(core, layer_index)
    self.layer_weights = resolve_layer_weights(self.resolved_layers,
                                               layer_weights)

  @property
  def resolved_layer(self):
    """The single block, or the list when the reward sums several."""
    if len(self.resolved_layers) == 1:
      return self.resolved_layers[0]
    return list(self.resolved_layers)

  @property
  def layer_description(self):
    if len(self.resolved_layers) == 1:
      return str(self.resolved_layers[0])
    return ' + '.join(f'{w:g}*L{i}' for i, w
                      in zip(self.resolved_layers, self.layer_weights))


class FrozenBackboneFeatures(_LayeredFeatures):
  """The frozen diffusion teacher, read at one block as a fixed feature map.

  It is the model being distilled, so its features already encode what a
  plausible CIFAR-10 image looks like, where a generic pretrained network only
  encodes plausible pictures.

  ON `noise_level`. A denoiser is a function of a *noised* latent, which is the
  argument for masking both rows before reading it, and on text that is what
  `noise_level > 0` does. **On CIFAR-10 it should stay 0.0.** `[MASK]` is not an
  out-of-band symbol here: `UNet._center_data` maps token id 256 to +0.992 on a
  [-1, 1] scale, i.e. a near-white pixel. Masking 10% of positions paints
  correlated white speckle into both rows and that speckle, not the content,
  dominates the bottleneck features the reward is trying to read.

  ON `sigma`. Defaulted to zeros, which under `time_conditioning: True` tells the
  UNet "t = 0, clean input". That is consistent only at `noise_level = 0.0`,
  which is checked. If a nonzero level is ever wanted, pass the matching
  `sigma` -- `_process_sigma(_sigma_from_alphat(alpha))` for the same alpha --
  rather than leaving the teacher conditioned on a noise level its input does not
  have.

  Either way the clean and generated rows get the **same** mask, because the
  comparison is paired: an independent mask per row would inject a difference
  into a statistic meant to isolate what the student wrote.
  """

  def __init__(self, denoiser, mask_index, layer_index=None, noise_level=0.0,
               layer_weights=None):
    self.denoiser = denoiser
    self.mask_index = mask_index
    self.noise_level = float(noise_level)
    self._resolve_layers(denoiser, layer_index, layer_weights)

  @torch.no_grad()
  def paired_reward(self, clean_tokens, generated_tokens, selected, eos_id=None,
                    sigma=None, return_per_layer=False):
    """Per-sequence reward comparing a generated sample against the clean one.

    `selected` is the TOKEN-level mask `(batch, length)`. It is used to noise the
    rows (when `noise_level > 0`) and to decide `active`; it does not reach the
    feature grid -- see the module docstring.
    """
    del eos_id  # same vocabulary, and CIFAR-10's tokenizer has no EOS anyway
    if clean_tokens.shape != generated_tokens.shape:
      raise ValueError('clean/generated token shapes must match, got '
                       f'{tuple(clean_tokens.shape)} vs '
                       f'{tuple(generated_tokens.shape)}')
    if self.noise_level > 0.0 and sigma is None:
      raise ValueError(
        f'mmd_noise_level={self.noise_level} masks the rows but no sigma was '
        'given, so the teacher would be conditioned on t=0 while reading a '
        'noised input. Pass the matching sigma, or use mmd_noise_level=0.0')
    batch = clean_tokens.shape[0]
    selected = selected.bool()

    # One mask, applied to both rows, over the scored positions only.
    if self.noise_level > 0.0:
      keep = torch.rand(clean_tokens.shape, device=clean_tokens.device)
      drop = (keep < self.noise_level) & selected
      clean_xt = torch.where(drop, self.mask_index, clean_tokens)
      gen_xt = torch.where(drop, self.mask_index, generated_tokens)
    else:
      clean_xt, gen_xt = clean_tokens, generated_tokens

    stacked = torch.cat([clean_xt, gen_xt], dim=0)
    # The backbone wants the post-`_process_sigma` form: 1-D, one entry per row.
    if sigma is None:
      sigma = torch.zeros(stacked.shape[0], device=stacked.device,
                          dtype=torch.float32)
    else:
      sigma = torch.cat([sigma, sigma], dim=0)

    # `active` comes from the tokens, not the feature grid. The rbf estimator
    # needs two scored positions to have an unbiased value at all; the linear
    # one only needs the row to be non-empty.
    if self.kernel == 'rbf':
      active = selected.sum(dim=1) >= 2
    else:
      active = selected.any(dim=1)

    with capture_layer_hidden(self.denoiser, self.layer_index) as (feats, _):
      self.denoiser(stacked, sigma)
    reward, active, per_layer = summed_paired_reward(
      feats, batch, active, self.resolved_layers, self.layer_weights,
      kernel=self.kernel, alpha=self.rbf_alpha,
      block_size=self.rbf_block_size, rows_of=self.kernel_batch_size)
    if return_per_layer:
      return reward, active, per_layer
    return reward, active


# Every key this module logs, on every call, whatever the config. Ranks that
# disagree on the key set deadlock inside Lightning's `log_dict` under DDP, and
# the empty-selection path used to return early without logging at all -- which
# is exactly the kind of thing that fires on one rank and not another.
STAT_KEYS = ('mmd/reward', 'mmd/advantage_absmean', 'mmd/reward_spread')


def zero_stats(device=None):
  """The `STAT_KEYS` dict, all zero. Logged when there is nothing to score."""
  return {key: torch.zeros((), device=device) for key in STAT_KEYS}


def mmd_reinforce_loss(student_probs, x0, selected, critic, *,
                       group_size, eos_id=None, reward_weight=1.0,
                       baseline='auto'):
  """REINFORCE loss driving the student toward the clean sample's features.

  `group_size` hard samples are drawn from the student's differentiable one-step
  prediction at the *same* state; each is scored against the clean `x0` by
  pooling the frozen teacher's features, and the group's leave-one-out mean is
  the baseline. Gradient reaches the student only through `log p(sampled token)`
  -- the sampling is not differentiable, which is the whole reason for a score
  estimator rather than a reparameterized one.

  One reward per sequence multiplies the summed log-probability of every position
  the student wrote. Masked diffusion predicts each masked position independently
  in a single step, so there is no sub-sequence decision unit to attribute credit
  to.

  Args:
    student_probs: `(batch, length, vocab)`, differentiable, sums to 1.
    x0: `(batch, length)` clean tokens.
    selected: `(batch, length)` bool; the positions the student generated (the
      masked ones), which are also the only ones whose log-probability is summed.
    critic: a `FrozenBackboneFeatures`.
    group_size: samples per state. At >= 2 the group's own leave-one-out mean is
      the baseline. At 1 there is no second sample at the same state, so
      `baseline` decides what stands in; the sampling cost falls with the group,
      since each member costs one critic forward.
    baseline: `auto` picks `group` at group_size >= 2 and `none` at 1. `batch`
      uses the other sequences in the batch -- unbiased, free, but noisier. `none`
      subtracts nothing, which is still unbiased but leaves every advantage
      negative (the reward is a negative distance), so every update pushes down
      whatever was sampled; it is for measuring the baseline's worth, not for
      training.

  Returns `(loss, stats)`; `loss` already carries `reward_weight`, and `stats`
  always has every key in `STAT_KEYS`.
  """
  if group_size < 1:
    raise ValueError(f'mmd group_size must be >= 1, got {group_size}')
  if baseline not in ('auto', 'group', 'batch', 'none'):
    raise ValueError('mmd_baseline must be auto|group|batch|none, got '
                     f'{baseline!r}')
  if baseline == 'auto':
    baseline = 'group' if group_size > 1 else 'none'
  if baseline == 'group' and group_size < 2:
    raise ValueError('mmd_baseline=group needs group_size >= 2 -- '
                     'leave-one-out has no baseline within a group of one; '
                     'use baseline=none (the default at group_size=1) or '
                     'baseline=batch')
  batch, length, _ = student_probs.shape
  rows_of = int(getattr(critic, 'kernel_batch_size', 1) or 1)
  if baseline == 'batch' and rows_of >= batch:
    raise ValueError(
      f'mmd_baseline=batch with mmd_kernel_batch_size={rows_of} at batch size '
      f'{batch} gives every row the identical reward, so the leave-one-out '
      'baseline equals it and every advantage is exactly zero -- the run would '
      'train on nothing. Use mmd_baseline=group with mmd_group_size >= 2')
  selected = selected.bool()

  # Sampling and the reward are no-grad; only the log-prob gather carries
  # gradient.
  with torch.no_grad():
    flat = student_probs.reshape(-1, student_probs.shape[-1])
    draws = torch.multinomial(flat.float(), group_size, replacement=True)
    draws = draws.view(batch, length, group_size).permute(2, 0, 1)
    # Outside the generated span the sample is the clean one, so the critic sees
    # a row differing from the reference *only* where the student wrote.
    samples = torch.where(selected.unsqueeze(0), draws, x0.unsqueeze(0))

    rewards, actives, breakdown = [], [], {}
    multi = len(getattr(critic, 'resolved_layers', [None])) > 1
    for k in range(group_size):
      out = critic.paired_reward(x0, samples[k], selected, eos_id,
                                 return_per_layer=multi)
      reward, active = out[0], out[1]
      if multi:
        for index, value in out[2].items():
          breakdown.setdefault(index, []).append(value)
      rewards.append(reward)
      actives.append(active)
    reward = torch.stack(rewards, dim=-1)          # (batch, group)
    active = actives[0]                            # identical across the group

    if baseline == 'group':
      advantage = group_advantages(reward)
    elif baseline == 'batch':
      advantage = batch_advantages(reward, active)
    else:  # 'none'
      advantage = reward
    advantage = torch.where(active.unsqueeze(-1), advantage,
                            torch.zeros_like(advantage))

  # log p(sampled token), summed over the positions the student wrote.
  logp = torch.log(student_probs.clamp_min(1e-12))
  surrogate = student_probs.new_zeros(())
  for k in range(group_size):
    token_logp = logp.gather(-1, samples[k].unsqueeze(-1)).squeeze(-1)
    seq_logp = (token_logp * selected.to(token_logp.dtype)).sum(dim=1)
    surrogate = surrogate + (advantage[:, k] * seq_logp).sum()

  # Negated: maximizing reward means descending this. Normalized by generated
  # tokens so the scale does not track batch or sequence length.
  denom = selected.sum().clamp_min(1).to(surrogate.dtype)
  loss = -reward_weight * surrogate / denom

  with torch.no_grad():
    scored = reward[active] if active.any() else reward.new_zeros(1)
    stats = {'mmd/reward': scored.mean().detach(),
             'mmd/advantage_absmean': advantage.abs().mean().detach(),
             'mmd/reward_spread': (reward.std(dim=-1).mean().detach()
                                   if group_size > 1 else
                                   reward.new_zeros(()))}
    # With several blocks summed, the terms differ by orders of magnitude, so the
    # total alone does not say which block is driving the ranking. `_spread` is
    # the part that survives the leave-one-out baseline.
    for index, values in breakdown.items():
      per = torch.stack(values, dim=-1)
      stats[f'mmd/reward_l{index}'] = per[active].mean().detach() \
        if active.any() else per.new_zeros(())
      stats[f'mmd/reward_spread_l{index}'] = per.std(dim=-1).mean().detach()
  return loss, stats
