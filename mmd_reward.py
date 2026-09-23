"""Paired feature-discrepancy reward, ported from yrOPDLM's `train/distill_ar.py`.

The idea, as implemented there for block-diffusion OPDLM: a *frozen autoregressive*
model is used not as a critic that scores likelihood, but as a fixed feature map. Run
it over the clean data and over the student's sample, take the hidden states of one
middle transformer block, mean-pool them per response block, and reward the student
for closing the gap between the two pooled features:

    R = -mean_h( sqrt(delta_h^2 + c^2) - c ),   delta = pooled_gen - pooled_clean

That is a pseudo-Huber distance (c = 1e-3): quadratic for small deltas, linear for
large ones, so a single badly-placed block cannot dominate the gradient the way a
squared distance would. The reward is per block, and it drives a REINFORCE step whose
baseline is the leave-one-out mean over a group of samples drawn at the same state.

The ``draws + sentence`` mode added below uses a different, explicit kernel
estimator: every sequence is mean-pooled first, then one fixed clean embedding
is compared with K sampled embeddings. It uses an unclamped U-statistic for
K > 1 and a biased fallback for K = 1, for normalized linear, RBF and
pseudo-Huber distance-induced kernels. The last recovers the historical
pseudo-Huber reward at K = 1. All MMD estimators retain negative estimates;
their negated rewards can therefore be positive.

The separate ``linear_l2`` kernel always scores negative MSE between empirical
feature means, in fp32. It uses the biased/V-statistic form at every draw count,
without the off-diagonal correction used by the sentence-level estimators above.

Two things differ here from the OPDLM implementation, both forced by the model:

* **No doubled row.** OPDLM stacks `[prompt|x0|x0]` against `[prompt|x0|x0_gen]` in one
  sequence, because in block diffusion the generated half has to attend to a clean
  prefix inside the same row, and comparing halves of one row guarantees identical
  attention topology and SDPA memory layout. IDLM's critic is a plain causal LM over
  `[prompt|answer]`, so two rows of identical shape already have identical topology;
  they are placed in the same batch so they also share a kernel launch.
* **The reward is per sequence, not per block.** OPDLM's `block_size: 4` is its
  block-diffusion generation width -- a real decision unit, since the model commits
  four tokens at a time, so per-block credit lines up with something the model does.
  Masked diffusion has no width: every masked position is predicted independently in
  one step, so cutting the answer into fixed spans would invent a structure the model
  does not have. One reward per generated answer, multiplying the summed
  log-probability of every position the student wrote, is the honest translation.
"""

import contextlib
import os

import torch

import mmd_rbf

# yrOPDLM's MMD_PSEUDO_HUBER_C. Kept at its value there so rewards are comparable.
PSEUDO_HUBER_C = 0.001


def ar_decoder_layers(core):
  """The causal model's ordered transformer blocks.

  Probes the same three attribute paths yrOPDLM does, so the hook keeps working
  across common HF causal wrappers, and fails loudly rather than silently hooking
  an unrelated module. `num_hidden_layers` bounds the result because some models
  append non-transformer modules (MTP heads) to the same list.
  """
  candidates = (
    getattr(getattr(core, 'model', None), 'layers', None),
    getattr(getattr(getattr(core, 'model', None), 'decoder', None), 'layers', None),
    getattr(getattr(core, 'transformer', None), 'h', None),
    # This repo's own DiT, so the frozen *diffusion teacher* can serve as the
    # feature map instead of an external causal LM.
    getattr(core, 'blocks', None),
  )
  for layers in candidates:
    if layers is not None and len(layers) > 0:
      configured = getattr(getattr(core, 'config', None),
                           'num_hidden_layers', None)
      n = len(layers) if configured is None else int(configured)
      if n < 1 or n > len(layers):
        raise ValueError(
          f'AR config.num_hidden_layers={n} is incompatible with the '
          f'{len(layers)} discovered decoder blocks')
      return layers[:n]
  raise ValueError(
    'MMD feature extraction could not find transformer blocks at '
    'core.model.layers, core.model.decoder.layers, or core.transformer.h')


def embedding_module(core):
  """The token-embedding module, whose output is layer -1.

  That is the bag-of-embeddings floor: a pure lookup table with no attention
  and no mixing, so its pooled features carry the answer's token *multiset* and
  nothing about order. `scripts/probe_mmd_features.py` measures it at exactly
  0.000 structure, against 0.841 for block 6 -- which is what makes it the
  reference any real block has to beat rather than a sensible reward on its own.
  """
  for attr in ('vocab_embed', 'embed_tokens', 'wte'):
    module = getattr(core, attr, None)
    if module is not None:
      return module
    inner = getattr(core, 'model', None)
    module = None if inner is None else getattr(inner, attr, None)
    if module is not None:
      return module
  getter = getattr(core, 'get_input_embeddings', None)
  if callable(getter):
    module = getter()
    if module is not None:
      return module
  raise ValueError(
    'mmd_feature_layer=-1 needs a token-embedding module; looked for '
    'vocab_embed, embed_tokens, wte and get_input_embeddings()')


def resolve_feature_layers(core, layer_index):
  """Resolve `mmd_feature_layer` to an ordered list of blocks and their indices.

  Accepts `None` (the middle block), a single zero-based integer, or a sequence
  of them. The middle is the default for the same reason yrOPDLM picks it: late
  layers specialize toward the next-token head and early ones barely differ from
  the embedding, so neither end says much about whether two sequences *mean* the
  same thing.

  `-1` selects the token embeddings themselves, matching the convention in
  `scripts/probe_mmd_features.py`. That is a lookup table with no attention and
  no mixing, so the pooled feature is the answer's token multiset and the reward
  reduces to a soft bag-of-words distance -- order-blind by construction. The
  probe measures it at exactly 0.000 structure against block 6's 0.841, so it is
  the floor a real block has to beat, useful as a control arm rather than as a
  reward to train on for its own sake.

  A sequence asks for a *summed* reward -- one pseudo-Huber term per block. That
  is worth knowing about before using it: on this repo's TinyGSM teacher the
  reward per corrupted token grows roughly 6x per four blocks
  (`scripts/probe_mmd_features.py` measures ~0.0008 at block 0 against ~0.046 at
  block 11), so an unweighted sum is very nearly the deepest block alone. Pass
  `mmd_feature_layer_weights` to put the terms on comparable footing.
  """
  layers = ar_decoder_layers(core)
  if layer_index is None:
    indices = [len(layers) // 2]
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
    # -1 is the embedding output, the bag-of-embeddings floor; it is a real
    # choice here rather than an out-of-range index, so it is admitted
    # explicitly instead of falling through to Python's from-the-end indexing,
    # which would silently hand back the last block.
    if index < -1 or index >= len(layers):
      raise ValueError(
        f'mmd_feature_layer={index} is outside [-1,{len(layers) - 1}] '
        '(-1 = raw token embeddings)')
  return ([embedding_module(core) if i == -1 else layers[i] for i in indices],
          indices)


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
  """Capture the output of each requested transformer block, in one forward.

  Yields `({index: hidden}, indices)`. Only the first call through a block is
  retained, so a second forward cannot quietly hold another full
  `(batch, length, hidden)` activation. Reading several blocks costs one forward
  and one activation per block, not one forward per block. Used inside a
  frozen-teacher `no_grad` region; the `detach` is belt-and-braces.
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
      if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise RuntimeError(
          'MMD feature hook expected a rank-3 hidden tensor, got '
          f'{type(hidden).__name__} with shape {getattr(hidden, "shape", None)}')
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
      f'MMD feature hook on transformer block(s) {missing} did not run')


def pseudo_huber_reward(feature_delta):
  """`-mean(sqrt(delta^2 + c^2) - c)` over the final (feature) dimension, in fp32.

  Leading dimensions are preserved, so a `(batch, blocks, hidden)` delta gives a
  `(batch, blocks)` reward. fp32 regardless of the teacher's dtype: the deltas are
  small and a bf16 square would quantise most of them to the same value.
  """
  delta = feature_delta.float()
  c = PSEUDO_HUBER_C
  return -(torch.sqrt(delta ** 2 + c ** 2) - c).mean(dim=-1)


def mean_squared_reward(feature_delta):
  """Negative MSE of pooled feature means, preserving leading axes, in fp32."""
  return -feature_delta.float().square().mean(dim=-1)


def pool_over_selected(features, selected):
  """Mean-pool `(rows, length, hidden)` over `selected`, ignoring the rest.

  Returns `(pooled, active)`, shapes `(rows, hidden)` and `(rows,)`. A row
  selecting nothing pools to zero and is reported inactive rather than scored
  on positions no token contributed to.
  """
  selected = selected.bool()
  weight = selected.unsqueeze(-1).to(torch.float32)
  denom = selected.sum(dim=1).clamp_min(1).to(torch.float32).unsqueeze(-1)
  return (features.float() * weight).sum(dim=1) / denom, selected.any(dim=1)


@torch.no_grad()
def aligned_feature_mse_reward(clean, generated, selected):
  """Negative MSE of aligned masked-token features, one value per sequence.

  Square the coordinate differences BEFORE averaging over hidden dimensions
  and selected positions. Token permutations and opposing feature errors must
  not cancel as they can when comparing pooled feature means. Reductions use
  fp32; unselected positions, including non-finite values, contribute nothing.
  Empty rows return zero and are inactive; one selected token is sufficient.
  """
  if clean.ndim != 3 or clean.shape != generated.shape:
    raise ValueError('feature_mse expects matching (batch, length, hidden) '
                     f'features, got {tuple(clean.shape)} and '
                     f'{tuple(generated.shape)}')
  if selected.shape != clean.shape[:2]:
    raise ValueError(f'feature_mse selected mask must be '
                     f'{tuple(clean.shape[:2])}, got {tuple(selected.shape)}')
  if clean.shape[-1] == 0:
    raise ValueError('feature_mse needs at least one hidden coordinate')
  selected = selected.bool()
  delta = (generated.detach().float() - clean.detach().float()).masked_fill(
    ~selected.unsqueeze(-1), 0.0)
  token_mse = delta.square().mean(dim=-1)
  counts = selected.sum(dim=-1)
  reward = -token_mse.sum(dim=-1) / counts.clamp_min(1).float()
  return reward, counts > 0


def sequence_pooled_delta(clean, generated, selected):
  """Mean-pool both feature streams over `selected`, and subtract.

  Args:
    clean, generated: `(batch, length, hidden)` features over the same span.
    selected: `(batch, length)` bool; which positions count. A row selecting
      nothing pools to zero and is reported inactive, so callers can drop it
      rather than score a reward no token contributed to.

  Returns `(feature_delta, active)`, shapes `(batch, hidden)` and `(batch,)`.
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

  clean_mean, active = pool_over_selected(clean, selected)
  gen_mean, _ = pool_over_selected(generated, selected)
  return gen_mean - clean_mean, active


def median_alpha(feat, selected, max_samples=512, generator=None,
                 distance='squared_euclidean', huber_delta=0.001):
  """`alpha = 1 / (2 * median(D))` from real pairwise distances.

  `D` is squared Euclidean by default, or the sum of coordinatewise smooth
  pseudo-Huber distances selected by `distance` and `huber_delta`.

  The standard bandwidth heuristic, measured on the REAL branch only, so it
  never needs a gradient and never moves with the student. A bandwidth fitted to
  the student's own features would drift as the student does, which turns the
  reward into a moving target.

  Subsampled because the median needs every pair: at `length` 1024 the exact
  version is the N^2 matrix this module exists to avoid. 512 samples give
  ~131k pairs, far more than the median needs to be stable.

  Falls back to `alpha = 1` when the distances degenerate -- an all-masked row,
  or identical features -- so a zero median cannot divide by zero.
  """
  mmd_rbf.validate_distance(distance, huber_delta)
  valid = feat[selected.bool()]
  if valid.shape[0] < 2:
    return 1.0
  if valid.shape[0] > max_samples:
    idx = torch.randperm(valid.shape[0], device=valid.device,
                         generator=generator)[:max_samples]
    valid = valid[idx]
  if distance == 'squared_euclidean':
    distances = torch.cdist(valid.float(), valid.float()).pow_(2)
  else:
    vectors = valid.detach().float().unsqueeze(0)
    distances = mmd_rbf.pairwise_distance(
      vectors, vectors, distance=distance, huber_delta=huber_delta)[0]
  # Off-diagonal only: the diagonal is exactly 0 and would drag the median down
  # by however many samples were drawn.
  n = distances.shape[0]
  off = distances[~torch.eye(n, dtype=torch.bool, device=distances.device)]
  med = off.median()
  if not torch.isfinite(med) or med <= 0:
    return 1.0
  return float(1.0 / (2.0 * med))


def rbf_sequence_reward(clean, generated, selected, alpha=None, block_size=0,
                        distance='squared_euclidean', huber_delta=0.001,
                        omit_real_real=False, omit_fake_fake=False):
  """`-MMD^2` per sequence under an exponential distance kernel.

  The linear-kernel reward compares the two feature sets by their MEANS, which
  is all a linear kernel can see -- `sequence_pooled_delta` is literally that
  mean difference. An RBF kernel compares the distributions, so a sample that
  matches the clean row on average but not in shape is no longer free.

  Both sets are the per-position features of one sequence, weighted by
  `selected`; masked positions carry weight 0 and contribute nothing to any of
  the three sums, so no masking of the kernel itself is needed.

  The unbiased estimator, with the self-pairs removed:

      MMD^2 = (S_xx - n)/(n(n-1)) + (S_yy - n)/(n(n-1)) - 2 S_xy/n^2

  `- n` is exact rather than approximate: the weights are 0/1, so the diagonal
  contributes `sum_i w_i^2 k(x_i, x_i)` = `sum_i w_i` = `n`. Rows with fewer
  than two scored positions have no unbiased estimate at all and are reported
  inactive rather than given a fabricated 0.

  ``omit_real_real=True`` skips the entire real-real contribution without
  changing the fake-fake or real-fake normalization or the active mask.
  ``omit_fake_fake=True`` independently skips the entire fake-fake term,
  including diagonal subtraction, while retaining the real-fake factor of 2.

  Returns `(reward, active)`, matching the linear path.
  """
  mmd_rbf.validate_distance(distance, huber_delta)
  weights = selected.to(torch.float32)
  n = weights.sum(dim=1)
  active = n >= 2
  if not bool(active.any()):
    zeros = clean.new_zeros(clean.shape[0], dtype=torch.float32)
    return zeros, active

  real = clean.detach().float()
  fake = generated.detach().float()
  if alpha is None:
    alpha = median_alpha(real, selected, distance=distance,
                         huber_delta=huber_delta)

  s_xx, s_yy, s_xy = mmd_rbf.rbf_pair_sums(
    real, fake, weights, alpha, block_size=block_size,
    distance=distance, huber_delta=huber_delta, omit_real_real=omit_real_real,
    omit_fake_fake=omit_fake_fake)

  safe_n = n.clamp_min(2.0)
  pairs = safe_n * (safe_n - 1.0)
  square = safe_n * safe_n
  if omit_fake_fake:
    # Drop the entire normalized term, including its diagonal subtraction.
    mmd2 = -2.0 * s_xy / square
    if not omit_real_real:
      mmd2 = (s_xx - safe_n) / pairs + mmd2
  elif omit_real_real:
    mmd2 = (s_yy - safe_n) / pairs - 2.0 * s_xy / square
  else:
    mmd2 = ((s_xx - safe_n) / pairs
            + (s_yy - safe_n) / pairs
            - 2.0 * s_xy / square)
  # A U-statistic can be negative; preserve it instead of biasing the reward
  # by clipping the estimate at zero.
  reward = -mmd2
  return torch.where(active, reward, torch.zeros_like(reward)), active


def merge_kernel_rows(clean, generated, selected, rows_of):
  """Fold `rows_of` adjacent sequences into one kernel row.

  The ROW is the unit the statistic is computed over, for both kernels: the
  linear kernel pools it to one mean, and the rbf kernel takes its valid tokens
  as the sample set. So merging N sequences means the reward stops asking "does
  this sample match its own clean row" and starts asking "does this GROUP of
  samples match that group of clean rows" -- a distribution match over N
  sequences rather than N separate comparisons.

  That is a real trade, not a free widening. A row of N sequences can be
  satisfied by samples that swap content between them: two rows whose answers
  are exchanged pool to the same mean and hold the same token multiset, so
  neither kernel can tell them apart. What it buys is a statistic with N times
  the tokens behind it, which for rbf means N^2 times the pairs.

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
                           alpha=None, block_size=0, rows_of=1,
                           level='token',
                           distance='squared_euclidean', huber_delta=0.001,
                           exclude_same_sentence=False,
                           omit_real_real=False, omit_fake_fake=False,
                           exclude_same_position=False):
  """One reward per sequence; rows selecting nothing score zero.

  `kernel='linear'` is the original: pool both sides and score the difference
  with pseudo-Huber. This historical robust score is not linear-kernel MMD^2.
  `kernel='linear_l2'` uses the same pooling and scores negative MSE between
  the empirical means, the biased/V-statistic normalized linear-kernel MMD.
  `kernel='rbf'` compares the two feature distributions instead. See
  `rbf_sequence_reward`.

  `level` chooses what one sample is. `token` (the default) hands the kernel
  every scored POSITION of the row. `sentence` mean-pools each sequence first
  and hands it the row's `rows_of` SENTENCE vectors instead.

  `rows_of > 1` computes the statistic over that many sequences at once (see
  `merge_kernel_rows`) and hands every sequence in a row its row's reward.
  REINFORCE multiplies each sequence's advantage by ITS OWN log-probability, so
  the return has to stay per sequence whatever the row size -- the row only
  changes what the number means, not its shape.
  """
  if exclude_same_position:
    raise ValueError('mmd_rbf_exclude_same_position requires '
                     'mmd_kernel_source=draws')
  if omit_real_real and kernel != 'rbf':
    raise ValueError('mmd_rbf_omit_real_real requires mmd_kernel=rbf')
  if omit_fake_fake and kernel != 'rbf':
    raise ValueError('mmd_rbf_omit_fake_fake requires mmd_kernel=rbf')
  if exclude_same_sentence:
    raise ValueError('mmd_rbf_exclude_same_sentence requires '
                     'mmd_kernel_source=draws')
  if kernel == 'pseudo_huber':
    raise ValueError('mmd_kernel=pseudo_huber requires '
                     'mmd_kernel_source=draws and mmd_kernel_level=sentence')
  batch = clean.shape[0]
  if level == 'sentence':
    # One sample per SEQUENCE: mean-pool it over its own scored positions
    # first, then let the kernel see the row's `rows_of` sentence vectors as
    # its sample set. Token level instead hands the kernel every scored
    # position of the row.
    #
    # For the linear kernel this is the same reward whenever the sequences in a
    # row carry equal numbers of scored positions, since a mean of means then
    # equals the pooled mean -- exactly so at `rows_of=1`. It differs when the
    # counts differ, because pooling weights a sequence by its length while
    # this weights every sequence alike. On TinyGSM those counts differ a lot:
    # answers vary in length and `t ~ U(0,1)` masks a different fraction of
    # each. For rbf it is a different statistic outright.
    if rows_of < 2 and kernel == 'rbf':
      raise ValueError(
        'mmd_kernel_level=sentence with mmd_kernel=rbf needs '
        'mmd_kernel_batch_size >= 2: a row of one sentence gives the unbiased '
        'MMD^2 a single sample per side, which has no estimate at all')
    if batch % rows_of:
      raise ValueError(
        f'mmd_kernel_batch_size={rows_of} does not divide the batch size '
        f'{batch}; a row would straddle the end of the batch')
    clean_m, active_seq = pool_over_selected(clean, selected)
    gen_m, _ = pool_over_selected(generated, selected)
    rows = batch // rows_of
    clean = clean_m.reshape(rows, rows_of, -1)
    generated = gen_m.reshape(rows, rows_of, -1)
    selected = active_seq.reshape(rows, rows_of)
  else:
    clean, generated, selected = merge_kernel_rows(
      clean, generated, selected, rows_of)
  if kernel == 'rbf':
    reward, active = rbf_sequence_reward(
      clean, generated, selected, alpha=alpha, block_size=block_size,
      distance=distance, huber_delta=huber_delta, omit_real_real=omit_real_real,
      omit_fake_fake=omit_fake_fake)
  elif kernel in ('linear', 'linear_l2'):
    delta, active = sequence_pooled_delta(clean, generated, selected)
    reward = (mean_squared_reward(delta) if kernel == 'linear_l2'
              else pseudo_huber_reward(delta))
    reward = torch.where(active, reward, torch.zeros_like(reward))
  else:
    raise ValueError(f'mmd_kernel must be linear|linear_l2|rbf, got {kernel!r}')
  if rows_of > 1:
    # Every sequence in a row carries that row's reward. The group baseline
    # then subtracts a per-row quantity from a per-row quantity, so the
    # advantage is still well defined -- it is just shared.
    reward = reward.repeat_interleave(rows_of)
    active = active.repeat_interleave(rows_of)
    assert reward.shape[0] == batch
  return reward, active


def summed_paired_reward(features, batch, selected, indices, weights,
                         kernel='linear', alpha=None, block_size=0,
                         rows_of=1, level='token',
                         distance='squared_euclidean', huber_delta=0.001,
                         exclude_same_sentence=False, reward_type='feature',
                         omit_real_real=False, omit_fake_fake=False,
                         exclude_same_position=False):
  """`R = sum_l w_l * R_l` over the requested blocks.

  Args:
    features: `{index: (2 * batch, length, hidden)}` as `capture_layer_hidden`
      yields it -- clean rows stacked above generated ones.
    batch: rows per half.
    indices, weights: parallel lists from `resolve_feature_layers` and
      `resolve_layer_weights`.

  Summing rather than, say, averaging keeps a single-block config numerically
  identical to what it was before multi-block support existed. `active` depends
  only on `selected`, so it is the same for every block.

  ``reward_type='feature_mse'`` compares aligned token features per row and
  ignores the kernel, row grouping and pooling options.

  Returns `(reward, active, per_layer)`; `per_layer` is unweighted, for logging.
  """
  if reward_type not in ('feature', 'feature_mse'):
    raise ValueError(f'unknown feature reward type {reward_type!r}')
  total, active, per_layer = None, None, {}
  for index, weight in zip(indices, weights):
    stacked = features[index]
    if reward_type == 'feature_mse':
      # Always compare each original sequence independently; kernel row
      # grouping and sentence pooling do not apply to aligned token MSE.
      reward, active = aligned_feature_mse_reward(
        stacked[:batch], stacked[batch:], selected)
    else:
      reward, active = paired_sequence_reward(
        stacked[:batch], stacked[batch:], selected, kernel=kernel,
        alpha=alpha, block_size=block_size, distance=distance,
        huber_delta=huber_delta, exclude_same_sentence=exclude_same_sentence,
        exclude_same_position=exclude_same_position,
        omit_real_real=omit_real_real, omit_fake_fake=omit_fake_fake,
        rows_of=rows_of, level=level)
    per_layer[index] = reward
    total = reward * weight if total is None else total + reward * weight
  return total, active, per_layer


@torch.no_grad()
def linear_mmd2_fixed_real(real, fake):
  """Linear-kernel MMD^2 for one fixed real and K fake vectors per row.

  The real vector is the observed conditional target. With K > 1, fake
  self-pairs are removed for an estimator unbiased over the fake draws. With
  K = 1, where no off-diagonal fake pair exists, this is the biased empirical
  MSE between the real and fake means. For K > 1 the same estimator is written
  as that MSE minus ``Var(fake) / K``; this is algebraically the fake-diagonal
  removal for the normalized linear kernel ``k(x,y) = mean_d x_d y_d``. No
  clamp is applied: the K > 1 U-statistic can legitimately be negative.
  """
  if real.ndim != 2 or fake.ndim != 3:
    raise ValueError('fixed-real linear MMD expects (batch, hidden) real and '
                     '(batch, draws, hidden) fake features, got '
                     f'{tuple(real.shape)} and {tuple(fake.shape)}')
  if real.shape[0] != fake.shape[0] or real.shape[1] != fake.shape[2]:
    raise ValueError('fixed-real linear MMD features must share batch and '
                     f'hidden dimensions, got {tuple(real.shape)} and '
                     f'{tuple(fake.shape)}')
  draws = fake.shape[1]
  if draws < 1:
    raise ValueError('fixed-real linear MMD needs at least one fake draw')

  real = real.detach().float()
  fake = fake.detach().float()
  mean_mse = (real - fake.mean(dim=1)).square().mean(dim=-1)
  if draws == 1:
    return mean_mse
  variance_correction = fake.var(
    dim=1, correction=1).mean(dim=-1) / draws
  return mean_mse - variance_correction


@torch.no_grad()
def pseudo_huber_mmd2_fixed_real(real, fake):
  """MMD^2 for the kernel induced by the historical pseudo-Huber distance.

  With rho(a,b) = mean_d(sqrt((a_d-b_d)^2+c^2)-c), the statistic is
  mean_i rho(real,fake_i) minus half the off-diagonal mean fake-pair distance.
  K > 1 is unbiased over independent fake draws conditional on the fixed real
  vector. K = 1 omits the unavailable pair correction and exactly recovers
  the historical distance. This defines a different kernel, rather than
  applying pseudo-Huber to a linear/RBF MMD estimate. Its coordinate-additive
  distance need not distinguish distributions differing only in dependence
  between feature dimensions. Negative U-statistics are retained.

  Pair distances are accumulated with (batch, hidden) scratch tensors, never
  a (batch, draws, draws, hidden) allocation. Arithmetic stays in fp32.
  """
  if real.ndim != 2 or fake.ndim != 3:
    raise ValueError('fixed-real pseudo-Huber MMD expects (batch, hidden) real '
                     'and (batch, draws, hidden) fake features, got '
                     f'{tuple(real.shape)} and {tuple(fake.shape)}')
  if real.shape[0] != fake.shape[0] or real.shape[1] != fake.shape[2]:
    raise ValueError('fixed-real pseudo-Huber MMD features must share batch '
                     f'and hidden dimensions, got {tuple(real.shape)} and '
                     f'{tuple(fake.shape)}')
  draws = fake.shape[1]
  if draws < 1:
    raise ValueError('fixed-real pseudo-Huber MMD needs at least one fake draw')

  real = real.detach().float()
  fake = fake.detach().float()
  cross = -pseudo_huber_reward(fake[:, 0] - real)
  if draws == 1:
    return cross
  for index in range(1, draws):
    cross = cross - pseudo_huber_reward(fake[:, index] - real)
  pair_sum = torch.zeros_like(cross)
  for left in range(draws):
    for right in range(left + 1, draws):
      pair_sum = pair_sum - pseudo_huber_reward(fake[:, left] - fake[:, right])
  return cross / draws - pair_sum / (draws * (draws - 1))


def summed_multi_draw_reward(features, batch, draw_count, length, selected,
                             indices, weights, kernel='linear', alpha=None,
                             block_size=0, level='token',
                             distance='squared_euclidean', huber_delta=0.001,
                             exclude_same_sentence=False, reward_type='feature',
                             omit_real_real=False, omit_fake_fake=False,
                             exclude_same_position=False):
  """Score K draws from each state as one feature distribution.

  ``features`` contains the clean batch followed by K generated batches.
  ``level='token'`` retains the historical multi-draw estimator: one side has
  ``n_scored`` real token features and the other ``K * n_scored`` fake token
  features. ``level='sentence'`` first mean-pools every sequence over its own
  scored positions, yielding one fixed real vector and K fake vectors.

  At sentence level linear, RBF and pseudo-Huber use the same estimator rule.
  K > 1 removes fake self-pairs and is unbiased over draws conditional on the
  fixed real vector; K = 1 uses the biased empirical estimate. The linear
  path is a normalized linear-kernel MMD rather than the historical
  pseudo-Huber transform used by paired/token rewards. The separate
  ``pseudo_huber`` kernel uses the distance-induced estimator and recovers the
  historical reward at K = 1.

  ``linear_l2`` scores negative MSE between the empirical real and fake means
  at either level, in fp32. It is the biased/V-statistic form for every K,
  with no fake variance correction and no aligned-token comparison.

  With RBF token kernels, ``exclude_same_sentence=True`` keeps only
  cross-draw fake-fake token pairs and omits the real-real term. It requires
  K >= 2, retains all real-fake pairs, and is active with one scored position.
  The reward is the negative of that signed statistic, not a full MMD^2.

  ``exclude_same_position=True`` additionally removes matching-position pairs
  from the cross-draw fake-fake term. Its denominator becomes K*(K-1)*n*(n-1),
  and rows need n >= 2 scored positions. Real-fake pairs and their K*n^2
  denominator are unchanged. This requires the RBF token cross-sentence mode.

  ``omit_real_real=True`` separately drops only the RBF real-real term,
  retaining the existing fake-fake pairs, normalization and active mask. It
  supports both levels and K >= 1; the cross-sentence mode already omits it.
  ``omit_fake_fake=True`` independently drops the entire RBF fake-fake term
  at either level, including the K = 1 sentence fallback constant. It also
  applies in cross-sentence mode without relaxing its K >= 2 constraint.

  ``reward_type='feature_mse'`` averages K aligned masked-token MSE rewards
  per original row, with no kernel or feature pooling.

  Returns ``(reward, active, per_layer)`` with one reward per original row.
  """
  if reward_type not in ('feature', 'feature_mse'):
    raise ValueError(f'unknown feature reward type {reward_type!r}')
  if selected.shape != (batch, length):
    raise ValueError(f'multi-draw selected mask must be {(batch, length)}, got '
                     f'{tuple(selected.shape)}')
  if draw_count < 1:
    raise ValueError(f'multi-draw needs at least one draw, got {draw_count}')
  if reward_type == 'feature':
    if exclude_same_position and (kernel != 'rbf' or level != 'token'
                                  or not exclude_same_sentence or draw_count < 2):
      raise ValueError('mmd_rbf_exclude_same_position requires mmd_kernel=rbf, '
                       'mmd_kernel_level=token, mmd_kernel_batch_size >= 2 '
                       'and mmd_rbf_exclude_same_sentence=True')
    if omit_real_real and kernel != 'rbf':
      raise ValueError('mmd_rbf_omit_real_real requires mmd_kernel=rbf')
    if omit_fake_fake and kernel != 'rbf':
      raise ValueError('mmd_rbf_omit_fake_fake requires mmd_kernel=rbf')
    if level not in ('token', 'sentence'):
      raise ValueError(f'mmd_kernel_level must be token|sentence, got {level!r}')
    if kernel == 'pseudo_huber' and level != 'sentence':
      raise ValueError('mmd_kernel=pseudo_huber requires '
                       'mmd_kernel_source=draws and mmd_kernel_level=sentence')
    if exclude_same_sentence:
      if kernel != 'rbf' or level != 'token':
        raise ValueError('mmd_rbf_exclude_same_sentence requires '
                         'mmd_kernel=rbf and mmd_kernel_level=token')
      if draw_count < 2:
        raise ValueError('mmd_rbf_exclude_same_sentence requires '
                         'mmd_kernel_batch_size >= 2')
  expected_rows = (draw_count + 1) * batch
  total, active, per_layer = None, None, {}
  for index, weight in zip(indices, weights):
    hidden = features[index]
    if hidden.shape[0] != expected_rows or hidden.shape[1] != length:
      raise ValueError(
        f'multi-draw layer {index} must be {(expected_rows, length, "D")}, '
        f'got {tuple(hidden.shape)}')
    real = hidden[:batch]
    fake_by_draw = hidden[batch:].reshape(draw_count, batch, length, -1)

    if reward_type == 'feature_mse':
      # Average the K independently aligned errors, not their feature means.
      # No cross-draw/token kernel terms or variance correction belong here.
      draw_rewards = []
      for generated in fake_by_draw:
        draw_reward, layer_active = aligned_feature_mse_reward(
          real, generated, selected)
        draw_rewards.append(draw_reward)
      reward = torch.stack(draw_rewards, dim=0).mean(dim=0)
    elif level == 'sentence':
      real_sentence, real_active = pool_over_selected(real, selected)
      fake_sentence, fake_active = pool_over_selected(
        fake_by_draw.reshape(draw_count * batch, length, -1),
        selected.repeat(draw_count, 1))
      fake_sentence = fake_sentence.reshape(
        draw_count, batch, -1).permute(1, 0, 2)
      layer_active = real_active & fake_active.reshape(
        draw_count, batch).all(dim=0)

      if kernel == 'rbf':
        layer_alpha = alpha
        if layer_alpha is None:
          # Estimate the sentence-scale bandwidth from real sentences across
          # the device batch, never from the moving student samples.
          layer_alpha = median_alpha(
            real_sentence.unsqueeze(1), real_active.unsqueeze(1),
            distance=distance, huber_delta=huber_delta)
        mmd2 = mmd_rbf.rbf_mmd2_fixed_real(
          real_sentence, fake_sentence, layer_alpha,
          block_size=block_size, distance=distance, huber_delta=huber_delta,
          omit_real_real=omit_real_real, omit_fake_fake=omit_fake_fake)
      elif kernel == 'linear':
        mmd2 = linear_mmd2_fixed_real(real_sentence, fake_sentence)
      elif kernel == 'linear_l2':
        mmd2 = -mean_squared_reward(
          fake_sentence.mean(dim=1) - real_sentence)
      elif kernel == 'pseudo_huber':
        mmd2 = pseudo_huber_mmd2_fixed_real(real_sentence, fake_sentence)
      else:
        raise ValueError('mmd_kernel must be linear|linear_l2|rbf|pseudo_huber, got '
                         f'{kernel!r}')
      reward = torch.where(layer_active, -mmd2, torch.zeros_like(mmd2))
    else:
      if not exclude_same_sentence:
        fake = fake_by_draw.permute(1, 0, 2, 3).reshape(
          batch, draw_count * length, -1)
        w_real = selected.to(real.dtype)
        w_fake = w_real.repeat(1, draw_count)
      if kernel == 'rbf':
        layer_alpha = alpha
        if layer_alpha is None:
          layer_alpha = median_alpha(
            real.detach().float(), selected, distance=distance,
            huber_delta=huber_delta)
        if exclude_same_sentence:
          # Keep independent sentence draws separate for the fake-fake term.
          # The single real sentence supplies only the real-fake term.
          mmd2, layer_active = mmd_rbf.rbf_mmd2_cross_sentence(
            real, fake_by_draw.permute(1, 0, 2, 3), selected, layer_alpha,
            block_size=block_size, distance=distance, huber_delta=huber_delta,
            omit_fake_fake=omit_fake_fake,
            exclude_same_position=exclude_same_position)
        else:
          mmd2, layer_active = mmd_rbf.rbf_mmd2_unequal(
            real, fake, w_real, w_fake, layer_alpha, block_size=block_size,
            distance=distance, huber_delta=huber_delta,
            omit_real_real=omit_real_real, omit_fake_fake=omit_fake_fake)
        # Keep negative U-statistics, just as in the sentence-level path.
        reward = -mmd2
      elif kernel in ('linear', 'linear_l2'):
        real_mean, layer_active = pool_over_selected(real, selected)
        fake_mean, _ = pool_over_selected(fake, w_fake.bool())
        delta = fake_mean - real_mean
        reward = (mean_squared_reward(delta) if kernel == 'linear_l2'
                  else pseudo_huber_reward(delta))
        reward = torch.where(layer_active, reward, torch.zeros_like(reward))
      else:
        raise ValueError(f'mmd_kernel must be linear|linear_l2|rbf, got {kernel!r}')

    per_layer[index] = reward
    total = reward * weight if total is None else total + reward * weight
    active = (layer_active if active is None else
              active & layer_active)
  return total, active, per_layer


def batch_advantages(reward, active):
  """Leave-one-out advantages over the batch axis for a group of one.

  There is no within-state baseline when `group_size=1`. The other active
  sequences provide an unbiased, though noisier, control variate because each
  row carries a different question, answer, and masking fraction. Inactive
  rows are excluded rather than treated as zero rewards.
  """
  if reward.shape[0] < 2:
    raise ValueError(
      'batch_advantages needs at least 2 sequences in the batch when '
      f'group_size=1, got {reward.shape[0]}; there is no baseline otherwise')
  weight = active.to(reward.dtype).unsqueeze(-1)
  count = weight.sum(dim=0, keepdim=True)
  total = (reward * weight).sum(dim=0, keepdim=True)
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
  """Shared bookkeeping for the two feature maps: which blocks, at what weight."""

  # Set by the critics' constructors; kept on the shared base so both feature
  # maps get the plumbing instead of each threading its own copy.
  reward_type = 'feature'
  kernel = 'linear'
  kernel_level = 'token'
  rbf_alpha = None
  rbf_distance = 'squared_euclidean'
  rbf_huber_delta = 0.001
  rbf_exclude_same_sentence = False
  rbf_exclude_same_position = False
  rbf_omit_real_real = False
  rbf_omit_fake_fake = False
  rbf_block_size = 0
  kernel_batch_size = 1
  # Only the model forward is chunked. Kernel and reward calculations retain
  # the complete logical batch, including its automatic RBF bandwidth.
  forward_batch_size = None

  def _capture_forward_chunks(self, core, batch, forward):
    size = self.forward_batch_size
    if size is None:
      size = batch
    else:
      size = int(size)
      if size < 1:
        raise ValueError('feature forward_batch_size must be a positive integer')
    if size >= batch:
      with capture_layer_hidden(core, self.layer_index) as (captured, _):
        forward(0, batch)
      return captured
    parts = {index: [] for index in self.resolved_layers}
    for start in range(0, batch, size):
      with capture_layer_hidden(core, self.layer_index) as (captured, _):
        forward(start, min(start + size, batch))
      for index in self.resolved_layers:
        parts[index].append(captured[index])
    return {index: torch.cat(values, dim=0) for index, values in parts.items()}

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

  @staticmethod
  def _layer_tag(index):
    """`-1` prints as `embed`, so a log line says what was actually read."""
    return 'embed' if index == -1 else str(index)

  @property
  def layer_description(self):
    if len(self.resolved_layers) == 1:
      return self._layer_tag(self.resolved_layers[0])
    return ' + '.join(f'{w:g}*L{self._layer_tag(i)}' for i, w
                      in zip(self.resolved_layers, self.layer_weights))


class FrozenARFeatures(_LayeredFeatures):
  """A frozen causal LM used as a fixed feature map, not as a likelihood critic.

  This is the role yrOPDLM gives Qwen3-0.6B: never trained, never asked for a
  probability, only for the hidden states of one middle block. The analogue for
  TinyGSM is SmolLM-135M, which shares the student's tokenizer -- a critic on a
  different vocabulary would be scoring token ids that mean something else.

  Token-id safety: the student's vocabulary is the AR's *plus* the padding and
  mask ids appended by the dataloader (49152, 49153 against SmolLM's 49152), so
  feeding a latent straight in would index past the embedding table. Committed
  answers contain neither, but padding does, and it is remapped to EOS rather
  than clamped to an arbitrary row. Those positions are excluded from `selected`
  anyway; the remap only keeps the forward legal, and because attention is
  causal it cannot disturb any earlier position's features.
  """

  def __init__(self, model_name, layer_index=None, device='cuda',
               dtype=torch.bfloat16, layer_weights=None, weights_dir=None):
    import transformers
    from model_assets import pretrained_kwargs
    self.model = transformers.AutoModelForCausalLM.from_pretrained(
      **pretrained_kwargs(model_name, weights_dir),
      torch_dtype=dtype).to(device).eval()
    for param in self.model.parameters():
      param.requires_grad_(False)
    self.vocab_size = int(self.model.config.vocab_size)
    self._resolve_layers(self.model, layer_index, layer_weights)
    self.device = device

  def _safe_ids(self, tokens, eos_id):
    """Map out-of-vocabulary student ids (pad, mask) onto EOS."""
    if int(tokens.max()) < self.vocab_size:
      return tokens
    return torch.where(tokens >= self.vocab_size,
                       torch.full_like(tokens, eos_id), tokens)

  @torch.no_grad()
  def features(self, tokens, eos_id, attention_mask=None):
    """`{index: (batch, length, hidden)}` for every configured block."""
    tokens = self._safe_ids(tokens, eos_id)
    return self._capture_forward_chunks(
      self.model, tokens.shape[0],
      lambda start, end: self.model(
        input_ids=tokens[start:end],
        attention_mask=(None if attention_mask is None else
                        attention_mask[start:end])))

  @torch.no_grad()
  def multi_draw_reward(self, clean_tokens, draws, selected, eos_id,
                        attention_mask=None, return_per_layer=False):
    """Compare K generated answers from one state with its clean answer."""
    if draws.ndim != 3:
      raise ValueError('multi_draw_reward expects (K, batch, length) draws, '
                       f'got {tuple(draws.shape)}')
    draw_count, batch, length = draws.shape
    if clean_tokens.shape != (batch, length):
      raise ValueError(f'clean tokens must be {(batch, length)}, got '
                       f'{tuple(clean_tokens.shape)}')

    stacked = torch.cat(
      [clean_tokens, draws.reshape(draw_count * batch, length)], dim=0)
    mask = None
    if attention_mask is not None:
      if attention_mask.shape != (batch, length):
        raise ValueError(f'attention mask must be {(batch, length)}, got '
                         f'{tuple(attention_mask.shape)}')
      mask = torch.cat([attention_mask,
                        attention_mask.repeat(draw_count, 1)], dim=0)
    feats = self.features(stacked, eos_id, attention_mask=mask)
    reward, active, per_layer = summed_multi_draw_reward(
      feats, batch, draw_count, length, selected, self.resolved_layers,
      self.layer_weights, kernel=self.kernel, alpha=self.rbf_alpha,
      reward_type=self.reward_type,
      distance=self.rbf_distance, huber_delta=self.rbf_huber_delta,
      exclude_same_sentence=self.rbf_exclude_same_sentence,
      exclude_same_position=self.rbf_exclude_same_position,
      omit_real_real=self.rbf_omit_real_real,
      omit_fake_fake=self.rbf_omit_fake_fake,
      block_size=self.rbf_block_size, level=self.kernel_level)
    if return_per_layer:
      return reward, active, per_layer
    return reward, active

  @torch.no_grad()
  def paired_reward(self, clean_tokens, generated_tokens, selected,
                    eos_id, attention_mask=None, return_per_layer=False):
    """Per-sequence reward comparing a generated answer against the clean one.

    Both rows go through the *same* forward so they share a kernel launch and
    cannot differ through batching alone -- the concern that makes yrOPDLM
    compare two halves of a single doubled row. Several blocks are read from
    that same forward, so a summed reward costs extra activations, not extra
    compute.
    """
    if clean_tokens.shape != generated_tokens.shape:
      raise ValueError('clean/generated token shapes must match, got '
                       f'{tuple(clean_tokens.shape)} vs '
                       f'{tuple(generated_tokens.shape)}')
    batch = clean_tokens.shape[0]
    stacked = torch.cat([clean_tokens, generated_tokens], dim=0)
    mask = None if attention_mask is None else torch.cat(
      [attention_mask, attention_mask], dim=0)
    feats = self.features(stacked, eos_id, attention_mask=mask)
    reward, active, per_layer = summed_paired_reward(
      feats, batch, selected, self.resolved_layers, self.layer_weights,
      kernel=self.kernel, alpha=self.rbf_alpha,
      reward_type=self.reward_type,
      distance=self.rbf_distance, huber_delta=self.rbf_huber_delta,
      exclude_same_sentence=self.rbf_exclude_same_sentence,
      exclude_same_position=self.rbf_exclude_same_position,
      omit_real_real=self.rbf_omit_real_real,
      omit_fake_fake=self.rbf_omit_fake_fake,
      block_size=self.rbf_block_size, rows_of=self.kernel_batch_size,
      level=self.kernel_level)
    if return_per_layer:
      return reward, active, per_layer
    return reward, active


def _prepare_reinforce_inputs(x0, selected, critic, *, group_size,
                              baseline, scored, kernel_source, kernel_draws):
  """Validate reward settings before any sampling or critic work."""
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
  if x0.ndim != 2:
    raise ValueError(f'clean tokens must be (batch, length), got {tuple(x0.shape)}')
  batch, length = x0.shape
  if selected.shape != x0.shape:
    raise ValueError(f'selected mask must be {tuple(x0.shape)}, got '
                     f'{tuple(selected.shape)}')
  selected = selected.bool()
  feature_mse = getattr(critic, 'reward_type', 'feature') == 'feature_mse'
  # The written span drives sampling and the surrogate; `scored` only ever
  # reaches the critic.
  scored = selected if scored is None else scored.bool()
  if scored.shape != selected.shape:
    raise ValueError(f'scored mask must be {tuple(selected.shape)}, got '
                     f'{tuple(scored.shape)}')
  if feature_mse and not torch.equal(scored, selected):
    raise ValueError('mmd_reward_type=feature_mse requires scored positions '
                     'to equal the written/masked positions '
                     '(mmd_score_positions=written)')

  if kernel_source not in ('rows', 'draws'):
    raise ValueError('mmd_kernel_source must be rows|draws, got '
                     f'{kernel_source!r}')
  if not feature_mse and getattr(critic, 'kernel', None) == 'pseudo_huber' and (
      kernel_source != 'draws' or
      getattr(critic, 'kernel_level', None) != 'sentence'):
    raise ValueError('mmd_kernel=pseudo_huber requires '
                     'mmd_kernel_source=draws and mmd_kernel_level=sentence')
  per_candidate = int(kernel_draws) if kernel_source == 'draws' else 1
  if per_candidate < 1:
    raise ValueError(f'mmd_kernel_batch_size must be >= 1, got {kernel_draws}')
  if kernel_source == 'draws':
    if not hasattr(critic, 'multi_draw_reward'):
      raise ValueError('mmd_kernel_source=draws needs a feature critic exposing '
                       'multi_draw_reward; mmd_reward_type=idlm is unsupported')

  return (batch, length, selected, scored, baseline, per_candidate,
          feature_mse)


@torch.no_grad()
def _score_reinforce_samples(x0, scored, samples, critic, *, group_size,
                             per_candidate, kernel_source, baseline, eos_id,
                             shuffle_rewards=False):
  """Detached scoring sees the complete logical batch, never forward chunks."""
  batch, length = x0.shape
  rewards, actives, breakdown = [], [], {}
  multi = len(getattr(critic, 'resolved_layers', [None])) > 1
  if kernel_source == 'draws':
    # Candidate g is a SET of K answers, scored together as one empirical
    # feature distribution. G remains the outer leave-one-out group.
    blocks = samples.view(group_size, per_candidate, batch, length)
    for group_index in range(group_size):
      if multi:
        out = critic.multi_draw_reward(
          x0, blocks[group_index], scored, eos_id, return_per_layer=True)
      else:
        out = critic.multi_draw_reward(
          x0, blocks[group_index], scored, eos_id)
      member_reward, member_active = out[0], out[1]
      if multi:
        for index, value in out[2].items():
          breakdown.setdefault(index, []).append(value)
      rewards.append(member_reward)
      actives.append(member_active)
  else:
    for group_index in range(group_size):
      out = critic.paired_reward(
        x0, samples[group_index], scored, eos_id,
        return_per_layer=multi)
      member_reward, member_active = out[0], out[1]
      if multi:
        for index, value in out[2].items():
          breakdown.setdefault(index, []).append(value)
      rewards.append(member_reward)
      actives.append(member_active)
  reward = torch.stack(rewards, dim=-1)          # (batch, group)
  active = actives[0]                            # identical across the group

  if shuffle_rewards:
    # Independent permutation per row. Shuffling the whole tensor would also
    # move rewards between sequences, changing each row's baseline and its
    # spread; permuting within the row leaves both exactly as they were and
    # isolates the one thing under test -- which sample gets the credit.
    reward = reward.gather(
      -1, torch.argsort(torch.rand_like(reward), dim=-1))

  if baseline == 'group':
    advantage = group_advantages(reward)
  elif baseline == 'batch':
    advantage = batch_advantages(reward, active)
  else:  # 'none'
    advantage = reward
  advantage = torch.where(active.unsqueeze(-1), advantage,
                          torch.zeros_like(advantage))
  return reward, active, advantage, breakdown


def _reduce_reinforce_samples(sampled_logp, selected, reward, active,
                              advantage, breakdown, *, group_size,
                              per_candidate, feature_mse, reward_weight):
  """Apply the original logical-batch score estimator and logging reductions."""
  batch = selected.shape[0]
  # log p(sampled token), summed over the positions the student wrote.
  surrogate = sampled_logp.new_zeros(())
  for group_index in range(group_size):
    # The action earning one reward is the whole K-draw set, so its joint log
    # probability is the sum of the member log probabilities.
    seq_logp = sampled_logp.new_zeros(batch)
    for draw_index in range(per_candidate):
      token_logp = sampled_logp[group_index * per_candidate + draw_index]
      seq_logp = seq_logp + (
        token_logp * selected.to(token_logp.dtype)).sum(dim=1)
    surrogate = surrogate + (advantage[:, group_index] * seq_logp).sum()

  # Negated: maximizing reward means descending this. Normalized by generated
  # tokens and by the K draws inside a candidate. Deliberately do NOT divide by
  # `group_size`: TinyGSM's historical runs omitted the 1/G normalization, and
  # preserving that omission is required for comparable reward strength.
  # feature_mse already averages K independent errors. Its joint-score
  # estimator needs no second /K: cross-draw score terms cancel in expectation.
  # Keep the historical kernel-MMD normalization unchanged.
  draw_divisor = 1 if feature_mse else per_candidate
  denom = (selected.sum().clamp_min(1) * draw_divisor).to(surrogate.dtype)
  loss = -reward_weight * surrogate / denom

  with torch.no_grad():
    scored = reward[active] if active.any() else reward.new_zeros(1)
    stats = {'mmd/reward': scored.mean().detach(),
             'mmd/advantage_absmean': advantage.abs().mean().detach(),
             'mmd/reward_spread': (reward.std(dim=-1).mean().detach()
                                   if group_size > 1 else
                                   reward.new_zeros(()))}
    # With several blocks summed, the terms differ by orders of magnitude, so
    # the total alone does not say which block is actually driving the ranking.
    # `_spread` is the part that survives the leave-one-out baseline.
    for index, values in breakdown.items():
      per = torch.stack(values, dim=-1)
      stats[f'mmd/reward_l{index}'] = per[active].mean().detach() \
        if active.any() else per.new_zeros(())
      stats[f'mmd/reward_spread_l{index}'] = (
        per.std(dim=-1).mean().detach() if group_size > 1
        else per.new_zeros(()))
  return loss, stats


def mmd_reinforce_loss(student_probs, x0, selected, critic, *,
                       group_size, eos_id, reward_weight=1.0,
                       shuffle_rewards=False, baseline='auto', scored=None,
                       kernel_source='rows', kernel_draws=1):
  """REINFORCE loss driving the student toward the clean answer's AR features.

  `group_size` hard answers are drawn from the student's differentiable
  one-step prediction at the *same* state; each is scored against the clean
  `x0` by pooling the frozen AR's features over the generated positions, and
  the group's leave-one-out mean is the baseline. Gradient reaches the student
  only through `log p(sampled token)` -- the sampling is not differentiable,
  which is the whole reason for a score estimator rather than a
  reparameterized one.

  One reward per sequence multiplies the summed log-probability of every
  position the student wrote. Masked diffusion predicts each masked position
  independently in a single step, so there is no sub-sequence decision unit to
  attribute credit to; OPDLM's per-block credit exists because its model
  commits a block of four tokens at a time.

  Args:
    student_probs: `(batch, length, vocab)`, differentiable, sums to 1.
    x0: `(batch, length)` clean tokens.
    scored: `(batch, length)` bool, or None for `selected`. Which positions the
      REWARD is measured over, as distinct from which the student wrote.
      Widening it is meaningful only because the features are CONTEXTUAL: the
      two token sequences are identical outside the written span, so the extra
      positions contribute through the hidden state moving when its neighbours
      change, not through any token difference. It must never reach the
      log-prob sum -- the student did not write those positions.
    selected: `(batch, length)` bool; the positions the student generated (the
      masked ones), and always the only positions entering the log-probability.
    critic: a `FrozenARFeatures`.
    group_size: samples per state. At >= 2 the group's own leave-one-out mean
      is the baseline. At 1 there is no second sample at the same state, so
      `baseline` decides what stands in; the sampling cost falls with the
      group, since each member costs one critic forward.
    baseline: which control variate subtracts off the reward's offset.
      `auto` picks `group` when group_size >= 2 and `none` when it is 1.
      `batch` uses the other sequences in the batch -- unbiased and free, but
      noisier because TinyGSM rows carry different conditions and masking
      fractions. `none` subtracts nothing; it is mainly a control because it
      leaves the reward's potentially large offset in the estimator.
    shuffle_rewards: permute each row's rewards across the group before the
      baseline, breaking the correspondence between a reward and the sample
      that earned it. A control, not a training mode: the reward *values* are
      untouched, so scale, spread and the leave-one-out statistics are
      distributed exactly as in a real run, and every logged `mmd/*` series
      looks normal. Only the credit assignment is destroyed. If an arm trained
      this way matches one trained honestly, the reward's content is not what
      is driving the result -- which no amount of staring at `mmd/reward` can
      tell you, since that series is identical by construction.
    kernel_source: ``rows`` preserves the historical one-draw candidate and
      lets the critic's kernel batch merge adjacent device-batch rows.
      ``draws`` makes each candidate a set of ``kernel_draws`` complete answers
      sampled from the same student state.
    kernel_draws: number of complete answers inside a candidate when
      ``kernel_source=draws``. This is ``mmd_kernel_batch_size`` at the config
      boundary; it is unrelated to the data-loader batch size in this mode.

  For ``feature_mse``, ``scored`` must equal the written ``selected`` mask.
  K-draw candidates average their K MSE rewards; the surrogate omits the
  historical kernel-MMD /K factor so its expected scale stays constant in K.

  Returns `(loss, stats)`; `loss` already carries `reward_weight`.
  """
  (batch, length, selected, scored, baseline, per_candidate,
   feature_mse) = _prepare_reinforce_inputs(
    x0, selected, critic, group_size=group_size, baseline=baseline,
    scored=scored, kernel_source=kernel_source, kernel_draws=kernel_draws)
  with torch.no_grad():
    flat = student_probs.reshape(-1, student_probs.shape[-1])
    draws = torch.multinomial(
      flat.float(), group_size * per_candidate, replacement=True)
    draws = draws.view(batch, length, -1).permute(2, 0, 1)
    samples = torch.where(selected.unsqueeze(0), draws, x0.unsqueeze(0))
  rewards = _score_reinforce_samples(
    x0, scored, samples, critic, group_size=group_size,
    per_candidate=per_candidate, kernel_source=kernel_source, baseline=baseline,
    shuffle_rewards=shuffle_rewards, eos_id=eos_id)
  logp = torch.log(student_probs.clamp_min(1e-12))
  sampled_logp = torch.stack([
    logp.gather(-1, drawn.unsqueeze(-1)).squeeze(-1) for drawn in samples])
  return _reduce_reinforce_samples(
    sampled_logp, selected, *rewards, group_size=group_size,
    per_candidate=per_candidate, feature_mse=feature_mse,
    reward_weight=reward_weight)


def mmd_reinforce_loss_from_samples(sampled_logp, x0, selected, samples,
                                    critic, *, group_size, eos_id,
                                    reward_weight=1.0,
                                    shuffle_rewards=False, baseline='auto',
                                    scored=None, kernel_source='rows',
                                    kernel_draws=1):
  """Score a logical reward batch after sampling in smaller forward chunks.

  `samples` and differentiable `sampled_logp` both have shape (D, B, L),
  where D = group_size * kernel_draws for source=draws, or group_size for rows.
  Sample ids outside `selected` must be the corresponding clean `x0` ids.
  Only selected positions enter the score estimator. All kernel groups,
  automatic bandwidths, baselines and normalization still see the full B rows,
  independent of the physical student/feature forward size.

  Other arguments and returned statistics match `mmd_reinforce_loss`.
  """
  (batch, length, selected, scored, baseline, per_candidate,
   feature_mse) = _prepare_reinforce_inputs(
    x0, selected, critic, group_size=group_size, baseline=baseline,
    scored=scored, kernel_source=kernel_source, kernel_draws=kernel_draws)
  expected = (group_size * per_candidate, batch, length)
  if samples.shape != expected or sampled_logp.shape != expected:
    raise ValueError('sample ids and sampled log-probabilities must both be '
                     f'{expected}, got {tuple(samples.shape)} and '
                     f'{tuple(sampled_logp.shape)}')
  if samples.dtype != torch.long:
    raise ValueError('sample ids must have dtype torch.long')
  if not sampled_logp.is_floating_point():
    raise ValueError('sampled log-probabilities must be floating point')
  rewards = _score_reinforce_samples(
    x0, scored, samples, critic, group_size=group_size,
    per_candidate=per_candidate, kernel_source=kernel_source, baseline=baseline,
    shuffle_rewards=shuffle_rewards, eos_id=eos_id)
  return _reduce_reinforce_samples(
    sampled_logp, selected, *rewards, group_size=group_size,
    per_candidate=per_candidate, feature_mse=feature_mse,
    reward_weight=reward_weight)

class FrozenBackboneFeatures(_LayeredFeatures):
  """One of this repo's own DiT backbones, frozen, read at a single block.

  Serves both in-vocabulary feature maps, which differ only in `noise_level`:

  * **The diffusion teacher** (`noise_level > 0`). It is the model being
    distilled, so its features already encode what a plausible TinyGSM solution
    looks like, where a generic pretrained LM only encodes plausible text. But a
    denoiser is a function of a *noised* latent and never saw a clean sequence,
    so both rows are masked to `noise_level` first -- lightly, since the point is
    to read the content that is there, not to ask the model to reconstruct it.
  * **A causal AR** (`noise_level = 0`), e.g. S-FLM's `tinygsm/ar.ckpt`. It *did*
    see clean sequences, so the masking is not merely unnecessary but harmful:
    it would inject a difference into a comparison meant to isolate what the
    student wrote. It is also independent of the student, where the teacher
    measures the reward in the student's own feature space.

  Either way the clean and generated rows get the **same** mask, because the
  comparison is paired. Both share the student's vocabulary exactly (S-FLM's AR
  carries 49154 embeddings, covering the pad and mask ids the dataloader
  appends), so the remapping an external tokenizer forces is unnecessary.
  """

  def __init__(self, denoiser, mask_index, layer_index=None, noise_level=0.1,
               layer_weights=None):
    self.denoiser = denoiser
    self.mask_index = mask_index
    self.noise_level = float(noise_level)
    self._resolve_layers(denoiser, layer_index, layer_weights)

  @torch.no_grad()
  def features(self, tokens, sigma):
    """Read all requested blocks with an optional physical forward limit."""
    return self._capture_forward_chunks(
      self.denoiser, tokens.shape[0],
      lambda start, end: self.denoiser(tokens[start:end], sigma[start:end]))

  @torch.no_grad()
  def multi_draw_reward(self, clean_tokens, draws, selected, eos_id,
                        sigma=None, return_per_layer=False):
    """Compare K generated answers from one state with its clean answer."""
    del eos_id  # This backbone uses the student's vocabulary directly.
    if draws.ndim != 3:
      raise ValueError('multi_draw_reward expects (K, batch, length) draws, '
                       f'got {tuple(draws.shape)}')
    draw_count, batch, length = draws.shape
    if clean_tokens.shape != (batch, length):
      raise ValueError(f'clean tokens must be {(batch, length)}, got '
                       f'{tuple(clean_tokens.shape)}')

    # Use one critic-corruption mask for the clean answer and all K draws. That
    # keeps masking luck out of the MMD comparison within a candidate.
    keep = torch.rand(clean_tokens.shape, device=clean_tokens.device)
    drop = (keep < self.noise_level) & selected.bool()
    clean_xt = torch.where(drop, self.mask_index, clean_tokens)
    gen_xt = torch.where(drop.unsqueeze(0), self.mask_index, draws)
    stacked = torch.cat(
      [clean_xt, gen_xt.reshape(draw_count * batch, length)], dim=0)

    if sigma is None:
      sigma = torch.zeros(stacked.shape[0], device=stacked.device,
                          dtype=torch.float32)
    else:
      if sigma.shape[0] != batch:
        raise ValueError(f'sigma must have {batch} rows, got {tuple(sigma.shape)}')
      repeats = (draw_count,) + (1,) * (sigma.ndim - 1)
      sigma = torch.cat([sigma, sigma.repeat(repeats)], dim=0)
    feats = self.features(stacked, sigma)

    reward, active, per_layer = summed_multi_draw_reward(
      feats, batch, draw_count, length, selected, self.resolved_layers,
      self.layer_weights, kernel=self.kernel, alpha=self.rbf_alpha,
      reward_type=self.reward_type,
      distance=self.rbf_distance, huber_delta=self.rbf_huber_delta,
      exclude_same_sentence=self.rbf_exclude_same_sentence,
      exclude_same_position=self.rbf_exclude_same_position,
      omit_real_real=self.rbf_omit_real_real,
      omit_fake_fake=self.rbf_omit_fake_fake,
      block_size=self.rbf_block_size, level=self.kernel_level)
    if return_per_layer:
      return reward, active, per_layer
    return reward, active

  @torch.no_grad()
  def paired_reward(self, clean_tokens, generated_tokens, selected, eos_id,
                    sigma=None, return_per_layer=False):
    """Per-sequence reward comparing a generated answer against the clean one."""
    del eos_id  # same vocabulary; nothing to remap
    if clean_tokens.shape != generated_tokens.shape:
      raise ValueError('clean/generated token shapes must match, got '
                       f'{tuple(clean_tokens.shape)} vs '
                       f'{tuple(generated_tokens.shape)}')
    batch = clean_tokens.shape[0]
    # One mask, applied to both rows, over the scored positions only.
    keep = torch.rand(clean_tokens.shape, device=clean_tokens.device)
    drop = (keep < self.noise_level) & selected.bool()
    clean_xt = torch.where(drop, self.mask_index, clean_tokens)
    gen_xt = torch.where(drop, self.mask_index, generated_tokens)

    stacked = torch.cat([clean_xt, gen_xt], dim=0)
    # The backbone wants the post-`_process_sigma` form: 1-D, one entry per
    # row. MDLM runs `time_conditioning=False`, so that tensor is zeros and the
    # noise level enters only through the masking above.
    if sigma is None:
      sigma = torch.zeros(stacked.shape[0], device=stacked.device,
                          dtype=torch.float32)
    else:
      sigma = torch.cat([sigma, sigma], dim=0)
    feats = self.features(stacked, sigma)
    reward, active, per_layer = summed_paired_reward(
      feats, batch, selected, self.resolved_layers, self.layer_weights,
      kernel=self.kernel, alpha=self.rbf_alpha,
      reward_type=self.reward_type,
      distance=self.rbf_distance, huber_delta=self.rbf_huber_delta,
      exclude_same_sentence=self.rbf_exclude_same_sentence,
      exclude_same_position=self.rbf_exclude_same_position,
      omit_real_real=self.rbf_omit_real_real,
      omit_fake_fake=self.rbf_omit_fake_fake,
      block_size=self.rbf_block_size, rows_of=self.kernel_batch_size,
      level=self.kernel_level)
    if return_per_layer:
      return reward, active, per_layer
    return reward, active


def load_ar_backbone(ckpt_path, config, tokenizer, device):
  """Load a frozen causal AR from one of this repo's checkpoints.

  Built for S-FLM's `tinygsm/ar.ckpt`: a causal DiT trained 250k steps on
  TinyGSM with the student's own SmolLM-135M tokenizer, which makes it a far
  better feature map for a TinyGSM reward than a generic pretrained LM. It loads
  here rather than through `AutoModelForCausalLM` because it is a Lightning
  checkpoint of `algo.AR`, not an HF model.

  The run's own config is reused for everything but the `algo` block, which is
  swapped for an autoregressive one -- the student and this AR are both
  `model=small` at `length=512`, which is why the architectures line up. Only
  the backbone is returned; `ar_decoder_layers` already finds its `.blocks`.
  """
  # Local, not module-level: `algo` imports this module (lazily, inside
  # `_mmd_critic`), so importing it at the top would close the cycle.
  import algo
  import checkpoint_ema
  import omegaconf
  import utils
  from model_assets import resolve_checkpoint

  ckpt_path = resolve_checkpoint(
    ckpt_path, getattr(config, 'model_weights_dir', None))
  if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
      f'MMD AR checkpoint not found: {ckpt_path}\n'
      'Fetch it with: bash scripts/fetch_sflm_ar.sh\n'
      'The released file must be stripped of its pickled tokenizer first -- it '
      'was written by a newer `tokenizers` than requirements.txt pins, so a '
      'plain torch.load fails before reading a single weight.')

  cfg = omegaconf.OmegaConf.create(
    omegaconf.OmegaConf.to_container(config, resolve=True))
  cfg.algo.name = 'ar'
  cfg.algo.parameterization = 'ar'
  cfg.algo.causal_attention = True
  cfg.algo.time_conditioning = False
  cfg.algo.T = 0
  cfg.algo.subs_masking = False
  # A distillation run's config carries the critics; this is a plain AR.
  cfg.adversarial_distill.is_distill = False

  model = algo.AR.load_from_checkpoint(
    ckpt_path, tokenizer=tokenizer, config=cfg, weights_only=False)
  if checkpoint_ema.finetune_weights(config) == 'ema':
    checkpoint_ema.initialize_backbones_from_ema(model)
  model = model.to(device).eval()
  utils.freeze_model(model)
  return model.backbone
