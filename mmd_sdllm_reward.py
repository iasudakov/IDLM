"""Frozen MMD rewards over native, variable-length SDLLM token features.

The student stays on its original vocabulary. Its sampled text is retokenized
by the feature model. OWT scores all native positions; TinyGSM scores native
answer positions with the prompt retained as context. There is no projection
back to student tokens. ``same_position`` means equal ordinal positions in
the independently retokenized scored spans (answer-relative for TinyGSM).
"""
from __future__ import annotations

import math

import torch

import mmd_rbf
from mmd_reward import FrozenBackboneFeatures, median_alpha


def _pack_ragged(rows):
  """Pad only kernel inputs, never inputs to the bidirectional feature model."""
  if not rows:
    raise ValueError('SDLLM feature kernel needs at least one sequence')
  first = rows[0]
  if first.ndim != 2:
    raise ValueError('SDLLM token features must have shape (tokens, hidden)')
  hidden = first.shape[1]
  width = max(1, max(row.shape[0] for row in rows))
  packed = first.new_zeros((len(rows), width, hidden), dtype=torch.float32)
  mask = first.new_zeros((len(rows), width), dtype=torch.float32)
  for i, row in enumerate(rows):
    if row.ndim != 2 or row.shape[1] != hidden or row.device != first.device:
      raise ValueError('SDLLM ragged features must share hidden size and device')
    count = row.shape[0]
    packed[i, :count] = row.detach().float()
    mask[i, :count] = 1.
  return packed, mask


@torch.no_grad()
def _kernel_sum(left, right, alpha, *, block_size=0,
                distance='squared_euclidean', huber_delta=.001,
                exclude_same_position=False):
  """One ragged pair's blocked kernel sum, with FP32 kernel arithmetic."""
  left, right = left.detach().float(), right.detach().float()
  if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
    raise ValueError('SDLLM kernel pairs must be (tokens, hidden) matrices')
  if left.device != right.device:
    raise ValueError('SDLLM kernel pairs must use the same device')
  block = (mmd_rbf.pick_block_size(1, max(left.shape[0], right.shape[0], 1))
           if int(block_size) <= 0 else int(block_size))
  block = max(1, block)
  total = left.new_zeros(())
  with torch.autocast(device_type=left.device.type, enabled=False):
    for i0 in range(0, left.shape[0], block):
      a = left[i0:i0 + block].unsqueeze(0)
      for j0 in range(0, right.shape[0], block):
        b = right[j0:j0 + block].unsqueeze(0)
        if distance == 'squared_euclidean':
          kernel = mmd_rbf._kernel_block(
            a, b, mmd_rbf._sq_norms(a), mmd_rbf._sq_norms(b), alpha)
        else:
          kernel = mmd_rbf.pairwise_distance(
            a, b, distance=distance, huber_delta=huber_delta,
            block_size=block)
          kernel.mul_(-alpha).exp_()
        if exclude_same_position and i0 == j0:
          kernel.diagonal(dim1=1, dim2=2).zero_()
        total.add_(kernel.sum())
  return total


@torch.no_grad()
def ragged_cross_sentence_mmd2(real, fake_draws, alpha, *, block_size=0,
                              distance='squared_euclidean', huber_delta=.001,
                              omit_fake_fake=False, exclude_same_position=False):
  """Cross-sentence signed statistic for one real row and K ragged draws.

  Each real/fake term is normalized by its own n*m_r valid native token
  pairs, then averaged over draws. Each distinct fake-sentence pair is
  normalized by m_r*m_s (minus min(m_r,m_s) when matching native positions
  are excluded), then averaged over ordered distinct draw pairs. This gives
  independent draws equal weight rather than weighting sentences by length.

  At equal lengths this is the existing cross-sentence RBF estimator. As in
  that estimator, the entire real-real term is omitted; negative estimates
  remain negative. No reference or generated features are position-aligned.
  """
  mmd_rbf.validate_distance(distance, huber_delta)
  draws = len(fake_draws)
  if draws < 2:
    raise ValueError('cross-sentence SDLLM RBF needs at least 2 fake draws')
  n = real.shape[0]
  lengths = [fake.shape[0] for fake in fake_draws]
  active = n > 0 and all(length > 0 for length in lengths)
  for r in range(draws):
    for s in range(r + 1, draws):
      denominator = lengths[r] * lengths[s]
      if exclude_same_position:
        denominator -= min(lengths[r], lengths[s])
      active = active and denominator > 0
  active_tensor = torch.tensor(active, device=real.device, dtype=torch.bool)
  if not active:
    return real.new_zeros((), dtype=torch.float32), active_tensor
  common = dict(block_size=block_size, distance=distance,
                huber_delta=huber_delta)
  cross = real.new_zeros((), dtype=torch.float32)
  fake = real.new_zeros((), dtype=torch.float32)
  for r in range(draws):
    cross.add_(_kernel_sum(real, fake_draws[r], alpha, **common)
               / (n * lengths[r]))
    if not omit_fake_fake:
      for s in range(r + 1, draws):
        denominator = lengths[r] * lengths[s]
        if exclude_same_position:
          denominator -= min(lengths[r], lengths[s])
        fake.add_(_kernel_sum(
          fake_draws[r], fake_draws[s], alpha,
          exclude_same_position=exclude_same_position, **common)
          / denominator, alpha=2.)
  statistic = fake / (draws * (draws - 1)) - 2. * cross / draws
  return statistic, active_tensor


class FrozenSdllmFeatures(FrozenBackboneFeatures):
  """Native-token feature map implementing the existing frozen-critic API.

  ``feature_model`` supplies:
    retokenize(student_ids, source_tokenizer, source_mask_index) -> list[list[int]]
    features(native_ids, layer_indices, forward_batch_size=...) ->
      dict[layer_index, list[Tensor[native_length, hidden]]]

  Native OWT positions include BOS/EOS and document separators. TinyGSM
  selects the answer after a separately tokenized context prefix. The
  feature model itself is responsible for unpadded forwards, strict context
  bounds, frozen/eval behavior and native tokenizer validation.
  """
  def __init__(self, feature_model, source_tokenizer, mask_index,
               layer_index=None, layer_weights=None, noise_level=0.0):
    if float(noise_level) != 0.:
      raise ValueError('native SDLLM MMD features require mmd_noise_level=0')
    super().__init__(feature_model, mask_index, layer_index=layer_index,
                     layer_weights=layer_weights, noise_level=0.)
    self.source_tokenizer = source_tokenizer
    self.source_family = getattr(feature_model, 'source_family', 'gpt2')
    if self.source_family not in ('gpt2', 'tinygsm'):
      raise ValueError(f'Unsupported SDLLM source family {self.source_family!r}')

  def _validate_settings(self, selected, tokens, *, sequence_mask=None,
                         draw_count=None):
    if self.reward_type != 'feature' or self.kernel != 'rbf' or self.kernel_level != 'token':
      raise ValueError('native SDLLM features support feature/token/RBF rewards only')
    if selected.shape != tokens.shape:
      raise ValueError('native SDLLM scored mask must match source token shape')
    if self.source_family == 'gpt2':
      if not bool(selected.bool().all()):
        raise ValueError('native SDLLM features require scoring all OWT positions')
      if sequence_mask is not None and (
          sequence_mask.shape != tokens.shape or not bool(sequence_mask.bool().all())):
        raise ValueError('native SDLLM OWT features do not support conditional/padded masks')
    else:
      # The reference content mask fixes the same answer window for every
      # candidate. It is not shortened at a sampled EOS or PAD token.
      for row in selected.bool():
        indices = row.nonzero(as_tuple=False).flatten()
        if not indices.numel() or int(indices[-1] - indices[0] + 1) != indices.numel():
          raise ValueError('TinyGSM SDLLM needs one nonempty contiguous answer span per row')
      if sequence_mask is not None and (
          sequence_mask.shape != tokens.shape or
          not torch.equal(sequence_mask.bool(), selected.bool())):
        raise ValueError('TinyGSM SDLLM sequence mask must match the reference answer span')
    if self.sentence_pooling != 'mean':
      raise ValueError('native SDLLM token features require sentence_pooling=mean')
    if self.rbf_exclude_same_position and not self.rbf_exclude_same_sentence:
      raise ValueError('same-position exclusion requires same-sentence exclusion')
    if self.rbf_exclude_same_sentence and (draw_count is None or draw_count < 2):
      raise ValueError('cross-sentence SDLLM RBF requires draws mode with K>=2')
    mmd_rbf.validate_distance(self.rbf_distance, self.rbf_huber_delta)
    if self.rbf_alpha is not None:
      alpha = float(self.rbf_alpha)
      if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('SDLLM RBF alpha must be finite and nonnegative or null')

  @torch.no_grad()
  def features(self, tokens, sigma=None, selected=None):
    del sigma  # Frozen MDLM feature extraction uses zero timestep conditioning.
    if self.source_family == 'tinygsm':
      if selected is None:
        raise ValueError('TinyGSM SDLLM features require the reference answer mask')
      native, starts = self.denoiser.retokenize_conditioned(
        tokens, self.source_tokenizer, self.mask_index, selected)
    else:
      native = self.denoiser.retokenize(
        tokens, self.source_tokenizer, self.mask_index)
      starts = [0] * len(native)
    if len(native) != tokens.shape[0]:
      raise ValueError('SDLLM retokenizer changed the number of sequences')
    if len(starts) != len(native) or any(
        not isinstance(start, int) or not 0 <= start <= len(row)
        for row, start in zip(native, starts)):
      raise ValueError('SDLLM retokenizer returned invalid scored-span offsets')
    # Existing REINFORCE code assumes every candidate has the same active mask.
    # Reject sample-dependent inactive rows instead of silently changing its
    # baseline or counting an unscorable candidate as zero reward.
    if any(len(row) - start < 2 for row, start in zip(native, starts)):
      raise ValueError('native SDLLM MMD needs at least 2 native tokens per '
                       'scored span; short rows cannot be silently deactivated')
    captured = self.denoiser.features(
      native, list(self.resolved_layers),
      forward_batch_size=self.forward_batch_size)
    for index in self.resolved_layers:
      values = captured[index]
      if len(values) != len(native):
        raise ValueError('SDLLM feature model changed the number of rows')
      for position, (ids, start, value) in enumerate(zip(native, starts, values)):
        if value.ndim != 2 or value.shape[0] != len(ids):
          raise ValueError('SDLLM native token features must match native ID lengths')
        if value.device != tokens.device:
          raise ValueError('SDLLM native features must be on the student device')
        values[position] = value[start:]
    return captured

  def _layer_alpha(self, real):
    if self.rbf_alpha is not None:
      return self.rbf_alpha
    packed, mask = _pack_ragged(real)
    return median_alpha(packed, mask.bool(), distance=self.rbf_distance,
                        huber_delta=self.rbf_huber_delta)

  def _ordinary_layer_reward(self, real, fake, alpha):
    packed_real, real_mask = _pack_ragged(real)
    packed_fake, fake_mask = _pack_ragged(fake)
    with torch.autocast(device_type=packed_real.device.type, enabled=False):
      statistic, active = mmd_rbf.rbf_mmd2_unequal(
        packed_real, packed_fake, real_mask, fake_mask, alpha,
        block_size=self.rbf_block_size, distance=self.rbf_distance,
        huber_delta=self.rbf_huber_delta,
        omit_real_real=self.rbf_omit_real_real,
        omit_fake_fake=self.rbf_omit_fake_fake)
    return -statistic, active

  @torch.no_grad()
  def paired_reward(self, clean_tokens, generated_tokens, selected, eos_id,
                    sigma=None, return_per_layer=False, sequence_mask=None):
    del eos_id, sigma
    if clean_tokens.ndim != 2 or clean_tokens.shape != generated_tokens.shape:
      raise ValueError('SDLLM paired rewards require matching (batch,length) IDs')
    self._validate_settings(selected, clean_tokens, sequence_mask=sequence_mask)
    batch = clean_tokens.shape[0]
    rows_of = int(self.kernel_batch_size)
    if rows_of < 1 or batch % rows_of:
      raise ValueError('SDLLM kernel rows must divide the student batch')
    # Preserve the original noise-level-zero critic's single RNG draw.
    torch.rand(clean_tokens.shape, device=clean_tokens.device)
    captured = self.features(torch.cat([clean_tokens, generated_tokens], dim=0),
                             selected=selected.repeat(2, 1))
    total, active, per_layer = None, None, {}
    for index, weight in zip(self.resolved_layers, self.layer_weights):
      values = captured[index]
      real, fake = values[:batch], values[batch:]
      alpha = self._layer_alpha(real)
      if rows_of > 1:
        real = [torch.cat(real[i:i + rows_of], dim=0) for i in range(0, batch, rows_of)]
        fake = [torch.cat(fake[i:i + rows_of], dim=0) for i in range(0, batch, rows_of)]
      reward, current_active = self._ordinary_layer_reward(real, fake, alpha)
      if rows_of > 1:
        reward = reward.repeat_interleave(rows_of)
        current_active = current_active.repeat_interleave(rows_of)
      per_layer[index] = reward
      total = reward * weight if total is None else total + reward * weight
      active = current_active if active is None else active & current_active
    return (total, active, per_layer) if return_per_layer else (total, active)

  @torch.no_grad()
  def multi_draw_reward(self, clean_tokens, draws, selected, eos_id,
                        sigma=None, return_per_layer=False, sequence_mask=None):
    del eos_id, sigma
    if draws.ndim != 3 or clean_tokens.ndim != 2 or draws.shape[1:] != clean_tokens.shape:
      raise ValueError('SDLLM multi-draw rewards need (K,batch,length) draws')
    count, batch, length = draws.shape
    if count < 1:
      raise ValueError('SDLLM multi-draw rewards need at least one fake draw')
    self._validate_settings(selected, clean_tokens, sequence_mask=sequence_mask,
                            draw_count=count)
    torch.rand(clean_tokens.shape, device=clean_tokens.device)
    captured = self.features(torch.cat(
      [clean_tokens, draws.reshape(count * batch, length)], dim=0),
      selected=selected.repeat(count + 1, 1))
    total, active, per_layer = None, None, {}
    for index, weight in zip(self.resolved_layers, self.layer_weights):
      values = captured[index]
      real = values[:batch]
      fake_by_row = [[values[(k + 1) * batch + i] for k in range(count)]
                     for i in range(batch)]
      alpha = self._layer_alpha(real)
      if self.rbf_exclude_same_sentence:
        outputs = [ragged_cross_sentence_mmd2(
          x, ys, alpha, block_size=self.rbf_block_size,
          distance=self.rbf_distance, huber_delta=self.rbf_huber_delta,
          omit_fake_fake=self.rbf_omit_fake_fake,
          exclude_same_position=self.rbf_exclude_same_position)
          for x, ys in zip(real, fake_by_row)]
        reward = -torch.stack([out[0] for out in outputs])
        current_active = torch.stack([out[1] for out in outputs])
      else:
        fake = [torch.cat(rows, dim=0) for rows in fake_by_row]
        reward, current_active = self._ordinary_layer_reward(real, fake, alpha)
      per_layer[index] = reward
      total = reward * weight if total is None else total + reward * weight
      active = current_active if active is None else active & current_active
    return (total, active, per_layer) if return_per_layer else (total, active)
