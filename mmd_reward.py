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
    # `algo.backbone=hf_dit`, which the OpenWebText recipe uses: the teacher is
    # then `kuleshov-group/mdlm-owt`'s `MDLM` wrapper, whose blocks sit one
    # level down on the `DITBackbone` it holds. Without this the diffusion
    # feature source raises on OWT while working on TinyGSM's local `dit`.
    getattr(getattr(core, 'backbone', None), 'blocks', None),
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


def resolve_feature_layers(core, layer_index):
  """Resolve `mmd_feature_layer` to an ordered list of blocks and their indices.

  Accepts `None` (the middle block), a single zero-based integer, or a sequence
  of them. The middle is the default for the same reason yrOPDLM picks it: late
  layers specialize toward the next-token head and early ones barely differ from
  the embedding, so neither end says much about whether two sequences *mean* the
  same thing.

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

  selected = selected.bool()
  weight = selected.unsqueeze(-1).to(torch.float32)
  denom = selected.sum(dim=1).clamp_min(1).to(torch.float32).unsqueeze(-1)
  clean_mean = (clean.float() * weight).sum(dim=1) / denom
  gen_mean = (generated.float() * weight).sum(dim=1) / denom
  return gen_mean - clean_mean, selected.any(dim=1)


def paired_sequence_reward(clean, generated, selected):
  """One pseudo-Huber reward per sequence; rows selecting nothing score zero."""
  delta, active = sequence_pooled_delta(clean, generated, selected)
  reward = pseudo_huber_reward(delta)
  return torch.where(active, reward, torch.zeros_like(reward)), active


def summed_paired_reward(features, batch, selected, indices, weights):
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

  Returns `(reward, active, per_layer)`; `per_layer` is unweighted, for logging.
  """
  total, active, per_layer = None, None, {}
  for index, weight in zip(indices, weights):
    stacked = features[index]
    reward, active = paired_sequence_reward(
      stacked[:batch], stacked[batch:], selected)
    per_layer[index] = reward
    total = reward * weight if total is None else total + reward * weight
  return total, active, per_layer


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
               dtype=torch.bfloat16, layer_weights=None):
    import transformers
    self.model = transformers.AutoModelForCausalLM.from_pretrained(
      model_name, torch_dtype=dtype).to(device).eval()
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
    with capture_layer_hidden(self.model, self.layer_index) as (captured, _):
      self.model(input_ids=tokens, attention_mask=attention_mask)
    return captured

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
      feats, batch, selected, self.resolved_layers, self.layer_weights)
    if return_per_layer:
      return reward, active, per_layer
    return reward, active


def mmd_reinforce_loss(student_probs, x0, selected, critic, *,
                       group_size, eos_id, reward_weight=1.0):
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
    selected: `(batch, length)` bool; the positions the student generated (the
      masked ones), which are also the only ones scored.
    critic: a `FrozenARFeatures`.
    group_size: samples per state. Must be >= 2 -- leave-one-out has no
      baseline with one, and an unbaselined score estimator on a reward this
      large is pure variance.

  Returns `(loss, stats)`; `loss` already carries `reward_weight`.
  """
  if group_size < 2:
    raise ValueError('mmd group_size must be >= 2 for a leave-one-out '
                     f'baseline, got {group_size}')
  batch, length, _ = student_probs.shape
  selected = selected.bool()

  # Sampling and the reward are no-grad; only the log-prob gather carries
  # gradient.
  with torch.no_grad():
    flat = student_probs.reshape(-1, student_probs.shape[-1])
    draws = torch.multinomial(flat.float(), group_size, replacement=True)
    draws = draws.view(batch, length, group_size).permute(2, 0, 1)
    # Outside the generated span the answer is the clean one, so the critic
    # sees a row differing from the reference *only* where the student wrote.
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

    advantage = group_advantages(reward)
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
    # With several blocks summed, the terms differ by orders of magnitude, so
    # the total alone does not say which block is actually driving the ranking.
    # `_spread` is the part that survives the leave-one-out baseline.
    for index, values in breakdown.items():
      per = torch.stack(values, dim=-1)
      stats[f'mmd/reward_l{index}'] = per[active].mean().detach() \
        if active.any() else per.new_zeros(())
      stats[f'mmd/reward_spread_l{index}'] = per.std(dim=-1).mean().detach()
  return loss, stats


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
    with capture_layer_hidden(self.denoiser, self.layer_index) as (feats, _):
      self.denoiser(stacked, sigma)
    reward, active, per_layer = summed_paired_reward(
      feats, batch, selected, self.resolved_layers, self.layer_weights)
    if return_per_layer:
      return reward, active, per_layer
    return reward, active


class _CausalBackboneAdapter(torch.nn.Module):
  """Give `owt-ar`'s `DITLegacy` the `(indices, sigma)` call shape used here.

  `FrozenBackboneFeatures` calls its denoiser positionally, as this repo's own
  DiT wants. `DITLegacy.forward` is `(input_ids, attention_mask, ...)`, so the
  same positional call would land `sigma` in `attention_mask` -- and a tensor of
  zeros there reads as "attend to nothing", which would not raise, it would
  quietly return features computed under a broken mask. Hence an explicit
  adapter rather than a positional call that happens to typecheck.

  `blocks` is a property, not an assignment: assigning the inner `ModuleList`
  would register the same 12 blocks a second time under this module.
  """

  def __init__(self, inner):
    super().__init__()
    self.inner = inner

  @property
  def blocks(self):
    return self.inner.blocks

  def forward(self, indices, sigma=None):
    # A causal LM has no noise conditioning, and owt-ar is `adaln: false`, so
    # the backbone ignores sigma anyway. Dropped explicitly to document that
    # the AR feature map reads *clean* sequences (`noise_level=0`).
    del sigma
    return self.inner(input_ids=indices)


def load_owt_ar_backbone(model_id_or_path, device):
  """Load `kuleshov-group/owt-ar` as a frozen feature map.

  The OpenWebText analogue of the TinyGSM branch's S-FLM AR: a causal DiT
  trained on the student's exact corpus and vocabulary (50258 embeddings,
  matching `mdlm-owt` token for token), so no id remapping is needed and
  `FrozenBackboneFeatures` can be used directly rather than `FrozenARFeatures`.

  Loaded by instantiating the vendored backbone rather than through
  `AutoModelForCausalLM`: the published `trust_remote_code` module imports
  `transformers.models.qwen3`, which transformers 4.38.2 in this environment
  does not have, so the auto path fails before reading a weight. Only the
  backbone is needed -- `ar_decoder_layers` finds its `.blocks` through the
  adapter.
  """
  import json

  import huggingface_hub

  from models import backbone_dit_legacy

  if os.path.isdir(model_id_or_path):
    config_path = os.path.join(model_id_or_path, 'config.json')
    weights_path = os.path.join(model_id_or_path, 'pytorch_model.bin')
  else:
    config_path = huggingface_hub.hf_hub_download(
      model_id_or_path, 'config.json')
    weights_path = huggingface_hub.hf_hub_download(
      model_id_or_path, 'pytorch_model.bin')

  with open(config_path) as f:
    hf_config = json.load(f)
  backbone_config = dict(hf_config['backbone_config'])
  backbone_config.pop('_target_', None)
  # Never let the vendored backbone pull weights of its own; the published
  # owt-ar tensors are loaded explicitly below.
  backbone_config['pretrained_model_name_or_path'] = None
  model = backbone_dit_legacy.DITLegacy(**backbone_config)

  state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
  if isinstance(state_dict, dict) and 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
  backbone_state = {key[len('backbone.'):]: value
                    for key, value in state_dict.items()
                    if key.startswith('backbone.')}
  missing, _ = model.load_state_dict(backbone_state, strict=False)
  if missing:
    raise RuntimeError(
      f'Missing weights when loading owt-ar backbone: {missing}')

  model = _CausalBackboneAdapter(model).to(device).eval()
  for param in model.parameters():
    param.requires_grad_(False)
  return model
