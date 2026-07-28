import os
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils
import hydra.utils
import models
from ipdb import set_trace as debug


def _maybe_save_periodic_checkpoint(module):
  # Periodic checkpointing is handled by main.PeriodicStepCheckpoint.
  return


def _conditional_valid_tokens(config, valid_tokens):
  if getattr(config.data, 'train', None) == 'tiny_gsm':
    return valid_tokens
  return None


def _apply_top_p_mask(log_probs: torch.Tensor, top_p: float, neg_infinity: float):
    """
    log_probs: (B, V) log-probabilities (already log_softmax'd)
    Returns: log_probs with tokens outside nucleus set to -inf.
    """
    if top_p is None or top_p >= 1.0:
        return log_probs  # p=1 -> no change -> same results as before

    # Convert to probs for sorting / cumulative mass
    probs = log_probs.exp()  # (B, V)

    # Sort probs descending
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  # (B, V)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, V)

    # Keep smallest set with cumulative mass >= top_p
    # We mark tokens to REMOVE where cum_probs > top_p, but ensure at least 1 token kept.
    sorted_remove = cum_probs > top_p
    sorted_remove[..., 0] = False  # always keep the top token

    # Map remove-mask back to original vocab positions
    remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_idx, sorted_remove)  # (B, V)

    # Mask out removed tokens in log space
    log_probs = log_probs.masked_fill(remove, neg_infinity)
    return log_probs

class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step, train_mode=None,
          valid_tokens=None):
    del current_accumulation_step, train_mode, valid_tokens
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]

  def generate_samples(self, num_samples, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    for i in range(num_pred_tokens):
      output = self.backbone(x[:, :i + 1], None)
      output[:, :, self.mask_index] = self.neg_infinity
      # log-probs for the next token
      log_probs = output[:, -1, :].log_softmax(-1)  # (B, V)

      log_probs = _apply_top_p_mask(log_probs, top_p=0.95, neg_infinity=self.neg_infinity)

      y = (log_probs + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None

def _load_owt_ar_backbone(model_id_or_path):
  """Loads the causal (autoregressive) DiT backbone from the
  `kuleshov-group/owt-ar` checkpoint (or a local directory holding the same
  `config.json` / `pytorch_model.bin`).

  The upstream repo ships its full `AutoModelForCausalLM` behind
  `trust_remote_code`, but that code imports `transformers.models.qwen3`, which
  is unavailable in this environment's transformers. We only need the causal
  DiT backbone (vendored at `models/backbone_dit_legacy.py`) that maps a token
  sequence to next-token logits, so we instantiate it directly and load just
  its weights.
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
  # Never let the vendored backbone try to load weights itself; we load the
  # published owt-ar weights explicitly below.
  backbone_config['pretrained_model_name_or_path'] = None
  model = backbone_dit_legacy.DITLegacy(**backbone_config)

  state_dict = torch.load(
    weights_path, map_location='cpu', weights_only=False)
  if isinstance(state_dict, dict) and 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
  backbone_state = {
    key[len('backbone.'):]: value
    for key, value in state_dict.items()
    if key.startswith('backbone.')}
  missing, _ = model.load_state_dict(backbone_state, strict=False)
  assert not missing, (
    f'Missing weights when loading owt-ar backbone: {missing}')
  return model


def _embedding_std(embed_module):
  """Per-entry std of a token-embedding module's weight matrix.

  Used to calibrate the noise-conditioning init to the loaded checkpoint rather
  than to a freshly-initialised backbone. Returns the DiT init scale (0.02) if
  no weight can be located, so the behaviour degrades to the previous constant.
  """
  for attr in ('embedding', 'weight'):
    w = getattr(embed_module, attr, None)
    if isinstance(w, torch.Tensor) and w.numel() > 0:
      return float(w.detach().float().std())
  return 0.02


class _NoiseConditionedBackbone(torch.nn.Module):
  """Wraps a denoiser backbone and injects a learned projection of Gaussian
  noise into the token-embedding residual (D-MMD Sec. 6.5). By default the
  noise is drawn independently per token position; with `shared_noise` a single
  vector per sample is broadcast across all embeddings (same noise everywhere).

  A masked-diffusion generator is otherwise a deterministic function of its
  (masked) input, so the moment-matching objective can only be minimised by
  collapsing its output distribution -- which destroys sample diversity. An
  input noise source lets the generator stay diverse while sharpening its
  factorised output. Fresh noise is drawn on every forward pass and added to
  the embedding via a forward hook, so it is applied uniformly across training
  (`multistep_generation`) and sampling (`generate_samples`).

  The projection's *output* is rescaled by `embed_std / sqrt(noise_dim)`, a
  constant measured from the *actual* embedding matrix of `embed_module` at
  construction time. That makes the parameterization dimensionless: a unit-RMS
  `W` injects a vector as large as the token identity, so `noise/snr ~= rms(W)`
  and, under Adam (per-entry `|dW| ~= lr` per step), snr grows at ~`noise_lr`
  per step whatever `noise_dim` or checkpoint is in use -- i.e. `noise_lr`
  transfers. The constant adds no expressiveness (`c * (W z)` spans the same
  functions as `W' z`); it fixes the units, which is what makes the *optimizer*
  behave comparably across settings.

  Calibrating to the loaded checkpoint matters when distilling: a hardcoded init
  calibrated to a freshly-initialised DiT (`kaiming_uniform_`, per-entry std
  ~= 0.021) is ~15x too small for `kuleshov-group/mdlm-owt`, whose trained
  embedding has std ~= 0.309. At that scale two noise draws move the student's
  output distribution by a total-variation distance of only ~0.09, i.e. the
  generator is effectively noise-free and behaves like a run with `noise_dim=0`.
  The mirror-image failure is a large init (snr ~= 10): the embedding residual
  is then ~90% noise, the student cannot read its own masked context, and a
  pretrained checkpoint has to spend thousands of steps just attenuating `W`
  before distillation means anything. Hence the zero-init default -- see
  `noise_scale` in `configs/config.yaml`.
  """

  def __init__(self, backbone, embed_module, noise_dim, hidden_size,
               shared_noise=False, noise_scale=1.0):
    super().__init__()
    self.backbone = backbone
    self.noise_dim = noise_dim
    # True => one Gaussian vector per sample, broadcast across every token
    # embedding; False => an independent vector per token position.
    self.shared_noise = shared_noise
    self.noise_scale = noise_scale
    self.noise_proj = torch.nn.Linear(noise_dim, hidden_size)
    # The projection output is rescaled by a constant measured once, here, from
    # the *actual* embedding matrix, so a unit-RMS `W` injects a vector the size
    # of the token identity: `snr == rms(W)`. Fixed at construction (the
    # embedding drifts during training, and recomputing it per forward would
    # both cost a reduction over the whole matrix and feed back into itself).
    # Falls back to the DiT init scale if the module exposes no weight.
    embed_std = _embedding_std(embed_module)
    # Non-persistent: recomputed from the embedding at every construction, so it
    # adds no state_dict key (an old noise-conditioned checkpoint would
    # otherwise fail Lightning's strict resume with a missing key) and it is not
    # a parameter, so it stays out of `_student_param_groups` / weight decay.
    self.register_buffer(
      'noise_out_scale',
      torch.tensor(embed_std / (noise_dim ** 0.5), dtype=torch.float32),
      persistent=False)
    # `noise_scale` is the initial RMS of `W`, i.e. the initial `noise/snr`.
    # 0.0 (default) => zero-init: the wrapped student starts as an exact copy of
    # the checkpoint's function and grows its noise usage only if the objective
    # pays for it. The gradient into `W` is `noise_out_scale * g_out z^T`, which
    # is non-zero because `z` is, so a zero start still trains (the standard
    # zero-init residual branch, cf. AdaLN-zero / zero-convs / LoRA `B=0`).
    if noise_scale:
      torch.nn.init.normal_(self.noise_proj.weight, std=noise_scale)
    else:
      torch.nn.init.zeros_(self.noise_proj.weight)
    torch.nn.init.zeros_(self.noise_proj.bias)
    self._noise = None
    embed_module.register_forward_hook(self._add_noise)

  def _add_noise(self, module, inputs, output):
    if self._noise is None:
      return output
    # Shared noise has length 1 and broadcasts over the position axis.
    injected = self.noise_out_scale * self.noise_proj(self._noise)
    return output + injected.to(output.dtype)

  def forward(self, xt, sigma=None, *args, **kwargs):
    length = 1 if self.shared_noise else xt.shape[1]
    self._noise = torch.randn(
      xt.shape[0], length, self.noise_dim,
      device=xt.device, dtype=torch.float32)
    try:
      return self.backbone(xt, sigma, *args, **kwargs)
    finally:
      self._noise = None


def _wrap_backbone_with_noise(backbone, noise_dim, default_hidden,
                              shared_noise=False, noise_scale=1.0):
  """Locate the token-embedding module of `backbone` and wrap it so a learned
  Gaussian-noise projection is added to the embedding residual."""
  embed_module = None
  for name, module in backbone.named_modules():
    if name.endswith('vocab_embed'):
      embed_module = module
      break
  assert embed_module is not None, (
    'Could not locate a `vocab_embed` module for noise conditioning.')
  if hasattr(embed_module, 'embedding'):
    hidden_size = embed_module.embedding.shape[-1]
  elif hasattr(embed_module, 'embedding_dim'):
    hidden_size = embed_module.embedding_dim
  else:
    hidden_size = default_hidden
  return _NoiseConditionedBackbone(
    backbone, embed_module, noise_dim, hidden_size,
    shared_noise=shared_noise, noise_scale=noise_scale)


class MDLM_with_AR(trainer_base.AbsorbingState):
  """MDLM student distilled with *autoregressive* teacher / fake critics.

  This follows the IDLM adversarial-distillation recipe, but replaces the
  diffusion teacher and fake models with autoregressive language models (the
  `owt-ar` checkpoint). The frozen teacher supplies the data-distribution
  signal and the trainable fake model tracks the student's own distribution;
  the generator loss remains `teacher_loss - fake_loss`.

  Both critics score the student's (clean) generated sequence by their
  per-position next-token log-probabilities, and those log-probs are combined
  with the differentiable student distribution `student_probs` exactly the way
  the diffusion denoiser log-probs were, so the gradient to the student flows
  through `student_probs` while the token context is treated as a constant.
  """

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()
    self.is_adversarial_distill = self.config.adversarial_distill.is_distill
    # Resolved lazily by `_student_embed_std` for the noise diagnostics.
    self._embed_module_cache = None
    if self.is_adversarial_distill:
      self.automatic_optimization = False
      # Critic type: 'ar' uses autoregressive (owt-ar) teacher/fake; 'diffusion'
      # uses copies of the masked-diffusion student as critics (original IDLM).
      self.critic_type = getattr(
        self.config.adversarial_distill, 'critic_type', 'ar')
      assert self.critic_type in ('ar', 'diffusion'), (
        f'Unknown adversarial_distill.critic_type: {self.critic_type}')

      # Whether to use the trainable fake critic at all. If False, the fake
      # model is not created and the generator is trained with only the teacher
      # term (no min-max, no fake steps).
      self.use_fake = getattr(self.config.adversarial_distill, 'use_fake', True)

      if self.critic_type == 'ar':
        ar_model_path = getattr(
          self.config.adversarial_distill, 'ar_model_path',
          'kuleshov-group/owt-ar')
        # Frozen AR teacher: the pretrained data-distribution model.
        self.teacher = _load_owt_ar_backbone(ar_model_path)
        # Trainable AR "fake" model, from the same owt-ar weights.
        self.fake_model = (
          _load_owt_ar_backbone(ar_model_path) if self.use_fake else None)
      else:
        # Original IDLM: teacher/fake are diffusion denoisers, copied from the
        # (raw, un-noise-conditioned) student before it is wrapped below.
        self.teacher = copy.deepcopy(self.backbone)
        self.fake_model = (
          copy.deepcopy(self.backbone) if self.use_fake else None)
      utils.freeze_model(self.teacher)
      if self.use_fake:
        utils.activate_model(self.fake_model)

      # Optional generator input-noise conditioning (D-MMD Sec. 6.5): wrap the
      # student backbone so a learned Gaussian-noise projection is injected into
      # the embedding residual. Done AFTER copying the diffusion critics (so they
      # stay noise-free) and before (re)building the EMA/optimizer so the
      # projection is tracked/trained like the rest of the backbone.
      noise_dim = getattr(self.config.adversarial_distill, 'noise_dim', 0)
      if noise_dim and noise_dim > 0:
        noise_shared = getattr(
          self.config.adversarial_distill, 'noise_shared', False)
        noise_scale = getattr(
          self.config.adversarial_distill, 'noise_scale', 1.0)
        self.backbone = _wrap_backbone_with_noise(
          self.backbone, noise_dim, self.config.model.hidden_size,
          shared_noise=noise_shared, noise_scale=noise_scale)
      utils.activate_model(self.backbone)
      self.ema_fake = None
      if self.ema is not None:
        del self.ema
        self.ema = models.ema.ExponentialMovingAverage(
          self.backbone.parameters(),
          decay=self.config.training.ema)

        if self.use_fake:
          self.ema_fake = models.ema.ExponentialMovingAverage(
            self.fake_model.parameters(),
            decay=self.config.training.ema)
      self.training_step_counter = 0
      self.accum_batches = self.config.adversarial_distill.accum_batches
      self._fake_accum_counter = 0
      self._student_accum_counter = 0
      self.fake_frequency = self.config.adversarial_distill.fake_frequency
      self.fake_iter = 0
      # With no fake critic, every step is a generator step.
      self.is_generator_step = not self.use_fake
      self.total_loss_student = 0
      self.total_loss_fake = 0
      self.f_loss_ar = []
      # Weight of the auxiliary->teacher regularizer in the fake loss (D-MMD
      # Eq. 12). 0 => original fake loss (hard NLL on student samples only).
      self.fake_teacher_weight = getattr(
        self.config.adversarial_distill, 'fake_teacher_weight', 0.0)
      # Fit the fake critic to the student's soft distribution instead of to the
      # one-hot sampled tokens (D-MMD Sec. 3.2). Applies to both critic types:
      # the AR critic switches between `_ar_cross_entropy` and `_ar_nll`, the
      # diffusion critic between `student_probs` and one-hot `x_hard` (see
      # `_fake_diffusion_target`). Only the *fake/auxiliary* step is affected;
      # the generator always differentiates the soft path.
      #
      # `null` keeps each critic's own original default rather than imposing one
      # on both, so no existing run changes behaviour: the AR critic was written
      # with the hard next-token NLL, the diffusion critic with the soft target.
      fake_soft_target = getattr(
        self.config.adversarial_distill, 'fake_soft_target', None)
      if fake_soft_target is None:
        fake_soft_target = (self.critic_type == 'diffusion')
      self.fake_soft_target = bool(fake_soft_target)
      print(f'[adversarial_distill] critic_type={self.critic_type}, '
            f'fake_soft_target={self.fake_soft_target}', flush=True)

  def _student_param_groups(self):
    """Student optimizer parameter groups. If the backbone is noise-conditioned
    and `adversarial_distill.noise_lr` is set, the learned noise-projection
    matrix gets its own learning rate; every other parameter uses optim.lr."""
    base_lr = self.config.optim.lr
    noise_lr = getattr(self.config.adversarial_distill, 'noise_lr', None)
    if noise_lr is None or not hasattr(self.backbone, 'noise_proj'):
      return [{'params': list(self.backbone.parameters()), 'lr': base_lr}]
    noise_params = list(self.backbone.noise_proj.parameters())
    noise_ids = {id(p) for p in noise_params}
    other_params = [p for p in self.backbone.parameters()
                    if id(p) not in noise_ids]
    return [
      {'params': other_params, 'lr': base_lr},
      {'params': noise_params, 'lr': noise_lr}]

  def configure_optimizers(self):
    if self.config.adversarial_distill.is_distill:
      optimizer_student = torch.optim.AdamW(
        self._student_param_groups(),
        lr=self.config.optim.lr,
        betas=(self.config.optim.beta1,
              self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)

      scheduler_student = hydra.utils.instantiate(
        self.config.lr_scheduler, optimizer=optimizer_student)
      scheduler_student_dict = {'scheduler': scheduler_student, 'name': 'trainer/lr'}

      if not self.use_fake:
        # No fake critic: a single student optimizer.
        return [optimizer_student], [scheduler_student_dict]

      # The fake critic adapts to the student's distribution and needs a real
      # learning rate; it should not share the (often tiny) generator LR.
      fake_lr = getattr(self.config.adversarial_distill, 'fake_lr', None)
      if fake_lr is None:
        fake_lr = self.config.optim.lr
      optimizer_fake = torch.optim.AdamW(
        self.fake_model.parameters(),
        lr=fake_lr,
        betas=(self.config.optim.beta1,
              self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)

      scheduler_fake = hydra.utils.instantiate(
        self.config.lr_scheduler, optimizer=optimizer_fake)
      scheduler_fake_dict = {'scheduler': scheduler_fake, 'name': 'trainer/lr_fake'}
      return [optimizer_student, optimizer_fake], [scheduler_student_dict, scheduler_fake_dict]
    else:
      return super().configure_optimizers()

  def _validate_configuration(self):
    # ancestral sampling isn't desirable because it's slow
    assert self.sampler == 'ancestral_cache'

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def on_train_start(self):
    if self.is_adversarial_distill:
      if self.use_fake:
        # Fake-first schedule: start on a fake step (student frozen).
        self.ema_fake.move_shadow_params_to_device(self.device)
        utils.activate_model(self.fake_model)
        utils.freeze_model(self.backbone)
      else:
        # No fake critic: the student trains every step.
        utils.activate_model(self.backbone)
        self.is_generator_step = True
    super().on_train_start()

  def _sync_submodule_state(self, state_dict, prefix, module):
    """Reconcile `state_dict` with `module`'s params under `prefix.`.

    The AR teacher / fake model have a different architecture than the student
    backbone, so a checkpoint may either lack their weights (e.g. distilling
    from a plain MDLM student) or carry incompatible ones (e.g. an old
    diffusion-critic checkpoint). We drop stray / shape-mismatched entries and
    fill any missing keys from the freshly-loaded owt-ar module, so a strict
    load succeeds without clobbering genuinely-resumed weights.
    """
    current = module.state_dict()
    valid = {f'{prefix}.{name}': param for name, param in current.items()}
    for key in list(state_dict.keys()):
      if key.startswith(prefix + '.') and (
          key not in valid or state_dict[key].shape != valid[key].shape):
        del state_dict[key]
    for key, param in valid.items():
      state_dict.setdefault(key, param.detach().clone())

  def on_load_checkpoint(self, checkpoint):
    if self.is_adversarial_distill:
      state_dict = collections.OrderedDict(checkpoint['state_dict'])
      self._sync_submodule_state(state_dict, 'teacher', self.teacher)
      if self.fake_model is not None:
        self._sync_submodule_state(state_dict, 'fake_model', self.fake_model)
      checkpoint['state_dict'] = state_dict
    super().on_load_checkpoint(checkpoint)
    if self.is_adversarial_distill and self.ema_fake is not None:
      # Only restore a genuine fake-model EMA; never fall back to the student
      # EMA, whose parameters have the (incompatible) student shapes.
      ema_fake_state = checkpoint.get('ema_fake', None)
      if ema_fake_state is not None:
        self.ema_fake.load_state_dict(ema_fake_state)

  def on_save_checkpoint(self, checkpoint):
    if self.is_adversarial_distill and self.critic_type == 'ar':
      # The frozen AR teacher is reloaded from owt-ar; don't persist it. The
      # diffusion teacher is a frozen copy of the pretrained student (not
      # reloadable from a fixed path), so it must stay in the checkpoint.
      checkpoint['state_dict'] = collections.OrderedDict(
        (k, v) for k, v in checkpoint['state_dict'].items()
        if not k.startswith('teacher.'))
    super().on_save_checkpoint(checkpoint)
    if self.is_adversarial_distill and self.ema_fake is not None:
      checkpoint['ema_fake'] = self.ema_fake.state_dict()

  def _preserve_prompt_probs(self, probs, x0, valid_tokens):
    if valid_tokens is None:
      return probs
    clean_probs = F.one_hot(x0, num_classes=probs.shape[-1]).to(probs.dtype)
    return torch.where(valid_tokens[..., None].bool(), probs, clean_probs)

  def _preserve_prompt_tokens(self, tokens, x0, valid_tokens):
    if valid_tokens is None:
      return tokens
    return torch.where(valid_tokens.bool(), tokens, x0)

  def multistep_generation(self, x0, valid_tokens, current_accumulation_step):
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)

    log_x_theta = self.forward(xt, sigma=sigma)
    student_probs = log_x_theta.exp()
    return self._preserve_prompt_probs(student_probs, x0, valid_tokens)

  def _ar_logprobs(self, ar_backbone, tokens):
    """Per-position next-token log-probabilities from an AR backbone.

    Args:
      ar_backbone: a causal `DITLegacy` backbone (teacher or fake model).
      tokens: int tensor (batch, length) of a full token sequence.

    Returns:
      log-probs of shape (batch, length - 1, vocab_size), where index `i`
      holds `log p(tokens[i + 1] | tokens[<= i])`, aligned so that it can be
      scored against `student_probs[:, 1:]`.
    """
    with torch.cuda.amp.autocast(dtype=torch.float32):
      output = ar_backbone(input_ids=tokens[:, :-1])
    logits = output.logits if hasattr(output, 'logits') else output[0]
    log_probs = logits.float().log_softmax(-1)
    # Align the AR vocabulary width to the student's vocabulary. The owt-ar
    # checkpoint already matches (both 50258 over the GPT-2 tokenizer), but be
    # defensive so a different checkpoint doesn't silently misalign.
    vocab = self.vocab_size
    if log_probs.shape[-1] > vocab:
      log_probs = log_probs[..., :vocab]
    elif log_probs.shape[-1] < vocab:
      log_probs = F.pad(
        log_probs, (0, vocab - log_probs.shape[-1]),
        value=self.neg_infinity)
    return log_probs

  def _ar_cross_entropy(self, ar_logprobs, target_probs, valid_tokens):
    """Token-averaged cross-entropy of the student distribution against an AR
    critic's next-token log-probabilities.

    `target_probs[:, i]` (the student distribution over token `i`) is scored by
    `ar_logprobs[:, i - 1]` (`log p(token_i | token_{<i})`), so we drop the
    first (contextless) position. Only one side carries gradient per call: in
    the generator step the critic log-probs are constants and the gradient
    reaches the student through `target_probs`; in the (soft-target) fake step
    `target_probs` is detached and the gradient trains the critic.
    """
    target = target_probs[:, 1:, :]
    loss = -(target * ar_logprobs).sum(-1)  # (batch, length - 1)
    if valid_tokens is not None:
      valid = valid_tokens[:, 1:]
      loss = loss * valid
      denom = valid.sum(-1).clamp_min(1)
      return loss.sum(-1) / denom
    return loss.sum(-1)

  def _ar_nll(self, ar_logprobs, tokens, valid_tokens):
    """Token-averaged NLL of the hard sampled sequence under an AR critic.

    Standard next-token cross-entropy: `ar_logprobs[:, i]` scores the sampled
    token `tokens[:, i + 1]`, so we gather against the tokens from position 1
    onward (dropping the first, contextless position).
    """
    targets = tokens[:, 1:]
    loss = -ar_logprobs.gather(
      -1, targets[..., None]).squeeze(-1)  # (batch, length - 1)
    if valid_tokens is not None:
      valid = valid_tokens[:, 1:]
      loss = loss * valid
      denom = valid.sum(-1).clamp_min(1)
      return loss.sum(-1) / denom
    return loss.sum(-1)

  def q_xt_with_probs(self, student_probs, alpha_t, x0=None, valid_tokens=None):
    """Diffuse a (soft) student distribution to noise level t for the diffusion
    critic (masked diffusion). Verbatim from the original MDLM implementation."""
    mask_probs = torch.zeros_like(student_probs)
    mask_probs[:, :, self.mask_index] = 1
    noise_probs = alpha_t[:, :, None] * student_probs + (1 - alpha_t[:, :, None])*(mask_probs)
    if x0 is not None and valid_tokens is not None:
      noise_probs = self._preserve_prompt_probs(noise_probs, x0, valid_tokens)
    return noise_probs

  def nll_per_token_with_probs(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, valid_tokens=None):
    """Soft weighted cross-entropy for the diffusion critic. Verbatim from the
    original MDLM implementation (`x0` is the target distribution)."""
    del xt
    loss = (x0 * log_x_theta).sum(-1)
    loss = loss * dalpha_t / (1 - alpha_t)
    if valid_tokens is not None:
      loss = loss * valid_tokens
      denom = valid_tokens.sum(-1).clamp_min(1)
      return loss.sum(-1) / denom
    return loss.sum(-1)

  def generator_loss(self, x0, valid_tokens, current_accumulation_step):
    if self.critic_type == 'diffusion':
      return self._generator_loss_diffusion(
        x0, valid_tokens, current_accumulation_step)
    return self._generator_loss_ar(x0, valid_tokens, current_accumulation_step)

  def _generator_loss_ar(self, x0, valid_tokens, current_accumulation_step):
    # Differentiable student distribution over the (clean) generated sequence.
    student_probs = self.multistep_generation(
      x0, valid_tokens, current_accumulation_step)

    # Discrete sequence fed as context to the AR critics (non-differentiable).
    x_gen = trainer_base.sample_categorical(student_probs.detach())
    x_gen = self._preserve_prompt_tokens(x_gen, x0, valid_tokens)

    # Both critics are frozen during the generator step; their outputs act as
    # constants and gradients reach the student only through `student_probs`.
    with torch.no_grad():
      teacher_logprobs = self._ar_logprobs(self.teacher, x_gen)
    teacher_loss = self._ar_cross_entropy(
      teacher_logprobs, student_probs, valid_tokens)
    if not self.use_fake:
      return teacher_loss

    with torch.no_grad():
      fake_logprobs = self._ar_logprobs(self.fake_model, x_gen)
    fake_loss = self._ar_cross_entropy(
      fake_logprobs, student_probs, valid_tokens)
    return teacher_loss - fake_loss

  def _generator_loss_diffusion(self, x0, valid_tokens,
                                current_accumulation_step):
    # Original IDLM (diffusion critic). Verbatim from the `MDLM` implementation.
    student_probs = self.multistep_generation(x0, valid_tokens, current_accumulation_step)

    num_noised_samples = getattr(self.config.adversarial_distill, 'generator_num_noised_samples', 1)

    teacher_loss_total = 0.0
    fake_loss_total = 0.0

    for _ in range(num_noised_samples):

      print(_)

      t = self._sample_t(x0.shape[0], current_accumulation_step)
      dalpha_t, alpha_t = self.noise(t)
      alpha_t = alpha_t.unsqueeze(-1)
      assert alpha_t.ndim == 2
      sigma = self._sigma_from_alphat(alpha_t)
      sigma = self._process_sigma(sigma)

      probs_xt = self.q_xt_with_probs(student_probs, alpha_t, x0=x0, valid_tokens=valid_tokens)
      xt = trainer_base.sample_categorical(probs_xt)
      xt = self._preserve_prompt_tokens(xt, x0, valid_tokens)

      with torch.cuda.amp.autocast(dtype=torch.float32):
        model_output_teacher = self.teacher(xt, sigma)
      logits_teacher = self._process_model_output(model_output=model_output_teacher, xt=xt, sigma=sigma)
      teacher_loss = self.nll_per_token_with_probs(logits_teacher, xt, student_probs,
                                  alpha_t=alpha_t, dalpha_t=dalpha_t,
                                  valid_tokens=valid_tokens)
      
      teacher_loss_total = teacher_loss_total + teacher_loss

      if self.use_fake:
        with torch.cuda.amp.autocast(dtype=torch.float32):
          model_output_fake = self.fake_model(xt, sigma)
        logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
        fake_loss = self.nll_per_token_with_probs(logits_fake, xt, student_probs,
                                    alpha_t=alpha_t, dalpha_t=dalpha_t,
                                    valid_tokens=valid_tokens)

        fake_loss_total = fake_loss_total + fake_loss

    teacher_loss = teacher_loss_total / num_noised_samples
    fake_loss = fake_loss_total / num_noised_samples
      
    if not self.use_fake:
      return teacher_loss
    
    return teacher_loss - fake_loss

  def _reduce_token_loss(self, per_token_loss, valid_tokens):
    """Reduce a (batch, length - 1) per-token loss to (batch,).

    Matches _ar_nll / _ar_cross_entropy: masked token-mean when valid_tokens is
    given (using positions 1..L-1), otherwise a plain sum over positions.
    """
    if valid_tokens is not None:
      valid = valid_tokens[:, 1:]
      per_token_loss = per_token_loss * valid
      return per_token_loss.sum(-1) / valid.sum(-1).clamp_min(1)
    return per_token_loss.sum(-1)

  def fake_loss(self, x0, valid_tokens, current_accumulation_step):
    if self.critic_type == 'diffusion':
      return self._fake_loss_diffusion(
        x0, valid_tokens, current_accumulation_step)
    return self._fake_loss_ar(x0, valid_tokens, current_accumulation_step)

  def _fake_loss_ar(self, x0, valid_tokens, current_accumulation_step):
    # Auxiliary/fake objective (D-MMD Eq. 12). Fit the fake AR critic to the
    # student's samples AND anchor it to the frozen teacher; the anchor
    # stabilizes the min-max and keeps the critic from drifting into a
    # degenerate state (it does not change the fixed point). The generator step
    # still scores the fake against the soft `student_probs` (the differentiable
    # DMD gradient path).
    with torch.no_grad():
      student_probs = self.multistep_generation(
        x0, valid_tokens, current_accumulation_step)
      x_gen = trainer_base.sample_categorical(student_probs)
      x_gen = self._preserve_prompt_tokens(x_gen, x0, valid_tokens)

    fake_logprobs = self._ar_logprobs(self.fake_model, x_gen)
    # Term 1: fit the critic to the student's own distribution.
    if self.fake_soft_target:
      # CE(student_probs || fake): score *every* category by the student's
      # probability rather than only the sampled one. Same expectation as the
      # hard NLL below (x_gen ~ student_probs) but without the per-position
      # sampling noise, and it is exactly the quantity the generator step
      # differentiates -- so critic and generator optimize the same functional.
      # The critic's context stays the hard sample `x_gen`, as it is in the
      # generator step (an AR backbone consumes discrete ids).
      loss = self._ar_cross_entropy(fake_logprobs, student_probs, valid_tokens)
    else:
      # CE(x | fake) -- standard next-token NLL on the hard student sample.
      loss = self._ar_nll(fake_logprobs, x_gen, valid_tokens)
    # Term 2: CE(teacher || fake) -- anchor the fake to the teacher's next-token
    # distribution on the same context.
    if self.fake_teacher_weight > 0:
      with torch.no_grad():
        teacher_probs = self._ar_logprobs(self.teacher, x_gen).exp()
      teacher_ce = -(teacher_probs * fake_logprobs).sum(-1)  # (batch, length-1)
      loss = loss + self.fake_teacher_weight * self._reduce_token_loss(
        teacher_ce, valid_tokens)
    return loss

  def _fake_diffusion_target(self, student_probs, x_hard):
    """Target the diffusion fake critic regresses onto (D-MMD Sec. 3.2).

    `fake_soft_target=True` -> the student's soft prediction `x_hat_eta(z_t)`;
    `False` -> the one-hot hard sample `x ~ Cat(x_hat_eta(z_t))` that the critic
    latent was actually drawn from. For *masked* diffusion the paper states both
    are valid ("it is therefore equally valid to use either the soft x_hat_eta or
    the hard x"), because a masked `z_s` carries no information about `x` at the
    masked positions; the soft target is just its Rao-Blackwellisation and has
    lower variance. Note the caveat in the same section: for **uniform**
    diffusion the auxiliary model must be trained on hard samples, so this switch
    is masked-diffusion-only.

    Returned as a distribution over the vocabulary either way, so both feed the
    same weighted soft cross-entropy `nll_per_token_with_probs`.
    """
    if self.fake_soft_target:
      return student_probs
    return F.one_hot(
      x_hard, num_classes=student_probs.shape[-1]).to(student_probs.dtype)

  def _fake_loss_diffusion(self, x0, valid_tokens, current_accumulation_step):
    # Original IDLM (diffusion critic), with the D-MMD Sec. 3.2 choice of hard
    # vs. soft target exposed as `fake_soft_target`.
    with torch.no_grad():
      student_probs = self.multistep_generation(x0, valid_tokens, current_accumulation_step)

    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)
    sigma = self._process_sigma(sigma)
    # Draw the hard student sample, then mask it to the critic latent. This is
    # marginally identical to the original one-shot draw from
    # `q_xt_with_probs(student_probs, alpha_t)` -- both give P(mask) = 1 - alpha_t
    # and P(c) = alpha_t * student_probs_c -- but it exposes the `x_hard` the
    # latent is coupled to, which the hard target needs. Doing it the same way in
    # both modes keeps the latent identical across a soft/hard A/B at fixed seed,
    # so the comparison isolates the target.
    x_hard = trainer_base.sample_categorical(student_probs)
    x_hard = self._preserve_prompt_tokens(x_hard, x0, valid_tokens)
    xt = self.q_xt(x_hard, alpha_t, valid_tokens=valid_tokens)
    xt = self._preserve_prompt_tokens(xt, x0, valid_tokens)

    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_fake = self.fake_model(xt, sigma)
    logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
    fake_loss = self.nll_per_token_with_probs(
                                logits_fake, xt,
                                self._fake_diffusion_target(
                                  student_probs, x_hard),
                                alpha_t=alpha_t, dalpha_t=dalpha_t,
                                valid_tokens=valid_tokens)

    # Optional aux->teacher anchor (D-MMD Eq. 12, second term). Disabled by
    # default (fake_teacher_weight=0), so the above matches the original MDLM.
    if self.fake_teacher_weight > 0:
      with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
        teacher_probs = self._process_model_output(
          model_output=self.teacher(xt, sigma), xt=xt, sigma=sigma).exp()
      anchor = self.nll_per_token_with_probs(logits_fake, xt, teacher_probs,
                                alpha_t=alpha_t, dalpha_t=dalpha_t,
                                valid_tokens=valid_tokens)
      fake_loss = fake_loss + self.fake_teacher_weight * anchor
    return fake_loss

  def _noise_conditioning_stats(self):
    """Diagnostics for the generator's input-noise conditioning, or `{}` when
    it is disabled (`noise_dim=0`).

    Logged every generator step so it is visible whether the student is
    actually *using* the noise source or ignoring it:

    * `noise/proj_norm` -- Frobenius norm of the learned projection. Flat means
      the generator is not learning to use the noise; the run is then
      equivalent to `noise_dim=0` regardless of `noise_lr`.
    * `noise/inject_std` -- per-hidden-dim std of the injected vector,
      `noise_out_scale * ||W||_F / sqrt(hidden)` for unit-Gaussian input. The
      `noise_out_scale` factor is what `_add_noise` actually applies; dropping it
      here would under-report the injection by `embed_std / sqrt(noise_dim)`
      (~37x at `noise_dim=128` on mdlm-owt). Compare against `noise/embed_std`.
    * `noise/embed_std` -- per-entry std of the student's token embedding
      (constant up to training drift), the scale the injection competes with.
    * `noise/snr` -- ratio of the two, and with the output rescaling also just
      `rms(W)`. At ~0.06 the injection is a few-percent perturbation that
      LayerNorm largely absorbs; ~1.0 means the noise is as large as the token
      identity itself; ~10 means the token identity is a tenth of the residual
      and the student can barely read its own input.
    """
    proj = getattr(self.backbone, 'noise_proj', None)
    if proj is None:
      return {}
    with torch.no_grad():
      w = proj.weight.detach().float()
      proj_norm = w.norm()
      # E[||Wz||^2] = ||W||_F^2 for z ~ N(0, I), spread over `hidden` dims,
      # then the constant output rescaling `_add_noise` applies.
      out_scale = float(getattr(self.backbone, 'noise_out_scale', 1.0))
      inject_std = out_scale * proj_norm / (w.shape[0] ** 0.5)
      embed_std = self._student_embed_std()
    return {
      'noise/proj_norm': proj_norm,
      'noise/inject_std': inject_std,
      'noise/embed_std': torch.as_tensor(embed_std, device=w.device),
      'noise/snr': inject_std / max(embed_std, 1e-12)}

  def _student_embed_std(self):
    """Per-entry std of the (possibly noise-wrapped) student's embedding."""
    if self._embed_module_cache is None:
      inner = getattr(self.backbone, 'backbone', self.backbone)
      for name, module in inner.named_modules():
        if name.endswith('vocab_embed'):
          self._embed_module_cache = module
          break
    if self._embed_module_cache is None:
      return 0.02
    return _embedding_std(self._embed_module_cache)

  def _clip_gradients(self, parameters, name):
    """Optionally clip gradients in the manual-optimization loop; log the norm.

    With ``automatic_optimization=False`` Lightning does NOT apply
    ``trainer.gradient_clip_val``, so this is the only clipping the adversarial
    steps get. Unclipped steps combined with the diffusion weight
    ``w(s) = dalpha_s / (1 - alpha_s)`` (huge near ``s -> eps``) can make the
    generator/critic losses diverge (student_loss -> -inf, fake_loss -> +inf,
    entropy -> 0), so watch the logged norm when clipping is off.

    Clipping is opt-in via ``adversarial_distill.use_gradient_clipping``
    (default False); when enabled the accumulated gradient is clipped to
    ``adversarial_distill.gradient_clipping_val`` (None or <= 0 also disables)
    right before ``optimizer.step()``. The pre-clip norm is logged in both
    cases, so disabling clipping does not cost the diagnostic.
    """
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
      return
    ad = self.config.adversarial_distill
    clip_val = getattr(ad, 'gradient_clipping_val', None)
    if (getattr(ad, 'use_gradient_clipping', False)
        and clip_val is not None and clip_val > 0):
      norm = torch.nn.utils.clip_grad_norm_(parameters, clip_val)
    else:
      # Monitoring only: the same total norm `clip_grad_norm_` would report,
      # with the gradients left untouched.
      norm = torch.norm(torch.stack(
        [p.grad.detach().norm(2) for p in parameters]), 2)
    self.log(name, norm, on_step=True, on_epoch=False,
             prog_bar=False, sync_dist=True)

  def _generator_only_step(self, input_tokens, valid_tokens,
                           current_accumulation_step):
    # No fake critic: every step trains the student on the (teacher-only)
    # generator loss.
    optimizer_student = self.optimizers()
    scheduler_student = self.lr_schedulers()
    self.toggle_optimizer(optimizer_student)
    g_loss = self.generator_loss(
      input_tokens, valid_tokens, current_accumulation_step)
    g_loss = g_loss.mean() / self.accum_batches
    self.manual_backward(g_loss)
    self.untoggle_optimizer(optimizer_student)

    self._student_accum_counter += 1
    self.total_loss_student += g_loss.detach()
    if self._student_accum_counter == self.accum_batches:
      self._student_accum_counter = 0
      self._clip_gradients(self.backbone.parameters(), 'grad_norm_student')
      optimizer_student.step()
      self.ema.update(self.backbone.parameters())
      optimizer_student.zero_grad()
      _maybe_save_periodic_checkpoint(self)
      self.log_dict(
        {"student_loss": self.total_loss_student,
         **self._noise_conditioning_stats()},
        prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
      scheduler_student.step()
      self.total_loss_student = 0

  def training_step(self, batch, batch_idx):
    if self.config.adversarial_distill.is_distill:
      current_accumulation_step = (batch_idx % self.accum_batches)

      (input_tokens, output_tokens, valid_tokens) = self._process_model_input(batch['input_ids'],
                                                                              batch['attention_mask'])
      valid_tokens = _conditional_valid_tokens(self.config, valid_tokens)

      if not self.use_fake:
        return self._generator_only_step(
          input_tokens, valid_tokens, current_accumulation_step)

      optimizer_student, optimizer_fake = self.optimizers()
      scheduler_student, scheduler_fake = self.lr_schedulers()
      finish_iter = False
      if self.is_generator_step:
        self.toggle_optimizer(optimizer_student)
        g_loss = self.generator_loss(input_tokens, valid_tokens, current_accumulation_step)
        g_loss = g_loss.mean()
        g_loss = g_loss / self.accum_batches
        self.manual_backward(g_loss)

        self.untoggle_optimizer(optimizer_student)

        self._student_accum_counter += 1
        self.total_loss_student += g_loss.detach()
        if self._student_accum_counter == self.accum_batches:
          self._student_accum_counter = 0
          self._clip_gradients(self.backbone.parameters(), 'grad_norm_student')
          optimizer_student.step()
          self.ema.update(self.backbone.parameters())
          optimizer_student.zero_grad()
          finish_iter = True
      else:
        self.toggle_optimizer(optimizer_fake)
        f_loss = self.fake_loss(input_tokens, valid_tokens, current_accumulation_step)
        f_loss = f_loss.mean()
        f_loss = f_loss / self.accum_batches
        self.manual_backward(f_loss)

        self.untoggle_optimizer(optimizer_fake)
        self._fake_accum_counter += 1
        self.total_loss_fake += f_loss.detach()
        if self._fake_accum_counter == self.accum_batches:
          self._fake_accum_counter = 0
          self._clip_gradients(self.fake_model.parameters(), 'grad_norm_fake')
          optimizer_fake.step()
          self.ema_fake.update(self.fake_model.parameters())
          optimizer_fake.zero_grad()
          finish_iter = True

      if finish_iter:
        _maybe_save_periodic_checkpoint(self)

      if finish_iter and self.is_generator_step:
        self.is_generator_step = False
        utils.activate_model(self.fake_model)
        utils.freeze_model(self.backbone)
        optimizer_fake.zero_grad()
        self.log_dict(
            {"student_loss": self.total_loss_student,
             "fake_loss": torch.stack(self.f_loss_ar).mean(),
             **self._noise_conditioning_stats()},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        scheduler_student.step()
        scheduler_fake.step()
        self.total_loss_student = 0
        self.f_loss_ar.clear()
      elif finish_iter:
        self.fake_iter += 1
        self.f_loss_ar.append(self.total_loss_fake)
        self.total_loss_fake = 0
        if self.fake_iter == self.fake_frequency:
          self.is_generator_step = True
          self.fake_iter = 0
          utils.activate_model(self.backbone)
          utils.freeze_model(self.fake_model)
          optimizer_student.zero_grad()

    else:
      return super().training_step(batch, batch_idx)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def _get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    # score(x, t) = p_t(y) / p_t(x)
    # => log score(x, t) = log p_t(y) - log p_t(x)
    
    # case 1: x = masked
    #   (i) y = unmasked
    #     log score(x, t) = log p_\theta(x)|_y + log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    #   (ii) y = masked
    #     log score(x, t) = 0

    # case 2: x = unmasked
    #   (i) y != masked, y != x
    #     log score(x_i, t) = - inf
    #   (ii) y = x 
    #     log score(x_i, t) = 0
    #   (iii) y = masked token
    #     log score(x_i, t) = - log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1
    
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_index] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(
      model_output)
    unmasked_score = torch.scatter(
      unmasked_score,
      -1,
      x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, self.mask_index] = - (
      log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == self.mask_index).to(
      model_output.dtype)[:, :, None]
    model_output = (
      masked_score * masked_indices
      + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class MDLM_DMMD(MDLM_with_AR):
  """MDLM distilled with the Discrete Moment-Matching (D-MMD) recipe of
  Hoogeboom et al., "Beyond Single Tokens" (arXiv:2603.20155).

  This subclasses `MDLM_with_AR` and inherits *all* of its machinery unchanged
  (optimizers, EMA, checkpointing, the fake/generator min-max schedule, the AR
  critic, generator input-noise conditioning, samplers, ...). It only replaces
  the diffusion-critic loss to add two features from the paper, each independently
  toggleable so their effect on training can be compared:

  1. Mask sampling scheme -- `adversarial_distill.dmmd_mask_sampling`:
     * 'posterior' ("s", D-MMD Algorithm 1, default): couple two times `s < t`,
       generate the student prediction `x_hat_eta(z_t)` at the *noisier* latent
       `z_t`, sample a hard `x ~ Cat(x_hat_eta(z_t))`, and draw the *less-noisy*
       critic latent from the masked-diffusion posterior `z_s ~ q(z_s | z_t, x)`
       (revealing some masked positions toward `x`). The critic is scored at `z_s`
       with weight `w(s)`. The step count `k` is `adversarial_distill.
       dmmd_num_steps`; the two times are coupled as in Algorithm 1 --
       `s ~ U(0, 1)`, `delta_t ~ U(0, 1/k)`, `t = min(1, s + delta_t)`.
     * 'renoise' ("t", original IDLM `MDLM_with_AR`): generate `student_probs`,
       then *re-noise* it back to an independent time `t` (`q_xt_with_probs`) and
       score the critic at `t`. This is the parent's exact scheme (identical when
       `generator_num_noised_samples == 1`), kept here so the two mask-sampling
       strategies can be A/B-tested with everything else held fixed.
     Aliases accepted: {'posterior','s','dmmd'} / {'renoise','t','idlm'}.

  2. Top-p (nucleus) teacher distillation (D-MMD Sec. 3.3), applied in *both*
     modes. To make the student mode-seek like a top-p sampler, the teacher
     target is restricted to the top-p nucleus. Hard-masking out-of-nucleus
     logits to a large negative value spikes the gradient reaching the student
     (the paper reports divergence), so instead their logits are *lowered by a
     constant* `Delta` (paper: `Delta = 2`):
     `s_theta <- s_theta - (1 - mask_top-p) * Delta`. Set via
     `adversarial_distill.top_p` (>= 1 disables) and `.top_p_delta`. Top-p also
     applies to the AR critic's teacher; the posterior mask sampling is specific
     to the diffusion critic.
  """

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    ad = self.config.adversarial_distill
    # Nucleus threshold for D-MMD Sec. 3.3 teacher-target top-p distillation.
    # None or >= 1 disables it (teacher target left unchanged).
    self.top_p = getattr(ad, 'top_p', 1.0)
    if self.top_p is None:
      self.top_p = 1.0
    # Constant by which out-of-nucleus teacher logits are lowered (paper Delta=2).
    self.top_p_delta = getattr(ad, 'top_p_delta', 2.0)
    # Number of D-MMD sampling steps k; the generation time t is drawn one step
    # above the uniform critic time s via delta_t ~ U(0, 1/k) (posterior only).
    self.dmmd_num_steps = getattr(ad, 'dmmd_num_steps', 32)
    # Mask-sampling scheme: 'posterior' (D-MMD, critic at s) or 'renoise' (IDLM,
    # critic at re-noised t). Normalized so 's'/'dmmd' and 't'/'idlm' also work.
    mode = str(getattr(ad, 'dmmd_mask_sampling', 'posterior')).lower()
    if mode in ('posterior', 's', 'dmmd'):
      self.dmmd_mask_sampling = 'posterior'
    elif mode in ('renoise', 't', 'idlm'):
      self.dmmd_mask_sampling = 'renoise'
    else:
      raise ValueError(
        f'Unknown adversarial_distill.dmmd_mask_sampling: {mode!r} '
        "(expected one of 'posterior'/'s' or 'renoise'/'t').")

  def _dmmd_sample_t(self, s):
    """Generation time `t` one D-MMD step above the critic time `s`.

    Algorithm 1's coupling: `delta_t ~ U(0, 1/k)` (k = `dmmd_num_steps`),
    `t = min(1, s + delta_t)`. `t >= s`, so `alpha_t <= alpha_s` -- `z_s` is a
    *less* noisy latent than `z_t`, matching the reverse (denoising) direction
    of the posterior.

    The paper draws the *critic* time uniformly and derives `t` from it, which
    is the opposite of an earlier version of this method (uniform `t`, then
    `s = t - delta_t`). It matters in two ways: the loss weight `w(s)` and the
    critic's mask density are now driven by a genuinely uniform `s`, and the
    clamp puts an atom of mass `1/(2k)` (~1.6% at k=32) on `t = 1`, i.e. the
    generator is sometimes trained on the near-fully-masked input it starts
    sampling from. Runs from before this change are therefore not exactly
    comparable.
    """
    delta = torch.rand_like(s) / self.dmmd_num_steps
    return (s + delta).clamp(max=1.0)

  def _dmmd_posterior(self, z_t, x_hard, alpha_t, alpha_s):
    """Masked-diffusion posterior sample `z_s ~ q(z_s | z_t, x_hard)`.

    Mirrors `AbsorbingState._ancestral_update` (a reverse step from `t` to `s`)
    but conditions on the *hard* student sample `x_hard` instead of a soft model
    prediction: positions already revealed in `z_t` are kept, and each still-
    masked position is revealed to `x_hard` with probability
    `(alpha_s - alpha_t) / (1 - alpha_t)`, otherwise stays masked.
    """
    reveal_prob = ((alpha_s - alpha_t) / (1 - alpha_t)).clamp(0.0, 1.0)
    reveal = torch.rand(z_t.shape, device=z_t.device) < reveal_prob
    newly_revealed = (z_t == self.mask_index) & reveal
    return torch.where(newly_revealed, x_hard, z_t)

  def _dmmd_generate(self, x0, valid_tokens, current_accumulation_step,
                     detach_student):
    """One critic draw: `(student_probs, crit_x, alpha, dalpha, x_hard)`.

    `student_probs` is the differentiable student prediction; `crit_x` is the
    discrete latent fed to the teacher/fake critic, scored with weight from
    `(alpha, dalpha)`; `x_hard ~ Cat(student_probs)` is the hard sample the
    latent is coupled to (Algorithm 1's `x`), needed by the hard fake target.
    The latent is produced by the D-MMD posterior ('posterior' mode) or by IDLM
    re-noising ('renoise' mode). When `detach_student` the student forward pass
    is run under `no_grad` (fake / auxiliary step).
    """
    if self.dmmd_mask_sampling == 'renoise':
      # IDLM: generate student_probs, then re-noise to an independent time t.
      if detach_student:
        with torch.no_grad():
          student_probs = self.multistep_generation(
            x0, valid_tokens, current_accumulation_step)
      else:
        student_probs = self.multistep_generation(
          x0, valid_tokens, current_accumulation_step)
      t = self._sample_t(x0.shape[0], current_accumulation_step)
      dalpha_t, alpha_t = self.noise(t)
      alpha_t = alpha_t.unsqueeze(-1)
      assert alpha_t.ndim == 2
      # Hard sample then mask, rather than one draw from the mixture: same
      # marginal, but exposes `x_hard` for the hard fake target (see
      # `MDLM_with_AR._fake_loss_diffusion`).
      x_hard = trainer_base.sample_categorical(student_probs.detach())
      x_hard = self._preserve_prompt_tokens(x_hard, x0, valid_tokens)
      xt = self.q_xt(x_hard, alpha_t, valid_tokens=valid_tokens)
      xt = self._preserve_prompt_tokens(xt, x0, valid_tokens)
      return student_probs, xt, alpha_t, dalpha_t, x_hard

    # D-MMD posterior (Algorithm 1): couple s < t and sample z_s from q(z_s|z_t,x).
    # `s ~ U(eps, 1)` is the critic time (`_sample_t` is just the repo's uniform
    # draw, with antithetic sampling / global-batch chunking); `t` is one step
    # above it. The `eps` floor is what keeps the `w(s) = -alpha'_s/(1-alpha_s)`
    # weight bounded (~1e3 at s=eps).
    s = self._sample_t(x0.shape[0], current_accumulation_step)
    t = self._dmmd_sample_t(s)
    _, alpha_t = self.noise(t)
    dalpha_s, alpha_s = self.noise(s)
    alpha_t = alpha_t.unsqueeze(-1)
    alpha_s = alpha_s.unsqueeze(-1)
    assert alpha_t.ndim == 2 and alpha_s.ndim == 2

    # Generate the student prediction at the (noisier) generation latent z_t.
    z_t = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)
    sigma_t = self._sigma_from_alphat(alpha_t)
    if detach_student:
      with torch.no_grad():
        student_probs = self.forward(z_t, sigma=sigma_t).exp()
    else:
      student_probs = self.forward(z_t, sigma=sigma_t).exp()
    student_probs = self._preserve_prompt_probs(student_probs, x0, valid_tokens)

    # Hard student sample -> posterior "sampling of masks" to the critic latent.
    x_hard = trainer_base.sample_categorical(student_probs.detach())
    x_hard = self._preserve_prompt_tokens(x_hard, x0, valid_tokens)
    z_s = self._dmmd_posterior(z_t, x_hard, alpha_t, alpha_s)
    z_s = self._preserve_prompt_tokens(z_s, x0, valid_tokens)
    return student_probs, z_s, alpha_s, dalpha_s, x_hard

  def _apply_top_p_to_teacher(self, teacher_log_probs):
    """D-MMD Sec. 3.3 top-p distillation of the teacher target.

    Lowers the log-probability of out-of-nucleus categories by `top_p_delta`
    instead of hard-masking them to a large negative value (which would spike the
    gradient reaching the student through its softmax Jacobian). No-op when
    `top_p >= 1`. The nucleus is the smallest set whose cumulative probability
    first exceeds `top_p` (top-1 always kept), matching the shift-by-one
    semantics of `utils.top_k_top_p_filtering`. Following the paper we do not
    renormalize; a per-position constant would not change the student gradient.
    Safe on the one-hot revealed positions produced by `_process_model_output`
    (their nucleus is the single revealed token, so they stay one-hot).
    """
    if self.top_p is None or self.top_p >= 1.0:
      return teacher_log_probs
    with torch.no_grad():
      probs = teacher_log_probs.exp()
      sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
      cum_probs = torch.cumsum(sorted_probs, dim=-1)
      sorted_remove = cum_probs > self.top_p
      # Keep the boundary category that first crosses top_p; always keep top-1.
      sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
      sorted_remove[..., 0] = False
      remove = torch.zeros_like(sorted_remove).scatter(
        -1, sorted_idx, sorted_remove)
    return teacher_log_probs - remove.to(teacher_log_probs.dtype) * self.top_p_delta

  def _generator_loss_diffusion(self, x0, valid_tokens,
                                current_accumulation_step):
    # D-MMD generator loss (diffusion critic). Same weighted soft cross-entropy
    # as the parent, but the critic latent comes from the selected mask-sampling
    # scheme (D-MMD posterior or IDLM re-noising) and the teacher target is
    # optionally top-p distilled.
    num_noised_samples = getattr(
      self.config.adversarial_distill, 'generator_num_noised_samples', 1)

    teacher_loss_total = 0.0
    fake_loss_total = 0.0

    for _ in range(num_noised_samples):
      student_probs, crit_x, alpha, dalpha, _ = self._dmmd_generate(
        x0, valid_tokens, current_accumulation_step, detach_student=False)
      sigma = self._sigma_from_alphat(alpha)
      sigma = self._process_sigma(sigma)

      with torch.cuda.amp.autocast(dtype=torch.float32):
        model_output_teacher = self.teacher(crit_x, sigma)
      logits_teacher = self._process_model_output(
        model_output=model_output_teacher, xt=crit_x, sigma=sigma)
      logits_teacher = self._apply_top_p_to_teacher(logits_teacher)
      teacher_loss = self.nll_per_token_with_probs(
        logits_teacher, crit_x, student_probs,
        alpha_t=alpha, dalpha_t=dalpha, valid_tokens=valid_tokens)
      teacher_loss_total = teacher_loss_total + teacher_loss

      if self.use_fake:
        with torch.cuda.amp.autocast(dtype=torch.float32):
          model_output_fake = self.fake_model(crit_x, sigma)
        logits_fake = self._process_model_output(
          model_output=model_output_fake, xt=crit_x, sigma=sigma)
        fake_loss = self.nll_per_token_with_probs(
          logits_fake, crit_x, student_probs,
          alpha_t=alpha, dalpha_t=dalpha, valid_tokens=valid_tokens)
        fake_loss_total = fake_loss_total + fake_loss

    teacher_loss = teacher_loss_total / num_noised_samples
    if not self.use_fake:
      return teacher_loss
    fake_loss = fake_loss_total / num_noised_samples
    return teacher_loss - fake_loss

  def _fake_loss_diffusion(self, x0, valid_tokens, current_accumulation_step):
    # D-MMD auxiliary/fake loss (diffusion critic). Fits the fake critic at the
    # critic latent to either the student's soft prediction or the hard sample it
    # was drawn from, per `fake_soft_target` (D-MMD Sec. 3.2 -- Algorithm 1's
    # "Optional use soft target x <- x_hat_eta(z_t)"). Keeps the optional
    # aux->teacher anchor (fake_teacher_weight, D-MMD Eq. 12, second term).
    student_probs, crit_x, alpha, dalpha, x_hard = self._dmmd_generate(
      x0, valid_tokens, current_accumulation_step, detach_student=True)
    sigma = self._sigma_from_alphat(alpha)
    sigma = self._process_sigma(sigma)

    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_fake = self.fake_model(crit_x, sigma)
    logits_fake = self._process_model_output(
      model_output=model_output_fake, xt=crit_x, sigma=sigma)
    fake_loss = self.nll_per_token_with_probs(
      logits_fake, crit_x,
      self._fake_diffusion_target(student_probs, x_hard),
      alpha_t=alpha, dalpha_t=dalpha, valid_tokens=valid_tokens)

    if self.fake_teacher_weight > 0:
      with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
        teacher_probs = self._process_model_output(
          model_output=self.teacher(crit_x, sigma), xt=crit_x, sigma=sigma).exp()
      anchor = self.nll_per_token_with_probs(
        logits_fake, crit_x, teacher_probs,
        alpha_t=alpha, dalpha_t=dalpha, valid_tokens=valid_tokens)
      fake_loss = fake_loss + self.fake_teacher_weight * anchor
    return fake_loss

  def _generator_loss_ar(self, x0, valid_tokens, current_accumulation_step):
    # AR critic: identical to the parent, plus D-MMD Sec. 3.3 top-p distillation
    # of the teacher's next-token log-probs. (The posterior mask sampling is
    # diffusion-specific and does not apply to a full-sequence AR critic.)
    student_probs = self.multistep_generation(
      x0, valid_tokens, current_accumulation_step)

    x_gen = trainer_base.sample_categorical(student_probs.detach())
    x_gen = self._preserve_prompt_tokens(x_gen, x0, valid_tokens)

    with torch.no_grad():
      teacher_logprobs = self._ar_logprobs(self.teacher, x_gen)
    teacher_logprobs = self._apply_top_p_to_teacher(teacher_logprobs)
    teacher_loss = self._ar_cross_entropy(
      teacher_logprobs, student_probs, valid_tokens)
    if not self.use_fake:
      return teacher_loss

    with torch.no_grad():
      fake_logprobs = self._ar_logprobs(self.fake_model, x_gen)
    fake_loss = self._ar_cross_entropy(
      fake_logprobs, student_probs, valid_tokens)
    return teacher_loss - fake_loss


# Backwards-compatible alias for the algo=mdlm_dmmd wiring in main.py.
MDLM_with_DMMD = MDLM_DMMD


# class MDLM(trainer_base.AbsorbingState):
#   def __init__(self, config, tokenizer):
#     super().__init__(config, tokenizer)
#     self._validate_configuration()
#     self.is_adversarial_distill = self.config.adversarial_distill.is_distill
#     if self.is_adversarial_distill:
#       self.automatic_optimization = False
#       self.teacher = copy.deepcopy(self.backbone)
#       utils.freeze_model(self.teacher)
      
#       self.fake_model = copy.deepcopy(self.backbone)
#       utils.activate_model(self.fake_model)
#       utils.activate_model(self.backbone)
#       if self.ema is not None:
#         del self.ema
#         self.ema = models.ema.ExponentialMovingAverage(
#           self.backbone.parameters(),
#           decay=self.config.training.ema)

#         self.ema_fake = models.ema.ExponentialMovingAverage(
#           self.fake_model.parameters(),
#           decay=self.config.training.ema)
#       self.training_step_counter = 0
#       self.accum_batches = self.config.adversarial_distill.accum_batches 
#       self._fake_accum_counter = 0
#       self._student_accum_counter = 0
#       self.fake_frequency = self.config.adversarial_distill.fake_frequency
#       self.fake_iter = 0 
#       self.is_generator_step = False
#       self.total_loss_student = 0
#       self.total_loss_fake = 0
#       self.f_loss_ar = []
  
#   def configure_optimizers(self):
#     if self.config.adversarial_distill.is_distill:
#       optimizer_student = torch.optim.AdamW(
#         self.backbone.parameters(),
#         lr=self.config.optim.lr,
#         betas=(self.config.optim.beta1,
#               self.config.optim.beta2),
#         eps=self.config.optim.eps,
#         weight_decay=self.config.optim.weight_decay)
      
#       optimizer_fake = torch.optim.AdamW(
#         self.fake_model.parameters(),
#         lr=self.config.optim.lr,
#         betas=(self.config.optim.beta1,
#               self.config.optim.beta2),
#         eps=self.config.optim.eps,
#         weight_decay=self.config.optim.weight_decay)

#       scheduler_student = hydra.utils.instantiate(
#         self.config.lr_scheduler, optimizer=optimizer_student)
#       scheduler_student_dict = {'scheduler': scheduler_student, 'name': 'trainer/lr'}

#       scheduler_fake = hydra.utils.instantiate(
#         self.config.lr_scheduler, optimizer=optimizer_fake)
#       scheduler_fake_dict = {'scheduler': scheduler_fake, 'name': 'trainer/lr_fake'}
#       return [optimizer_student, optimizer_fake], [scheduler_student_dict, scheduler_fake_dict]
#     else:
#       return super().configure_optimizers()

#   def _validate_configuration(self):
#     # ancestral sampling isn't desirable because it's slow
#     assert self.sampler == 'ancestral_cache'

#   def _process_model_output(self, model_output, xt, sigma):
#     del sigma
#     model_output[:, :, self.mask_index] += self.neg_infinity
    
#     # Normalize the model_output such that x.exp() is
#     # a probability distribution over vocab_size.
#     model_output = model_output - torch.logsumexp(
#       model_output, dim=-1, keepdim=True)
#     # Apply updates directly in the logits matrix.
#     # For the logits of the unmasked tokens, set all values
#     # to -infinity except for the indices corresponding to
#     # the unmasked tokens.
#     unmasked_indices = (xt != self.mask_index)
#     model_output[unmasked_indices] = self.neg_infinity
#     model_output[unmasked_indices, xt[unmasked_indices]] = 0
#     return model_output

#   def on_train_start(self):
#     if self.is_adversarial_distill:
#       self.ema_fake.move_shadow_params_to_device(self.device)
#       utils.activate_model(self.fake_model)
#       utils.freeze_model(self.backbone)
#     super().on_train_start()

#   def on_load_checkpoint(self, checkpoint):
#     if self.is_adversarial_distill:
#       state_dict = collections.OrderedDict(checkpoint['state_dict'])
#       for key, value in list(state_dict.items()):
#         if key.startswith('backbone.'):
#           suffix = key[len('backbone.'):]
#           state_dict.setdefault(f'teacher.{suffix}', value)
#           state_dict.setdefault(f'fake_model.{suffix}', value)
#       checkpoint['state_dict'] = state_dict
#     super().on_load_checkpoint(checkpoint)
#     if self.is_adversarial_distill and self.ema_fake is not None:
#       ema_fake_state = checkpoint.get('ema_fake', checkpoint.get('ema'))
#       if ema_fake_state is not None:
#         self.ema_fake.load_state_dict(ema_fake_state)

#   def on_save_checkpoint(self, checkpoint):
#     super().on_save_checkpoint(checkpoint)
#     if self.is_adversarial_distill and self.ema_fake is not None:
#       checkpoint['ema_fake'] = self.ema_fake.state_dict()

#   def _preserve_prompt_probs(self, probs, x0, valid_tokens):
#     if valid_tokens is None:
#       return probs
#     clean_probs = F.one_hot(x0, num_classes=probs.shape[-1]).to(probs.dtype)
#     return torch.where(valid_tokens[..., None].bool(), probs, clean_probs)

#   def _preserve_prompt_tokens(self, tokens, x0, valid_tokens):
#     if valid_tokens is None:
#       return tokens
#     return torch.where(valid_tokens.bool(), tokens, x0)

#   def multistep_generation(self, x0, valid_tokens, current_accumulation_step):
#     t = self._sample_t(x0.shape[0], current_accumulation_step)
#     dalpha_t, alpha_t = self.noise(t)
#     alpha_t = alpha_t.unsqueeze(-1)
#     assert alpha_t.ndim == 2
#     sigma = self._sigma_from_alphat(alpha_t)

#     xt = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)

#     log_x_theta = self.forward(xt, sigma=sigma)
#     student_probs = log_x_theta.exp()
#     return self._preserve_prompt_probs(student_probs, x0, valid_tokens)

#   def q_xt_with_probs(self, student_probs, alpha_t, x0=None, valid_tokens=None):
#     mask_probs = torch.zeros_like(student_probs)
#     mask_probs[:, :, self.mask_index] = 1
#     noise_probs = alpha_t[:, :, None] * student_probs + (1 - alpha_t[:, :, None])*(mask_probs)
#     if x0 is not None and valid_tokens is not None:
#       noise_probs = self._preserve_prompt_probs(noise_probs, x0, valid_tokens)
#     return noise_probs

#   def nll_per_token_with_probs(self, log_x_theta, xt, x0, alpha_t,
#                     dalpha_t, low_var=False, valid_tokens=None):
#     del xt
#     loss = (x0 * log_x_theta).sum(-1)
#     loss = loss * dalpha_t / (1 - alpha_t)
#     if valid_tokens is not None:
#       loss = loss * valid_tokens
#       denom = valid_tokens.sum(-1).clamp_min(1)
#       return loss.sum(-1) / denom
#     return loss.sum(-1)

#   def generator_loss(self, x0, valid_tokens, current_accumulation_step):
#     student_probs = self.multistep_generation(x0, valid_tokens, current_accumulation_step)
    
#     t = self._sample_t(x0.shape[0], current_accumulation_step)
#     dalpha_t, alpha_t = self.noise(t)
#     alpha_t = alpha_t.unsqueeze(-1)
#     assert alpha_t.ndim == 2
#     sigma = self._sigma_from_alphat(alpha_t)
#     sigma = self._process_sigma(sigma)

#     probs_xt = self.q_xt_with_probs(student_probs, alpha_t, x0=x0, valid_tokens=valid_tokens)
#     xt = trainer_base.sample_categorical(probs_xt)
#     xt = self._preserve_prompt_tokens(xt, x0, valid_tokens)
    
#     with torch.cuda.amp.autocast(dtype=torch.float32):
#       model_output_teacher = self.teacher(xt, sigma)
#     logits_teacher = self._process_model_output(model_output=model_output_teacher, xt=xt, sigma=sigma)
#     teacher_loss = self.nll_per_token_with_probs(logits_teacher, xt, student_probs, 
#                                 alpha_t=alpha_t, dalpha_t=dalpha_t,
#                                 valid_tokens=valid_tokens)

#     with torch.cuda.amp.autocast(dtype=torch.float32):
#       model_output_fake = self.fake_model(xt, sigma)
#     logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
#     fake_loss = self.nll_per_token_with_probs(logits_fake, xt, student_probs, 
#                                 alpha_t=alpha_t, dalpha_t=dalpha_t,
#                                 valid_tokens=valid_tokens)
#     return teacher_loss - fake_loss

#   def fake_loss(self, x0, valid_tokens, current_accumulation_step):
#     with torch.no_grad():
#       student_probs = self.multistep_generation(x0, valid_tokens, current_accumulation_step)

#     t = self._sample_t(x0.shape[0], current_accumulation_step)
#     dalpha_t, alpha_t = self.noise(t)
#     alpha_t = alpha_t.unsqueeze(-1)
#     assert alpha_t.ndim == 2
#     sigma = self._sigma_from_alphat(alpha_t)
#     sigma = self._process_sigma(sigma)
#     probs_xt = self.q_xt_with_probs(student_probs, alpha_t, x0=x0, valid_tokens=valid_tokens)
#     xt = trainer_base.sample_categorical(probs_xt)
#     xt = self._preserve_prompt_tokens(xt, x0, valid_tokens)

#     with torch.cuda.amp.autocast(dtype=torch.float32):
#       model_output_fake = self.fake_model(xt, sigma)
#     logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
#     fake_loss = self.nll_per_token_with_probs(logits_fake, xt, student_probs, 
#                                 alpha_t=alpha_t, dalpha_t=dalpha_t,
#                                 valid_tokens=valid_tokens)

#     return fake_loss

#   def training_step(self, batch, batch_idx):
#     if self.config.adversarial_distill.is_distill:
#       optimizer_student, optimizer_fake = self.optimizers()
#       scheduler_student, scheduler_fake = self.lr_schedulers()
#       current_accumulation_step = (batch_idx % self.accum_batches)
#       finish_iter = False

#       (input_tokens, output_tokens, valid_tokens) = self._process_model_input(batch['input_ids'], 
#                                                                               batch['attention_mask'])
#       valid_tokens = _conditional_valid_tokens(self.config, valid_tokens)
#       if self.is_generator_step:
#         self.toggle_optimizer(optimizer_student)
#         g_loss = self.generator_loss(input_tokens, valid_tokens, current_accumulation_step)
#         g_loss = g_loss.mean()
#         g_loss = g_loss / self.accum_batches
#         self.manual_backward(g_loss)

#         self.untoggle_optimizer(optimizer_student)

#         self._student_accum_counter += 1
#         self.total_loss_student += g_loss.detach()
#         if self._student_accum_counter == self.accum_batches:
#           self._student_accum_counter = 0 
#           optimizer_student.step()
#           self.ema.update(self.backbone.parameters())
#           optimizer_student.zero_grad()
#           finish_iter = True
#       else:
#         self.toggle_optimizer(optimizer_fake)
#         f_loss = self.fake_loss(input_tokens, valid_tokens, current_accumulation_step)
#         f_loss = f_loss.mean()
#         f_loss = f_loss / self.accum_batches
#         self.manual_backward(f_loss)

#         self.untoggle_optimizer(optimizer_fake)
#         self._fake_accum_counter += 1
#         self.total_loss_fake += f_loss.detach()
#         if self._fake_accum_counter == self.accum_batches:
#           self._fake_accum_counter = 0 
#           optimizer_fake.step()
#           self.ema_fake.update(self.fake_model.parameters())
#           optimizer_fake.zero_grad()
#           finish_iter = True

#       if finish_iter:
#         _maybe_save_periodic_checkpoint(self)

#       if finish_iter and self.is_generator_step:
#         self.is_generator_step = False
#         utils.activate_model(self.fake_model)
#         utils.freeze_model(self.backbone)
#         optimizer_fake.zero_grad()
#         self.log_dict(
#             {"student_loss": self.total_loss_student, "fake_loss": torch.stack(self.f_loss_ar).mean()},
#             prog_bar=True,
#             on_step=True,
#             on_epoch=False,
#             sync_dist=True,
#         )
#         scheduler_student.step()
#         scheduler_fake.step()
#         self.total_loss_student = 0
#         self.f_loss_ar.clear()
#       elif finish_iter:
#         self.fake_iter += 1
#         self.f_loss_ar.append(self.total_loss_fake)
#         self.total_loss_fake = 0
#         if self.fake_iter == self.fake_frequency:
#           self.is_generator_step = True
#           self.fake_iter = 0
#           utils.activate_model(self.backbone)
#           utils.freeze_model(self.fake_model)
#           optimizer_student.zero_grad()

#     else:
#       return super().training_step(batch, batch_idx)

#   def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
#                     dalpha_t, low_var=False):
#     del xt
#     log_p_theta = torch.gather(
#       input=log_x_theta,
#       dim=-1,
#       index=x0[:, :, None]).squeeze(-1)
#     return log_p_theta * dalpha_t / (1 - alpha_t)

#   def _get_score(self, x, sigma):
#     model_output = self.forward(x, sigma)
#     # score(x, t) = p_t(y) / p_t(x)
#     # => log score(x, t) = log p_t(y) - log p_t(x)
    
#     # case 1: x = masked
#     #   (i) y = unmasked
#     #     log score(x, t) = log p_\theta(x)|_y + log k
#     #     where k = exp(- sigma) / (1 - exp(- sigma))
#     #   (ii) y = masked
#     #     log score(x, t) = 0

#     # case 2: x = unmasked
#     #   (i) y != masked, y != x
#     #     log score(x_i, t) = - inf
#     #   (ii) y = x 
#     #     log score(x_i, t) = 0
#     #   (iii) y = masked token
#     #     log score(x_i, t) = - log k
#     #     where k = exp(- sigma) / (1 - exp(- sigma))
    
#     log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
#     assert log_k.ndim == 1
    
#     masked_score = model_output + log_k[:, None, None]
#     masked_score[:, :, self.mask_index] = 0

#     unmasked_score = self.neg_infinity * torch.ones_like(
#       model_output)
#     unmasked_score = torch.scatter(
#       unmasked_score,
#       -1,
#       x[..., None],
#       torch.zeros_like(unmasked_score[..., :1]))
#     unmasked_score[:, :, self.mask_index] = - (
#       log_k[:, None] * torch.ones_like(x))
    
#     masked_indices = (x == self.mask_index).to(
#       model_output.dtype)[:, :, None]
#     model_output = (
#       masked_score * masked_indices
#       + unmasked_score * (1 - masked_indices))
#     return model_output.exp()


class D3PMAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.noise.type == 'log-linear'
    assert self.parameterization == 'mean'

  def _process_model_output(self, model_output, xt, sigma):
    del xt 
    del sigma
    if self.subs_masking:
      model_output[:, :, self.mask_index] += self.neg_infinity
    return model_output.log_softmax(dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del dalpha_t
    assert not low_var
    dt = 1 / self.T
    t = 1 - alpha_t  # Only valid for log-linear schedule.
    t = t.clamp(0., 1.0 - 1e-4)
    alpha_t = alpha_t + torch.zeros_like(xt)
    alpha_s = t - dt + torch.zeros_like(xt)
    assert alpha_s.shape == xt.shape
    assert alpha_t.shape == xt.shape
    log_x_theta_at_x0 = torch.gather(
      log_x_theta, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(
      alpha_s * x_theta_at_m / (t - dt) + 1)
    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
    return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.predictor == 'analytic'

  def _get_score(self, x, sigma):
    return self.forward(x, sigma).exp()

  def _process_model_output(self, model_output, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(model_output.dtype)
    # logits shape
    # (batch_size, context_length, vocab_size)
    model_output = (model_output
                    - esigm1_log[:, None, None]
                    - np.log(model_output.shape[-1] - 1))
    # The below scatter operation sets the log score
    # for the input word to 0.
    model_output = torch.scatter(
      model_output, -1, xt[..., None],
      torch.zeros_like(model_output[..., :1]))
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    """Computes the SEDD loss for the Absorbing State Diffusion.

    Args:
      log_x_theta: float torch.Tensor with shape (batch_size,
          context_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          context_length), input.
      x0: int torch.Tensor with shape (batch_size,
          context_length), input.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
          time derivative of signal level.
      low_var: bool, low variance loss during training.
    
    Returns:
      loss with shape (batch_size, context_length).
    """
    assert not low_var
    masked_indices = xt == self.mask_index
    sigma = self._sigma_from_alphat(alpha_t)
    dsigma = - dalpha_t / alpha_t

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_x_theta[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_x_theta[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return dsigma * entropy


class DUO_BASE(trainer_base.UniformState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    return model_output.log_softmax(dim=-1)

  def _compute_posterior(self, x, xt, alpha_s, alpha_t):
    """Computes the posterior / approximate posterior.

    Args:
      x: Either clean input `x0` (one-hot),
        or model's predicted `x_theta` of shape (B, L, V).
      xt: The noisy latent (as indices) of shape (B, L).
      alpha_s: Noise level at s of shape (B, [L | 1], 1).
      alpha_t: Noise level at t of shape (B, [L | 1], 1).

    Returns:
      Posterior / approximate posterior of shape (B, L, V).
    """
    if self.config.sampling.use_float64:
      x = x.to(torch.float64)
    if alpha_s.ndim == 2:
      alpha_s = alpha_s.unsqueeze(-1)
    if alpha_t.ndim == 2:
      alpha_t = alpha_t.unsqueeze(-1)
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, self.vocab_size).to(
      self.dtype).to(self.device)
    return (
      (alpha_t * self.vocab_size * x * xt_one_hot + (
        alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
          1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
            alpha_t * self.vocab_size * torch.gather(
              x, -1, xt[..., None]) + (1 - alpha_t)))

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
        :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    term1 = self.vocab_size * (1 / xbar_xt
                                - 1 / xbar_theta_xt)
    
    const = (1 - alpha_t) / (self.vocab_size * alpha_t
                             + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1)
      - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss

  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False):
    del p_x0
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    sigma_t = self._sigma_from_alphat(alpha_t)
    assert alpha_t.ndim == 2
    
    q_xs = self._compute_posterior(
      x=self.forward(x, sigma_t).exp(),
      xt=x,
      alpha_s=alpha_s,
      alpha_t=alpha_t)
    if self.p_nucleus < 1:
      q_xs = utils.top_k_top_p_filtering(
        q_xs.log(), top_p=self.p_nucleus)
    return None, trainer_base.sample_categorical(q_xs)


class Integral(torch.autograd.Function):
  """
  torch module calculating UDLM's p_t 
  """

  @staticmethod
  def forward(ctx, gamma_t, data):
    gamma_max = data['gamma_max']
    gamma_min = data['gamma_min']
    if (gamma_t.max() > gamma_max) or (
      gamma_t.min() < gamma_min):
      print('max:{} {}'.format(gamma_t.max(), gamma_max))
      print('min:{} {}'.format(gamma_t.min(), gamma_min))
      gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
    indices = torch.round(
      (data['num_points'] - 1) * (gamma_t - gamma_min) / (
          gamma_max - gamma_min)).long()
    grad_pt = data['grad_pt']
    ctx.grad_pt = grad_pt[indices]
    
    pt = data['pt'][indices]
    assert pt.shape == gamma_t.shape
    return pt

  @staticmethod
  def backward(ctx, grad_output):
    return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    with fsspec.open(self.config.algo.integral_cache_path,
                     'rb') as f:
      self.integral_cache = pickle.load(f)
    self.integral_cache['pt'] = torch.from_numpy(
      self.integral_cache['pt'])
    self.integral_cache['grad_pt'] = torch.from_numpy(
      self.integral_cache['grad_pt'])
    self.gamma_min = self.config.algo.gamma_min
    self.gamma_max = self.config.algo.gamma_max
    self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
    self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
    self.curriculum_start = self.config.algo.curriculum_start
    self.curriculum_end = self.config.algo.curriculum_end
    self.loss_type = self.config.algo.loss_type
    self.is_adversarial_distill = self.config.adversarial_distill.is_distill
    if self.is_adversarial_distill:
      self.automatic_optimization = False
      self.teacher = copy.deepcopy(self.backbone)
      utils.freeze_model(self.teacher)
      
      self.fake_model = copy.deepcopy(self.backbone)
      utils.activate_model(self.fake_model)
      utils.activate_model(self.backbone)
      if self.ema is not None:
        del self.ema
        self.ema = models.ema.ExponentialMovingAverage(
          self.backbone.parameters(),
          decay=self.config.training.ema)

        self.ema_fake = models.ema.ExponentialMovingAverage(
          self.fake_model.parameters(),
          decay=self.config.training.ema)
      self.training_step_counter = 0
      self.accum_batches = self.config.adversarial_distill.accum_batches 
      self._fake_accum_counter = 0
      self._student_accum_counter = 0
      self.fake_frequency = self.config.adversarial_distill.fake_frequency
      self.fake_iter = 0 
      self.is_generator_step = False
      self.total_loss_student = 0
      self.total_loss_fake = 0
      self.f_loss_ar = []
    self._validate_configuration()

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.integral_cache['pt'] = self.integral_cache[
      'pt'].to(*args, **kwargs)
    self.integral_cache['grad_pt'] = self.integral_cache[
      'grad_pt'].to(*args, **kwargs)
    return self

  def _compute_gumbel_tau_inverse(self):
    start = self.gumbel_tau_log10_start
    end = self.gumbel_tau_log10_end
    delta = end - start
    if self.global_step < self.curriculum_start:
      tau = start
    elif self.global_step < self.curriculum_end:
      frac = (self.global_step - self.curriculum_start) / (
        self.curriculum_end - self.curriculum_start)
      tau = start + frac * delta
    else:
      tau = -10
    return 10 ** (-tau)

  def _gaussian_update(self, x, t, dt):
    t = t.squeeze()
    gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    sigma = self._sigma_from_alphat(usdm_alpha_t)
    log_x_theta = self.forward(x, sigma=sigma)
    x0 = trainer_base.sample_categorical(log_x_theta.exp())
    x0_one_hot = F.one_hot(x0, self.vocab_size)
    

    gamma_t_mdt = self.gamma_min + (t - dt) * (self.gamma_max - self.gamma_min)
    xt = self._q_xt_gaussian(x0_one_hot, gamma_t_mdt)
    return xt

  def _clean_probs(self, x0):
    return F.one_hot(x0, self.vocab_size).to(self.dtype)

  def _force_prompt_probs(self, probs, clean_probs, valid_tokens):
    if valid_tokens is None:
      return probs
    prompt = ~valid_tokens.bool().unsqueeze(-1)
    return torch.where(prompt, clean_probs, probs)

  def _scale_and_restore_prompt(self, xt, clean_probs, valid_tokens):
    tau_inv = self._compute_gumbel_tau_inverse()
    xt = xt * tau_inv
    if valid_tokens is not None:
      prompt = ~valid_tokens.bool().unsqueeze(-1)
      xt = torch.where(prompt, clean_probs * tau_inv, xt)
    return xt

  def _argmax_with_clean_prompt(self, xt, x0, valid_tokens):
    xt_tokens = xt.argmax(-1)
    if valid_tokens is not None:
      xt_tokens = torch.where(valid_tokens.bool(), xt_tokens, x0)
    return xt_tokens

  def _masked_mean(self, loss, valid_tokens):
    if valid_tokens is None:
      return loss.sum(-1).mean()
    return (loss * valid_tokens).sum() / valid_tokens.sum().clamp(min=1)

  def multistep_generation(self, x0, current_accumulation_step,
                           valid_tokens=None, eps_probs=1e-12):
    t = self._sample_t(x0.shape[0], current_accumulation_step)

    gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)  

    x0_one_hot = self._clean_probs(x0)
    xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
    xt = self._scale_and_restore_prompt(xt, x0_one_hot, valid_tokens)

    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(usdm_alpha_t)
    
    log_x_theta = self.forward(xt, sigma=sigma)
    student_probs = log_x_theta.exp()
    return self._force_prompt_probs(student_probs, x0_one_hot, valid_tokens)

  def nll_per_token_with_probs(self, log_x_theta, xt, x0, alpha_t, dalpha_t):
    assert alpha_t.ndim == 2
    assert x0.ndim == 3
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[:, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    x_bar = self.vocab_size * alpha_t[:, :, None] * x0 + 1 - alpha_t[:, :, None]

    coeff = dalpha_t / (self.vocab_size * alpha_t)
    
    x_bar_zt = torch.gather(x_bar, -1, xt[..., None])  # B, L, 1
    x_bar_theta_zt = torch.gather(x_bar_theta, -1, xt[..., None])  # B, L, 1
    term1 = ((self.vocab_size / x_bar_zt) - (self.vocab_size / x_bar_theta_zt))  # B, L, 1
    
    term2 = (  # B, L, V before summing --> B, L, 1 after
          (x_bar / x_bar_zt) *
          (
              x_bar_theta_zt.log() - x_bar_theta.log() +
              x_bar.log() - x_bar_zt.log()
          )
      )
    
    term2 = term2.sum(dim=-1, keepdim=True)
    diffusion_loss = (coeff.unsqueeze(-1) * (term1 - term2)).squeeze(-1)
    assert diffusion_loss.ndim == 2
    return diffusion_loss

  def q_xt_with_probs(self, student_probs, alpha_t):
    noise_probs = alpha_t[:, :, None] * student_probs + (1 - alpha_t[:, :, None])*(1/self.vocab_size)
    return noise_probs
  
  def gumbel_softmax_from_probs(self, probs, tau=1e-3, eps=1e-12):
    probs = probs.clamp(min=0.0) + eps
    probs = probs / probs.sum(dim=-1, keepdim=True)

    logits = torch.log(probs)              

    u = torch.rand_like(logits).clamp_(eps, 1 - eps)
    g = -torch.log(-torch.log(u))
    log_y_soft =(logits + g) / tau

    return log_y_soft

  def generator_loss(self, x0, current_accumulation_step, valid_tokens=None):
    student_probs = self.multistep_generation(
      x0, current_accumulation_step, valid_tokens)
    clean_probs = self._clean_probs(x0)
    student_probs = self._force_prompt_probs(
      student_probs, clean_probs, valid_tokens)

    t = self._sample_t(x0.shape[0], current_accumulation_step)
    gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)   
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)

    sigma = self._sigma_from_alphat(usdm_alpha_t)
    sigma = self._process_sigma(sigma)

    xt = self._q_xt_gaussian(student_probs, gamma_t)
    xt = self._scale_and_restore_prompt(xt, clean_probs, valid_tokens)
    xt_usdm = self._argmax_with_clean_prompt(xt, x0, valid_tokens)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_teacher = self.teacher(xt, sigma)
    logits_teacher = self._process_model_output(model_output=model_output_teacher, xt=xt, sigma=sigma)
    teacher_loss = self.nll_per_token_with_probs(logits_teacher, xt_usdm, student_probs, 
                                alpha_t=usdm_alpha_t, dalpha_t=usdm_dalpha_t)
    
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_fake = self.fake_model(xt, sigma)
    logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
    fake_loss = self.nll_per_token_with_probs(logits_fake, xt_usdm, student_probs, 
                                alpha_t=usdm_alpha_t, dalpha_t=usdm_dalpha_t)
    return self._masked_mean(teacher_loss - fake_loss, valid_tokens)
  
  def configure_optimizers(self):
    if self.config.adversarial_distill.is_distill:
      optimizer_student = torch.optim.AdamW(
        self.backbone.parameters(),
        lr=self.config.optim.lr,
        betas=(self.config.optim.beta1,
              self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)
      
      optimizer_fake = torch.optim.AdamW(
        self.fake_model.parameters(),
        lr=self.config.optim.lr,
        betas=(self.config.optim.beta1,
              self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)

      scheduler_student = hydra.utils.instantiate(
        self.config.lr_scheduler, optimizer=optimizer_student)
      scheduler_student_dict = {'scheduler': scheduler_student, 'name': 'trainer/lr'}

      scheduler_fake = hydra.utils.instantiate(
        self.config.lr_scheduler, optimizer=optimizer_fake)
      scheduler_fake_dict = {'scheduler': scheduler_fake, 'name': 'trainer/lr_fake'}
      return [optimizer_student, optimizer_fake], [scheduler_student_dict, scheduler_fake_dict]
    else:
      return super().configure_optimizers()

  def fake_loss(self, x0, current_accumulation_step, valid_tokens=None):
    with torch.no_grad():
      student_probs = self.multistep_generation(
        x0, current_accumulation_step, valid_tokens)
    clean_probs = self._clean_probs(x0)
    student_probs = self._force_prompt_probs(
      student_probs, clean_probs, valid_tokens)

    t = self._sample_t(x0.shape[0], current_accumulation_step)

    gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)   
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)

    sigma = self._sigma_from_alphat(usdm_alpha_t)
    sigma = self._process_sigma(sigma)

    xt = self._q_xt_gaussian(student_probs, gamma_t)
    xt = self._scale_and_restore_prompt(xt, clean_probs, valid_tokens)
    xt_usdm = self._argmax_with_clean_prompt(xt, x0, valid_tokens)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output_fake = self.fake_model(xt, sigma)
    logits_fake = self._process_model_output(model_output=model_output_fake, xt=xt, sigma=sigma)
    fake_loss = self.nll_per_token_with_probs(logits_fake, xt_usdm, student_probs, 
                                alpha_t=usdm_alpha_t, dalpha_t=usdm_dalpha_t)
    return self._masked_mean(fake_loss, valid_tokens)

  def _apply_checkpoint_ema_to_backbone_state(self, state_dict, checkpoint):
    ema_state = checkpoint.get('ema', None)
    if not (ema_state and 'shadow_params' in ema_state):
      return
    for (name, _), shadow in zip(
        self.backbone.named_parameters(), ema_state['shadow_params']):
      key = f'backbone.{name}'
      if key in state_dict:
        state_dict[key] = shadow.detach().clone().to(state_dict[key].dtype)

  def _backbone_shadow_params_from_state(self, state_dict):
    shadow_params = []
    for name, _ in self.backbone.named_parameters():
      key = f'backbone.{name}'
      if key not in state_dict:
        return None
      shadow_params.append(state_dict[key].detach().clone())
    return shadow_params

  def _reset_distill_ema(self, shadow_params):
    if shadow_params is None:
      return
    ema_state = {
      'decay': self.config.training.ema,
      'num_updates': 0,
      'shadow_params': shadow_params,
    }
    if self.ema is not None:
      self.ema.load_state_dict(copy.deepcopy(ema_state))
    if getattr(self, 'ema_fake', None) is not None:
      self.ema_fake.load_state_dict(copy.deepcopy(ema_state))

  def on_train_start(self):
    if self.is_adversarial_distill:
      self.ema_fake.move_shadow_params_to_device(self.device)
      utils.activate_model(self.fake_model)
      utils.freeze_model(self.backbone)
    super().on_train_start()

  def on_load_checkpoint(self, checkpoint):
    state_dict = collections.OrderedDict(checkpoint['state_dict'])
    has_distill_state = any(
      key.startswith(('teacher.', 'fake_model.'))
      for key in state_dict)
    if self.is_adversarial_distill:
      if not has_distill_state:
        self._apply_checkpoint_ema_to_backbone_state(state_dict, checkpoint)
      for key, value in list(state_dict.items()):
        if key.startswith('backbone.'):
          suffix = key[len('backbone.'):]
          state_dict.setdefault(f'teacher.{suffix}', value)
          state_dict.setdefault(f'fake_model.{suffix}', value)
    else:
      state_dict = collections.OrderedDict(
        (k, v) for k, v in state_dict.items()
        if not k.startswith(('teacher.', 'fake_model.')))
    checkpoint['state_dict'] = state_dict
    if self.is_adversarial_distill and not has_distill_state:
      self._reset_distill_ema(
        self._backbone_shadow_params_from_state(state_dict))
      return
    trainer_base.TrainerBase.on_load_checkpoint(self, checkpoint)
    if (self.is_adversarial_distill
        and getattr(self, 'ema_fake', None) is not None):
      ema_fake_state = checkpoint.get('ema_fake', checkpoint.get('ema'))
      if ema_fake_state is not None:
        self.ema_fake.load_state_dict(ema_fake_state)

  def on_save_checkpoint(self, checkpoint):
    if self.is_adversarial_distill:
      checkpoint['state_dict'] = collections.OrderedDict(
        (k, v) for k, v in checkpoint['state_dict'].items()
        if not k.startswith('teacher.'))
      trainer_base.TrainerBase.on_save_checkpoint(self, checkpoint)
    else:
      super().on_save_checkpoint(checkpoint)
    if (self.is_adversarial_distill
        and getattr(self, 'ema_fake', None) is not None):
      checkpoint['ema_fake'] = self.ema_fake.state_dict()

  def training_step(self, batch, batch_idx):
    self.log(name='gumbel_tau_log10',
              value=1 / self._compute_gumbel_tau_inverse(),
              on_step=True,
              on_epoch=False,
              sync_dist=True)
    if self.config.adversarial_distill.is_distill:
      optimizer_student, optimizer_fake = self.optimizers()
      scheduler_student, scheduler_fake = self.lr_schedulers()
      current_accumulation_step = (batch_idx % self.accum_batches)
      finish_iter = False

      (input_tokens, output_tokens, valid_tokens) = self._process_model_input(batch['input_ids'], 
                                                                              batch['attention_mask'])
      valid_tokens = _conditional_valid_tokens(self.config, valid_tokens)
      if self.is_generator_step:
        self.toggle_optimizer(optimizer_student)
        g_loss = self.generator_loss(
          input_tokens, current_accumulation_step, valid_tokens)
        g_loss = g_loss.mean()
        g_loss = g_loss / self.accum_batches
        self.manual_backward(g_loss)

        self.untoggle_optimizer(optimizer_student)

        self._student_accum_counter += 1
        self.total_loss_student += g_loss.detach()
        if self._student_accum_counter == self.accum_batches:
          self._student_accum_counter = 0 
          optimizer_student.step()
          self.ema.update(self.backbone.parameters())
          optimizer_student.zero_grad()
          finish_iter = True
      else:
        self.toggle_optimizer(optimizer_fake)
        f_loss = self.fake_loss(
          input_tokens, current_accumulation_step, valid_tokens)
        f_loss = f_loss.mean()
        f_loss = f_loss / self.accum_batches
        self.manual_backward(f_loss)

        self.untoggle_optimizer(optimizer_fake)
        self._fake_accum_counter += 1
        self.total_loss_fake += f_loss.detach()
        if self._fake_accum_counter == self.accum_batches:
          self._fake_accum_counter = 0 
          optimizer_fake.step()
          self.ema_fake.update(self.fake_model.parameters())
          optimizer_fake.zero_grad()
          finish_iter = True

      if finish_iter:
        _maybe_save_periodic_checkpoint(self)

      if finish_iter and self.is_generator_step:
        self.is_generator_step = False
        utils.activate_model(self.fake_model)
        utils.freeze_model(self.backbone)
        optimizer_fake.zero_grad()
        self.log_dict(
            {"student_loss": self.total_loss_student, "fake_loss": torch.stack(self.f_loss_ar).mean()},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        scheduler_student.step()
        scheduler_fake.step()
        self.total_loss_student = 0
        self.f_loss_ar.clear()
      elif finish_iter:
        self.fake_iter += 1
        self.f_loss_ar.append(self.total_loss_fake)
        self.total_loss_fake = 0
        if self.fake_iter == self.fake_frequency:
          self.is_generator_step = True
          self.fake_iter = 0
          utils.activate_model(self.backbone)
          utils.freeze_model(self.fake_model)
          optimizer_student.zero_grad()

    else:
      return super().training_step(batch, batch_idx)

  def _gamma_to_alphat(self, gamma_t):
    integral = Integral.apply(gamma_t, self.integral_cache)
    return (self.vocab_size * integral - 1) / (
      self.vocab_size - 1)

  def _prior_loss(self):
    alpha_1 = self._gamma_to_alphat(
      torch.tensor(self.gamma_max))
    loss = ((alpha_1 + (1 - alpha_1) / self.vocab_size) * torch.log(
      (self.vocab_size - 1) * alpha_1 + 1) + (
        1 - 1 / self.vocab_size) * (1 - alpha_1) * torch.log(1 - alpha_1))
    return loss.item()

  def _q_xt_gaussian(self, x, gamma_t):
    """Computes the noisy sample xt."""
    assert gamma_t.ndim == 1
    assert x.ndim == 3
    gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()
    epsilon = torch.randn(x.shape, dtype=torch.float32,
                          device=self.device)
    return alpha_t * x + sigma_t * epsilon

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    use_true_nll = (self.global_step > self.curriculum_end
                    or not train_mode)
    if use_true_nll:
      return super().nll(x0, output_tokens,
                         current_accumulation_step,
                         valid_tokens=valid_tokens)
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)    
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (
      self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(usdm_alpha_t)

    x0_one_hot = F.one_hot(x0, self.vocab_size)
    xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
    xt = xt * self._compute_gumbel_tau_inverse()
    xt_usdm = xt.argmax(-1)
    log_x_theta = self.forward(xt, sigma=sigma)

    return self.nll_per_token(log_x_theta=log_x_theta,
                              xt=xt_usdm,
                              x0=x0,
                              alpha_t=usdm_alpha_t,
                              dalpha_t=usdm_dalpha_t,
                              low_var=False)


class Distillation(DUO):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.update_teacher_every = config.algo.update_teacher_every
    self.save_hyperparameters()
    self.teacher = None
    self.teacher_ema = config.algo.teacher_ema
    self.linear_growth_dt = config.algo.linear_growth_dt
    self.linear_growth_min = config.algo.linear_growth_min
    self.linear_growth_max = config.algo.linear_growth_max

  def _validate_configuration(self):
    assert os.path.exists(
      self.config.algo.integral_cache_path), (
        'The integral cache (Eq. 10 in the paper) for '
        f'the {self.config.data.tokenizer_name_or_path} '
        ' tokenizer doesnt exist at '
        f'{self.config.algo.integral_cache_path}. '
        'Please generate it by running the utils.py script, '
        'and ensure the correct path is specified using the '
        'algo.integral_cache_path flag.')
    assert self.loss_type in {
      'kl-fwd', 'kl-bwd', 'posterior', 'kl-posterior'}

  def _maybe_update_teacher_weights(self):
    if self.global_step % self.update_teacher_every != 0:
      return
    if self.teacher_ema:
      self.ema.copy_to(self.teacher.parameters())
    else:
      for better_param, current_param in zip(
        self.backbone.parameters(), self.teacher.parameters()):
        if current_param.requires_grad:
          current_param.data.copy_(better_param.data)

  @torch.no_grad()
  def _teacher_logits(self, xt, sigma):
    if self.teacher is None:
      self.teacher = copy.deepcopy(self.backbone)
    self._maybe_update_teacher_weights()

    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output = self.teacher(xt, sigma)
    logits = self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)
    return logits.detach()

  def _sample_trajectory(self, x0, gamma_t, gamma_s):
    """Computes the noisy sample xt."""
    assert gamma_t.ndim == 1
    assert gamma_s.ndim == 1
    assert x0.ndim == 2
    x0 = F.one_hot(x0, self.vocab_size).to(
      self.dtype).to(self.device)
    gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()

    gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
    alpha_s = torch.sigmoid(-gamma_s).sqrt()
    sigma_s = torch.sigmoid(gamma_s).sqrt()
    
    epsilon = torch.randn(x0.shape, dtype=torch.float32,
                          device=self.device)
    xt = alpha_t * x0 + sigma_t * epsilon
    xs = alpha_s * x0 + sigma_s * epsilon
    return xt, xs

  def _compute_dt(self):
    if self.linear_growth_dt:
      scale = self.global_step / self.trainer.max_steps
      return self.linear_growth_min + scale * (
        self.linear_growth_max -  self.linear_growth_min)
    n = self.global_step // self.update_teacher_every
    return 2 ** n / self.T

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=None,
          valid_tokens=None):
    del output_tokens, train_mode
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dt = self._compute_dt()
    t = torch.clip(t + dt, 0, 1)

    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)
    gamma_s = self.gamma_min + (
      t - dt) * (self.gamma_max - self.gamma_min)

    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    usdm_alpha_s = self._gamma_to_alphat(gamma_s)
    usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
    assert usdm_alpha_s.ndim == 2

    xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
    xt_discrete = xt.argmax(-1)
    xs_discrete = xs.argmax(-1)
    log_x_theta_student = self.forward(
      xt_discrete, sigma=self._sigma_from_alphat(usdm_alpha_t))
    log_x_theta_teacher = self._teacher_logits(
      xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s))
    if self.config.training.loss_precision == 'float64':
      log_x_theta_student = log_x_theta_student.to(torch.float64)
      log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
    if self.loss_type == 'kl-fwd':
      return (log_x_theta_teacher.exp() * (
        log_x_theta_teacher - log_x_theta_student)).sum(-1)
    elif self.loss_type == 'kl-bwd':
      return (log_x_theta_student.exp() * (
        log_x_theta_student - log_x_theta_teacher)).sum(-1)
    
  def training_step(self, batch, batch_idx):
    self.log(name='dt',
             value=self._compute_dt(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return super().training_step(batch, batch_idx)
