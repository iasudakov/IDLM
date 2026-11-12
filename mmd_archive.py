"""Connect the corrected archive objectives to the existing MDLM trainer.

The source in mmd/objective{,_fast}.py is copied verbatim from the archive.
Only config names, student microbatch forwards, and metric logging live here.
Training, EMA, checkpoint loading/saving and paired GSM8K evaluation stay in
the existing trainer. B is draws per candidate, never rows from other prompts.
"""
import math

import torch
from torch.utils.checkpoint import checkpoint

from mmd.objective import FeatureReward, reinforce_loss
from mmd.objective_fast import FastFeatureReward, ScoredHead, reinforce_loss_fast


def validate_config(config):
  cfg = config.adversarial_distill
  required = {
    'is_distill': True, 'dmd_weight': 0.0, 'mmd_reward_type': 'feature',
    'mmd_feature_source': 'diffusion', 'mmd_noise_level': 0.0,
    'mmd_kernel': 'rbf', 'mmd_kernel_level': 'token',
    'mmd_kernel_source': 'draws', 'mmd_rbf_distance': 'squared_euclidean',
    'mmd_rbf_exclude_same_sentence': True, 'mmd_rbf_omit_fake_fake': False,
    'mmd_score_positions': 'written', 'mmd_scored_positions': 'answer',
    'mmd_shuffle_rewards': False,
  }
  for key, expected in required.items():
    if cfg.get(key, expected) != expected:
      raise ValueError(f'Archive MMD requires adversarial_distill.{key}={expected!r}; '
                       'use mmd_implementation=legacy for the old objective')
  if cfg.mmd_implementation not in ('fast', 'reference'):
    raise ValueError('mmd_implementation must be legacy, fast or reference')
  if config.algo.name != 'mdlm' or config.algo.backbone != 'dit':
    raise ValueError('Archive integration requires algo=mdlm and backbone=dit')
  if config.training.get('continue_teacher', False):
    raise ValueError('Archive MMD cannot be combined with continue_teacher')
  if config.algo.causal_attention or config.algo.time_conditioning:
    raise ValueError('Archive features require bidirectional, non-time-conditioned MDLM')
  if config.trainer.accumulate_grad_batches != 1:
    raise ValueError('Archive integration uses manual accumulation; trainer.accumulate_grad_batches must be 1')
  devices = config.trainer.devices
  if isinstance(devices, int) and devices > 0:
    effective = (devices * config.trainer.num_nodes * config.loader.batch_size
                 * cfg.accum_batches)
    if effective != config.loader.global_batch_size:
      raise ValueError('global_batch_size must equal devices * nodes * batch_size * '
                       'adversarial_distill.accum_batches')
  for key, minimum in [('mmd_group_size', 1), ('mmd_kernel_batch_size', 2),
                       ('mmd_fast_feature_batch_size', 1), ('mmd_fast_kernel_elements', 1)]:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
      raise ValueError(f'{key} must be an integer >= {minimum}')
  layer = cfg.mmd_feature_layer
  if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < config.model.n_blocks:
    raise ValueError(f'mmd_feature_layer must be in [0, {config.model.n_blocks - 1}]')
  if cfg.get('mmd_feature_layer_weights') is not None:
    raise ValueError('Archive MMD supports a single unweighted feature layer')
  alpha = cfg.mmd_rbf_alpha
  if alpha is None or not math.isfinite(float(alpha)) or float(alpha) <= 0:
    raise ValueError('Archive MMD requires a fixed positive mmd_rbf_alpha')
  if not math.isfinite(float(cfg.mmd_reward_weight)) or cfg.mmd_reward_weight <= 0:
    raise ValueError('mmd_reward_weight must be positive and finite')
  if cfg.get('mmd_baseline', 'auto') not in ('auto', 'group'):
    raise ValueError('Archive MMD uses the within-group leave-one-out baseline')
  if cfg.get('mmd_baseline') == 'group' and cfg.mmd_group_size < 2:
    raise ValueError('A group baseline requires G >= 2; use auto for G=1')
  for key in ('mmd_average_candidates', 'mmd_fast_scored_head', 'mmd_rbf_exclude_same_position'):
    if not isinstance(cfg.get(key), bool):
      raise ValueError(f'{key} must be boolean')
  if cfg.get('mmd_rbf_block_size', 0) < 0:
    raise ValueError('mmd_rbf_block_size must be nonnegative')


class ArchiveObjective:
  def __init__(self, model):
    validate_config(model.config)
    cfg = model.config.adversarial_distill
    kwargs = dict(layer=cfg.mmd_feature_layer, kernel='token_rbf',
                  alpha=cfg.mmd_rbf_alpha, block_size=cfg.mmd_rbf_block_size,
                  feature_batch_size=model._mmd_forward_batch_size(),
                  exclude_same_position=cfg.mmd_rbf_exclude_same_position)
    self.fast = cfg.mmd_implementation == 'fast'
    if self.fast:
      self.reward = FastFeatureReward(
        model.teacher, fast_feature_batch_size=cfg.mmd_fast_feature_batch_size,
        kernel_elements=cfg.mmd_fast_kernel_elements, **kwargs)
    else:
      self.reward = FeatureReward(model.teacher, **kwargs)
    # Installed only after the teacher was copied; idle during inference.
    self.head = ScoredHead(model.backbone) if self.fast and cfg.mmd_fast_scored_head else None

  def log_probs(self, model, xt, sigma, selected):
    size = model._mmd_forward_batch_size() or len(xt)

    def forward(tokens, noise, mask):
      if self.head is None:
        return model(tokens, noise)
      # Matches the archive: FP32 outside the backbone's internal BF16 region.
      with torch.autocast(device_type=tokens.device.type, enabled=False):
        logits = self.head.logits(tokens, model._process_sigma(noise), mask.nonzero(as_tuple=True))
      logits[:, model.mask_index] += model.neg_infinity
      return logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    parts = []
    for start in range(0, len(xt), size):
      args = xt[start:start + size], sigma[start:start + size], selected[start:start + size]
      parts.append(checkpoint(forward, *args, use_reentrant=False, preserve_rng_state=True)
                   if torch.is_grad_enabled() else forward(*args))
    return torch.cat(parts, dim=0)

  def loss(self, model, xt, sigma, x0, selected):
    cfg = model.config.adversarial_distill
    logp = self.log_probs(model, xt, sigma, selected)
    kwargs = dict(group_size=cfg.mmd_group_size,
                  draws_per_candidate=cfg.mmd_kernel_batch_size,
                  average_candidates=cfg.mmd_average_candidates, log_paper=True)
    if self.fast:
      return reinforce_loss_fast(logp, x0, selected, self.reward,
                                 scored=self.head is not None, **kwargs)
    return reinforce_loss(logp.exp(), x0, selected, self.reward, **kwargs)


def generator_loss(model, x0, valid_tokens, accumulation_step):
  objective = getattr(model, '_archive_mmd_objective', None)
  if objective is None:
    objective = model._archive_mmd_objective = ArchiveObjective(model)
  xt, sigma = model._distill_generator_state(x0, valid_tokens, accumulation_step)
  content = model._mmd_content_mask(x0, valid_tokens)
  selected = xt.eq(model.mask_index) & content
  loss, stats = objective.loss(model, xt, sigma, x0, selected)
  # Counts are returned by the archive for weighted aggregation; retain them
  # as diagnostics without treating them as rewards.
  metrics = {k: v for k, v in stats.items() if k.startswith('mmd/')}
  metrics['mmd/generator_mask_fraction'] = selected.sum().float() / content.sum().clamp_min(1)
  model.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=True)
  return loss * model.config.adversarial_distill.mmd_reward_weight
