"""Optional BD3-LMs-style clipping of distillation generator masking rates.

Adaptive bounds minimize held-out, per-sequence denoising-loss variance.
This is a proxy for training variance, not an estimate of MMD gradient variance.
Only generator input corruption changes; critic noise and reported NLL retain
their original sampling distributions.
"""
import math

import torch
import torch.distributed as dist


DEFAULT_INTERVALS = (
  (0.001, 1.0), (0.1, 0.6), (0.2, 0.7), (0.3, 0.8),
  (0.4, 0.9), (0.45, 0.95), (0.5, 1.0))


class ClippedMaskSchedule:
  @classmethod
  def from_config(cls, distill_cfg, sampling_eps, noise_eps):
    return cls(distill_cfg, sampling_eps, noise_eps)

  def __init__(self, distill_cfg, sampling_eps, noise_eps):
    cfg = distill_cfg.get('mask_schedule', {}) or {}
    if not hasattr(cfg, 'get'):
      raise ValueError('adversarial_distill.mask_schedule must be a mapping')
    self.mode = str(cfg.get('mode', 'original'))
    if self.mode not in {'original', 'clipped', 'adaptive'}:
      raise ValueError('mask_schedule.mode must be original, clipped, or adaptive')
    self.sampling_eps = float(sampling_eps)
    self.noise_eps = float(noise_eps)
    self.bounds = None
    self.candidates = ()
    self.search_batches = 0
    if self.mode == 'original':
      return
    if not distill_cfg.get('is_distill', False):
      raise ValueError('Clipped masking requires adversarial_distill.is_distill=True')
    if float(distill_cfg.get('high_mask_fraction', 0.0)) != 0.0:
      raise ValueError('Clipped masking requires high_mask_fraction=0; '
                       'choose one generator masking option')
    if distill_cfg.get('generator_force_full_mask', False):
      raise ValueError('Clipped masking cannot be combined with generator_force_full_mask')
    if not (0 < self.sampling_eps < 1 and 0 <= self.noise_eps < 1):
      raise ValueError('Clipped masking requires 0 < sampling_eps < 1 and '
                       '0 <= noise_eps < 1')
    self.bounds = self._effective_bounds((cfg.get('min', 0.45), cfg.get('max', 0.95)))
    self.initial_bounds = self.bounds
    if self.mode == 'adaptive':
      batches = cfg.get('search_batches', 4)
      if isinstance(batches, bool) or int(batches) != batches or batches < 1:
        raise ValueError('mask_schedule.search_batches must be a positive integer')
      self.search_batches = int(batches)
      intervals = cfg.get('search_intervals', DEFAULT_INTERVALS)
      if not intervals:
        raise ValueError('mask_schedule.search_intervals must not be empty')
      candidates = [self._effective_bounds(interval) for interval in intervals]
      # Keep the initial interval eligible, including when custom candidates
      # are supplied. Ties can then retain the currently selected interval.
      self.candidates = tuple(dict.fromkeys([self.bounds] + candidates))

  def _effective_bounds(self, interval):
    if len(interval) != 2:
      raise ValueError('Each masking interval must contain [min, max]')
    lower, upper = map(float, interval)
    if not (math.isfinite(lower) and math.isfinite(upper)
            and 0 <= lower < upper <= 1):
      raise ValueError(f'Masking intervals require 0 <= min < max <= 1: {interval}')
    # Limit the endpoints before sampling, avoiding probability mass piling
    # up at t=1 when a requested mask maximum is 1 but alpha(1)=noise_eps.
    lower = max(lower, self.sampling_eps * (1 - self.noise_eps))
    upper = min(upper, 1 - self.noise_eps)
    if lower >= upper:
      raise ValueError(f'Masking interval {interval} has no width within the noise support')
    return lower, upper

  def sample(self, base_t, bounds=None):
    if self.mode == 'original':
      return base_t
    lower, upper = self.bounds if bounds is None else bounds
    # Reuse the original stratified draws across ranks/accumulation steps.
    # No additional random numbers are needed for a clipped interval.
    u = ((base_t - self.sampling_eps) / (1 - self.sampling_eps)).clamp(0, 1)
    return ((lower + (upper - lower) * u) / (1 - self.noise_eps)).clamp(
      self.sampling_eps, 1.0)

  def _signature(self):
    return (self.mode, self.initial_bounds, self.candidates,
            self.sampling_eps, self.noise_eps, self.search_batches)

  def state_dict(self):
    if self.mode != 'adaptive':
      return {}
    return {'version': 1, 'signature': self._signature(), 'bounds': self.bounds}

  def load_state_dict(self, state):
    # A fresh schedule configuration intentionally starts from its new bounds.
    # No model buffers are added, so old strict state_dict loads still work.
    if (self.mode != 'adaptive' or not state or state.get('version') != 1
        or state.get('signature') != self._signature()):
      return False
    bounds = tuple(state['bounds'])
    if bounds not in self.candidates:
      return False
    self.bounds = bounds
    return True

  def new_statistics(self, device):
    # count, sum, squared sum, nonfinite count; sufficient for an exact
    # pooled variance across batches/ranks rather than averaging rank variances.
    return torch.zeros(len(self.candidates), 4, dtype=torch.float64, device=device)

  def observe(self, statistics, index, per_row_losses):
    values = per_row_losses.detach().to(dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(values)
    values_finite = values[finite]
    statistics[index] += torch.stack((
      finite.sum(), values_finite.sum(), values_finite.square().sum(),
      (~finite).sum())).to(statistics)

  def variances(self, statistics):
    count, total, squares, bad = statistics.unbind(-1)
    safe_count = count.clamp_min(1)
    variance = (squares / safe_count - (total / safe_count).square()).clamp_min(0)
    valid = (count >= 2) & (bad == 0) & torch.isfinite(variance)
    return torch.where(valid, variance, torch.full_like(variance, float('inf')))

  def select(self, statistics):
    variances = self.variances(statistics)
    best = int(variances.argmin().item())
    if not torch.isfinite(variances[best]):
      return None
    current = self.candidates.index(self.bounds)
    if variances[current] == variances[best]:
      best = current
    self.bounds = self.candidates[best]
    return best


class DistillMaskScheduleMixin:
  """Shared MDLM hooks; does not depend on Lightning or either branch's algo."""

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._mask_schedule = ClippedMaskSchedule.from_config(
      config.adversarial_distill, self.sampling_eps, self.noise.eps)
    self._mask_schedule_statistics = None

  def on_load_checkpoint(self, checkpoint):
    super().on_load_checkpoint(checkpoint)
    schedule = getattr(self, '_mask_schedule', None)
    if schedule is not None:
      schedule.load_state_dict(checkpoint.get('distill_mask_schedule'))

  def on_save_checkpoint(self, checkpoint):
    super().on_save_checkpoint(checkpoint)
    schedule = getattr(self, '_mask_schedule', None)
    if schedule is not None and schedule.mode == 'adaptive':
      checkpoint['distill_mask_schedule'] = schedule.state_dict()

  def on_validation_epoch_start(self):
    super().on_validation_epoch_start()
    self._mask_schedule_statistics = None
    schedule = getattr(self, '_mask_schedule', None)
    if (schedule is not None and schedule.mode == 'adaptive'
        and self.config.mode == 'train' and not self.trainer.sanity_checking
        and self.global_step > 0):
      self._mask_schedule_statistics = schedule.new_statistics(self.device)

  def validation_step(self, batch, batch_idx):
    result = super().validation_step(batch, batch_idx)
    statistics = getattr(self, '_mask_schedule_statistics', None)
    if statistics is not None and batch_idx < self._mask_schedule.search_batches:
      self._probe_mask_schedule(batch, statistics)
    return result

  @torch.no_grad()
  def _probe_mask_schedule(self, batch, statistics):
    schedule = self._mask_schedule
    x0, _, valid_tokens = self._process_model_input(
      batch['input_ids'], batch['attention_mask'])
    scored = (torch.ones_like(x0, dtype=torch.bool) if valid_tokens is None
              else valid_tokens.bool().clone())
    if self.ignore_bos:
      scored[:, 0] = False
    counts = scored.sum(-1)
    active = counts > 0
    if not active.any():
      return
    # Extra validation probes must not advance the training/normal validation
    # RNG stream. Reuse both time draws and corruption randomness for each
    # candidate to reduce differences caused merely by corruption luck.
    devices = [x0.device.index] if x0.is_cuda else []
    with torch.random.fork_rng(devices=devices):
      base_t = self._sample_t(x0.shape[0], None)
      cpu_state = torch.random.get_rng_state()
      cuda_state = torch.cuda.get_rng_state(x0.device) if x0.is_cuda else None
      for index, bounds in enumerate(schedule.candidates):
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
          torch.cuda.set_rng_state(cuda_state, x0.device)
        t = schedule.sample(base_t, bounds)
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        xt = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)
        log_probs = self.forward(xt, sigma=self._sigma_from_alphat(alpha_t))
        losses = self.nll_per_token(
          log_x_theta=log_probs, xt=xt, x0=x0, alpha_t=alpha_t,
          dalpha_t=dalpha_t, low_var=False)
        row_losses = losses.masked_fill(~scored, 0).sum(-1) / counts.clamp_min(1)
        schedule.observe(statistics, index, row_losses[active])
        del log_probs, losses

  def on_validation_epoch_end(self):
    statistics = getattr(self, '_mask_schedule_statistics', None)
    schedule = getattr(self, '_mask_schedule', None)
    if statistics is not None:
      if dist.is_available() and dist.is_initialized():
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
      selected = schedule.select(statistics)
      stats = {'mask_schedule/updated': float(selected is not None),
               'mask_schedule/min': schedule.bounds[0],
               'mask_schedule/max': schedule.bounds[1]}
      if selected is not None:
        stats['mask_schedule/denoising_variance'] = schedule.variances(statistics)[selected]
        stats['mask_schedule/probe_rows'] = statistics[selected, 0]
      # All ranks have the same pooled statistics and chosen bounds.
      self.log_dict(stats, on_step=False, on_epoch=True, sync_dist=False)
      self._mask_schedule_statistics = None
    super().on_validation_epoch_end()
