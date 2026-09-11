"""Fixed clipping of distillation generator masking probabilities.

Uses the same interval intersection and stratified time remapping as OWT.
Only generator inputs change; loss corruption and evaluation keep their
original schedules. Fixed bounds live in the config, not checkpoint tensors.
"""
import math


class ClippedMaskSchedule:
  @classmethod
  def from_config(cls, distill_cfg, sampling_eps, noise_eps):
    return cls(distill_cfg, sampling_eps, noise_eps)

  def __init__(self, distill_cfg, sampling_eps, noise_eps):
    cfg = distill_cfg.get('mask_schedule', {}) or {}
    if not hasattr(cfg, 'get'):
      raise ValueError('adversarial_distill.mask_schedule must be a mapping')
    self.mode = str(cfg.get('mode', 'original'))
    if self.mode not in {'original', 'clipped'}:
      raise ValueError('mask_schedule.mode must be original or clipped')
    self.sampling_eps = float(sampling_eps)
    self.noise_eps = float(noise_eps)
    self.bounds = None
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

  def _effective_bounds(self, interval):
    if len(interval) != 2:
      raise ValueError('Each masking interval must contain [min, max]')
    lower, upper = map(float, interval)
    if not (math.isfinite(lower) and math.isfinite(upper)
            and 0 <= lower < upper <= 1):
      raise ValueError(f'Masking intervals require 0 <= min < max <= 1: {interval}')
    # Intersect the interval before drawing, so bounds beyond the noise
    # support do not pile up probability mass at the clipping endpoints.
    lower = max(lower, self.sampling_eps * (1 - self.noise_eps))
    upper = min(upper, 1 - self.noise_eps)
    if lower >= upper:
      raise ValueError(f'Masking interval {interval} has no width within the noise support')
    return lower, upper

  def sample(self, base_t):
    if self.mode == 'original':
      return base_t
    lower, upper = self.bounds
    # Reuse the original stratified draws across ranks/accumulation steps.
    # No additional random numbers are needed for a clipped interval.
    u = ((base_t - self.sampling_eps) / (1 - self.sampling_eps)).clamp(0, 1)
    return ((lower + (upper - lower) * u) / (1 - self.noise_eps)).clamp(
      self.sampling_eps, 1.0)
