"""Cosine-kernel MMD rewards without token-by-token Gram matrices.

The feature map is phi(x) = x / max(||x||_2, 1e-12), and the kernel is
k(x,y) = phi(x).dot(phi(y)). There is no hidden-dimension average, pseudo-Huber
transform, or nonnegative clamp. Zero vectors map to zero, so their self-kernel
is zero rather than one.

All reward calculations detach their inputs and use float32. Mean vectors and
centered second moments give the exact diagonal-excluded statistic in linear
space/time; centered moments avoid subtracting nearly equal kernel sums.
"""

import torch


NORMALIZATION_EPS = 1e-12


def _normalized_features(features):
    """Stable fp32 L2 normalization, including zero and sub-epsilon vectors."""
    values = features.detach().float()
    # Scaling first avoids overflowing the norm for large, finite fp32 features.
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(NORMALIZATION_EPS)
    scaled = values / scale
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    return scaled / torch.maximum(norm, NORMALIZATION_EPS / scale)


def _validate_features(real, fake, real_dimensions):
    if not isinstance(real, torch.Tensor) or not isinstance(fake, torch.Tensor):
        raise ValueError('cosine MMD features must be tensors')
    if real.ndim != real_dimensions or fake.ndim != 3:
        raise ValueError(f'cosine MMD expects rank-{real_dimensions} real and '
                         'rank-3 fake features, got '
                         f'{tuple(real.shape)} and {tuple(fake.shape)}')
    if real.shape[0] != fake.shape[0] or real.shape[-1] != fake.shape[-1]:
        raise ValueError('cosine MMD features must share batch and hidden dimensions')
    if real.shape[-1] < 1:
        raise ValueError('cosine MMD needs at least one hidden dimension')
    if real.device != fake.device:
        raise ValueError('cosine MMD features must share a device')
    if real.is_complex() or fake.is_complex():
        raise ValueError('cosine MMD expects real-valued features')


def _binary_weights(weights, shape, device, label):
    if not isinstance(weights, torch.Tensor) or tuple(weights.shape) != tuple(shape):
        raise ValueError(f'cosine MMD {label} mask must have shape {tuple(shape)}')
    if weights.device != device:
        raise ValueError(f'cosine MMD {label} mask must share the features device')
    if weights.is_complex() or not bool(((weights == 0) | (weights == 1)).all()):
        raise ValueError(f'cosine MMD {label} mask must contain only binary 0/1 weights')
    return weights.detach().float()


def _mean_and_correction(features, weights):
    """Weighted mean and trace(sample variance)/n for a binary mask.

    For n selected phi vectors, the correction equals
    [sum_i ||phi_i||^2 - ||sum_i phi_i||^2/n] / [n(n-1)].
    Using centered differences preserves actual diagonal norms, including zeros,
    while improving numerical accuracy for almost identical vectors.
    """
    count = weights.sum(dim=1)
    weighted = weights.unsqueeze(-1)
    mean = (features * weighted).sum(dim=1) / count.clamp_min(1).unsqueeze(-1)
    centered = (features - mean.unsqueeze(1)) * weighted
    correction = centered.square().sum(dim=(1, 2)) / (
        count * (count - 1)).clamp_min(1)
    return mean, correction, count


@torch.no_grad()
def cosine_mmd2_unequal(real, fake, w_real, w_fake):
    """Diagonal-excluded cosine MMD between two possibly unequal sample sets.

    Inputs: real=(B,N,D), fake=(B,M,D), binary masks=(B,N)/(B,M).
    The result is algebraically

      (||sum phi(x)||^2 - sum ||phi(x)||^2) / [n(n-1)]
      + (||sum phi(y)||^2 - sum ||phi(y)||^2) / [m(m-1)]
      - 2 sum phi(x).dot(sum phi(y)) / (nm).

    The implementation uses the equivalent squared mean difference minus the
    two centered-variance corrections. Returns (mmd2, active), both shaped (B,).
    Rows with fewer than two selected vectors on either side are inactive and
    return zero. Negative finite-sample estimates are retained.
    """
    _validate_features(real, fake, real_dimensions=3)
    w_real = _binary_weights(w_real, real.shape[:2], real.device, 'real')
    w_fake = _binary_weights(w_fake, fake.shape[:2], fake.device, 'fake')
    real = _normalized_features(real)
    fake = _normalized_features(fake)
    real_mean, real_correction, n = _mean_and_correction(real, w_real)
    fake_mean, fake_correction, m = _mean_and_correction(fake, w_fake)
    active = (n >= 2) & (m >= 2)
    mmd2 = (real_mean - fake_mean).square().sum(dim=-1)
    mmd2 = mmd2 - real_correction - fake_correction
    return torch.where(active, mmd2, torch.zeros_like(mmd2)), active


@torch.no_grad()
def cosine_mmd2_fixed_real(real, fake):
    """Cosine MMD for one fixed real vector and K sampled fake vectors.

    Inputs: real=(B,D), fake=(B,K,D). For K>1 the fake self-pairs are removed,
    giving the estimator conditional on the fixed real feature. For K=1 use
    the biased fallback ||phi(real)-phi(fake)||^2, equal to 2(1-cosine) when
    both vectors have norms at least epsilon. No factor 1/D or clamp is applied.
    """
    _validate_features(real, fake, real_dimensions=2)
    count = fake.shape[1]
    if count < 1:
        raise ValueError('fixed-real cosine MMD needs at least one fake draw')
    real = _normalized_features(real)
    fake = _normalized_features(fake)
    if count == 1:
        return (real - fake[:, 0]).square().sum(dim=-1)
    mean = fake.mean(dim=1)
    # torch.var uses a centered reduction, retaining precision near agreement.
    correction = fake.var(dim=1, correction=1).sum(dim=-1) / count
    return (real - mean).square().sum(dim=-1) - correction
