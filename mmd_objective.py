"""TinyGSM representation-MMD objective used by paper run 3e0ac6b1.

This is the historical paired-token RBF estimator: G independent one-answer
candidates at one masked state, one clean target per example, and leave-one-out
REINFORCE. It keeps the real-real term, removes self-pairs within each token
set, and sums candidate contributions without 1/G normalization. The released
dlm-mmd two-answer cross-draw estimator is a different objective.

Adapted from IDLM tinygsm-upstream-repro (14594e9), mmd_reward.py and mmd_rbf.py.
The selected paper checkpoint is step 7000 of run
3e0ac6b1-7a8f-43c3-af58-d9e21b886da7 (G=8, kernel batch=1).
"""

import math

import torch


def _kernel_block(left, right, left2, right2, alpha, *, self_tile=False):
    kernel = torch.bmm(left, right.transpose(1, 2))
    kernel.mul_(-2.0).add_(left2.unsqueeze(2)).add_(right2.unsqueeze(1))
    kernel.clamp_(min=0).mul_(-alpha).exp_()
    if self_tile:
        # The estimator subtracts this diagonal analytically as exactly n.
        kernel.diagonal(dim1=1, dim2=2).fill_(1.0)
    return kernel


def _weighted_sum(kernel, left, right):
    return torch.bmm(left.unsqueeze(1), torch.bmm(kernel, right.unsqueeze(2))).reshape(-1)


@torch.no_grad()
def paired_rbf_reward(clean, generated, selected, *, alpha=6e-5, block_size=0):
    """Return detached (-MMD², active) for (N,L,D) paired feature tensors.

    Each row uses only its selected positions. Rows with fewer than two selected
    tokens are inactive. Negative finite-sample MMD² estimates are retained.
    Kernel arithmetic uses FP32 even when the feature backbone uses BF16.
    """
    if clean.ndim != 3 or generated.shape != clean.shape or selected.shape != clean.shape[:2]:
        raise ValueError('Expected matching clean/generated (N,L,D) and selected (N,L)')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('RBF alpha must be finite and positive')
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 0:
        raise ValueError('Kernel block_size must be a nonnegative integer')
    weights = selected.bool().float()
    count = weights.sum(1)
    active = count >= 2
    if not bool(active.any()):
        return torch.zeros_like(count), active
    real, fake = clean.detach().float(), generated.detach().float()
    rows, length = weights.shape
    if block_size == 0:
        block_size = (length if rows * length * length <= (1 << 26)
                      else max(64, min(length, int(((1 << 26) / max(rows, 1)) ** .5))))
    spans = [(start, min(start + block_size, length)) for start in range(0, length, block_size)]
    with torch.autocast(device_type=real.device.type, enabled=False):
        real2, fake2 = (real * real).sum(-1), (fake * fake).sum(-1)
        real_real, fake_fake, cross = [real.new_zeros(rows) for _ in range(3)]
        for index, (i0, i1) in enumerate(spans):
            wi = weights[:, i0:i1]
            for j0, j1 in spans[index:]:
                wj = weights[:, j0:j1]
                diagonal = i0 == j0
                multiplier = 1.0 if diagonal else 2.0
                kernel = _kernel_block(real[:, i0:i1], real[:, j0:j1],
                                       real2[:, i0:i1], real2[:, j0:j1], alpha,
                                       self_tile=diagonal)
                real_real += _weighted_sum(kernel, wi, wj) * multiplier
                kernel = _kernel_block(fake[:, i0:i1], fake[:, j0:j1],
                                       fake2[:, i0:i1], fake2[:, j0:j1], alpha,
                                       self_tile=diagonal)
                fake_fake += _weighted_sum(kernel, wi, wj) * multiplier
            for j0, j1 in spans:
                kernel = _kernel_block(real[:, i0:i1], fake[:, j0:j1],
                                       real2[:, i0:i1], fake2[:, j0:j1], alpha)
                cross += _weighted_sum(kernel, wi, weights[:, j0:j1])
        safe_count = count.clamp_min(2.0)
        pairs = safe_count * (safe_count - 1.0)
        mmd2 = ((real_real - safe_count) / pairs + (fake_fake - safe_count) / pairs
                - 2.0 * cross / (safe_count * safe_count))
        return torch.where(active, -mmd2, torch.zeros_like(mmd2)), active


class _FeaturesReady(Exception):
    pass


class FeatureReward:
    """Frozen clean-MDLM features from one zero-based DiT block.

    The owner registers/moves/checkpoints the backbone. This plain holder does
    not add a second registered copy. Physical feature chunks change neither
    the per-example kernel statistic nor the REINFORCE group.
    """

    def __init__(self, backbone, layer=6, kernel='rbf', alpha=6e-5,
                 block_size=0, feature_batch_size=8):
        if kernel != 'rbf':
            raise ValueError('The historical TinyGSM objective requires kernel=rbf')
        core = getattr(backbone, 'backbone', backbone)
        blocks = getattr(core, 'blocks', None)
        if blocks is None or isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < len(blocks):
            raise ValueError('Feature layer must be a valid zero-based DiT block index')
        if (feature_batch_size is not None and (isinstance(feature_batch_size, bool)
                or not isinstance(feature_batch_size, int) or feature_batch_size <= 0)):
            raise ValueError('feature_batch_size must be a positive integer or None')
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('RBF alpha must be finite and positive')
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 0:
            raise ValueError('Kernel block_size must be a nonnegative integer')
        self.backbone, self.block = backbone, blocks[layer]
        self.alpha, self.block_size = alpha, block_size
        self.feature_batch_size = feature_batch_size
        self.backbone.requires_grad_(False).eval()

    @torch.no_grad()
    def features(self, tokens):
        captured = []

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(hidden.detach())
            raise _FeaturesReady

        self.backbone.eval()
        handle = self.block.register_forward_hook(capture)
        try:
            chunk_size = self.feature_batch_size or tokens.shape[0]
            for start in range(0, tokens.shape[0], chunk_size):
                chunk = tokens[start:start + chunk_size]
                sigma = torch.zeros(chunk.shape[0], device=chunk.device, dtype=torch.float32)
                try:
                    self.backbone(chunk, sigma)
                except _FeaturesReady:
                    continue
                raise RuntimeError('The selected MDLM feature block did not run')
        finally:
            handle.remove()
        return torch.cat(captured)

    @torch.no_grad()
    def __call__(self, x0, draws, selected):
        if draws.shape != (1, *x0.shape) or selected.shape != x0.shape:
            raise ValueError('The historical kernel needs one draw per candidate: (1,N,L)')
        # The historical feature extractor consumes this draw even with noise=0.
        torch.rand(x0.shape, device=x0.device)
        hidden = self.features(torch.cat([x0, draws[0]], dim=0))
        count = x0.shape[0]
        return paired_rbf_reward(hidden[:count], hidden[count:], selected,
                                 alpha=self.alpha, block_size=self.block_size)


def _validate_reinforce_settings(group_size, draws_per_candidate, average_candidates):
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError('group_size must be a positive integer')
    if draws_per_candidate != 1 or isinstance(draws_per_candidate, bool):
        raise ValueError('Historical TinyGSM uses draws_per_candidate=1; B>1 is a different estimator')
    if not isinstance(average_candidates, bool):
        raise ValueError('average_candidates must be a boolean')


def _zero_loss(values):
    zero = values.sum() * 0
    return zero, {name: zero.detach() for name in
                  ('mmd/reward', 'mmd/advantage_absmean', 'mmd/reward_spread', 'reward_count')}


def reinforce_loss(student_probs, x0, selected, reward_fn, *, group_size=8,
                   draws_per_candidate=1, average_candidates=False):
    """Historical TinyGSM score estimator, with no candidate averaging by default.

    Reward is detached; gradients reach the student only through sampled-token
    log probabilities. Each example's group baseline excludes the scored draw.
    G=1 uses the raw reward. All candidates share one student prediction/mask.
    """
    _validate_reinforce_settings(group_size, draws_per_candidate, average_candidates)
    if (student_probs.ndim != 3 or student_probs.shape[:2] != x0.shape
            or selected.shape != x0.shape):
        raise ValueError('Expected probabilities (N,L,V), tokens (N,L), and mask (N,L)')
    selected = selected.bool()
    if not bool(selected.any()):
        return _zero_loss(student_probs)
    with torch.no_grad():
        sampled = torch.multinomial(student_probs.reshape(-1, student_probs.shape[-1]).float(),
                                    group_size, replacement=True)
        sampled = sampled.view(*x0.shape, group_size).permute(2, 0, 1)
        samples = torch.where(selected.unsqueeze(0), sampled, x0.unsqueeze(0))
    logp = torch.log(student_probs.clamp_min(1e-12))
    sampled_logp = torch.stack([logp.gather(-1, sample.unsqueeze(-1)).squeeze(-1)
                                for sample in samples])
    return reinforce_loss_from_samples(
        sampled_logp, x0, selected, samples, reward_fn, group_size=group_size,
        draws_per_candidate=draws_per_candidate, average_candidates=average_candidates)


def reinforce_loss_from_samples(sampled_logp, x0, selected, samples, reward_fn, *,
                                group_size=8, draws_per_candidate=1,
                                average_candidates=False):
    """Apply full logical-batch MMD after checkpointed physical student forwards.

    ``samples`` and differentiable ``sampled_logp`` have shape (G,N,L).
    Callers may construct them in smaller N chunks and concatenate on axis 1;
    feature rewards, leave-one-out baselines and normalization still see the
    full loader batch. Unselected tokens must remain equal to the clean input.
    """
    _validate_reinforce_settings(group_size, draws_per_candidate, average_candidates)
    expected = (group_size, *x0.shape)
    if (x0.ndim != 2 or selected.shape != x0.shape or samples.shape != expected
            or sampled_logp.shape != expected):
        raise ValueError('Expected tokens/mask (N,L) and samples/sampled_logp (G,N,L)')
    if samples.dtype != torch.long or not sampled_logp.is_floating_point():
        raise ValueError('Samples must be torch.long and sampled_logp floating point')
    selected = selected.bool()
    if bool(((samples != x0.unsqueeze(0)) & ~selected.unsqueeze(0)).any()):
        raise ValueError('Unselected sample tokens must equal the clean input')
    if not bool(selected.any()):
        return _zero_loss(sampled_logp)
    with torch.no_grad():
        rewards = []
        active = None
        for index in range(group_size):
            value, row_active = reward_fn(x0, samples[index:index + 1], selected)
            if value.shape != (x0.shape[0],) or row_active.shape != value.shape:
                raise ValueError('Reward and active mask must have one entry per example')
            if active is None:
                active = row_active.bool()
            elif not torch.equal(active, row_active.bool()):
                raise ValueError('The same selected mask must give identical active rows for each candidate')
            rewards.append(value.detach())
        reward = torch.stack(rewards, -1)
        advantage = (reward - (reward.sum(-1, keepdim=True) - reward) / (group_size - 1)
                     if group_size > 1 else reward)
        advantage = torch.where(active.unsqueeze(-1), advantage, torch.zeros_like(advantage))
    # Keep the historical sequential sum over candidates and its dtype behavior.
    surrogate = sampled_logp.new_zeros(())
    for index in range(group_size):
        token_logp = sampled_logp[index]
        seq_logp = (token_logp * selected.to(token_logp.dtype)).sum(1)
        surrogate = surrogate + (advantage[:, index] * seq_logp).sum()
    denominator = selected.sum().clamp_min(1).to(surrogate.dtype)
    if average_candidates:
        denominator = denominator * group_size
    loss = -surrogate / denominator
    scored = reward[active] if bool(active.any()) else reward.new_zeros(1)
    return loss, {
        'mmd/reward': scored.mean().detach(),
        'mmd/advantage_absmean': advantage.abs().mean().detach(),
        'mmd/reward_spread': (reward.std(-1).mean().detach()
                              if group_size > 1 else reward.new_zeros(())),
        'reward_count': active.sum().detach(),
    }
