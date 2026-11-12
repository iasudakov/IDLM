"""Efficient implementation of the grouped REINFORCE / feature-MMD objective.

Same estimator as ``mmd.objective`` (``reinforce_loss`` + ``FeatureReward``):
identical rewards, advantages, loss and gradient for the same sampled tokens,
up to float32 rounding. Selected by ``mmd.implementation=fast`` (the default
here; ``reference`` selects ``mmd.objective``). What changes is only how the
work is organised:

Feature extraction
  * The real sequences are encoded once per step instead of once per
    candidate: (1 + G*B) * N extractor inputs instead of G * (1 + B) * N.
  * Rows whose reward is not used (no scored position) are not encoded.
  * Identical sequences of a row are encoded once. Samples of a confident
    model often coincide (about half of the TinyGSM extractor inputs at the
    start of training; almost none on OpenWebText).
  * Extractor forwards use ``fast_feature_batch_size`` sequences (the reference
    uses ``feature_batch_size``, 8), and only the features of scored positions
    are kept.

MMD
  * Kernels are evaluated between scored positions only, C x C per pair of
    sequences instead of L x L (C = largest number of scored positions of a row
    in the microbatch), and for all G candidates in one batched product.

REINFORCE
  * Tokens are drawn only at scored positions (the reference draws G*B tokens
    for every position of every sequence and discards the rest).
  * The log-probabilities of the sampled tokens are gathered from the model's
    log-probabilities; the reference materialises probabilities and their
    clamped logarithm for the full (N, L, V) tensor and backpropagates through
    both.
  * Optionally (``scored=True``; ``mmd.fast_scored_head`` in training) the
    student's log-probabilities are given for the scored positions only, so the
    vocabulary-sized output layer and log-softmax, and their backward, skip
    the positions the objective never reads (``ScoredHead``).

The sampling distribution is unchanged, but the random stream is not the
reference's (fewer draws are consumed), so runs are statistically, not
bitwise, comparable. ``draws=`` injects reference draws for exact checks; see
``tests/test_objective_fast.py``.
"""

import torch

from mmd.objective import FeatureReward, _FeaturesReady


class ScoredHead:
    """Evaluate an MDLM backbone's output layer at chosen positions only.

    ``logits(tokens, sigma, positions)`` returns the rows ``backbone(tokens,
    sigma)[positions]`` (C,V), ``positions`` being ``mask.nonzero(as_tuple=True)``.
    The transformer blocks run on full sequences as usual; only the final
    normalisation, modulation and vocabulary projection are restricted.
    """

    def __init__(self, backbone):
        core = getattr(backbone, 'backbone', backbone)
        if not hasattr(core, 'output_layer'):
            raise ValueError('Expected an MDLM backbone with an output_layer')
        self.backbone = backbone
        self.positions = None
        core.output_layer.register_forward_pre_hook(self._restrict, with_kwargs=True)

    def _restrict(self, _module, args, kwargs):
        if self.positions is None:
            return None
        hidden = args[0]
        condition = kwargs['c'] if 'c' in kwargs else args[1]
        # (C,1,D) hidden states with their rows' conditioning: the layer's (N,L,D) code path.
        return (hidden[self.positions][:, None], condition[self.positions[0]]), {}

    def logits(self, tokens, sigma, positions):
        self.positions = positions
        try:
            return self.backbone(tokens, sigma).squeeze(1)
        finally:
            self.positions = None


class FastFeatureReward(FeatureReward):
    """``FeatureReward`` scoring all G candidates of a microbatch in one call."""

    def __init__(self, *args, fast_feature_batch_size=64, kernel_elements=1 << 27,
                 share_repeated=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_repeated = share_repeated
        if fast_feature_batch_size is not None and fast_feature_batch_size < 1:
            raise ValueError('fast_feature_batch_size must be positive or None')
        self.fast_feature_batch_size = fast_feature_batch_size
        # Largest kernel block (elements) held at once; rows are chunked above it.
        self.kernel_elements = kernel_elements
        # Running totals: sequences scored and sequences actually encoded.
        self.sequences_scored = 0
        self.sequences_encoded = 0

    @torch.no_grad()
    def features_at(self, tokens, index):
        """Block output at chosen positions: tokens (M,L), index (M,C) -> float32 (M,C,D)."""
        captured = []

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(hidden.detach())
            # No downstream layer or vocabulary-sized output is used by MMD.
            raise _FeaturesReady

        self.backbone.eval()
        handle = self.block.register_forward_hook(capture)
        chunks = []
        try:
            size = self.fast_feature_batch_size or tokens.shape[0]
            for start in range(0, tokens.shape[0], size):
                chunk = tokens[start:start + size]
                try:
                    # As in the student's forward: outside the backbone's own bf16 region
                    # everything is float32. Under the trainer's bf16 autocast the rotary
                    # phases would otherwise be computed and cached in bf16.
                    with torch.autocast(device_type=chunk.device.type, enabled=False):
                        self.backbone(chunk, torch.zeros(chunk.shape[0], device=chunk.device))
                except _FeaturesReady:
                    hidden = captured.pop()
                    where = index[start:start + size, :, None].expand(-1, -1, hidden.shape[-1])
                    chunks.append(hidden.gather(1, where).float())
                    continue
                raise RuntimeError('The selected MDLM feature block did not run')
        finally:
            handle.remove()
        return torch.cat(chunks)

    def _weighted_kernel(self, left, right, left2, right2, weight_left, weight_right):
        """RBF kernel (R,P,Q) from (R,P,D), (R,Q,D) with unscored entries zeroed."""
        kernel = torch.bmm(left, right.transpose(1, 2))
        kernel.mul_(-2).add_(left2[:, :, None]).add_(right2[:, None, :]).clamp_min_(0)
        kernel.mul_(-self.alpha).exp_()
        return kernel.mul_(weight_left[:, :, None]).mul_(weight_right[:, None, :])

    @torch.no_grad()
    def score_candidates(self, x0, samples, selected, group_size, *, return_paper=False):
        """Rewards (N,G) and active rows (N,) for samples (G*B,N,L), candidate-major.

        With ``return_paper`` also the paper Eq. 6 surrogate (N,G). Values match
        ``FeatureReward.__call__`` applied to each candidate separately.
        """
        if self.kernel != 'token_rbf':
            raise ValueError(f'Only token_rbf is supported, got {self.kernel}')
        n, length = x0.shape
        total = samples.shape[0]
        draws = total // group_size
        if samples.shape[1:] != x0.shape or selected.shape != x0.shape or draws * group_size != total:
            raise ValueError('Expected x0 (N,L), samples (G*B,N,L), selected (N,L)')
        if draws < 2:
            raise ValueError('token_rbf needs B >= 2')
        selected = selected.bool()
        count = selected.sum(-1)
        minimum = 2 if self.exclude_same_position else 1
        active = count >= minimum
        rewards = torch.zeros(n, group_size, device=x0.device)
        paper = torch.zeros_like(rewards) if return_paper else None
        # Rows that contribute a reward (or an Eq. 6 value); the others are not encoded.
        host = count.cpu()  # the only synchronisation for the row bookkeeping
        rows = (host >= (1 if return_paper else minimum)).nonzero().flatten().to(x0.device)
        if rows.numel():
            width = int(host.max())
            # Scored positions first, in ascending order; the tail is padding with weight 0.
            index = selected[rows].to(torch.uint8).argsort(dim=-1, descending=True, stable=True)[:, :width]
            kept = count[rows]
            weight = (torch.arange(width, device=x0.device)[None] < kept[:, None]).float()
            # Real rows, then each row's G*B samples (row-major, candidate-major within a row).
            r = rows.numel()
            tokens = torch.cat([x0[rows], samples[:, rows].transpose(0, 1).reshape(-1, length)])
            owner = torch.arange(r, device=x0.device)
            owner = torch.cat([owner, owner.repeat_interleave(total)])
            first, inverse = self._distinct(tokens, owner)
            self.sequences_scored += tokens.shape[0]
            if first is None:
                self.sequences_encoded += tokens.shape[0]
                features = self.features_at(tokens, index[owner])
            else:
                self.sequences_encoded += first.shape[0]
                features = self.features_at(tokens[first], index[owner[first]])[inverse]
            real = features[:r]
            fake = features[r:].reshape(r, total, width, -1)
            with torch.autocast(device_type=features.device.type, enabled=False):
                cross = torch.empty(r, group_size, device=x0.device)
                within = torch.zeros_like(cross)
                diagonal = torch.zeros_like(cross)
                step = max(1, self.kernel_elements // (total * width * width))
                for start in range(0, r, step):
                    stop = min(r, start + step)
                    cross[start:stop], within[start:stop], diagonal[start:stop] = self._kernel_sums(
                        real[start:stop], fake[start:stop], weight[start:stop], group_size)
                kept = kept.float()[:, None]
                squares = kept.square()
                reward = 2 * cross / (draws * squares).clamp_min(1)
                pairs = kept * (kept - 1) if self.exclude_same_position else squares
                reward -= within / (draws * (draws - 1) * pairs).clamp_min(1)
                rewards[rows] = torch.where(active[rows, None], reward, torch.zeros_like(reward))
                if return_paper:
                    value = (within + diagonal) / (draws * (draws - 1) * squares).clamp_min(1)
                    paper[rows] = value - 2 * cross / (draws * squares).clamp_min(1)
        return (rewards, active, paper) if return_paper else (rewards, active)

    def _distinct(self, tokens, owner):
        """One representative per distinct (row, sequence) and the map back, or (None, None).

        Sequences are grouped by a 64-bit hash and the grouping is then verified
        token by token, so sharing features is exact; a hash collision only
        disables sharing for that call.
        """
        if not self.share_repeated:
            return None, None
        generator = torch.Generator(device='cpu').manual_seed(0x5EED)
        weights = torch.randint(-(1 << 62), 1 << 62, (tokens.shape[1] + 1,), generator=generator)
        weights = (weights * 2 + 1).to(tokens.device)
        key = (tokens * weights[1:]).sum(-1) + owner * weights[0]  # wraps modulo 2**64
        distinct, inverse = torch.unique(key, return_inverse=True)
        if distinct.shape[0] == tokens.shape[0]:
            return None, None
        order = torch.arange(tokens.shape[0], device=tokens.device)
        first = torch.empty_like(distinct).scatter_(0, inverse, order)
        same = torch.equal(tokens[first][inverse], tokens) and torch.equal(owner[first][inverse], owner)
        return (first, inverse) if same else (None, None)

    def _kernel_sums(self, real, fake, weight, group_size):
        """Per row and candidate: real-fake sum, fake-fake sum over ordered pairs of
        distinct sequences (same positions excluded when configured), and the
        excluded same-position part."""
        r, total, width, dim = fake.shape
        draws = total // group_size
        real2 = real.square().sum(-1)
        fake2 = fake.square().sum(-1)
        kernel = self._weighted_kernel(real, fake.reshape(r, total * width, dim), real2,
                                       fake2.reshape(r, total * width), weight, weight.repeat(1, total))
        cross = kernel.sum(1).reshape(r, group_size, draws * width).sum(-1)
        del kernel
        grouped = fake.reshape(r * group_size, draws, width, dim)
        grouped2 = fake2.reshape(r * group_size, draws, width)
        group_weight = weight.repeat_interleave(group_size, 0)
        within = cross.new_zeros(r * group_size)
        diagonal = torch.zeros_like(within)
        for i in range(draws):
            for j in range(i + 1, draws):
                kernel = self._weighted_kernel(grouped[:, i], grouped[:, j], grouped2[:, i], grouped2[:, j],
                                               group_weight, group_weight)
                if self.exclude_same_position:
                    same = kernel.diagonal(dim1=1, dim2=2)
                    diagonal.add_(same.sum(-1), alpha=2)
                    same.zero_()
                within.add_(kernel.sum((1, 2)), alpha=2)
        return cross, within.reshape(r, group_size), diagonal.reshape(r, group_size)


def reinforce_loss_fast(student_log_probs, x0, selected, reward_fn, *, group_size=4,
                        draws_per_candidate=2, average_candidates=True, log_paper=False, draws=None,
                        scored=False):
    """``mmd.objective.reinforce_loss`` from log-probabilities (N,L,V), restructured.

    ``reward_fn`` is a ``FastFeatureReward``. With ``scored=True`` the
    log-probabilities are (C,V): the rows of the scored positions, in the order
    of ``selected.nonzero()``. ``draws`` (N,L,G*B) optionally replaces the
    internal sampling (only scored positions are read); it is used to compare
    against the reference on identical samples.
    """
    if group_size < 1 or draws_per_candidate < 1:
        raise ValueError('group_size and draws_per_candidate must be positive')
    selected = selected.bool()
    total = group_size * draws_per_candidate
    where = selected.nonzero(as_tuple=True)
    if selected.shape != x0.shape or (
            student_log_probs.shape[:-1] != where[0].shape if scored else student_log_probs.shape[:2] != x0.shape):
        raise ValueError('Student log-probabilities, tokens and mask must share (N,L), '
                         'or with scored=True give one row per scored position')
    if not where[0].numel():
        zero = student_log_probs.sum() * 0
        stats = {'mmd/reward': zero.detach(), 'mmd/advantage_absmean': zero.detach(),
                 'reward_count': zero.detach()}
        if log_paper:
            stats.update({'mmd/paper_surrogate': zero.detach(), 'paper_count': zero.detach()})
        return zero, stats
    with torch.no_grad():
        if draws is None:
            rows = student_log_probs.detach()
            rows = (rows if scored else rows[where]).float().exp()
            scored_draws = torch.multinomial(rows, total, replacement=True)
            del rows
        else:
            scored_draws = draws[where]
        tokens = x0[:, :, None].repeat(1, 1, total)
        tokens[where] = scored_draws
        samples = tokens.permute(2, 0, 1)  # (G*B, N, L), candidate-major as in the reference
        result = reward_fn.score_candidates(x0, samples, selected, group_size, return_paper=log_paper)
        rewards, active = result[:2]
        if group_size > 1:
            advantages = rewards - (rewards.sum(-1, keepdim=True) - rewards) / (group_size - 1)
        else:
            advantages = rewards
        advantages = advantages.masked_fill(~active[:, None], 0)
    if scored:
        sampled_logp = student_log_probs.gather(-1, scored_draws)
        placed = sampled_logp.new_zeros(*x0.shape, total)
        placed[where] = sampled_logp
        seq_logp = placed.sum(1)
    else:
        sampled_logp = student_log_probs.gather(-1, tokens)
        seq_logp = (sampled_logp * selected[:, :, None]).sum(1)
    seq_logp = seq_logp.reshape(-1, group_size, draws_per_candidate).sum(-1)
    group_divisor = group_size if average_candidates else 1
    denominator = selected.sum().clamp_min(1) * group_divisor * draws_per_candidate
    loss = -(advantages * seq_logp).sum() / denominator
    # Means over active rows, written without boolean indexing (no host synchronisation).
    reward_count = active.sum()
    stats = {
        'mmd/reward': (rewards * active[:, None]).sum() / (reward_count * group_size).clamp_min(1),
        'reward_count': reward_count,
        'mmd/advantage_absmean': advantages.abs().mean(),
    }
    if log_paper:
        paper_active = selected.any(-1)
        paper_count = paper_active.sum()
        stats.update({'mmd/paper_surrogate': (result[2] * paper_active[:, None]).sum()
                      / (paper_count * group_size).clamp_min(1),
                      'paper_count': paper_count})
    return loss, stats
