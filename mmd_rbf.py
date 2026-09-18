"""Blocked MMD kernels without retaining the (R, N, N) Gram matrix.

The default is the Gaussian kernel discussed below. The optional
``distance='pseudo_huber'`` uses exp(-alpha * D), where
D = sum_d(sqrt((a_d-b_d)**2 + delta**2) - delta). It uses fused Triton
distance/gradient reductions on supported CUDA inputs and a tiled PyTorch
fallback otherwise, recomputing its own exact fake-only derivative.

The rbf branch of an MMD loss only ever consumes three SCALARS per kernel row:

    S_xx = sum_ij w_i w_j k(x_i, x_j)
    S_yy = sum_ij w_i w_j k(y_i, y_j)
    S_xy = sum_ij w_i w_j k(x_i, y_j)          k(a, b) = exp(-alpha ||a - b||^2)

so the (R, N, N) Gram matrices those sums are reduced out of are pure scratch.
Building them anyway is what caps the batch: a row of N tokens costs R * N^2
floats PER MATRIX and the straight-line version keeps about a dozen of them
alive at once (three kernels, each with its pre-clamp distance saved by
`clamp_min` and its result saved by `exp`, plus the Gram temporaries), all of it
held until the backward.

This module reduces the same sums a TILE at a time and never keeps a tile:
memory becomes O(block^2) of scratch plus O(N * H) of saved inputs, i.e. it
stops depending on N^2 at all. It is a different summation order, not the same
arithmetic, so it agrees with the dense path to fp32/TF32 rounding rather than
exactly.

The backward is the reason this is a `torch.autograd.Function` rather than a
chunked forward: autograd would have to keep the tiles to differentiate them,
which is the thing being avoided. Differentiating the sums by hand instead,

    d S_yy / d y_i = -4 alpha w_i ( y_i * sum_j w_j k_ij  -  sum_j w_j k_ij y_j )
    d S_xy / d y_i = -2 alpha w_i ( y_i * sum_a w_a k_ai  -  sum_a w_a k_ai x_a )

(the 4 vs 2 is because y appears in both arguments of k in S_yy and in one of
them in S_xy), and both `sum_j w_j k_ij` (R, N) and `sum_j w_j k_ij y_j`
(R, N, H) are reductions over j -- so the backward recomputes the kernel tiles
and folds them into those two accumulators, again keeping nothing.
Recomputation costs one extra pass over the pairs, which the dense path spends
anyway on the elementwise backward of its three `exp`s.

`x` (the real branch) is detached by the caller, so only `y` gets a gradient.

The two symmetric sums are reduced over block pairs I <= J only, doubling the
off-diagonal contribution, which halves their pair count in both directions.

`shard` splits the TILES across ranks. It is for the case where every rank holds
the same gathered features and so recomputes the same global kernel in full.
Sharding is sound because the tiles are summed: each rank reduces a disjoint
subset and one all-reduce of the three (R,) sums makes every rank's result the
complete one. The backward all-reduces its two accumulators for the same reason.
"""

import math

import torch
import torch.distributed as dist

import mmd_laplace_cuda


def _sq_norms(a):
    """||a_i||^2, (R, N) -- the a + b - 2ab form of the squared distance."""
    return (a * a).sum(-1)


def _kernel_block(a, b, a2, b2, alpha, self_tile=False):
    """exp(-alpha ||a_i - b_j||^2) for one (R, Bi, Bj) tile, in one buffer.

    Everything after the Gram matmul is in-place, so a tile allocates exactly
    one (R, Bi, Bj) tensor instead of the six the expression form would leave
    behind. Safe because this only ever runs with grad disabled (inside
    `Function.forward` and `Function.backward`).

    The clamp at 0 is the dense path's: `a + b - 2ab` is a difference of large
    numbers and rounding can push it below zero.

    `self_tile` marks a DIAGONAL tile of a symmetric kernel, where k_ii is
    analytically 1 and is pinned there. Computing the norms separately is more
    accurate off the diagonal, but it leaves d_ii at the matmul's error instead
    of at 0, and under TF32 that is enough to drop k_ii visibly below 1. The
    estimator subtracts the diagonal as exactly `n`, so that error would land in
    the loss undivided.
    """
    k = torch.bmm(a, b.transpose(1, 2))
    k.mul_(-2.0).add_(a2.unsqueeze(2)).add_(b2.unsqueeze(1))
    k.clamp_(min=0).mul_(-alpha).exp_()
    if self_tile:
        k.diagonal(dim1=1, dim2=2).fill_(1.0)
    return k


def _wKw(k, wi, wj):
    """(R,) = wi^T k wj, as two bmms -- the (R, Bi, Bj) weight matrix is never built."""
    return torch.bmm(wi.unsqueeze(1), torch.bmm(k, wj.unsqueeze(2))).reshape(-1)


def _blocks(n, block):
    return [(i, min(i + block, n)) for i in range(0, n, block)]


def pick_block_size(rows: int, n: int, budget_elems: int = 1 << 26) -> int:
    """Tile edge such that one (rows, block, block) tile fits the scratch budget.

    The default budget is 2^26 elements = 256 MB of fp32, which is the size of
    the ONE tile in flight (its transpose in the symmetric backward is a view).
    Returns `n` when the whole thing already fits, so small rows keep running as
    a single dense block with no loop overhead.
    """
    if rows * n * n <= budget_elems:
        return n
    block = int((budget_elems / max(rows, 1)) ** 0.5)
    return max(64, min(n, block))


def _shard_ctx(shard):
    """(rank, world) when tiles should be split, else None."""
    if not shard or not dist.is_available() or not dist.is_initialized():
        return None
    world = dist.get_world_size()
    return (dist.get_rank(), world) if world > 1 else None


class _RBFPairSums(torch.autograd.Function):
    """(S_xx, S_yy, S_xy) per row, tiled; gradient to `y` only."""

    @staticmethod
    def forward(ctx, x, y, w, alpha, block, shard, omit_real_real):
        x2, y2 = _sq_norms(x), _sq_norms(y)
        rows, n = w.shape
        if block <= 0:
            block = pick_block_size(rows, n)
        spans = _blocks(n, block)
        # Every rank walks the SAME tile list and skips the tiles it does not
        # own, so the collective below sees the same call on every rank whatever
        # the split works out to.
        part = _shard_ctx(shard)
        t = 0

        s_xx = x.new_zeros(rows)
        s_yy = x.new_zeros(rows)
        s_xy = x.new_zeros(rows)

        for bi, (i0, i1) in enumerate(spans):
            wi = w[:, i0:i1]
            for j0, j1 in spans[bi:]:               # symmetric: I <= J only
                t += 1
                if part and t % part[1] != part[0]:
                    continue
                wj = w[:, j0:j1]
                diag = j0 == i0
                if not omit_real_real:
                    k = _kernel_block(x[:, i0:i1], x[:, j0:j1], x2[:, i0:i1], x2[:, j0:j1],
                                      alpha, self_tile=diag)
                    s_xx += _wKw(k, wi, wj) * (1.0 if diag else 2.0)
                k = _kernel_block(y[:, i0:i1], y[:, j0:j1], y2[:, i0:i1], y2[:, j0:j1],
                                  alpha, self_tile=diag)
                s_yy += _wKw(k, wi, wj) * (1.0 if diag else 2.0)
            for j0, j1 in spans:                    # k(x_i, y_j) is not symmetric
                t += 1
                if part and t % part[1] != part[0]:
                    continue
                k = _kernel_block(x[:, i0:i1], y[:, j0:j1], x2[:, i0:i1], y2[:, j0:j1], alpha)
                s_xy += _wKw(k, wi, w[:, j0:j1])
        k = None

        if part:
            sums = torch.stack((s_xx, s_yy, s_xy))
            dist.all_reduce(sums)
            s_xx, s_yy, s_xy = sums.unbind(0)

        ctx.save_for_backward(x, y, w, x2, y2)
        ctx.alpha = alpha
        ctx.block = block
        ctx.shard = shard
        return s_xx, s_yy, s_xy

    @staticmethod
    def backward(ctx, g_xx, g_yy, g_xy):
        if not ctx.needs_input_grad[1]:
            return (None,) * 7
        x, y, w, x2, y2 = ctx.saved_tensors
        alpha, block = ctx.alpha, ctx.block
        rows, n = w.shape
        spans = _blocks(n, block)
        part = _shard_ctx(ctx.shard)
        t = 0

        # The gradient of both sums has the same shape,
        #     w_i * ( y_i * sum_p w_p k_ip  -  sum_p w_p k_ip z_p ),
        # with z = y over the k(y, y) pairs and z = x over the k(x, y) ones, so
        # the two only differ by their per-ROW coefficient. Folding that
        # coefficient in AT ACCUMULATION TIME lets both land in one pair of
        # accumulators instead of two, which is where the (R, N, H) buffers are.
        c_yy = (g_yy * (-4.0 * alpha)).reshape(rows, 1)      # y is in both args
        c_xy = (g_xy * (-2.0 * alpha)).reshape(rows, 1)      # ...and in one
        s_acc = torch.zeros_like(w)                          # (R, N)
        v_acc = torch.zeros_like(y)                          # (R, N, H)

        def accumulate(k, dst, src, wsrc, coef):
            """dst-block += coef * (k @ w_src, k @ (w_src * src))."""
            i0, i1 = dst
            p0, p1 = src
            wp = w[:, p0:p1]
            s_acc[:, i0:i1] += coef * torch.bmm(k, wp.unsqueeze(2)).squeeze(2)
            v_acc[:, i0:i1] += coef.unsqueeze(2) * torch.bmm(k, wp.unsqueeze(2) * wsrc[:, p0:p1])

        for bi, (i0, i1) in enumerate(spans):
            for j0, j1 in spans[bi:]:
                t += 1
                if part and t % part[1] != part[0]:
                    continue
                k = _kernel_block(y[:, i0:i1], y[:, j0:j1], y2[:, i0:i1], y2[:, j0:j1],
                                  alpha, self_tile=j0 == i0)
                accumulate(k, (i0, i1), (j0, j1), y, c_yy)
                if j0 != i0:                            # the mirrored tile, for free
                    accumulate(k.transpose(1, 2), (j0, j1), (i0, i1), y, c_yy)
            for a0, a1 in spans:                        # y_i against every x_a
                t += 1
                if part and t % part[1] != part[0]:
                    continue
                k = _kernel_block(y[:, i0:i1], x[:, a0:a1], y2[:, i0:i1], x2[:, a0:a1], alpha)
                accumulate(k, (i0, i1), (a0, a1), x, c_xy)
        k = None

        if part:
            # A tile of the symmetric kernel writes into BOTH its blocks, so a
            # rank's accumulators are partial for indices it never queried.
            # Summing them is what makes each rank's gradient the whole one.
            dist.all_reduce(s_acc)
            dist.all_reduce(v_acc)

        # grad = w * (y * s_acc - v_acc), in place so no (R, N, H) temporary.
        v_acc.addcmul_(y, s_acc.unsqueeze(-1), value=-1.0).neg_().mul_(w.unsqueeze(-1))
        return None, v_acc, None, None, None, None, None


def rbf_pair_sums(feat_real, feat_fake, w, alpha, block_size: int = 0, shard: bool = False,
                  distance: str = 'squared_euclidean', huber_delta: float = 0.001,
                  omit_real_real: bool = False):
    """Per row: (sum_ij w_i w_j k(x_i,x_j), ... k(y_i,y_j), ... k(x_i,y_j)).

    `alpha` = 1 / (2 sigma^2), a float or a 0-dim tensor; it is measured from the
    detached real branch, so it never needs a gradient. `block_size` 0 picks a
    tile from the scratch budget (see `pick_block_size`).

    `distance='pseudo_huber'` replaces squared Euclidean distance with
    SUM_d(sqrt((x_d-y_d)**2 + huber_delta**2) - huber_delta), using a finite
    positive `huber_delta`. Its coordinate scratch budget can further reduce
    an explicitly requested tile size. The default Gaussian path is unchanged.

    `shard` splits the tiles across ranks and all-reduces, which only makes sense
    when every rank holds the same samples. The result is the unsharded one on
    every rank. With `omit_real_real=True`, the first sum is zero and its
    kernel blocks are skipped; the fake-fake and real-fake sums are unchanged.
    """
    distance, huber_delta = validate_distance(distance, huber_delta)
    if distance == 'pseudo_huber':
        return _PseudoHuberPairSums.apply(
            feat_real, feat_fake, w, alpha, int(block_size), bool(shard), huber_delta,
            bool(omit_real_real))
    if torch.is_tensor(alpha):
        alpha = alpha.detach().to(dtype=feat_fake.dtype, device=feat_fake.device)
    return _RBFPairSums.apply(
        feat_real, feat_fake, w, alpha, int(block_size), bool(shard), bool(omit_real_real))


@torch.no_grad()
def rbf_mmd2_unequal(real, fake, w_real, w_fake, alpha,
                     block_size: int = 0, distance: str = 'squared_euclidean',
                     huber_delta: float = 0.001, omit_real_real: bool = False):
    """Unbiased MMD^2 between two differently sized feature sets, per row.

    Multi-draw rewards compare one clean sequence with K generated sequences,
    so the real and fake sets contain different numbers of token features. The
    ordinary ``rbf_pair_sums`` path cannot express that because it shares one
    mask between equal-sized sets.

        MMD^2 = (S_xx-n)/(n(n-1)) + (S_yy-m)/(m(m-1)) - 2 S_xy/(nm)

    The reward is detached and reaches the student only through REINFORCE, so
    this path needs no custom backward.
    `omit_real_real=True` skips the real-real sum and removes its entire
    normalized term. Fake/cross terms and the inactive-row rule are unchanged.
    """
    if real.ndim != 3 or fake.ndim != 3:
        raise ValueError('RBF feature sets must be rank-3, got '
                         f'{tuple(real.shape)} and {tuple(fake.shape)}')
    if real.shape[0] != fake.shape[0] or real.shape[2] != fake.shape[2]:
        raise ValueError('RBF feature sets must share row and feature dimensions, '
                         f'got {tuple(real.shape)} and {tuple(fake.shape)}')
    if w_real.shape != real.shape[:2] or w_fake.shape != fake.shape[:2]:
        raise ValueError('RBF masks must match their feature sets, got '
                         f'{tuple(w_real.shape)} and {tuple(w_fake.shape)}')

    distance, huber_delta = validate_distance(distance, huber_delta)
    if distance == 'pseudo_huber':
        return _pseudo_huber_mmd2_unequal(
            real, fake, w_real, w_fake, alpha, block_size, huber_delta, omit_real_real)

    real = real.detach().float()
    fake = fake.detach().float()
    w_real = w_real.to(real.dtype)
    w_fake = w_fake.to(fake.dtype)
    rows = real.shape[0]
    n_eff = w_real.sum(dim=1)
    m_eff = w_fake.sum(dim=1)
    active = (n_eff >= 2) & (m_eff >= 2)

    if torch.is_tensor(alpha):
        alpha = alpha.detach().to(dtype=real.dtype, device=real.device)
    r2, f2 = _sq_norms(real), _sq_norms(fake)
    block = (pick_block_size(rows, max(real.shape[1], fake.shape[1]))
             if block_size <= 0 else block_size)

    def _sum(a, b, a2, b2, wa, wb, symmetric):
        total = real.new_zeros(rows)
        for i0, i1 in _blocks(a.shape[1], block):
            for j0, j1 in _blocks(b.shape[1], block):
                kernel = _kernel_block(
                    a[:, i0:i1], b[:, j0:j1], a2[:, i0:i1], b2[:, j0:j1],
                    alpha, self_tile=symmetric and i0 == j0)
                total = total + _wKw(
                    kernel, wa[:, i0:i1], wb[:, j0:j1])
        return total

    s_xx = (real.new_zeros(rows) if omit_real_real else
            _sum(real, real, r2, r2, w_real, w_real, True))
    s_yy = _sum(fake, fake, f2, f2, w_fake, w_fake, True)
    s_xy = _sum(real, fake, r2, f2, w_real, w_fake, False)

    nn = (n_eff * (n_eff - 1)).clamp_min(1.0)
    mm = (m_eff * (m_eff - 1)).clamp_min(1.0)
    nm = (n_eff * m_eff).clamp_min(1.0)
    mmd2 = ((s_yy - m_eff) / mm - 2.0 * s_xy / nm if omit_real_real else
            (s_xx - n_eff) / nn + (s_yy - m_eff) / mm - 2.0 * s_xy / nm)
    return torch.where(active, mmd2, torch.zeros_like(mmd2)), active


@torch.no_grad()
def rbf_mmd2_fixed_real(real, fake, alpha, block_size: int = 0,
                       distance: str = 'squared_euclidean', huber_delta: float = 0.001,
                       omit_real_real: bool = False):
    """MMD^2 from one fixed real vector and K sampled fake vectors per row.

    ``real`` is ``(batch, hidden)`` and ``fake`` is
    ``(batch, draws, hidden)``.  The real vector is the observed conditional
    target, so ``k(real, real)`` is known exactly.  When K > 1 the fake-fake
    term is the unbiased U-statistic (its diagonal is removed):

        k(x,x) + sum_{i != j} k(y_i,y_j) / (K(K-1))
               - 2 sum_i k(x,y_i) / K

    With K = 1 there is no off-diagonal fake pair, so this deliberately falls
    back to the biased empirical estimate ``2 - 2 k(x,y)``.  The K > 1 result
    is not clamped: a U-statistic can legitimately be negative, and clamping it
    would make the requested estimator biased. `omit_real_real=True` removes
    only k(x,x)=1, retaining the fake self-term for the K=1 fallback.
    """
    if real.ndim != 2 or fake.ndim != 3:
        raise ValueError('fixed-real RBF expects (batch, hidden) real and '
                         '(batch, draws, hidden) fake features, got '
                         f'{tuple(real.shape)} and {tuple(fake.shape)}')
    if real.shape[0] != fake.shape[0] or real.shape[1] != fake.shape[2]:
        raise ValueError('fixed-real RBF features must share batch and hidden '
                         f'dimensions, got {tuple(real.shape)} and '
                         f'{tuple(fake.shape)}')
    draws = fake.shape[1]
    if draws < 1:
        raise ValueError('fixed-real RBF needs at least one fake draw')

    distance, huber_delta = validate_distance(distance, huber_delta)
    if distance == 'pseudo_huber':
        return _pseudo_huber_mmd2_fixed_real(
            real, fake, alpha, block_size, huber_delta, omit_real_real)

    real = real.detach().float()
    fake = fake.detach().float()
    if torch.is_tensor(alpha):
        alpha = alpha.detach().to(dtype=real.dtype, device=real.device)
    real2, fake2 = _sq_norms(real), _sq_norms(fake)
    rows = real.shape[0]
    block = (pick_block_size(rows, draws)
             if block_size <= 0 else block_size)

    fake_sum = real.new_zeros(rows)
    cross_sum = real.new_zeros(rows)
    for i0, i1 in _blocks(draws, block):
        for j0, j1 in _blocks(draws, block):
            kernel = _kernel_block(
                fake[:, i0:i1], fake[:, j0:j1],
                fake2[:, i0:i1], fake2[:, j0:j1], alpha,
                self_tile=i0 == j0)
            fake_sum += kernel.sum(dim=(1, 2))
        cross = _kernel_block(
            real.unsqueeze(1), fake[:, i0:i1], real2.unsqueeze(1),
            fake2[:, i0:i1], alpha)
        cross_sum += cross.sum(dim=(1, 2))

    if draws == 1:
        fake_term = fake_sum
    else:
        fake_term = (fake_sum - draws) / (draws * (draws - 1))
    return (fake_term - 2.0 * cross_sum / draws if omit_real_real else
            1.0 + fake_term - 2.0 * cross_sum / draws)



@torch.no_grad()
def rbf_mmd2_cross_sentence(real, fake, selected, alpha, block_size=0,
                            distance='squared_euclidean', huber_delta=0.001):
    """Cross-sentence token-kernel statistic with no reference self-term.

    real is (batch, length, hidden), fake is
    (batch, K, length, hidden), and selected is the shared token mask.
    For n selected positions, this returns

      sum_{r != s} sum_{i,j} k(y_ri, y_sj) / (K (K-1) n**2)
        - 2 sum_r sum_{i,j} k(x_i, y_rj) / (K n**2).

    All pairs of tokens within the same generated sentence are excluded;
    matching positions in different sentences are retained. The reference
    sentence's entire real-real term is deliberately omitted. This is a
    signed training statistic, not the complete nonnegative population MMD^2.
    The fake-fake term is unbiased over independent sentence draws conditional
    on the state and feature map, without assuming tokens within a sentence
    are independent. K >= 2 is required, and rows with n >= 1 are active.

    Sentence pairs and token tiles are traversed separately, so excluded
    within-sentence blocks are never computed or materialized.
    """
    distance, huber_delta = validate_distance(distance, huber_delta)
    if real.ndim != 3 or fake.ndim != 4:
        raise ValueError('cross-sentence RBF expects (batch, length, hidden) real '
                         'and (batch, draws, length, hidden) fake features, got '
                         f'{tuple(real.shape)} and {tuple(fake.shape)}')
    if (fake.shape[0] != real.shape[0]
            or fake.shape[2:] != real.shape[1:]):
        raise ValueError('cross-sentence RBF features must share batch, length '
                         f'and hidden dimensions, got {tuple(real.shape)} and '
                         f'{tuple(fake.shape)}')
    if selected.shape != real.shape[:2]:
        raise ValueError('cross-sentence RBF selected mask must match real '
                         f'features, got {tuple(selected.shape)} and '
                         f'{tuple(real.shape)}')
    draws = fake.shape[1]
    if draws < 2:
        raise ValueError('cross-sentence RBF requires at least 2 independent '
                         f'fake draws (mmd_kernel_batch_size >= 2), got {draws}')
    if real.device != fake.device:
        raise ValueError('cross-sentence RBF features must be on the same device')

    real, fake = real.detach().float(), fake.detach().float()
    weight = selected.to(dtype=real.dtype, device=real.device)
    if torch.is_tensor(alpha):
        alpha = alpha.detach().to(dtype=real.dtype, device=real.device)
    rows, length, hidden = real.shape
    if distance == 'pseudo_huber':
        block, chunk = _pseudo_huber_layout(
            rows, length, hidden, real.element_size(), int(block_size),
            fused=mmd_laplace_cuda.supports_pseudo_huber_cuda(real, huber_delta))
        real2 = fake2 = None
    else:
        block = (pick_block_size(rows, length)
                 if block_size <= 0 else int(block_size))
        block = max(1, block)
        real2, fake2 = _sq_norms(real), _sq_norms(fake)
    spans = _blocks(length, block)

    def pair_sum(left, right, left2, right2):
        total = real.new_zeros(rows)
        for i0, i1 in spans:
            for j0, j1 in spans:
                if distance == 'pseudo_huber':
                    kernel = _pseudo_huber_kernel_block(
                        left[:, i0:i1], right[:, j0:j1],
                        alpha, huber_delta, chunk)
                else:
                    kernel = _kernel_block(
                        left[:, i0:i1], right[:, j0:j1],
                        left2[:, i0:i1], right2[:, j0:j1], alpha)
                total.add_(_wKw(kernel, weight[:, i0:i1], weight[:, j0:j1]))
        return total

    fake_sum, cross_sum = real.new_zeros(rows), real.new_zeros(rows)
    with torch.autocast(device_type=real.device.type, enabled=False):
        for left in range(draws):
            left2 = None if fake2 is None else fake2[:, left]
            cross_sum.add_(pair_sum(real, fake[:, left], real2, left2))
            for right in range(left + 1, draws):
                right2 = None if fake2 is None else fake2[:, right]
                fake_sum.add_(
                    pair_sum(fake[:, left], fake[:, right], left2, right2),
                    alpha=2.0)
    count = weight.sum(dim=1)
    active = count >= 1
    nn = count.square()
    statistic = (fake_sum / (draws * (draws - 1) * nn).clamp_min(1.0)
                 - 2.0 * cross_sum / (draws * nn).clamp_min(1.0))
    return torch.where(active, statistic, torch.zeros_like(statistic)), active




def validate_distance(distance='squared_euclidean', huber_delta=0.001):
    """Validate the kernel distance; delta is used only by pseudo-Huber."""
    if distance not in ('squared_euclidean', 'pseudo_huber'):
        raise ValueError(
            "RBF distance must be 'squared_euclidean' or 'pseudo_huber', "
            f'got {distance!r}')
    if distance == 'pseudo_huber':
        try:
            delta = float(huber_delta)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('pseudo-Huber delta must be finite and positive') from exc
        if isinstance(huber_delta, bool) or not math.isfinite(delta) or delta <= 0:
            raise ValueError(
                f'pseudo-Huber delta must be finite and positive, got {huber_delta!r}')
        huber_delta = delta
    return distance, huber_delta


def _distance_inputs(real, fake):
    if real.ndim != 3 or fake.ndim != 3:
        raise ValueError('RBF feature sets must be rank-3, got '
                         f'{tuple(real.shape)} and {tuple(fake.shape)}')
    if real.shape[0] != fake.shape[0] or real.shape[2] != fake.shape[2]:
        raise ValueError('RBF feature sets must share row and feature dimensions, '
                         f'got {tuple(real.shape)} and {tuple(fake.shape)}')
    if real.device != fake.device:
        raise ValueError('RBF feature sets must be on the same device')
    dtype = (torch.float64 if torch.float64 in (real.dtype, fake.dtype)
             else torch.float32)
    return real.detach().to(dtype=dtype), fake.detach().to(dtype=dtype)


_PSEUDO_HUBER_SCRATCH_BYTES = 128 << 20


def _pseudo_huber_layout(rows, n, hidden, element_size, block_size, fused=False):
    """Cap all pair-coordinate scratch, even for an explicit oversized tile.

    Four coordinate buffers fit in 128 MiB. The distance/kernel and reduction
    buffers are smaller by at least the feature-chunk factor. Inputs, outputs
    and the full fake-feature gradient are separate linear-sized storage.
    Fused CUDA keeps coordinate intermediates in registers, so its pair tiles
    can use the same budget without the feature-chunk multiplier.
    """
    pair_budget = max(1, _PSEUDO_HUBER_SCRATCH_BYTES // (4 * element_size * max(rows, 1)))
    feature_chunk = max(1, min(hidden, 32, pair_budget))
    safe_edge = max(1, math.isqrt(max(1, pair_budget // (1 if fused else feature_chunk))))
    requested = n if block_size <= 0 else int(block_size)
    return max(1, min(max(n, 1), max(requested, 1), safe_edge)), feature_chunk


def _pseudo_huber_distance_block(a, b, delta, feature_chunk):
    """SUM sqrt(diff**2 + delta**2) - delta without near-zero cancellation."""
    if mmd_laplace_cuda.supports_pseudo_huber_cuda(a, delta):
        return mmd_laplace_cuda.pseudo_huber_pairwise_distance_cuda(a, b, delta)
    distance = a.new_zeros((a.shape[0], a.shape[1], b.shape[1]))
    delta = torch.as_tensor(delta, dtype=a.dtype, device=a.device)
    for h0, h1 in _blocks(a.shape[2], feature_chunk):
        diff = a[:, :, None, h0:h1] - b[:, None, :, h0:h1]
        denominator = torch.hypot(diff, delta).add_(delta)
        # diff**2 / (hypot(diff, delta) + delta), arranged to avoid overflow
        # when squaring a large finite difference.
        contribution = diff / denominator
        contribution.mul_(diff)
        # A positive Python delta can underflow to zero in the working dtype.
        # Equal coordinates still have exactly zero distance.
        contribution.masked_fill_(diff == 0, 0.0)
        distance.add_(contribution.sum(dim=-1))
    return distance


def _pseudo_huber_kernel_block(a, b, alpha, delta, feature_chunk, self_tile=False):
    kernel = _pseudo_huber_distance_block(a, b, delta, feature_chunk)
    kernel.mul_(-alpha).exp_()
    if self_tile:
        kernel.diagonal(dim1=1, dim2=2).fill_(1.0)
    return kernel


@torch.no_grad()
def pairwise_distance(real, fake, distance='squared_euclidean', huber_delta=0.001,
                      block_size=0):
    """Detached (rows, n, m) distances for two rank-3 feature sets.

    Pseudo-Huber uses SUM_d(sqrt((x_d-y_d)**2 + delta**2) - delta).
    Half/BF16 features are promoted to FP32; FP64 inputs retain FP64. Pair and
    coordinate tiles bound intermediate storage; the returned matrix is dense.
    """
    distance, huber_delta = validate_distance(distance, huber_delta)
    real, fake = _distance_inputs(real, fake)
    rows, n, hidden = real.shape
    m = fake.shape[1]
    output = real.new_empty((rows, n, m))
    if distance == 'pseudo_huber':
        block, chunk = _pseudo_huber_layout(
            rows, max(n, m), hidden, real.element_size(), int(block_size),
            fused=mmd_laplace_cuda.supports_pseudo_huber_cuda(real, huber_delta))
    else:
        block = (pick_block_size(rows, max(n, m))
                 if block_size <= 0 else int(block_size))
        block = max(1, block)
        r2, f2 = _sq_norms(real), _sq_norms(fake)
    # Do not let surrounding mixed precision demote the squared-distance bmm.
    with torch.autocast(device_type=real.device.type, enabled=False):
        for i0, i1 in _blocks(n, block):
            for j0, j1 in _blocks(m, block):
                if distance == 'pseudo_huber':
                    tile = _pseudo_huber_distance_block(
                        real[:, i0:i1], fake[:, j0:j1], huber_delta, chunk)
                else:
                    tile = torch.bmm(real[:, i0:i1], fake[:, j0:j1].transpose(1, 2))
                    tile.mul_(-2.0).add_(r2[:, i0:i1, None]).add_(f2[:, None, j0:j1])
                    tile.clamp_(min=0)
                output[:, i0:i1, j0:j1] = tile
    return output


def _pseudo_huber_sums(real, fake, w_real, w_fake, alpha, block, delta, chunk,
                       shard=False, omit_real_real=False):
    """Blocked weighted sums, supporting unequal set sizes and tile sharding."""
    rows = real.shape[0]
    part = _shard_ctx(shard)
    tile_index = 0
    sums = []
    for term, (a, b, wa, wb, symmetric) in enumerate((
            (real, real, w_real, w_real, True),
            (fake, fake, w_fake, w_fake, True),
            (real, fake, w_real, w_fake, False))):
        total = real.new_zeros(rows)
        spans_b = _blocks(b.shape[1], block)
        if term == 0 and omit_real_real:
            # Preserve the fake/cross tile assignments when sharding is used,
            # while skipping every real-real kernel evaluation.
            tile_index += len(spans_b) * (len(spans_b) + 1) // 2
            sums.append(total)
            continue
        for bi, (i0, i1) in enumerate(_blocks(a.shape[1], block)):
            for j0, j1 in (spans_b[bi:] if symmetric else spans_b):
                tile_index += 1
                if part and tile_index % part[1] != part[0]:
                    continue
                diagonal = symmetric and i0 == j0
                kernel = _pseudo_huber_kernel_block(
                    a[:, i0:i1], b[:, j0:j1], alpha, delta, chunk, diagonal)
                # Elementwise reduction avoids autocast changing FP32 to BF16.
                kernel.mul_(wa[:, i0:i1, None]).mul_(wb[:, None, j0:j1])
                total.add_(kernel.sum(dim=(1, 2)),
                           alpha=2.0 if symmetric and not diagonal else 1.0)
        sums.append(total)
    if part:
        reduced = torch.stack(sums)
        dist.all_reduce(reduced)
        return tuple(reduced.unbind(0))
    return tuple(sums)


class _PseudoHuberPairSums(torch.autograd.Function):
    """Recomputed pseudo-Huber kernels with an exact fake-only first derivative."""

    @staticmethod
    def forward(ctx, real, fake, w, alpha, block_size, shard, delta, omit_real_real):
        x, y = _distance_inputs(real, fake)
        if x.shape != y.shape or w.shape != x.shape[:2]:
            raise ValueError('paired RBF features and their shared mask must have matching shapes')
        weights = w.detach().to(dtype=x.dtype, device=x.device)
        if torch.is_tensor(alpha):
            alpha = alpha.detach().to(dtype=x.dtype, device=x.device)
        block, chunk = _pseudo_huber_layout(
            x.shape[0], x.shape[1], x.shape[2], x.element_size(), block_size,
            fused=mmd_laplace_cuda.supports_pseudo_huber_cuda(x, delta))
        sums = _pseudo_huber_sums(
            x, y, weights, weights, alpha, block, delta, chunk, shard, omit_real_real)
        ctx.save_for_backward(x, y, weights)
        ctx.alpha, ctx.block, ctx.chunk = alpha, block, chunk
        ctx.delta, ctx.shard, ctx.fake_dtype = delta, shard, fake.dtype
        return sums

    @staticmethod
    @torch.no_grad()
    def backward(ctx, g_xx, g_yy, g_xy):
        if not ctx.needs_input_grad[1]:
            return (None,) * 8
        x, y, w = ctx.saved_tensors
        rows, n, hidden = y.shape
        grad = torch.zeros_like(y)
        part = _shard_ctx(ctx.shard)
        tile_index = 0
        delta = torch.as_tensor(ctx.delta, dtype=y.dtype, device=y.device)
        # y appears twice in S_yy, once in S_xy. The coordinate derivative
        # of the distance is diff/hypot(diff, delta), not 2*diff.
        for other, upstream, factor in ((y, g_yy, -2.0), (x, g_xy, -1.0)):
            coefficient = (upstream * (factor * ctx.alpha)).reshape(rows, 1, 1)
            for i0, i1 in _blocks(n, ctx.block):
                for j0, j1 in _blocks(n, ctx.block):
                    tile_index += 1
                    if part and tile_index % part[1] != part[0]:
                        continue
                    kernel = _pseudo_huber_kernel_block(
                        y[:, i0:i1], other[:, j0:j1], ctx.alpha,
                        ctx.delta, ctx.chunk, other is y and i0 == j0)
                    kernel.mul_(w[:, i0:i1, None]).mul_(w[:, None, j0:j1])
                    kernel.mul_(coefficient)
                    if mmd_laplace_cuda.supports_pseudo_huber_cuda(y, ctx.delta):
                        grad[:, i0:i1].add_(mmd_laplace_cuda.pseudo_huber_weighted_grad_sum(
                            y[:, i0:i1], other[:, j0:j1], kernel, ctx.delta))
                        continue
                    for h0, h1 in _blocks(hidden, ctx.chunk):
                        diff = y[:, i0:i1, None, h0:h1] - other[:, None, j0:j1, h0:h1]
                        denominator = torch.hypot(diff, delta)
                        diff.div_(denominator).masked_fill_(denominator == 0, 0.0)
                        diff.mul_(kernel.unsqueeze(-1))
                        grad[:, i0:i1, h0:h1].add_(diff.sum(dim=2))
                        # Release coordinate scratch before recomputing another kernel.
                        del diff, denominator
        if part:
            dist.all_reduce(grad)
        return None, grad.to(dtype=ctx.fake_dtype), None, None, None, None, None, None


def _pseudo_huber_mmd2_unequal(real, fake, w_real, w_fake, alpha, block_size, delta,
                              omit_real_real=False):
    real, fake = _distance_inputs(real, fake)
    if w_real.shape != real.shape[:2] or w_fake.shape != fake.shape[:2]:
        raise ValueError('RBF masks must match their feature sets')
    # Match the reward estimators' existing FP32 precision.
    real, fake = real.float(), fake.float()
    w_real = w_real.to(dtype=real.dtype, device=real.device)
    w_fake = w_fake.to(dtype=fake.dtype, device=fake.device)
    if torch.is_tensor(alpha):
        alpha = alpha.detach().to(dtype=real.dtype, device=real.device)
    block, chunk = _pseudo_huber_layout(
        real.shape[0], max(real.shape[1], fake.shape[1]), real.shape[2],
        real.element_size(), int(block_size),
        fused=mmd_laplace_cuda.supports_pseudo_huber_cuda(real, delta))
    s_xx, s_yy, s_xy = _pseudo_huber_sums(
        real, fake, w_real, w_fake, alpha, block, delta, chunk,
        omit_real_real=omit_real_real)
    n, m = w_real.sum(dim=1), w_fake.sum(dim=1)
    active = (n >= 2) & (m >= 2)
    if omit_real_real:
        result = ((s_yy - m) / (m * (m - 1)).clamp_min(1.0)
                  - 2.0 * s_xy / (n * m).clamp_min(1.0))
    else:
        result = ((s_xx - n) / (n * (n - 1)).clamp_min(1.0)
                  + (s_yy - m) / (m * (m - 1)).clamp_min(1.0)
                  - 2.0 * s_xy / (n * m).clamp_min(1.0))
    return torch.where(active, result, torch.zeros_like(result)), active


def _pseudo_huber_mmd2_fixed_real(real, fake, alpha, block_size, delta,
                                 omit_real_real=False):
    real, fake = real.detach().float(), fake.detach().float()
    if torch.is_tensor(alpha):
        alpha = alpha.detach().to(dtype=real.dtype, device=real.device)
    rows, draws, hidden = fake.shape
    block, chunk = _pseudo_huber_layout(
        rows, draws, hidden, fake.element_size(), int(block_size),
        fused=mmd_laplace_cuda.supports_pseudo_huber_cuda(fake, delta))
    # A singleton real set has known self-kernel 1; do not use the unequal
    # estimator's inactive-row rule for this conditional target.
    ones_real = real.new_ones((rows, 1))
    ones_fake = fake.new_ones((rows, draws))
    _, fake_sum, cross_sum = _pseudo_huber_sums(
        real.unsqueeze(1), fake, ones_real, ones_fake, alpha, block, delta, chunk,
        omit_real_real=omit_real_real)
    fake_term = fake_sum if draws == 1 else (fake_sum - draws) / (draws * (draws - 1))
    return (fake_term - 2.0 * cross_sum / draws if omit_real_real else
            1.0 + fake_term - 2.0 * cross_sum / draws)
