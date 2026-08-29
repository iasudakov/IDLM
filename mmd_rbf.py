"""Blocked Gaussian-kernel MMD: the same estimator, without the (R, N, N) matrix.

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

import torch
import torch.distributed as dist


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
    def forward(ctx, x, y, w, alpha, block, shard):
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
            return None, None, None, None, None, None
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
        return None, v_acc, None, None, None, None


def rbf_pair_sums(feat_real, feat_fake, w, alpha, block_size: int = 0, shard: bool = False):
    """Per row: (sum_ij w_i w_j k(x_i,x_j), ... k(y_i,y_j), ... k(x_i,y_j)).

    `alpha` = 1 / (2 sigma^2), a float or a 0-dim tensor; it is measured from the
    detached real branch, so it never needs a gradient. `block_size` 0 picks a
    tile from the scratch budget (see `pick_block_size`).

    `shard` splits the tiles across ranks and all-reduces, which only makes sense
    when every rank holds the same samples. The result is the unsharded one on
    every rank.
    """
    if torch.is_tensor(alpha):
        alpha = alpha.detach().to(dtype=feat_fake.dtype, device=feat_fake.device)
    return _RBFPairSums.apply(feat_real, feat_fake, w, alpha, int(block_size), bool(shard))
