"""Optional fused CUDA pair distances and MMD gradient reductions.

The reduction computes ``sum_j k[r, i, j] * sign(a[r, i, h] - b[r, j, h])``
without materializing a pair-by-hidden-dimension tensor in device memory.
Pseudo-Huber helpers use the same tiled reductions for smooth L1 distances.
Triton is loaded only on the CUDA path; importing this module on a CPU-only
installation does not require Triton. Callers retain their portable fallback.
"""

from functools import lru_cache
import math

import torch


@lru_cache(maxsize=1)
def _triton_available():
    try:
        import triton
        import triton.language
    except (ImportError, OSError):
        # A package can be present but unusable (e.g. a missing shared library).
        # Do not hide kernel compilation/runtime failures after a valid import.
        return False
    return True


def supports_laplace_cuda(tensor):
    """Whether the optional float32 CUDA kernels can handle this tensor."""
    return (
        tensor.device.type == "cuda"
        and tensor.dtype == torch.float32
        and _triton_available()
    )


@lru_cache(maxsize=1)
def _load_kernel():
    # Keep language symbols in this module's globals for Triton's JIT source
    # resolver while deferring the optional imports until an actual CUDA call.
    global triton, tl
    import triton
    import triton.language as tl

    # Compaction changes pair lengths and parent row strides every batch. Keep
    # them runtime values and disable integer/alignment specialization, so an
    # arbitrary new tail does not trigger another compilation during training.
    @triton.jit(do_not_specialize=["I", "J", "A_R", "B_R", "K_R", "K_I", "K_J"])
    def weighted_sign_kernel(
        a_ptr, b_ptr, k_ptr, out_ptr,
        I, J, H: tl.constexpr,
        A_R, A_I: tl.constexpr, A_H: tl.constexpr,
        B_R, B_J: tl.constexpr, B_H: tl.constexpr,
        K_R, K_I, K_J,
        K_LAYOUT: tl.constexpr,
        BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        # Nearby programs share a row: its feature tile can remain in L2.
        row = tl.program_id(2)
        ii = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
        hh = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        jj = tl.arange(0, BLOCK_J)
        a = tl.load(
            a_ptr + row * A_R + ii[:, None] * A_I + hh[None, :] * A_H,
            mask=(ii[:, None] < I) & (hh[None, :] < H), other=0.0,
        )
        result = tl.full((BLOCK_I, BLOCK_H), 0.0, tl.float32)
        for start in range(tl.cdiv(J, BLOCK_J)):
            j = start * BLOCK_J + jj
            b = tl.load(
                b_ptr + row * B_R + j[:, None] * B_J + hh[None, :] * B_H,
                mask=(j[:, None] < J) & (hh[None, :] < H), other=0.0,
            )
            if K_LAYOUT == 1:
                k_offsets = ii[:, None] * K_I + j[None, :]
            elif K_LAYOUT == 2:
                k_offsets = ii[:, None] + j[None, :] * K_J
            else:
                k_offsets = ii[:, None] * K_I + j[None, :] * K_J
            k = tl.load(
                k_ptr + row * K_R + k_offsets,
                mask=(ii[:, None] < I) & (j[None, :] < J), other=0.0,
            )
            delta = a[:, None, :] - b[None, :, :]
            # torch.sign chooses the zero subgradient at equal coordinates.
            sign = tl.where(delta > 0, 1.0, tl.where(delta < 0, -1.0, 0.0))
            result += tl.sum(k[:, :, None] * sign, axis=1)
        tl.store(
            out_ptr + (row * I + ii[:, None]) * H + hh[None, :],
            result, mask=(ii[:, None] < I) & (hh[None, :] < H),
        )

    return weighted_sign_kernel


@lru_cache(maxsize=1)
def _load_distance_kernel():
    _load_kernel()  # Populate the lazily imported JIT symbols.

    @triton.jit(do_not_specialize=["I", "J", "A_R", "B_R"])
    def distance_kernel(
        a_ptr, b_ptr, out_ptr,
        I, J, H: tl.constexpr,
        A_R, A_I: tl.constexpr, A_H: tl.constexpr,
        B_R, B_J: tl.constexpr, B_H: tl.constexpr,
        BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(2)
        ii = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
        jj = tl.program_id(1) * BLOCK_J + tl.arange(0, BLOCK_J)
        hh = tl.arange(0, BLOCK_H)
        result = tl.full((BLOCK_I, BLOCK_J), 0.0, tl.float32)
        for start in range(tl.cdiv(H, BLOCK_H)):
            h = start * BLOCK_H + hh
            a = tl.load(
                a_ptr + row * A_R + ii[:, None] * A_I + h[None, :] * A_H,
                mask=(ii[:, None] < I) & (h[None, :] < H), other=0.0,
            )
            b = tl.load(
                b_ptr + row * B_R + jj[:, None] * B_J + h[None, :] * B_H,
                mask=(jj[:, None] < J) & (h[None, :] < H), other=0.0,
            )
            result += tl.sum(tl.abs(a[:, None, :] - b[None, :, :]), axis=2)
        tl.store(
            out_ptr + (row * I + ii[:, None]) * J + jj[None, :],
            result, mask=(ii[:, None] < I) & (jj[None, :] < J),
        )

    return distance_kernel


def l1_pairwise_distance_cuda(a, b):
    """Compute an L1 pair tile with feature reuse, without autograd."""
    if not supports_laplace_cuda(a) or b.device != a.device or b.dtype != a.dtype:
        raise ValueError("The fused Laplace distance requires float32 CUDA tensors and Triton")
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
        raise ValueError("Incompatible Laplace distance input shapes")
    rows, count_i, hidden = a.shape
    count_j = b.shape[1]
    out = torch.empty((rows, count_i, count_j), dtype=a.dtype, device=a.device)
    if not out.numel():
        return out
    if not hidden:
        return out.zero_()
    kernel = _load_distance_kernel()
    # Reduce a complete feature vector when practical; only the small pair
    # tile is held in registers, rather than a global (R, I, J, H) tensor.
    block_i, block_j, block_h = 4, 4, triton.next_power_of_2(min(hidden, 1024))
    with torch.cuda.device(a.device):
        kernel[(triton.cdiv(count_i, block_i), triton.cdiv(count_j, block_j), rows)](
            a, b, out, count_i, count_j, hidden,
            *a.stride(), *b.stride(), block_i, block_j, block_h,
            num_warps=4,
        )
    return out


def l1_weighted_sign_sum(a, b, k):
    """Reduce weighted L1 coordinate derivatives on CUDA, without autograd.

    Inputs have shapes ``(R, I, H)``, ``(R, J, H)``, and ``(R, I, J)``;
    noncontiguous slices and transposed pair tiles are supported. All inputs
    must be float32 on the same CUDA device. Returns a contiguous ``(R, I, H)``
    tensor. The caller supplies any kernel weights and gradient coefficients
    through ``k`` and is responsible for differentiating the enclosing loss.
    """
    if not supports_laplace_cuda(a):
        raise ValueError("The fused Laplace reduction requires float32 CUDA tensors and Triton")
    if any(t.device != a.device or t.dtype != a.dtype for t in (b, k)):
        raise ValueError("Laplace reduction inputs must share a CUDA device and float32 dtype")
    if a.ndim != 3 or b.ndim != 3 or k.ndim != 3:
        raise ValueError("Laplace reduction inputs must have three dimensions")
    rows, count_i, hidden = a.shape
    if b.shape[0] != rows or b.shape[2] != hidden or k.shape != (rows, count_i, b.shape[1]):
        raise ValueError("Incompatible Laplace reduction input shapes")
    out = torch.empty((rows, count_i, hidden), dtype=a.dtype, device=a.device)
    if not out.numel():
        return out
    if b.shape[1] == 0:
        return out.zero_()
    kernel = _load_kernel()
    block_i, block_j, block_h = 4, 64, 128
    # Only three fixed layout variants, never a distinct variant for each tail.
    k_layout = 1 if k.stride(2) == 1 else (2 if k.stride(1) == 1 else 0)
    with torch.cuda.device(a.device):
        kernel[(triton.cdiv(count_i, block_i), triton.cdiv(hidden, block_h), rows)](
            a, b, k, out,
            count_i, b.shape[1], hidden,
            *a.stride(), *b.stride(), *k.stride(),
            k_layout,
            block_i, block_j, block_h,
            num_warps=4,
        )
    return out


def supports_pseudo_huber_cuda(tensor, delta=None):
    """Whether the optional float32 CUDA pseudo-Huber kernels are available."""
    if delta is not None:
        try:
            value = float(delta)
        except (TypeError, ValueError):
            return False
        if (
            isinstance(delta, bool)
            or not math.isfinite(value)
            or value <= 0.0
            or value > torch.finfo(torch.float32).max
        ):
            return False
    return supports_laplace_cuda(tensor)


@lru_cache(maxsize=1)
def _load_pseudo_huber_kernels():
    _load_kernel()  # Populate the lazily imported JIT symbols.
    # Older Triton versions resolve called JIT functions through globals only.
    global _pseudo_huber_coordinate_distance, _pseudo_huber_coordinate_grad

    @triton.jit
    def _pseudo_huber_coordinate_distance(diff, delta):
        absolute = tl.abs(diff)
        scale = tl.maximum(absolute, delta)
        # Delta may round to zero in float32. At equal coordinates the distance
        # and its derivative must still be exactly zero rather than 0 / 0.
        safe_scale = tl.where(scale > 0.0, scale, 1.0)
        scaled_diff = absolute / safe_scale
        scaled_delta = delta / safe_scale
        root = tl.sqrt(scaled_diff * scaled_diff + scaled_delta * scaled_delta)
        denominator = root + scaled_delta
        denominator = tl.where(denominator > 0.0, denominator, 1.0)
        # Rationalize hypot(diff, delta) - delta to retain small distances;
        # scale before squaring so large finite float32 inputs do not overflow.
        distance = absolute * (scaled_diff / denominator)
        # Opposite, finite extreme coordinates can overflow their subtraction.
        # Their distance legitimately overflows, but must not become NaN.
        return tl.where(absolute == float("inf"), float("inf"), distance)

    @triton.jit
    def _pseudo_huber_coordinate_grad(diff, delta):
        absolute = tl.abs(diff)
        scale = tl.maximum(absolute, delta)
        safe_scale = tl.where(scale > 0.0, scale, 1.0)
        scaled_diff = diff / safe_scale
        scaled_delta = delta / safe_scale
        root = tl.sqrt(scaled_diff * scaled_diff + scaled_delta * scaled_delta)
        root = tl.where(root > 0.0, root, 1.0)
        grad = scaled_diff / root
        overflow_sign = tl.where(diff > 0.0, 1.0, -1.0)
        return tl.where(absolute == float("inf"), overflow_sign, grad)

    @triton.jit(do_not_specialize=["I", "J", "A_R", "B_R", "DELTA"])
    def distance_kernel(
        a_ptr, b_ptr, out_ptr,
        I, J, H: tl.constexpr, DELTA,
        A_R, A_I: tl.constexpr, A_H: tl.constexpr,
        B_R, B_J: tl.constexpr, B_H: tl.constexpr,
        BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(2)
        ii = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
        jj = tl.program_id(1) * BLOCK_J + tl.arange(0, BLOCK_J)
        hh = tl.arange(0, BLOCK_H)
        result = tl.full((BLOCK_I, BLOCK_J), 0.0, tl.float32)
        for start in range(tl.cdiv(H, BLOCK_H)):
            h = start * BLOCK_H + hh
            a = tl.load(
                a_ptr + row * A_R + ii[:, None] * A_I + h[None, :] * A_H,
                mask=(ii[:, None] < I) & (h[None, :] < H), other=0.0,
            )
            b = tl.load(
                b_ptr + row * B_R + jj[:, None] * B_J + h[None, :] * B_H,
                mask=(jj[:, None] < J) & (h[None, :] < H), other=0.0,
            )
            distance = _pseudo_huber_coordinate_distance(a[:, None, :] - b[None, :, :], DELTA)
            result += tl.sum(distance, axis=2)
        tl.store(
            out_ptr + (row * I + ii[:, None]) * J + jj[None, :],
            result, mask=(ii[:, None] < I) & (jj[None, :] < J),
        )

    @triton.jit(do_not_specialize=["I", "J", "A_R", "B_R", "K_R", "K_I", "K_J", "DELTA"])
    def weighted_grad_kernel(
        a_ptr, b_ptr, k_ptr, out_ptr,
        I, J, H: tl.constexpr, DELTA,
        A_R, A_I: tl.constexpr, A_H: tl.constexpr,
        B_R, B_J: tl.constexpr, B_H: tl.constexpr,
        K_R, K_I, K_J,
        K_LAYOUT: tl.constexpr,
        BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(2)
        ii = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
        hh = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        jj = tl.arange(0, BLOCK_J)
        a = tl.load(
            a_ptr + row * A_R + ii[:, None] * A_I + hh[None, :] * A_H,
            mask=(ii[:, None] < I) & (hh[None, :] < H), other=0.0,
        )
        result = tl.full((BLOCK_I, BLOCK_H), 0.0, tl.float32)
        for start in range(tl.cdiv(J, BLOCK_J)):
            j = start * BLOCK_J + jj
            b = tl.load(
                b_ptr + row * B_R + j[:, None] * B_J + hh[None, :] * B_H,
                mask=(j[:, None] < J) & (hh[None, :] < H), other=0.0,
            )
            if K_LAYOUT == 1:
                k_offsets = ii[:, None] * K_I + j[None, :]
            elif K_LAYOUT == 2:
                k_offsets = ii[:, None] + j[None, :] * K_J
            else:
                k_offsets = ii[:, None] * K_I + j[None, :] * K_J
            k = tl.load(
                k_ptr + row * K_R + k_offsets,
                mask=(ii[:, None] < I) & (j[None, :] < J), other=0.0,
            )
            grad = _pseudo_huber_coordinate_grad(a[:, None, :] - b[None, :, :], DELTA)
            result += tl.sum(k[:, :, None] * grad, axis=1)
        tl.store(
            out_ptr + (row * I + ii[:, None]) * H + hh[None, :],
            result, mask=(ii[:, None] < I) & (hh[None, :] < H),
        )

    return distance_kernel, weighted_grad_kernel


def _validate_pseudo_huber_delta(delta):
    try:
        value = float(delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pseudo-Huber delta must be finite and positive") from exc
    if isinstance(delta, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError("Pseudo-Huber delta must be finite and positive")
    if value > torch.finfo(torch.float32).max:
        raise ValueError("The fused pseudo-Huber delta must fit in float32")
    return value


def pseudo_huber_pairwise_distance_cuda(a, b, delta):
    """Return ``sum_h (hypot(a-b, delta) - delta)`` without autograd.

    Inputs are float32 CUDA tensors of shape ``(R, I, H)`` and ``(R, J, H)``.
    Strided slices and empty dimensions are supported. Only a small feature
    tile lives in registers; no global ``(R, I, J, H)`` scratch is allocated.
    """
    delta = _validate_pseudo_huber_delta(delta)
    if not supports_pseudo_huber_cuda(a) or b.device != a.device or b.dtype != a.dtype:
        raise ValueError("The fused pseudo-Huber distance requires float32 CUDA tensors and Triton")
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
        raise ValueError("Incompatible pseudo-Huber distance input shapes")
    rows, count_i, hidden = a.shape
    count_j = b.shape[1]
    out = torch.empty((rows, count_i, count_j), dtype=a.dtype, device=a.device)
    if not out.numel():
        return out
    if not hidden:
        return out.zero_()
    kernel, _ = _load_pseudo_huber_kernels()
    block_i, block_j, block_h = 4, 4, triton.next_power_of_2(min(hidden, 1024))
    with torch.cuda.device(a.device):
        kernel[(triton.cdiv(count_i, block_i), triton.cdiv(count_j, block_j), rows)](
            a, b, out, count_i, count_j, hidden, delta,
            *a.stride(), *b.stride(), block_i, block_j, block_h,
            num_warps=4,
        )
    return out


def pseudo_huber_weighted_grad_sum(a, b, k, delta):
    """Return ``sum_j k * (a-b) / hypot(a-b, delta)`` without autograd.

    Shapes are ``(R, I, H)``, ``(R, J, H)`` and ``(R, I, J)``. All inputs
    must share a CUDA device and float32 dtype; arbitrary strides are valid.
    The output is contiguous with shape ``(R, I, H)``. The enclosing loss
    supplies kernel weights and gradient coefficients through ``k``.
    """
    delta = _validate_pseudo_huber_delta(delta)
    if not supports_pseudo_huber_cuda(a):
        raise ValueError("The fused pseudo-Huber reduction requires float32 CUDA tensors and Triton")
    if any(t.device != a.device or t.dtype != a.dtype for t in (b, k)):
        raise ValueError("Pseudo-Huber reduction inputs must share a CUDA device and float32 dtype")
    if a.ndim != 3 or b.ndim != 3 or k.ndim != 3:
        raise ValueError("Pseudo-Huber reduction inputs must have three dimensions")
    rows, count_i, hidden = a.shape
    if b.shape[0] != rows or b.shape[2] != hidden or k.shape != (rows, count_i, b.shape[1]):
        raise ValueError("Incompatible pseudo-Huber reduction input shapes")
    out = torch.empty((rows, count_i, hidden), dtype=a.dtype, device=a.device)
    if not out.numel():
        return out
    if b.shape[1] == 0:
        return out.zero_()
    _, kernel = _load_pseudo_huber_kernels()
    block_i, block_j, block_h = 4, 64, 128
    k_layout = 1 if k.stride(2) == 1 else (2 if k.stride(1) == 1 else 0)
    with torch.cuda.device(a.device):
        kernel[(triton.cdiv(count_i, block_i), triton.cdiv(hidden, block_h), rows)](
            a, b, k, out, count_i, b.shape[1], hidden, delta,
            *a.stride(), *b.stride(), *k.stride(), k_layout,
            block_i, block_j, block_h, num_warps=4,
        )
    return out
