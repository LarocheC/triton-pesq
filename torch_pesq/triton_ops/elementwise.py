"""Small fused element-wise stages of the PESQ pipeline as Triton kernels.

This module holds the four "glue" operators of the pipeline. Each one is a
:class:`torch.autograd.Function` whose forward *and* backward pass are Triton
kernels, exposed through a thin function wrapper:

``peak_normalize``
    Divide reference and degraded signal by their common peak amplitude.
``edge_ramp``
    Fade the first and last 15 samples, the edge treatment which precedes the
    pre-emphasize filter.
``align_scale``
    Scale a signal such that the power of a band pass filtered copy hits the
    ``1e7`` target of the PESQ reference implementation.
``pesq_epilogue``
    Turn the symmetric and asymmetric distances into a MOS estimate and the
    loss value.

All reductions are deterministic two-stage tree reductions -- no atomics -- so
repeated calls with identical inputs return bitwise identical results.
"""

import torch

from ._common import HAS_TRITON, check_input

if HAS_TRITON:  # pragma: no branch - Triton is a hard requirement at runtime
    import triton
    import triton.language as tl
else:  # pragma: no cover - keeps the module importable without Triton
    import types

    triton = types.SimpleNamespace(jit=lambda fn: fn, cdiv=lambda a, b: -(-a // b))
    tl = types.SimpleNamespace(constexpr=None)


__all__ = ["peak_normalize", "edge_ramp", "align_scale", "pesq_epilogue"]


#: Number of elements processed by a single program iteration.
_BLOCK = 1024

#: Number of partial results of the first reduction stage. Fixed (and a power
#: of two) so that the reduction tree only depends on the tensor shape, which
#: makes the result run-to-run deterministic.
_NCHUNK = 64

#: Block size of the small per batch element kernels.
_BLOCK_B = 128

#: Largest supported row length. The batch stride is computed in 64 bit, but
#: the offsets *inside* a row are 32 bit (``tl.arange`` and ``tl.program_id``
#: are ``int32``), so ``pid * BLOCK`` would silently wrap beyond this. No audio
#: signal comes anywhere near it; the guard just turns a wrong answer into an
#: error.
_MAX_ROW = 2**31 - _BLOCK


def _check_row_length(n, op):
    """Reject row lengths for which the 32 bit offset arithmetic would wrap."""

    if n > _MAX_ROW:
        raise RuntimeError(
            f"{op} supports at most {_MAX_ROW} samples per row, got {n}; the "
            "offsets inside a row are 32 bit."
        )


# ---------------------------------------------------------------------------
# peak normalisation
# ---------------------------------------------------------------------------


@triton.jit
def _peak_partial_kernel(
    ref_ptr, deg_ptr, part_ptr, n, BLOCK: tl.constexpr, NCHUNK: tl.constexpr
):
    """First stage: per chunk maximum of ``max(|ref|, |deg|)``."""

    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    base = pid_b.to(tl.int64) * n
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    num_blocks = tl.cdiv(n, BLOCK)
    for blk in range(pid_c, num_blocks, NCHUNK):
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        ref = tl.load(ref_ptr + base + offs, mask=mask, other=0.0)
        deg = tl.load(deg_ptr + base + offs, mask=mask, other=0.0)
        acc = tl.maximum(acc, tl.maximum(tl.abs(ref), tl.abs(deg)))

    tl.store(part_ptr + pid_b * NCHUNK + pid_c, tl.max(acc, axis=0))


@triton.jit
def _peak_apply_kernel(
    part_ptr,
    ref_ptr,
    deg_ptr,
    oref_ptr,
    odeg_ptr,
    max_ptr,
    n,
    NCHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Second stage plus the division, fused to save a kernel launch.

    Every program repeats the (cheap, ``NCHUNK`` wide) second reduction stage
    instead of waiting for a separate kernel; the first program along the
    sample axis also writes the peak amplitude out for the backward pass.
    """

    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    part = tl.load(part_ptr + pid_b * NCHUNK + tl.arange(0, NCHUNK))
    max_val = tl.max(part, axis=0)

    if pid_n == 0:
        tl.store(max_ptr + pid_b, max_val)

    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = pid_b.to(tl.int64) * n

    ref = tl.load(ref_ptr + base + offs, mask=mask, other=0.0)
    deg = tl.load(deg_ptr + base + offs, mask=mask, other=0.0)

    tl.store(oref_ptr + base + offs, tl.div_rn(ref, max_val), mask=mask)
    tl.store(odeg_ptr + base + offs, tl.div_rn(deg, max_val), mask=mask)


@triton.jit
def _div_pair_kernel(
    a_ptr,
    b_ptr,
    s_ptr,
    oa_ptr,
    ob_ptr,
    n,
    HAS_A: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Divide up to two ``[batch, n]`` tensors by a per batch scalar."""

    pid_b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    base = pid_b.to(tl.int64) * n
    scalar = tl.load(s_ptr + pid_b)

    if HAS_A:
        val = tl.load(a_ptr + base + offs, mask=mask, other=0.0)
        tl.store(oa_ptr + base + offs, tl.div_rn(val, scalar), mask=mask)
    if HAS_B:
        val = tl.load(b_ptr + base + offs, mask=mask, other=0.0)
        tl.store(ob_ptr + base + offs, tl.div_rn(val, scalar), mask=mask)


class _PeakNormalize(torch.autograd.Function):
    """Equalise reference and degraded signal to the ``[-1, 1]`` range.

    The peak amplitude is treated as a **constant** in the backward pass, i.e.
    the gradient is simply ``grad / max_val``. This is not an approximation of
    the composed pipeline but exactly correct for it: the very next stage,
    :func:`align_scale`, rescales every signal so that the power of its band
    pass filtered copy equals ``1e7``. The pipeline output is therefore
    invariant under multiplication of either input signal with an arbitrary
    positive scalar, which makes the omitted term -- the derivative flowing
    through ``max_val`` -- analytically zero.
    """

    @staticmethod
    def forward(ctx, ref, deg, stack=False):
        ref = check_input(ref, "ref")
        deg = check_input(deg, "deg")

        if ref.ndim != 2 or deg.ndim != 2:
            raise RuntimeError(
                "peak_normalize expects [batch, sample] tensors, got "
                f"{tuple(ref.shape)} and {tuple(deg.shape)}."
            )
        if ref.shape != deg.shape:
            raise RuntimeError(
                "peak_normalize expects matching shapes, got "
                f"{tuple(ref.shape)} and {tuple(deg.shape)}."
            )

        batch, n = ref.shape
        if n == 0:
            raise RuntimeError("peak_normalize needs at least one sample.")
        _check_row_length(n, "peak_normalize")

        part = torch.empty((batch, _NCHUNK), device=ref.device, dtype=ref.dtype)
        max_val = torch.empty((batch,), device=ref.device, dtype=ref.dtype)

        if stack:
            # write both signals into one [2 * batch, sample] buffer so that
            # every following stage of the pipeline runs as a single launch
            stacked = torch.empty((2 * batch, n), device=ref.device, dtype=ref.dtype)
            ref_out, deg_out = stacked[:batch], stacked[batch:]
        else:
            ref_out = torch.empty_like(ref)
            deg_out = torch.empty_like(deg)

        with torch.cuda.device(ref.device):
            _peak_partial_kernel[(batch, _NCHUNK)](
                ref, deg, part, n, BLOCK=_BLOCK, NCHUNK=_NCHUNK
            )
            _peak_apply_kernel[(batch, triton.cdiv(n, _BLOCK))](
                part,
                ref,
                deg,
                ref_out,
                deg_out,
                max_val,
                n,
                NCHUNK=_NCHUNK,
                BLOCK=_BLOCK,
            )

        ctx.save_for_backward(max_val)
        ctx.shape = (batch, n)
        ctx.stack = stack

        return stacked if stack else (ref_out, deg_out)

    @staticmethod
    def backward(ctx, *grads):
        (max_val,) = ctx.saved_tensors
        batch, n = ctx.shape

        need_ref, need_deg = ctx.needs_input_grad[:2]
        if not need_ref and not need_deg:
            return None, None, None

        if ctx.stack:
            stacked = check_input(grads[0], "grad")
            grad_ref, grad_deg = stacked[:batch], stacked[batch:]
        else:
            grad_ref = check_input(grads[0], "grad_ref")
            grad_deg = check_input(grads[1], "grad_deg")

        out_ref = torch.empty_like(grad_ref) if need_ref else grad_ref
        out_deg = torch.empty_like(grad_deg) if need_deg else grad_deg

        with torch.cuda.device(grad_ref.device):
            _div_pair_kernel[(batch, triton.cdiv(n, _BLOCK))](
                grad_ref,
                grad_deg,
                max_val,
                out_ref,
                out_deg,
                n,
                HAS_A=need_ref,
                HAS_B=need_deg,
                BLOCK=_BLOCK,
            )

        return (
            (out_ref if need_ref else None),
            (out_deg if need_deg else None),
            None,
        )


def peak_normalize(ref, deg, stack=False):
    """Divide both signals by their common peak amplitude.

    ``max_val[b] = max(max_n |deg[b, n]|, max_n |ref[b, n]|)`` and the returned
    tensors are ``ref / max_val`` and ``deg / max_val``. See
    :class:`_PeakNormalize` for the (deliberate) gradient semantics: ``max_val``
    is treated as a constant, which is exact for the composed PESQ pipeline.

    Parameters
    ----------
    ref : torch.Tensor
        Reference signal with shape ``[batch, sample]``
    deg : torch.Tensor
        Degraded signal with shape ``[batch, sample]``
    stack : bool
        Return a single ``[2 * batch, sample]`` tensor with the reference in the
        first half instead of two tensors. Both signals go through identical
        stages afterwards, so stacking them lets the rest of the pipeline run
        with half the kernel launches. No copy is involved, the normalisation
        kernel writes straight into the stacked buffer.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor] or torch.Tensor
        Normalised reference and degraded signal, both ``[batch, sample]``, or
        the stacked ``[2 * batch, sample]`` tensor when ``stack`` is set
    """

    return _PeakNormalize.apply(ref, deg, stack)


# ---------------------------------------------------------------------------
# edge ramp
# ---------------------------------------------------------------------------


@triton.jit
def _edge_ramp_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Multiply the first and last 15 samples with a linear ramp.

    The reference applies the two ramps as two *sequential* in place
    assignments, so for ``n < 30`` -- where the leading and the trailing slice
    overlap -- the overlapping samples are scaled by both weights. Evaluating
    ``(val * head) * tail`` in exactly that order reproduces the reference bit
    for bit, and both weights are ``1.0`` outside their ramp, which makes the
    common ``n >= 30`` case an exact no-op multiply.
    """

    pid_b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    base = pid_b.to(tl.int64) * n
    val = tl.load(x_ptr + base + offs, mask=mask, other=0.0)

    one = tl.full([BLOCK], 1.0, dtype=tl.float32)
    head = tl.where(offs < 15, (offs + 1).to(tl.float32) / 16.0, one)
    tail = tl.where(offs >= n - 15, (n - offs).to(tl.float32) / 16.0, one)

    tl.store(out_ptr + base + offs, (val * head) * tail, mask=mask)


class _EdgeRamp(torch.autograd.Function):
    """Linear fade of the first and last 15 samples.

    Reproduces the two masked assignments at the top of
    :func:`torch_pesq.triton_ops.reference.ref_preemphasize`, including their
    sequential semantics when the two 15 sample slices overlap. The weights
    ``k / 16`` for ``k = 1 .. 15`` and their pairwise products ``j * k / 256``
    are exact in binary floating point, so the forward pass is bit exact with
    the oracle. The backward pass is the same multiplication.
    """

    @staticmethod
    def forward(ctx, x):
        x = check_input(x, "x")

        if x.ndim != 2:
            raise RuntimeError(
                f"edge_ramp expects a [batch, sample] tensor, got {tuple(x.shape)}."
            )

        batch, n = x.shape
        if n < 15:
            raise RuntimeError(
                "edge_ramp needs at least 15 samples, the length of a single "
                f"ramp, got {n}. The reference cannot be evaluated below that "
                "either -- its `signal[:, :15] * emp` would fail to broadcast."
            )
        _check_row_length(n, "edge_ramp")

        out = torch.empty_like(x)
        with torch.cuda.device(x.device):
            _edge_ramp_kernel[(batch, triton.cdiv(n, _BLOCK))](x, out, n, BLOCK=_BLOCK)

        ctx.shape = (batch, n)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        batch, n = ctx.shape

        if not ctx.needs_input_grad[0]:
            return None

        grad_out = check_input(grad_out, "grad_out")
        grad_in = torch.empty_like(grad_out)

        with torch.cuda.device(grad_out.device):
            _edge_ramp_kernel[(batch, triton.cdiv(n, _BLOCK))](
                grad_out, grad_in, n, BLOCK=_BLOCK
            )

        return grad_in


def edge_ramp(x):
    """Fade the first and last 15 samples of a signal.

    With ``emp = torch.linspace(0, 15, 16)[1:] / 16`` the result is
    ``y[:, :15] = x[:, :15] * emp``, then ``y[:, -15:] = y[:, -15:] * flip(emp)``
    and a plain copy in between. The two assignments are applied in that order,
    so for ``sample < 30`` the overlapping samples receive both weights.

    Parameters
    ----------
    x : torch.Tensor
        Time signal with shape ``[batch, sample]``, at least 15 samples

    Returns
    -------
    torch.Tensor
        Ramped signal with shape ``[batch, sample]``
    """

    return _EdgeRamp.apply(x)


# ---------------------------------------------------------------------------
# level alignment scaling
# ---------------------------------------------------------------------------


@triton.jit
def _dot_partial_kernel(
    a_ptr, b_ptr, part_ptr, n, BLOCK: tl.constexpr, NCHUNK: tl.constexpr
):
    """First stage: per chunk value of ``sum_n a[n] * b[n]``.

    The chunk accumulator is a full ``BLOCK`` wide vector which is only folded
    into a scalar by a ``tl.sum`` tree at the very end. Together with the
    second stage this gives a reduction depth of ``log2(BLOCK) + log2(NCHUNK) +
    n / (BLOCK * NCHUNK)`` instead of the ``n`` of a serial accumulator, which
    matters for the ~1e5 large sums of squares of this pipeline.
    """

    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    base = pid_b.to(tl.int64) * n
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    num_blocks = tl.cdiv(n, BLOCK)
    for blk in range(pid_c, num_blocks, NCHUNK):
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        lhs = tl.load(a_ptr + base + offs, mask=mask, other=0.0)
        rhs = tl.load(b_ptr + base + offs, mask=mask, other=0.0)
        acc += lhs * rhs

    tl.store(part_ptr + pid_b * NCHUNK + pid_c, tl.sum(acc, axis=0))


@triton.jit
def _align_apply_kernel(
    part_ptr,
    signal_ptr,
    out_ptr,
    power_ptr,
    scale_ptr,
    n,
    denom,
    NCHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Second reduction stage, ``sqrt(1e7 / power)`` and the scaling.

    Fused into a single kernel launch; the first program along the sample axis
    also writes ``power`` and ``scale`` out for the backward pass.
    """

    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    part = tl.load(part_ptr + pid_b * NCHUNK + tl.arange(0, NCHUNK))

    power = tl.div_rn(tl.div_rn(tl.sum(part, axis=0), denom), 1.04684)
    scale = tl.sqrt_rn(tl.div_rn(1e7, power))

    if pid_n == 0:
        tl.store(power_ptr + pid_b, power)
        tl.store(scale_ptr + pid_b, scale)

    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = pid_b.to(tl.int64) * n

    val = tl.load(signal_ptr + base + offs, mask=mask, other=0.0)
    tl.store(out_ptr + base + offs, val * scale, mask=mask)


@triton.jit
def _mul_bcast_kernel(x_ptr, s_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Multiply a ``[batch, n]`` tensor with a per batch scalar."""

    pid_b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    base = pid_b.to(tl.int64) * n
    scalar = tl.load(s_ptr + pid_b)
    val = tl.load(x_ptr + base + offs, mask=mask, other=0.0)

    tl.store(out_ptr + base + offs, val * scalar, mask=mask)


@triton.jit
def _align_grad_filtered_kernel(
    part_ptr,
    scale_ptr,
    power_ptr,
    filtered_ptr,
    out_ptr,
    n,
    denom,
    NCHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gradient w.r.t. the band pass filtered signal.

    ``part`` holds the partial results of ``S = sum_n grad[n] * signal[n]``.
    With ``grad_power = -0.5 * S * scale / power`` and
    ``d power / d sum = 1 / ((n + 5120) * 1.04684)`` the result is
    ``2 * filtered * grad_power / ((n + 5120) * 1.04684)``.
    """

    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    part = tl.load(part_ptr + pid_b * NCHUNK + tl.arange(0, NCHUNK))
    dot = tl.sum(part, axis=0)

    scale = tl.load(scale_ptr + pid_b)
    power = tl.load(power_ptr + pid_b)

    grad_power = tl.div_rn(-0.5 * dot * scale, power)
    grad_sumsq = tl.div_rn(tl.div_rn(grad_power, 1.04684), denom)

    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = pid_b.to(tl.int64) * n

    val = tl.load(filtered_ptr + base + offs, mask=mask, other=0.0)
    tl.store(out_ptr + base + offs, 2.0 * grad_sumsq * val, mask=mask)


class _AlignScale(torch.autograd.Function):
    """Scale ``signal`` so that the power of ``filtered`` reaches ``1e7``.

    Split out of :func:`torch_pesq.triton_ops.reference.ref_align_level`: the
    band pass filtering itself is a separate operator, this one only consumes
    its output. With ``D = (N + 5120) * 1.04684`` and
    ``power = sum_n filtered[n]^2 / D`` the forward pass is
    ``out = signal * sqrt(1e7 / power)``.

    The backward pass follows from ``d scale / d power = -0.5 * scale / power``
    and ``d power / d filtered = 2 * filtered / D``:

    ``grad_signal = grad * scale``,
    ``grad_filtered = 2 * filtered * (-0.5 * S * scale / power) / D`` with
    ``S = sum_n grad[n] * signal[n]``.
    """

    @staticmethod
    def forward(ctx, signal, filtered):
        signal = check_input(signal, "signal")
        filtered = check_input(filtered, "filtered")

        if signal.ndim != 2 or filtered.ndim != 2:
            raise RuntimeError(
                "align_scale expects [batch, sample] tensors, got "
                f"{tuple(signal.shape)} and {tuple(filtered.shape)}."
            )
        if signal.shape[0] != filtered.shape[0]:
            raise RuntimeError(
                "align_scale expects a matching batch size, got "
                f"{signal.shape[0]} and {filtered.shape[0]}."
            )

        batch, n_sig = signal.shape
        n_flt = filtered.shape[1]
        if n_sig == 0 or n_flt == 0:
            raise RuntimeError("align_scale needs at least one sample.")
        _check_row_length(n_sig, "align_scale")
        _check_row_length(n_flt, "align_scale")

        denom = float(n_flt + 5120)

        part = torch.empty((batch, _NCHUNK), device=signal.device, dtype=signal.dtype)
        power = torch.empty((batch,), device=signal.device, dtype=signal.dtype)
        scale = torch.empty((batch,), device=signal.device, dtype=signal.dtype)
        out = torch.empty_like(signal)

        with torch.cuda.device(signal.device):
            _dot_partial_kernel[(batch, _NCHUNK)](
                filtered, filtered, part, n_flt, BLOCK=_BLOCK, NCHUNK=_NCHUNK
            )
            _align_apply_kernel[(batch, triton.cdiv(n_sig, _BLOCK))](
                part,
                signal,
                out,
                power,
                scale,
                n_sig,
                denom,
                NCHUNK=_NCHUNK,
                BLOCK=_BLOCK,
            )

        ctx.save_for_backward(signal, filtered, power, scale)
        ctx.denom = denom

        return out

    @staticmethod
    def backward(ctx, grad_out):
        signal, filtered, power, scale = ctx.saved_tensors
        denom = ctx.denom

        need_sig, need_flt = ctx.needs_input_grad[:2]
        if not need_sig and not need_flt:
            return None, None

        grad_out = check_input(grad_out, "grad_out")

        batch, n_sig = signal.shape
        n_flt = filtered.shape[1]

        grad_signal = None
        grad_filtered = None

        with torch.cuda.device(signal.device):
            if need_sig:
                grad_signal = torch.empty_like(signal)
                _mul_bcast_kernel[(batch, triton.cdiv(n_sig, _BLOCK))](
                    grad_out, scale, grad_signal, n_sig, BLOCK=_BLOCK
                )

            if need_flt:
                part = torch.empty(
                    (batch, _NCHUNK), device=signal.device, dtype=signal.dtype
                )
                grad_filtered = torch.empty_like(filtered)

                _dot_partial_kernel[(batch, _NCHUNK)](
                    grad_out, signal, part, n_sig, BLOCK=_BLOCK, NCHUNK=_NCHUNK
                )
                _align_grad_filtered_kernel[(batch, triton.cdiv(n_flt, _BLOCK))](
                    part,
                    scale,
                    power,
                    filtered,
                    grad_filtered,
                    n_flt,
                    denom,
                    NCHUNK=_NCHUNK,
                    BLOCK=_BLOCK,
                )

        return grad_signal, grad_filtered


def align_scale(signal, filtered):
    """Align the power of a band pass filtered signal to ``1e7``.

    ``power = sum_n filtered[n]^2 / (N + 5120) / 1.04684`` with
    ``N = filtered.shape[1]``, ``scale = sqrt(1e7 / power)`` and the result is
    ``signal * scale``. The reduction is a deterministic two-stage tree.

    Parameters
    ----------
    signal : torch.Tensor
        Time signal with shape ``[batch, sample]``
    filtered : torch.Tensor
        Band pass filtered copy of ``signal`` with shape ``[batch, sample]``

    Returns
    -------
    torch.Tensor
        Scaled time signal with the shape of ``signal``
    """

    return _AlignScale.apply(signal, filtered)


# ---------------------------------------------------------------------------
# MOS epilogue
# ---------------------------------------------------------------------------


@triton.jit
def _epilogue_kernel(
    ds_ptr, da_ptr, mos_ptr, loss_ptr, factor, batch, BLOCK: tl.constexpr
):
    """Compression curve and loss from the two distances."""

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch

    d_symm = tl.load(ds_ptr + offs, mask=mask, other=0.0)
    d_asymm = tl.load(da_ptr + offs, mask=mask, other=0.0)

    inner = 4.5 - 0.1 * d_symm - 0.0309 * d_asymm
    mos = 0.999 + tl.div_rn(4.0, 1.0 + tl.exp(-1.3669 * inner + 3.8224))
    loss = factor * (0.1 * d_symm + 0.0309 * d_asymm)

    tl.store(mos_ptr + offs, mos, mask=mask)
    tl.store(loss_ptr + offs, loss, mask=mask)


@triton.jit
def _epilogue_bwd_kernel(
    ds_ptr,
    da_ptr,
    gmos_ptr,
    gloss_ptr,
    gds_ptr,
    gda_ptr,
    factor,
    batch,
    HAS_MOS: tl.constexpr,
    HAS_LOSS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gradients of :func:`_epilogue_kernel` w.r.t. both distances."""

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch

    grad_ds = tl.zeros([BLOCK], dtype=tl.float32)
    grad_da = tl.zeros([BLOCK], dtype=tl.float32)

    if HAS_MOS:
        d_symm = tl.load(ds_ptr + offs, mask=mask, other=0.0)
        d_asymm = tl.load(da_ptr + offs, mask=mask, other=0.0)
        gmos = tl.load(gmos_ptr + offs, mask=mask, other=0.0)

        inner = 4.5 - 0.1 * d_symm - 0.0309 * d_asymm
        expo = tl.exp(-1.3669 * inner + 3.8224)
        denom = 1.0 + expo

        # d mos / d inner, written in the same order as autograd evaluates it
        grad_inner = gmos * tl.div_rn(-4.0, denom * denom) * expo * -1.3669

        grad_ds += grad_inner * -0.1
        grad_da += grad_inner * -0.0309

    if HAS_LOSS:
        gloss = tl.load(gloss_ptr + offs, mask=mask, other=0.0)

        grad_ds += gloss * factor * 0.1
        grad_da += gloss * factor * 0.0309

    tl.store(gds_ptr + offs, grad_ds, mask=mask)
    tl.store(gda_ptr + offs, grad_da, mask=mask)


class _PesqEpilogue(torch.autograd.Function):
    """Mean opinion score and loss value from the two PESQ distances.

    ``mos = 0.999 + 4 / (1 + exp(-1.3669 * (4.5 - 0.1 * d_symm -
    0.0309 * d_asymm) + 3.8224))`` and
    ``loss = factor * (0.1 * d_symm + 0.0309 * d_asymm)``.

    Both outputs are differentiable and the backward pass accepts upstream
    gradients for either or both of them.
    """

    @staticmethod
    def forward(ctx, d_symm, d_asymm, factor):
        d_symm = check_input(d_symm, "d_symm")
        d_asymm = check_input(d_asymm, "d_asymm")

        if d_symm.ndim != 1 or d_asymm.ndim != 1:
            raise RuntimeError(
                "pesq_epilogue expects [batch] tensors, got "
                f"{tuple(d_symm.shape)} and {tuple(d_asymm.shape)}."
            )
        if d_symm.shape != d_asymm.shape:
            raise RuntimeError(
                "pesq_epilogue expects matching shapes, got "
                f"{tuple(d_symm.shape)} and {tuple(d_asymm.shape)}."
            )
        if isinstance(factor, torch.Tensor) and factor.requires_grad:
            # the backward pass returns ``None`` for ``factor``; accepting a
            # tensor that requires a gradient here would silently drop it
            raise RuntimeError(
                "pesq_epilogue treats `factor` as a constant, pass a Python "
                "float. A tensor that requires a gradient would receive none."
            )

        batch = d_symm.shape[0]

        mos = torch.empty_like(d_symm)
        loss = torch.empty_like(d_symm)

        with torch.cuda.device(d_symm.device):
            _epilogue_kernel[(triton.cdiv(batch, _BLOCK_B),)](
                d_symm, d_asymm, mos, loss, float(factor), batch, BLOCK=_BLOCK_B
            )

        ctx.save_for_backward(d_symm, d_asymm)
        ctx.factor = float(factor)

        return mos, loss

    @staticmethod
    def backward(ctx, grad_mos, grad_loss):
        d_symm, d_asymm = ctx.saved_tensors
        batch = d_symm.shape[0]

        need_ds, need_da = ctx.needs_input_grad[:2]
        if not need_ds and not need_da:
            return None, None, None

        has_mos = grad_mos is not None
        has_loss = grad_loss is not None

        grad_mos = check_input(grad_mos, "grad_mos") if has_mos else d_symm
        grad_loss = check_input(grad_loss, "grad_loss") if has_loss else d_symm

        grad_ds = torch.empty_like(d_symm)
        grad_da = torch.empty_like(d_asymm)

        with torch.cuda.device(d_symm.device):
            _epilogue_bwd_kernel[(triton.cdiv(batch, _BLOCK_B),)](
                d_symm,
                d_asymm,
                grad_mos,
                grad_loss,
                grad_ds,
                grad_da,
                ctx.factor,
                batch,
                HAS_MOS=has_mos,
                HAS_LOSS=has_loss,
                BLOCK=_BLOCK_B,
            )

        return (grad_ds if need_ds else None), (grad_da if need_da else None), None


def pesq_epilogue(d_symm, d_asymm, factor):
    """Turn the PESQ distances into a MOS estimate and a loss value.

    Parameters
    ----------
    d_symm : torch.Tensor
        Symmetric distance with shape ``[batch]``
    d_asymm : torch.Tensor
        Asymmetric distance with shape ``[batch]``
    factor : float
        Scaling of the loss function

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Mean opinion score in range ``(1.08, 4.999)`` and the loss value, both
        with shape ``[batch]``
    """

    return _PesqEpilogue.apply(d_symm, d_asymm, factor)
