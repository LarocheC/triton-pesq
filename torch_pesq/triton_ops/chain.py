"""Fused Triton kernels for the PESQ perceptual chain.

This module covers the whole second half of :meth:`torch_pesq.loss.PesqLoss.raw`,
i.e. everything between the silent frame detection and the overlapping sums.  The
pure PyTorch oracle for this stage is
:func:`torch_pesq.triton_ops.reference.ref_chain`.

The stage is a chain of reductions with genuine data dependencies, therefore the
forward pass is split into six stages (seven kernel launches):

  1. ``_partial_pow_kernel``    -- silent frames and per block band partials
  2. ``_band_ratio_kernel``     -- deterministic reduction plus ``band_pow_ratio``
  3. ``_frame_ratio_kernel``    -- total audible energies, ``frame_pow_ratio`` and ``h``
  4. ``_fir_kernel``            -- the two tap FIR over ``frame_pow_ratio``
  5. ``_distortion_kernel``     -- loudness, disturbance and both weighted norms
  6. ``_psqm_kernel`` / ``_reduce_dist_kernel`` -- the overlapping sums

The backward pass mirrors that structure with six kernels of its own.  Every
reduction is a two stage tree reduction with a fixed traversal order, no atomics
are used, so the result is bitwise reproducible from run to run.  Nothing of size
``[batch, frame, bark]`` besides the two inputs and the two gradients is ever
materialised, the backward kernels recompute the loudness and the disturbance.

Accuracy note
-------------
The oracle silently promotes the loudness, disturbance and norm computation to
float64, because :attr:`torch_pesq.loudness.Loudness.threshs` is a float64
buffer.  These kernels deliberately stay in float32 (float64 runs at 1/32 rate on
the target hardware), so the agreement with the oracle is limited by float32
round off.

On well conditioned data the forward stays below ``6e-06`` and both gradients
below ``3e-05`` relative to their own largest magnitude.  Two regimes are
genuinely worse and no float32 implementation can do better there:

``deg`` almost equal to ``ref``
    ``dist = sign(u) (|u| - 0.25 min(deg_loud, ref_loud))`` subtracts two nearly
    equal quantities twice over, first in ``u = deg_loud - ref_loud`` and then in
    the deadzone.  A pure float32 PyTorch evaluation of
    :func:`~torch_pesq.triton_ops.reference.ref_chain` loses exactly the same
    digits; ``tests/test_triton_chain.py::test_conditioning_matches_float32``
    pins the kernels to that bound instead of to an absolute tolerance.

bands sitting a hair above their hearing threshold
    the loudness itself is then ``O(eps)`` of the terms it is built from.  See
    :func:`_loudness`, which is written so that only the float32 rounding of
    ``equ_ref``/``equ_deg`` -- not the formula -- limits the result.
"""

import math

import torch

from ._common import HAS_TRITON, check_input, next_power_of_two, require_triton

if HAS_TRITON:  # pragma: no cover - depends on the installation
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
else:  # pragma: no cover - depends on the installation
    import types

    triton = types.SimpleNamespace(jit=lambda fn: fn, cdiv=lambda a, b: -(-a // b))
    tl = types.SimpleNamespace(constexpr=int)
    libdevice = None


__all__ = ["TritonPesqChain"]


# Frames per program of the ``[batch, frame, bark]`` kernels.  A tile is
# ``BLOCK_T x BLOCK_K`` with ``BLOCK_K = 64``; keeping it small avoids register
# spills on Pascal, where the disturbance backward holds ~25 live tiles.  Swept
# over ``{2, 4, 8, 16, 32} x {1, 2, 4, 8}`` warps on a GTX 1080 Ti, 4 with the
# default 4 warps was the best forward plus backward combination.
_BLOCK_T = 4

# Frames per program of the cheap ``[batch, frame]`` kernels.
_BLOCK_T_SMALL = 128

# Windows per program of the overlapping sum kernels.
_BLOCK_L = 32

_WINDOW = 20
_STRIDE = 10

# Crossover of the two loudness evaluations in :func:`_loudness`, in units of
# ``base - 1``.  Both are within 2.3e-07 relative of float64 at the crossover,
# see the table there.  Kernels can only read globals that are `tl.constexpr`.
#
# Careful when tuning this: Triton 3.1 does not put the *value* of a global
# constexpr into its on disk cache key, so editing the number alone hands back
# the previously compiled kernel.  Clear ``~/.triton/cache`` (or point
# ``TRITON_CACHE_DIR`` somewhere fresh) to see the change.
_LOUD_SPLIT = tl.constexpr(1.0e3)


# ---------------------------------------------------------------------------
# forward kernels
# ---------------------------------------------------------------------------


@triton.jit
def _partial_pow_kernel(
    ref_ptr,
    deg_ptr,
    th_ptr,
    pref_ptr,
    pdeg_ptr,
    T,
    K,
    NB,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Silent frame detection and per block partials of the mean band powers."""

    b = tl.program_id(0)
    nb = tl.program_id(1)

    t = nb * BLOCK_T + tl.arange(0, BLOCK_T)
    k = tl.arange(0, BLOCK_K)
    tmask = t < T
    kmask = k < K
    mask = tmask[:, None] & kmask[None, :]

    off = b * T * K + t[:, None] * K + k[None, :]
    ref = tl.load(ref_ptr + off, mask=mask, other=0.0)
    deg = tl.load(deg_ptr + off, mask=mask, other=0.0)
    th100 = tl.load(th_ptr + k, mask=kmask, other=1.0)[None, :] * 100.0

    aud_ref = tl.where(ref > th100, ref, 0.0)
    keep = (tl.sum(aud_ref, axis=1) >= 1e7)[:, None]

    aud_deg = tl.where(deg > th100, deg, 0.0)

    poff = b * NB * K + nb * K + k
    tl.store(pref_ptr + poff, tl.sum(tl.where(keep, aud_ref, 0.0), axis=0), mask=kmask)
    tl.store(pdeg_ptr + poff, tl.sum(tl.where(keep, aud_deg, 0.0), axis=0), mask=kmask)


@triton.jit
def _band_ratio_kernel(
    pref_ptr,
    pdeg_ptr,
    mref_ptr,
    ratio_ptr,
    r_ptr,
    Tf,
    K,
    NB,
    BLOCK_K: tl.constexpr,
):
    """Reduce the band partials and form the clamped ``band_pow_ratio``."""

    b = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    kmask = k < K

    acc_ref = tl.zeros([BLOCK_K], dtype=tl.float32)
    acc_deg = tl.zeros([BLOCK_K], dtype=tl.float32)
    for i in range(NB):
        o = b * NB * K + i * K + k
        acc_ref += tl.load(pref_ptr + o, mask=kmask, other=0.0)
        acc_deg += tl.load(pdeg_ptr + o, mask=kmask, other=0.0)

    mref = acc_ref / Tf
    mdeg = acc_deg / Tf
    ratio = (mdeg + 1000.0) / (mref + 1000.0)

    o = b * K + k
    tl.store(mref_ptr + o, mref, mask=kmask)
    tl.store(ratio_ptr + o, ratio, mask=kmask)
    tl.store(r_ptr + o, tl.minimum(tl.maximum(ratio, 0.01), 100.0), mask=kmask)


@triton.jit
def _frame_ratio_kernel(
    ref_ptr,
    deg_ptr,
    r_ptr,
    th_ptr,
    taer_ptr,
    tad_ptr,
    fpr0_ptr,
    h_ptr,
    T,
    K,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Total audible energies, the raw frame power ratio and the weighting ``h``."""

    b = tl.program_id(0)
    nb = tl.program_id(1)

    t = nb * BLOCK_T + tl.arange(0, BLOCK_T)
    k = tl.arange(0, BLOCK_K)
    tmask = t < T
    kmask = k < K
    mask = tmask[:, None] & kmask[None, :]

    off = b * T * K + t[:, None] * K + k[None, :]
    ref = tl.load(ref_ptr + off, mask=mask, other=0.0)
    deg = tl.load(deg_ptr + off, mask=mask, other=0.0)
    th = tl.load(th_ptr + k, mask=kmask, other=1.0)[None, :]
    r = tl.load(r_ptr + b * K + k, mask=kmask, other=0.0)[None, :]

    equ_ref = r * ref
    taer = tl.sum(tl.where(equ_ref > th, equ_ref, 0.0), axis=1)
    tad = tl.sum(tl.where(deg > th, deg, 0.0), axis=1)

    o = b * T + t
    tl.store(taer_ptr + o, taer, mask=tmask)
    tl.store(tad_ptr + o, tad, mask=tmask)
    tl.store(fpr0_ptr + o, (taer + 5e3) / (tad + 5e3), mask=tmask)
    tl.store(h_ptr + o, libdevice.pow((taer + 1e5) / 1e7, 0.04), mask=tmask)


@triton.jit
def _fir_kernel(fpr0_ptr, fpr1_ptr, T, BLOCK_T: tl.constexpr):
    """Two tap FIR over the frame power ratio, evaluated on the original values."""

    b = tl.program_id(0)
    nb = tl.program_id(1)

    t = nb * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < T
    base = b * T

    cur = tl.load(fpr0_ptr + base + t, mask=tmask, other=0.0)
    prev = tl.load(fpr0_ptr + base + t - 1, mask=tmask & (t > 0), other=0.0)

    tl.store(
        fpr1_ptr + base + t,
        tl.where(t == 0, cur, cur * 0.8 + prev * 0.2),
        mask=tmask,
    )


@triton.jit
def _base_minus_one(x, th):
    """``0.5 + 0.5 x / th - 1``, evaluated without cancellation."""

    # ``0.5 * (x - th) / th`` instead of ``0.5 + 0.5 * x / th - 1``: for ``x`` in
    # ``[th/2, 2 th]`` the subtraction ``x - th`` is exact (Sterbenz), so the
    # distance of the loudness base from one keeps full float32 precision.  This
    # matters because the loudness is ``base ** e - 1``, which cancels away every
    # significant bit once ``base`` approaches one.
    return 0.5 * (x - th) / th


@triton.jit
def _loudness(bm1, above, ls, e):
    """Zwicker loudness ``sl (2 th)^e ((0.5 + 0.5 x / th)^e - 1)``, zero below ``th``.

    Takes ``bm1 = base - 1`` from :func:`_base_minus_one` and the ``x > th``
    predicate, so that the forward and the backward kernel evaluate bit
    identical loudness values.
    """

    # Two evaluations of the same expression, each accurate in one half of the
    # range (measured against float64 on this GPU, worst relative error over
    # 4096 samples per decade of ``x / th``):
    #
    #   x / th        1+1e-3    1.01     1.1       10      1e4      1e9     1e18
    #   pow - 1       1.3e-3  1.1e-4  1.2e-05  3.0e-07  1.4e-07  1.4e-07  1.5e-07
    #   expm1/log1p   1.9e-7  1.7e-7  1.6e-07  1.9e-07  2.3e-07  4.8e-07  9.2e-07
    #
    # ``pow(base, e) - 1`` cancels to nothing once ``base`` approaches one, which
    # is every band sitting just above its hearing threshold.  ``expm1`` removes
    # that, but its argument ``e log1p(bm1)`` grows with ``x``, and its rounding
    # is a *relative* error of the result, so it slowly loses accuracy at the
    # top of the range where ``pow`` keeps an extended precision logarithm.
    near = libdevice.expm1(e * libdevice.log1p(bm1))
    far = libdevice.pow(1.0 + bm1, e) - 1.0

    return tl.where(above, ls * tl.where(bm1 <= _LOUD_SPLIT, near, far), 0.0)


@triton.jit
def _distortion_kernel(
    ref_ptr,
    deg_ptr,
    r_ptr,
    fpr1_ptr,
    h_ptr,
    th_ptr,
    exp_ptr,
    ls_ptr,
    w_ptr,
    sa_ptr,
    saraw_ptr,
    T,
    K,
    BTS,
    W,
    SQRT_W,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Loudness, disturbance, both weighted norms and the ``h`` weighting."""

    b = tl.program_id(0)
    nb = tl.program_id(1)

    t = nb * BLOCK_T + tl.arange(0, BLOCK_T)
    k = tl.arange(0, BLOCK_K)
    tmask = t < T
    kmask = k < K
    mask = tmask[:, None] & kmask[None, :]
    bmask = (kmask & (k >= 1))[None, :] & tmask[:, None]

    off = b * T * K + t[:, None] * K + k[None, :]
    ref = tl.load(ref_ptr + off, mask=mask, other=0.0)
    deg = tl.load(deg_ptr + off, mask=mask, other=0.0)
    th = tl.load(th_ptr + k, mask=kmask, other=1.0)[None, :]
    e = tl.load(exp_ptr + k, mask=kmask, other=1.0)[None, :]
    ls = tl.load(ls_ptr + k, mask=kmask, other=0.0)[None, :]
    w = tl.load(w_ptr + k, mask=kmask, other=0.0)[None, :]
    r = tl.load(r_ptr + b * K + k, mask=kmask, other=0.0)[None, :]

    fo = b * T + t
    fpr1 = tl.load(fpr1_ptr + fo, mask=tmask, other=0.0)
    fpr = tl.minimum(tl.maximum(fpr1, 3e-4), 5.0)
    h = tl.load(h_ptr + fo, mask=tmask, other=1.0)

    equ_ref = r * ref
    equ_deg = fpr[:, None] * deg

    ref_loud = _loudness(_base_minus_one(equ_ref, th), equ_ref > th, ls, e)
    deg_loud = _loudness(_base_minus_one(equ_deg, th), equ_deg > th, ls, e)

    u = deg_loud - ref_loud
    z = tl.abs(u) - 0.25 * tl.minimum(deg_loud, ref_loud)
    sgn = tl.where(u > 0, 1.0, tl.where(u < 0, -1.0, 0.0))
    dist = sgn * tl.maximum(z, 0.0)

    v = w * dist / SQRT_W
    symm_raw = W * tl.sqrt(tl.sum(tl.where(bmask, v * v, 0.0), axis=1))

    as_raw = libdevice.pow((equ_deg + 50.0) / (equ_ref + 50.0), 1.2)
    asc = tl.where(as_raw < 3.0, 0.0, tl.minimum(as_raw, 12.0))
    y = w * (dist * asc) / W
    asymm_raw = W * tl.sum(tl.where(bmask, tl.abs(y), 0.0), axis=1)

    symm = tl.minimum(tl.maximum(symm_raw, 1e-20) / h, 45.0)
    asymm = tl.minimum(tl.maximum(asymm_raw, 1e-20) / h, 45.0)

    tl.store(sa_ptr + fo, symm, mask=tmask)
    tl.store(sa_ptr + BTS + fo, asymm, mask=tmask)
    tl.store(saraw_ptr + fo, symm_raw, mask=tmask)
    tl.store(saraw_ptr + BTS + fo, asymm_raw, mask=tmask)


@triton.jit
def _psqm_kernel(s_ptr, p_ptr, T, L, BLOCK_L: tl.constexpr, BLOCK_J: tl.constexpr):
    """Sixth power mean over every window of 20 frames, stride 10."""

    c = tl.program_id(0)
    lb = tl.program_id(1)

    l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    j = tl.arange(0, BLOCK_J)
    lmask = l < L
    m = lmask[:, None] & (j[None, :] < 20)

    s = tl.load(s_ptr + c * T + l[:, None] * 10 + j[None, :], mask=m, other=0.0)

    # normalise by the window maximum, ``s**6`` underflows float32 for the
    # ``1e-20`` clamp floor of the disturbance
    mx = tl.max(s, axis=1)
    scale = tl.where(mx > 0, mx, 1.0)[:, None]
    u = s / scale
    u2 = u * u
    mean6 = tl.sum(tl.where(m, u2 * u2 * u2, 0.0), axis=1) / 20.0

    tl.store(p_ptr + c * L + l, mx * libdevice.pow(mean6, 1.0 / 6.0), mask=lmask)


@triton.jit
def _reduce_dist_kernel(p_ptr, d_ptr, L, Lf, BLOCK_L: tl.constexpr):
    """Root mean square over all windows."""

    c = tl.program_id(0)

    mx = 0.0
    for i in range(0, L, BLOCK_L):
        l = i + tl.arange(0, BLOCK_L)
        p = tl.load(p_ptr + c * L + l, mask=l < L, other=0.0)
        mx = tl.maximum(mx, tl.max(p, axis=0))

    scale = tl.where(mx > 0, mx, 1.0)
    acc = 0.0
    for i in range(0, L, BLOCK_L):
        l = i + tl.arange(0, BLOCK_L)
        p = tl.load(p_ptr + c * L + l, mask=l < L, other=0.0) / scale
        acc += tl.sum(p * p, axis=0)

    tl.store(d_ptr + c, mx * tl.sqrt(acc / Lf))


# ---------------------------------------------------------------------------
# backward kernels
# ---------------------------------------------------------------------------


@triton.jit
def _psqm_bwd_kernel(
    s_ptr, p_ptr, d_ptr, gd_ptr, gs_ptr, T, L, Lf, BLOCK_T: tl.constexpr
):
    """Adjoint of the overlapping sums, at most two windows touch a frame."""

    c = tl.program_id(0)
    nb = tl.program_id(1)

    t = nb * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < T

    s = tl.load(s_ptr + c * T + t, mask=tmask, other=0.0)
    d = tl.load(d_ptr + c)
    gd = tl.load(gd_ptr + c)
    dsafe = tl.where(d > 0, d, 1.0)

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    for shift in range(2):
        l = t // 10 - shift
        ok = tmask & (l >= 0) & (l < L)
        p = tl.load(p_ptr + c * L + l, mask=ok, other=1.0)
        ok = ok & (p > 0)
        psafe = tl.where(p > 0, p, 1.0)
        # dd/dp_l * dp_l/ds_t = p_l / (L d) * (s_t / p_l)**5 / 20
        rat = s / psafe
        rat2 = rat * rat
        acc += tl.where(ok, (p / (Lf * dsafe)) * (rat2 * rat2 * rat) * 0.05, 0.0)

    acc = tl.where(d > 0, acc, 0.0)
    tl.store(gs_ptr + c * T + t, acc * gd, mask=tmask)


@triton.jit
def _distortion_bwd_kernel(
    ref_ptr,
    deg_ptr,
    r_ptr,
    fpr1_ptr,
    h_ptr,
    saraw_ptr,
    gsa_ptr,
    th_ptr,
    exp_ptr,
    ls_ptr,
    w_ptr,
    gref_ptr,
    gdeg_ptr,
    gfpr_ptr,
    gh_ptr,
    grpart_ptr,
    T,
    K,
    NB,
    BTS,
    W,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Adjoint of :func:`_distortion_kernel`, recomputing all intermediates."""

    b = tl.program_id(0)
    nb = tl.program_id(1)

    t = nb * BLOCK_T + tl.arange(0, BLOCK_T)
    k = tl.arange(0, BLOCK_K)
    tmask = t < T
    kmask = k < K
    mask = tmask[:, None] & kmask[None, :]
    bmask = (kmask & (k >= 1))[None, :] & tmask[:, None]

    off = b * T * K + t[:, None] * K + k[None, :]
    ref = tl.load(ref_ptr + off, mask=mask, other=0.0)
    deg = tl.load(deg_ptr + off, mask=mask, other=0.0)
    th = tl.load(th_ptr + k, mask=kmask, other=1.0)[None, :]
    e = tl.load(exp_ptr + k, mask=kmask, other=1.0)[None, :]
    ls = tl.load(ls_ptr + k, mask=kmask, other=0.0)[None, :]
    w = tl.load(w_ptr + k, mask=kmask, other=0.0)[None, :]
    r = tl.load(r_ptr + b * K + k, mask=kmask, other=0.0)[None, :]

    fo = b * T + t
    fpr1 = tl.load(fpr1_ptr + fo, mask=tmask, other=0.0)
    fpr = tl.minimum(tl.maximum(fpr1, 3e-4), 5.0)
    h = tl.load(h_ptr + fo, mask=tmask, other=1.0)
    symm_raw = tl.load(saraw_ptr + fo, mask=tmask, other=0.0)
    asymm_raw = tl.load(saraw_ptr + BTS + fo, mask=tmask, other=0.0)
    g_symm = tl.load(gsa_ptr + fo, mask=tmask, other=0.0)
    g_asymm = tl.load(gsa_ptr + BTS + fo, mask=tmask, other=0.0)

    # --- recompute the forward intermediates
    equ_ref = r * ref
    equ_deg = fpr[:, None] * deg

    bm1_ref = _base_minus_one(equ_ref, th)
    bm1_deg = _base_minus_one(equ_deg, th)
    ref_pos = equ_ref > th
    deg_pos = equ_deg > th
    # identical expression to `_distortion_kernel`, so every comparison below
    # reproduces the branch the forward pass took
    ref_loud = _loudness(bm1_ref, ref_pos, ls, e)
    deg_loud = _loudness(bm1_deg, deg_pos, ls, e)

    u = deg_loud - ref_loud
    z = tl.abs(u) - 0.25 * tl.minimum(deg_loud, ref_loud)
    sgn = tl.where(u > 0, 1.0, tl.where(u < 0, -1.0, 0.0))
    dist = sgn * tl.maximum(z, 0.0)

    as_raw = libdevice.pow((equ_deg + 50.0) / (equ_ref + 50.0), 1.2)
    asc = tl.where(as_raw < 3.0, 0.0, tl.minimum(as_raw, 12.0))

    # --- clamps, the division by h and the 1e-20 floor
    symm_pre = tl.maximum(symm_raw, 1e-20)
    asymm_pre = tl.maximum(asymm_raw, 1e-20)
    g_sq = tl.where(symm_pre / h <= 45.0, g_symm, 0.0)
    g_aq = tl.where(asymm_pre / h <= 45.0, g_asymm, 0.0)
    g_h = -(g_sq * symm_pre + g_aq * asymm_pre) / (h * h)
    g_symm_raw = tl.where(symm_raw >= 1e-20, g_sq / h, 0.0)
    g_asymm_raw = tl.where(asymm_raw >= 1e-20, g_aq / h, 0.0)

    # --- weighted 2-norm: d symm_raw / d dist_k = W w_k^2 dist_k / symm_raw
    coef = g_symm_raw * W / tl.where(symm_raw > 0, symm_raw, 1.0)
    g_dist = coef[:, None] * (w * w * dist)

    # --- weighted 1-norm: d asymm_raw / d (dist_k asc_k) = w_k sign(dist_k asc_k)
    y = dist * asc
    sy = tl.where(y > 0, 1.0, tl.where(y < 0, -1.0, 0.0))
    g_y = g_asymm_raw[:, None] * (w * sy)
    g_dist = tl.where(bmask, g_dist + g_y * asc, 0.0)
    g_asc = tl.where(bmask, g_y * dist, 0.0)

    # --- deadzone
    inzone = z >= 0
    g_u = tl.where(inzone, g_dist * sgn * sgn, 0.0)
    g_min = tl.where(inzone, -0.25 * g_dist * sgn, 0.0)
    tie = deg_loud == ref_loud
    wd = tl.where(deg_loud < ref_loud, 1.0, tl.where(tie, 0.5, 0.0))
    wr = tl.where(ref_loud < deg_loud, 1.0, tl.where(tie, 0.5, 0.0))
    g_deg_loud = g_u + g_min * wd
    g_ref_loud = -g_u + g_min * wr

    # --- loudness, d loud / d x = ls e (1 + bm1)^(e - 1) / (2 th); the power is
    # close to one for near threshold bands, so `pow` is accurate here
    scale = ls * e * 0.5 / th
    dl_ref = tl.where(ref_pos, scale * libdevice.pow(1.0 + bm1_ref, e - 1.0), 0.0)
    dl_deg = tl.where(deg_pos, scale * libdevice.pow(1.0 + bm1_deg, e - 1.0), 0.0)
    g_equ_ref = g_ref_loud * dl_ref
    g_equ_deg = g_deg_loud * dl_deg

    # --- asymmetric scaling, d as_raw / d equ_deg = 1.2 as_raw / (equ_deg + 50)
    live = (as_raw >= 3.0) & (as_raw <= 12.0)
    g_equ_deg += tl.where(live, g_asc * 1.2 * as_raw / (equ_deg + 50.0), 0.0)
    g_equ_ref += tl.where(live, -g_asc * 1.2 * as_raw / (equ_ref + 50.0), 0.0)

    g_equ_ref = tl.where(mask, g_equ_ref, 0.0)
    g_equ_deg = tl.where(mask, g_equ_deg, 0.0)

    tl.store(gref_ptr + off, g_equ_ref * r, mask=mask)
    tl.store(gdeg_ptr + off, g_equ_deg * fpr[:, None], mask=mask)
    tl.store(gfpr_ptr + fo, tl.sum(g_equ_deg * deg, axis=1), mask=tmask)
    tl.store(gh_ptr + fo, g_h, mask=tmask)
    tl.store(
        grpart_ptr + b * NB * K + nb * K + k,
        tl.sum(g_equ_ref * ref, axis=0),
        mask=kmask,
    )


@triton.jit
def _frame_adjoint_kernel(
    gfpr_ptr,
    fpr1_ptr,
    gh_ptr,
    h_ptr,
    taer_ptr,
    tad_ptr,
    fpr0_ptr,
    gtaer_ptr,
    gtad_ptr,
    T,
    BLOCK_T: tl.constexpr,
):
    """Adjoint of the FIR, the clamp and the two total audible energies."""

    b = tl.program_id(0)
    nb = tl.program_id(1)

    t = nb * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < T
    base = b * T

    g_cur = tl.load(gfpr_ptr + base + t, mask=tmask, other=0.0)
    f_cur = tl.load(fpr1_ptr + base + t, mask=tmask, other=0.0)
    g_cur = tl.where((f_cur >= 3e-4) & (f_cur <= 5.0), g_cur, 0.0)

    nmask = tmask & (t + 1 < T)
    g_nxt = tl.load(gfpr_ptr + base + t + 1, mask=nmask, other=0.0)
    f_nxt = tl.load(fpr1_ptr + base + t + 1, mask=nmask, other=0.0)
    g_nxt = tl.where(nmask & (f_nxt >= 3e-4) & (f_nxt <= 5.0), g_nxt, 0.0)

    g_fpr0 = tl.where(t == 0, g_cur, 0.8 * g_cur) + 0.2 * g_nxt

    taer = tl.load(taer_ptr + base + t, mask=tmask, other=0.0)
    tad = tl.load(tad_ptr + base + t, mask=tmask, other=0.0)
    fpr0 = tl.load(fpr0_ptr + base + t, mask=tmask, other=0.0)
    h = tl.load(h_ptr + base + t, mask=tmask, other=1.0)
    g_h = tl.load(gh_ptr + base + t, mask=tmask, other=0.0)

    inv = 1.0 / (tad + 5e3)
    # dh/d taer = 0.04 * ((taer + 1e5) / 1e7)**-0.96 / 1e7 = 0.04 h / (taer + 1e5)
    tl.store(
        gtaer_ptr + base + t,
        g_fpr0 * inv + g_h * 0.04 * h / (taer + 1e5),
        mask=tmask,
    )
    tl.store(gtad_ptr + base + t, -g_fpr0 * fpr0 * inv, mask=tmask)


@triton.jit
def _audible_bwd_kernel(
    ref_ptr,
    deg_ptr,
    r_ptr,
    th_ptr,
    gtaer_ptr,
    gtad_ptr,
    gref_ptr,
    gdeg_ptr,
    grpart_ptr,
    T,
    K,
    NB,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Route the total audible gradients back onto ``ref``, ``deg`` and ``r``."""

    b = tl.program_id(0)
    nb = tl.program_id(1)

    t = nb * BLOCK_T + tl.arange(0, BLOCK_T)
    k = tl.arange(0, BLOCK_K)
    tmask = t < T
    kmask = k < K
    mask = tmask[:, None] & kmask[None, :]

    off = b * T * K + t[:, None] * K + k[None, :]
    ref = tl.load(ref_ptr + off, mask=mask, other=0.0)
    deg = tl.load(deg_ptr + off, mask=mask, other=0.0)
    th = tl.load(th_ptr + k, mask=kmask, other=1.0)[None, :]
    r = tl.load(r_ptr + b * K + k, mask=kmask, other=0.0)[None, :]

    fo = b * T + t
    g_taer = tl.load(gtaer_ptr + fo, mask=tmask, other=0.0)[:, None]
    g_tad = tl.load(gtad_ptr + fo, mask=tmask, other=0.0)[:, None]

    g_equ_ref = tl.where(mask & (r * ref > th), g_taer, 0.0)
    g_deg = tl.where(mask & (deg > th), g_tad, 0.0)

    prev = tl.load(gref_ptr + off, mask=mask, other=0.0)
    tl.store(gref_ptr + off, prev + g_equ_ref * r, mask=mask)
    prev = tl.load(gdeg_ptr + off, mask=mask, other=0.0)
    tl.store(gdeg_ptr + off, prev + g_deg, mask=mask)

    tl.store(
        grpart_ptr + b * NB * K + nb * K + k,
        tl.sum(g_equ_ref * ref, axis=0),
        mask=kmask,
    )


@triton.jit
def _band_ratio_bwd_kernel(
    p1_ptr,
    p2_ptr,
    ratio_ptr,
    mref_ptr,
    gmref_ptr,
    gmdeg_ptr,
    K,
    NB,
    BLOCK_K: tl.constexpr,
):
    """Reduce both ``band_pow_ratio`` partials and undo the ratio and its clamp."""

    b = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    kmask = k < K

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for i in range(NB):
        o = b * NB * K + i * K + k
        acc += tl.load(p1_ptr + o, mask=kmask, other=0.0)
        acc += tl.load(p2_ptr + o, mask=kmask, other=0.0)

    o = b * K + k
    ratio = tl.load(ratio_ptr + o, mask=kmask, other=0.0)
    mref = tl.load(mref_ptr + o, mask=kmask, other=0.0)

    g_ratio = tl.where((ratio >= 0.01) & (ratio <= 100.0), acc, 0.0)
    inv = 1.0 / (mref + 1000.0)

    tl.store(gmref_ptr + o, -g_ratio * ratio * inv, mask=kmask)
    tl.store(gmdeg_ptr + o, g_ratio * inv, mask=kmask)


@triton.jit
def _mean_pow_bwd_kernel(
    ref_ptr,
    deg_ptr,
    th_ptr,
    gmref_ptr,
    gmdeg_ptr,
    gref_ptr,
    gdeg_ptr,
    T,
    K,
    Tf,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Scatter the mean band power gradients, recomputing the silent frame mask."""

    b = tl.program_id(0)
    nb = tl.program_id(1)

    t = nb * BLOCK_T + tl.arange(0, BLOCK_T)
    k = tl.arange(0, BLOCK_K)
    tmask = t < T
    kmask = k < K
    mask = tmask[:, None] & kmask[None, :]

    off = b * T * K + t[:, None] * K + k[None, :]
    ref = tl.load(ref_ptr + off, mask=mask, other=0.0)
    deg = tl.load(deg_ptr + off, mask=mask, other=0.0)
    th100 = tl.load(th_ptr + k, mask=kmask, other=1.0)[None, :] * 100.0

    keep = (tl.sum(tl.where(ref > th100, ref, 0.0), axis=1) >= 1e7)[:, None]

    g_mref = tl.load(gmref_ptr + b * K + k, mask=kmask, other=0.0)[None, :] / Tf
    g_mdeg = tl.load(gmdeg_ptr + b * K + k, mask=kmask, other=0.0)[None, :] / Tf

    prev = tl.load(gref_ptr + off, mask=mask, other=0.0)
    tl.store(
        gref_ptr + off,
        prev + tl.where(keep & (ref > th100), g_mref, 0.0),
        mask=mask,
    )
    prev = tl.load(gdeg_ptr + off, mask=mask, other=0.0)
    tl.store(
        gdeg_ptr + off,
        prev + tl.where(keep & (deg > th100), g_mdeg, 0.0),
        mask=mask,
    )


# ---------------------------------------------------------------------------
# autograd glue
# ---------------------------------------------------------------------------


class _ChainFunction(torch.autograd.Function):
    """Autograd wrapper around the perceptual chain kernels."""

    @staticmethod
    def forward(ctx, ref, deg, threshs, exps, loud_scale, width_bark, total_width):
        batch, nframe, nband = ref.shape

        if nframe < _WINDOW:
            raise ValueError(
                f"The PESQ chain needs at least {_WINDOW} frames for the overlapping "
                f"sums, got {nframe}."
            )

        block_k = next_power_of_two(nband)
        nblock = triton.cdiv(nframe, _BLOCK_T)
        nwin = (nframe - _WINDOW) // _STRIDE + 1

        opts = dict(device=ref.device, dtype=torch.float32)
        pref = torch.empty(batch, nblock, nband, **opts)
        pdeg = torch.empty(batch, nblock, nband, **opts)
        mref = torch.empty(batch, nband, **opts)
        ratio = torch.empty(batch, nband, **opts)
        r = torch.empty(batch, nband, **opts)
        taer = torch.empty(batch, nframe, **opts)
        tad = torch.empty(batch, nframe, **opts)
        fpr0 = torch.empty(batch, nframe, **opts)
        fpr1 = torch.empty(batch, nframe, **opts)
        h = torch.empty(batch, nframe, **opts)
        sa = torch.empty(2, batch, nframe, **opts)
        saraw = torch.empty(2, batch, nframe, **opts)
        psqm = torch.empty(2, batch, nwin, **opts)
        dist = torch.empty(2, batch, **opts)

        _partial_pow_kernel[(batch, nblock)](
            ref,
            deg,
            threshs,
            pref,
            pdeg,
            nframe,
            nband,
            nblock,
            BLOCK_T=_BLOCK_T,
            BLOCK_K=block_k,
        )
        _band_ratio_kernel[(batch,)](
            pref,
            pdeg,
            mref,
            ratio,
            r,
            float(nframe),
            nband,
            nblock,
            BLOCK_K=block_k,
        )
        _frame_ratio_kernel[(batch, nblock)](
            ref,
            deg,
            r,
            threshs,
            taer,
            tad,
            fpr0,
            h,
            nframe,
            nband,
            BLOCK_T=_BLOCK_T,
            BLOCK_K=block_k,
        )
        _fir_kernel[(batch, triton.cdiv(nframe, _BLOCK_T_SMALL))](
            fpr0, fpr1, nframe, BLOCK_T=_BLOCK_T_SMALL
        )
        _distortion_kernel[(batch, nblock)](
            ref,
            deg,
            r,
            fpr1,
            h,
            threshs,
            exps,
            loud_scale,
            width_bark,
            sa,
            saraw,
            nframe,
            nband,
            batch * nframe,
            total_width,
            math.sqrt(total_width),
            BLOCK_T=_BLOCK_T,
            BLOCK_K=block_k,
        )
        _psqm_kernel[(2 * batch, triton.cdiv(nwin, _BLOCK_L))](
            sa, psqm, nframe, nwin, BLOCK_L=_BLOCK_L, BLOCK_J=32
        )
        _reduce_dist_kernel[(2 * batch,)](psqm, dist, nwin, float(nwin), BLOCK_L=256)

        ctx.save_for_backward(
            ref,
            deg,
            threshs,
            exps,
            loud_scale,
            width_bark,
            r,
            ratio,
            mref,
            taer,
            tad,
            fpr0,
            fpr1,
            h,
            sa,
            saraw,
            psqm,
            dist,
        )
        ctx.shape = (batch, nframe, nband, block_k, nblock, nwin)
        ctx.total_width = total_width

        return dist

    @staticmethod
    def backward(ctx, grad_dist):
        (
            ref,
            deg,
            threshs,
            exps,
            loud_scale,
            width_bark,
            r,
            ratio,
            mref,
            taer,
            tad,
            fpr0,
            fpr1,
            h,
            sa,
            saraw,
            psqm,
            dist,
        ) = ctx.saved_tensors
        batch, nframe, nband, block_k, nblock, nwin = ctx.shape
        total_width = ctx.total_width

        if grad_dist is None:
            return (None,) * 7
        grad_dist = grad_dist.contiguous()

        opts = dict(device=ref.device, dtype=torch.float32)
        gsa = torch.empty(2, batch, nframe, **opts)
        gref = torch.empty(batch, nframe, nband, **opts)
        gdeg = torch.empty(batch, nframe, nband, **opts)
        gfpr = torch.empty(batch, nframe, **opts)
        gh = torch.empty(batch, nframe, **opts)
        gtaer = torch.empty(batch, nframe, **opts)
        gtad = torch.empty(batch, nframe, **opts)
        gp1 = torch.empty(batch, nblock, nband, **opts)
        gp2 = torch.empty(batch, nblock, nband, **opts)
        gmref = torch.empty(batch, nband, **opts)
        gmdeg = torch.empty(batch, nband, **opts)

        _psqm_bwd_kernel[(2 * batch, triton.cdiv(nframe, _BLOCK_T_SMALL))](
            sa,
            psqm,
            dist,
            grad_dist,
            gsa,
            nframe,
            nwin,
            float(nwin),
            BLOCK_T=_BLOCK_T_SMALL,
        )
        _distortion_bwd_kernel[(batch, nblock)](
            ref,
            deg,
            r,
            fpr1,
            h,
            saraw,
            gsa,
            threshs,
            exps,
            loud_scale,
            width_bark,
            gref,
            gdeg,
            gfpr,
            gh,
            gp1,
            nframe,
            nband,
            nblock,
            batch * nframe,
            total_width,
            BLOCK_T=_BLOCK_T,
            BLOCK_K=block_k,
        )
        _frame_adjoint_kernel[(batch, triton.cdiv(nframe, _BLOCK_T_SMALL))](
            gfpr,
            fpr1,
            gh,
            h,
            taer,
            tad,
            fpr0,
            gtaer,
            gtad,
            nframe,
            BLOCK_T=_BLOCK_T_SMALL,
        )
        _audible_bwd_kernel[(batch, nblock)](
            ref,
            deg,
            r,
            threshs,
            gtaer,
            gtad,
            gref,
            gdeg,
            gp2,
            nframe,
            nband,
            nblock,
            BLOCK_T=_BLOCK_T,
            BLOCK_K=block_k,
        )
        _band_ratio_bwd_kernel[(batch,)](
            gp1, gp2, ratio, mref, gmref, gmdeg, nband, nblock, BLOCK_K=block_k
        )
        _mean_pow_bwd_kernel[(batch, nblock)](
            ref,
            deg,
            threshs,
            gmref,
            gmdeg,
            gref,
            gdeg,
            nframe,
            nband,
            float(nframe),
            BLOCK_T=_BLOCK_T,
            BLOCK_K=block_k,
        )

        if not ctx.needs_input_grad[0]:
            gref = None
        if not ctx.needs_input_grad[1]:
            gdeg = None

        return gref, gdeg, None, None, None, None, None


class TritonPesqChain(torch.nn.Module):
    """Symmetric and asymmetric PESQ distance of two Bark spectrograms.

    Triton implementation of :func:`torch_pesq.triton_ops.reference.ref_chain`,
    covering the silent frame detection, the band and frame power equalisation,
    the Zwicker loudness, the disturbance with its deadzone, both width weighted
    norms and the overlapping sums.

    Parameters
    ----------
    threshs : torch.Tensor
        Hearing thresholds, any shape broadcastable to ``[bark]``
    exp : torch.Tensor
        Loudness exponents with shape ``[bark]``
    width_bark : torch.Tensor
        Band widths in Bark with shape ``[bark]``
    total_width : float
        Sum of all band widths, excluding the first band
    sl : float
        Loudness calibration constant

    Attributes
    ----------
    threshs : torch.Tensor
        Hearing thresholds as a float32 buffer with shape ``[bark]``
    exp : torch.Tensor
        Loudness exponents as a float32 buffer with shape ``[bark]``
    loud_scale : torch.Tensor
        Precomputed ``sl * (2 threshs) ** exp`` with shape ``[bark]``
    width_bark : torch.Tensor
        Band widths as a float32 buffer with shape ``[bark]``
    """

    def __init__(self, threshs, exp, width_bark, total_width, sl=0.1866055):
        super(TritonPesqChain, self).__init__()

        threshs = torch.as_tensor(threshs).reshape(-1).double()
        exps = torch.as_tensor(exp).reshape(-1).double()
        widths = torch.as_tensor(width_bark).reshape(-1).double()

        if not (threshs.shape == exps.shape == widths.shape):
            raise ValueError(
                "`threshs`, `exp` and `width_bark` need the same number of bands, got "
                f"{threshs.numel()}, {exps.numel()} and {widths.numel()}."
            )
        if torch.any(threshs <= 0):
            raise ValueError("Hearing thresholds have to be strictly positive.")

        self.total_width = float(total_width)
        self.sl = float(sl)

        self.register_buffer("threshs", threshs.float())
        self.register_buffer("exp", exps.float())
        self.register_buffer("width_bark", widths.float())
        self.register_buffer("loud_scale", (self.sl * (2.0 * threshs) ** exps).float())

    def forward(self, ref_bark, deg_bark):
        """Symmetric and asymmetric distance of two Bark spectrograms.

        Parameters
        ----------
        ref_bark : torch.Tensor
            Bark spectrogram of the reference with shape ``[batch, frame, bark]``
        deg_bark : torch.Tensor
            Bark spectrogram of the degraded signal, same shape

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Symmetric and asymmetric distance, both with shape ``[batch]``
        """

        require_triton()

        ref = check_input(ref_bark, "ref_bark")
        deg = check_input(deg_bark, "deg_bark")

        if ref.dim() != 3 or deg.dim() != 3:
            raise ValueError(
                "The PESQ chain expects `[batch, frame, bark]` inputs, got "
                f"{tuple(ref.shape)} and {tuple(deg.shape)}."
            )
        if ref.shape != deg.shape:
            raise ValueError(
                "`ref_bark` and `deg_bark` need the same shape, got "
                f"{tuple(ref.shape)} and {tuple(deg.shape)}."
            )
        if ref.shape[2] != self.threshs.shape[0]:
            raise ValueError(
                f"Expected {self.threshs.shape[0]} Bark bands, got {ref.shape[2]}."
            )

        dist = _ChainFunction.apply(
            ref,
            deg,
            self.threshs,
            self.exp,
            self.loud_scale,
            self.width_bark,
            self.total_width,
        )

        return dist[0], dist[1]
