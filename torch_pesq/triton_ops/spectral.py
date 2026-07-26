"""Fused window -> real DFT -> power -> Bark band aggregation in Triton.

The module in here replaces the ``torch.stft`` / power / Bark filterbank part of
:meth:`torch_pesq.loss.PesqLoss.raw` by two Triton kernels for the forward and
two for the backward pass. The oracle it reproduces is
:func:`torch_pesq.triton_ops.reference.ref_stft_bark`.

Structure that is exploited
---------------------------

The Bark filterbank produced by :class:`torch_pesq.bark.BarkScale` is *binary*,
*contiguous* and *non overlapping*: band ``i`` covers the half open bin range
``[start_i, end_i)`` with ``start_i >= end_{i - 1}``. Bands may be empty and the
upper bins may stay uncovered when ``nbarks < 49``. Consequently

  * the ``[batch, frame, n_fft // 2 + 1]`` power spectrogram is never written to
    global memory, the band reduction happens on the register tile,
  * bin ``0`` is dropped (the energy feature is unused) and bin ``n_fft // 2`` is
    never touched, so only bins ``1 .. max(end_i) - 1`` are evaluated at all,
  * everything above ``max(end_i)`` is skipped, which saves the top bins for
    ``nbarks < 49``.

DFT variants that were benchmarked
----------------------------------

All numbers are for ``batch = 8``, ``128000`` samples, ``n_fft = 512``,
``nbarks = 49`` on a GTX 1080 Ti; ``rel2peak`` is the largest absolute deviation
from a float64 evaluation divided by the peak of the spectrogram.

=================================== ========= ==========
variant                             time      rel2peak
=================================== ========= ==========
dense matmul over all ``n_fft`` taps 0.406 ms  3.9e-07
inline ``tl.math.cos`` / ``sin``     0.504 ms  9.4e-06
real input fold (**used**)           0.316 ms  2.0e-07
=================================== ========= ==========

The fold uses ``a_k = u_k + u_{n - k}`` and ``b_k = u_k - u_{n - k}`` so that
``Re_f = sum_{k < n/2} a_k cos(2 pi f k / n) + u_{n/2} (-1)^f`` and
``Im_f = -sum_{k < n/2} b_k sin(2 pi f k / n)``, which is exactly half the MAC
count and half the twiddle memory of the dense variant. Computing the twiddles
inline is both slower and two decimal digits less accurate, because the phase
``2 pi f k / n`` reaches ~800 rad and the float32 argument reduction of
``tl.math.cos`` loses most of the mantissa there.

The band reduction is a ``tl.dot`` against the binary filterbank tile, which is
built from the host side segment boundaries. A literal segment sum -- one masked
``tl.sum`` per band followed by a separate store -- was implemented and measured
too: it does fewer flops (``sum_i width_i = n_fft / 2`` instead of
``n_fft / 2 * nbarks``) and is equally accurate (rel2peak 2.0e-07), but the
``nbarks`` uncoalesced ``[BLOCK_T]`` stores make it slower (0.397 ms) and the
``nbarks``-way unroll pushes the compile time to 3.9 s. The segment boundaries
are still validated at construction time and used to build the filterbank tile,
to clip the evaluated bin range and, transposed, as the bin to band gather of
the backward pass.

Backward
--------

With ``u_k = w_k x_k``, ``P_f = Re_f^2 + Im_f^2`` the VJP is

``dL/du_k = 2 sum_f gP_f (Re_f cos(theta_fk) - Im_f sin(theta_fk))``

where ``gP_f`` is the band gradient broadcast over the band's bin range times
the power density correction, and ``gP_0 = 0``. The same ``k <-> n - k`` fold is
used, ``dL/du_k`` and ``dL/du_{n-k}`` differ only in the sign of the sine term.
The per frame gradients are written to a ``[batch, frame, n_fft]`` scratch
buffer and a second kernel *gathers* them into ``dL/dx`` -- every sample looks
up the at most ``ceil(n_fft / hop_length)`` frames that contain it. No atomics,
so the result is bitwise deterministic.

``Re`` and ``Im`` are saved by the forward pass rather than recomputed. Storing
them costs 0.020 ms of the 0.317 ms forward, while recomputing the DFT would add
the full ~0.30 ms to the 0.46 ms backward.

Accuracy
--------

A direct DFT accumulates ``O(sqrt(n_fft))`` rounding error where an FFT
accumulates ``O(sqrt(log n_fft))``, so per element relative errors on the
smallest band powers reach ~1e-4 in the worst configurations. The block sizes in
:func:`_fwd_config` were picked for accuracy as much as for speed: with
``BLOCK_K = 16`` and eight warps the deviation from a float64 evaluation stays
at ~2e-7 of the spectrogram peak, which is *better* than what
``torch.stft`` in float32 achieves on the same data. The meaningful tolerance
for both this op and the oracle is therefore an absolute one scaled by the peak
of the output, not a per element relative one -- the same holds for the
gradient, where the sum over frequencies cancels almost completely for samples
in the ramped edges of the window.

The peak the tolerance is scaled by is the peak of the *frame*, not of the whole
spectrogram: the rounding error of bin ``f`` of frame ``t`` is proportional to
the energy of that frame, so a globally scaled tolerance would be blind to a
defect that only shows up in the quiet part of a signal.

Limitations
-----------

* ``n_fft`` has to be a **power of two** and at least ``32``. Both kernels tile
  the folded reduction with a single ``tl.arange(0, n_fft // 2)``, which Triton
  only accepts for power of two lengths, and the forward ``k`` loop steps in
  blocks of ``16`` without a tail mask. A non power of two ``n_fft`` used to
  either read the twiddle tables out of bounds -- silently wrong by tens of
  percent -- or fail with a raw Triton ``CompilationError``; it is rejected at
  construction time now.
* All three kernels put the long grid axis on ``program_id(0)``, because CUDA
  caps grid dimensions ``y`` and ``z`` at ``65535``. The batch sits on axis
  ``1``, so batches beyond ``65535`` are unsupported while the number of frames
  and samples is not limited by the launch geometry.
* Every pointer offset is 32 bit, so a single tensor may not exceed ``2 ** 31``
  elements, i.e. 8.6 GB in float32. That is far beyond what fits next to the
  ``[batch, frame, n_fft]`` scratch buffer of the backward pass anyway.
"""

import math

import numpy as np
import torch

from ._common import HAS_TRITON, check_input, next_power_of_two, require_triton

if HAS_TRITON:  # pragma: no cover - depends on the installation
    import triton
    import triton.language as tl
else:  # pragma: no cover - depends on the installation

    class _Dummy:
        def jit(self, fn):
            return fn

    triton = _Dummy()
    tl = None


__all__ = ["TritonStftBark", "bark_segments"]


def bark_segments(fbank: torch.Tensor):
    """Segment boundaries of a binary, contiguous, non overlapping filterbank.

    Parameters
    ----------
    fbank : torch.Tensor
        Filterbank matrix with shape ``[bark, n_fft // 2]``

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Start and end bin of every band, ``[bark]`` each. Empty bands get
        ``start == end``.

    Raises
    ------
    ValueError
        If the filterbank is not binary, not contiguous or overlapping.
    """

    mat = fbank.detach().cpu().numpy()

    if mat.ndim != 2:
        raise ValueError(f"The filterbank has to be 2d, got {mat.ndim} dimensions.")
    if not np.all((mat == 0.0) | (mat == 1.0)):
        raise ValueError(
            "The Triton spectral op needs a binary filterbank, got values "
            f"outside {{0, 1}} (min {mat.min()}, max {mat.max()})."
        )

    nbarks, nfreqs = mat.shape
    starts = np.zeros(nbarks, dtype=np.int64)
    ends = np.zeros(nbarks, dtype=np.int64)

    prev = 0
    for band in range(nbarks):
        nonzero = np.flatnonzero(mat[band])

        if nonzero.size == 0:
            starts[band], ends[band] = prev, prev
            continue

        start, end = int(nonzero[0]), int(nonzero[-1]) + 1

        if end - start != nonzero.size:
            raise ValueError(
                f"Band {band} of the filterbank is not contiguous, it covers "
                f"{nonzero.size} bins spread over [{start}, {end})."
            )
        if start < prev:
            raise ValueError(
                f"Band {band} of the filterbank starts at bin {start} but the "
                f"previous band ends at {prev}, the bands have to be disjoint "
                "and sorted."
            )

        starts[band], ends[band] = start, end
        prev = end

    if prev > nfreqs:
        raise ValueError(
            f"The filterbank covers bin {prev - 1} but only has {nfreqs} bins."
        )

    return starts, ends


def _twiddles(n_fft: int, nfreqs: int):
    """Folded cosine and sine tables in float64.

    Returns ``cos[k, f]`` and ``sin[k, f]`` for ``k < n_fft // 2`` and
    ``f < nfreqs``. Column ``0`` is zeroed so that the DC bin drops out of both
    the forward and the backward pass.
    """

    half = n_fft // 2
    taps = np.arange(half, dtype=np.int64)[:, None]
    freqs = np.arange(nfreqs, dtype=np.int64)[None, :]

    # reduce the argument exactly before going to floating point
    phase = (taps * freqs) % n_fft * (2.0 * math.pi / n_fft)

    cos, sin = np.cos(phase), np.sin(phase)
    cos[:, 0] = 0.0
    sin[:, 0] = 0.0

    return cos, sin


if HAS_TRITON:  # pragma: no cover - depends on the installation

    @triton.jit
    def _fwd_kernel(
        X,
        W,
        COS,
        SIN,
        SGN,
        FBT,
        CORR,
        OUT,
        RE,
        IM,
        nsamples,
        nframes,
        NFFT: tl.constexpr,
        HALF: tl.constexpr,
        HOP: tl.constexpr,
        NFREQ: tl.constexpr,
        NBARK: tl.constexpr,
        KPAD: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_F: tl.constexpr,
        SAVE_RI: tl.constexpr,
    ):
        """Window, folded real DFT, power and Bark segment sum for a frame tile."""

        # the frame tile is the *first* grid axis: it is the one that can grow
        # without bound and CUDA caps grid dimensions y and z at 65535
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)

        frame = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        fmask = frame < nframes
        base = frame * HOP

        xptr = X + pid_b * nsamples
        bacc = tl.zeros((BLOCK_T, KPAD), dtype=tl.float32)

        # the k == HALF tap has no partner under the k <-> n - k fold
        jmid = base + HALF
        umid = tl.load(xptr + jmid, mask=fmask & (jmid < nsamples), other=0.0)
        umid = umid * tl.load(W + HALF)

        for f0 in range(0, NFREQ, BLOCK_F):
            foff = f0 + tl.arange(0, BLOCK_F)
            fvalid = foff < NFREQ

            re = tl.zeros((BLOCK_T, BLOCK_F), dtype=tl.float32)
            im = tl.zeros((BLOCK_T, BLOCK_F), dtype=tl.float32)

            for k0 in range(0, HALF, BLOCK_K):
                koff = k0 + tl.arange(0, BLOCK_K)
                upper = koff > 0

                jlo = base[:, None] + koff[None, :]
                ulo = tl.load(
                    xptr + jlo, mask=fmask[:, None] & (jlo < nsamples), other=0.0
                )
                ulo = ulo * tl.load(W + koff)[None, :]

                jhi = base[:, None] + (NFFT - koff)[None, :]
                whi = tl.load(W + (NFFT - koff), mask=upper, other=0.0)
                uhi = tl.load(
                    xptr + jhi,
                    mask=fmask[:, None] & upper[None, :] & (jhi < nsamples),
                    other=0.0,
                )
                uhi = uhi * whi[None, :]

                tptr = koff[:, None] * NFREQ + foff[None, :]
                tmask = fvalid[None, :]
                cos = tl.load(COS + tptr, mask=tmask, other=0.0)
                sin = tl.load(SIN + tptr, mask=tmask, other=0.0)

                re += tl.dot(ulo + uhi, cos, allow_tf32=False)
                im -= tl.dot(ulo - uhi, sin, allow_tf32=False)

            re += umid[:, None] * tl.load(SGN + foff, mask=fvalid, other=0.0)[None, :]

            if SAVE_RI:
                riptr = (pid_b * nframes + frame[:, None]) * NFREQ + foff[None, :]
                rimask = fmask[:, None] & fvalid[None, :]
                tl.store(RE + riptr, re, mask=rimask)
                tl.store(IM + riptr, im, mask=rimask)

            power = re * re + im * im

            kpad = tl.arange(0, KPAD)
            fbt = tl.load(
                FBT + foff[:, None] * KPAD + kpad[None, :],
                mask=fvalid[:, None],
                other=0.0,
            )
            bacc += tl.dot(power, fbt, allow_tf32=False)

        kpad = tl.arange(0, KPAD)
        bacc = bacc * tl.load(CORR + kpad)[None, :]

        tl.store(
            OUT + (pid_b * nframes + frame[:, None]) * NBARK + kpad[None, :],
            bacc,
            mask=fmask[:, None] & (kpad < NBARK)[None, :],
        )

    @triton.jit
    def _bwd_frame_kernel(
        GB,
        RE,
        IM,
        COSB,
        SINB,
        SGN,
        BANDID,
        CORRBIN,
        GU,
        nframes,
        NFFT: tl.constexpr,
        HALF: tl.constexpr,
        NFREQ: tl.constexpr,
        NBARK: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        """Per frame gradient with respect to the windowed samples ``u_k``."""

        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)

        frame = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        fmask = frame < nframes
        gbbase = (pid_b * nframes + frame[:, None]) * NBARK
        ribase = (pid_b * nframes + frame[:, None]) * NFREQ

        koff = tl.arange(0, HALF)
        csum = tl.zeros((BLOCK_T, HALF), dtype=tl.float32)
        ssum = tl.zeros((BLOCK_T, HALF), dtype=tl.float32)
        gmid = tl.zeros((BLOCK_T,), dtype=tl.float32)

        for f0 in range(0, NFREQ, BLOCK_F):
            foff = f0 + tl.arange(0, BLOCK_F)

            # the band gradient is broadcast back over the bins of its segment,
            # which is a plain gather with the precomputed bin -> band map
            band = tl.load(BANDID + foff)
            gpow = tl.load(
                GB + gbbase + band[None, :],
                mask=fmask[:, None] & (band >= 0)[None, :],
                other=0.0,
            )
            gpow = gpow * tl.load(CORRBIN + foff)[None, :]

            riptr = ribase + foff[None, :]
            gre = gpow * tl.load(RE + riptr, mask=fmask[:, None], other=0.0)
            gim = gpow * tl.load(IM + riptr, mask=fmask[:, None], other=0.0)

            tptr = foff[:, None] * HALF + koff[None, :]
            csum += tl.dot(gre, tl.load(COSB + tptr), allow_tf32=False)
            ssum += tl.dot(gim, tl.load(SINB + tptr), allow_tf32=False)

            # the unpaired k == HALF tap, cos(pi f) == (-1) ** f
            gmid += tl.sum(gre * tl.load(SGN + foff)[None, :], axis=1)

        gurow = (pid_b * nframes + frame) * NFFT
        gubase = gurow[:, None]

        tl.store(GU + gubase + koff[None, :], 2.0 * (csum - ssum), mask=fmask[:, None])
        tl.store(GU + gurow + HALF, 2.0 * gmid, mask=fmask)

        # the mirrored taps n - k for k = 1 .. HALF - 1, flipped into ascending
        # order so that the store stays coalesced
        tl.store(
            GU + gubase + (HALF + 1) + koff[None, :],
            tl.flip(2.0 * (csum + ssum), 1),
            mask=fmask[:, None] & (koff < HALF - 1)[None, :],
        )

    @triton.jit
    def _bwd_gather_kernel(
        GU,
        W,
        DX,
        nsamples,
        nframes,
        NFFT: tl.constexpr,
        HOP: tl.constexpr,
        NOVER: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Gather the per frame gradients back onto the input samples."""

        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)

        sample = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        smask = sample < nsamples

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        last = sample // HOP

        for over in range(NOVER):
            frame = last - over
            tap = sample - frame * HOP
            valid = smask & (frame >= 0) & (frame < nframes) & (tap < NFFT)
            safe = tl.maximum(frame, 0)

            grad = tl.load(
                GU + (pid_b * nframes + safe) * NFFT + tap, mask=valid, other=0.0
            )
            acc += grad * tl.load(W + tap, mask=valid, other=0.0)

        tl.store(DX + pid_b * nsamples + sample, acc, mask=smask)


def _fwd_config(nfreq: int, kpad: int):
    """Block sizes for the forward kernel, hand tuned on a GTX 1080 Ti.

    ``BLOCK_K = 16`` with eight warps is not only the fastest configuration but
    also by far the most accurate one: Triton spreads the ``tl.dot`` reduction
    over more threads, which turns the accumulation into a wider tree. Larger
    reduction blocks were measured to lose up to 1.5 decimal digits.

    ``BLOCK_F`` also drives the shared memory footprint of the band reduction,
    whose right hand operand is a ``[BLOCK_F, kpad]`` tile. Pascal offers 48 KiB
    of shared memory per block, so ``BLOCK_F`` is capped at ``8192 // kpad``;
    without that cap a filterbank with more than 64 bands fails to launch with
    ``OutOfResources``. For the 49 band PESQ filterbank the cap is inactive.
    """

    return 16, 16, min(nfreq, 128, max(16, 8192 // kpad)), 8, 1


def _bwd_config(nfreq: int):
    """Block sizes for the frame gradient kernel, hand tuned on a GTX 1080 Ti."""

    return 16, min(nfreq, 16), 8, 1


class _StftBarkFunction(torch.autograd.Function):
    """Autograd wrapper around the fused STFT / Bark kernels."""

    @staticmethod
    def forward(ctx, signal, module, n_padded):
        nbatch, nsamples = signal.shape
        nframes = 1 + (n_padded - module.n_fft) // module.hop_length

        out = torch.empty(
            (nbatch, nframes, module.nbarks), device=signal.device, dtype=torch.float32
        )

        needs_grad = ctx.needs_input_grad[0]
        shape = (nbatch, nframes, module.nfreq) if needs_grad else (1,)
        re = torch.empty(shape, device=signal.device, dtype=torch.float32)
        im = torch.empty(shape, device=signal.device, dtype=torch.float32)

        block_t, block_k, block_f, warps, stages = _fwd_config(
            module.nfreq, module.kpad
        )
        grid = (triton.cdiv(nframes, block_t), nbatch)

        _fwd_kernel[grid](
            signal,
            module.window,
            module.cos_fwd,
            module.sin_fwd,
            module.sgn,
            module.fbank_fwd,
            module.correction,
            out,
            re,
            im,
            nsamples,
            nframes,
            NFFT=module.n_fft,
            HALF=module.n_fft // 2,
            HOP=module.hop_length,
            NFREQ=module.nfreq,
            NBARK=module.nbarks,
            KPAD=module.kpad,
            BLOCK_T=block_t,
            BLOCK_K=block_k,
            BLOCK_F=block_f,
            SAVE_RI=needs_grad,
            num_warps=warps,
            num_stages=stages,
        )

        ctx.module = module
        ctx.nsamples = nsamples
        ctx.nframes = nframes
        if needs_grad:
            ctx.save_for_backward(re, im)

        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        module = ctx.module
        re, im = ctx.saved_tensors

        grad_out = grad_out.contiguous().float()
        nbatch = grad_out.shape[0]
        nframes, nsamples = ctx.nframes, ctx.nsamples

        gu = torch.empty(
            (nbatch, nframes, module.n_fft),
            device=grad_out.device,
            dtype=torch.float32,
        )

        block_t, block_f, warps, stages = _bwd_config(module.nfreq)

        _bwd_frame_kernel[(triton.cdiv(nframes, block_t), nbatch)](
            grad_out,
            re,
            im,
            module.cos_bwd,
            module.sin_bwd,
            module.sgn,
            module.bandid,
            module.corrbin,
            gu,
            nframes,
            NFFT=module.n_fft,
            HALF=module.n_fft // 2,
            NFREQ=module.nfreq,
            NBARK=module.nbarks,
            BLOCK_T=block_t,
            BLOCK_F=block_f,
            num_warps=warps,
            num_stages=stages,
        )

        grad_in = torch.empty(
            (nbatch, nsamples), device=grad_out.device, dtype=torch.float32
        )
        block_n = 256

        _bwd_gather_kernel[(triton.cdiv(nsamples, block_n), nbatch)](
            gu,
            module.window,
            grad_in,
            nsamples,
            nframes,
            NFFT=module.n_fft,
            HOP=module.hop_length,
            NOVER=module.novertaps,
            BLOCK_N=block_n,
            num_warps=4,
        )

        return grad_in, None, None


class TritonStftBark(torch.nn.Module):
    """Fused STFT, power spectrum and Bark band aggregation.

    Numerically equivalent to
    :func:`torch_pesq.triton_ops.reference.ref_stft_bark`, i.e.
    ``torch.stft(center=False)`` followed by ``|X| ** 2``, dropping the DC bin,
    a contraction with the binary Bark filterbank over bins
    ``0 .. n_fft // 2 - 1`` and a per band power density correction.

    The reference returns ``float64`` because
    :attr:`torch_pesq.bark.BarkScale.pow_dens_correction` is a ``float64``
    buffer; this module keeps everything in ``float32`` as required by the
    Triton backend contract.

    Parameters
    ----------
    window : torch.Tensor
        Analysis window with ``win_length <= n_fft`` taps, centred into an
        ``n_fft`` long window exactly like ``torch.stft`` does
    fbank : torch.Tensor
        Binary Bark filterbank with shape ``[bark, n_fft // 2]``
    pow_dens_correction : torch.Tensor
        Per band power density correction with shape ``[bark]``
    n_fft : int
        Number of frequency bins, a power of two of at least ``32``
    hop_length : int
        Distance between frames

    Attributes
    ----------
    window : torch.Tensor
        Zero padded analysis window with ``n_fft`` taps
    cos_fwd, sin_fwd : torch.Tensor
        Folded twiddle tables with shape ``[n_fft // 2, nfreq]``
    cos_bwd, sin_bwd : torch.Tensor
        Transposed twiddle tables with shape ``[nfreq, n_fft // 2]``
    sgn : torch.Tensor
        ``(-1) ** f`` with a zeroed DC entry, shape ``[nfreq]``
    fbank_fwd : torch.Tensor
        Binary segment scatter matrix with shape ``[nfreq, kpad]``
    correction : torch.Tensor
        Zero padded power density correction with shape ``[kpad]``
    bandid : torch.Tensor
        Bin to band map with shape ``[nfreq]``, ``-1`` for uncovered bins
    corrbin : torch.Tensor
        Power density correction of the band owning a bin, shape ``[nfreq]``
    starts, ends : torch.Tensor
        Segment boundaries of every Bark band, shape ``[bark]`` each
    """

    def __init__(
        self,
        window: torch.Tensor,
        fbank: torch.Tensor,
        pow_dens_correction: torch.Tensor,
        n_fft: int = 512,
        hop_length: int = 256,
    ):
        super(TritonStftBark, self).__init__()

        require_triton()

        if n_fft % 2 != 0:
            raise ValueError(f"`n_fft` has to be even, got {n_fft}.")
        if n_fft & (n_fft - 1) != 0:
            # both kernels tile the folded reduction with a single
            # `tl.arange(0, n_fft // 2)` and step the forward k loop in blocks
            # of 16 without a tail mask; either is silently wrong (the twiddle
            # tables get read past their last row) when n_fft // 2 is not a
            # power of two, so refuse the configuration instead
            raise ValueError(f"`n_fft` has to be a power of two, got {n_fft}.")
        if n_fft < 32:
            # tl.dot needs every dimension to be at least 16 and the folded
            # reduction runs over n_fft // 2 taps
            raise ValueError(f"`n_fft` has to be at least 32, got {n_fft}.")
        if hop_length <= 0:
            raise ValueError(f"`hop_length` has to be positive, got {hop_length}.")
        if window.ndim != 1 or window.shape[0] > n_fft:
            raise ValueError(
                f"The window has to be 1d with at most {n_fft} taps, got "
                f"{tuple(window.shape)}."
            )
        if fbank.shape[1] != n_fft // 2:
            raise ValueError(
                f"The filterbank needs {n_fft // 2} bins, got {fbank.shape[1]}."
            )
        if pow_dens_correction.shape[0] != fbank.shape[0]:
            raise ValueError(
                "The power density correction needs one entry per band, got "
                f"{pow_dens_correction.shape[0]} for {fbank.shape[0]} bands."
            )

        starts, ends = bark_segments(fbank)

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.nbarks = int(fbank.shape[0])
        self.kpad = max(16, next_power_of_two(self.nbarks))
        self.novertaps = -(-self.n_fft // self.hop_length)

        # only bins below the last covered one contribute, bin 0 is dropped by
        # the zeroed twiddle column and bin n_fft // 2 is never evaluated
        covered = int(ends.max()) if len(ends) else 0
        self.nfreq = min(n_fft // 2, max(16, next_power_of_two(covered)))

        # centre the window into an n_fft long buffer, matching torch.stft
        full = torch.zeros(n_fft, dtype=torch.float32)
        left = (n_fft - window.shape[0]) // 2
        full[left : left + window.shape[0]] = window.detach().cpu().float()

        cos, sin = _twiddles(self.n_fft, self.nfreq)

        sgn = np.ones(self.nfreq)
        sgn[1::2] = -1.0
        sgn[0] = 0.0

        corr = np.zeros(self.kpad)
        corr[: self.nbarks] = pow_dens_correction.detach().cpu().double().numpy()

        # segment sum: [nfreq, kpad] scatter matrix for the forward pass and the
        # bin -> band map plus per bin correction for the backward broadcast
        dense = np.zeros((self.kpad, self.nfreq))
        bandid = np.full(self.nfreq, -1, dtype=np.int32)
        corrbin = np.zeros(self.nfreq)

        for band in range(self.nbarks):
            lo, hi = int(starts[band]), min(int(ends[band]), self.nfreq)
            dense[band, lo:hi] = 1.0
            bandid[lo:hi] = band
            corrbin[lo:hi] = corr[band]

        # bin 0 carries no gradient, its twiddles are zero anyway
        bandid[0], corrbin[0] = -1, 0.0

        def _buf(array):
            return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)

        self.register_buffer("window", full)
        self.register_buffer("cos_fwd", _buf(cos))
        self.register_buffer("sin_fwd", _buf(sin))
        self.register_buffer("cos_bwd", _buf(cos.T))
        self.register_buffer("sin_bwd", _buf(sin.T))
        self.register_buffer("sgn", _buf(sgn))
        self.register_buffer("fbank_fwd", _buf(dense.T))
        self.register_buffer("correction", _buf(corr))
        self.register_buffer("corrbin", _buf(corrbin))
        self.register_buffer(
            "bandid", torch.as_tensor(np.ascontiguousarray(bandid), dtype=torch.int32)
        )
        self.register_buffer("starts", torch.as_tensor(starts))
        self.register_buffer("ends", torch.as_tensor(ends))

    def forward(self, signal: torch.Tensor, n_padded: int = None) -> torch.Tensor:
        """Bark scaled power spectrogram of a time signal.

        Parameters
        ----------
        signal : torch.Tensor
            Time signal with shape ``[batch, sample]``
        n_padded : int
            Number of samples the frames are laid out over. Samples beyond
            ``signal.shape[1]`` read as zero, which folds the trailing zero
            padding of the PESQ pipeline into this op. Defaults to
            ``signal.shape[1]``.

        Returns
        -------
        torch.Tensor
            Bark scaled power spectrogram with shape ``[batch, frame, bark]``
        """

        signal = check_input(signal, "signal")

        if signal.ndim != 2:
            raise ValueError(
                f"`signal` has to be 2d, got {signal.ndim} dimensions instead."
            )

        n_padded = signal.shape[1] if n_padded is None else int(n_padded)

        if n_padded < signal.shape[1]:
            raise ValueError(
                f"`n_padded` ({n_padded}) is smaller than the signal length "
                f"({signal.shape[1]})."
            )
        if n_padded < self.n_fft:
            raise ValueError(
                f"`n_padded` ({n_padded}) is shorter than one frame ({self.n_fft})."
            )

        return _StftBarkFunction.apply(signal, self, n_padded)
