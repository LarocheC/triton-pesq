"""Triton implementation of the polyphase sinc resampler.

:class:`TritonResample` is a drop-in replacement for
:class:`torchaudio.transforms.Resample`. The band limited interpolation kernel
is *not* redesigned here -- it is taken verbatim from torchaudio's host side
``_get_sinc_resample_kernel`` and registered as a buffer. Only the strided
convolution which applies that kernel, and its adjoint, are Triton kernels.

torchaudio evaluates the resampler as

.. code-block:: python

    padded = pad(waveform, (width, width + orig_step))
    out = conv1d(padded[:, None], kernel, stride=orig_step)
    out = out.transpose(1, 2).reshape(batch, -1)[..., :target_length]

with ``orig_step = orig_freq // gcd`` and ``new_step = new_freq // gcd``.
Splitting the flat output index into ``m = frame * new_step + phase`` this is

.. math::

    y[\\mathrm{frame}, \\mathrm{phase}] = \\sum_k K[\\mathrm{phase}, k] \\;
        x[\\mathrm{frame} \\cdot \\mathrm{orig\\_step} + k - \\mathrm{width}]

with ``x`` read as zero outside ``[0, n\\_samples)``. Both kernels below are
written as *outer product* accumulations over that expression:

* the forward tiles ``(frame, phase)``, loads ``K[:, k]`` once per tile column
  and ``x[frame * orig_step + k - width]`` once per tile row and accumulates
  their outer product, so every tap is reused ``BLOCK_L`` times,
* the backward tiles ``(frame, residue)``. Input sample ``n`` is uniquely
  written as ``n + width = frame * orig_step + residue``; it is read by the
  outputs ``(frame - i, phase)`` through tap ``residue + i * orig_step``, which
  is again an outer product -- ``grad_out`` depends only on the row and ``K``
  only on the column.

The backward is therefore a deterministic gather over input samples: no
atomics, no run to run variation.

Launch geometry
---------------
The frame axis is the only unbounded one -- a one minute utterance at 48 kHz
already needs close to a million frames. CUDA limits ``gridDim.y`` and
``gridDim.z`` to 65535 blocks, so the frame axis *must* live on ``gridDim.x``
(limit ``2**31 - 1``). Batch and frame block are therefore folded into
``program_id(0)`` and split again inside the kernel; only the phase/residue
axis, which is bounded by the resampling ratio, is left on ``program_id(1)``.
"""

import math
from typing import Optional, Tuple

import torch
from torchaudio.functional.functional import _get_sinc_resample_kernel

from ._common import HAS_TRITON, check_input, next_power_of_two, require_triton

if HAS_TRITON:  # pragma: no cover - depends on the installation
    import triton
    import triton.language as tl
else:  # pragma: no cover - depends on the installation
    import types

    # Stubs which keep the kernel definitions below importable without Triton.
    # Launching them is prevented by ``require_triton``.
    triton = types.SimpleNamespace(
        jit=lambda fn: fn,
        autotune=lambda **kwargs: (lambda fn: fn),
        Config=lambda *args, **kwargs: None,
    )
    tl = types.SimpleNamespace(constexpr=int)


__all__ = ["TritonResample"]


# All indices inside the kernels are 32 bit: the sample index, the flat output
# index and the tap offsets. Every one of them is bounded by
# ``max(n_samples, n_out)`` plus a tile of slack, so a signal that stays below
# this many samples in *and* out can never wrap. Reaching the bound needs an
# 8.6 GB tensor, but a silent wrap would be far worse than a clear error, so
# the sizes are checked on the host instead of paying for 64 bit offsets in
# the inner loop.
_INDEX_LIMIT = 2**31 - 1


# How the outer axis of the tile is split into warps is the one choice that
# matters on Pascal: the tap loop is serial, so too few resident warps starves
# the load pipeline while a tile that does not divide cleanly into warps pays
# for a shared memory round trip on every broadcast. The landscape is not
# monotone in either knob, so the configurations are measured instead of
# guessed. Every configuration computes each output entirely inside one lane
# and in the same tap order, so the choice never changes the result -- only how
# long it takes.
_CONFIGS = [
    triton.Config({"BLOCK_L": block_l}, num_warps=warps)
    for block_l, warps in [
        (128, 1),
        (128, 2),
        (128, 4),
        (64, 1),
        (64, 2),
        (32, 1),
        (16, 4),
        (4, 2),
        (2, 2),
    ]
]

_TUNE_KEY = ["orig_step", "new_step", "width_total", "size_class"]


@triton.autotune(configs=_CONFIGS, key=_TUNE_KEY, warmup=1, rep=5)
@triton.jit
def _resample_fwd_kernel(
    x_ptr,
    kernel_t_ptr,
    out_ptr,
    n_samples,
    n_out,
    orig_step,
    new_step,
    width,
    width_total,
    n_frames,
    size_class,
    BLOCK_L: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """``y[l, j] = sum_k K[j, k] * x[l * orig_step + k - width]``.

    ``kernel_t`` is the ``[width_total, new_step]`` transpose of the filter
    bank, so the ``new_step`` taps of one ``k`` are contiguous and the load is
    coalesced along the phase axis of the tile.

    ``program_id(0)`` enumerates ``batch * ceil(n_frames / BLOCK_L)`` tiles,
    frame block fastest so that neighbouring blocks read neighbouring samples;
    see the module docstring for why the frame axis cannot sit on a higher
    grid dimension.
    """

    n_blocks_l = tl.cdiv(n_frames, BLOCK_L)

    pid = tl.program_id(0)
    pid_b = pid // n_blocks_l
    pid_l = pid % n_blocks_l
    pid_j = tl.program_id(1)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

    mask_l = offs_l < n_frames
    mask_j = offs_j < new_step

    x_base = x_ptr + pid_b.to(tl.int64) * n_samples
    start = offs_l * orig_step - width

    acc = tl.zeros([BLOCK_L, BLOCK_J], dtype=tl.float32)

    for k in range(0, width_total):
        taps = tl.load(kernel_t_ptr + k * new_step + offs_j, mask=mask_j, other=0.0)

        pos = start + k
        samples = tl.load(
            x_base + pos, mask=mask_l & (pos >= 0) & (pos < n_samples), other=0.0
        )

        acc += samples[:, None] * taps[None, :]

    idx_m = offs_l[:, None] * new_step + offs_j[None, :]
    tl.store(
        out_ptr + pid_b.to(tl.int64) * n_out + idx_m,
        acc,
        mask=mask_l[:, None] & mask_j[None, :] & (idx_m < n_out),
    )


@triton.autotune(configs=_CONFIGS, key=_TUNE_KEY, warmup=1, rep=5)
@triton.jit
def _resample_bwd_kernel(
    grad_out_ptr,
    kernel_ptr,
    grad_x_ptr,
    n_samples,
    n_out,
    orig_step,
    new_step,
    width,
    width_total,
    n_taps,
    n_frames,
    size_class,
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Adjoint of :func:`_resample_fwd_kernel`, gathered over input samples.

    Every input index is hit exactly once by the bijection
    ``n + width = frame * orig_step + residue`` with ``residue`` in
    ``[0, orig_step)``. Sample ``n`` is read by the outputs
    ``m = (frame - i) * new_step + j`` through tap ``residue + i * orig_step``,
    for ``i`` in ``[0, n_taps)`` -- all taps that can reach ``n`` are congruent
    to ``residue`` modulo ``orig_step``. ``grad_out`` only depends on ``frame``
    and the filter only on ``residue``, hence the outer product.

    The grid is folded exactly like the forward kernel.
    """

    n_blocks_l = tl.cdiv(n_frames, BLOCK_L)

    pid = tl.program_id(0)
    pid_b = pid // n_blocks_l
    pid_l = pid % n_blocks_l
    pid_r = tl.program_id(1)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < orig_step

    idx_n = offs_l[:, None] * orig_step + offs_r[None, :] - width
    keep = mask_r[None, :] & (idx_n >= 0) & (idx_n < n_samples)

    grad_base = grad_out_ptr + pid_b.to(tl.int64) * n_out
    acc = tl.zeros([BLOCK_L, BLOCK_R], dtype=tl.float32)

    for i in range(0, n_taps):
        tap = offs_r + i * orig_step
        mask_tap = mask_r & (tap < width_total)

        idx_l = offs_l - i
        mask_frame = idx_l >= 0

        for j in range(0, new_step):
            idx_m = idx_l * new_step + j

            grads = tl.load(
                grad_base + idx_m, mask=mask_frame & (idx_m < n_out), other=0.0
            )
            taps = tl.load(kernel_ptr + j * width_total + tap, mask=mask_tap, other=0.0)

            acc += grads[:, None] * taps[None, :]

    tl.store(grad_x_ptr + pid_b.to(tl.int64) * n_samples + idx_n, acc, mask=keep)


def _block_inner(inner: int) -> int:
    """Width of the contiguous tile axis.

    Capped at 32 so the tap vector of one loop iteration is a single coalesced
    transaction; a wider tile would only add masked lanes for the rates that
    matter (``new_step`` is 1, 2, 160 or 320 for the usual PESQ inputs).
    """

    return min(32, next_power_of_two(inner))


def _size_class(work: int) -> int:
    """Coarse problem size bucket, part of the autotuning key.

    The best tile is very different for a single short utterance and for a
    large batch, but retuning for every length would be far more expensive
    than the kernel itself. Bucketing by order of magnitude keeps the number
    of tuning runs at a handful per rate pair.
    """

    return min(6, max(0, (work // 1024).bit_length()))


def _target_length(n_samples: int, orig_step: int, new_step: int) -> int:
    """Output length of torchaudio's ``_apply_sinc_resample_kernel``.

    torchaudio rounds ``new_step * n_samples / orig_step`` up *in float32*.
    That computation is replicated bit for bit so both implementations always
    agree on the number of samples they emit.
    """

    return int(torch.ceil(torch.as_tensor(new_step * n_samples / orig_step)).long())


def _check_index_range(n_samples: int, n_out: int, sizes: Tuple[int, ...]) -> None:
    """Refuse sizes for which the 32 bit kernel indices could wrap.

    The largest index any kernel forms is bounded by the number of samples it
    walks plus one tile of over-hang, so a generous slack of one maximal tile
    (``128``) times the resampling step is enough to make the check safe.
    """

    _, _, orig_step, new_step, _, width_total = sizes
    slack = 128 * max(orig_step, new_step) + width_total

    if max(n_samples, n_out) > _INDEX_LIMIT - slack:
        raise RuntimeError(
            "The Triton resampler indexes samples with 32 bit integers, "
            f"{n_samples} input and {n_out} output samples would overflow "
            "them. Split the signal into shorter chunks."
        )


def _launch_forward(
    waveform: torch.Tensor, kernel_t: torch.Tensor, sizes
) -> torch.Tensor:
    """Run the forward kernel on a ``[batch, n_samples]`` contiguous input."""

    n_samples, n_out, orig_step, new_step, width, width_total = sizes
    batch = waveform.shape[0]

    out = torch.empty((batch, n_out), device=waveform.device, dtype=waveform.dtype)

    if batch == 0 or n_out == 0:
        return out

    n_frames = -(-n_out // new_step)
    block_j = _block_inner(new_step)
    columns = triton.cdiv(new_step, block_j)

    def grid(meta):
        return (batch * triton.cdiv(n_frames, meta["BLOCK_L"]), columns)

    _resample_fwd_kernel[grid](
        waveform,
        kernel_t,
        out,
        n_samples,
        n_out,
        orig_step,
        new_step,
        width,
        width_total,
        n_frames,
        _size_class(batch * n_frames * columns),
        BLOCK_J=block_j,
    )

    return out


def _launch_backward(
    grad_out: torch.Tensor, kernel: torch.Tensor, sizes
) -> torch.Tensor:
    """Run the adjoint kernel on a ``[batch, n_out]`` contiguous gradient."""

    n_samples, n_out, orig_step, new_step, width, width_total = sizes
    batch = grad_out.shape[0]

    grad_x = torch.empty(
        (batch, n_samples), device=grad_out.device, dtype=grad_out.dtype
    )

    if batch == 0 or n_samples == 0:
        return grad_x

    # every tap reaching a sample is congruent to its residue modulo
    # ``orig_step``, so at most this many of them exist
    n_taps = -(-width_total // orig_step)
    n_frames = (n_samples - 1 + width) // orig_step + 1
    block_r = _block_inner(orig_step)
    columns = triton.cdiv(orig_step, block_r)

    def grid(meta):
        return (batch * triton.cdiv(n_frames, meta["BLOCK_L"]), columns)

    _resample_bwd_kernel[grid](
        grad_out,
        kernel,
        grad_x,
        n_samples,
        n_out,
        orig_step,
        new_step,
        width,
        width_total,
        n_taps,
        n_frames,
        _size_class(batch * n_frames * columns),
        BLOCK_R=block_r,
    )

    return grad_x


class _ResampleAdjoint(torch.autograd.Function):
    """The adjoint convolution, itself differentiable.

    Resampling is linear, so the derivative of ``grad_x = A^T grad_y`` with
    respect to ``grad_y`` is ``A`` again. Wrapping the adjoint in its own
    :class:`torch.autograd.Function` is what gives the module a working double
    backward -- ``torchaudio``'s ``conv1d`` has one, and a single
    :class:`torch.autograd.Function` for both directions would not.
    """

    @staticmethod
    def forward(ctx, grad_out, kernel, kernel_t, sizes):
        ctx.save_for_backward(kernel_t)
        ctx.sizes = sizes

        return _launch_backward(grad_out.contiguous(), kernel, sizes)

    @staticmethod
    def backward(ctx, grad_grad_x):
        (kernel_t,) = ctx.saved_tensors

        if not ctx.needs_input_grad[0]:
            return None, None, None, None

        return (
            _launch_forward(grad_grad_x.contiguous(), kernel_t, ctx.sizes),
            None,
            None,
            None,
        )


class _ResampleFunction(torch.autograd.Function):
    """Autograd wrapper around the two Triton kernels."""

    @staticmethod
    def forward(ctx, waveform, kernel, kernel_t, orig_step, new_step, width):
        n_samples = waveform.shape[1]
        width_total = kernel.shape[1]
        n_out = _target_length(n_samples, orig_step, new_step)

        sizes = (n_samples, n_out, orig_step, new_step, width, width_total)
        _check_index_range(n_samples, n_out, sizes)

        ctx.save_for_backward(kernel, kernel_t)
        ctx.sizes = sizes

        return _launch_forward(waveform, kernel_t, sizes)

    @staticmethod
    def backward(ctx, grad_out):
        kernel, kernel_t = ctx.saved_tensors

        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        grad_x = _ResampleAdjoint.apply(grad_out, kernel, kernel_t, ctx.sizes)

        return grad_x, None, None, None, None, None


class TritonResample(torch.nn.Module):
    """Resample a signal from one frequency to another with Triton kernels.

    Numerically equivalent to :class:`torchaudio.transforms.Resample`
    constructed with the same arguments. The interpolation kernel is designed
    once on the host by torchaudio and registered as a buffer, so ``.to(device)``
    and ``.cuda()`` behave as expected.

    Parameters
    ----------
    orig_freq : int
        Sampling rate of the input signal
    new_freq : int
        Sampling rate of the output signal
    lowpass_filter_width : int
        Number of zero crossings kept in the sinc kernel, larger is sharper
    rolloff : float
        Cut-off of the anti aliasing filter as a fraction of the Nyquist rate
    resampling_method : str
        Either ``"sinc_interp_hann"`` or ``"sinc_interp_kaiser"``
    beta : Optional[float]
        Shape parameter of the Kaiser window

    Attributes
    ----------
    kernel : torch.Tensor
        Filter bank with shape ``[new_step, width_total]``, read by the
        backward kernel. Only present when ``orig_freq != new_freq``
    kernel_t : torch.Tensor
        Transposed filter bank with shape ``[width_total, new_step]``, read by
        the forward kernel. Both layouts are stored because forward and
        backward stream the taps along different axes
    width : int
        Number of samples the input is zero padded with on the left

    Notes
    -----
    When ``orig_freq == new_freq`` torchaudio skips the filtering completely.
    This module does the same and returns its input unchanged, without
    launching a kernel and without requiring a CUDA tensor.

    Signals longer than roughly ``2**31`` samples, in or out, are rejected
    because the kernels index with 32 bit integers.
    """

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        resampling_method: str = "sinc_interp_hann",
        beta: Optional[float] = None,
    ):
        super(TritonResample, self).__init__()

        if orig_freq <= 0 or new_freq <= 0:
            raise ValueError("Original and desired frequency have to be positive.")

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.gcd = math.gcd(int(orig_freq), int(new_freq))
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.resampling_method = resampling_method
        self.beta = beta

        self.orig_step = int(orig_freq) // self.gcd
        self.new_step = int(new_freq) // self.gcd

        if self.orig_freq != self.new_freq:
            kernel, self.width = _get_sinc_resample_kernel(
                self.orig_freq,
                self.new_freq,
                self.gcd,
                self.lowpass_filter_width,
                self.rolloff,
                self.resampling_method,
                self.beta,
            )

            # torchaudio keeps the [new_step, 1, width_total] conv1d layout
            kernel = kernel.reshape(self.new_step, -1)

            self.register_buffer("kernel", kernel.contiguous())
            self.register_buffer("kernel_t", kernel.t().contiguous())
        else:
            self.width = 0

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Resample a batch of time signals.

        Parameters
        ----------
        waveform : torch.Tensor
            Time signal with shape ``[batch, sample]``

        Returns
        -------
        torch.Tensor
            Resampled signal with shape ``[batch, resampled]``
        """

        if self.orig_freq == self.new_freq:
            return waveform

        require_triton()

        if self.kernel.device != waveform.device:
            raise RuntimeError(
                f"The resampling kernel lives on {self.kernel.device} but the "
                f"waveform on {waveform.device}, move the module with `.to(...)`."
            )

        shape = waveform.shape

        # `reshape(-1, n)` is ambiguous for empty tensors, pack explicitly
        batch = math.prod(shape[:-1])
        packed = check_input(waveform.reshape(batch, shape[-1]), "waveform")

        out = _ResampleFunction.apply(
            packed,
            self.kernel,
            self.kernel_t,
            self.orig_step,
            self.new_step,
            self.width,
        )

        return out.reshape(shape[:-1] + out.shape[-1:])

    def extra_repr(self) -> str:
        return f"orig_freq={self.orig_freq}, new_freq={self.new_freq}"
