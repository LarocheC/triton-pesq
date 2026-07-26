"""Parity tests for the fused Triton STFT / Bark op.

Tolerances
----------

The oracle (:func:`ref_stft_bark`) uses ``torch.stft``, i.e. an FFT, while the
Triton op evaluates a direct DFT. In float32 an FFT accumulates
``O(sqrt(log n))`` rounding error and a direct DFT ``O(sqrt(n))``, so the two
cannot agree to a per element ``rtol`` on band powers that are close to zero --
neither of them is anywhere near correct there, as the comparison against a
float64 evaluation in ``test_accuracy_vs_float64`` shows.

The tolerance used here is therefore

    ``|triton - oracle| <= RTOL * |oracle| + ATOL_SCALE * local_scale``

which is the natural error model of a fixed length dot product: the absolute
error of an output is proportional to the norm of the inputs it was reduced
over, not to its own size.

``local_scale`` is deliberately *local*. The rounding error of bin ``f`` of
frame ``t`` is proportional to the energy of frame ``t`` alone, so scaling by
the peak of the whole spectrogram would make the check blind to a defect that
only shows up in the quiet part of a signal -- and speech has 40 dB of dynamics
between its loud and its quiet frames. The forward therefore scales by the peak
of each individual frame (:func:`_frame_scale`) and the backward by the peak of
the frames that a sample takes part in (:func:`_sample_scale`).

The constants are ``ATOL_SCALE_FWD = 5e-6`` and ``ATOL_SCALE_BWD = 1e-5``
against measured worst cases of ``5.0e-7`` and ``1.4e-6``, i.e. roughly a
factor of ten of headroom. ``test_tolerance_is_not_vacuous`` pins that the
check still fires when a single frequency bin is dropped from a single band,
and ``test_error_stays_well_inside_the_budget`` pins the headroom itself so
that a regression cannot silently eat it.

A flat ``rtol=1e-5, atol=1e-6`` is not meaningful for this op: the output is a
power spectrum whose peak is routinely ``1e9`` and whose smallest entries are
``1e-20`` of that, so ``atol=1e-6`` is simultaneously absurdly loose at the top
and unreachable at the bottom -- for the oracle just as much as for the kernel.

Subnormals
----------

Band powers below ``1.2e-38`` are float32 subnormals and lose relative
precision in *both* implementations. ``test_subnormal_powers`` states the only
claim that can be made there, namely that the Triton op is no worse than
``torch.stft``; the ordinary parity tests stay above that regime on purpose.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torch_pesq.bark import BarkScale
from torch_pesq.triton_ops._common import HAS_TRITON
from torch_pesq.triton_ops.reference import ref_stft_bark

if HAS_TRITON:
    from torch_pesq.triton_ops.spectral import TritonStftBark, bark_segments

cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON),
    reason="the Triton backend needs a CUDA device and the triton package",
)

RTOL = 1e-5
ATOL_SCALE_FWD = 5e-6
ATOL_SCALE_BWD = 1e-5

_CACHE = {}


def _setup(n_fft=512, nbarks=49, hop_length=256):
    """Window, filterbank and module for a configuration, compiled only once.

    The module is shared between tests, so never mutate its buffers -- use
    :func:`_fresh` for that.
    """

    key = (n_fft, nbarks, hop_length)

    if key not in _CACHE:
        window = torch.hann_window(n_fft)
        scale = BarkScale(n_fft // 2, nbarks)
        module = TritonStftBark(
            window, scale.fbank, scale.pow_dens_correction, n_fft, hop_length
        ).cuda()
        _CACHE[key] = (window.cuda(), scale.fbank.cuda(), module)

    return _CACHE[key]


def _fresh(n_fft=512, nbarks=49, hop_length=256):
    """An unshared module whose constant buffers may be perturbed."""

    scale = BarkScale(n_fft // 2, nbarks)

    return TritonStftBark(
        torch.hann_window(n_fft),
        scale.fbank,
        scale.pow_dens_correction,
        n_fft,
        hop_length,
    ).cuda()


def _oracle(signal, window, fbank, module, n_padded):
    """Reference output for a signal that is zero padded to ``n_padded``."""

    padded = torch.nn.functional.pad(signal, (0, n_padded - signal.shape[1]))

    return ref_stft_bark(
        padded,
        window.to(signal.dtype),
        fbank.to(signal.dtype),
        module.correction[: module.nbarks].to(signal.dtype),
        module.n_fft,
        module.hop_length,
    )


def _frame_scale(ref):
    """Per frame magnitude of a ``[batch, frame, bark]`` spectrogram."""

    return ref.abs().amax(dim=2, keepdim=True)


def _sample_scale(grad, hop_length, novertaps):
    """Per sample magnitude of a ``[batch, sample]`` gradient.

    The gradient of a sample is a sum over the at most ``novertaps`` frames it
    belongs to, so its rounding error scales with the largest of those frames.
    The gradient is reduced to one value per ``hop_length`` long block and the
    block maximum is then dilated by ``novertaps`` blocks in both directions.
    """

    nsamples = grad.shape[1]
    padding = (-nsamples) % hop_length
    blocks = (
        F.pad(grad.abs(), (0, padding))
        .reshape(grad.shape[0], -1, hop_length)
        .amax(dim=2)
    )
    dilated = F.max_pool1d(
        blocks.unsqueeze(1),
        kernel_size=2 * novertaps + 1,
        stride=1,
        padding=novertaps,
    ).squeeze(1)

    return dilated.repeat_interleave(hop_length, dim=1)[:, :nsamples]


def _close(got, ref, atol_scale, scale=None):
    """Largest violation of the scaled tolerance and the error metrics.

    Returns ``(violation, max_abs_error, max_error_over_scale)``. The test
    passes when ``violation <= 0``.
    """

    ref = ref.double()
    scale = ref.abs().max() if scale is None else scale.double()
    delta = (got.double() - ref).abs()

    return (
        (delta - (RTOL * ref.abs() + atol_scale * scale)).max().item(),
        delta.max().item(),
        (delta / torch.clamp(scale.expand_as(delta), min=1e-300)).max().item(),
    )


# ---------------------------------------------------------------------------
# structural assumptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_fft", [512, 256])
@pytest.mark.parametrize("nbarks", [49, 32])
def test_filterbank_is_a_segmentation(n_fft, nbarks):
    """The Bark filterbank has to be binary, contiguous and non overlapping."""

    fbank = BarkScale(n_fft // 2, nbarks).fbank
    starts, ends = bark_segments(fbank)

    assert torch.all((fbank == 0.0) | (fbank == 1.0))

    prev = 0
    for band in range(nbarks):
        assert starts[band] >= prev
        assert ends[band] <= n_fft // 2

        expected = torch.zeros(n_fft // 2)
        expected[starts[band] : ends[band]] = 1.0
        assert torch.equal(fbank[band], expected)

        prev = ends[band]


@cuda
def test_rejects_broken_filterbanks():
    """A filterbank that violates the segment structure has to be refused."""

    window = torch.hann_window(64)
    correction = torch.ones(4)

    overlapping = torch.zeros(4, 32)
    overlapping[0, 0:8] = 1.0
    overlapping[1, 4:12] = 1.0
    with pytest.raises(ValueError, match="disjoint"):
        TritonStftBark(window, overlapping, correction, 64, 32)

    gappy = torch.zeros(4, 32)
    gappy[0, 0] = 1.0
    gappy[0, 5] = 1.0
    with pytest.raises(ValueError, match="not contiguous"):
        TritonStftBark(window, gappy, correction, 64, 32)

    weighted = torch.zeros(4, 32)
    weighted[0, 0:4] = 0.5
    with pytest.raises(ValueError, match="binary"):
        TritonStftBark(window, weighted, correction, 64, 32)

    fine = torch.zeros(4, 32)
    fine[0, 0:4] = 1.0
    with pytest.raises(ValueError, match="at least 32"):
        TritonStftBark(window[:16], fine[:, :8], correction, 16, 8)
    with pytest.raises(ValueError, match="512 bins"):
        TritonStftBark(window, fine, correction, 1024, 32)
    with pytest.raises(ValueError, match="one entry per band"):
        TritonStftBark(window, fine, correction[:2], 64, 32)
    with pytest.raises(ValueError, match="at most 64 taps"):
        TritonStftBark(torch.hann_window(128), fine, correction, 64, 32)


@cuda
@pytest.mark.parametrize("n_fft", [40, 48, 96, 1000])
def test_rejects_non_power_of_two_n_fft(n_fft):
    """``n_fft // 2`` has to be a power of two, see the module docstring.

    Both kernels tile the folded reduction with a single
    ``tl.arange(0, n_fft // 2)`` and the forward steps its ``k`` loop in blocks
    of 16 without a tail mask. With ``n_fft = 40`` the kernel used to read the
    twiddle tables 12 rows past their end and produced a forward that was off
    by 38% of the spectrogram peak; with ``n_fft = 48`` it died inside the
    Triton compiler. Both are rejected at construction time now.
    """

    half = n_fft // 2
    fbank = torch.zeros(4, half)
    for band in range(4):
        fbank[band, 4 * band : 4 * band + 4] = 1.0

    with pytest.raises(ValueError, match="power of two"):
        TritonStftBark(torch.hann_window(n_fft), fbank, torch.ones(4), n_fft, half)


@cuda
def test_short_window_is_centred():
    """A ``win_length < n_fft`` window is centred exactly like ``torch.stft``."""

    scale = BarkScale(256, 49)

    torch.manual_seed(23)
    signal = torch.randn(2, 4096, device="cuda") * 1000.0

    for win_length in (400, 401, 33):
        window = torch.hann_window(win_length)
        module = TritonStftBark(
            window, scale.fbank, scale.pow_dens_correction, 512, 256
        ).cuda()

        ref = ref_stft_bark(
            signal,
            window.cuda(),
            scale.fbank.cuda(),
            scale.pow_dens_correction.cuda(),
            512,
            256,
        )

        violation, absolute, _ = _close(
            module(signal), ref, ATOL_SCALE_FWD, _frame_scale(ref)
        )
        assert violation <= 0.0, f"win_length {win_length}, maxabs {absolute:.3e}"


# ---------------------------------------------------------------------------
# forward parity
# ---------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("n_fft,nbarks", [(512, 49), (512, 32), (256, 49), (256, 32)])
@pytest.mark.parametrize("batch,nsamples", [(1, 4096), (3, 5003), (5, 1024)])
def test_forward_parity(n_fft, nbarks, batch, nsamples):
    """Forward matches the oracle over shapes and configurations."""

    window, fbank, module = _setup(n_fft, nbarks)

    torch.manual_seed(nsamples + n_fft + nbarks)
    signal = torch.randn(batch, nsamples, device="cuda") * 3000.0

    got = module(signal)
    ref = _oracle(signal, window, fbank, module, nsamples)

    assert got.shape == ref.shape

    violation, absolute, relative = _close(got, ref, ATOL_SCALE_FWD, _frame_scale(ref))
    assert violation <= 0.0, f"maxabs {absolute:.3e}, rel to frame {relative:.3e}"


@cuda
@pytest.mark.parametrize(
    "name",
    [
        "zeros",
        "small",
        "huge",
        "alternating",
        "impulse",
        "dc",
        "wide_dynamic_range",
        "pure_tone",
        "sign_flip",
    ],
)
def test_forward_adversarial(name):
    """Degenerate inputs still match the oracle."""

    window, fbank, module = _setup()
    signal = _adversarial(name)

    got = module(signal)
    ref = _oracle(signal, window, fbank, module, signal.shape[1])

    violation, absolute, relative = _close(got, ref, ATOL_SCALE_FWD, _frame_scale(ref))
    assert violation <= 0.0, f"maxabs {absolute:.3e}, rel to frame {relative:.3e}"


def _adversarial(name, batch=2, nsamples=4096):
    """Degenerate test signals.

    ``small`` stops at ``1e-15``: the band powers are then ``~1e-30``, which is
    still a normal float32. At ``1e-20`` the powers underflow into the subnormal
    range, which is covered separately by :func:`test_subnormal_powers`.
    """

    torch.manual_seed(7)
    ramp = torch.arange(nsamples, device="cuda", dtype=torch.float32)

    return {
        "zeros": lambda: torch.zeros(batch, nsamples, device="cuda"),
        "small": lambda: torch.randn(batch, nsamples, device="cuda") * 1e-15,
        "huge": lambda: torch.randn(batch, nsamples, device="cuda") * 1e15,
        "alternating": lambda: ((-1.0) ** ramp).expand(batch, nsamples) * 1e3,
        "impulse": lambda: torch.zeros(batch, nsamples, device="cuda").index_fill_(
            1, torch.tensor([1000], device="cuda"), 1e6
        ),
        "dc": lambda: torch.full((batch, nsamples), 1e4, device="cuda"),
        "wide_dynamic_range": lambda: torch.randn(batch, nsamples, device="cuda")
        * torch.exp(-ramp / 200.0),
        # a tone exactly on bin 64 leaves every other bin near zero, the worst
        # case for a relative comparison of a direct DFT against an FFT
        "pure_tone": lambda: torch.sin(2.0 * np.pi * 64.0 * ramp / 512.0).expand(
            batch, nsamples
        )
        * 1e4,
        "sign_flip": lambda: torch.where(
            ramp < nsamples // 2,
            -1e5 * torch.ones_like(ramp),
            1e5 * torch.ones_like(ramp),
        ).expand(batch, nsamples),
    }[name]().contiguous()


@cuda
@pytest.mark.parametrize("nsamples,n_padded", [(4000, 4096), (3900, 4096), (777, 1024)])
def test_forward_trailing_padding(nsamples, n_padded):
    """``n_padded`` reproduces a separate zero padding of the input."""

    window, fbank, module = _setup()

    torch.manual_seed(nsamples)
    signal = torch.randn(3, nsamples, device="cuda") * 1000.0

    got = module(signal, n_padded)
    ref = _oracle(signal, window, fbank, module, n_padded)

    assert got.shape[1] == 1 + (n_padded - 512) // 256

    violation, absolute, relative = _close(got, ref, ATOL_SCALE_FWD, _frame_scale(ref))
    assert violation <= 0.0, f"maxabs {absolute:.3e}, rel to frame {relative:.3e}"


@cuda
def test_forward_drops_the_partial_frame():
    """Samples behind the last complete frame do not reach the output."""

    _, _, module = _setup()

    torch.manual_seed(11)
    signal = torch.randn(2, 4000, device="cuda") * 1000.0

    # 1 + (4000 - 512) // 256 == 14 frames, covering 3840 samples
    assert module(signal).shape[1] == 14

    tail = signal.clone()
    tail[:, 3840:] = 1e9

    assert torch.equal(module(signal), module(tail))


@cuda
def test_handles_awkward_inputs():
    """Non contiguous inputs, ``no_grad`` and inputs without a gradient."""

    window, fbank, module = _setup()

    torch.manual_seed(3)
    strided = (torch.randn(3, 4096, 2, device="cuda") * 1000.0)[..., 0]
    assert not strided.is_contiguous()

    ref = _oracle(strided.contiguous(), window, fbank, module, 4096)
    violation, absolute, _ = _close(
        module(strided), ref, ATOL_SCALE_FWD, _frame_scale(ref)
    )
    assert violation <= 0.0, f"maxabs {absolute:.3e}"

    with torch.no_grad():
        blocked = module(strided.contiguous().requires_grad_(True))
    assert not blocked.requires_grad

    plain = torch.randn(1, 2048, device="cuda")
    assert not module(plain).requires_grad

    # an empty batch must not trip the launch geometry
    assert module(torch.randn(0, 4096, device="cuda")).shape == (0, 15, 49)


@cuda
def test_rejects_bad_calls():
    """Argument validation of :meth:`TritonStftBark.forward`."""

    _, _, module = _setup()

    with pytest.raises(ValueError, match="2d"):
        module(torch.randn(2, 3, 4096, device="cuda"))
    with pytest.raises(ValueError, match="smaller than the signal"):
        module(torch.randn(2, 4096, device="cuda"), 1024)
    with pytest.raises(ValueError, match="shorter than one frame"):
        module(torch.randn(2, 256, device="cuda"))
    with pytest.raises(RuntimeError, match="GPU"):
        module(torch.randn(2, 4096))
    with pytest.raises(RuntimeError, match="float32"):
        module(torch.randn(2, 4096, device="cuda", dtype=torch.float64))


# ---------------------------------------------------------------------------
# backward parity
# ---------------------------------------------------------------------------


def _both(module, window, fbank, signal, n_padded, upstream=None):
    """Run the op and the oracle on the same input and take both gradients."""

    triton_in = signal.clone().requires_grad_(True)
    oracle_in = signal.clone().requires_grad_(True)

    got = module(triton_in, n_padded)
    ref = _oracle(oracle_in, window, fbank, module, n_padded)

    if upstream is None:
        # a random upstream gradient, ones would hide index permutations
        upstream = torch.randn_like(ref)

    (grad_got,) = torch.autograd.grad(got, (triton_in,), upstream)
    (grad_ref,) = torch.autograd.grad(ref, (oracle_in,), upstream)

    return got, ref, grad_got, grad_ref


@cuda
@pytest.mark.parametrize("n_fft,nbarks", [(512, 49), (512, 32), (256, 49), (256, 32)])
@pytest.mark.parametrize(
    "batch,nsamples,n_padded", [(1, 4096, 4096), (3, 5003, 5120), (2, 4000, 4000)]
)
def test_backward_parity(n_fft, nbarks, batch, nsamples, n_padded):
    """Gradients match ``torch.autograd.grad`` of the oracle."""

    window, fbank, module = _setup(n_fft, nbarks)

    torch.manual_seed(nsamples + n_fft * nbarks)
    signal = torch.randn(batch, nsamples, device="cuda") * 1000.0

    _, _, grad_got, grad_ref = _both(module, window, fbank, signal, n_padded)

    assert grad_got.shape == signal.shape

    violation, absolute, relative = _close(
        grad_got,
        grad_ref,
        ATOL_SCALE_BWD,
        _sample_scale(grad_ref, module.hop_length, module.novertaps),
    )
    assert violation <= 0.0, f"maxabs {absolute:.3e}, rel to frame {relative:.3e}"


@cuda
@pytest.mark.parametrize(
    "name",
    ["zeros", "small", "huge", "impulse", "dc", "wide_dynamic_range", "pure_tone"],
)
def test_backward_adversarial(name):
    """Degenerate inputs also have to give the right gradient.

    The forward is blind to the sign of the imaginary part -- ``|X| ** 2``
    throws it away -- so the sine table is only ever validated here.
    """

    window, fbank, module = _setup()
    signal = _adversarial(name)

    _, _, grad_got, grad_ref = _both(module, window, fbank, signal, signal.shape[1])

    assert not torch.isnan(grad_got).any()
    assert not torch.isinf(grad_got).any()

    violation, absolute, relative = _close(
        grad_got,
        grad_ref,
        ATOL_SCALE_BWD,
        _sample_scale(grad_ref, module.hop_length, module.novertaps),
    )
    assert violation <= 0.0, f"maxabs {absolute:.3e}, rel to frame {relative:.3e}"


@cuda
def test_backward_awkward_inputs():
    """Non contiguous signal, non contiguous upstream and a float64 upstream."""

    window, fbank, module = _setup()

    torch.manual_seed(29)
    signal = torch.randn(3, 5003, device="cuda") * 1000.0

    # the gradient has to reach the *strided* leaf, not just the copy that
    # `check_input` makes
    lanes = torch.stack([signal, torch.zeros_like(signal)], dim=2).requires_grad_(True)
    plain = signal.clone().requires_grad_(True)

    got = module(lanes[..., 0], 5120)
    ref = _oracle(plain, window, fbank, module, 5120)

    # a non contiguous upstream gradient, too
    upstream = torch.randn(*ref.shape, 2, device="cuda")[..., 0]
    assert not lanes[..., 0].is_contiguous() and not upstream.is_contiguous()

    (grad_got,) = torch.autograd.grad(got, (lanes,), upstream)
    (grad_ref,) = torch.autograd.grad(ref, (plain,), upstream)

    assert torch.count_nonzero(grad_got[..., 1]) == 0

    violation, absolute, _ = _close(
        grad_got[..., 0],
        grad_ref,
        ATOL_SCALE_BWD,
        _sample_scale(grad_ref, module.hop_length, module.novertaps),
    )
    assert violation <= 0.0, f"maxabs {absolute:.3e}"

    # a float64 upstream gradient must not crash or change the result
    leaf = signal.clone().requires_grad_(True)
    (from_double,) = torch.autograd.grad(
        module(leaf, 5120), (leaf,), upstream.double().contiguous()
    )
    leaf = signal.clone().requires_grad_(True)
    (from_single,) = torch.autograd.grad(
        module(leaf, 5120), (leaf,), upstream.contiguous()
    )
    assert torch.equal(from_double, from_single)


@cuda
def test_backward_ignores_the_padding():
    """Samples that no frame covers receive a zero gradient."""

    _, _, module = _setup()

    torch.manual_seed(5)
    signal = (torch.randn(2, 4000, device="cuda") * 1000.0).requires_grad_(True)

    out = module(signal)
    (grad,) = torch.autograd.grad(out, (signal,), torch.randn_like(out))

    assert torch.count_nonzero(grad[:, 3840:]) == 0
    assert torch.count_nonzero(grad[:, :3840]) > 0


@cuda
def test_backward_matches_finite_differences():
    """Independent check of the VJP against central differences in float64."""

    window, fbank, module = _setup()

    torch.manual_seed(13)
    signal = torch.randn(1, 1024, device="cuda", dtype=torch.float64)
    upstream = torch.randn(1, 3, 49, device="cuda", dtype=torch.float64)

    def loss(vector):
        out = _oracle(vector, window, fbank, module, 1024)
        return (out * upstream).sum()

    single = signal.float().requires_grad_(True)
    out = module(single)
    (grad,) = torch.autograd.grad(out, (single,), upstream.float())

    step, taken = 1e-3, torch.randperm(1024)[:24]
    numeric = torch.zeros(24, dtype=torch.float64)
    for pos, index in enumerate(taken):
        plus, minus = signal.clone(), signal.clone()
        plus[0, index] += step
        minus[0, index] -= step
        numeric[pos] = ((loss(plus) - loss(minus)) / (2.0 * step)).cpu()

    analytic = grad[0, taken.cuda()].double().cpu()
    scale = numeric.abs().max()

    assert torch.allclose(analytic, numeric, rtol=1e-4, atol=1e-5 * scale)


# ---------------------------------------------------------------------------
# hop length, the frame overlap of the backward gather
# ---------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("hop_length", [128, 600, 100])
def test_hop_length_sweep(hop_length):
    """Hops other than ``n_fft // 2`` in both directions.

    ``hop_length`` decides how many frames a sample belongs to, which is the
    trip count of the backward gather loop: 4 for 128, 1 for 600 (no overlap at
    all, and gaps between the frames on top) and 6 for 100, which does not
    divide ``n_fft`` so the last trip of the loop is only partially valid.
    """

    window, fbank, module = _setup(512, 49, hop_length)

    torch.manual_seed(hop_length)
    signal = torch.randn(2, 4096, device="cuda") * 1000.0

    assert module.novertaps == -(-512 // hop_length)

    got, ref, grad_got, grad_ref = _both(module, window, fbank, signal, 4096)

    assert got.shape[1] == 1 + (4096 - 512) // hop_length

    violation, absolute, _ = _close(got, ref, ATOL_SCALE_FWD, _frame_scale(ref))
    assert violation <= 0.0, f"forward maxabs {absolute:.3e}"

    violation, absolute, _ = _close(
        grad_got,
        grad_ref,
        ATOL_SCALE_BWD,
        _sample_scale(grad_ref, hop_length, module.novertaps),
    )
    assert violation <= 0.0, f"backward maxabs {absolute:.3e}"


@cuda
def test_gradient_is_zero_in_the_gaps_between_frames():
    """With ``hop_length > n_fft`` the uncovered samples get exactly zero."""

    _, _, module = _setup(512, 49, 600)

    torch.manual_seed(31)
    signal = (torch.randn(1, 3000, device="cuda") * 1000.0).requires_grad_(True)

    out = module(signal)
    (grad,) = torch.autograd.grad(out, (signal,), torch.randn_like(out))

    for frame in range(out.shape[1]):
        covered = grad[0, frame * 600 : frame * 600 + 512]
        gap = grad[0, frame * 600 + 512 : (frame + 1) * 600]
        assert torch.count_nonzero(covered) > 0
        assert torch.count_nonzero(gap) == 0

    # everything behind the last frame is a gap as well
    assert torch.count_nonzero(grad[0, (out.shape[1] - 1) * 600 + 512 :]) == 0


# ---------------------------------------------------------------------------
# determinism, memory hygiene and accuracy
# ---------------------------------------------------------------------------


@cuda
def test_determinism():
    """Identical inputs give bitwise identical outputs and gradients."""

    _, _, module = _setup()

    torch.manual_seed(17)
    signal = torch.randn(4, 8192, device="cuda") * 1000.0
    upstream = torch.randn(4, 31, 49, device="cuda")

    first = module(signal, 8192)
    second = module(signal, 8192)
    assert torch.equal(first, second)

    grads = []
    for _ in range(2):
        leaf = signal.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(module(leaf, 8192), (leaf,), upstream)
        grads.append(grad)

    assert torch.equal(grads[0], grads[1])

    # calling the backward twice on the same graph has to be stable, too
    leaf = signal.clone().requires_grad_(True)
    out = module(leaf, 8192)
    (again,) = torch.autograd.grad(out, (leaf,), upstream, retain_graph=True)
    (once_more,) = torch.autograd.grad(out, (leaf,), upstream)
    assert torch.equal(again, once_more)


@cuda
def test_scratch_buffers_are_fully_written():
    """No element of the uninitialised scratch buffers survives into a result.

    ``out``, ``Re``, ``Im`` and the ``[batch, frame, n_fft]`` frame gradient are
    all allocated with ``torch.empty``. The caching allocator is primed with
    NaN filled blocks of exactly those shapes first, so a tile that the kernels
    fail to cover shows up as a NaN instead of as a silently plausible zero.
    """

    _, _, module = _setup()

    batch, nsamples = 3, 5003
    nframes = 1 + (5120 - 512) // 256

    for _ in range(3):
        junk = [
            torch.full((batch, nframes, width), float("nan"), device="cuda")
            for width in (module.nbarks, module.nfreq, module.nfreq, 512)
        ]
        junk.append(torch.full((batch, nsamples), float("nan"), device="cuda"))
        del junk

        signal = (torch.randn(batch, nsamples, device="cuda") * 1000.0).requires_grad_(
            True
        )
        out = module(signal, 5120)
        (grad,) = torch.autograd.grad(out, (signal,), torch.randn_like(out))

        assert torch.isfinite(out).all()
        assert torch.isfinite(grad).all()


@cuda
def test_accuracy_vs_float64():
    """The op is not worse than the float32 oracle against a float64 baseline."""

    window, fbank, module = _setup()

    torch.manual_seed(19)
    signal = torch.randn(2, 16384, device="cuda") * 3000.0

    exact = _oracle(signal.double(), window.double(), fbank.double(), module, 16384)
    peak = exact.abs().max()

    triton_error = (module(signal).double() - exact).abs().max() / peak
    oracle_error = (_oracle(signal, window, fbank, module, 16384) - exact).abs().max()

    assert triton_error <= oracle_error / peak * 2.0, (
        f"triton {triton_error:.3e} vs oracle {oracle_error / peak:.3e} "
        "of the spectrogram peak"
    )
    assert triton_error < 1e-6


@cuda
def test_subnormal_powers():
    """Below the float32 normal range only a relative claim is possible.

    An amplitude of ``1e-20`` puts the band powers at ``~4e-40``, which is a
    float32 subnormal (the smallest normal is ``1.2e-38``). Both the FFT of the
    oracle and the direct DFT of the kernel then lose the same three to four
    decimal digits -- measured against a float64 baseline both are off by
    ``1.7e-6`` of the peak, so the only meaningful assertion is that the Triton
    op does not do worse.
    """

    window, fbank, module = _setup()

    torch.manual_seed(19)
    signal = torch.randn(2, 8192, device="cuda") * 1e-20

    exact = _oracle(signal.double(), window.double(), fbank.double(), module, 8192)
    peak = exact.abs().max()

    assert peak < 1e-37, "the test is supposed to run in the subnormal regime"

    triton_error = ((module(signal).double() - exact).abs().max() / peak).item()
    oracle_error = (
        (_oracle(signal, window, fbank, module, 8192) - exact).abs().max() / peak
    ).item()

    assert (
        triton_error <= 2.0 * oracle_error
    ), f"triton {triton_error:.3e} vs oracle {oracle_error:.3e} of the peak"
    assert triton_error < 1e-4


@cuda
def test_tolerance_is_not_vacuous():
    """The scaled tolerance still detects a one bin defect.

    Dropping a single frequency bin from a single Bark band, or scaling one
    band correction by 1%, has to make :func:`_close` report a violation --
    otherwise the parity tests above would prove nothing.
    """

    window, fbank, clean = _setup()

    torch.manual_seed(37)
    signal = torch.randn(2, 4096, device="cuda") * 1000.0

    # the reference always comes from the pristine module: `_oracle` reads
    # `module.correction`, so deriving it from the perturbed one would move the
    # reference along with the defect
    leaf = signal.clone().requires_grad_(True)
    ref = _oracle(leaf, window, fbank, clean, 4096)
    upstream = torch.randn_like(ref)
    (grad_ref,) = torch.autograd.grad(ref, (leaf,), upstream)
    frame_scale = _frame_scale(ref)
    sample_scale = _sample_scale(grad_ref, clean.hop_length, clean.novertaps)

    defects = {
        "drop bin 1": lambda m: m.fbank_fwd[1].zero_(),
        "drop bin 100": lambda m: m.fbank_fwd[100].zero_(),
        "drop bin 255": lambda m: m.fbank_fwd[255].zero_(),
        "correction off by 1%": lambda m: m.correction[48].mul_(1.01),
    }

    for name, defect in defects.items():
        module = _fresh()
        defect(module)

        violation, _, _ = _close(module(signal), ref, ATOL_SCALE_FWD, frame_scale)
        assert violation > 0.0, f"{name} slipped through the forward tolerance"

    gradient_defects = {
        "bin -> band map shifted": lambda m: m.bandid.copy_(torch.roll(m.bandid, 1)),
        "sine sign flipped": lambda m: m.sin_bwd.neg_(),
        "unpaired n/2 tap dropped": lambda m: m.sgn.zero_(),
        "per bin correction off by 1%": lambda m: m.corrbin[1:].mul_(1.01),
    }

    for name, defect in gradient_defects.items():
        module = _fresh()
        defect(module)

        leaf = signal.clone().requires_grad_(True)
        (grad_got,) = torch.autograd.grad(module(leaf, 4096), (leaf,), upstream)

        violation, _, _ = _close(grad_got, grad_ref, ATOL_SCALE_BWD, sample_scale)
        assert violation > 0.0, f"{name} slipped through the backward tolerance"


@cuda
def test_error_stays_well_inside_the_budget():
    """Pin the headroom, so that a regression cannot silently consume it.

    Measured on this GPU the worst case is ``5.0e-7`` of the frame peak for the
    forward and ``1.4e-6`` for the backward, against budgets of ``5e-6`` and
    ``1e-5``. Anything above a fifth of the budget is a regression, not noise.
    """

    window, fbank, module = _setup()

    worst_fwd, worst_bwd = 0.0, 0.0

    for name in ("zeros", "huge", "impulse", "wide_dynamic_range", "pure_tone"):
        signal = _adversarial(name)
        _, ref, grad_got, grad_ref = _both(module, window, fbank, signal, 4096)

        _, _, relative = _close(module(signal), ref, ATOL_SCALE_FWD, _frame_scale(ref))
        worst_fwd = max(worst_fwd, relative)

        _, _, relative = _close(
            grad_got,
            grad_ref,
            ATOL_SCALE_BWD,
            _sample_scale(grad_ref, module.hop_length, module.novertaps),
        )
        worst_bwd = max(worst_bwd, relative)

    assert worst_fwd < ATOL_SCALE_FWD / 5.0, f"forward margin down to {worst_fwd:.3e}"
    assert worst_bwd < ATOL_SCALE_BWD / 5.0, f"backward margin down to {worst_bwd:.3e}"


# ---------------------------------------------------------------------------
# launch geometry and filterbank shapes
# ---------------------------------------------------------------------------


@cuda
def test_long_signal_clears_the_grid_limit():
    """A signal past ``65535 * 256`` samples still launches the gather kernel.

    CUDA caps grid dimensions ``y`` and ``z`` at 65535. The backward gather
    used to put the sample tile on ``program_id(1)``, so anything longer than
    ``16 776 960`` samples -- 17.5 minutes at 16 kHz, well within the memory of
    an 11 GB card -- died with ``Triton Error [CUDA]: invalid argument``.
    """

    _, _, module = _setup()

    nsamples = 65536 * 256 + 1024
    assert -(-nsamples // 256) > 65535

    signal = torch.ones(1, nsamples, device="cuda").requires_grad_(True)
    out = module(signal)
    (grad,) = torch.autograd.grad(out, (signal,), torch.ones_like(out))

    torch.cuda.synchronize()

    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


@cuda
def test_many_bands_fit_into_shared_memory():
    """More than 64 bands must not blow the 48 KiB shared memory of Pascal.

    ``BLOCK_F`` sizes the ``[BLOCK_F, kpad]`` operand of the band reduction, so
    it has to shrink when ``kpad`` grows. With a fixed ``BLOCK_F = 128`` a 100
    band filterbank asked for 72 KiB and failed with ``OutOfResources``.
    """

    nbarks = 100
    edges = np.linspace(0, 256, nbarks + 1).astype(int)
    fbank = torch.zeros(nbarks, 256)
    for band in range(nbarks):
        fbank[band, edges[band] : edges[band + 1]] = 1.0

    torch.manual_seed(41)
    correction = torch.rand(nbarks) + 0.5
    module = TritonStftBark(torch.hann_window(512), fbank, correction, 512, 256).cuda()

    assert module.kpad == 128

    signal = torch.randn(2, 4096, device="cuda") * 1000.0
    got, ref, grad_got, grad_ref = _both(
        module, torch.hann_window(512).cuda(), fbank.cuda(), signal, 4096
    )

    violation, absolute, _ = _close(got, ref, ATOL_SCALE_FWD, _frame_scale(ref))
    assert violation <= 0.0, f"forward maxabs {absolute:.3e}"

    violation, absolute, _ = _close(
        grad_got, grad_ref, ATOL_SCALE_BWD, _sample_scale(grad_ref, 256, 2)
    )
    assert violation <= 0.0, f"backward maxabs {absolute:.3e}"


@cuda
def test_narrow_filterbank_and_empty_bands():
    """A filterbank that leaves the top bins free shortens the DFT.

    Also covers empty bands, a first band that does not start at bin 0 and the
    ``nfreq < n_fft // 2`` path that drops the uncovered bins entirely.
    """

    fbank = torch.zeros(8, 256)
    for band in range(8):
        fbank[band, 12 * band : 12 * band + 12] = 1.0
    fbank[3].zero_()  # an empty band in the middle
    fbank[0].zero_()  # nothing covers bins 0 .. 11 any more

    torch.manual_seed(43)
    correction = torch.rand(8) + 0.5
    window = torch.hann_window(512)
    module = TritonStftBark(window, fbank, correction, 512, 256).cuda()

    # bins 96 .. 255 are uncovered, so only the lower half is evaluated
    assert module.nfreq == 128
    assert torch.all(module.bandid[96:] == -1)
    assert torch.all(module.corrbin[:12] == 0.0)

    signal = torch.randn(2, 4096, device="cuda") * 1000.0
    got, ref, grad_got, grad_ref = _both(
        module, window.cuda(), fbank.cuda(), signal, 4096
    )

    assert torch.all(got[:, :, 0] == 0.0)
    assert torch.all(got[:, :, 3] == 0.0)

    violation, absolute, _ = _close(got, ref, ATOL_SCALE_FWD, _frame_scale(ref))
    assert violation <= 0.0, f"forward maxabs {absolute:.3e}"

    violation, absolute, _ = _close(
        grad_got, grad_ref, ATOL_SCALE_BWD, _sample_scale(grad_ref, 256, 2)
    )
    assert violation <= 0.0, f"backward maxabs {absolute:.3e}"


@cuda
def test_device_move_and_dtype():
    """The constant tables follow the module and the output stays float32."""

    window = torch.hann_window(512)
    scale = BarkScale(256, 49)
    module = TritonStftBark(window, scale.fbank, scale.pow_dens_correction, 512, 256)

    assert module.cos_fwd.device.type == "cpu"
    module = module.cuda()
    assert module.cos_fwd.device.type == "cuda"
    assert module.cos_fwd.dtype == torch.float32
    assert module.bandid.dtype == torch.int32

    out = module(torch.randn(1, 2048, device="cuda"))
    assert out.dtype == torch.float32
    assert out.shape == (1, 7, 49)


@cuda
def test_segment_boundaries_are_used():
    """``nbarks < 49`` leaves the top bins uncovered and they are skipped."""

    scale = BarkScale(256, 32)
    starts, ends = bark_segments(scale.fbank)

    assert int(ends.max()) < 256

    module = TritonStftBark(
        torch.hann_window(512), scale.fbank, scale.pow_dens_correction, 512, 256
    ).cuda()

    assert module.nfreq == 256
    assert np.array_equal(module.starts.cpu().numpy(), starts)
    assert np.array_equal(module.ends.cpu().numpy(), ends)

    # uncovered bins carry neither a band nor a correction
    assert torch.all(module.bandid[int(ends.max()) :] == -1)
    assert torch.all(module.corrbin[int(ends.max()) :] == 0.0)
    assert module.bandid[0] == -1
