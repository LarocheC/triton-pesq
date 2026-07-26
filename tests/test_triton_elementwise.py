"""Tests for the fused element-wise Triton ops of the PESQ pipeline.

Tolerances
----------
Unless noted otherwise every comparison uses ``rtol=1e-5, atol=1e-6``, the
tolerance mandated by the Triton backend spec. Three of the four ops are
actually *bit exact* with their PyTorch oracle (the kernels use the IEEE
correctly rounded ``tl.div_rn`` / ``tl.sqrt_rn`` instead of the fast
approximations Triton emits for ``/`` and ``tl.sqrt``):

* ``peak_normalize`` -- ``max`` is exact, the division is correctly rounded
* ``edge_ramp`` -- the ramp weights ``k / 16`` and their pairwise products
  ``j * k / 256`` are exact in binary floating point
* ``pesq_epilogue`` -- only ``exp`` differs, by at most one ulp

``align_scale`` sums ~1e5 squares, so its reduction order differs from
``torch.sum`` and the result carries roughly one ulp of relative error. The
measured numbers are printed by every test, run with ``-s`` to see them.

Known limitations, pinned by the tests below
--------------------------------------------
* ``peak_normalize`` treats the peak amplitude as a constant in the backward
  pass. That is exact for the composed pipeline, not for the op in isolation;
  :func:`test_peak_normalize_detach_matches_full_pipeline_gradient` proves it
  end to end against full autograd, in float64.
* If ``sum(filtered**2)`` overflows float32 -- it takes samples of order
  ``1e18``, the pipeline feeds ``align_scale`` peak normalised signals bounded
  by one -- the forward passes of both implementations agree (``scale == 0``)
  but their backward passes disagree in kind: the kernel yields zeros where
  autograd yields ``NaN``. Both are meaningless; see
  :func:`test_align_scale_adversarial_magnitudes`.
* The backward passes are not themselves differentiable, so ``create_graph``
  raises instead of returning second order gradients.
"""

import math

import pytest
import torch

from torchaudio.functional import lfilter

from torch_pesq.loss import PesqLoss
from torch_pesq.triton_ops._common import HAS_TRITON
from torch_pesq.triton_ops.elementwise import (
    _BLOCK,
    _NCHUNK,
    align_scale,
    edge_ramp,
    peak_normalize,
    pesq_epilogue,
)
from torch_pesq.triton_ops.reference import (
    ref_align_level,
    ref_chain,
    ref_pipeline,
    ref_preemphasize,
    ref_stft_bark,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON),
    reason="the Triton backend needs a CUDA device and the triton package",
)

DEVICE = "cuda"

#: Tolerance used across the suite, see the module docstring.
RTOL, ATOL = 1e-5, 1e-6

#: Shapes covering batch == 1, lengths that are not a multiple of the 1024
#: element block size and a realistically long signal.
SHAPES = [(1, 31), (1, 1024), (2, 1023), (3, 5000), (4, 4097), (2, 100_000)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _errors(got, expected):
    """Maximum absolute and relative deviation between two tensors."""

    got, expected = got.double(), expected.double()
    finite = torch.isfinite(got) & torch.isfinite(expected)

    if not finite.any():
        return 0.0, 0.0

    diff = (got[finite] - expected[finite]).abs()
    rel = diff / expected[finite].abs().clamp(min=1e-30)

    return diff.max().item(), rel.max().item()


def _check(got, expected, name, worst, equal_nan=False):
    """Assert closeness, track and return the observed error."""

    abs_err, rel_err = _errors(got, expected)

    previous = worst.get(name, (0.0, 0.0))
    worst[name] = (max(previous[0], abs_err), max(previous[1], rel_err))

    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL, equal_nan=equal_nan)

    return abs_err, rel_err


def _signals(batch, n, kind, generator):
    """Benign and adversarial inputs of shape ``[batch, n]``."""

    base = torch.randn(batch, n, device=DEVICE, generator=generator)

    if kind == "randn":
        return base
    if kind == "tiny":
        return base * 1e-25
    if kind == "huge":
        return base * 1e18
    if kind == "zeros":
        return torch.zeros(batch, n, device=DEVICE)
    if kind == "sparse":  # a single spike, everything else exactly zero
        out = torch.zeros(batch, n, device=DEVICE)
        out[:, n // 3] = -3.5
        return out
    if kind == "signs":
        signs = torch.arange(n, device=DEVICE) % 2 * 2 - 1
        return base.abs() * signs

    raise ValueError(kind)


def _oracle_peak_normalize(ref, deg):
    """Oracle of :func:`peak_normalize`, ``max_val`` detached on purpose.

    The op treats the peak amplitude as a constant in the backward pass, see
    :class:`torch_pesq.triton_ops.elementwise._PeakNormalize` and
    :func:`test_pipeline_is_invariant_to_reference_scaling`.
    """

    max_val = torch.max(
        torch.amax(deg.abs(), dim=1, keepdim=True),
        torch.amax(ref.abs(), dim=1, keepdim=True),
    ).detach()

    return ref / max_val, deg / max_val


def _oracle_edge_ramp(x):
    """Oracle of :func:`edge_ramp`, the head of ``ref_preemphasize``.

    Written as the two *sequential* out-of-place assignments the reference
    performs, rather than as a concatenation of three slices. The difference
    only shows for ``n < 30``, where the leading and the trailing 15 sample
    slice overlap: the reference then applies both weights to the overlapping
    samples, while a three way split would produce a tensor of the wrong shape.
    """

    emp = torch.linspace(0, 15, 16, device=x.device, dtype=x.dtype)[1:] / 16.0

    out = x.clone()
    out[:, :15] = out[:, :15] * emp
    out[:, -15:] = out[:, -15:] * torch.flip(emp, dims=(0,))

    return out


def _noncontiguous(tensor):
    """A non-contiguous view carrying the values and shape of ``tensor``."""

    holder = torch.empty(
        tensor.shape[0], 2 * tensor.shape[1], device=tensor.device, dtype=tensor.dtype
    )
    holder[:, ::2] = tensor
    view = holder[:, ::2]

    assert not view.is_contiguous()

    return view


def _oracle_align_scale(signal, filtered):
    """Oracle of :func:`align_scale`, the tail of ``ref_align_level``."""

    power = (
        (filtered**2).sum(dim=1, keepdim=True) / (filtered.shape[1] + 5120) / 1.04684
    )

    return signal * (10**7 / power).sqrt()


def _oracle_epilogue(d_symm, d_asymm, factor):
    """Oracle of :func:`pesq_epilogue`, see ``PesqLoss.mos`` / ``forward``."""

    mos = 4.5 - 0.1 * d_symm - 0.0309 * d_asymm
    mos = 0.999 + 4 / (1 + torch.exp(-1.3669 * mos + 3.8224))
    loss = factor * (0.1 * d_symm + 0.0309 * d_asymm)

    return mos, loss


def _timeit(fn, repeats=25, inner=10):
    """CUDA event timing of ``fn``, returns ``(median, minimum)`` in ms/call.

    ``inner`` calls are timed together so that the event overhead and the
    launch latency of the very first kernel are amortised, then ``repeats``
    such batches are taken. The GPU is shared with other tenants, so the mean
    is useless; the median is reported for comparability and the minimum as
    the best estimate of the uncontended cost.
    """

    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    times = []
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        for _ in range(inner):
            fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) / inner)

    times.sort()

    return times[len(times) // 2], times[0]


# ---------------------------------------------------------------------------
# peak normalisation
# ---------------------------------------------------------------------------


def test_peak_normalize_forward():
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    worst = {}

    for batch, n in SHAPES:
        for kind in ["randn", "tiny", "huge", "sparse", "signs"]:
            ref = _signals(batch, n, kind, generator)
            deg = _signals(batch, n, kind, generator) * 2.0

            got = peak_normalize(ref, deg)
            expected = _oracle_peak_normalize(ref, deg)

            _check(got[0], expected[0], "ref", worst)
            _check(got[1], expected[1], "deg", worst)

    # the all zero case divides by zero in the oracle as well
    zeros = torch.zeros(2, 512, device=DEVICE)
    got = peak_normalize(zeros, zeros)
    expected = _oracle_peak_normalize(zeros, zeros)
    _check(got[0], expected[0], "zeros", worst, equal_nan=True)

    print(f"\npeak_normalize forward max |abs|, |rel|: {worst}")
    assert worst["ref"] == (0.0, 0.0) and worst["deg"] == (0.0, 0.0)


def test_peak_normalize_non_contiguous():
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    dense = torch.randn(3, 2048, device=DEVICE, generator=generator)

    ref, deg = dense[:, ::2], dense.flip(0)[:, 1::2]
    assert not ref.is_contiguous() and not deg.is_contiguous()

    got = peak_normalize(ref, deg)
    expected = _oracle_peak_normalize(ref, deg)

    torch.testing.assert_close(got[0], expected[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(got[1], expected[1], rtol=RTOL, atol=ATOL)


def test_peak_normalize_backward():
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    worst = {}

    for batch, n in [(1, 31), (2, 1023), (3, 5000)]:
        ref = _signals(batch, n, "randn", generator).requires_grad_(True)
        deg = (_signals(batch, n, "randn", generator) * 3.0).requires_grad_(True)

        # random upstream gradients, not ones
        g_ref = torch.randn(batch, n, device=DEVICE, generator=generator)
        g_deg = torch.randn(batch, n, device=DEVICE, generator=generator)

        out = peak_normalize(ref, deg)
        got = torch.autograd.grad(
            (out[0] * g_ref).sum() + (out[1] * g_deg).sum(), [ref, deg]
        )

        exp_out = _oracle_peak_normalize(ref, deg)
        expected = torch.autograd.grad(
            (exp_out[0] * g_ref).sum() + (exp_out[1] * g_deg).sum(), [ref, deg]
        )

        _check(got[0], expected[0], "d_ref", worst)
        _check(got[1], expected[1], "d_deg", worst)

        # only the reference output is used downstream
        out = peak_normalize(ref, deg)
        (partial,) = torch.autograd.grad((out[0] * g_ref).sum(), [ref])
        exp_out = _oracle_peak_normalize(ref, deg)
        (exp_partial,) = torch.autograd.grad((exp_out[0] * g_ref).sum(), [ref])
        _check(partial, exp_partial, "d_ref_only", worst)

        # only one of the inputs needs a gradient
        out = peak_normalize(ref.detach(), deg)
        (partial,) = torch.autograd.grad((out[1] * g_deg).sum(), [deg])
        _check(partial, expected[1], "d_deg_only", worst)

    print(f"\npeak_normalize backward max |abs|, |rel|: {worst}")


def test_all_ops_under_no_grad():
    """Every op runs inside ``torch.no_grad`` and returns detached results."""

    generator = torch.Generator(device=DEVICE).manual_seed(3)
    ref = torch.randn(2, 4096, device=DEVICE, generator=generator).requires_grad_(True)
    deg = torch.randn(2, 4096, device=DEVICE, generator=generator).requires_grad_(True)
    distance = (
        torch.rand(2, device=DEVICE, generator=generator) * 45.0
    ).requires_grad_(True)

    with torch.no_grad():
        results = [
            *peak_normalize(ref, deg),
            edge_ramp(ref),
            align_scale(ref, deg),
            *pesq_epilogue(distance, distance, 0.5),
        ]

    assert not any(result.requires_grad for result in results)

    with torch.no_grad():
        expected = [
            *_oracle_peak_normalize(ref, deg),
            _oracle_edge_ramp(ref),
            _oracle_align_scale(ref, deg),
            *_oracle_epilogue(distance, distance, 0.5),
        ]

    for got, want in zip(results, expected):
        torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_peak_normalize_deterministic():
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    ref = torch.randn(3, 100_000, device=DEVICE, generator=generator)
    deg = torch.randn(3, 100_000, device=DEVICE, generator=generator)

    first, second = peak_normalize(ref, deg), peak_normalize(ref, deg)

    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_peak_normalize_peak_position_sweep():
    """The maximum has to be found wherever it sits.

    ``n`` needs more than ``_NCHUNK`` blocks of ``_BLOCK`` samples and ends in a
    partial tile, so a wrong tail mask, a wrong grid stride in the first
    reduction stage or an off-by-one in the chunk indexing all drop the spike
    for at least one of the probed positions.
    """

    n = _BLOCK * _NCHUNK + 500
    positions = [
        0,
        1,
        _BLOCK - 1,
        _BLOCK,
        _BLOCK + 1,
        _BLOCK * (_NCHUNK - 1),
        _BLOCK * _NCHUNK - 1,
        _BLOCK * _NCHUNK,
        n - 2,
        n - 1,
    ]

    for pos in positions:
        for spike, tag in [(-7.0, "in ref"), (0.05, "not the maximum")]:
            ref = torch.full((2, n), 0.1, device=DEVICE)
            ref[:, pos] = spike
            deg = torch.full((2, n), 0.2, device=DEVICE)

            got = peak_normalize(ref, deg)
            expected = _oracle_peak_normalize(ref, deg)

            assert torch.equal(got[0], expected[0]), f"spike {tag} at {pos}"
            assert torch.equal(got[1], expected[1]), f"spike {tag} at {pos}"


def test_peak_normalize_backward_deterministic():
    generator = torch.Generator(device=DEVICE).manual_seed(31)
    ref = torch.randn(3, 100_003, device=DEVICE, generator=generator).requires_grad_(
        True
    )
    deg = torch.randn(3, 100_003, device=DEVICE, generator=generator).requires_grad_(
        True
    )
    upstream = torch.randn(3, 100_003, device=DEVICE, generator=generator)

    def once():
        out = peak_normalize(ref, deg)
        return torch.autograd.grad(
            (out[0] * upstream).sum() + (out[1] * upstream).sum(), [ref, deg]
        )

    first = once()
    for _ in range(4):
        again = once()
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


# ---------------------------------------------------------------------------
# edge ramp
# ---------------------------------------------------------------------------


def test_edge_ramp_forward():
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    worst = {}

    for batch, n in SHAPES + [(1, 30)]:
        for kind in ["randn", "tiny", "huge", "zeros", "signs"]:
            x = _signals(batch, n, kind, generator)
            _check(edge_ramp(x), _oracle_edge_ramp(x), "out", worst)

    print(f"\nedge_ramp forward max |abs|, |rel|: {worst}")
    assert worst["out"] == (0.0, 0.0)


def test_edge_ramp_non_contiguous():
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    dense = torch.randn(4, 800, device=DEVICE, generator=generator)

    x = dense[::2, 3:400]
    assert not x.is_contiguous()

    torch.testing.assert_close(edge_ramp(x), _oracle_edge_ramp(x), rtol=RTOL, atol=ATOL)


def test_edge_ramp_backward():
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    worst = {}

    for batch, n in [(1, 30), (2, 1023), (3, 5000)]:
        x = torch.randn(batch, n, device=DEVICE, generator=generator).requires_grad_(
            True
        )
        upstream = torch.randn(batch, n, device=DEVICE, generator=generator)

        (got,) = torch.autograd.grad((edge_ramp(x) * upstream).sum(), [x])
        (expected,) = torch.autograd.grad((_oracle_edge_ramp(x) * upstream).sum(), [x])

        _check(got, expected, "d_x", worst)

    print(f"\nedge_ramp backward max |abs|, |rel|: {worst}")
    assert worst["d_x"] == (0.0, 0.0)


def test_edge_ramp_rejects_short_input():
    """Below 15 samples the reference itself cannot be evaluated.

    ``signal[:, :15] * emp`` would fail to broadcast a 15 tap ramp onto a
    shorter slice, so there is nothing to be equivalent to.
    """

    for n in [1, 14]:
        with pytest.raises(RuntimeError, match="at least 15 samples"):
            edge_ramp(torch.randn(2, n, device=DEVICE))


def test_edge_ramp_short_inputs_apply_both_ramps():
    """For ``15 <= n < 30`` the two ramps overlap and *both* weights apply.

    The reference performs two sequential in place assignments, so a sample in
    the overlap is multiplied by ``(i + 1) / 16`` and then by ``(n - i) / 16``.
    A kernel that lets the trailing ramp overwrite the leading one -- the
    obvious ``tl.where`` formulation -- silently disagrees here.
    """

    generator = torch.Generator(device=DEVICE).manual_seed(30)

    for n in range(15, 30):
        x = torch.randn(3, n, device=DEVICE, generator=generator)
        got, expected = edge_ramp(x), _oracle_edge_ramp(x)

        assert torch.equal(got, expected), f"n={n}"

        # and the weights really are the product of the two ramps
        emp = torch.linspace(0, 15, 16, device=DEVICE)[1:] / 16.0
        head = torch.cat([emp, torch.ones(max(n - 15, 0), device=DEVICE)])[:n]
        tail = torch.cat([torch.ones(max(n - 15, 0), device=DEVICE), emp.flip(0)])[-n:]
        assert torch.equal(got, x * head * tail), f"n={n}"


def test_edge_ramp_matches_ref_preemphasize():
    """Composed with the pre-emphasize filter this is ``ref_preemphasize``."""

    from torchaudio.functional import lfilter

    generator = torch.Generator(device=DEVICE).manual_seed(21)
    model = PesqLoss(0.5, sample_rate=48000).to(DEVICE)

    signal = torch.randn(2, 16_000, device=DEVICE, generator=generator)

    got = lfilter(
        edge_ramp(signal), model.pre_filter[1], model.pre_filter[0], clamp=False
    )
    expected = ref_preemphasize(signal, model.pre_filter)

    abs_err, rel_err = _errors(got, expected)
    print(f"\nedge_ramp vs ref_preemphasize: abs {abs_err:.3e}, rel {rel_err:.3e}")

    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


def test_edge_ramp_backward_matches_ref_preemphasize():
    """Gradient of the composition, not just of the forward pass."""

    generator = torch.Generator(device=DEVICE).manual_seed(37)
    model = PesqLoss(0.5, sample_rate=48000).to(DEVICE)

    signal = torch.randn(2, 16_000, device=DEVICE, generator=generator)
    upstream = torch.randn(2, 16_000, device=DEVICE, generator=generator)

    got_in = signal.clone().requires_grad_(True)
    ramped = lfilter(
        edge_ramp(got_in), model.pre_filter[1], model.pre_filter[0], clamp=False
    )
    (got,) = torch.autograd.grad((ramped * upstream).sum(), [got_in])

    exp_in = signal.clone().requires_grad_(True)
    (expected,) = torch.autograd.grad(
        (ref_preemphasize(exp_in, model.pre_filter) * upstream).sum(), [exp_in]
    )

    abs_err, rel_err = _errors(got, expected)
    print(
        f"\nedge_ramp backward vs ref_preemphasize: abs {abs_err:.3e}, rel {rel_err:.3e}"
    )

    assert torch.equal(got, expected)


def test_edge_ramp_deterministic():
    generator = torch.Generator(device=DEVICE).manual_seed(8)
    x = torch.randn(2, 100_000, device=DEVICE, generator=generator)

    assert torch.equal(edge_ramp(x), edge_ramp(x))


# ---------------------------------------------------------------------------
# level alignment scaling
# ---------------------------------------------------------------------------


def test_align_scale_forward():
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    worst = {}

    for batch, n in SHAPES:
        for scale in [1e-8, 1.0, 1e3]:
            signal = torch.randn(batch, n, device=DEVICE, generator=generator) * scale
            filtered = torch.randn(batch, n, device=DEVICE, generator=generator) * scale

            _check(
                align_scale(signal, filtered),
                _oracle_align_scale(signal, filtered),
                "out",
                worst,
            )

    print(f"\nalign_scale forward max |abs|, |rel|: {worst}")


def test_align_scale_reduction_accuracy():
    """Compare the fp32 tree reduction against a float64 oracle.

    ``signal = 1`` turns the output into the broadcast scale factor, which is a
    direct read-out of the internal reduction. Reported for both the Triton
    kernel and ``torch.sum`` in fp32, against the same sum in float64.
    """

    generator = torch.Generator(device=DEVICE).manual_seed(10)

    # 1_000_000 makes every lane of the BLOCK wide accumulator fold ~15 values
    # serially, which is where a naive scalar accumulator would start to drift
    for n in [100_000, 160_000, 1_000_000]:
        filtered = torch.randn(2, n, device=DEVICE, generator=generator) * 1e3
        ones = torch.ones(2, n, device=DEVICE)

        triton_scale = align_scale(ones, filtered)[:, 0]
        torch_scale = _oracle_align_scale(ones, filtered)[:, 0]

        power64 = (filtered.double() ** 2).sum(dim=1) / (n + 5120) / 1.04684
        scale64 = (10**7 / power64).sqrt()

        rel_triton = ((triton_scale.double() - scale64) / scale64).abs().max().item()
        rel_torch = ((torch_scale.double() - scale64) / scale64).abs().max().item()

        print(
            f"\nalign_scale n={n} scale rel. error vs float64: "
            f"triton {rel_triton:.3e}, torch.sum {rel_torch:.3e}"
        )

        # one fp32 ulp is 6e-8, allow a couple of them for the whole reduction
        assert rel_triton < 5e-7


def test_align_scale_matches_ref_align_level():
    """Composed with the band pass filter this is ``ref_align_level``."""

    from torchaudio.functional import lfilter

    generator = torch.Generator(device=DEVICE).manual_seed(11)
    model = PesqLoss(0.5, sample_rate=48000).to(DEVICE)

    signal = torch.randn(2, 16_000, device=DEVICE, generator=generator)
    filtered = lfilter(
        signal, model.power_filter[1], model.power_filter[0], clamp=False
    )

    got = align_scale(signal, filtered)
    expected = ref_align_level(signal, model.power_filter)

    abs_err, rel_err = _errors(got, expected)
    print(f"\nalign_scale vs ref_align_level: abs {abs_err:.3e}, rel {rel_err:.3e}")

    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


def test_align_scale_backward():
    generator = torch.Generator(device=DEVICE).manual_seed(12)
    worst = {}

    for batch, n in [(1, 31), (2, 1023), (3, 5000), (2, 100_000)]:
        signal = torch.randn(batch, n, device=DEVICE, generator=generator)
        filtered = torch.randn(batch, n, device=DEVICE, generator=generator) * 2.0
        signal, filtered = signal.requires_grad_(True), filtered.requires_grad_(True)

        upstream = torch.randn(batch, n, device=DEVICE, generator=generator)

        got = torch.autograd.grad(
            (align_scale(signal, filtered) * upstream).sum(), [signal, filtered]
        )
        expected = torch.autograd.grad(
            (_oracle_align_scale(signal, filtered) * upstream).sum(),
            [signal, filtered],
        )

        _check(got[0], expected[0], "d_signal", worst)
        _check(got[1], expected[1], "d_filtered", worst)

        # only the filtered input needs a gradient
        (partial,) = torch.autograd.grad(
            (align_scale(signal.detach(), filtered) * upstream).sum(), [filtered]
        )
        _check(partial, expected[1], "d_filtered_only", worst)

        # only the signal needs a gradient
        (partial,) = torch.autograd.grad(
            (align_scale(signal, filtered.detach()) * upstream).sum(), [signal]
        )
        _check(partial, expected[0], "d_signal_only", worst)

    print(f"\nalign_scale backward max |abs|, |rel|: {worst}")


def test_align_scale_deterministic():
    generator = torch.Generator(device=DEVICE).manual_seed(13)
    signal = torch.randn(2, 100_000, device=DEVICE, generator=generator)
    filtered = torch.randn(2, 100_000, device=DEVICE, generator=generator)

    assert torch.equal(align_scale(signal, filtered), align_scale(signal, filtered))


def test_align_scale_backward_deterministic():
    generator = torch.Generator(device=DEVICE).manual_seed(32)
    signal = torch.randn(2, 100_003, device=DEVICE, generator=generator).requires_grad_(
        True
    )
    filtered = torch.randn(
        2, 100_003, device=DEVICE, generator=generator
    ).requires_grad_(True)
    upstream = torch.randn(2, 100_003, device=DEVICE, generator=generator)

    def once():
        return torch.autograd.grad(
            (align_scale(signal, filtered) * upstream).sum(), [signal, filtered]
        )

    first = once()
    for _ in range(4):
        again = once()
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


def test_align_scale_non_contiguous():
    """Both inputs and the upstream gradient may be strided views."""

    generator = torch.Generator(device=DEVICE).manual_seed(33)
    worst = {}

    signal = _noncontiguous(torch.randn(3, 1997, device=DEVICE, generator=generator))
    filtered = _noncontiguous(
        torch.randn(3, 1997, device=DEVICE, generator=generator) * 2.0
    )

    _check(
        align_scale(signal, filtered),
        _oracle_align_scale(signal, filtered),
        "fwd",
        worst,
    )

    signal = signal.detach().requires_grad_(True)
    filtered = filtered.detach().requires_grad_(True)
    upstream = _noncontiguous(torch.randn(3, 1997, device=DEVICE, generator=generator))

    got = torch.autograd.grad(
        (align_scale(signal, filtered) * upstream).sum(), [signal, filtered]
    )
    expected = torch.autograd.grad(
        (_oracle_align_scale(signal, filtered) * upstream).sum(), [signal, filtered]
    )

    _check(got[0], expected[0], "d_signal", worst)
    _check(got[1], expected[1], "d_filtered", worst)

    print(f"\nalign_scale non-contiguous max |abs|, |rel|: {worst}")


def test_align_scale_mismatched_lengths():
    """``signal`` and ``filtered`` only have to agree in the batch dimension.

    The power denominator uses ``filtered.shape[1]``, the scaling runs over
    ``signal.shape[1]`` and the dot product of the backward pass over
    ``signal.shape[1]`` again -- three different loop bounds that a single
    shared ``n`` would silently conflate.
    """

    generator = torch.Generator(device=DEVICE).manual_seed(34)
    worst = {}

    for n_sig, n_flt in [(1000, 3000), (3000, 1000), (31, 100_000), (100_000, 31)]:
        signal = torch.randn(2, n_sig, device=DEVICE, generator=generator)
        filtered = torch.randn(2, n_flt, device=DEVICE, generator=generator) * 1.5

        _check(
            align_scale(signal, filtered),
            _oracle_align_scale(signal, filtered),
            "fwd",
            worst,
        )

        signal = signal.requires_grad_(True)
        filtered = filtered.requires_grad_(True)
        upstream = torch.randn(2, n_sig, device=DEVICE, generator=generator)

        got = torch.autograd.grad(
            (align_scale(signal, filtered) * upstream).sum(), [signal, filtered]
        )
        expected = torch.autograd.grad(
            (_oracle_align_scale(signal, filtered) * upstream).sum(),
            [signal, filtered],
        )

        _check(got[0], expected[0], "d_signal", worst)
        _check(got[1], expected[1], "d_filtered", worst)

    print(f"\nalign_scale mismatched lengths max |abs|, |rel|: {worst}")


def test_align_scale_adversarial_magnitudes():
    """Degenerate levels have to degenerate the same way as the oracle.

    ``power`` is zero for a digitally silent signal -- which is reachable, a
    silent reference is a legal input -- so ``scale`` is infinite and the
    output is ``+-inf``, or ``NaN`` wherever the signal itself is zero. Both
    the forward and the backward pass have to reproduce that exactly.

    Levels of ``1e-25`` and ``1e18`` additionally underflow respectively
    overflow the sum of squares. The forward passes still agree there; the
    backward passes do not, but the kernel is the *more* accurate of the two,
    see :func:`test_align_scale_backward_conditioning_sweep`.
    """

    generator = torch.Generator(device=DEVICE).manual_seed(35)
    n = 5000

    for factor in [0.0, 1e-25, 1e18]:
        signal = torch.randn(2, n, device=DEVICE, generator=generator) * factor
        filtered = torch.randn(2, n, device=DEVICE, generator=generator) * factor

        torch.testing.assert_close(
            align_scale(signal, filtered),
            _oracle_align_scale(signal, filtered),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
        )

    # the reachable degenerate case: an exactly silent input, gradients included
    zeros = torch.zeros(2, n, device=DEVICE, requires_grad=True)
    upstream = torch.randn(2, n, device=DEVICE, generator=generator)

    got = torch.autograd.grad((align_scale(zeros, zeros) * upstream).sum(), [zeros])
    expected = torch.autograd.grad(
        (_oracle_align_scale(zeros, zeros) * upstream).sum(), [zeros]
    )
    torch.testing.assert_close(
        got[0], expected[0], rtol=RTOL, atol=ATOL, equal_nan=True
    )

    # a silent band pass output with a non-silent signal: pure +-inf, no NaN
    signal = torch.randn(2, n, device=DEVICE, generator=generator)
    filtered = torch.zeros(2, n, device=DEVICE)

    got = align_scale(signal, filtered)

    assert torch.equal(got, _oracle_align_scale(signal, filtered))
    assert not torch.isnan(got).any()


def test_align_scale_backward_conditioning_sweep():
    """``grad_filtered`` has to stay accurate over the whole usable level range.

    The kernel evaluates ``d scale / d power`` as ``-0.5 * scale / power``
    while autograd evaluates the algebraically equal
    ``0.5 / scale * (-1e7 / power**2)``. They agree to ~7e-7 relative wherever
    both are representable, but the kernel's form has a much wider dynamic
    range, because it never forms ``power**2``:

    ====================  ========================  ====================
    ``power``             float32 autograd          this kernel
    ====================  ========================  ====================
    ``< 4e-23``           ``1e7 / power**2`` inf    overflows too
    ``4e-23 .. 4e-19``    ``1e7 / power**2`` inf    accurate
    ``4e-19 .. 4e+19``    accurate                  accurate
    ``4e+19 .. 4e+24``    underflows towards zero   accurate
    ``> 4e+24``           underflows to zero        underflows too
    ====================  ========================  ====================

    Both parts are checked below: parity with float32 autograd inside the
    common range, and agreement with a float64 evaluation of the same
    expression outside it, where float32 autograd is demonstrably wrong. The
    pipeline itself lives at ``power`` around one -- ``align_scale`` is what
    puts it there -- so the whole table is head room.
    """

    n = 4000
    worst = 0.0

    for exponent in range(-8, 10):
        generator = torch.Generator(device=DEVICE).manual_seed(41)
        signal = torch.randn(2, n, device=DEVICE, generator=generator)
        filtered = (
            torch.randn(2, n, device=DEVICE, generator=generator) * 10.0**exponent
        )
        upstream = torch.randn(2, n, device=DEVICE, generator=generator)

        inputs = [signal.requires_grad_(True), filtered.requires_grad_(True)]
        got = torch.autograd.grad((align_scale(*inputs) * upstream).sum(), inputs)
        expected = torch.autograd.grad(
            (_oracle_align_scale(*inputs) * upstream).sum(), inputs
        )

        for a, b in zip(got, expected):
            worst = max(worst, _errors(a, b)[1])
            torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)

    print(f"\nalign_scale backward level sweep 1e-8..1e9, max rel: {worst:.3e}")

    # outside that range float32 autograd breaks down and the kernel does not
    for exponent in [-10, 11, 12]:
        generator = torch.Generator(device=DEVICE).manual_seed(41)
        signal = torch.randn(2, n, device=DEVICE, generator=generator)
        filtered = (
            torch.randn(2, n, device=DEVICE, generator=generator) * 10.0**exponent
        )
        upstream = torch.randn(2, n, device=DEVICE, generator=generator)

        truth_inputs = [
            signal.double().requires_grad_(True),
            filtered.double().requires_grad_(True),
        ]
        _, truth = torch.autograd.grad(
            (_oracle_align_scale(*truth_inputs) * upstream.double()).sum(), truth_inputs
        )
        peak = truth.abs().max()

        inputs = [signal.requires_grad_(True), filtered.requires_grad_(True)]
        _, kernel = torch.autograd.grad((align_scale(*inputs) * upstream).sum(), inputs)
        _, autograd = torch.autograd.grad(
            (_oracle_align_scale(*inputs) * upstream).sum(), inputs
        )

        def relative(value):
            # measured against the peak of the gradient, so that a single
            # near-zero entry cannot dominate the ratio
            return ((value.double() - truth).abs().max() / peak).item()

        print(
            f"  level 1e{exponent:+03d}: kernel vs float64 {relative(kernel):.3e}, "
            f"float32 autograd vs float64 {relative(autograd):.3e}"
        )

        assert relative(kernel) < 1e-5
        assert relative(autograd) > 1e-3


def test_align_scale_backward_through_bandpass():
    """The real usage: ``filtered`` is a linear function of ``signal``.

    Both gradient paths then meet in the same leaf, which is where a sign error
    or a missing factor in ``grad_filtered`` stops cancelling out.

    This one cannot be compared at ``rtol=1e-5`` element by element, and the
    reason is not the kernel. ``grad_filtered`` is fed into the backward pass
    of ``torchaudio.functional.lfilter``, a fifth order Butterworth band pass
    run in reverse over 16000 samples, and that recursion is wildly
    ill-conditioned: perturbing *its* upstream gradient by ``1e-7`` relative
    moves its output by ``~8e-3`` relative, an amplification of order ``1e5``.
    The test therefore measures that amplification on the pure PyTorch path and
    requires the kernel's deviation to stay below it -- a bound the kernel beats
    by more than two orders of magnitude -- plus a fixed regression ceiling.
    """

    generator = torch.Generator(device=DEVICE).manual_seed(36)
    model = PesqLoss(0.5, sample_rate=48000).to(DEVICE)

    signal = torch.randn(2, 16_000, device=DEVICE, generator=generator)
    upstream = torch.randn(2, 16_000, device=DEVICE, generator=generator)

    def reference(weights):
        leaf = signal.clone().requires_grad_(True)
        return torch.autograd.grad(
            (ref_align_level(leaf, model.power_filter) * weights).sum(), [leaf]
        )[0]

    expected = reference(upstream)

    leaf = signal.clone().requires_grad_(True)
    filtered = lfilter(leaf, model.power_filter[1], model.power_filter[0], clamp=False)
    (got,) = torch.autograd.grad((align_scale(leaf, filtered) * upstream).sum(), [leaf])

    def deviation(value):
        return ((value - expected).norm() / expected.norm()).item()

    # calibration: the kernel and autograd agree on grad_filtered to ~1e-6
    # relative, so this is how far apart the IIR backward can push them
    budget = deviation(reference(upstream * (1.0 + 1e-6)))
    observed = deviation(got)

    print(
        f"\nalign_scale backward vs ref_align_level: relative L2 {observed:.3e}, "
        f"lfilter-backward budget for a 1e-6 upstream change {budget:.3e}"
    )

    assert observed < budget
    assert observed < 1e-4

    # the split form itself is bit exact, so the deviation really is the IIR
    leaf = signal.clone().requires_grad_(True)
    filtered = lfilter(leaf, model.power_filter[1], model.power_filter[0], clamp=False)
    (split,) = torch.autograd.grad(
        (_oracle_align_scale(leaf, filtered) * upstream).sum(), [leaf]
    )
    assert torch.equal(split, expected)


# ---------------------------------------------------------------------------
# MOS epilogue
# ---------------------------------------------------------------------------


def test_pesq_epilogue_forward():
    generator = torch.Generator(device=DEVICE).manual_seed(14)
    worst = {}

    # d_symm and d_asymm are clamped to [1e-20, 45] by the chain, the 200 is
    # well outside of the reachable range
    cases = [
        torch.zeros(1, device=DEVICE),
        torch.full((3,), 1e-20, device=DEVICE),
        torch.full((3,), 45.0, device=DEVICE),
        torch.full((3,), 200.0, device=DEVICE),
        torch.rand(64, device=DEVICE, generator=generator) * 45.0,
    ]

    for factor in [0.5, 1.0, 12.5]:
        for d_symm in cases:
            d_asymm = torch.flip(d_symm, dims=(0,)) * 0.5
            got = pesq_epilogue(d_symm, d_asymm, factor)
            expected = _oracle_epilogue(d_symm, d_asymm, factor)

            _check(got[0], expected[0], "mos", worst)
            _check(got[1], expected[1], "loss", worst)

    print(f"\npesq_epilogue forward max |abs|, |rel|: {worst}")


def test_pesq_epilogue_non_contiguous():
    generator = torch.Generator(device=DEVICE).manual_seed(15)
    dense = torch.rand(64, device=DEVICE, generator=generator) * 40.0

    d_symm, d_asymm = dense[::2], dense[1::2]
    assert not d_symm.is_contiguous()

    got = pesq_epilogue(d_symm, d_asymm, 0.5)
    expected = _oracle_epilogue(d_symm, d_asymm, 0.5)

    torch.testing.assert_close(got[0], expected[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(got[1], expected[1], rtol=RTOL, atol=ATOL)


def test_pesq_epilogue_backward():
    generator = torch.Generator(device=DEVICE).manual_seed(16)
    worst = {}

    for batch in [1, 7, 64]:
        d_symm = (
            torch.rand(batch, device=DEVICE, generator=generator) * 45.0
        ).requires_grad_(True)
        d_asymm = (
            torch.rand(batch, device=DEVICE, generator=generator) * 45.0
        ).requires_grad_(True)

        g_mos = torch.randn(batch, device=DEVICE, generator=generator)
        g_loss = torch.randn(batch, device=DEVICE, generator=generator)

        for use_mos, use_loss, name in [
            (True, True, "both"),
            (True, False, "mos"),
            (False, True, "loss"),
        ]:
            mos, loss = pesq_epilogue(d_symm, d_asymm, 0.5)
            exp_mos, exp_loss = _oracle_epilogue(d_symm, d_asymm, 0.5)

            objective = torch.zeros((), device=DEVICE)
            exp_objective = torch.zeros((), device=DEVICE)
            if use_mos:
                objective = objective + (mos * g_mos).sum()
                exp_objective = exp_objective + (exp_mos * g_mos).sum()
            if use_loss:
                objective = objective + (loss * g_loss).sum()
                exp_objective = exp_objective + (exp_loss * g_loss).sum()

            got = torch.autograd.grad(objective, [d_symm, d_asymm])
            expected = torch.autograd.grad(exp_objective, [d_symm, d_asymm])

            _check(got[0], expected[0], f"d_symm/{name}", worst)
            _check(got[1], expected[1], f"d_asymm/{name}", worst)

        # only one of the inputs needs a gradient
        mos, loss = pesq_epilogue(d_symm.detach(), d_asymm, 0.5)
        (partial,) = torch.autograd.grad((mos * g_mos).sum(), [d_asymm])
        exp_mos, _ = _oracle_epilogue(d_symm.detach(), d_asymm, 0.5)
        (exp_partial,) = torch.autograd.grad((exp_mos * g_mos).sum(), [d_asymm])
        _check(partial, exp_partial, "d_asymm/only", worst)

    print(f"\npesq_epilogue backward max |abs|, |rel|: {worst}")


def test_pesq_epilogue_matches_loss_module():
    """The epilogue reproduces ``PesqLoss.mos`` and ``PesqLoss.forward``."""

    generator = torch.Generator(device=DEVICE).manual_seed(17)
    model = PesqLoss(2.5, sample_rate=48000).to(DEVICE)

    d_symm = torch.rand(5, device=DEVICE, generator=generator) * 20.0
    d_asymm = torch.rand(5, device=DEVICE, generator=generator) * 20.0

    mos, loss = pesq_epilogue(d_symm, d_asymm, model.factor)

    reference_mos = 4.5 - 0.1 * d_symm - 0.0309 * d_asymm
    reference_mos = 0.999 + 4 / (1 + torch.exp(-1.3669 * reference_mos + 3.8224))

    torch.testing.assert_close(mos, reference_mos, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        loss, model.factor * (0.1 * d_symm + 0.0309 * d_asymm), rtol=RTOL, atol=ATOL
    )


def test_pesq_epilogue_deterministic():
    generator = torch.Generator(device=DEVICE).manual_seed(18)
    d_symm = torch.rand(64, device=DEVICE, generator=generator) * 45.0
    d_asymm = torch.rand(64, device=DEVICE, generator=generator) * 45.0

    first = pesq_epilogue(d_symm, d_asymm, 0.5)
    second = pesq_epilogue(d_symm, d_asymm, 0.5)

    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_pesq_epilogue_batch_sweep():
    """Batch sizes around the ``_BLOCK_B == 128`` element tile of the kernel."""

    generator = torch.Generator(device=DEVICE).manual_seed(38)
    worst = {}

    for batch in [1, 2, 127, 128, 129, 255, 256, 1000]:
        d_symm = torch.rand(batch, device=DEVICE, generator=generator) * 45.0
        d_asymm = torch.rand(batch, device=DEVICE, generator=generator) * 45.0

        got = pesq_epilogue(d_symm, d_asymm, 2.5)
        expected = _oracle_epilogue(d_symm, d_asymm, 2.5)

        _check(got[0], expected[0], "mos", worst)
        _check(got[1], expected[1], "loss", worst)

        d_symm = d_symm.requires_grad_(True)
        d_asymm = d_asymm.requires_grad_(True)
        g_mos = torch.randn(batch, device=DEVICE, generator=generator)
        g_loss = torch.randn(batch, device=DEVICE, generator=generator)

        mos, loss = pesq_epilogue(d_symm, d_asymm, 2.5)
        exp_mos, exp_loss = _oracle_epilogue(d_symm, d_asymm, 2.5)

        got = torch.autograd.grad(
            (mos * g_mos).sum() + (loss * g_loss).sum(), [d_symm, d_asymm]
        )
        expected = torch.autograd.grad(
            (exp_mos * g_mos).sum() + (exp_loss * g_loss).sum(), [d_symm, d_asymm]
        )

        _check(got[0], expected[0], "d_symm", worst)
        _check(got[1], expected[1], "d_asymm", worst)

    print(f"\npesq_epilogue batch sweep max |abs|, |rel|: {worst}")


def test_pesq_epilogue_rejects_differentiable_factor():
    """A tensor ``factor`` would silently receive no gradient, so reject it."""

    d_symm = torch.rand(4, device=DEVICE) * 10.0
    d_asymm = torch.rand(4, device=DEVICE) * 10.0

    factor = torch.tensor(0.5, device=DEVICE, requires_grad=True)
    with pytest.raises(RuntimeError, match="treats `factor` as a constant"):
        pesq_epilogue(d_symm, d_asymm, factor)

    # a constant tensor is fine, it is just a scalar
    mos, loss = pesq_epilogue(d_symm, d_asymm, factor.detach())
    torch.testing.assert_close(
        loss, 0.5 * (0.1 * d_symm + 0.0309 * d_asymm), rtol=RTOL, atol=ATOL
    )
    assert torch.isfinite(mos).all()


# ---------------------------------------------------------------------------
# cross cutting: strided gradients, degenerate shapes, input validation
# ---------------------------------------------------------------------------


def test_backward_accepts_non_contiguous_upstream_gradients():
    """Every backward pass gets a strided ``grad_output``.

    An upstream gradient produced by a slicing op is the common case in a real
    graph; a kernel that indexes it as if it were dense reads every other
    element and produces a plausible looking but wrong result.
    """

    generator = torch.Generator(device=DEVICE).manual_seed(39)
    worst = {}
    batch, n = 3, 2999

    ref = torch.randn(batch, n, device=DEVICE, generator=generator).requires_grad_(True)
    deg = torch.randn(batch, n, device=DEVICE, generator=generator).requires_grad_(True)
    g_ref = _noncontiguous(torch.randn(batch, n, device=DEVICE, generator=generator))
    g_deg = _noncontiguous(torch.randn(batch, n, device=DEVICE, generator=generator))

    out = peak_normalize(ref, deg)
    got = torch.autograd.grad(
        (out[0] * g_ref).sum() + (out[1] * g_deg).sum(), [ref, deg]
    )
    exp_out = _oracle_peak_normalize(ref, deg)
    expected = torch.autograd.grad(
        (exp_out[0] * g_ref).sum() + (exp_out[1] * g_deg).sum(), [ref, deg]
    )
    _check(got[0], expected[0], "peak/d_ref", worst)
    _check(got[1], expected[1], "peak/d_deg", worst)

    (got_x,) = torch.autograd.grad((edge_ramp(ref) * g_ref).sum(), [ref])
    (exp_x,) = torch.autograd.grad((_oracle_edge_ramp(ref) * g_ref).sum(), [ref])
    _check(got_x, exp_x, "ramp/d_x", worst)

    got = torch.autograd.grad((align_scale(ref, deg) * g_ref).sum(), [ref, deg])
    expected = torch.autograd.grad(
        (_oracle_align_scale(ref, deg) * g_ref).sum(), [ref, deg]
    )
    _check(got[0], expected[0], "align/d_signal", worst)
    _check(got[1], expected[1], "align/d_filtered", worst)

    d_symm = (
        torch.rand(batch, device=DEVICE, generator=generator) * 45.0
    ).requires_grad_(True)
    d_asymm = (
        torch.rand(batch, device=DEVICE, generator=generator) * 45.0
    ).requires_grad_(True)
    strided = torch.randn(2 * batch, device=DEVICE, generator=generator)[::2]
    assert not strided.is_contiguous()

    mos, loss = pesq_epilogue(d_symm, d_asymm, 0.5)
    got = torch.autograd.grad(
        (mos * strided).sum() + (loss * strided).sum(), [d_symm, d_asymm]
    )
    exp_mos, exp_loss = _oracle_epilogue(d_symm, d_asymm, 0.5)
    expected = torch.autograd.grad(
        (exp_mos * strided).sum() + (exp_loss * strided).sum(), [d_symm, d_asymm]
    )
    _check(got[0], expected[0], "epilogue/d_symm", worst)
    _check(got[1], expected[1], "epilogue/d_asymm", worst)

    print(f"\nnon-contiguous upstream gradients max |abs|, |rel|: {worst}")


def test_empty_batch():
    """A zero sized batch must produce zero sized outputs, not crash."""

    empty = torch.zeros(0, 512, device=DEVICE)

    ref_out, deg_out = peak_normalize(empty, empty)
    assert ref_out.shape == (0, 512) and deg_out.shape == (0, 512)
    assert edge_ramp(empty).shape == (0, 512)
    assert align_scale(empty, empty).shape == (0, 512)

    mos, loss = pesq_epilogue(
        torch.zeros(0, device=DEVICE), torch.zeros(0, device=DEVICE), 0.5
    )
    assert mos.shape == (0,) and loss.shape == (0,)


def test_input_validation():
    """Bad inputs raise instead of reading out of bounds or returning junk."""

    good = torch.randn(2, 512, device=DEVICE)

    cases = [
        (lambda: peak_normalize(good.double(), good.double()), "float32"),
        (lambda: peak_normalize(good.cpu(), good.cpu()), "GPU"),
        (lambda: peak_normalize(good[0], good[0]), r"\[batch, sample\]"),
        (
            lambda: peak_normalize(good.unsqueeze(0), good.unsqueeze(0)),
            r"\[batch, sample\]",
        ),
        (lambda: peak_normalize(good, good[:, :256]), "matching shapes"),
        (lambda: peak_normalize(good[:, :0], good[:, :0]), "at least one sample"),
        (lambda: edge_ramp(good[0]), r"\[batch, sample\]"),
        (lambda: align_scale(good, good[:1]), "matching batch size"),
        (lambda: align_scale(good[:, :0], good), "at least one sample"),
        (lambda: pesq_epilogue(good, good, 0.5), r"\[batch\] tensors"),
        (
            lambda: pesq_epilogue(good[0], good[0, :256], 0.5),
            "matching shapes",
        ),
    ]

    for call, message in cases:
        with pytest.raises(RuntimeError, match=message):
            call()


# ---------------------------------------------------------------------------
# the omitted peak normalisation term really is zero
# ---------------------------------------------------------------------------


def _pipeline_tail(ref, deg, model):
    """Everything :func:`ref_pipeline` does *after* the peak normalisation."""

    deg, ref = model.resampler(deg), model.resampler(ref)

    ref = ref_align_level(ref, model.power_filter)
    deg = ref_align_level(deg, model.power_filter)
    ref = ref_preemphasize(ref, model.pre_filter)
    deg = ref_preemphasize(deg, model.pre_filter)

    deg = torch.nn.functional.pad(deg, (0, deg.shape[1] % 256))
    ref = torch.nn.functional.pad(ref, (0, ref.shape[1] % 256))

    spec_args = (
        model.to_spec.window,
        model.fbank.fbank,
        model.fbank.pow_dens_correction,
        model.to_spec.n_fft,
        model.to_spec.hop_length,
    )

    return ref_chain(
        ref_stft_bark(ref, *spec_args),
        ref_stft_bark(deg, *spec_args),
        model.loudness.threshs,
        model.loudness.exp,
        model.fbank.width_bark,
        model.fbank.total_width,
        0.1866055,
    )


def _autograd_peak_normalize(ref, deg):
    """What ``ref_pipeline`` does: ``max_val`` is *not* detached."""

    max_val = torch.max(
        torch.amax(deg.abs(), dim=1, keepdim=True),
        torch.amax(ref.abs(), dim=1, keepdim=True),
    )

    return ref / max_val, deg / max_val


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_peak_normalize_detach_matches_full_pipeline_gradient(dtype):
    """Detaching ``max_val`` may not change the gradient of the pipeline.

    This is the direct form of the argument: run the *whole* pipeline twice,
    once with the peak amplitude differentiated through and once with it
    detached, and compare ``dL/dref`` and ``dL/ddeg`` element by element. The
    float64 run is the actual proof -- the two gradients agree to ~1e-15
    relative, i.e. the dropped term is analytically zero, because everything
    downstream of ``align_level`` depends on each signal only through its
    direction, never its amplitude.

    The float32 run additionally checks the Triton kernel itself: its gradient
    has to be as close to full autograd as the pure PyTorch detached variant
    is, no worse. The residual there is float32 round-off of a chain
    containing a resampler, two IIR filters, an STFT and several clamps, not a
    missing term -- which is exactly what the float64 run establishes.
    """

    generator = torch.Generator(device=DEVICE).manual_seed(40)
    model = PesqLoss(0.5, sample_rate=48000).to(DEVICE).to(dtype)

    noise = torch.randn(2, 24_000, device=DEVICE, generator=generator).to(dtype)
    base_ref = (torch.randn(2, 24_000, device=DEVICE, generator=generator) * 0.4).to(
        dtype
    )
    base_deg = (base_ref + 0.15 * noise).detach()
    base_ref = base_ref.detach()

    # random weights, so the two distances cannot cancel each other
    weights = torch.randn(2, 2, device=DEVICE, generator=generator).to(dtype)

    def gradients(normalize):
        ref = base_ref.clone().requires_grad_(True)
        deg = base_deg.clone().requires_grad_(True)

        normalized_ref, normalized_deg = normalize(ref, deg)
        d_symm, d_asymm = _pipeline_tail(normalized_ref, normalized_deg, model)
        objective = (d_symm * weights[0]).sum() + (d_asymm * weights[1]).sum()

        return torch.autograd.grad(objective, [ref, deg])

    def detached(ref, deg):
        max_val = torch.max(
            torch.amax(deg.abs(), dim=1, keepdim=True),
            torch.amax(ref.abs(), dim=1, keepdim=True),
        ).detach()
        return ref / max_val, deg / max_val

    exact = gradients(_autograd_peak_normalize)
    approx = gradients(detached)

    def relative(a, b):
        return ((a - b).double().norm() / b.double().norm()).item()

    for name, got, want in zip(("d_ref", "d_deg"), approx, exact):
        error = relative(got, want)
        print(f"\n{dtype} detached vs autograd {name}: relative L2 {error:.3e}")
        # float64 pins the analytic statement, float32 only bounds round-off
        assert error < (1e-4 if dtype == torch.float32 else 1e-12)

    if dtype != torch.float32:
        return

    triton = gradients(peak_normalize)

    for name, got, want, reference in zip(("d_ref", "d_deg"), triton, exact, approx):
        error, pytorch_error = relative(got, want), relative(reference, want)
        print(
            f"triton vs autograd {name}: relative L2 {error:.3e} "
            f"(pytorch detached: {pytorch_error:.3e}), "
            f"triton vs pytorch detached: {relative(got, reference):.3e}"
        )

        # the kernel may not be worse than the PyTorch detached variant, and
        # the two detached variants have to agree far more tightly than either
        # agrees with full autograd
        assert error <= 2.0 * pytorch_error + 1e-9
        assert relative(got, reference) < 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pipeline_is_invariant_to_reference_scaling(dtype):
    """``peak_normalize`` may detach ``max_val`` -- here is the proof.

    ``ref_pipeline(c * ref, deg, model)`` does not depend on ``c`` because
    ``align_level`` rescales both signals to a fixed band power. Consequently
    ``d/dc`` at ``c = 1`` is analytically zero and the term that
    :class:`_PeakNormalize` drops in the backward pass cannot contribute.

    Measured relative to ``||dL/dref|| * ||ref||``, the Cauchy-Schwarz bound on
    ``dL/dc = <dL/dref, ref>``.
    """

    generator = torch.Generator(device=DEVICE).manual_seed(19)
    model = PesqLoss(0.5, sample_rate=48000).to(DEVICE).to(dtype)

    noise = torch.randn(2, 24_000, device=DEVICE, generator=generator).to(dtype)
    ref = torch.randn(2, 24_000, device=DEVICE, generator=generator).to(dtype) * 0.4
    deg = (ref + 0.15 * noise).detach()
    ref = ref.detach().requires_grad_(True)

    # random weights, so no cancellation between the two distances
    weights = torch.randn(2, 2, device=DEVICE, generator=generator).to(dtype)

    factor = torch.tensor(1.0, device=DEVICE, dtype=dtype, requires_grad=True)
    d_symm, d_asymm = ref_pipeline(factor * ref, deg, model)
    objective = (d_symm * weights[0]).sum() + (d_asymm * weights[1]).sum()

    grad_factor, grad_ref = torch.autograd.grad(objective, [factor, ref])
    bound = (grad_ref.norm() * ref.norm()).item()
    ratio = abs(grad_factor.item()) / bound

    print(
        f"\n{dtype}: |dL/dc| = {abs(grad_factor.item()):.3e}, "
        f"||dL/dref|| * ||ref|| = {bound:.3e}, ratio = {ratio:.3e}"
    )

    # pure round-off, the analytic value is exactly zero
    assert ratio < (1e-7 if dtype == torch.float32 else 1e-15)

    with torch.no_grad():
        scaled = ref_pipeline(3.7 * ref, deg, model)

    # the same statement without derivatives: the distances do not move. In
    # float32 the long chain (resampling, two IIR filters, STFT, clamps)
    # amplifies the round-off of the changed peak amplitude to ~1e-4 relative,
    # in float64 the invariance is exact to 1e-13
    for value, reference in zip(scaled, (d_symm, d_asymm)):
        rel = ((value - reference).abs() / reference.abs()).max().item()
        print(f"  rescaled by 3.7, max relative output change: {rel:.3e}")
        assert rel < (1e-3 if dtype == torch.float32 else 1e-12)


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------


def test_report_timings():
    """Not an assertion, prints forward and backward timings vs the oracle.

    These ops move a few megabytes and are entirely launch-latency bound: a
    bare Triton launch costs ~8.5 us of CPU time on this machine while the
    actual device work for 2.5 MB is ~10 us, so the CUDA event time and the
    wall clock time coincide. Treat the medians as noise-prone -- the GPU is
    shared -- and the minima as the uncontended cost.
    """

    generator = torch.Generator(device=DEVICE).manual_seed(20)
    batch, n = 4, 160_000  # 10 s of 16 kHz audio

    ref = torch.randn(batch, n, device=DEVICE, generator=generator)
    deg = torch.randn(batch, n, device=DEVICE, generator=generator)
    upstream = torch.randn(batch, n, device=DEVICE, generator=generator)
    d_symm = torch.rand(batch, device=DEVICE, generator=generator) * 45.0
    d_asymm = torch.rand(batch, device=DEVICE, generator=generator) * 45.0

    def backward_of(fn, inputs, grads):
        def run():
            outputs = fn()
            outputs = outputs if isinstance(outputs, tuple) else (outputs,)
            objective = sum((out * g).sum() for out, g in zip(outputs, grads))
            torch.autograd.grad(objective, inputs)

        return run

    grad_ref, grad_deg = ref.requires_grad_(True), deg.requires_grad_(True)
    grad_symm = d_symm.clone().requires_grad_(True)
    grad_asymm = d_asymm.clone().requires_grad_(True)
    upstream_b = torch.randn(batch, device=DEVICE, generator=generator)

    rows = [
        (
            "peak_normalize fwd",
            _timeit(lambda: peak_normalize(ref.detach(), deg.detach())),
            _timeit(lambda: _oracle_peak_normalize(ref.detach(), deg.detach())),
        ),
        (
            "peak_normalize bwd",
            _timeit(
                backward_of(
                    lambda: peak_normalize(grad_ref, grad_deg),
                    [grad_ref, grad_deg],
                    [upstream, upstream],
                )
            ),
            _timeit(
                backward_of(
                    lambda: _oracle_peak_normalize(grad_ref, grad_deg),
                    [grad_ref, grad_deg],
                    [upstream, upstream],
                )
            ),
        ),
        (
            "edge_ramp fwd",
            _timeit(lambda: edge_ramp(ref.detach())),
            _timeit(lambda: _oracle_edge_ramp(ref.detach())),
        ),
        (
            "edge_ramp bwd",
            _timeit(backward_of(lambda: edge_ramp(grad_ref), [grad_ref], [upstream])),
            _timeit(
                backward_of(lambda: _oracle_edge_ramp(grad_ref), [grad_ref], [upstream])
            ),
        ),
        (
            "align_scale fwd",
            _timeit(lambda: align_scale(ref.detach(), deg.detach())),
            _timeit(lambda: _oracle_align_scale(ref.detach(), deg.detach())),
        ),
        (
            "align_scale bwd",
            _timeit(
                backward_of(
                    lambda: align_scale(grad_ref, grad_deg),
                    [grad_ref, grad_deg],
                    [upstream],
                )
            ),
            _timeit(
                backward_of(
                    lambda: _oracle_align_scale(grad_ref, grad_deg),
                    [grad_ref, grad_deg],
                    [upstream],
                )
            ),
        ),
        (
            "pesq_epilogue fwd",
            _timeit(lambda: pesq_epilogue(d_symm, d_asymm, 0.5)),
            _timeit(lambda: _oracle_epilogue(d_symm, d_asymm, 0.5)),
        ),
        (
            "pesq_epilogue bwd",
            _timeit(
                backward_of(
                    lambda: pesq_epilogue(grad_symm, grad_asymm, 0.5),
                    [grad_symm, grad_asymm],
                    [upstream_b, upstream_b],
                )
            ),
            _timeit(
                backward_of(
                    lambda: _oracle_epilogue(grad_symm, grad_asymm, 0.5),
                    [grad_symm, grad_asymm],
                    [upstream_b, upstream_b],
                )
            ),
        ),
    ]

    print(f"\ntimings for batch={batch}, n={n} [ms/call, 10 warmups + 25x10]")
    header = f"{'op':<20}{'triton':>19}{'pytorch':>19}{'speedup':>18}"
    print(header)
    print(
        f"{'':<20}{'median':>10}{'min':>9}{'median':>10}{'min':>9}{'median':>9}{'min':>9}"
    )
    for name, (triton_ms, triton_lo), (torch_ms, torch_lo) in rows:
        print(
            f"{name:<20}{triton_ms:>10.3f}{triton_lo:>9.3f}"
            f"{torch_ms:>10.3f}{torch_lo:>9.3f}"
            f"{torch_ms / triton_ms:>8.2f}x{torch_lo / triton_lo:>8.2f}x"
        )

    assert all(math.isfinite(triton_ms) for _, (triton_ms, _), _ in rows)
