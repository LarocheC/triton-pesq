"""Tests for the Triton IIR filter.

Tolerances
----------
The oracle named in the contract,
``torchaudio.functional.lfilter(x, a, b, clamp=False)``, evaluates a
direct-form-I recursion in float32.  For the order 10 Butterworth bandpass used
by PESQ that recursion is badly conditioned: measured against the *same* call
in float64 it has a peak relative error of ``8.5e-3`` forward and ``1.5e-2``
backward.  :class:`TritonIIR` uses a cascaded second order state space
realisation and reaches ``2.4e-7`` on the same input, so a direct
float32-vs-float32 comparison can only ever reproduce the error of the oracle,
not of the code under test.

Every parity test therefore compares both implementations against a **float64**
``lfilter`` and requires

  1. the Triton filter to be within :data:`TOL` of the float64 result, and
  2. the Triton filter to be at least as accurate as the float32 oracle, down
     to the float32 floor of the chunked matmul (:data:`FLOOR`).

Errors are relative to the peak magnitude of the reference, which is the
meaningful scale for a filter output; a per-element relative error is
meaningless where the reference crosses zero.
"""

import math

import numpy as np
import pytest
import torch

from scipy.signal import butter, cheby1, ellip
from torchaudio.functional import lfilter

from torch_pesq.triton_ops._common import HAS_TRITON, next_power_of_two

#: Relative to peak error the Triton filter has to stay below.  Measured worst
#: case over both filters, both methods, chunk sizes 64/128/256/512, 18 shapes
#: and forward plus backward is 2.03e-6, dominated by float32 rounding of the
#: chunk matmul; 1e-5 leaves headroom for the tuner settling on a different
#: tile size.
TOL = 1e-5

#: Accuracy the module reaches on *any* input, i.e. the float32 rounding floor
#: of the chunked matmul (measured 2.03e-6).  Used to make "at least as
#: accurate as the oracle" a real assertion: whenever the float32 oracle is
#: better than this floor the module only has to reach the floor, everywhere
#: else it has to actually beat the oracle.
FLOOR = 3e-6

CUDA = torch.cuda.is_available() and HAS_TRITON

pytestmark = pytest.mark.skipif(
    not CUDA, reason="the Triton IIR filter needs a GPU and the triton package"
)

if CUDA:
    from torch_pesq.triton_ops import iir as iir_module
    from torch_pesq.triton_ops.iir import TritonIIR

#: The two filters the PESQ pipeline actually uses, at 16kHz.
FILTERS = {
    "order2": (
        np.array([2.740826, -5.4816519, 2.740826]),
        np.array([1.0, -1.9444777, 0.94597794]),
    ),
    "order10": tuple(
        np.asarray(c) for c in butter(5, [325, 3250], fs=16000, btype="band")
    ),
}

#: Coefficient sets that break naive state space constructions: a defective
#: companion matrix (repeated poles, trailing zero poles) has fewer
#: eigenvectors than states, so the modal basis is singular.  Numerator and
#: denominator are padded to the same length because ``lfilter`` insists on it.
DEGENERATE = {
    "b_longer_than_a": (
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1.0, -0.5, 0.0, 0.0, 0.0]),
    ),
    "double_pole": (np.array([1.0, 0.0, 0.0]), np.array([1.0, -1.0, 0.25])),
    "triple_pole": (
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, -2.1, 1.47, -0.343]),
    ),
    "unnormalised_a0": (np.array([2.0, 1.0, 0.0]), np.array([4.0, -2.0, 0.5])),
    "single_pole": (np.array([1.0, 0.5]), np.array([1.0, -0.9])),
    "odd_order": tuple(np.asarray(c) for c in butter(3, 3000, fs=16000, btype="low")),
}


def reference(signal, numerator, denominator, dtype=torch.float64):
    """``lfilter`` oracle, on the CPU in float64 unless told otherwise."""

    device = signal.device if dtype == torch.float32 else "cpu"

    return lfilter(
        signal.to(device=device, dtype=dtype),
        torch.as_tensor(denominator, dtype=dtype, device=device),
        torch.as_tensor(numerator, dtype=dtype, device=device),
        clamp=False,
    )


def peak_error(value, target):
    """Largest deviation from ``target``, relative to its peak magnitude."""

    scale = target.abs().max().item()
    if scale == 0.0:
        return value.abs().max().item()

    return (value.double().cpu() - target.double().cpu()).abs().max().item() / scale


def check_accuracy(mine, oracle, exact, what=""):
    """Assert the module beats both :data:`TOL` and the float32 oracle."""

    ours, theirs = peak_error(mine, exact), peak_error(oracle, exact)

    # the float32 oracle does not merely lose digits on a high order bandpass,
    # it returns `NaN` outright -- `butter(8, [325, 3250])` does.  `max(nan, x)`
    # is `nan` in Python, which would turn the strongest possible win into a
    # failed comparison, so score a broken oracle as infinitely bad.
    if not math.isfinite(theirs):
        theirs = math.inf

    assert math.isfinite(ours), f"{what}: the module itself returned {ours}"
    assert ours < TOL, f"{what}: {ours} >= {TOL}"
    assert ours <= max(theirs, FLOOR), (
        f"{what}: {ours} is worse than the float32 oracle ({theirs}) and worse "
        f"than the float32 floor ({FLOOR})"
    )

    return ours


def noise(*shape, seed=0, scale=1.0):
    """Reproducible white noise on the GPU."""

    generator = torch.Generator(device="cuda").manual_seed(seed)

    return torch.randn(*shape, device="cuda", generator=generator) * scale


@pytest.fixture(scope="module")
def modules():
    """One filter per (name, method, chunk size), built once for the module."""

    out = {}
    for name, (numerator, denominator) in FILTERS.items():
        for method in ("matmul", "scan"):
            for taps in (64, 128):
                out[(name, method, taps)] = TritonIIR(
                    numerator, denominator, chunk_size=taps, method=method
                ).cuda()

    return out


# ---------------------------------------------------------------------------
# forward parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(FILTERS))
@pytest.mark.parametrize("method", ["matmul", "scan"])
@pytest.mark.parametrize(
    "batch,samples",
    [
        (1, 16000),  # the shape the PESQ pipeline uses, batch of one
        (4, 4097),  # not a multiple of any chunk or tile size
        (3, 100),  # shorter than a single chunk
        (2, 129),  # spills into a second, mostly masked chunk
        (17, 1001),  # more rows than the carry kernel handles in one block
    ],
)
def test_forward_parity(modules, name, method, batch, samples):
    """Forward output matches a float64 ``lfilter`` on white noise."""

    numerator, denominator = FILTERS[name]
    module = modules[(name, method, 128)]

    signal = noise(batch, samples, seed=batch * 1000 + samples)

    check_accuracy(
        module(signal),
        reference(signal, numerator, denominator, dtype=torch.float32),
        reference(signal, numerator, denominator),
        f"{name}/{method}/{batch}x{samples}",
    )


@pytest.mark.parametrize("name", list(FILTERS))
@pytest.mark.parametrize(
    "kind",
    ["zeros", "tiny", "huge", "level", "alternating", "impulse", "step", "last"],
)
def test_forward_adversarial(modules, name, kind):
    """Degenerate and badly scaled inputs stay accurate and finite."""

    numerator, denominator = FILTERS[name]
    module = modules[(name, "matmul", 128)]

    batch, samples = 2, 3000
    white = noise(batch, samples, seed=7)

    if kind == "zeros":
        signal = torch.zeros(batch, samples, device="cuda")
    elif kind == "tiny":
        signal = white * 1e-25
    elif kind == "huge":
        signal = white * 1e12
    elif kind == "level":
        # the pre-emphasize filter runs on level aligned audio of this size
        signal = white * 3000.0
    elif kind == "alternating":
        sign = torch.where(
            torch.arange(samples, device="cuda") % 2 == 0, 1.0, -1.0
        ).expand(batch, samples)
        signal = (white.abs() + 1.0) * sign
    elif kind == "impulse":
        signal = torch.zeros(batch, samples, device="cuda")
        signal[:, 13] = 1.0
    elif kind == "last":
        # only the very last sample is nonzero, so every dropped tail element
        # or off-by-one in the chunk masks shows up as a 100% error
        signal = torch.zeros(batch, samples, device="cuda")
        signal[:, -1] = 1.0
    else:
        signal = torch.ones(batch, samples, device="cuda")

    out = module(signal)
    assert torch.isfinite(out).all()

    exact = reference(signal, numerator, denominator)
    assert peak_error(out, exact) < TOL


@pytest.mark.parametrize("samples", [1, 2, 3, 7, 15, 16, 17, 63, 64, 65, 127, 129])
def test_forward_tiny_lengths(modules, samples):
    """Signals shorter than one chunk, around every tile boundary."""

    numerator, denominator = FILTERS["order10"]
    module = modules[("order10", "matmul", 64)]

    signal = noise(1, samples, seed=samples)
    exact = reference(signal, numerator, denominator)

    assert peak_error(module(signal), exact) < TOL


@pytest.mark.parametrize("batch", [1, 15, 16, 17, 31, 32, 33, 64])
def test_batch_sweep(modules, batch):
    """The carry kernel walks the batch in blocks of 16, cross every boundary."""

    numerator, denominator = FILTERS["order10"]
    module = modules[("order10", "matmul", 128)]

    signal = noise(batch, 777, seed=batch)
    exact = reference(signal, numerator, denominator)

    out = module(signal)
    assert peak_error(out, exact) < TOL

    # every row has to be filtered, an over- or under-masked carry block would
    # leave whole rows at zero
    assert (out.abs().amax(dim=1) > 0).all()


def test_default_chunk_and_auto_method():
    """The shipped defaults -- ``chunk_size=None`` and ``method="auto"``."""

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator).cuda()

    assert module.candidates == (iir_module.DEFAULT_CHUNK,)
    assert module.method == "matmul"

    signal = noise(2, 5000, seed=61)
    exact = reference(signal, numerator, denominator)
    oracle = reference(signal, numerator, denominator, dtype=torch.float32)

    check_accuracy(module(signal), oracle, exact, "default")

    leaf = signal.clone().requires_grad_(True)
    upstream = noise(2, 5000, seed=62)
    grad = torch.autograd.grad(module(leaf), leaf, upstream)[0]

    exact_in = signal.double().cpu().requires_grad_(True)
    exact_grad = torch.autograd.grad(
        reference(exact_in, numerator, denominator), exact_in, upstream.double().cpu()
    )[0]

    assert peak_error(grad, exact_grad) < TOL


@pytest.mark.parametrize("taps", [64, 128, 256, 512])
def test_every_chunk_size(taps):
    """Each supported chunk length describes the very same filter."""

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator, chunk_size=taps).cuda()

    # 5000 is not a multiple of any of the chunk lengths, so the last chunk is
    # always partially masked
    signal = noise(3, 5000, seed=3)
    exact = reference(signal, numerator, denominator)
    oracle = reference(signal, numerator, denominator, dtype=torch.float32)

    check_accuracy(module(signal), oracle, exact, f"taps={taps}")


# ---------------------------------------------------------------------------
# backward parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(FILTERS))
@pytest.mark.parametrize("method", ["matmul", "scan"])
@pytest.mark.parametrize("batch,samples", [(1, 8000), (3, 2049), (20, 333)])
def test_backward_parity(modules, name, method, batch, samples):
    """Gradient matches autograd through ``lfilter`` for a random cotangent."""

    numerator, denominator = FILTERS[name]
    module = modules[(name, method, 128)]

    signal = noise(batch, samples, seed=samples, scale=3000.0)
    upstream = noise(batch, samples, seed=samples + 1)

    mine_in = signal.clone().requires_grad_(True)
    mine = torch.autograd.grad(module(mine_in), mine_in, upstream)[0]

    exact_in = signal.double().cpu().requires_grad_(True)
    exact = torch.autograd.grad(
        reference(exact_in, numerator, denominator),
        exact_in,
        upstream.double().cpu(),
    )[0]

    oracle_in = signal.clone().requires_grad_(True)
    oracle = torch.autograd.grad(
        reference(oracle_in, numerator, denominator, dtype=torch.float32),
        oracle_in,
        upstream,
    )[0]

    check_accuracy(mine, oracle, exact, f"{name}/{method}/{batch}x{samples}")

    # a filter gradient is dense, a silently zeroed tail or head would still
    # pass a peak relative check on a lucky cotangent
    assert (mine != 0).float().mean().item() > 0.99


@pytest.mark.parametrize("samples", [1, 5, 65, 127, 129])
def test_backward_tiny_lengths(modules, samples):
    """The reversed index arithmetic has to survive short, masked chunks."""

    numerator, denominator = FILTERS["order10"]
    module = modules[("order10", "matmul", 64)]

    signal = noise(2, samples, seed=samples + 500)
    upstream = noise(2, samples, seed=samples + 900)

    leaf = signal.clone().requires_grad_(True)
    mine = torch.autograd.grad(module(leaf), leaf, upstream)[0]

    exact_in = signal.double().cpu().requires_grad_(True)
    exact = torch.autograd.grad(
        reference(exact_in, numerator, denominator), exact_in, upstream.double().cpu()
    )[0]

    assert peak_error(mine, exact) < TOL


def test_backward_is_the_reversed_filter(modules):
    """The VJP of a causal LTI filter is ``flip(T(flip(g)))``, check that."""

    module = modules[("order10", "matmul", 128)]

    signal = noise(2, 1500, seed=11)
    upstream = noise(2, 1500, seed=12)

    leaf = signal.clone().requires_grad_(True)
    mine = torch.autograd.grad(module(leaf), leaf, upstream)[0]

    manual = torch.flip(module(torch.flip(upstream, (1,)).contiguous()), (1,))

    # both are float32 evaluations of the same expression, but the forward and
    # the reverse launch tune their tile sizes separately, so the summation
    # order of the chunk matmul may differ
    assert peak_error(mine, manual) < FLOOR


@pytest.mark.parametrize("method", ["matmul", "scan"])
def test_double_backward(modules, method):
    """The backward is itself a filter, so second order gradients must flow."""

    numerator, denominator = FILTERS["order10"]
    module = modules[("order10", method, 128)]

    signal = noise(2, 1200, seed=41)
    upstream = noise(2, 1200, seed=42)
    probe = noise(2, 1200, seed=43)

    leaf = signal.clone().requires_grad_(True)
    cotangent = upstream.clone().requires_grad_(True)

    (first,) = torch.autograd.grad(
        module(leaf), leaf, cotangent, create_graph=True
    )  # T^T g
    (second,) = torch.autograd.grad((first * probe).sum(), cotangent)

    # d/dg of (T^T g) . p is (T^T)^T p, i.e. the plain forward filter on p
    assert peak_error(second, reference(probe, numerator, denominator)) < TOL


def test_backward_through_a_strided_view(modules):
    """Gradients reach a non contiguous slice of a leaf, and only that slice."""

    numerator, denominator = FILTERS["order2"]
    module = modules[("order2", "matmul", 128)]

    leaf = noise(6, 4001, seed=23).requires_grad_(True)
    strided = leaf[1::2, ::3]
    assert not strided.is_contiguous()

    upstream = noise(*strided.shape, seed=24)
    mine = torch.autograd.grad(module(strided), leaf, upstream)[0]

    exact_leaf = leaf.detach().double().cpu().requires_grad_(True)
    exact = torch.autograd.grad(
        reference(exact_leaf[1::2, ::3].contiguous(), numerator, denominator),
        exact_leaf,
        upstream.double().cpu(),
    )[0]

    assert peak_error(mine, exact) < TOL
    assert torch.equal(mine[0::2], torch.zeros_like(mine[0::2]))
    assert (mine[1::2, ::3] != 0).all()


def test_backward_of_a_zero_signal(modules):
    """A zero input still has a nonzero Jacobian, the filter is linear."""

    numerator, denominator = FILTERS["order10"]
    module = modules[("order10", "matmul", 128)]

    leaf = torch.zeros(2, 2000, device="cuda", requires_grad=True)
    upstream = noise(2, 2000, seed=77)

    mine = torch.autograd.grad(module(leaf), leaf, upstream)[0]

    exact_in = torch.zeros(2, 2000, dtype=torch.float64, requires_grad=True)
    exact = torch.autograd.grad(
        reference(exact_in, numerator, denominator), exact_in, upstream.double().cpu()
    )[0]

    assert torch.isfinite(mine).all()
    assert peak_error(mine, exact) < TOL


# ---------------------------------------------------------------------------
# determinism, aliasing and the Triton only contract
# ---------------------------------------------------------------------------


def test_determinism(modules):
    """Same input twice gives bitwise identical output and gradient."""

    module = modules[("order10", "matmul", 128)]

    signal = noise(3, 7777, seed=17)
    upstream = noise(3, 7777, seed=18)

    first, second = module(signal), module(signal)
    assert torch.equal(first, second)

    grads = []
    for _ in range(2):
        leaf = signal.clone().requires_grad_(True)
        grads.append(torch.autograd.grad(module(leaf), leaf, upstream)[0])

    assert torch.equal(grads[0], grads[1])


def test_interleaved_shapes_do_not_alias(modules):
    """The shared scratch buffers must not leak state between calls."""

    numerator, denominator = FILTERS["order10"]
    module = modules[("order10", "matmul", 128)]

    first, second = noise(2, 1000, seed=31), noise(2, 1000, seed=32)
    third = noise(5, 4321, seed=33)

    alone = module(first)
    module(second)
    module(third)
    module(second)

    assert torch.equal(module(first), alone)
    assert peak_error(alone, reference(first, numerator, denominator)) < TOL
    assert peak_error(module(third), reference(third, numerator, denominator)) < TOL


def test_output_is_fully_written(modules):
    """Every output element is stored, none is left uninitialised.

    ``_launch`` hands out a ``torch.empty_like`` buffer, so a chunk mask that
    skips an element would return whatever the caching allocator had there.
    Poisoning the allocator with NaN makes such a hole visible.
    """

    module = modules[("order10", "matmul", 128)]

    signal = noise(3, 4001, seed=51)  # 4001 is not a multiple of 128
    expected = module(signal).clone()

    torch.cuda.synchronize()
    poison = [torch.full_like(signal, float("nan")) for _ in range(8)]
    del poison

    again = module(signal)
    assert torch.isfinite(again).all()
    assert torch.equal(again, expected)


def test_only_triton_kernels_touch_the_data(modules):
    """Forward *and* backward have to run entirely in Triton kernels."""

    module = modules[("order10", "matmul", 128)]

    signal = noise(2, 4000, seed=71)
    upstream = noise(2, 4000, seed=72)

    leaf = signal.clone().requires_grad_(True)
    torch.autograd.grad(module(leaf), leaf, upstream)  # warm the tuner

    leaf = signal.clone().requires_grad_(True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        torch.autograd.grad(module(leaf), leaf, upstream)
        torch.cuda.synchronize()

    launched = {
        event.key for event in prof.key_averages() if event.self_device_time_total > 0
    }

    assert launched == {
        "_state_matmul_kernel",
        "_carry_kernel",
        "_output_matmul_kernel",
    }, launched


def test_scratch_memory_is_bounded(modules):
    """A per shape scratch cache would leak on variable length audio."""

    module = modules[("order2", "matmul", 128)]

    module(noise(2, 20000, seed=1))  # fix the high water mark
    plan = module._plan(128)
    assert len(plan._scratch) == 1
    before = next(iter(plan._scratch.values()))[0].numel()

    for samples in range(1000, 1009):
        module(noise(2, samples, seed=samples))

    assert len(plan._scratch) == 1
    assert next(iter(plan._scratch.values()))[0].numel() == before


def test_nan_does_not_cross_rows(modules):
    """A poisoned row must not contaminate its neighbours."""

    module = modules[("order10", "matmul", 128)]

    signal = noise(3, 2000, seed=81)
    signal[1, 500] = float("nan")

    out = module(signal)

    assert torch.isfinite(out[0]).all()
    assert torch.isfinite(out[2]).all()
    assert torch.isnan(out[1, 500:]).any()


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def test_chunk_sizes_agree(modules):
    """Every chunk length has to describe the very same filter."""

    numerator, denominator = FILTERS["order10"]

    signal = noise(2, 5000, seed=3)
    exact = reference(signal, numerator, denominator)

    for method in ("matmul", "scan"):
        for taps in (64, 128):
            error = peak_error(modules[("order10", method, taps)](signal), exact)
            assert error < TOL, f"{method}/{taps}: {error}"


def test_tuned_chunk_size():
    """``chunk_size="tune"`` picks a candidate and stays correct."""

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator, chunk_size="tune").cuda()

    signal = noise(4, 6000, seed=5)

    out = module(signal)
    assert module._tuned[(4, 6000)] in module.candidates
    assert peak_error(out, reference(signal, numerator, denominator)) < TOL

    # the cached choice has to keep producing the same answer
    assert torch.equal(out, module(signal))


def test_non_contiguous_and_no_grad(modules):
    """Strided inputs, ``no_grad`` and non differentiable inputs all work."""

    numerator, denominator = FILTERS["order2"]
    module = modules[("order2", "matmul", 128)]

    wide = noise(3, 4000, seed=23)

    strided = wide[::2, ::2]
    assert not strided.is_contiguous()

    exact = reference(strided.contiguous(), numerator, denominator)
    assert peak_error(module(strided), exact) < TOL

    transposed = noise(2500, 4, seed=25).t()
    assert not transposed.is_contiguous()
    assert (
        peak_error(
            module(transposed),
            reference(transposed.contiguous(), numerator, denominator),
        )
        < TOL
    )

    with torch.no_grad():
        out = module(wide.clone().requires_grad_(True))
    assert not out.requires_grad

    plain = module(wide)
    assert not plain.requires_grad

    assert module(wide.requires_grad_(True)).requires_grad
    wide.requires_grad_(False)


def test_batch_one_matches_batched(modules):
    """Every row is filtered independently, whatever the batch size is."""

    module = modules[("order10", "matmul", 128)]

    signal = noise(5, 3333, seed=29)

    batched = module(signal)
    for row in range(signal.shape[0]):
        single = module(signal[row : row + 1].contiguous())
        # the tile sizes are tuned per shape, so batch 1 and batch 5 may sum
        # the chunk matmul in a different order
        assert peak_error(single[0], batched[row]) < FLOOR


def test_empty_inputs(modules):
    """Zero samples and zero rows are handled without a launch failure."""

    module = modules[("order2", "matmul", 128)]

    assert module(torch.zeros(2, 0, device="cuda")).shape == (2, 0)
    assert module(torch.zeros(0, 1000, device="cuda")).shape == (0, 1000)


def test_moves_with_the_module():
    """Tables are buffers, so ``.to(device)`` has to carry them along."""

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator, chunk_size=128)

    assert module.get_buffer("_impulse_128").device.type == "cpu"

    module = module.to("cuda")
    assert module.get_buffer("_impulse_128").is_cuda

    signal = noise(2, 2500, seed=2)
    assert peak_error(module(signal), reference(signal, numerator, denominator)) < TOL


def test_wrong_device_is_reported():
    """A module left on the host must say so instead of failing obscurely."""

    module = TritonIIR(*FILTERS["order2"], chunk_size=128)

    with pytest.raises(RuntimeError, match="move the module"):
        module(noise(2, 512, seed=1))


def test_rejects_oversized_input(modules, monkeypatch):
    """The kernels index with int32, so the element count has to be checked."""

    module = modules[("order2", "matmul", 128)]
    monkeypatch.setattr(iir_module, "_MAX_ELEMENTS", 100)

    with pytest.raises(RuntimeError, match="int32"):
        module(noise(2, 512, seed=1))


def test_rejects_bad_input(modules):
    """Shape, dtype, device and constructor arguments are validated."""

    module = modules[("order2", "matmul", 128)]

    with pytest.raises(RuntimeError):
        module(torch.randn(2, 3, 4, device="cuda"))
    with pytest.raises(RuntimeError):
        module(torch.randn(2, 128, device="cuda", dtype=torch.float64))
    with pytest.raises(RuntimeError):
        module(torch.randn(2, 128))

    for bad in (100, 63, 0, -64):
        with pytest.raises(ValueError):
            TritonIIR(*FILTERS["order2"], chunk_size=bad)

    with pytest.raises(ValueError):
        TritonIIR(*FILTERS["order2"], method="magic")

    with pytest.raises(ValueError):
        TritonIIR(np.array([1.0]), np.array([0.0, 1.0]))  # a[0] == 0


# ---------------------------------------------------------------------------
# state space realisation
# ---------------------------------------------------------------------------


def test_state_space_is_well_conditioned():
    """The order 10 bandpass must not end up in the companion basis.

    In the companion basis ``max|A^t|`` reaches ``3e4`` and the float32
    recursion loses every significant digit, so this is a correctness guard,
    not cosmetics.
    """

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator, chunk_size=64)

    assert module.order == 10
    assert module.states == 10
    assert module.basis in ("cascade", "modal")

    for name in ("_carry_64", "_power_64", "_drive_64", "_impulse_64", "_state_64"):
        assert module.get_buffer(name).abs().max().item() < 1e2


@pytest.mark.parametrize("taps", [64, 256])
@pytest.mark.parametrize(
    "name", list(FILTERS) + list(DEGENERATE) + ["resonant", "narrow"]
)
def test_conditioning_covers_every_table(name, taps):
    """The scored table entry has to bound every table the kernels read.

    ``_condition`` used to look at the read out ``G[t] = e^T A^t`` and the
    transition powers only, and stopped one step short of ``A^L``.  The kernels
    also multiply the input by ``W[:, k] = A^(L-1-k) beta``, and that table is
    *not* implied by the other two: ``b_longer_than_a`` above scored ``1.0``
    while its ``drive`` table reaches ``8.1``, so the realisation was ranked as
    eight times better conditioned than it is.  Counting the input map costs
    nothing, is what the realisations are actually ranked on, and leaves the
    choice for both PESQ filters unchanged.

    ``impulse`` is deliberately excluded: it is the filter's own impulse
    response, a property of the coefficients rather than of the realisation, and
    every realisation of one filter carries the same one.
    """

    extra = {
        "resonant": (
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            np.poly(
                [
                    0.999 * np.exp(1j * 0.30),
                    0.999 * np.exp(-1j * 0.30),
                    0.999 * np.exp(1j * 0.31),
                    0.999 * np.exp(-1j * 0.31),
                ]
            ).real,
        ),
        "narrow": tuple(np.asarray(c) for c in butter(3, [0.01, 0.02], btype="band")),
    }
    numerator, denominator = {**FILTERS, **DEGENERATE, **extra}[name]

    numerator = iir_module._as_coeffs(numerator)
    denominator = iir_module._as_coeffs(denominator)

    state, drive, read, feed, _ = iir_module._state_space(numerator, denominator, taps)
    exact = iir_module._direct_response(numerator, denominator, taps)
    scale = max(float(np.abs(exact).max()), 1e-30)

    _, entry = iir_module._score(state, drive, read, feed, exact, scale, taps)

    padded = max(16, next_power_of_two(state.shape[0]))
    tables = iir_module._tables(state, drive, read, feed, taps, padded)

    # `impulse` is the filter's own impulse response, a property of the
    # coefficients rather than of the realisation, so it is not scored
    largest = max(
        float(np.abs(table).max()) for key, table in tables.items() if key != "impulse"
    )

    assert entry >= largest * (1.0 - 1e-9), (
        f"{name}: scored table entry {entry:.4g} does not cover the largest "
        f"entry {largest:.4g} of the tables the kernels read"
    )


@pytest.mark.parametrize("name", list(DEGENERATE))
def test_degenerate_coefficients(name):
    """Repeated poles, zero poles and an unnormalised ``a[0]`` all work.

    Their companion matrix is defective, so the modal basis is singular; the
    constructor has to fall back instead of raising ``LinAlgError``.
    """

    numerator, denominator = DEGENERATE[name]
    module = TritonIIR(numerator, denominator, chunk_size=128).cuda()

    signal = noise(2, 3000, seed=hash(name) % 1000)
    out = module(signal)

    assert torch.isfinite(out).all()
    assert peak_error(out, reference(signal, numerator, denominator)) < TOL


def test_high_order_filter_is_not_silently_nan():
    """An order 16 bandpass used to select the companion basis and return NaN."""

    numerator, denominator = (
        np.asarray(c) for c in butter(8, [200, 3800], fs=16000, btype="band")
    )
    module = TritonIIR(numerator, denominator, chunk_size=128).cuda()

    assert module.states == 16

    signal = noise(2, 4000, seed=91)
    out = module(signal)

    assert torch.isfinite(out).all()
    assert peak_error(out, reference(signal, numerator, denominator)) < TOL


def test_unrepresentable_filter_raises_instead_of_returning_nan():
    """No realisation of an order 30 bandpass survives float32 -- say so."""

    numerator, denominator = (
        np.asarray(c) for c in butter(15, [325, 3250], fs=16000, btype="band")
    )

    with pytest.raises(RuntimeError, match="realisation"):
        TritonIIR(numerator, denominator, chunk_size=128)


def test_degraded_filter_warns():
    """An order 20 bandpass is usable but lossy, and has to say so."""

    numerator, denominator = (
        np.asarray(c) for c in butter(10, [325, 3250], fs=16000, btype="band")
    )

    with pytest.warns(RuntimeWarning, match="degraded"):
        TritonIIR(numerator, denominator, chunk_size=128)


# ---------------------------------------------------------------------------
# adversarial review: launch geometry, basis selection and coverage holes
# ---------------------------------------------------------------------------


def test_batch_beyond_the_grid_y_limit():
    """A batch larger than 65535 must not hit the CUDA ``gridDim.y`` cap.

    CUDA caps the second and third grid dimensions at 65535 while the first one
    goes to ``2**31 - 1``.  Putting the batch on ``tl.program_id(1)`` therefore
    fails the *launch* -- not a wrong answer, an ``invalid configuration
    argument`` -- for any batch beyond that, which ``_MAX_ELEMENTS`` happily
    allows: 70000 rows of 64 samples are only 4.5M elements and 18MB.
    """

    numerator, denominator = FILTERS["order2"]
    module = TritonIIR(numerator, denominator, chunk_size=64).cuda()

    signal = noise(70000, 64, seed=101)
    out = module(signal)

    torch.cuda.synchronize()
    assert torch.isfinite(out).all()

    # the rows past 65535 are exactly the ones a y-grid launch cannot reach, so
    # check them against the float64 oracle rather than the batch as a whole
    tail = signal[-200:].contiguous()
    assert peak_error(out[-200:], reference(tail, numerator, denominator)) < TOL

    # and every remaining row has to be filtered, not left at whatever the
    # caching allocator had in the buffer
    assert (out.abs().amax(dim=1) > 0).all()


def test_many_chunks_on_a_single_row():
    """A long signal puts many chunk tiles on the first grid axis."""

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator, chunk_size=64).cuda()

    # 480000 samples at chunk 64 is 7500 chunks, i.e. 7500/BLOCK_C * TAPS/BLOCK_T
    # output programs per row -- the axis the batch now shares
    signal = noise(2, 480000, seed=102)
    out = module(signal)

    assert torch.isfinite(out).all()
    assert peak_error(out, reference(signal, numerator, denominator)) < TOL


def test_state_buffer_offsets_are_checked_too(modules, monkeypatch):
    """The int32 guard has to cover the state buffer, not only the signal.

    The state offset is ``(batch * nchunk + chunk) * padded``, which exceeds
    ``batch * sample`` whenever the padded state is wider than a chunk holds
    samples.  A guard that only looks at the signal would let the state offset
    wrap while reporting the shape as fine.
    """

    module = modules[("order2", "matmul", 64)]
    padded = module.padded

    nbatch, ntime = 4, 8  # one chunk of 64 taps, mostly masked
    signal_elements = nbatch * ntime
    state_elements = nbatch * 1 * padded

    assert state_elements > signal_elements, (state_elements, signal_elements)

    # a limit between the two: the signal fits, the state buffer does not
    monkeypatch.setattr(
        iir_module, "_MAX_ELEMENTS", (signal_elements + state_elements) // 2
    )

    with pytest.raises(RuntimeError, match="int32"):
        module(torch.zeros(nbatch, ntime, device="cuda"))


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7, 9, 12, 16])
def test_filter_order_sweep(order):
    """Every order between 1 and 16, not only the two PESQ filters.

    Orders other than 2 and 10 exercise odd state counts (the cascade pads to
    an even number), a one dimensional state and, at order 16, the case where
    picking the realisation by conditioning alone used to select the modal
    basis and lose four digits.
    """

    if order == 1:
        numerator, denominator = (np.asarray(c) for c in butter(1, 3000, fs=16000))
    elif order % 2:
        numerator, denominator = (
            np.asarray(c) for c in butter(order, 3000, fs=16000, btype="low")
        )
    else:
        numerator, denominator = (
            np.asarray(c)
            for c in butter(order // 2, [325, 3250], fs=16000, btype="band")
        )

    module = TritonIIR(numerator, denominator, chunk_size=256).cuda()
    assert module.order == order

    signal = noise(2, 16000, seed=200 + order)
    upstream = noise(2, 16000, seed=300 + order)

    exact = reference(signal, numerator, denominator)
    oracle = reference(signal, numerator, denominator, dtype=torch.float32)
    check_accuracy(module(signal), oracle, exact, f"order{order} forward")

    leaf = signal.clone().requires_grad_(True)
    mine = torch.autograd.grad(module(leaf), leaf, upstream)[0]

    exact_in = signal.double().cpu().requires_grad_(True)
    exact_grad = torch.autograd.grad(
        reference(exact_in, numerator, denominator), exact_in, upstream.double().cpu()
    )[0]

    assert peak_error(mine, exact_grad) < TOL, f"order{order} backward"


def test_order_sixteen_picks_the_accurate_realisation():
    """Conditioning alone is the wrong criterion, fidelity has to count too.

    For ``butter(8, [325, 3250])`` the modal realisation has the smaller table
    entries (66 against 81) but its float64 impulse response is already off by
    ``2.3e-05``, against ``8.0e-08`` for the cascade.  Selecting on the table
    size alone therefore shipped a filter with a peak relative error of
    ``5.1e-05``, five times outside :data:`TOL`, on a filter for which
    ``4.9e-07`` was available.
    """

    numerator, denominator = (
        np.asarray(c) for c in butter(8, [325, 3250], fs=16000, btype="band")
    )
    module = TritonIIR(numerator, denominator, chunk_size=256).cuda()

    assert module.states == 16
    assert module.basis == "cascade"

    signal = noise(2, 16000, seed=113)
    error = peak_error(module(signal), reference(signal, numerator, denominator))

    assert error < 2e-6, error


@pytest.mark.parametrize(
    "name,coeffs",
    [
        ("bp12", lambda: butter(6, [325, 3250], fs=16000, btype="band")),
        ("bp14", lambda: butter(7, [325, 3250], fs=16000, btype="band")),
        ("cheby1", lambda: cheby1(4, 1, [325, 3250], fs=16000, btype="band")),
        ("ellip", lambda: ellip(4, 1, 60, [325, 3250], fs=16000, btype="band")),
        ("cheby1_hi", lambda: cheby1(6, 1, [325, 3250], fs=16000, btype="band")),
    ],
)
def test_other_filter_families(name, coeffs):
    """Chebyshev and elliptic responses, whose poles sit closer to the circle."""

    numerator, denominator = (np.asarray(c) for c in coeffs())
    module = TritonIIR(numerator, denominator, chunk_size=256).cuda()

    signal = noise(2, 16000, seed=hash(name) % 997)
    exact = reference(signal, numerator, denominator)
    oracle = reference(signal, numerator, denominator, dtype=torch.float32)

    check_accuracy(module(signal), oracle, exact, name)


@pytest.mark.parametrize("radius", [0.99, 0.999, 0.9999])
@pytest.mark.parametrize("samples", [16000, 300000])
def test_poles_close_to_the_unit_circle(radius, samples):
    """A resonator with a very long ring down, over a very long signal.

    This is the regime in which the inter chunk scan accumulates: the carry
    recursion runs ``samples / chunk`` steps and a pole at ``0.9999`` decays by
    less than 3% over a 256 sample chunk, so nothing damps the round off.  The
    tolerance is the usual one; ``1e-5`` is only left behind beyond a radius of
    ``0.99999``, which is documented in the module docstring.
    """

    denominator = np.array([1.0, -2 * radius * np.cos(0.3), radius * radius])
    numerator = np.array([1.0, 0.0, 0.0])

    module = TritonIIR(numerator, denominator, chunk_size=256).cuda()

    signal = noise(2, samples, seed=int(radius * 1e4) + samples)
    exact = reference(signal, numerator, denominator)
    oracle = reference(signal, numerator, denominator, dtype=torch.float32)

    check_accuracy(module(signal), oracle, exact, f"r={radius} n={samples}")


def test_extreme_pole_is_documented_not_silent():
    """At a radius of ``0.99999`` accuracy degrades -- but stays ahead of the
    oracle by more than an order of magnitude, and never turns into ``NaN``."""

    radius = 0.99999
    denominator = np.array([1.0, -2 * radius * np.cos(0.3), radius * radius])
    numerator = np.array([1.0, 0.0, 0.0])

    module = TritonIIR(numerator, denominator, chunk_size=256).cuda()

    signal = noise(2, 480000, seed=131)
    exact = reference(signal, numerator, denominator)
    oracle = reference(signal, numerator, denominator, dtype=torch.float32)

    out = module(signal)
    assert torch.isfinite(out).all()

    ours, theirs = peak_error(out, exact), peak_error(oracle, exact)

    # measured 4.4e-05 against 1.2e-03; the loose bound is the documented
    # limitation of this regime, the ratio is the actual assertion
    assert ours < 1e-4, ours
    assert ours < theirs / 10.0, (ours, theirs)


@pytest.mark.parametrize("exponent", [-30, -35, -38])
def test_denormal_scale_inputs(exponent):
    """Inputs near the float32 denormal floor stay finite and stay linear.

    Down to ``1e-38`` the state of the cascade is still normal and the filter
    keeps five digits.  Below that the *state*, which the section gains scale
    down further, underflows before the signal does -- see the module
    docstring; the assertion here is only that nothing turns into ``NaN`` and
    that the result is the scaled version of the well scaled answer.
    """

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator, chunk_size=128).cuda()

    unit = noise(2, 2000, seed=141)
    scale = float(10.0**exponent)

    small = module(unit * scale)
    assert torch.isfinite(small).all()

    exact = reference(unit * scale, numerator, denominator)
    tolerance = TOL if exponent >= -35 else 1e-5 * 10.0 ** (-35 - exponent)

    assert peak_error(small, exact) < tolerance, peak_error(small, exact)


def test_subnormal_input_does_not_produce_nan():
    """True subnormals lose precision but must not poison the output."""

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator, chunk_size=128).cuda()

    signal = noise(2, 2000, seed=142) * 1e-42
    out = module(signal)

    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("taps", [64, 128, 256, 512])
@pytest.mark.parametrize("method", ["matmul", "scan"])
def test_backward_at_every_chunk_size(taps, method):
    """Gradients, not just forwards, at every chunk length and both methods.

    The shipped ``chunk_size`` is 256 and ``"tune"`` reaches 512, but only the
    forward pass was covered there; the reverse index flip interacts with the
    partial last chunk differently from the forward one.
    """

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator, chunk_size=taps, method=method).cuda()

    # 3001 is not a multiple of 64, 128, 256 or 512
    signal = noise(2, 3001, seed=taps + 400, scale=3000.0)
    upstream = noise(2, 3001, seed=taps + 401)

    leaf = signal.clone().requires_grad_(True)
    mine = torch.autograd.grad(module(leaf), leaf, upstream)[0]

    exact_in = signal.double().cpu().requires_grad_(True)
    exact = torch.autograd.grad(
        reference(exact_in, numerator, denominator), exact_in, upstream.double().cpu()
    )[0]

    oracle_in = signal.clone().requires_grad_(True)
    oracle = torch.autograd.grad(
        reference(oracle_in, numerator, denominator, dtype=torch.float32),
        oracle_in,
        upstream,
    )[0]

    check_accuracy(mine, oracle, exact, f"{method}/{taps} backward")

    # a filter gradient is dense; a silently zeroed head or tail would survive
    # a peak relative check on a lucky cotangent
    assert (mine != 0).float().mean().item() > 0.99


@pytest.mark.parametrize("taps", [64, 128, 256, 512])
def test_batch_one_at_every_chunk_size(taps):
    """Batch one, both methods, a length that is not a multiple of the chunk."""

    numerator, denominator = FILTERS["order10"]

    for method in ("matmul", "scan"):
        module = TritonIIR(
            numerator, denominator, chunk_size=taps, method=method
        ).cuda()

        signal = noise(1, taps * 3 + 7, seed=taps)
        exact = reference(signal, numerator, denominator)

        assert peak_error(module(signal), exact) < TOL, f"{method}/{taps}"


def test_non_contiguous_backward_for_both_methods(modules):
    """A strided input through the scan formulation too, not just the matmul."""

    numerator, denominator = FILTERS["order10"]

    for method in ("matmul", "scan"):
        module = modules[("order10", method, 128)]

        leaf = noise(4, 3000, seed=151).requires_grad_(True)
        strided = leaf[::2, 1::2]
        assert not strided.is_contiguous()

        upstream = noise(*strided.shape, seed=152)
        mine = torch.autograd.grad(module(strided), leaf, upstream, retain_graph=False)[
            0
        ]

        exact_leaf = leaf.detach().double().cpu().requires_grad_(True)
        exact = torch.autograd.grad(
            reference(exact_leaf[::2, 1::2].contiguous(), numerator, denominator),
            exact_leaf,
            upstream.double().cpu(),
        )[0]

        assert peak_error(mine, exact) < TOL, method
        assert torch.equal(mine[1::2], torch.zeros_like(mine[1::2]))


def test_tolerance_is_not_vacuous(modules):
    """:data:`TOL` has to be a real bound, so record the margin it actually has.

    A tolerance that no realistic input comes near is not a test.  Over the
    shapes and filters this suite uses the module stays below ``1e-6``, so
    ``TOL = 1e-5`` is a ten fold margin -- tight enough to catch a regression
    of a single float32 digit.
    """

    worst = 0.0
    for name in FILTERS:
        numerator, denominator = FILTERS[name]
        for method in ("matmul", "scan"):
            module = modules[(name, method, 128)]
            for batch, samples in ((1, 16000), (5, 4097)):
                signal = noise(batch, samples, seed=batch + samples)
                worst = max(
                    worst,
                    peak_error(
                        module(signal), reference(signal, numerator, denominator)
                    ),
                )

    assert 0.0 < worst < TOL / 5.0, f"worst {worst}, TOL {TOL}"


def test_scan_and_matmul_agree_bitwise_in_kind(modules):
    """The two formulations are the same filter, only summed differently."""

    numerator, denominator = FILTERS["order10"]

    signal = noise(3, 2049, seed=161)
    exact = reference(signal, numerator, denominator)

    matmul = modules[("order10", "matmul", 128)](signal)
    scan = modules[("order10", "scan", 128)](signal)

    assert peak_error(matmul, exact) < TOL
    assert peak_error(scan, exact) < TOL
    assert peak_error(matmul, scan.double().cpu()) < FLOOR


def test_zero_batch_backward(modules):
    """An empty batch has to survive the launch in both directions."""

    module = modules[("order10", "matmul", 128)]

    leaf = torch.zeros(0, 1000, device="cuda", requires_grad=True)
    out = module(leaf)
    assert out.shape == (0, 1000)

    grad = torch.autograd.grad(out, leaf, torch.zeros_like(out))[0]
    assert grad.shape == (0, 1000)


def test_scan_output_is_fully_written():
    """The scan formulation gets the same uninitialised memory check."""

    numerator, denominator = FILTERS["order10"]
    module = TritonIIR(numerator, denominator, chunk_size=128, method="scan").cuda()

    signal = noise(3, 4001, seed=171)
    expected = module(signal).clone()

    torch.cuda.synchronize()
    poison = [torch.full_like(signal, float("nan")) for _ in range(8)]
    del poison

    again = module(signal)
    assert torch.isfinite(again).all()
    assert torch.equal(again, expected)


def test_only_triton_kernels_touch_the_data_scan(modules):
    """The scan formulation must also stay entirely inside Triton."""

    module = modules[("order10", "scan", 128)]

    signal = noise(2, 4000, seed=181)
    upstream = noise(2, 4000, seed=182)

    leaf = signal.clone().requires_grad_(True)
    torch.autograd.grad(module(leaf), leaf, upstream)  # warm the tuner

    leaf = signal.clone().requires_grad_(True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        torch.autograd.grad(module(leaf), leaf, upstream)
        torch.cuda.synchronize()

    launched = {
        event.key for event in prof.key_averages() if event.self_device_time_total > 0
    }

    assert launched == {
        "_state_scan_kernel",
        "_carry_kernel",
        "_output_scan_kernel",
    }, launched
