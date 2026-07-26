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

import numpy as np
import pytest
import torch

from scipy.signal import butter
from torchaudio.functional import lfilter

from torch_pesq.triton_ops._common import HAS_TRITON

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
