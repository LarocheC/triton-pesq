"""Parity tests for :class:`torch_pesq.triton_ops.resample.TritonResample`.

The oracle is :class:`torchaudio.transforms.Resample` constructed with the very
same arguments; both share the host side sinc kernel design, so any deviation
comes from the convolution itself.

Tolerances
----------
The comparison is *scale aware*: ``max |triton - torchaudio|`` is required to
stay below ``ATOL + RTOL * max |torchaudio|``. An element-wise ``rtol`` is the
wrong criterion for a convolution because outputs pass through zero while their
neighbours do not, so an element of size ``1e-9`` next to elements of size
``1`` can never be matched relatively. With ``width_total`` between 5 and 621
taps a float32 dot product carries roughly ``sqrt(width_total) * 6e-8 ~ 1.5e-6``
relative error, and cuDNN sums in a different order than the Triton kernel, so
``RTOL = 2e-6`` on the signal scale is the tightest bound that is physically
meaningful here. The measured worst case over the whole sweep is ~8e-7 relative,
and :func:`test_tolerance_rejects_wrong_answers` pins the bound down from the
other side by checking that it still rejects a one-sample shift, a dropped tail
and a single perturbed element.
"""

import gc
import statistics

import pytest
import torch

triton = pytest.importorskip("triton")
torchaudio = pytest.importorskip("torchaudio")

from torch_pesq.triton_ops import resample as resample_module  # noqa: E402
from torch_pesq.triton_ops.resample import TritonResample  # noqa: E402

# See the module docstring for the justification of these numbers.
ATOL = 1e-6
RTOL = 2e-6

RATES = [(48000, 16000), (44100, 16000), (16000, 16000), (8000, 16000), (22050, 16000)]

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the Triton backend needs a CUDA device"
)


def errors(got, expected):
    """Maximum absolute error and the same error relative to the signal scale."""

    scale = expected.abs().max().clamp(min=1e-30)
    abs_err = (got - expected).abs().max()

    return abs_err.item(), (abs_err / scale).item()


def assert_close(got, expected, what=""):
    """Scale aware comparison, see the module docstring."""

    assert got.shape == expected.shape, f"{what}: {got.shape} != {expected.shape}"

    if expected.numel() == 0:
        return 0.0, 0.0

    abs_err, rel_err = errors(got, expected)
    bound = ATOL + RTOL * expected.abs().max().item()

    assert (
        abs_err <= bound
    ), f"{what}: abs {abs_err:.3e} > {bound:.3e} (rel {rel_err:.3e})"

    return abs_err, rel_err


def build(orig, new, device="cuda", **kwargs):
    """Oracle and Triton module for a rate pair, both on ``device``."""

    oracle = torchaudio.transforms.Resample(orig, new, **kwargs).to(device)
    triton_mod = TritonResample(orig, new, **kwargs).to(device)

    return oracle, triton_mod


def both_grads(oracle, triton_mod, signal, upstream=None, seed=0):
    """Gradient of both implementations for the same random upstream gradient."""

    ref_in = signal.clone().requires_grad_(True)
    tri_in = signal.clone().requires_grad_(True)

    ref_out, tri_out = oracle(ref_in), triton_mod(tri_in)

    if upstream is None:
        generator = torch.Generator(device=signal.device).manual_seed(seed)
        upstream = torch.randn(ref_out.shape, device=signal.device, generator=generator)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_in, upstream)
    (tri_grad,) = torch.autograd.grad(tri_out, tri_in, upstream)

    return tri_grad, ref_grad


def run_one_config(kernel, grid, args, **constexprs):
    """Launch a single autotuning candidate, bypassing the tuner."""

    kernel.fn[grid](*args, **constexprs)


@cuda_only
@pytest.mark.parametrize("orig,new", RATES)
@pytest.mark.parametrize("batch", [1, 3, 8])
@pytest.mark.parametrize("length", [1, 17, 999, 4001, 16384, 33333])
def test_forward_parity(orig, new, batch, length):
    """Forward pass matches torchaudio for odd shapes and batch sizes."""

    torch.manual_seed(orig + new + batch + length)
    signal = torch.randn(batch, length, device="cuda")

    oracle, triton_mod = build(orig, new)
    assert_close(triton_mod(signal), oracle(signal), f"{orig}->{new} {batch}x{length}")


@cuda_only
@pytest.mark.parametrize("orig,new", RATES)
@pytest.mark.parametrize(
    "kind", ["zeros", "tiny", "huge", "alternating", "impulse", "dc", "ramp"]
)
def test_forward_adversarial(orig, new, kind):
    """Forward pass survives degenerate and badly scaled inputs."""

    torch.manual_seed(1234)
    length = 5000
    base = torch.randn(2, length, device="cuda")

    if kind == "zeros":
        signal = torch.zeros(2, length, device="cuda")
    elif kind == "tiny":
        signal = base * 1e-12
    elif kind == "huge":
        signal = base * 1e6
    elif kind == "alternating":
        signal = base * torch.tensor([1.0, -1.0], device="cuda").repeat(length // 2)
    elif kind == "impulse":
        signal = torch.zeros(2, length, device="cuda")
        signal[:, 0] = 1.0
        signal[:, length // 3] = -5.0
        signal[:, -1] = 3.0
    elif kind == "dc":
        signal = torch.full((2, length), 0.5, device="cuda")
    else:
        signal = torch.linspace(-1e3, 1e3, length, device="cuda").expand(2, length)

    oracle, triton_mod = build(orig, new)
    out = triton_mod(signal)

    assert torch.isfinite(out).all(), f"{orig}->{new} {kind}: non finite output"
    assert_close(out, oracle(signal), f"{orig}->{new} {kind}")


@cuda_only
@pytest.mark.parametrize("orig,new", RATES)
@pytest.mark.parametrize("batch,length", [(1, 4001), (4, 9600), (3, 257)])
def test_backward_parity(orig, new, batch, length):
    """Gradients match autograd through torchaudio for a random upstream grad."""

    torch.manual_seed(orig + batch * length)
    signal = torch.randn(batch, length, device="cuda")

    oracle, triton_mod = build(orig, new)
    tri_grad, ref_grad = both_grads(oracle, triton_mod, signal, seed=length)

    assert torch.isfinite(tri_grad).all()
    assert_close(tri_grad, ref_grad, f"grad {orig}->{new} {batch}x{length}")


@cuda_only
@pytest.mark.parametrize("orig,new", RATES)
def test_backward_short_lengths(orig, new):
    """Signals shorter than the filter still differentiate correctly.

    ``width_total`` reaches 475 taps, so every length below that is a pure
    boundary case: no tap of the filter bank is ever fully inside the signal.
    """

    oracle, triton_mod = build(orig, new)

    for length in [1, 2, 3, 5, 13, 16, 17, 31, 33, 64, 65, 127, 129, 256, 441, 476]:
        torch.manual_seed(length)
        signal = torch.randn(2, length, device="cuda")

        tri_grad, ref_grad = both_grads(oracle, triton_mod, signal, seed=length)
        assert_close(tri_grad, ref_grad, f"grad {orig}->{new} len {length}")


@cuda_only
@pytest.mark.parametrize("orig,new", RATES)
def test_backward_scales(orig, new):
    """Gradients stay accurate for badly scaled upstream gradients."""

    torch.manual_seed(7)
    signal = torch.randn(2, 6000, device="cuda")

    oracle, triton_mod = build(orig, new)

    for factor in [1e-10, 1e5]:
        ref_in = signal.clone().requires_grad_(True)
        tri_in = signal.clone().requires_grad_(True)

        ref_out, tri_out = oracle(ref_in), triton_mod(tri_in)
        upstream = torch.randn_like(ref_out) * factor

        (ref_grad,) = torch.autograd.grad(ref_out, ref_in, upstream)
        (tri_grad,) = torch.autograd.grad(tri_out, tri_in, upstream)

        assert_close(tri_grad, ref_grad, f"grad {orig}->{new} scale {factor}")


@cuda_only
@pytest.mark.parametrize("orig,new", [(48000, 16000), (22050, 16000), (8000, 16000)])
def test_gradient_is_not_trivial(orig, new):
    """The gradient really depends on the upstream gradient, order included.

    A kernel which drops the upstream gradient, or which reads it under a
    permuted index, would still pass a forward-only comparison. Reversing the
    upstream vector has to change the result, and every input sample within a
    filter length of a non-zero upstream entry has to receive something.
    """

    torch.manual_seed(21)
    signal = torch.randn(2, 3000, device="cuda")

    _, triton_mod = build(orig, new)

    tri_in = signal.clone().requires_grad_(True)
    out = triton_mod(tri_in)

    upstream = torch.randn_like(out)
    (grad_a,) = torch.autograd.grad(out, tri_in, upstream, retain_graph=True)
    (grad_b,) = torch.autograd.grad(out, tri_in, upstream.flip(-1))

    assert grad_a.abs().max() > 0, "gradient is identically zero"
    assert not torch.allclose(grad_a, grad_b), "gradient ignores the upstream order"

    # a single non-zero upstream sample must light up exactly one filter window
    spike = torch.zeros_like(out)
    spike[:, out.shape[1] // 2] = 1.0
    (grad_spike,) = torch.autograd.grad(triton_mod(tri_in), tri_in, spike)

    hit = (grad_spike != 0).sum(dim=1)
    assert (hit > 0).all(), "an isolated upstream sample produced no gradient"
    assert (hit <= triton_mod.kernel.shape[1]).all(), "gradient leaked past the filter"


@cuda_only
@pytest.mark.parametrize("orig,new", [(48000, 16000), (22050, 16000), (8000, 16000)])
def test_double_backward(orig, new):
    """``create_graph=True`` works and matches torchaudio.

    Resampling is linear, so the second derivative with respect to the upstream
    gradient is the forward operator again. torchaudio's ``conv1d`` supports
    this and the Triton adjoint is wrapped in its own autograd function so that
    it does too.
    """

    torch.manual_seed(31)
    signal = torch.randn(2, 3000, device="cuda")

    oracle, triton_mod = build(orig, new)
    results = []

    for module in (oracle, triton_mod):
        tri_in = signal.clone().requires_grad_(True)
        out = module(tri_in)

        torch.manual_seed(32)
        upstream = torch.randn_like(out).requires_grad_(True)
        cotangent = torch.randn(signal.shape, device="cuda")

        (grad_in,) = torch.autograd.grad(out, tri_in, upstream, create_graph=True)
        assert grad_in.requires_grad, "the backward is not differentiable"

        (second,) = torch.autograd.grad(grad_in, upstream, cotangent)
        results.append(second)

    assert_close(results[1], results[0], f"double backward {orig}->{new}")


@cuda_only
@pytest.mark.parametrize("orig,new", RATES)
def test_determinism(orig, new):
    """The same input yields bitwise identical results, forward and backward."""

    torch.manual_seed(3)
    signal = torch.randn(4, 12345, device="cuda")

    _, triton_mod = build(orig, new)

    first, second = triton_mod(signal), triton_mod(signal)
    assert torch.equal(first, second)

    upstream = torch.randn_like(first)
    grads = []

    for _ in range(2):
        tri_in = signal.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(triton_mod(tri_in), tri_in, upstream)
        grads.append(grad)

    assert torch.equal(grads[0], grads[1])


def _fwd_launch_args(module, signal, out):
    """Positional kernel arguments of a forward launch, and its geometry."""

    length = signal.shape[-1]
    n_out = resample_module._target_length(length, module.orig_step, module.new_step)
    n_frames = -(-n_out // module.new_step)
    block_j = resample_module._block_inner(module.new_step)

    args = (
        signal,
        module.kernel_t,
        out,
        length,
        n_out,
        module.orig_step,
        module.new_step,
        module.width,
        module.kernel.shape[1],
        n_frames,
        0,
    )

    return args, n_frames, triton.cdiv(module.new_step, block_j), block_j


def _bwd_launch_args(module, grad_out, grad_x):
    """Positional kernel arguments of a backward launch, and its geometry."""

    length = grad_x.shape[-1]
    width_total = module.kernel.shape[1]
    n_out = grad_out.shape[-1]
    n_frames = (length - 1 + module.width) // module.orig_step + 1
    block_r = resample_module._block_inner(module.orig_step)

    args = (
        grad_out,
        module.kernel,
        grad_x,
        length,
        n_out,
        module.orig_step,
        module.new_step,
        module.width,
        width_total,
        -(-width_total // module.orig_step),
        n_frames,
        0,
    )

    return args, n_frames, triton.cdiv(module.orig_step, block_r), block_r


@cuda_only
@pytest.mark.parametrize("orig,new", [(48000, 16000), (44100, 16000), (8000, 16000)])
def test_autotune_configs_agree(orig, new):
    """Every autotuning candidate produces a bitwise identical result.

    Each output element is summed inside a single lane and in tap order, so the
    tile shape only changes the speed. This is what keeps the module
    deterministic even though the tuner may settle on a different candidate on
    another machine or run. Both directions are covered: the backward has its
    own tile shape and its own tap loop.
    """

    torch.manual_seed(9)
    batch, length = 3, 5000
    signal = torch.randn(batch, length, device="cuda")

    module = TritonResample(orig, new).to("cuda")
    n_out = resample_module._target_length(length, module.orig_step, module.new_step)
    upstream = torch.randn(batch, n_out, device="cuda")

    forwards, backwards = [], []
    for config in resample_module._CONFIGS:
        block_l = config.kwargs["BLOCK_L"]

        out = torch.empty((batch, n_out), device="cuda")
        args, n_frames, columns, block_j = _fwd_launch_args(module, signal, out)
        run_one_config(
            resample_module._resample_fwd_kernel,
            (batch * triton.cdiv(n_frames, block_l), columns),
            args,
            BLOCK_L=block_l,
            BLOCK_J=block_j,
            num_warps=config.num_warps,
        )
        forwards.append(out)

        grad_x = torch.empty((batch, length), device="cuda")
        args, n_frames, columns, block_r = _bwd_launch_args(module, upstream, grad_x)
        run_one_config(
            resample_module._resample_bwd_kernel,
            (batch * triton.cdiv(n_frames, block_l), columns),
            args,
            BLOCK_L=block_l,
            BLOCK_R=block_r,
            num_warps=config.num_warps,
        )
        backwards.append(grad_x)

    for other in forwards[1:]:
        assert torch.equal(forwards[0], other)
    for other in backwards[1:]:
        assert torch.equal(backwards[0], other)

    oracle, triton_mod = build(orig, new)
    assert_close(forwards[0], oracle(signal), "config sweep forward")
    assert_close(backwards[0], both_grads(oracle, triton_mod, signal, upstream)[1], "c")


@cuda_only
@pytest.mark.parametrize(
    "orig,new,length", [(48000, 16000, 400_000), (8000, 16000, 200_000)]
)
def test_long_signal_launches(orig, new, length):
    """Regression: long signals used to exceed the CUDA grid limit.

    ``gridDim.y`` and ``gridDim.z`` are capped at 65535 blocks. With the frame
    axis on ``gridDim.y`` the ``BLOCK_L = 2`` candidate died with
    ``CUDA error: invalid argument`` for anything above ~131070 frames, which
    is 8.2 s of 48 kHz audio -- i.e. an ordinary PESQ input. Every candidate is
    launched explicitly here because the autotuner caches its choice per size
    class and would otherwise never retry the small tiles at this size.
    """

    torch.manual_seed(41)
    signal = torch.randn(1, length, device="cuda")

    module = TritonResample(orig, new).to("cuda")
    n_out = resample_module._target_length(length, module.orig_step, module.new_step)

    for config in resample_module._CONFIGS:
        block_l = config.kwargs["BLOCK_L"]

        out = torch.empty((1, n_out), device="cuda")
        args, n_frames, columns, block_j = _fwd_launch_args(module, signal, out)
        grid = (triton.cdiv(n_frames, block_l), columns)
        assert grid[1] <= 65535, "the second grid axis must stay bounded"

        run_one_config(
            resample_module._resample_fwd_kernel,
            grid,
            args,
            BLOCK_L=block_l,
            BLOCK_J=block_j,
            num_warps=config.num_warps,
        )
        torch.cuda.synchronize()

        grad_x = torch.empty((1, length), device="cuda")
        args, n_frames, columns, block_r = _bwd_launch_args(module, out, grad_x)
        grid = (triton.cdiv(n_frames, block_l), columns)
        assert grid[1] <= 65535, "the second grid axis must stay bounded"

        run_one_config(
            resample_module._resample_bwd_kernel,
            grid,
            args,
            BLOCK_L=block_l,
            BLOCK_R=block_r,
            num_warps=config.num_warps,
        )
        torch.cuda.synchronize()

    oracle = torchaudio.transforms.Resample(orig, new).to("cuda")
    assert_close(module(signal), oracle(signal), f"long {orig}->{new} {length}")

    tri_grad, ref_grad = both_grads(oracle, module, signal, seed=42)
    assert_close(tri_grad, ref_grad, f"long grad {orig}->{new} {length}")


@cuda_only
def test_identity_is_exact():
    """``orig_freq == new_freq`` returns the input untouched."""

    signal = torch.randn(3, 1234, device="cuda", requires_grad=True)
    module = TritonResample(16000, 16000).to("cuda")

    out = module(signal)

    assert out is signal
    assert not hasattr(module, "kernel")

    upstream = torch.randn_like(out)
    (grad,) = torch.autograd.grad(out, signal, upstream)
    assert torch.equal(grad, upstream)


@cuda_only
@pytest.mark.parametrize("orig,new", [(48000, 16000), (44100, 16000)])
def test_noncontiguous_and_no_grad(orig, new):
    """Strided inputs, ``no_grad`` and inputs without gradients all work."""

    torch.manual_seed(11)
    dense = torch.randn(3, 8000, device="cuda")
    strided = dense[:, ::2]
    assert not strided.is_contiguous()

    oracle, triton_mod = build(orig, new)

    assert_close(triton_mod(strided), oracle(strided), "strided")

    with torch.no_grad():
        assert_close(triton_mod(dense), oracle(dense), "no_grad")

    with torch.inference_mode():
        assert_close(triton_mod(dense), oracle(dense), "inference_mode")

    plain = dense.clone()
    assert not triton_mod(plain).requires_grad


@cuda_only
@pytest.mark.parametrize("orig,new", [(48000, 16000), (22050, 16000), (8000, 16000)])
def test_backward_noncontiguous(orig, new):
    """Gradients survive strided inputs and a strided upstream gradient."""

    torch.manual_seed(13)
    oracle, triton_mod = build(orig, new)

    for slicer in [lambda t: t[:, ::2], lambda t: t[::2], lambda t: t[1:, 3:-3]]:
        dense = torch.randn(4, 8000, device="cuda")

        ref_root = dense.clone().requires_grad_(True)
        tri_root = dense.clone().requires_grad_(True)

        ref_out = oracle(slicer(ref_root))
        tri_out = triton_mod(slicer(tri_root))

        upstream = torch.randn_like(ref_out)
        (ref_grad,) = torch.autograd.grad(ref_out, ref_root, upstream)
        (tri_grad,) = torch.autograd.grad(tri_out, tri_root, upstream)

        assert_close(tri_grad, ref_grad, f"strided grad {orig}->{new}")

    # a non contiguous upstream gradient
    dense = torch.randn(3, 8000, device="cuda")
    ref_in = dense.clone().requires_grad_(True)
    tri_in = dense.clone().requires_grad_(True)

    ref_out, tri_out = oracle(ref_in), triton_mod(tri_in)
    upstream = torch.randn(3, 2 * ref_out.shape[1], device="cuda")[:, ::2]
    assert not upstream.is_contiguous()

    (ref_grad,) = torch.autograd.grad(ref_out, ref_in, upstream)
    (tri_grad,) = torch.autograd.grad(tri_out, tri_in, upstream)

    assert_close(tri_grad, ref_grad, f"strided upstream {orig}->{new}")


@cuda_only
@pytest.mark.parametrize("orig,new", [(48000, 16000), (22050, 16000)])
@pytest.mark.parametrize("shape", [(3000,), (2, 3, 1500), (2, 1, 3, 700)])
def test_leading_dimensions(orig, new, shape):
    """Any number of leading dimensions is packed and unpacked like torchaudio."""

    torch.manual_seed(17)
    signal = torch.randn(*shape, device="cuda")

    oracle, triton_mod = build(orig, new)
    assert_close(triton_mod(signal), oracle(signal), f"shape {shape}")

    tri_grad, ref_grad = both_grads(oracle, triton_mod, signal, seed=sum(shape))
    assert_close(tri_grad, ref_grad, f"grad shape {shape}")


@cuda_only
@pytest.mark.parametrize("orig,new", [(48000, 16000), (44100, 16000), (8000, 16000)])
def test_no_uninitialised_memory(orig, new):
    """Both kernels write every element of their ``torch.empty`` output.

    The caching allocator is primed with NaN blocks of exactly the right size
    and then released, so a tile that is skipped by a wrong mask hands back a
    NaN instead of a plausible looking stale value.
    """

    torch.manual_seed(19)
    oracle, triton_mod = build(orig, new)

    for length in [999, 4001, 12345]:
        signal = torch.randn(3, length, device="cuda")
        reference = oracle(signal)

        poison = [
            torch.full(reference.shape, float("nan"), device="cuda") for _ in range(8)
        ]
        del poison
        gc.collect()

        out = triton_mod(signal)
        assert torch.isfinite(out).all(), f"forward left holes at {length}"
        assert_close(out, reference, f"poisoned forward {length}")

        upstream = torch.randn_like(reference)
        ref_in = signal.clone().requires_grad_(True)
        (ref_grad,) = torch.autograd.grad(oracle(ref_in), ref_in, upstream)

        poison = [
            torch.full(signal.shape, float("nan"), device="cuda") for _ in range(8)
        ]
        del poison
        gc.collect()

        tri_in = signal.clone().requires_grad_(True)
        (tri_grad,) = torch.autograd.grad(triton_mod(tri_in), tri_in, upstream)

        assert torch.isfinite(tri_grad).all(), f"backward left holes at {length}"
        assert_close(tri_grad, ref_grad, f"poisoned backward {length}")


@cuda_only
@pytest.mark.parametrize("orig,new", RATES)
def test_output_length_matches_oracle(orig, new):
    """The emitted length agrees with torchaudio's float32 ``ceil`` everywhere.

    torchaudio rounds ``new_step * n / orig_step`` up in float32, which starts
    to disagree with an exact integer ceil above ~180k samples at 44.1 kHz.
    """

    oracle, triton_mod = build(orig, new)
    lengths = list(range(1, 32)) + [
        255,
        256,
        257,
        4096,
        100_000,
        180_000,
        180_001,
        180_224,
        262_144,
        524_288,
    ]

    for length in lengths:
        signal = torch.zeros(1, length, device="cuda")
        assert (
            triton_mod(signal).shape == oracle(signal).shape
        ), f"{orig}->{new} length {length}"


@cuda_only
@pytest.mark.parametrize(
    "orig,new", [(1000, 999), (13, 7), (7, 13), (8000, 11025), (44100, 48000), (2, 1)]
)
def test_exotic_rate_pairs(orig, new):
    """Rate pairs outside the PESQ defaults, including coprime ones."""

    torch.manual_seed(23)
    signal = torch.randn(2, 3001, device="cuda")

    oracle, triton_mod = build(orig, new)
    assert_close(triton_mod(signal), oracle(signal), f"{orig}->{new}")

    tri_grad, ref_grad = both_grads(oracle, triton_mod, signal, seed=orig)
    assert_close(tri_grad, ref_grad, f"grad {orig}->{new}")


@cuda_only
@pytest.mark.parametrize(
    "options",
    [
        dict(lowpass_filter_width=1),
        dict(lowpass_filter_width=64),
        dict(rolloff=0.5),
        dict(resampling_method="sinc_interp_kaiser", beta=8.0),
        dict(
            lowpass_filter_width=16,
            rolloff=0.945,
            resampling_method="sinc_interp_kaiser",
        ),
    ],
)
@pytest.mark.parametrize("orig,new", [(48000, 16000), (22050, 16000)])
def test_filter_design_variants(orig, new, options):
    """Non default filter designs are forwarded to torchaudio unchanged.

    ``lowpass_filter_width=64`` pushes ``width_total`` to 621 taps, the longest
    accumulation the kernels ever run.
    """

    torch.manual_seed(5)
    signal = torch.randn(2, 4321, device="cuda")

    oracle, triton_mod = build(orig, new, **options)
    assert_close(triton_mod(signal), oracle(signal), f"{options}")

    tri_grad, ref_grad = both_grads(oracle, triton_mod, signal, seed=7)
    assert_close(tri_grad, ref_grad, f"grad {options}")


@cuda_only
def test_moved_after_construction():
    """The kernel buffer follows ``.cuda()`` after a CPU construction."""

    torch.manual_seed(2)
    signal = torch.randn(2, 4096, device="cuda")

    module = TritonResample(48000, 16000)
    assert module.kernel.device.type == "cpu"

    with pytest.raises(RuntimeError, match="move the module"):
        module(signal)

    module = module.cuda()
    oracle = torchaudio.transforms.Resample(48000, 16000).to("cuda")

    assert_close(module(signal), oracle(signal), "moved")


@cuda_only
def test_input_validation():
    """Bad dtypes, bad devices and out of range sizes fail loudly."""

    module = TritonResample(48000, 16000).cuda()

    with pytest.raises(RuntimeError, match="float32"):
        module(torch.randn(2, 100, device="cuda", dtype=torch.float64))

    # a CUDA module fed a CPU tensor is caught by the device check ...
    with pytest.raises(RuntimeError, match="move the module"):
        module(torch.randn(2, 100))

    # ... and a CPU module fed a CPU tensor by the backend check
    with pytest.raises(RuntimeError, match="GPU"):
        TritonResample(48000, 16000)(torch.randn(2, 100))

    with pytest.raises(ValueError):
        TritonResample(0, 16000)

    # 32 bit indices are checked on the host instead of wrapping silently
    sizes = (0, 0, module.orig_step, module.new_step, module.width, 41)
    with pytest.raises(RuntimeError, match="32 bit"):
        resample_module._check_index_range(2**31, 2**30, sizes)

    resample_module._check_index_range(1_000_000, 400_000, sizes)


@cuda_only
def test_degenerate_shapes():
    """Empty batches and empty signals return empty tensors instead of crashing.

    torchaudio raises on both, so there is nothing to compare against; the
    contract here is only that no kernel is launched with a broken grid.
    """

    module = TritonResample(48000, 16000).cuda()

    assert module(torch.randn(0, 1000, device="cuda")).shape == (0, 334)
    assert module(torch.randn(2, 0, device="cuda")).shape == (2, 0)
    assert module(torch.randn(0, 0, device="cuda")).shape == (0, 0)


@cuda_only
def test_tolerance_rejects_wrong_answers():
    """The tolerance is tight enough to catch the bugs it is supposed to catch.

    A scale aware bound is only useful if it still fails for a plausible kernel
    bug. Three of them are injected into a correct result: a one sample shift
    (an off-by-one in the tap index), a zeroed tail (a dropped boundary tile)
    and a single element perturbed by ``1e-4`` (a mis-masked lane).
    """

    torch.manual_seed(29)
    signal = torch.randn(2, 5000, device="cuda")

    oracle, triton_mod = build(48000, 16000)
    reference = oracle(signal)

    assert_close(triton_mod(signal), reference, "sanity")

    for name, mutate in [
        ("shift", lambda t: t.roll(1, dims=-1)),
        (
            "dropped tail",
            lambda t: torch.cat([t[:, :-8], torch.zeros_like(t[:, -8:])], -1),
        ),
        ("one element", lambda t: t + torch.eye(1, t.shape[1], device="cuda") * 1e-4),
    ]:
        with pytest.raises(AssertionError):
            assert_close(mutate(reference), reference, name)


@cuda_only
def test_report_accuracy_and_timing(capsys):
    """Print the measured accuracy and timing, this test never fails on speed."""

    def median_ms(fn, reps=25):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()

        samples = []
        for _ in range(reps):
            start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            start.record()
            fn()
            stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop))

        return statistics.median(samples)

    torch.manual_seed(0)
    batch, length = 8, 48000

    lines = []
    for orig, new in RATES:
        signal = torch.randn(batch, length, device="cuda")
        oracle, triton_mod = build(orig, new)

        ref_out, tri_out = oracle(signal), triton_mod(signal)
        fwd_abs, fwd_rel = errors(tri_out, ref_out)

        ref_in = signal.clone().requires_grad_(True)
        tri_in = signal.clone().requires_grad_(True)
        upstream = torch.randn_like(ref_out)

        (ref_grad,) = torch.autograd.grad(oracle(ref_in), ref_in, upstream)
        (tri_grad,) = torch.autograd.grad(triton_mod(tri_in), tri_in, upstream)
        bwd_abs, bwd_rel = errors(tri_grad, ref_grad)

        ref_fwd = median_ms(lambda: oracle(signal))
        tri_fwd = median_ms(lambda: triton_mod(signal))
        ref_bwd = median_ms(
            lambda: torch.autograd.grad(oracle(ref_in), ref_in, upstream)
        )
        tri_bwd = median_ms(
            lambda: torch.autograd.grad(triton_mod(tri_in), tri_in, upstream)
        )

        lines.append(
            f"{orig:>6} -> {new:<6} fwd abs {fwd_abs:.2e} rel {fwd_rel:.2e} | "
            f"bwd abs {bwd_abs:.2e} rel {bwd_rel:.2e} | "
            f"fwd {tri_fwd:7.3f} ms vs {ref_fwd:7.3f} ms ({tri_fwd / ref_fwd:5.2f}x) | "
            f"fwd+bwd {tri_bwd:7.3f} ms vs {ref_bwd:7.3f} ms "
            f"({tri_bwd / ref_bwd:5.2f}x)"
        )

    with capsys.disabled():
        print(f"\n[resample] batch={batch} length={length}")
        for line in lines:
            print("[resample] " + line)
