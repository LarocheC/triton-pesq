"""End to end tests of the Triton backend against the PyTorch implementation."""

import pytest
import torch

from torch_pesq import PesqLoss
from torch_pesq.triton_ops import HAS_TRITON
from torch_pesq.triton_ops.reference import (
    ref_align_level,
    ref_chain,
    ref_preemphasize,
    ref_stft_bark,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON),
    reason="needs a CUDA GPU with Triton",
)

if HAS_TRITON and torch.cuda.is_available():
    from torch_pesq.triton_ops.loss import PesqLossTriton

# Comparing the two *backends* can never be tight, and not because of this one:
# `torchaudio.functional.lfilter` evaluates the order 10 level alignment filter
# as a float32 direct-form-I recursion and is itself off by ~1e-2 relative
# against float64, which moves the alignment gain of the PyTorch backend by
# ~1e-4. The chain also contains hard thresholds (silent frames, the asymmetric
# scaling cut at 3.0), so a rounding difference can flip a single band. These
# budgets cover that; what actually pins the kernels down is
# `test_matches_float64_reference` below, which compares against a float64
# evaluation of the same pipeline instead of against the other backend.
RTOL, ATOL = 2e-3, 2e-3
GRAD_RTOL, GRAD_ATOL = 5e-3, 5e-3

# Triton against a float64 evaluation of the same pipeline. Measured worst case
# over the signals of this file is 3.1e-07 on the distances; the budget leaves
# room for a single flipped band decision.
F64_RTOL = 1e-4

# Below this the two backends are equally right and the comparison in
# `test_matches_float64_reference` carries no information.
F64_FLOOR = 1e-6


def float32_reference(model: PesqLoss) -> PesqLoss:
    """Cast the float64 constants of a PesqLoss to float32."""

    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is not None and param.dtype == torch.float64:
                module._parameters[name] = torch.nn.Parameter(
                    param.float(), requires_grad=False
                )

    model.fbank.total_width = model.fbank.total_width.float()

    return model


def make_signals(batch, samples, seed=0, device="cuda", noise=0.2, fricative=0.3):
    """Speech like reference and a noisy degraded version of it.

    The reference is a voiced harmonic stack **plus** a broadband component, and
    the second half of that is not decoration. A purely harmonic reference has
    no energy at all between its partials, so additive white noise lands in
    bands where the reference is silent, the asymmetric scaling saturates at
    ``12`` everywhere and both distances come out pinned at their ``45.0``
    clamp: with ``fricative=0`` and ``noise=0.2``, 90% of the frames hit the
    symmetric clamp and 100% hit the asymmetric one, and ``d_asymm`` stays
    exactly ``45.0`` even down to ``noise=0.001``. Every differential test in
    this file would then be comparing two constants.

    With the broadband term the distances land around ``15`` and ``36``, in the
    middle of the range the loss is actually used in, and the comparisons below
    have something to compare.
    """

    generator = torch.Generator(device=device).manual_seed(seed)

    time = torch.arange(samples, device=device) / 16000.0
    envelope = 0.5 + 0.5 * torch.sin(2 * torch.pi * 3.0 * time)

    ref = torch.zeros(batch, samples, device=device)
    for harmonic in range(1, 12):
        phase = torch.rand(batch, 1, device=device, generator=generator) * 6.28
        ref += (
            torch.sin(2 * torch.pi * 120.0 * harmonic * time + phase)
            / harmonic
            * envelope
        )
    ref = ref + fricative * envelope * torch.randn(
        batch, samples, device=device, generator=generator
    )
    ref = ref / ref.abs().amax(dim=1, keepdim=True)

    deg = ref + noise * torch.randn(batch, samples, device=device, generator=generator)

    return ref.contiguous(), deg.contiguous()


def float64_distances(model, ref, deg):
    """The same pipeline evaluated in float64 on the CPU, as ground truth.

    Uses the stage decomposition of :mod:`torch_pesq.triton_ops.reference`,
    which is verified to reproduce :meth:`PesqLoss.raw` bit for bit, with every
    constant promoted to float64. Only valid at 16 kHz, where no resampling
    happens.
    """

    assert model.source_sample_rate == 16000

    ref = ref.detach().double().cpu()
    deg = deg.detach().double().cpu()

    peak = torch.max(deg.abs().amax(1, True), ref.abs().amax(1, True))
    ref, deg = ref / peak, deg / peak

    power = model.power_filter.detach().double().cpu()
    pre = model.pre_filter.detach().double().cpu()

    ref, deg = ref_align_level(ref, power), ref_align_level(deg, power)
    ref, deg = ref_preemphasize(ref, pre), ref_preemphasize(deg, pre)

    ref = torch.nn.functional.pad(ref, (0, ref.shape[1] % 256))
    deg = torch.nn.functional.pad(deg, (0, deg.shape[1] % 256))

    window = model.to_spec.window.detach().double().cpu()
    fbank = model.fbank.fbank.detach().double().cpu()
    correction = model.fbank.pow_dens_correction.detach().double().cpu()
    bark = [
        ref_stft_bark(
            signal,
            window,
            fbank,
            correction,
            model.to_spec.n_fft,
            model.to_spec.hop_length,
        )
        for signal in (ref, deg)
    ]

    return ref_chain(
        bark[0],
        bark[1],
        model.loudness.threshs.detach().double().cpu(),
        model.loudness.exp.detach().double().cpu(),
        model.fbank.width_bark.detach().double().cpu(),
        torch.as_tensor(model.fbank.total_width).double().cpu(),
        0.1866055,
    )


@pytest.mark.parametrize(
    "batch,samples,sample_rate",
    [
        (1, 16000, 16000),
        (4, 16000, 16000),
        (3, 20000, 16000),  # length that is not a multiple of the hop size
        (2, 48000, 48000),
        (2, 44100, 44100),
        (8, 32000, 16000),
    ],
)
def test_forward_matches_pytorch(batch, samples, sample_rate):
    """Distances, MOS and loss agree with the PyTorch implementation."""

    torch_loss = PesqLoss(0.5, sample_rate=sample_rate).cuda()
    triton_loss = PesqLossTriton(0.5, sample_rate=sample_rate).cuda()

    ref, deg = make_signals(batch, samples, seed=batch + samples)

    expected = torch_loss.raw(ref, deg)
    actual = triton_loss.raw(ref, deg)

    for name, want, got in zip(("d_symm", "d_asymm"), expected, actual):
        torch.testing.assert_close(
            got.double(), want.double(), rtol=RTOL, atol=ATOL, msg=lambda m: name + m
        )

    torch.testing.assert_close(
        triton_loss.mos(ref, deg).double(),
        torch_loss.mos(ref, deg).double(),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        triton_loss(ref, deg).double(),
        torch_loss(ref, deg).double(),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "batch,samples,seed", [(2, 16000, 0), (2, 16000, 7), (3, 20000, 13)]
)
def test_signals_are_not_clamp_saturated(batch, samples, seed):
    """The test signals have to live inside the range the loss is used in.

    Both distances are clamped at ``45.0`` per frame. A reference without
    broadband energy pushes every frame into that clamp (see
    :func:`make_signals`), which turns every differential test in this file
    into a comparison of two constants and hides real disagreement between the
    backends. This guards against that happening again.
    """

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, deg = make_signals(batch, samples, seed=seed)

    d_symm, d_asymm = (d.double() for d in triton_loss.raw(ref, deg))

    assert torch.all(d_symm > 1.0) and torch.all(d_symm < 42.0)
    assert torch.all(d_asymm > 1.0) and torch.all(d_asymm < 42.0)


@pytest.mark.parametrize(
    "batch,samples,noise", [(2, 16000, 0.2), (2, 16000, 0.05), (3, 20000, 0.1)]
)
def test_matches_float64_reference(batch, samples, noise):
    """Triton agrees with a float64 evaluation of the same pipeline.

    This is the test with teeth. Comparing the two backends against each other
    only bounds their *disagreement*, and most of that disagreement is the
    PyTorch backend: its ``lfilter`` carries ~1e-2 relative error on the level
    alignment filter, which is three orders of magnitude more than the Triton
    kernels contribute. Measured against float64 the Triton distances land at
    ~1e-7 while the PyTorch ones land at ~1e-4, so the assertion is both that
    Triton is close to float64 in absolute terms and that it is not the worse
    of the two.
    """

    torch_loss = PesqLoss(1.0, sample_rate=16000).cuda()
    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()

    ref, deg = make_signals(batch, samples, seed=batch + samples, noise=noise)

    exact = float64_distances(torch_loss, ref, deg)
    with torch.no_grad():
        torch_out = torch_loss.raw(ref, deg)
        triton_out = triton_loss.raw(ref, deg)

    for name, want, other, got in zip(
        ("d_symm", "d_asymm"), exact, torch_out, triton_out
    ):
        want = want.double()
        scale = want.abs().clamp(min=1e-12)

        triton_err = ((got.double().cpu() - want).abs() / scale).max().item()
        torch_err = ((other.double().cpu() - want).abs() / scale).max().item()

        assert triton_err < F64_RTOL, f"{name}: {triton_err:.3e} against float64"
        assert triton_err <= max(torch_err, F64_FLOOR), (
            f"{name}: Triton is further from float64 ({triton_err:.3e}) than the "
            f"PyTorch backend ({torch_err:.3e})"
        )


@pytest.mark.parametrize("batch,samples", [(1, 16000), (4, 24000)])
def test_backward_matches_pytorch(batch, samples):
    """Gradients agree with autograd through the PyTorch implementation."""

    torch_loss = PesqLoss(1.0, sample_rate=16000).cuda()
    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()

    ref, deg = make_signals(batch, samples, seed=7)
    grad_out = torch.randn(batch, device="cuda", generator=None)

    def grads(model, inputs):
        a, b = (t.detach().clone().requires_grad_(True) for t in inputs)
        out = model(a, b)
        return torch.autograd.grad(out, [a, b], grad_outputs=grad_out.to(out.dtype))

    expected = grads(torch_loss, (ref, deg))
    actual = grads(triton_loss, (ref, deg))

    for name, want, got in zip(("d_ref", "d_deg"), expected, actual):
        scale = want.abs().max()
        torch.testing.assert_close(
            got.double() / scale,
            want.double() / scale,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            msg=lambda m: name + m,
        )


def test_identical_signals():
    """A degraded signal identical to the reference has (almost) no loss."""

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, _ = make_signals(3, 16000, seed=11)

    assert torch.all(triton_loss(ref, ref) <= 1e-6)
    assert torch.all(triton_loss(ref, ref) >= 0.0)


def test_loss_positive_and_mos_range():
    """The loss is positive and the MOS stays inside its range."""

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, deg = make_signals(8, 16000, seed=13)

    assert torch.all(triton_loss(ref, deg) > 0.0)

    mos = triton_loss.mos(ref, deg)
    assert torch.all(mos > 1.0) and torch.all(mos < 5.0)


def test_gradient_flows_and_is_finite():
    """Gradients exist, are finite and are non zero for both inputs."""

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, deg = make_signals(2, 16000, seed=17)
    ref.requires_grad_(True)
    deg.requires_grad_(True)

    triton_loss(ref, deg).sum().backward()

    for grad in (ref.grad, deg.grad):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0.0


def test_deterministic():
    """Repeated evaluation is bitwise reproducible."""

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, deg = make_signals(4, 16000, seed=19)
    ref.requires_grad_(True)

    first = triton_loss(ref, deg)
    (grad_first,) = torch.autograd.grad(first.sum(), ref)
    second = triton_loss(ref, deg)
    (grad_second,) = torch.autograd.grad(second.sum(), ref)

    assert torch.equal(first, second)
    assert torch.equal(grad_first, grad_second)


def test_batch_independence():
    """Every item of a batch is scored independently."""

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, deg = make_signals(4, 16000, seed=23)

    batched = triton_loss(ref, deg)
    single = torch.cat([triton_loss(ref[i : i + 1], deg[i : i + 1]) for i in range(4)])

    torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-6)


def test_non_contiguous_input():
    """Strided inputs are handled."""

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, deg = make_signals(4, 32000, seed=29)

    strided_ref, strided_deg = ref[:, ::2], deg[:, ::2]
    assert not strided_ref.is_contiguous()

    expected = triton_loss(strided_ref.contiguous(), strided_deg.contiguous())
    torch.testing.assert_close(triton_loss(strided_ref, strided_deg), expected)


def test_one_dimensional_input():
    """Unbatched input is promoted to a batch of one."""

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, deg = make_signals(1, 16000, seed=31)

    torch.testing.assert_close(triton_loss(ref[0], deg[0]), triton_loss(ref, deg))


@pytest.mark.parametrize("seed", range(6))
def test_random_signals_match_pytorch(seed):
    """Differential test over random shapes, levels and degradation types."""

    generator = torch.Generator(device="cuda").manual_seed(1000 + seed)

    batch = int(torch.randint(1, 6, (1,), generator=generator, device="cuda").item())
    samples = int(
        torch.randint(16000, 40000, (1,), generator=generator, device="cuda").item()
    )
    level = 10 ** (
        torch.rand(1, generator=generator, device="cuda").item() * 4 - 2
    )  # 1e-2 .. 1e2

    ref, deg = make_signals(batch, samples, seed=seed)
    ref = ref * level
    deg = deg * level

    if seed % 3 == 1:  # heavily degraded
        deg = deg + 2.0 * torch.randn(
            batch, samples, device="cuda", generator=generator
        )
    elif seed % 3 == 2:  # clipped
        deg = deg.clamp(-0.3 * level, 0.3 * level)

    torch_loss = PesqLoss(1.0, sample_rate=16000).cuda()
    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()

    with torch.no_grad():
        want = torch_loss(ref, deg).double()
        got = triton_loss(ref, deg).double()

    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("sample_rate,seconds", [(48000, 12.0), (16000, 30.0)])
def test_long_signal(sample_rate, seconds):
    """Long inputs work; the resampler grid used to overflow past ~8.2 seconds."""

    samples = int(sample_rate * seconds)
    torch_loss = PesqLoss(1.0, sample_rate=sample_rate).cuda()
    triton_loss = PesqLossTriton(1.0, sample_rate=sample_rate).cuda()

    ref, deg = make_signals(1, samples, seed=41)

    with torch.no_grad():
        want = torch_loss(ref, deg).double()
        got = triton_loss(ref, deg).double()

    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_cuda_graph_is_bit_exact():
    """A captured graph reproduces the eager loss and gradients exactly."""

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, deg = make_signals(2, 16000, seed=43)
    ref, deg = ref.requires_grad_(True), deg.requires_grad_(True)

    eager = triton_loss(ref, deg)
    eager_grads = torch.autograd.grad(eager.sum(), [ref, deg])

    graphed = triton_loss.graphed(ref, deg)
    captured = graphed(ref, deg)
    captured_grads = torch.autograd.grad(captured.sum(), [ref, deg])

    assert torch.equal(captured, eager)
    for want, got in zip(eager_grads, captured_grads):
        assert torch.equal(got, want)


def test_cpu_input_raises():
    """A helpful error is raised for tensors that are not on the GPU."""

    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()
    ref, deg = make_signals(1, 16000, seed=37, device="cpu")

    with pytest.raises(RuntimeError):
        triton_loss(ref, deg)
