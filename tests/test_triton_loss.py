"""End to end tests of the Triton backend against the PyTorch implementation."""

import pytest
import torch

from torch_pesq import PesqLoss
from torch_pesq.triton_ops import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON),
    reason="needs a CUDA GPU with Triton",
)

if HAS_TRITON and torch.cuda.is_available():
    from torch_pesq.triton_ops.loss import PesqLossTriton

# The PyTorch implementation evaluates the perceptual model in float64 (the Bark
# tables are built by scipy and never cast down), the Triton backend runs in
# float32 throughout. Together with a different STFT round off this limits the
# agreement to roughly single precision on the distances. The chain also
# contains hard thresholds (silent frames, the asymmetric scaling cut at 3.0),
# so individual band decisions can flip; the tolerances below are chosen to
# cover that while still being tight enough to catch real errors.
RTOL, ATOL = 2e-3, 2e-3
GRAD_RTOL, GRAD_ATOL = 5e-3, 5e-3


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


def make_signals(batch, samples, seed=0, device="cuda"):
    """Speech like reference and a noisy degraded version of it."""

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
    ref = ref / ref.abs().amax(dim=1, keepdim=True)

    noise = torch.randn(batch, samples, device=device, generator=generator)
    deg = ref + 0.2 * noise

    return ref.contiguous(), deg.contiguous()


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
