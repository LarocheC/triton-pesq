"""Parity tests for the Triton implementation of the PESQ perceptual chain.

The oracle is :func:`torch_pesq.triton_ops.reference.ref_chain`, which is
verified to reproduce :meth:`torch_pesq.loss.PesqLoss.raw` bit for bit.  Because
:attr:`torch_pesq.loudness.Loudness.threshs` is a float64 buffer the oracle
silently runs the whole chain in float64, so every comparison here is float32
against float64.

Tolerances
----------
Two different budgets are used, and neither is a knob that was turned until the
suite went green:

``FORWARD_RTOL`` / ``GRAD_RTOL`` (``6e-06`` / ``3e-05``)
    used wherever the chain is well conditioned, i.e. everywhere the two
    spectra differ by at least a few percent.  Measured worst case over a
    60 point random fuzz (batch 1..8, 20..260 frames, spectra spread over 0.05
    to 4 decades, levels over 1 to 9 decades, ``test_random_fuzz``) is
    ``3.0e-06`` on the forward and ``1.1e-05`` on the gradients, both relative
    to the largest oracle magnitude.
    The gradient budget is the loose one of the pair because the disturbance
    passes through a subtraction of two loudnesses whose derivatives are
    ``O(1/th)`` apart.

a *relative to float32 PyTorch* bound
    the disturbance ``sign(u) (|u| - 0.25 min(deg_loud, ref_loud))`` cancels
    twice when ``deg`` is almost equal to ``ref``, and by ``spread = 0.005`` a
    plain float32 PyTorch evaluation of the same formulas is already off by
    ``9e-05``.  Asserting a fixed number there would only measure how ill
    conditioned the input happens to be, so ``test_conditioning_matches_float32``
    asserts that the kernels are no worse than that float32 reference instead.
    See :func:`_chain_float32`.

``test_near_threshold_bands`` carries its own table of budgets, see its
docstring: bands within a hair of their hearing threshold are a third regime,
and one where the *formulation* of the loudness decides the answer.
"""

import functools
import time

import pytest
import torch

from torch.nn.functional import unfold

from torch_pesq.bark import BarkScale
from torch_pesq.loudness import Loudness
from torch_pesq.triton_ops._common import HAS_TRITON
from torch_pesq.triton_ops.reference import ref_chain

if HAS_TRITON:
    from torch_pesq.triton_ops.chain import TritonPesqChain

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON),
    reason="the Triton backend needs a CUDA device and the triton package",
)

SL = 0.1866055
NBARK = 49

FORWARD_RTOL = 6e-6
GRAD_RTOL = 3e-5


@functools.lru_cache(maxsize=None)
def _params(device="cuda"):
    """Hearing thresholds, exponents and band widths, exactly as ``PesqLoss`` has them."""

    loudness = Loudness(NBARK).to(device)
    fbank = BarkScale(256, NBARK).to(device)

    return (
        loudness.threshs.detach(),
        loudness.exp.detach(),
        fbank.width_bark.detach(),
        fbank.total_width,
    )


def _oracle(ref, deg):
    threshs, exps, width, total = _params(str(ref.device))

    return ref_chain(ref, deg, threshs, exps, width, total, SL)


@functools.lru_cache(maxsize=None)
def _module(device="cuda"):
    threshs, exps, width, total = _params(device)

    return TritonPesqChain(threshs, exps, width, total, SL).to(device)


def _spectra(batch, nframe, seed, spread=0.5, level=6.0, device="cuda"):
    """Log-uniform Bark spectra straddling the hearing thresholds.

    The per frame level spans nine decades so that the silent frame detection,
    the ``band_pow_ratio`` clamp, the ``frame_pow_ratio`` clamp, the loudness
    threshold and the deadzone all trigger somewhere in the batch.
    """

    gen = torch.Generator(device=device).manual_seed(seed)
    threshs = _params(device)[0].reshape(1, 1, -1).float()

    shape = (batch, nframe, NBARK)
    level = 10 ** (torch.rand((batch, nframe, 1), device=device, generator=gen) * level)
    tilt = 0.2 + 2.0 * torch.rand(shape, device=device, generator=gen)
    ref = threshs * level * tilt

    pert = 10 ** ((torch.rand(shape, device=device, generator=gen) - 0.5) * 2 * spread)

    return ref.float().contiguous(), (ref * pert).float().contiguous()


def _errors(mine, oracle):
    """Maximum absolute error and error relative to the largest oracle magnitude."""

    mine, oracle = mine.double(), oracle.double()
    abs_err = (mine - oracle).abs().max().item()
    scale = oracle.abs().max().item()

    return abs_err, abs_err / max(scale, 1e-300)


def _upstream(batch, seed, device="cuda"):
    """A random, never all-ones, upstream gradient for each of the two outputs."""

    gen = torch.Generator(device=device).manual_seed(seed + 1)

    return (
        torch.randn(batch, device=device, generator=gen),
        torch.randn(batch, device=device, generator=gen),
    )


def _run_triton(ref, deg, seed):
    """Forward and both gradients of the Triton chain for a random upstream."""

    module = _module(str(ref.device))
    ref_a = ref.clone().requires_grad_(True)
    deg_a = deg.clone().requires_grad_(True)

    symm, asymm = module(ref_a, deg_a)
    up_s, up_a = _upstream(ref.shape[0], seed, str(ref.device))
    grads = torch.autograd.grad(
        (symm * up_s).sum() + (asymm * up_a).sum(), (ref_a, deg_a)
    )

    return symm, asymm, grads[0], grads[1]


def _run_oracle(ref, deg, seed):
    """Same quantities from the float64 oracle."""

    ref_b = ref.clone().requires_grad_(True)
    deg_b = deg.clone().requires_grad_(True)

    symm, asymm = _oracle(ref_b, deg_b)
    up_s, up_a = _upstream(ref.shape[0], seed, str(ref.device))
    grads = torch.autograd.grad(
        (symm * up_s.double()).sum() + (asymm * up_a.double()).sum(), (ref_b, deg_b)
    )

    return symm, asymm, grads[0], grads[1]


def _compare(ref, deg, seed=0, forward_rtol=FORWARD_RTOL, grad_rtol=GRAD_RTOL):
    """Run both implementations, check them and return the four error pairs."""

    mine = _run_triton(ref, deg, seed)
    theirs = _run_oracle(ref, deg, seed)

    names = ("symm", "asymm", "gref", "gdeg")
    out = {n: _errors(m, o) for n, m, o in zip(names, mine, theirs)}

    for tensor in mine:
        assert torch.isfinite(tensor).all(), "the Triton chain produced NaN or Inf"

    for name in ("symm", "asymm"):
        assert out[name][1] <= forward_rtol, f"{name}: {out[name]}"
    for name in ("gref", "gdeg"):
        assert out[name][1] <= grad_rtol, f"{name}: {out[name]}"

    return out


def _show(tag, errs):
    print(
        f"\n{tag} " + " ".join(f"{k}=({v[0]:.3e}, {v[1]:.3e})" for k, v in errs.items())
    )


# ---------------------------------------------------------------------------
# a float32 PyTorch evaluation of the very same formulas, used as the
# conditioning yardstick for the near identical spectra regime
# ---------------------------------------------------------------------------


def _chain_float32(ref, deg, threshs, exps, width_bark, total_width, sl):
    """``ref_chain`` with every constant kept in the dtype of ``ref``.

    Identical to :func:`torch_pesq.triton_ops.reference.ref_chain` except that
    the sixth power mean of the overlapping sums is evaluated on window
    normalised values.  That is not a numerical liberty but a necessity: the
    disturbance is clamped to a floor of ``1e-20`` and ``1e-20 ** 6 = 1e-120``
    flushes to zero in float32, which makes the *oracle itself* return NaN
    gradients.  The Triton kernels normalise for the same reason.
    """

    silent = (ref * (ref > threshs * 1e2)).sum(dim=2) < 1e7
    mask_ref = (ref > threshs * 100.0) * (~silent.unsqueeze(2))
    mask_deg = (deg > threshs * 100.0) * (~silent.unsqueeze(2))

    band_pow_ratio = (
        (((deg * mask_deg).mean(dim=1) + 1000) / ((ref * mask_ref).mean(dim=1) + 1000))
        .unsqueeze(1)
        .clamp(min=0.01, max=100.0)
    )
    equ_ref = band_pow_ratio * ref

    taer = (equ_ref * (equ_ref > threshs)).sum(dim=2)
    fpr = (taer + 5e3) / ((deg * (deg > threshs)).sum(dim=2) + 5e3)
    fpr = torch.cat([fpr[:, :1], fpr[:, 1:] * 0.8 + fpr[:, :-1] * 0.2], dim=1)
    fpr = fpr.clamp(min=3e-4, max=5.0)
    equ_deg = fpr.unsqueeze(2) * deg

    def loudness(pow_dens):
        loud = (2.0 * threshs) ** exps * ((0.5 + 0.5 * pow_dens / threshs) ** exps - 1)
        return torch.where(pow_dens <= threshs, torch.zeros_like(loud), loud) * sl

    deg_loud, ref_loud = loudness(equ_deg), loudness(equ_ref)
    distu = deg_loud - ref_loud
    distu = distu.sign() * (distu.abs() - 0.25 * torch.min(deg_loud, ref_loud)).clamp(
        min=0
    )

    def weighted_norm(tensor, p):
        return total_width * (width_bark * tensor / total_width ** (1 / p))[
            :, :, 1:
        ].norm(p, dim=2)

    asymm_scaling = ((equ_deg + 50.0) / (equ_ref + 50.0)) ** 1.2
    asymm_scaling = torch.where(
        asymm_scaling < 3.0, torch.zeros_like(asymm_scaling), asymm_scaling
    ).clamp(max=12.0)

    h = ((taer + 1e5) / 1e7) ** 0.04
    symm = (weighted_norm(distu, 2.0).clamp(min=1e-20) / h).clamp(max=45.0)
    asymm = (weighted_norm(distu * asymm_scaling, 1.0).clamp(min=1e-20) / h).clamp(
        max=45.0
    )

    def overlapping(dist):
        win = unfold(dist.unsqueeze(1).unsqueeze(1), (1, 20), stride=10)
        scale = win.amax(dim=1, keepdim=True).clamp(min=torch.finfo(dist.dtype).tiny)
        psqm = scale.squeeze(1) * ((win / scale) ** 6).mean(dim=1) ** (1.0 / 6)
        scale = psqm.amax(dim=1, keepdim=True).clamp(min=torch.finfo(dist.dtype).tiny)
        return scale.squeeze(1) * (psqm / scale).square().mean(dim=1).sqrt()

    return overlapping(symm), overlapping(asymm)


def _run_float32(ref, deg, seed):
    threshs, exps, width, total = _params(str(ref.device))
    ref_c = ref.clone().requires_grad_(True)
    deg_c = deg.clone().requires_grad_(True)

    symm, asymm = _chain_float32(
        ref_c, deg_c, threshs.float(), exps.float(), width.float(), float(total), SL
    )
    up_s, up_a = _upstream(ref.shape[0], seed, str(ref.device))
    grads = torch.autograd.grad(
        (symm * up_s).sum() + (asymm * up_a).sum(), (ref_c, deg_c)
    )

    return symm, asymm, grads[0], grads[1]


# ---------------------------------------------------------------------------
# forward and backward parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("nframe", [20, 21, 62, 200])
def test_parity_sweep(batch, nframe):
    """Forward and both gradients agree over the batch/length sweep."""

    ref, deg = _spectra(batch, nframe, seed=batch * 1000 + nframe)
    _show(f"batch={batch} nframe={nframe}", _compare(ref, deg, seed=nframe))


@pytest.mark.parametrize("nframe", [29, 30, 31, 39, 40, 41])
def test_parity_window_boundaries(nframe):
    """Lengths straddling the 20 frame window and its stride of 10.

    ``nwin = (nframe - 20) // 10 + 1`` changes at 30 and 40, and none of these
    lengths is a multiple of the ``BLOCK_T = 4`` frame tile either.
    """

    ref, deg = _spectra(2, nframe, seed=nframe)
    _show(f"nframe={nframe}", _compare(ref, deg, seed=nframe))


@pytest.mark.parametrize("spread", [0.05, 3.0])
def test_parity_disturbance_levels(spread):
    """Near identical and wildly different spectra, exercising both norm clamps."""

    ref, deg = _spectra(4, 63, seed=int(spread * 100), spread=spread)
    _show(f"spread={spread}", _compare(ref, deg, seed=7))


def test_many_silent_frames():
    """Most frames fall below the ``1e7`` total audible energy of the silence test."""

    ref, deg = _spectra(4, 71, seed=11)
    gen = torch.Generator(device="cuda").manual_seed(12)
    silent = torch.rand((4, 71, 1), device="cuda", generator=gen) < 0.8

    ref = torch.where(silent, ref * 1e-7, ref).contiguous()
    deg = torch.where(silent, deg * 1e-7, deg).contiguous()

    threshs = _params()[0].reshape(1, 1, -1).float()
    is_silent = (ref * (ref > threshs * 1e2)).sum(dim=2) < 1e7
    assert 0.5 < is_silent.float().mean().item() < 1.0

    _show("silent", _compare(ref, deg, seed=13))


@pytest.mark.parametrize("factor", [1e-6, 1e6])
def test_ratio_clamp_branches(factor):
    """A globally quieter or louder ``deg``, driving the two ratios onto a clamp.

    ``deg = 1e-6 ref`` pins ``band_pow_ratio`` to its lower clamp of ``0.01``
    and ``deg = 1e6 ref`` pins ``frame_pow_ratio`` to its lower clamp of
    ``3e-4``; neither is reached by the random sweeps above, and both gate a
    branch of the backward pass.
    """

    ref, _ = _spectra(3, 63, seed=101)
    deg = (ref * factor).contiguous()

    threshs, _, _, _ = _params()
    silent = (ref * (ref > threshs * 1e2)).sum(dim=2) < 1e7
    mask_ref = (ref > threshs * 100.0) * (~silent.unsqueeze(2))
    mask_deg = (deg > threshs * 100.0) * (~silent.unsqueeze(2))
    ratio = ((deg * mask_deg).mean(dim=1) + 1000) / (
        (ref * mask_ref).mean(dim=1) + 1000
    )

    equ_ref = ratio.unsqueeze(1).clamp(0.01, 100.0) * ref
    taer = (equ_ref * (equ_ref > threshs)).sum(dim=2)
    fpr = (taer + 5e3) / ((deg * (deg > threshs)).sum(dim=2) + 5e3)
    fpr = torch.cat([fpr[:, :1], fpr[:, 1:] * 0.8 + fpr[:, :-1] * 0.2], dim=1)

    if factor < 1.0:
        assert (ratio < 0.01).any(), "the band_pow_ratio lower clamp did not engage"
    else:
        assert (fpr < 3e-4).any(), "the frame_pow_ratio lower clamp did not engage"

    _show(f"factor={factor:g}", _compare(ref, deg, seed=2))


def test_identical_inputs():
    """``ref == deg`` drives both norms onto their ``1e-20`` floor without NaN."""

    ref, _ = _spectra(3, 62, seed=17)
    deg = ref.clone()

    module = _module()
    ref_a, deg_a = ref.clone().requires_grad_(True), deg.clone().requires_grad_(True)
    symm, asymm = module(ref_a, deg_a)

    # symm = asymm = 1e-20 / h, far below anything the clamp at 45 could hit
    assert (symm < 1e-15).all() and (asymm < 1e-15).all()
    assert torch.isfinite(symm).all() and torch.isfinite(asymm).all()

    grads = torch.autograd.grad(symm.sum() + asymm.sum(), (ref_a, deg_a))
    for grad in grads:
        assert torch.isfinite(grad).all(), "identical inputs produced NaN gradients"

    _show("identical", _compare(ref, deg, seed=19))


def test_aliased_inputs():
    """Passing one and the same tensor as both arguments accumulates both paths."""

    ref, _ = _spectra(2, 62, seed=401)
    module = _module()

    both = ref.clone().requires_grad_(True)
    symm, asymm = module(both, both)
    (grad,) = torch.autograd.grad(symm.sum() + asymm.sum(), both)

    both_b = ref.clone().requires_grad_(True)
    symm_ref, asymm_ref = _oracle(both_b, both_b)
    (grad_ref,) = torch.autograd.grad(symm_ref.sum() + asymm_ref.sum(), both_b)

    assert torch.isfinite(grad).all()
    assert _errors(grad, grad_ref)[1] <= GRAD_RTOL


@pytest.mark.parametrize("which", [0, 1])
def test_single_output_grad(which):
    """Differentiating only ``symm`` or only ``asymm`` matches the oracle."""

    ref, deg = _spectra(2, 62, seed=109)
    module = _module()
    up = _upstream(2, 3)[0]

    ref_a, deg_a = ref.clone().requires_grad_(True), deg.clone().requires_grad_(True)
    grads = torch.autograd.grad(
        (module(ref_a, deg_a)[which] * up).sum(), (ref_a, deg_a)
    )

    ref_b, deg_b = ref.clone().requires_grad_(True), deg.clone().requires_grad_(True)
    grads_ref = torch.autograd.grad(
        (_oracle(ref_b, deg_b)[which] * up.double()).sum(), (ref_b, deg_b)
    )

    for mine, oracle in zip(grads, grads_ref):
        assert torch.isfinite(mine).all()
        assert _errors(mine, oracle)[1] <= GRAD_RTOL


def test_exact_zeros_and_extremes():
    """Exact zeros, denormal sized and very large band powers stay finite."""

    ref, deg = _spectra(2, 62, seed=23)

    ref = ref.clone()
    deg = deg.clone()
    ref[:, ::3, :] = 0.0
    ref[:, :, ::7] = 0.0
    deg[:, ::4, :] = 0.0
    deg[:, 5, :] = 0.0

    _show("zeros", _compare(ref.contiguous(), deg.contiguous(), seed=29))

    zero = torch.zeros(2, 62, NBARK, device="cuda")
    _show("all-zero", _compare(zero, zero, seed=31))

    ref, deg = _spectra(2, 62, seed=211)
    _show("ref=0", _compare(torch.zeros_like(ref), deg, seed=2))
    _show("deg=0", _compare(ref, torch.zeros_like(deg), seed=2))

    ref, deg = _spectra(2, 62, seed=37)
    # 1e-12 lands every band far below the hearing threshold, 1e6 far above it
    for factor in (1e-12, 1e6, 1e9):
        _show(
            f"factor={factor:g}",
            _compare((ref * factor).contiguous(), (deg * factor).contiguous(), seed=41),
        )


@pytest.mark.parametrize(
    "u, forward_rtol, grad_rtol", [(1e-2, 1e-5, 5e-5), (1e-3, 1.2e-4, 6e-4)]
)
def test_near_threshold_bands(u, forward_rtol, grad_rtol):
    """Every band sitting a relative ``u`` above its hearing threshold.

    ``loudness = sl (2 th)^e ((0.5 + 0.5 x / th)^e - 1)`` has its base going to
    one as ``x`` approaches the threshold, so spelling it ``pow(base, e) - 1``
    throws away the whole mantissa.  Measured on exactly these inputs, with the
    two spellings of :func:`torch_pesq.triton_ops.chain._loudness`:

        u       pow(base, e) - 1        expm1(e log1p(base - 1))
        1e-2    3.2e-05 / 1.2e-04       3.4e-06 / 2.4e-05
        1e-3    3.6e-04 / 1.2e-03       6.0e-05 / 3.7e-04
        1e-4    2.2e-03 / 1.7e-02       2.8e-04 / 3.1e-03
                (forward / gradient, relative to the largest oracle magnitude)

    The budgets sit between the two columns, so this fails if anyone puts the
    plain power back.  What is left is not the formula any more but the float32
    rounding of ``equ_ref = band_pow_ratio * ref``: at ``u = 1e-3`` that product
    only pins ``x - th`` to three digits, which is why the budget has to grow as
    ``u`` shrinks.
    """

    threshs = _params()[0].reshape(1, 1, -1).float()

    gen = torch.Generator(device="cuda").manual_seed(7)
    shape = (3, 63, NBARK)
    ref = threshs * (1.0 + u * (0.5 + torch.rand(shape, device="cuda", generator=gen)))
    deg = threshs * (1.0 + u * (0.5 + torch.rand(shape, device="cuda", generator=gen)))

    errs = _compare(
        ref.contiguous(),
        deg.contiguous(),
        seed=1,
        forward_rtol=forward_rtol,
        grad_rtol=grad_rtol,
    )
    _show(f"near-threshold u={u:g}", errs)


def test_conditioning_matches_float32():
    """Ill conditioned spectra: no worse than float32 PyTorch on the same formulas.

    For ``spread <= 0.02`` the deadzone subtraction cancels away most of the
    mantissa and any absolute tolerance would just be a measurement of the
    input.  What is testable is that the kernels do not lose *more* than the
    same arithmetic in float32 does, so the bound here is ``8x`` the error of
    :func:`_chain_float32` plus the well conditioned budget.  A kernel bug shows
    up as a factor far beyond that, a conditioning problem does not.
    """

    for spread in (0.02, 0.005, 0.001):
        ref, deg = _spectra(4, 63, seed=5, spread=spread)
        mine = _run_triton(ref, deg, 3)
        f32 = _run_float32(ref, deg, 3)
        f64 = _run_oracle(ref, deg, 3)

        line = []
        for name, a, b, c in zip(("symm", "asymm", "gref", "gdeg"), mine, f32, f64):
            triton_err = _errors(a, c)[1]
            torch_err = _errors(b, c)[1]
            budget = 8.0 * torch_err + GRAD_RTOL
            assert triton_err <= budget, (
                f"spread={spread} {name}: triton {triton_err:.3e} vs "
                f"float32 pytorch {torch_err:.3e}"
            )
            line.append(f"{name}=(T {triton_err:.2e}, P {torch_err:.2e})")
        print(f"\nspread={spread} " + " ".join(line))


def test_random_fuzz():
    """Randomised sweep over shapes, spreads and levels, at the standard budget.

    This is the measurement the module tolerances are taken from, so it runs
    the same 60 points that are quoted at the top of this file.
    """

    import random

    rng = random.Random(0)
    worst = {"symm": 0.0, "asymm": 0.0, "gref": 0.0, "gdeg": 0.0}
    for i in range(60):
        batch = rng.choice([1, 1, 2, 3, 5, 8])
        nframe = rng.randint(20, 260)
        spread = rng.choice([0.05, 0.2, 0.5, 1.0, 2.0, 4.0])
        level = rng.choice([1.0, 3.0, 6.0, 9.0])

        ref, deg = _spectra(batch, nframe, seed=i * 977 + 3, spread=spread, level=level)
        errs = _compare(ref, deg, seed=i)
        for name, value in errs.items():
            worst[name] = max(worst[name], value[1])

    print("\nfuzz worst " + " ".join(f"{k}={v:.3e}" for k, v in worst.items()))


# ---------------------------------------------------------------------------
# structural properties: determinism, independence, memory hygiene
# ---------------------------------------------------------------------------


def test_determinism():
    """Identical inputs give bitwise identical outputs and gradients."""

    ref, deg = _spectra(4, 97, seed=43)

    first, second = _run_triton(ref, deg, 47), _run_triton(ref, deg, 47)
    for a, b in zip(first, second):
        assert torch.equal(a, b), "the chain is not run-to-run deterministic"


def test_batch_independence():
    """Every row of a batched call equals the same row computed on its own.

    Catches a batch offset that is off by one or a reduction that leaks across
    the batch: those survive a comparison against the oracle whenever the
    tolerance is taken over the whole tensor.
    """

    ref, deg = _spectra(8, 63, seed=999)
    module = _module()
    up_s, up_a = _upstream(8, 5)

    ref_a, deg_a = ref.clone().requires_grad_(True), deg.clone().requires_grad_(True)
    symm, asymm = module(ref_a, deg_a)
    grads = torch.autograd.grad(
        (symm * up_s).sum() + (asymm * up_a).sum(), (ref_a, deg_a)
    )

    for b in range(8):
        ref_b = ref[b : b + 1].clone().requires_grad_(True)
        deg_b = deg[b : b + 1].clone().requires_grad_(True)
        symm_b, asymm_b = module(ref_b, deg_b)
        grads_b = torch.autograd.grad(
            (symm_b * up_s[b : b + 1]).sum() + (asymm_b * up_a[b : b + 1]).sum(),
            (ref_b, deg_b),
        )

        assert torch.equal(symm[b : b + 1], symm_b)
        assert torch.equal(asymm[b : b + 1], asymm_b)
        assert torch.equal(grads[0][b : b + 1], grads_b[0])
        assert torch.equal(grads[1][b : b + 1], grads_b[1])


@pytest.mark.parametrize("poison", [float("nan"), -1e30])
def test_no_uninitialised_reads(poison):
    """Every scratch buffer is fully written before it is read.

    All temporaries come from ``torch.empty``; filling the caching allocator
    with NaN first means any tile the kernels forget to store, or read outside
    their mask, propagates into the result.
    """

    ref, deg = _spectra(3, 63, seed=63)
    clean = [t.clone() for t in _run_triton(ref, deg, 63)]

    blocks = []
    for numel in (1, 8, 64, 512, 4096, 32768, 262144, 1048576):
        blocks.extend(torch.full((numel,), poison, device="cuda") for _ in range(8))
    del blocks

    dirty = _run_triton(ref, deg, 63)
    for a, b in zip(clean, dirty):
        assert torch.isfinite(b).all(), "a scratch buffer is read before it is written"
        assert torch.equal(a, b)


def test_non_contiguous_input():
    """Non-contiguous views give the same result as their contiguous copies."""

    ref, deg = _spectra(2, 62, seed=53)

    wide_ref = torch.zeros(2, 62, 2 * NBARK, device="cuda")
    wide_deg = torch.zeros(2, 62, 2 * NBARK, device="cuda")
    wide_ref[:, :, ::2] = ref
    wide_deg[:, :, ::2] = deg
    wide_ref.requires_grad_(True)
    wide_deg.requires_grad_(True)

    view_ref, view_deg = wide_ref[:, :, ::2], wide_deg[:, :, ::2]
    assert not view_ref.is_contiguous()

    module = _module()
    symm_a, asymm_a = module(view_ref, view_deg)
    grads = torch.autograd.grad(symm_a.sum() + asymm_a.sum(), (wide_ref, wide_deg))

    ref_a, deg_a = ref.clone().requires_grad_(True), deg.clone().requires_grad_(True)
    symm_b, asymm_b = module(ref_a, deg_a)
    grads_b = torch.autograd.grad(symm_b.sum() + asymm_b.sum(), (ref_a, deg_a))

    assert torch.equal(symm_a, symm_b) and torch.equal(asymm_a, asymm_b)
    for wide, dense in zip(grads, grads_b):
        assert torch.equal(wide[:, :, ::2], dense)
        assert (wide[:, :, 1::2] == 0).all()


def test_non_contiguous_frames_match_oracle():
    """A view that is strided over frames still matches the oracle end to end."""

    ref, deg = _spectra(2, 125, seed=57)
    view_ref, view_deg = ref[:, ::2], deg[:, ::2]
    assert not view_ref.is_contiguous() and view_ref.shape[1] == 63

    _show("strided-frames", _compare(view_ref, view_deg, seed=57))


# ---------------------------------------------------------------------------
# configuration, call patterns and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nband", [17, 65])
def test_other_band_counts(nband):
    """Band counts that are not 49, i.e. a different ``BLOCK_K`` and mask.

    ``BLOCK_K`` is the next power of two, so 17 leaves 15 masked lanes in a
    32 wide tile and 65 leaves 63 in a 128 wide one.  A missing ``other=0.0``
    or a dropped ``kmask`` shows up here and nowhere in the 49 band sweep.
    """

    gen = torch.Generator().manual_seed(nband)
    threshs = (10 ** (torch.rand(nband, generator=gen) * 8 - 1)).double().cuda()
    exps = (0.15 + 0.15 * torch.rand(nband, generator=gen)).double().cuda()
    widths = (0.15 + 0.45 * torch.rand(nband, generator=gen)).double().cuda()
    total = widths[1:].sum()

    module = TritonPesqChain(threshs, exps, widths, total, SL).cuda()

    gen = torch.Generator(device="cuda").manual_seed(nband + 7)
    shape = (2, 63, nband)
    level = 10 ** (torch.rand((2, 63, 1), device="cuda", generator=gen) * 6)
    tilt = 0.2 + 2.0 * torch.rand(shape, device="cuda", generator=gen)
    ref = (threshs.reshape(1, 1, -1).float() * level * tilt).float().contiguous()
    pert = 10 ** (torch.rand(shape, device="cuda", generator=gen) - 0.5)
    deg = (ref * pert).float().contiguous()

    up_s, up_a = _upstream(2, nband)

    ref_a, deg_a = ref.clone().requires_grad_(True), deg.clone().requires_grad_(True)
    symm, asymm = module(ref_a, deg_a)
    grads = torch.autograd.grad(
        (symm * up_s).sum() + (asymm * up_a).sum(), (ref_a, deg_a)
    )

    ref_b, deg_b = ref.clone().requires_grad_(True), deg.clone().requires_grad_(True)
    symm_ref, asymm_ref = ref_chain(
        ref_b.double(), deg_b.double(), threshs, exps, widths, total, SL
    )
    grads_ref = torch.autograd.grad(
        (symm_ref * up_s.double()).sum() + (asymm_ref * up_a.double()).sum(),
        (ref_b, deg_b),
    )

    names = ("symm", "asymm", "gref", "gdeg")
    errs = {
        n: _errors(m, o)
        for n, m, o in zip(
            names, (symm, asymm) + grads, (symm_ref, asymm_ref) + grads_ref
        )
    }
    _show(f"nband={nband}", errs)

    for name in ("symm", "asymm"):
        assert errs[name][1] <= FORWARD_RTOL, f"{name}: {errs[name]}"
    for name in ("gref", "gdeg"):
        assert errs[name][1] <= GRAD_RTOL, f"{name}: {errs[name]}"


def test_exact_threshold_values():
    """Band powers exactly at the hearing threshold take the ``x > th`` branch off.

    ``ref == threshs`` makes every loudness exactly zero, so both norms end up
    on their ``1e-20`` floor and the clamp blocks the gradient of everything
    downstream of it, without a ``0/0`` anywhere.
    """

    module = _module()
    threshs = module.threshs.reshape(1, 1, -1)

    for tag, scale_ref, scale_deg in (("th/th", 1.0, 1.0), ("th/100th", 1.0, 100.0)):
        ref = (threshs * scale_ref).expand(2, 62, NBARK).contiguous()
        deg = (threshs * scale_deg).expand(2, 62, NBARK).contiguous()

        ref_a, deg_a = ref.clone().requires_grad_(True), deg.clone().requires_grad_(
            True
        )
        symm, asymm = module(ref_a, deg_a)
        grads = torch.autograd.grad(symm.sum() + asymm.sum(), (ref_a, deg_a))

        # h = ((0 + 1e5) / 1e7) ** 0.04 = 0.83175, so the floor lands at 1.2e-20
        assert torch.isfinite(symm).all() and torch.isfinite(asymm).all()
        assert (symm < 1e-19).all() and (asymm < 1e-19).all(), tag
        for grad in grads:
            assert torch.isfinite(grad).all(), tag


def test_no_grad_and_partial_grad():
    """Works under ``torch.no_grad`` and when only one input needs a gradient."""

    ref, deg = _spectra(2, 62, seed=59)
    module = _module()

    with torch.no_grad():
        symm, asymm = module(ref, deg)
    assert not symm.requires_grad and torch.isfinite(symm).all()

    deg_a = deg.clone().requires_grad_(True)
    symm, asymm = module(ref, deg_a)
    (grad,) = torch.autograd.grad(symm.sum() + asymm.sum(), deg_a)

    deg_b = deg.clone().requires_grad_(True)
    symm_ref, asymm_ref = _oracle(ref, deg_b)
    (grad_ref,) = torch.autograd.grad(symm_ref.sum() + asymm_ref.sum(), deg_b)

    assert _errors(grad, grad_ref)[1] <= GRAD_RTOL


def test_frames_outside_windows():
    """Frames past the last full window only feed the mean band powers."""

    nframe = 29  # a single window, frames 20..28 are dropped by the unfold
    ref, deg = _spectra(2, nframe, seed=61)

    module = _module()
    ref_a, deg_a = ref.clone().requires_grad_(True), deg.clone().requires_grad_(True)
    ref_b, deg_b = ref.clone().requires_grad_(True), deg.clone().requires_grad_(True)

    symm, asymm = module(ref_a, deg_a)
    symm_ref, asymm_ref = _oracle(ref_b, deg_b)

    grads = torch.autograd.grad(symm.sum() + asymm.sum(), (ref_a, deg_a))
    grads_ref = torch.autograd.grad(symm_ref.sum() + asymm_ref.sum(), (ref_b, deg_b))

    scale = max(g.abs().max().item() for g in grads_ref)
    for mine, oracle in zip(grads, grads_ref):
        # the tail is only reachable through the mean band powers, which is
        # exactly what the oracle does as well
        assert (mine[:, 20:] - oracle[:, 20:]).abs().max().item() <= GRAD_RTOL * scale
        assert (mine - oracle).abs().max().item() <= GRAD_RTOL * scale
        assert mine[:, 20:].abs().max().item() > 0.0


def test_empty_batch():
    """A zero sized batch is a no-op rather than a launch failure."""

    module = _module()
    empty = torch.zeros(0, 62, NBARK, device="cuda")
    symm, asymm = module(empty, empty)

    assert symm.shape == (0,) and asymm.shape == (0,)


def test_too_few_frames_raises():
    """Fewer than 20 frames cannot form a single overlapping sum."""

    ref, deg = _spectra(1, 20, seed=71)
    module = _module()

    with pytest.raises(ValueError, match="at least 20 frames"):
        module(ref[:, :19], deg[:, :19])


def test_input_validation():
    """Shape, dtype and device mistakes are rejected with a clear message."""

    ref, deg = _spectra(1, 62, seed=73)
    module = _module()

    with pytest.raises(RuntimeError, match="float32"):
        module(ref.double(), deg.double())
    with pytest.raises(RuntimeError, match="float32"):
        module(ref.half(), deg.half())
    with pytest.raises(RuntimeError, match="GPU"):
        module(ref.cpu(), deg.cpu())
    with pytest.raises(ValueError, match="same shape"):
        module(ref, deg[:, :40])
    with pytest.raises(ValueError, match="batch, frame, bark"):
        module(ref[0], deg[0])
    with pytest.raises(ValueError, match="49 Bark bands"):
        module(ref[:, :, :30].contiguous(), deg[:, :, :30].contiguous())


def test_module_construction_validation():
    """Inconsistent or non positive band parameters are rejected."""

    threshs, exps, widths, total = _params()

    with pytest.raises(ValueError, match="same number of bands"):
        TritonPesqChain(threshs, exps[..., :10], widths, total, SL)
    with pytest.raises(ValueError, match="strictly positive"):
        TritonPesqChain(torch.zeros_like(threshs), exps, widths, total, SL)


def test_benchmark():
    """Report the wall clock of the kernels against the PyTorch oracle."""

    ref, deg = _spectra(8, 200, seed=79)
    module = _module()

    def timed(fn, backward):
        for _ in range(5):
            fn(backward)
        torch.cuda.synchronize()
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            fn(backward)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        samples.sort()
        return samples[len(samples) // 2] * 1e3

    def mine(backward):
        ref_a, deg_a = ref.clone().requires_grad_(True), deg.clone().requires_grad_(
            True
        )
        symm, asymm = module(ref_a, deg_a)
        if backward:
            torch.autograd.grad(symm.sum() + asymm.sum(), (ref_a, deg_a))

    def oracle(backward):
        ref_a, deg_a = ref.clone().requires_grad_(True), deg.clone().requires_grad_(
            True
        )
        symm, asymm = _oracle(ref_a, deg_a)
        if backward:
            torch.autograd.grad(symm.sum() + asymm.sum(), (ref_a, deg_a))

    print(
        f"\n[8, 200, 49] forward         triton {timed(mine, False):7.3f} ms"
        f"   oracle {timed(oracle, False):7.3f} ms"
    )
    print(
        f"[8, 200, 49] forward+backward triton {timed(mine, True):7.3f} ms"
        f"   oracle {timed(oracle, True):7.3f} ms"
    )
