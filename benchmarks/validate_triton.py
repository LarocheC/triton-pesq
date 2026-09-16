"""Validate the Triton backend against PyTorch and the ITU reference.

Sweeps a range of degradation levels and compares the MOS estimates of

  * :class:`torch_pesq.PesqLoss` (PyTorch),
  * :class:`torch_pesq.triton_ops.loss.PesqLossTriton` (Triton),
  * the ITU-T P.862 reference implementation from the ``pesq`` package,
    if it is installed.

The two backends do *not* agree to float32 precision, and the Triton one is not
the reason: ``torchaudio.functional.lfilter`` evaluates the level alignment
filter as a float32 direct-form-I recursion and carries ~1e-2 relative error of
its own. :func:`accuracy_against_float64` is therefore the comparison that
means something -- it scores both backends against a float64 evaluation of the
same pipeline, where the Triton one lands two to three orders of magnitude
closer. Agreement with the ITU reference is limited by the approximations of
the torch-pesq model itself (no time alignment, IIR level alignment) and is only
reported for context.

Usage
-----
    python benchmarks/validate_triton.py [--samples 32] [--seed 0]
"""

import argparse

import torch

from torch_pesq import PesqLoss
from torch_pesq.triton_ops.loss import PesqLossTriton


def speech_like(batch: int, samples: int, seed: int, device: str = "cuda"):
    """Harmonic signal with formant like shaping and a slow envelope."""

    generator = torch.Generator(device=device).manual_seed(seed)
    time_axis = torch.arange(samples, device=device) / 16000.0

    signal = torch.zeros(batch, samples, device=device)
    for _ in range(3):
        pitch = 90.0 + 120.0 * torch.rand(batch, 1, device=device, generator=generator)
        envelope = 0.5 + 0.5 * torch.sin(
            2
            * torch.pi
            * (2.0 + 4.0 * torch.rand(batch, 1, device=device, generator=generator))
            * time_axis
        )
        for harmonic in range(1, 25):
            gain = 1.0 / (1.0 + (harmonic * pitch / 900.0) ** 2)
            phase = torch.rand(batch, 1, device=device, generator=generator) * 6.28
            signal += (
                gain
                * envelope
                * torch.sin(2 * torch.pi * pitch * harmonic * time_axis + phase)
            )

    return (signal / signal.abs().amax(dim=1, keepdim=True) * 0.7).contiguous()


def accuracy_against_float64(ref, deg, torch_loss, triton_loss):
    """Compare both float32 backends against a float64 evaluation.

    The PyTorch backend inherits the error of ``torchaudio.functional.lfilter``,
    whose float32 direct-form-I recursion is inaccurate for the order 10 band
    pass of the level alignment. The Triton backend evaluates the same filter as
    a chunked scan in a modal state basis and is much closer to the truth, so
    this comparison is the fair way to judge the deviation between the two.
    """

    import copy

    from torch_pesq.triton_ops.reference import ref_pipeline

    double_model = copy.deepcopy(torch_loss).double()
    with torch.no_grad():
        exact = ref_pipeline(ref.double(), deg.double(), double_model)
        approx = torch_loss.raw(ref, deg)
        triton = triton_loss.raw(ref, deg)

    print()
    print("Deviation from a float64 evaluation of the same pipeline:")
    for index, name in enumerate(("d_symm", "d_asymm")):
        scale = exact[index].abs().max()
        torch_error = (
            (approx[index].double() - exact[index]).abs().max() / scale
        ).item()
        triton_error = (
            (triton[index].double() - exact[index]).abs().max() / scale
        ).item()

        # Heavily degraded signals push the asymmetric distance into its clamp at
        # 45, where every backend returns the same saturated value and the ratio
        # below is meaningless. Say so instead of printing a bogus factor.
        if max(torch_error, triton_error) < 1e-12:
            verdict = "both exact (saturated)"
        else:
            verdict = f"{torch_error / max(triton_error, 1e-30):.0f}x better"

        print(
            f"  {name:8s} pytorch {torch_error:.3e}   triton {triton_error:.3e}   "
            f"({verdict})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=32000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This script needs a CUDA GPU.")

    torch_loss = PesqLoss(1.0, sample_rate=16000).cuda()
    triton_loss = PesqLossTriton(1.0, sample_rate=16000).cuda()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    clean = speech_like(1, args.length, args.seed)
    noise = torch.randn(
        args.samples, args.length, device="cuda", generator=generator
    ) * 0.5 + 0.5 * speech_like(args.samples, args.length, args.seed + 1)

    levels = torch.linspace(0.0, 0.7, args.samples, device="cuda").unsqueeze(1)
    reference = clean.expand(args.samples, -1).contiguous()
    degraded = ((1 - levels) * reference + levels * noise).contiguous()

    with torch.no_grad():
        mos_torch = torch_loss.mos(reference, degraded).double().cpu()
        mos_triton = triton_loss.mos(reference, degraded).double().cpu()

    delta = (mos_torch - mos_triton).abs()
    print(f"{'level':>7} {'pytorch':>9} {'triton':>9} {'|diff|':>9}", end="")

    mos_itu = None
    try:
        import numpy as np
        from pesq import pesq as pesq_reference

        mos_itu = torch.as_tensor(
            [
                pesq_reference(
                    16000,
                    np.asarray(reference[0].cpu()),
                    np.asarray(item.cpu()),
                    mode="wb",
                )
                for item in degraded
            ],
            dtype=torch.float64,
        )
        print(f" {'ITU P.862':>10}")
    except ImportError:
        print()

    for index in range(args.samples):
        line = (
            f"{levels[index, 0].item():7.3f} {mos_torch[index]:9.4f} "
            f"{mos_triton[index]:9.4f} {delta[index]:9.2e}"
        )
        if mos_itu is not None:
            line += f" {mos_itu[index]:10.4f}"
        print(line)

    print()
    print(f"max |pytorch - triton|      : {delta.max():.3e}")
    print(f"mean |pytorch - triton|     : {delta.mean():.3e}")

    accuracy_against_float64(reference, degraded, torch_loss, triton_loss)

    if mos_itu is not None:
        for name, values in (("pytorch", mos_torch), ("triton", mos_triton)):
            error = (values - mos_itu).abs()
            correlation = torch.corrcoef(torch.stack([values, mos_itu]))[0, 1]
            print(
                f"{name:>7} vs ITU: max {error.max():.3f}, "
                f"mean {error.mean():.3f}, pearson r {correlation:.4f}"
            )


if __name__ == "__main__":
    main()
