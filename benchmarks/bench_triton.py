"""Benchmark the Triton backend against the PyTorch implementation.

Usage
-----
    python benchmarks/bench_triton.py            # full run
    python benchmarks/bench_triton.py --quick    # a single shape
    python benchmarks/bench_triton.py --json out.json

Every measurement is a synchronised wall clock time, after a warmup phase, taken
as the median of several repetitions. Four backends are compared:

``torch``
    :class:`torch_pesq.PesqLoss` exactly as shipped. Note that its Bark tables
    are float64, so the perceptual model runs in double precision.
``torch-fp32``
    the same implementation with all constants cast to float32. It is measured
    to show that the double precision is not what makes it slow, and left out of
    the summary table.
``triton``
    :class:`torch_pesq.triton_ops.loss.PesqLossTriton`.
``triton+graph``
    the same module replayed from a CUDA graph. The Triton pipeline is bound by
    the host side dispatch of its ~40 small kernels rather than by the GPU, so
    removing that dispatch is worth more than any further kernel tuning.
"""

import argparse
import json
import statistics
import time

import torch

from torch_pesq import PesqLoss
from torch_pesq.triton_ops.loss import PesqLossTriton

SHAPES = [
    (1, 16000),
    (4, 16000),
    (8, 16000),
    (32, 16000),
    (8, 64000),
    (4, 160000),
]


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


def timeit(fn, repeats: int = 20, warmup: int = 10) -> float:
    """Median wall clock time of ``fn`` in milliseconds, host time included.

    CUDA events are deliberately *not* used here. The Triton pipeline issues
    around forty small kernels whose host side dispatch costs more than the
    device work they enqueue, so the CPU is the bottleneck and an event pair
    only measures the device part, reporting a number several times too good.
    """

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    return statistics.median(samples)


def make_signals(batch: int, samples: int, device: str = "cuda"):
    """Deterministic speech like test signals."""

    generator = torch.Generator(device=device).manual_seed(1234)

    time_axis = torch.arange(samples, device=device) / 16000.0
    envelope = 0.5 + 0.5 * torch.sin(2 * torch.pi * 3.0 * time_axis)

    ref = torch.zeros(batch, samples, device=device)
    for harmonic in range(1, 12):
        phase = torch.rand(batch, 1, device=device, generator=generator) * 6.28
        ref += (
            torch.sin(2 * torch.pi * 120.0 * harmonic * time_axis + phase)
            / harmonic
            * envelope
        )
    ref = ref / ref.abs().amax(dim=1, keepdim=True)
    deg = ref + 0.2 * torch.randn(batch, samples, device=device, generator=generator)

    return ref.contiguous(), deg.contiguous()


def forward_backward(model, ref, deg):
    """One forward and backward pass, returns nothing."""

    ref = ref.detach().requires_grad_(True)
    deg = deg.detach().requires_grad_(True)
    torch.autograd.grad(model(ref, deg).sum(), [ref, deg])


def run(shapes, repeats: int, skip_slow: bool):
    """Benchmark all backends over ``shapes``."""

    results = []

    for batch, samples in shapes:
        ref, deg = make_signals(batch, samples)

        models = {
            "torch": PesqLoss(1.0, sample_rate=16000).cuda(),
            "torch-fp32": float32_reference(PesqLoss(1.0, sample_rate=16000).cuda()),
            "triton": PesqLossTriton(1.0, sample_rate=16000).cuda(),
        }

        row = {"batch": batch, "samples": samples}

        with torch.no_grad():
            reference = models["torch"](ref, deg).double()

        for name, model in models.items():
            with torch.no_grad():
                value = model(ref, deg).double()
            error = (value - reference).abs().max().item()
            relative = error / reference.abs().max().item()

            slow = skip_slow and name.startswith("torch") and batch * samples > 5e5
            forward = (
                float("nan")
                if slow
                else timeit(lambda: model(ref, deg), repeats=repeats)
            )
            both = (
                float("nan")
                if slow
                else timeit(
                    lambda: forward_backward(model, ref, deg),
                    repeats=max(3, repeats // 4),
                )
            )

            row[name] = {"forward": forward, "forward_backward": both, "rel": relative}

        # the same Triton pipeline replayed from a CUDA graph, which removes the
        # host side dispatch that dominates this workload
        graph_ref = ref.detach().clone().requires_grad_(True)
        graph_deg = deg.detach().clone().requires_grad_(True)
        graphed = models["triton"].graphed(graph_ref, graph_deg)

        def graph_step():
            output = graphed(graph_ref, graph_deg)
            torch.autograd.grad(output.sum(), [graph_ref, graph_deg])

        row["triton-graph"] = {
            "forward": float("nan"),
            "forward_backward": timeit(graph_step, repeats=repeats),
            "rel": row["triton"]["rel"],
        }

        results.append(row)

        print(
            f"  batch={batch:3d} samples={samples:7d}  "
            + "  ".join(
                f"{name}: {row[name]['forward']:8.3f} / {row[name]['forward_backward']:8.3f} ms"
                for name in list(models) + ["triton-graph"]
            )
        )

    return results


def print_table(results):
    """Print the results as a markdown table."""

    def speedup(row, key, mode):
        base, triton = row[key][mode], row["triton"][mode]
        return "-" if base != base else f"{base / triton:.0f}x"

    print()
    print(
        "| batch | samples | audio | torch | triton | speedup | triton+graph | speedup |"
    )
    print(
        "|------:|--------:|------:|------:|-------:|--------:|-------------:|--------:|"
    )
    for mode, label in (("forward", "forward"), ("forward_backward", "fwd+bwd")):
        print(f"| **{label}** | | | | | | | |")
        for row in results:
            seconds = row["samples"] / 16000.0 * row["batch"]
            graph = row["triton-graph"][mode]
            base = row["torch"][mode]
            graph_cell = "-" if graph != graph else f"{graph:.3f} ms"
            graph_speed = (
                "-" if graph != graph or base != base else f"{base / graph:.0f}x"
            )
            print(
                f"| {row['batch']} | {row['samples']} | {seconds:.1f}s "
                f"| {base:.2f} ms "
                f"| {row['triton'][mode]:.3f} ms "
                f"| {speedup(row, 'torch', mode)} "
                f"| {graph_cell} | {graph_speed} |"
            )

    print()
    print("Maximum relative deviation of the loss from the PyTorch reference:")
    for row in results:
        print(
            f"  batch={row['batch']:3d} samples={row['samples']:7d}: "
            f"{row['triton']['rel']:.2e}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="only run one shape")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="also time the PyTorch backend on the large shapes (very slow)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")

    shapes = [(8, 16000)] if args.quick else SHAPES
    started = time.time()
    results = run(shapes, args.repeats, skip_slow=not args.all_shapes)
    print_table(results)
    print(f"\ntotal benchmark time: {time.time() - started:.0f}s")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
