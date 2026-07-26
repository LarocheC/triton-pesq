"""Shared helpers for the Triton kernels."""

import torch

try:  # pragma: no cover - depends on the installation
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on the installation
    triton = None
    tl = None
    HAS_TRITON = False


__all__ = [
    "HAS_TRITON",
    "next_power_of_two",
    "require_triton",
    "check_input",
]


def next_power_of_two(value: int) -> int:
    """Smallest power of two which is greater or equal to ``value``."""

    return 1 << max(0, int(value) - 1).bit_length()


def require_triton() -> None:
    """Raise a descriptive error when Triton is unavailable."""

    if not HAS_TRITON:
        raise RuntimeError(
            "The Triton backend requires the `triton` package, install it with "
            "`pip install triton` or use the PyTorch backend instead."
        )


def check_input(tensor: torch.Tensor, name: str = "input") -> torch.Tensor:
    """Validate a kernel input and return a contiguous CUDA view of it."""

    require_triton()

    if not tensor.is_cuda:
        raise RuntimeError(f"The Triton backend needs `{name}` to live on a GPU.")
    if tensor.dtype != torch.float32:
        raise RuntimeError(
            f"The Triton backend only supports float32, `{name}` is {tensor.dtype}."
        )

    return tensor.contiguous()
