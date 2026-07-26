"""Triton GPU backend for the PESQ loss.

The submodules implement forward and backward passes of every pipeline stage as
Triton kernels. :mod:`torch_pesq.triton_ops.reference` holds the pure PyTorch
oracle the kernels are validated against.
"""

from ._common import HAS_TRITON

__all__ = ["HAS_TRITON"]
