"""Triton GPU backend for the PESQ loss.

The submodules implement forward and backward passes of every pipeline stage as
Triton kernels. :mod:`torch_pesq.triton_ops.reference` holds the pure PyTorch
oracle the kernels are validated against.

This package is an execution backend only. The PESQ model it evaluates -- the
Bark filterbank, the loudness model, the disturbance processing and every
parameter table -- comes from torch-pesq by Lorenz Schmidt, Nils Werner and
Nils Peters (International Audio Laboratories Erlangen), MIT licensed, see
https://github.com/audiolabs/torch-pesq. Nothing here changes its behaviour.
"""

from ._common import HAS_TRITON

__all__ = ["HAS_TRITON"]
