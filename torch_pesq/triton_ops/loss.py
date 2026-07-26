"""Triton accelerated PESQ loss."""

import torch

from typing import Tuple

from ..loss import PesqLoss
from ._common import require_triton
from .chain import TritonPesqChain
from .elementwise import align_scale, edge_ramp, peak_normalize, pesq_epilogue
from .iir import TritonIIR
from .resample import TritonResample
from .spectral import TritonStftBark

__all__ = ["PesqLossTriton"]


class PesqLossTriton(PesqLoss):
    """Perceptual Evaluation of Speech Quality, evaluated with Triton kernels.

    Drop-in replacement for :class:`torch_pesq.loss.PesqLoss` in which every
    stage of the pipeline, forward and backward, is a Triton kernel. The module
    derives from the PyTorch implementation so that both share exactly the same
    filter coefficients, filterbank and loudness parametrization; only
    :meth:`raw`, :meth:`mos` and :meth:`forward` are overridden.

    The scores differ from the PyTorch implementation only by floating point
    round off: the reference accidentally evaluates the perceptual model in
    double precision, the Triton backend uses single precision throughout.

    Parameters
    ----------
    factor : float
        Scaling of the loss function
    sample_rate : int
        Sampling rate of the time signal, re-samples if different from 16kHz
    nbarks : int
        Number of bark bands
    win_length : int
        Window size used in the STFT
    n_fft : int
        Number of frequency bins
    hop_length : int
        Distance between different frames

    Attributes
    ----------
    resample_op : TritonResample or None
        Resampling to 16kHz, ``None`` when the input is already at 16kHz
    power_iir : TritonIIR
        Bandpass filter used for the level alignment
    pre_iir : TritonIIR
        Pre-emphasize filter
    spectral : TritonStftBark
        Fused spectrogram and Bark filterbank
    chain : TritonPesqChain
        Perceptual model, disturbance and overlapping sums
    """

    def __init__(
        self,
        factor: float,
        sample_rate: int = 48000,
        nbarks: int = 49,
        win_length: int = 512,
        n_fft: int = 512,
        hop_length: int = 256,
    ):
        require_triton()

        super(PesqLossTriton, self).__init__(
            factor, sample_rate, nbarks, win_length, n_fft, hop_length
        )

        self.resample_op = (
            None if sample_rate == 16000 else TritonResample(sample_rate, 16000)
        )
        self.power_iir = TritonIIR(self.power_filter[0], self.power_filter[1])
        self.pre_iir = TritonIIR(self.pre_filter[0], self.pre_filter[1])
        self.spectral = TritonStftBark(
            self.to_spec.window,
            self.fbank.fbank,
            self.fbank.pow_dens_correction,
            n_fft,
            hop_length,
        )
        self.chain = TritonPesqChain(
            self.loudness.threshs.flatten(),
            self.loudness.exp.flatten(),
            self.fbank.width_bark,
            float(self.fbank.total_width),
        )

    def align_level(self, signal: torch.Tensor) -> torch.Tensor:
        """Align power to 10**7 for band 325 to 3.25kHz, see :meth:`PesqLoss.align_level`."""

        return align_scale(signal, self.power_iir(signal))

    def preemphasize(self, signal: torch.Tensor) -> torch.Tensor:
        """Pre-emphasize a signal, see :meth:`PesqLoss.preemphasize`."""

        return self.pre_iir(edge_ramp(signal))

    def raw(
        self, ref: torch.Tensor, deg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate symmetric and asymmetric distances.

        Parameters
        ----------
        ref : torch.Tensor
            Reference signal with shape ``[batch, sample]``
        deg : torch.Tensor
            Degraded signal with shape ``[batch, sample]``

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Symmetric and asymmetric distance, both with shape ``[batch]``
        """

        deg, ref = torch.atleast_2d(deg), torch.atleast_2d(ref)

        if ref.shape != deg.shape:
            raise ValueError(
                f"Reference and degraded signal must have the same shape, "
                f"got {tuple(ref.shape)} and {tuple(deg.shape)}."
            )

        # Both signals share a single scaling factor and go through identical
        # stages, so they are stacked into one batch of twice the size. That
        # halves the number of kernel launches up to the Bark spectrogram.
        signal = peak_normalize(ref, deg, stack=True)

        if self.resample_op is not None:
            signal = self.resample_op(signal)

        signal = self.align_level(signal)
        signal = self.preemphasize(signal)

        # the reference implementation pads with `length % 256` zeros
        n_padded = signal.shape[1] + signal.shape[1] % 256
        bark = self.spectral(signal, n_padded)

        batch = bark.shape[0] // 2

        return self.chain(bark[:batch], bark[batch:])

    def graphed(self, ref: torch.Tensor, deg: torch.Tensor):
        """Capture the loss in a CUDA graph, specialised to one input shape.

        The pipeline issues roughly forty small kernels, so at these tensor
        sizes the host side launch cost is larger than the device work. Replaying
        a captured graph removes that cost entirely and roughly halves the
        latency of a single call. Throughput in a loop that already overlaps
        dispatch with execution benefits much less -- measure before adopting it.

        The captured callable is bit exact with respect to the eager one, for
        both the loss and the gradients.

        Parameters
        ----------
        ref : torch.Tensor
            Example reference signal, fixes the shape the graph is valid for
        deg : torch.Tensor
            Example degraded signal, same shape

        Returns
        -------
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            Replacement for ``self.__call__`` valid for this shape only

        Notes
        -----
        The example tensors must have the same ``requires_grad`` flags as the
        ones used later. Warm the module up once before capturing so that no
        Triton autotuning happens during the capture.
        """

        require_triton()

        # capture from detached copies, a live autograd graph on the example
        # tensors makes the capture depend on the legacy stream and fail
        sample = tuple(
            tensor.detach().clone().requires_grad_(tensor.requires_grad)
            for tensor in (ref, deg)
        )

        # trigger Triton compilation and autotuning before the capture starts
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            output = self(*sample)
            if output.requires_grad:
                torch.autograd.grad(
                    output.sum(), [t for t in sample if t.requires_grad]
                )
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        return torch.cuda.make_graphed_callables(self, sample)

    def mos(self, ref: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Calculate Mean Opinion Score, see :meth:`PesqLoss.mos`."""

        d_symm, d_asymm = self.raw(ref, deg)

        return pesq_epilogue(d_symm, d_asymm, self.factor)[0]

    def forward(self, ref: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Calculate a loss variant of the MOS score, see :meth:`PesqLoss.forward`."""

        d_symm, d_asymm = self.raw(ref, deg)

        return pesq_epilogue(d_symm, d_asymm, self.factor)[1]
