"""Pure PyTorch reference implementations of the PESQ pipeline stages.

The functions in this module are a faithful, stage-by-stage decomposition of
:meth:`torch_pesq.loss.PesqLoss.raw`. They serve two purposes:

  1. They define the exact contract that every Triton kernel has to fulfil,
     including the gradient behaviour implied by autograd.
  2. They are used as an oracle in the test suite; :func:`ref_pipeline` is
     verified to reproduce :meth:`PesqLoss.raw` bit for bit.

Nothing in here is meant to be fast -- it is the specification.
"""

import torch

from torch.nn.functional import unfold
from torchaudio.functional import lfilter

__all__ = [
    "ref_align_level",
    "ref_preemphasize",
    "ref_stft_bark",
    "ref_chain",
    "ref_pipeline",
]


def ref_align_level(signal: torch.Tensor, power_filter: torch.Tensor) -> torch.Tensor:
    """Align the power of the 325Hz-3.25kHz band to ``1e7``.

    Parameters
    ----------
    signal : torch.Tensor
        Time signal with shape ``[batch, sample]``
    power_filter : torch.Tensor
        Bandpass coefficients with shape ``[2, order + 1]``, ``[0]`` are the
        numerator and ``[1]`` the denominator coefficients

    Returns
    -------
    torch.Tensor
        Scaled time signal with shape ``[batch, sample]``
    """

    filtered = lfilter(signal, power_filter[1], power_filter[0], clamp=False)

    power = (
        (filtered**2).sum(dim=1, keepdim=True) / (filtered.shape[1] + 5120) / 1.04684
    )

    return signal * (10**7 / power).sqrt()


def ref_preemphasize(signal: torch.Tensor, pre_filter: torch.Tensor) -> torch.Tensor:
    """Ramp the signal edges and apply the pre-emphasize filter.

    Parameters
    ----------
    signal : torch.Tensor
        Time signal with shape ``[batch, sample]``
    pre_filter : torch.Tensor
        Filter coefficients with shape ``[2, order + 1]``

    Returns
    -------
    torch.Tensor
        Pre-emphasized signal with shape ``[batch, sample]``
    """

    emp = torch.linspace(0, 15, 16, device=signal.device)[1:] / 16.0

    signal = signal.clone()
    signal[:, :15] = signal[:, :15] * emp
    signal[:, -15:] = signal[:, -15:] * torch.flip(emp, dims=(0,))

    return lfilter(signal, pre_filter[1], pre_filter[0], clamp=False)


def ref_stft_bark(
    signal: torch.Tensor,
    window: torch.Tensor,
    fbank: torch.Tensor,
    pow_dens_correction: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 256,
) -> torch.Tensor:
    """Power spectrogram, DC removal and Bark band aggregation.

    Parameters
    ----------
    signal : torch.Tensor
        Time signal with shape ``[batch, sample]``, already zero padded
    window : torch.Tensor
        Analysis window with ``n_fft`` taps
    fbank : torch.Tensor
        Binary filterbank matrix with shape ``[bark, n_fft // 2]``
    pow_dens_correction : torch.Tensor
        Per band power density correction with shape ``[bark]``
    n_fft : int
        Number of frequency bins
    hop_length : int
        Distance between frames

    Returns
    -------
    torch.Tensor
        Bark scaled power spectrogram with shape ``[batch, frame, bark]``
    """

    spec = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window.shape[0],
        window=window,
        center=False,
        normalized=False,
        return_complex=True,
    )
    spec = spec.abs().pow(2).swapaxes(1, 2)

    # the energy feature is not used
    spec = torch.cat([torch.zeros_like(spec[:, :, :1]), spec[:, :, 1:]], dim=2)

    bark = torch.einsum("ij,klj->kli", fbank, spec[:, :, :-1])

    return bark * pow_dens_correction


def _total_audible(
    tensor: torch.Tensor, threshs: torch.Tensor, factor: float
) -> torch.Tensor:
    """Total audible energy per frame, see :meth:`Loudness.total_audible`."""

    return (tensor * (tensor > threshs * factor)).sum(dim=2)


def _loudness(
    pow_dens: torch.Tensor, threshs: torch.Tensor, exp: torch.Tensor, sl: float
) -> torch.Tensor:
    """Zwicker loudness of a Bark spectrogram, see :meth:`Loudness.forward`."""

    loudness = (2.0 * threshs) ** exp * ((0.5 + 0.5 * pow_dens / threshs) ** exp - 1)
    loudness = torch.where(pow_dens <= threshs, torch.zeros_like(loudness), loudness)

    return loudness * sl


def _weighted_norm(
    tensor: torch.Tensor, width_bark: torch.Tensor, total_width: float, p: float
) -> torch.Tensor:
    """Width weighted p-norm, see :meth:`BarkScale.weighted_norm`."""

    return total_width * (width_bark * tensor / total_width ** (1 / p))[:, :, 1:].norm(
        p, dim=2
    )


def ref_chain(
    ref: torch.Tensor,
    deg: torch.Tensor,
    threshs: torch.Tensor,
    exp: torch.Tensor,
    width_bark: torch.Tensor,
    total_width: float,
    sl: float,
):
    """Symmetric and asymmetric distance from two Bark spectrograms.

    This covers everything from the silent frame detection down to the
    overlapping sums, i.e. the whole second half of :meth:`PesqLoss.raw`.

    Parameters
    ----------
    ref : torch.Tensor
        Bark spectrogram of the reference with shape ``[batch, frame, bark]``
    deg : torch.Tensor
        Bark spectrogram of the degraded signal, same shape
    threshs : torch.Tensor
        Hearing thresholds with shape ``[bark]``
    exp : torch.Tensor
        Loudness exponents with shape ``[bark]``
    width_bark : torch.Tensor
        Band widths in Bark with shape ``[bark]``
    total_width : float
        Sum of all band widths, excluding the first band
    sl : float
        Loudness calibration constant

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Symmetric and asymmetric distance, both with shape ``[batch]``
    """

    silent = _total_audible(ref, threshs, 1e2) < 1e7

    mask_ref = (ref > threshs * 100.0) * (~silent.unsqueeze(2))
    mask_deg = (deg > threshs * 100.0) * (~silent.unsqueeze(2))

    mean_ref_pow = (ref * mask_ref).mean(dim=1)
    mean_deg_pow = (deg * mask_deg).mean(dim=1)

    band_pow_ratio = (
        ((mean_deg_pow + 1000) / (mean_ref_pow + 1000))
        .unsqueeze(1)
        .clamp(min=0.01, max=100.0)
    )
    equ_ref = band_pow_ratio * ref

    total_audible_ref = _total_audible(equ_ref, threshs, 1.0)
    frame_pow_ratio = (total_audible_ref + 5e3) / (
        _total_audible(deg, threshs, 1.0) + 5e3
    )

    frame_pow_ratio = torch.cat(
        [
            frame_pow_ratio[:, :1],
            frame_pow_ratio[:, 1:] * 0.8 + frame_pow_ratio[:, :-1] * 0.2,
        ],
        dim=1,
    ).clamp(min=3e-4, max=5.0)

    equ_deg = frame_pow_ratio.unsqueeze(2) * deg

    deg_loud = _loudness(equ_deg, threshs, exp, sl)
    ref_loud = _loudness(equ_ref, threshs, exp, sl)

    deadzone = 0.25 * torch.min(deg_loud, ref_loud)
    disturbance = deg_loud - ref_loud
    disturbance = disturbance.sign() * (disturbance.abs() - deadzone).clamp(min=0)

    symm_distu = _weighted_norm(disturbance, width_bark, total_width, 2.0)
    symm_distu = symm_distu.clamp(min=1e-20)

    asymm_scaling = ((equ_deg + 50.0) / (equ_ref + 50.0)) ** 1.2
    asymm_scaling = torch.where(
        asymm_scaling < 3.0, torch.zeros_like(asymm_scaling), asymm_scaling
    ).clamp(max=12.0)

    asymm_distu = _weighted_norm(
        disturbance * asymm_scaling, width_bark, total_width, 1.0
    )
    asymm_distu = asymm_distu.clamp(min=1e-20)

    h = ((total_audible_ref + 1e5) / 1e7) ** 0.04
    symm_distu = (symm_distu / h).clamp(max=45.0)
    asymm_distu = (asymm_distu / h).clamp(max=45.0)

    psqm = (unfold(symm_distu.unsqueeze(1).unsqueeze(1), (1, 20), stride=10) ** 6).mean(
        dim=1
    ) ** (1.0 / 6)
    d_symm = psqm.square().mean(dim=1).sqrt()

    psqm = (
        unfold(asymm_distu.unsqueeze(1).unsqueeze(1), (1, 20), stride=10) ** 6
    ).mean(dim=1) ** (1.0 / 6)
    d_asymm = psqm.square().mean(dim=1).sqrt()

    return d_symm, d_asymm


def ref_pipeline(ref: torch.Tensor, deg: torch.Tensor, model):
    """Full reference pipeline, composed from the stages above.

    Parameters
    ----------
    ref : torch.Tensor
        Reference signal with shape ``[batch, sample]``
    deg : torch.Tensor
        Degraded signal with shape ``[batch, sample]``
    model : PesqLoss
        Module holding the filter coefficients and filterbanks

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Symmetric and asymmetric distance, both with shape ``[batch]``
    """

    deg, ref = torch.atleast_2d(deg), torch.atleast_2d(ref)

    max_val = torch.max(
        torch.amax(deg.abs(), dim=1, keepdim=True),
        torch.amax(ref.abs(), dim=1, keepdim=True),
    )
    deg, ref = deg / max_val, ref / max_val
    deg, ref = model.resampler(deg), model.resampler(ref)

    ref = ref_align_level(ref, model.power_filter)
    deg = ref_align_level(deg, model.power_filter)
    ref = ref_preemphasize(ref, model.pre_filter)
    deg = ref_preemphasize(deg, model.pre_filter)

    deg = torch.nn.functional.pad(deg, (0, deg.shape[1] % 256))
    ref = torch.nn.functional.pad(ref, (0, ref.shape[1] % 256))

    ref_bark = ref_stft_bark(
        ref,
        model.to_spec.window,
        model.fbank.fbank,
        model.fbank.pow_dens_correction,
        model.to_spec.n_fft,
        model.to_spec.hop_length,
    )
    deg_bark = ref_stft_bark(
        deg,
        model.to_spec.window,
        model.fbank.fbank,
        model.fbank.pow_dens_correction,
        model.to_spec.n_fft,
        model.to_spec.hop_length,
    )

    return ref_chain(
        ref_bark,
        deg_bark,
        model.loudness.threshs,
        model.loudness.exp,
        model.fbank.width_bark,
        model.fbank.total_width,
        0.1866055,
    )
