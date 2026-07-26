"""Triton IIR filter, a drop-in replacement for :func:`torchaudio.functional.lfilter`.

``torchaudio`` has no CUDA kernel for :func:`~torchaudio.functional.lfilter`; on
the GPU it falls back to a Python loop over the time axis, which dominates the
runtime of the PESQ pipeline. This module replaces that loop by a **chunked
parallel scan**.

Algorithm
---------
The difference equation is written as a linear state space recursion, using the
direct-form-II-transposed state :math:`s \\in \\mathbb{R}^P`

.. math::

    s[n] = A\\,s[n-1] + \\beta\\,x[n], \\qquad y[n] = b_0 x[n] + e_1^T s[n-1]

with :math:`A` the companion matrix of the normalised denominator,
:math:`A_{j1} = -a_j`, :math:`A_{j,j+1} = 1`, and
:math:`\\beta_j = b_j - a_j b_0`.  Splitting the signal into ``C`` chunks of
length ``L`` and exploiting linearity gives, per chunk,

.. math::

    y_c = H x_c + G S_c, \\qquad S_{c+1} = M S_c + W x_c

where :math:`H[t,k] = h[t-k]` is the lower triangular Toeplitz matrix of the
impulse response, :math:`G[t] = e_1^T A^t`, :math:`M = A^L` and
:math:`W[:,k] = A^{L-1-k}\\beta`.  All four tables are constant and are
precomputed on the host in float64.  Three kernels evaluate this: the chunk
local final states, a short sequential scan over the ``C`` chunks, and the
chunk outputs.  Only the middle one is sequential, and it runs over ``C``
instead of ``N`` steps.

State basis
-----------
The companion matrix of a high order filter is strongly non-normal.  For the
order 10 bandpass used by PESQ ``max|A^t|`` reaches ``2.2e3`` and ``A^64``
reaches ``1.1e4``, which makes the float32 tables above useless -- the
recursion overflows to ``inf`` within a few hundred chunks.  The state is
therefore expressed in a better conditioned basis.  Three realisations are
built on the host, each is validated against the difference equation and the
one with the smallest table entries wins:

``companion``
    the plain direct form II transposed state, only competitive for biquads;
``modal``
    :math:`z = T^{-1} s` from the eigendecomposition of :math:`A`, rebalanced
    per eigenvalue block, so :math:`A_z` is block diagonal with
    rotation-scaling blocks;
``cascade``
    the series of second order sections of :func:`scipy.signal.tf2sos`, stacked
    into one block *lower triangular* :math:`A`.

All three are similarity-like reformulations of the same recursion, so the
three kernels are byte identical -- only the constant tables differ.  For the
order 10 PESQ bandpass the largest table entry is ``3.0e4`` (companion),
``4.5e1`` (modal) and ``6.9e0`` (cascade), so the cascade is selected.  A
filter whose best realisation still needs entries beyond
:data:`_FLOAT32_LIMIT` is rejected with an error instead of silently returning
``NaN``.

The resulting filter is far *more* accurate than the oracle it replaces.
Measured against a float64 :func:`~torchaudio.functional.lfilter` on 8x16000
samples of white noise, peak relative error of the order 10 bandpass:

===================  ==========  ==============
pass                 this module  float32 oracle
===================  ==========  ==============
forward                 2.5e-07        8.5e-03
backward                2.3e-07        1.5e-02
===================  ==========  ==============

Backward
--------
For a causal LTI filter :math:`y = \\mathcal{T} x` the matrix
:math:`\\mathcal{T}` is lower triangular Toeplitz, hence
:math:`\\mathcal{T}^T = J \\mathcal{T} J` with :math:`J` the reversal
permutation, and the vector-Jacobian product is exactly

.. math::

    \\bar{x} = J\\,\\mathcal{T}\\,J\\,\\bar{y}

i.e. the very same filter applied to the time reversed upstream gradient.  The
reversal is folded into the kernels as a ``REVERSE`` index flip
:math:`n \\rightarrow N-1-n`, no flipped copy of the data is materialised.
"""

import math
import threading
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import torch

from scipy.signal import tf2sos

from ._common import HAS_TRITON, check_input, next_power_of_two, require_triton

if HAS_TRITON:  # pragma: no cover - depends on the installation
    import triton
    import triton.language as tl
else:  # pragma: no cover - depends on the installation
    triton = None
    tl = None


__all__ = ["TritonIIR", "CHUNK_CANDIDATES", "METHODS"]


#: Chunk lengths considered by ``chunk_size="tune"``.
CHUNK_CANDIDATES = (64, 128, 256, 512)

#: Chunk length used when nothing is specified, measured on a GTX 1080 Ti.
DEFAULT_CHUNK = 256

#: Available inner formulations, see :class:`TritonIIR`.
METHODS = ("scan", "matmul")

#: Largest table entry a realisation may need before it is rejected outright.
#: float32 carries 7 decimal digits, so ``1e5`` still leaves two digits of the
#: filtered signal; beyond that the recursion is pure cancellation noise and,
#: for the ill conditioned companion form of a high order filter, overflows to
#: ``inf``.
_FLOAT32_LIMIT = 1e5

#: Table entry above which float32 accuracy is degraded but still usable.
_FLOAT32_WARN = 1e3

#: Peak relative deviation from the difference equation a realisation may show
#: before it is discarded.  This is a *validity* gate, not a precision one: a
#: broken eigendecomposition is off by 100%, while a healthy realisation of a
#: high order resonant filter can legitimately differ from the float64 direct
#: form by ~1e-6 simply because the direct form itself accumulates error.
_VALID_TOL = 1e-4

#: Element offsets are computed in int32 inside the kernels.
_MAX_ELEMENTS = 2**31 - 1


# ---------------------------------------------------------------------------
# host side filter design
# ---------------------------------------------------------------------------


def _as_coeffs(values: Union[Sequence[float], np.ndarray, torch.Tensor]) -> np.ndarray:
    """Return filter coefficients as a flat float64 numpy array."""

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    return np.asarray(values, dtype=np.float64).ravel()


def _companion(b: np.ndarray, a: np.ndarray):
    """Companion state space form of a difference equation.

    Parameters
    ----------
    b : np.ndarray
        Numerator coefficients
    a : np.ndarray
        Denominator coefficients, ``a[0]`` must not be zero

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        Transition matrix with shape ``[P, P]``, input vector with shape
        ``[P]`` and the direct feed through
    """

    size = max(b.shape[0], a.shape[0])
    b = np.pad(b, (0, size - b.shape[0]))
    a = np.pad(a, (0, size - a.shape[0]))

    if a[0] == 0.0:
        raise ValueError("The leading denominator coefficient a[0] must not be zero.")

    b, a = b / a[0], a / a[0]
    order = size - 1

    if order < 1:
        raise ValueError("The Triton IIR filter needs at least one pole.")

    trans = np.zeros((order, order), dtype=np.float64)
    trans[:, 0] = -a[1:]
    trans[np.arange(order - 1), np.arange(1, order)] = 1.0

    return trans, b[1:] - a[1:] * b[0], float(b[0])


def _cascade(b: np.ndarray, a: np.ndarray):
    """Series of second order sections, stacked into one state space form.

    :func:`scipy.signal.tf2sos` factors the transfer function into biquads.
    Realising section ``k`` in its own companion form and feeding its output
    into section ``k + 1`` gives a block *lower triangular* transition matrix
    whose diagonal blocks are the individual biquads.  Every block is a well
    conditioned 2x2, and the coupling blocks are bounded by the section gains,
    so ``max|A^t|`` stays small even for high order filters where both the
    companion and the modal realisation fall apart.

    Parameters
    ----------
    b : np.ndarray
        Numerator coefficients
    a : np.ndarray
        Denominator coefficients

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]
        Transition matrix, input vector, read out vector and feed through, or
        ``None`` when the factorisation is not available
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sections = np.asarray(tf2sos(b, a), dtype=np.float64)
    except Exception:  # pragma: no cover - needs a degenerate transfer function
        return None

    count = sections.shape[0]
    order = 2 * count

    # per section: s[n] = A s[n-1] + push u[n], out[n] = feed u[n] + tap . s[n-1]
    blocks, pushes, feeds = [], [], []
    for row in sections:
        num, den = row[:3], row[3:]
        if den[0] == 0.0:  # pragma: no cover - scipy normalises a0 to one
            return None

        num, den = num / den[0], den / den[0]
        blocks.append(np.array([[-den[1], 1.0], [-den[2], 0.0]]))
        pushes.append(np.array([num[1] - den[1] * num[0], num[2] - den[2] * num[0]]))
        feeds.append(float(num[0]))

    tap = np.array([1.0, 0.0])
    trans = np.zeros((order, order), dtype=np.float64)
    drive = np.zeros(order, dtype=np.float64)
    read = np.zeros(order, dtype=np.float64)

    for index in range(count):
        rows = slice(2 * index, 2 * index + 2)
        trans[rows, rows] = blocks[index]

        for other in range(index):
            gain = float(np.prod(feeds[other + 1 : index]))
            trans[rows, 2 * other : 2 * other + 2] = np.outer(pushes[index], tap) * gain

        drive[rows] = pushes[index] * float(np.prod(feeds[:index]))
        read[rows] = tap * float(np.prod(feeds[index + 1 :]))

    return trans, drive, read, float(np.prod(feeds))


def _modal_basis(trans: np.ndarray) -> Optional[np.ndarray]:
    """Real block diagonalising basis of ``trans``, ``None`` when it fails.

    The columns span the real invariant subspaces of the eigenvalues, so that
    ``T^-1 A T`` is block diagonal with 1x1 blocks for real poles and 2x2
    rotation-scaling blocks for conjugate pairs.
    """

    order = trans.shape[0]

    try:
        values, vectors = np.linalg.eig(trans)
    except np.linalg.LinAlgError:  # pragma: no cover - needs a pathological matrix
        return None

    # sorting by decaying magnitude keeps conjugate pairs adjacent
    perm = np.argsort(-np.abs(values), kind="stable")
    values, vectors = values[perm], vectors[:, perm]

    used = np.zeros(order, dtype=bool)
    columns = []

    for index in range(order):
        if used[index]:
            continue

        value = values[index]
        if abs(value.imag) <= 1e-13 * max(1.0, abs(value)):
            used[index] = True
            columns.append(vectors[:, index].real.copy())
            continue

        distance = np.where(used, np.inf, np.abs(values - np.conj(value)))
        mate = int(np.argmin(distance))
        if mate == index or used[mate] or not np.isfinite(distance[mate]):
            return None

        used[index] = used[mate] = True
        columns.append(vectors[:, index].real.copy())
        columns.append(vectors[:, index].imag.copy())

    basis = np.stack(columns, axis=1)
    norms = np.linalg.norm(basis, axis=0)

    if not np.all(np.isfinite(basis)) or np.any(norms == 0.0):
        return None

    return basis / norms


def _diagonal_blocks(state: np.ndarray):
    """Start index and size of every diagonal block of a quasi diagonal matrix."""

    order = state.shape[0]
    scale = max(float(np.abs(state).max()), 1e-300)

    blocks, index = [], 0
    while index < order:
        if index + 1 < order and abs(state[index, index + 1]) > 1e-10 * scale:
            blocks.append((index, 2))
            index += 2
        else:
            blocks.append((index, 1))
            index += 1

    return blocks


def _impulse_response(
    state: np.ndarray, drive: np.ndarray, read: np.ndarray, feed: float, taps: int
) -> np.ndarray:
    """First ``taps`` samples of the impulse response of a state space form.

    Evaluates ``h[0] = feed`` and ``h[m] = read . A^(m-1) . drive``, the
    response of ``s[n] = A s[n-1] + drive x[n]``,
    ``y[n] = feed x[n] + read . s[n-1]`` to a unit impulse.
    """

    out = np.zeros(taps, dtype=np.float64)
    out[0] = feed

    vector = drive.copy()
    for tap in range(1, taps):
        out[tap] = float(read @ vector)
        vector = state @ vector

    return out


def _direct_response(b: np.ndarray, a: np.ndarray, taps: int) -> np.ndarray:
    """First ``taps`` samples of the impulse response of ``b / a``.

    Evaluated straight from the difference equation, which is the numerically
    stable way to get it: unlike :func:`_impulse_response` on a companion
    matrix it never forms the badly conditioned powers :math:`A^t`.  This is
    the ground truth every realisation is validated against.
    """

    size = max(b.shape[0], a.shape[0])
    num = np.pad(b, (0, size - b.shape[0]))
    den = np.pad(a, (0, size - a.shape[0]))
    num, den = num / den[0], den / den[0]

    out = np.zeros(taps, dtype=np.float64)
    for step in range(taps):
        value = num[step] if step < size else 0.0
        for lag in range(1, min(size, step + 1)):
            value -= den[lag] * out[step - lag]
        out[step] = value

    return out


def _modal_realisation(trans: np.ndarray, beta: np.ndarray):
    """Modal realisation of a companion form, ``None`` when it cannot be built."""

    basis = _modal_basis(trans)
    if basis is None:
        return None

    try:
        inverse = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        # a defective companion matrix has fewer eigenvectors than states, so
        # the modal basis is singular -- there simply is no modal realisation
        return None

    state = inverse @ trans @ basis

    # a scalar rescaling per block keeps the block form intact and balances the
    # input against the read out vector
    drive, read = inverse @ beta, basis[0, :].copy()
    for start, size in _diagonal_blocks(state):
        gain = np.linalg.norm(read[start : start + size])
        push = np.linalg.norm(drive[start : start + size])
        if gain > 0.0 and push > 0.0:
            alpha = math.sqrt(push / gain)
            read[start : start + size] *= alpha
            drive[start : start + size] /= alpha

    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(drive)):
        return None

    return state, drive, read


def _condition(state: np.ndarray, read: np.ndarray, taps: int) -> float:
    """Largest table entry the kernels would have to handle for this form."""

    power, current = np.eye(state.shape[0]), 0.0
    for _ in range(taps):
        current = max(current, np.abs(read @ power).max(), np.abs(power).max())
        power = state @ power

    return float(current)


def _state_space(b: np.ndarray, a: np.ndarray, taps: int):
    """Well conditioned state space realisation of a difference equation.

    The companion, the modal and the cascaded realisation are built, each is
    validated against :func:`_direct_response` and the best conditioned of the
    survivors is returned.  Conditioning is measured as the largest table entry
    any kernel would have to handle.

    Parameters
    ----------
    b : np.ndarray
        Numerator coefficients
    a : np.ndarray
        Denominator coefficients
    taps : int
        Longest chunk length the tables will be built for, the realisations are
        only scored over that horizon

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]
        Transition matrix, input vector, read out vector and feed through of
        the realisation, plus the name of the selected basis

    Raises
    ------
    RuntimeError
        When no realisation reproduces the difference equation, or when the
        best one still needs table entries float32 cannot carry
    """

    trans, beta, feed = _companion(b, a)
    unit = np.eye(trans.shape[0])[0]
    reference = _direct_response(b, a, taps)
    scale = max(np.abs(reference).max(), 1e-30)

    candidates = [("companion", trans, beta, unit, feed)]

    modal = _modal_realisation(trans, beta)
    if modal is not None:
        candidates.append(("modal",) + modal + (feed,))

    cascade = _cascade(b, a)
    if cascade is not None:
        candidates.append(("cascade",) + cascade)

    best, score = None, np.inf
    for name, state, drive, read, direct in candidates:
        response = _impulse_response(state, drive, read, direct, taps)
        error = np.abs(response - reference).max() / scale
        if not np.isfinite(error) or error > _VALID_TOL:
            continue

        current = _condition(state, read, taps)
        if not math.isfinite(current):  # pragma: no cover - caught by the gate
            continue

        if current < score:
            best, score = (name, state, drive, read, direct), current

    if best is None:  # pragma: no cover - needs a pathological filter
        raise RuntimeError(
            "Could not build a numerically valid state space realisation for the "
            "given coefficients."
        )

    name, state, drive, read, direct = best

    if score > _FLOAT32_LIMIT:
        raise RuntimeError(
            f"The best state space realisation of this filter (`{name}`, order "
            f"{state.shape[0]}) needs table entries up to {score:.3g}, which "
            f"float32 cannot carry -- the recursion would return `inf` or "
            f"`NaN`. Split the filter into second order sections and chain "
            f"several `TritonIIR` modules instead."
        )
    if score > _FLOAT32_WARN:
        warnings.warn(
            f"The state space realisation of this filter (`{name}`, order "
            f"{state.shape[0]}) needs table entries up to {score:.3g}; float32 "
            f"accuracy will be degraded by roughly that factor.",
            RuntimeWarning,
            stacklevel=3,
        )

    return state, drive, read, direct, name


def _tables(state, drive, read, feed, taps, padded):
    """Constant tables of the chunked scan, transposed for ``tl.dot``.

    Parameters
    ----------
    state : np.ndarray
        Transition matrix with shape ``[P, P]``
    drive : np.ndarray
        Input vector with shape ``[P]``
    read : np.ndarray
        Read out vector with shape ``[P]``
    feed : float
        Direct feed through
    taps : int
        Chunk length ``L``
    padded : int
        State size the tables are zero padded to, at least 16 for ``tl.dot``

    Returns
    -------
    Dict[str, np.ndarray]
        ``impulse`` ``[L, L]``, ``carry`` ``[padded, L]``, ``drive``
        ``[L, padded]``, ``power`` and ``state`` ``[padded, padded]``, ``input``
        and ``read`` ``[padded]``
    """

    order = state.shape[0]

    # G[t] = read . A^t, and the loop leaves power = A^L = M behind
    gain = np.zeros((taps, order), dtype=np.float64)
    power = np.eye(order, dtype=np.float64)
    for tap in range(taps):
        gain[tap] = read @ power
        power = state @ power

    # W[:, k] = A^(L-1-k) . drive
    inject = np.zeros((order, taps), dtype=np.float64)
    accum = np.eye(order, dtype=np.float64)
    for tap in range(taps - 1, -1, -1):
        inject[:, tap] = accum @ drive
        accum = state @ accum

    impulse = np.zeros(taps, dtype=np.float64)
    impulse[0] = feed
    if taps > 1:
        impulse[1:] = gain[:-1] @ drive

    # HT[k, t] = h[t - k] for t >= k, zero above the diagonal
    index = np.arange(taps)
    delta = index[None, :] - index[:, None]
    toeplitz = np.where(delta >= 0, impulse[np.clip(delta, 0, taps - 1)], 0.0)

    def pad(matrix, rows, cols):
        out = np.zeros((rows, cols), dtype=np.float64)
        out[: matrix.shape[0], : matrix.shape[1]] = matrix
        return out

    return {
        "impulse": toeplitz,
        "carry": pad(gain.T, padded, taps),
        "drive": pad(inject.T, taps, padded),
        "power": pad(power.T, padded, padded),
        "state": pad(state.T, padded, padded),
        "input": np.pad(drive, (0, padded - order)),
        "read": np.pad(read, (0, padded - order)),
    }


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:  # pragma: no cover - depends on the installation

    @triton.jit
    def _state_matmul_kernel(
        x_ptr,
        state_ptr,
        drive_ptr,
        ntime,
        nchunk,
        stride_batch,
        TAPS: tl.constexpr,
        PADDED: tl.constexpr,
        REVERSE: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Zero state final state of every chunk, as ``S = X @ W^T``."""

        pid_c = tl.program_id(0)
        pid_b = tl.program_id(1)

        rows = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        valid = rows < nchunk
        cols = tl.arange(0, PADDED)

        acc = tl.zeros((BLOCK_C, PADDED), dtype=tl.float32)
        for start in range(0, TAPS, BLOCK_K):
            taps = start + tl.arange(0, BLOCK_K)
            index = rows[:, None] * TAPS + taps[None, :]
            mask = valid[:, None] & (index < ntime)
            if REVERSE:
                index = tl.maximum(ntime - 1 - index, 0)

            values = tl.load(x_ptr + pid_b * stride_batch + index, mask=mask, other=0.0)
            table = tl.load(drive_ptr + taps[:, None] * PADDED + cols[None, :])
            acc += tl.dot(values, table, allow_tf32=False)

        out = state_ptr + (pid_b * nchunk + rows[:, None]) * PADDED + cols[None, :]
        tl.store(out, acc, mask=valid[:, None])

    @triton.jit
    def _state_scan_kernel(
        x_ptr,
        state_ptr,
        trans_ptr,
        input_ptr,
        ntime,
        nchunk,
        stride_batch,
        TAPS: tl.constexpr,
        PADDED: tl.constexpr,
        REVERSE: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Zero state final state of every chunk, by a sequential recursion.

        One lane per chunk, so the per step loads are strided by ``TAPS`` and
        cannot coalesce -- that is the price of this formulation.
        """

        pid_c = tl.program_id(0)
        pid_b = tl.program_id(1)

        rows = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        valid = rows < nchunk
        cols = tl.arange(0, PADDED)

        trans = tl.load(trans_ptr + cols[:, None] * PADDED + cols[None, :])
        inject = tl.load(input_ptr + cols)

        acc = tl.zeros((BLOCK_C, PADDED), dtype=tl.float32)
        for tap in range(0, TAPS):
            index = rows * TAPS + tap
            mask = valid & (index < ntime)
            if REVERSE:
                index = tl.maximum(ntime - 1 - index, 0)

            values = tl.load(x_ptr + pid_b * stride_batch + index, mask=mask, other=0.0)
            acc = tl.sum(acc[:, :, None] * trans[None, :, :], axis=1)
            acc += inject[None, :] * values[:, None]

        out = state_ptr + (pid_b * nchunk + rows[:, None]) * PADDED + cols[None, :]
        tl.store(out, acc, mask=valid[:, None])

    @triton.jit
    def _carry_kernel(
        state_ptr,
        carry_ptr,
        power_ptr,
        nbatch,
        nchunk,
        PADDED: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        """Sequential scan ``S_{c+1} = M S_c + s_c`` over the chunks.

        The whole batch is handled by one program so that the only sequential
        dependency left in the op is this ``nchunk`` step chain.  A broadcast
        multiply plus reduction beats ``tl.dot`` here because the tiles are at
        the ``tl.dot`` minimum size and the chain is latency bound.
        """

        pid = tl.program_id(0)
        rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        valid = rows < nbatch
        cols = tl.arange(0, PADDED)

        power = tl.load(power_ptr + cols[:, None] * PADDED + cols[None, :])
        acc = tl.zeros((BLOCK_B, PADDED), dtype=tl.float32)

        base = rows[:, None] * nchunk * PADDED + cols[None, :]
        for chunk in range(0, nchunk):
            offset = base + chunk * PADDED
            local = tl.load(state_ptr + offset, mask=valid[:, None], other=0.0)
            tl.store(carry_ptr + offset, acc, mask=valid[:, None])
            acc = tl.sum(acc[:, :, None] * power[None, :, :], axis=1) + local

    @triton.jit
    def _output_matmul_kernel(
        x_ptr,
        y_ptr,
        carry_ptr,
        impulse_ptr,
        gain_ptr,
        ntime,
        nchunk,
        stride_batch,
        TAPS: tl.constexpr,
        PADDED: tl.constexpr,
        REVERSE: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Chunk output ``Y = X @ H^T + S @ G^T``."""

        tiles = TAPS // BLOCK_T
        pid = tl.program_id(0)
        pid_c = pid // tiles
        pid_t = pid % tiles
        pid_b = tl.program_id(1)

        rows = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        valid = rows < nchunk
        times = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = tl.arange(0, PADDED)

        carry = tl.load(
            carry_ptr + (pid_b * nchunk + rows[:, None]) * PADDED + cols[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        gain = tl.load(gain_ptr + cols[:, None] * TAPS + times[None, :])
        acc = tl.dot(carry, gain, allow_tf32=False)

        # H is lower triangular, only k < (pid_t + 1) * BLOCK_T contributes
        for start in range(0, (pid_t + 1) * BLOCK_T, BLOCK_K):
            taps = start + tl.arange(0, BLOCK_K)
            index = rows[:, None] * TAPS + taps[None, :]
            mask = valid[:, None] & (index < ntime)
            if REVERSE:
                index = tl.maximum(ntime - 1 - index, 0)

            values = tl.load(x_ptr + pid_b * stride_batch + index, mask=mask, other=0.0)
            table = tl.load(impulse_ptr + taps[:, None] * TAPS + times[None, :])
            acc += tl.dot(values, table, allow_tf32=False)

        index = rows[:, None] * TAPS + times[None, :]
        mask = valid[:, None] & (index < ntime)
        if REVERSE:
            index = tl.maximum(ntime - 1 - index, 0)
        tl.store(y_ptr + pid_b * stride_batch + index, acc, mask=mask)

    @triton.jit
    def _output_scan_kernel(
        x_ptr,
        y_ptr,
        carry_ptr,
        trans_ptr,
        input_ptr,
        read_ptr,
        feed,
        ntime,
        nchunk,
        stride_batch,
        TAPS: tl.constexpr,
        PADDED: tl.constexpr,
        REVERSE: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Chunk output by a sequential recursion over the ``TAPS`` steps."""

        pid_c = tl.program_id(0)
        pid_b = tl.program_id(1)

        rows = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        valid = rows < nchunk
        cols = tl.arange(0, PADDED)

        trans = tl.load(trans_ptr + cols[:, None] * PADDED + cols[None, :])
        inject = tl.load(input_ptr + cols)
        read = tl.load(read_ptr + cols)

        acc = tl.load(
            carry_ptr + (pid_b * nchunk + rows[:, None]) * PADDED + cols[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        for tap in range(0, TAPS):
            index = rows * TAPS + tap
            mask = valid & (index < ntime)
            if REVERSE:
                index = tl.maximum(ntime - 1 - index, 0)

            values = tl.load(x_ptr + pid_b * stride_batch + index, mask=mask, other=0.0)
            out = feed * values + tl.sum(acc * read[None, :], axis=1)
            tl.store(y_ptr + pid_b * stride_batch + index, out, mask=mask)
            acc = tl.sum(acc[:, :, None] * trans[None, :, :], axis=1)
            acc += inject[None, :] * values[:, None]

    # (constexpr kwargs, num_warps, num_stages); every supported chunk length is
    # a multiple of 64, so all tile sizes below divide it
    _STATE_CONFIGS = (
        ({"BLOCK_C": 16, "BLOCK_K": 64}, 2, 2),
        ({"BLOCK_C": 32, "BLOCK_K": 64}, 4, 2),
        ({"BLOCK_C": 64, "BLOCK_K": 64}, 4, 2),
        ({"BLOCK_C": 32, "BLOCK_K": 32}, 2, 3),
    )

    _OUTPUT_CONFIGS = (
        ({"BLOCK_C": 16, "BLOCK_T": 32, "BLOCK_K": 32}, 2, 2),
        ({"BLOCK_C": 16, "BLOCK_T": 64, "BLOCK_K": 64}, 4, 2),
        ({"BLOCK_C": 32, "BLOCK_T": 32, "BLOCK_K": 32}, 4, 2),
        ({"BLOCK_C": 32, "BLOCK_T": 64, "BLOCK_K": 64}, 4, 2),
        ({"BLOCK_C": 64, "BLOCK_T": 64, "BLOCK_K": 64}, 8, 2),
        ({"BLOCK_C": 32, "BLOCK_T": 64, "BLOCK_K": 64}, 8, 3),
    )

    _CARRY_CONFIGS = (({}, 2, 2), ({}, 4, 2), ({}, 8, 2))

    _SCAN_CONFIGS = (
        ({"BLOCK_C": 16}, 1, 2),
        ({"BLOCK_C": 32}, 2, 2),
        ({"BLOCK_C": 64}, 4, 2),
        ({"BLOCK_C": 128}, 8, 2),
    )


# ---------------------------------------------------------------------------
# launch helpers
# ---------------------------------------------------------------------------

#: Resolved kernel configuration per (kernel, problem shape).
_CONFIG_CACHE = {}

#: Devices whose CUDA context has been bound on the calling thread.
_CONTEXT = threading.local()


def _bind_context(device: torch.device) -> None:
    """Make the CUDA primary context current on the calling thread.

    Triton's ``load_binary`` retains and activates the primary context on
    whichever thread first compiles a kernel, but its launcher does not.  A
    kernel compiled on the main thread therefore fails with ``invalid device
    context`` when the autograd engine launches it from its worker thread --
    exactly what happens to the backward pass here.  Any real driver call binds
    the context; a non blocking stream query is the cheapest one, and it is
    only paid once per thread and device.
    """

    seen = getattr(_CONTEXT, "seen", None)
    if seen is None:
        seen = _CONTEXT.seen = set()

    if device.index not in seen:
        torch.cuda.current_stream(device).query()
        seen.add(device.index)


def _select(slot, configs, launch):
    """Run ``launch`` with the fastest of ``configs``, benchmarked once.

    ``launch(config)`` has to be a pure function of its inputs, every kernel in
    this module is, so running a candidate during the search already produces
    the right answer.  The winner is cached under ``slot``, hence a given shape
    always ends up in the same kernel and the op is bitwise reproducible within
    a process.  A challenger has to be clearly faster than the incumbent to
    displace it, which keeps the choice -- and with it the float32 summation
    order -- stable against benchmark noise on a busy GPU.
    """

    chosen = _CONFIG_CACHE.get(slot)
    if chosen is not None:
        launch(chosen)
        return

    best, score, failure = None, math.inf, None
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    for config in configs:
        try:
            launch(config)
            torch.cuda.synchronize()
        except Exception as error:  # out of registers, shared memory, ...
            failure = failure or error
            continue

        start.record()
        for _ in range(4):
            launch(config)
        stop.record()
        torch.cuda.synchronize()

        elapsed = start.elapsed_time(stop)
        if best is None or elapsed < 0.95 * score:
            best, score = config, elapsed

    if best is None:
        # never swallow the reason: a wrong device, an illegal access or an out
        # of memory all arrive here and look identical without the chained cause
        raise RuntimeError(
            f"No usable Triton configuration for `{slot[0]}` at this shape."
        ) from failure

    _CONFIG_CACHE[slot] = best
    launch(best)


def _launch(signal: torch.Tensor, plan, reverse: bool) -> torch.Tensor:
    """Run the three kernels of the chunked scan on ``signal``."""

    if plan.impulse.device != signal.device:
        raise RuntimeError(
            f"The filter tables live on {plan.impulse.device} but the signal on "
            f"{signal.device}; move the module with `.to(signal.device)` first."
        )

    _bind_context(signal.device)

    nbatch, ntime = signal.shape
    taps, padded = plan.taps, plan.padded
    nchunk = (ntime + taps - 1) // taps

    if nbatch * ntime > _MAX_ELEMENTS:
        # the kernels index with int32 offsets, `pid_b * ntime + n` would wrap
        raise RuntimeError(
            f"The Triton IIR filter indexes with int32 offsets and cannot take "
            f"more than {_MAX_ELEMENTS} elements at once, got {nbatch * ntime}; "
            f"split the batch."
        )

    out = torch.empty_like(signal)
    states, carry = plan.scratch(signal.device, nbatch, nchunk)
    shape = (taps, padded, reverse, nbatch, nchunk, ntime)

    if plan.method == "matmul":

        def state(config):
            kwargs, warps, stages = config
            grid = (triton.cdiv(nchunk, kwargs["BLOCK_C"]), nbatch)
            _state_matmul_kernel[grid](
                signal,
                states,
                plan.drive,
                ntime,
                nchunk,
                ntime,
                TAPS=taps,
                PADDED=padded,
                REVERSE=reverse,
                num_warps=warps,
                num_stages=stages,
                **kwargs,
            )

        _select(("state_matmul",) + shape, _STATE_CONFIGS, state)
    else:

        def state(config):
            kwargs, warps, stages = config
            grid = (triton.cdiv(nchunk, kwargs["BLOCK_C"]), nbatch)
            _state_scan_kernel[grid](
                signal,
                states,
                plan.state,
                plan.input,
                ntime,
                nchunk,
                ntime,
                TAPS=taps,
                PADDED=padded,
                REVERSE=reverse,
                num_warps=warps,
                num_stages=stages,
                **kwargs,
            )

        _select(("state_scan",) + shape, _SCAN_CONFIGS, state)

    def scan(config):
        _, warps, stages = config
        _carry_kernel[(triton.cdiv(nbatch, 16),)](
            states,
            carry,
            plan.power,
            nbatch,
            nchunk,
            PADDED=padded,
            BLOCK_B=16,
            num_warps=warps,
            num_stages=stages,
        )

    _select(("carry",) + shape, _CARRY_CONFIGS, scan)

    if plan.method == "matmul":

        def output(config):
            kwargs, warps, stages = config
            grid = (
                triton.cdiv(nchunk, kwargs["BLOCK_C"]) * (taps // kwargs["BLOCK_T"]),
                nbatch,
            )
            _output_matmul_kernel[grid](
                signal,
                out,
                carry,
                plan.impulse,
                plan.carry,
                ntime,
                nchunk,
                ntime,
                TAPS=taps,
                PADDED=padded,
                REVERSE=reverse,
                num_warps=warps,
                num_stages=stages,
                **kwargs,
            )

        _select(("output_matmul",) + shape, _OUTPUT_CONFIGS, output)
    else:

        def output(config):
            kwargs, warps, stages = config
            grid = (triton.cdiv(nchunk, kwargs["BLOCK_C"]), nbatch)
            _output_scan_kernel[grid](
                signal,
                out,
                carry,
                plan.state,
                plan.input,
                plan.read,
                plan.feed,
                ntime,
                nchunk,
                ntime,
                TAPS=taps,
                PADDED=padded,
                REVERSE=reverse,
                num_warps=warps,
                num_stages=stages,
                **kwargs,
            )

        _select(("output_scan",) + shape, _SCAN_CONFIGS, output)

    return out


class _Plan:
    """Everything a launch needs, gathered once per chunk length.

    Attributes
    ----------
    taps : int
        Chunk length ``L``
    padded : int
        Zero padded state size
    method : str
        Inner formulation, ``"matmul"`` or ``"scan"``
    feed : float
        Direct feed through of the filter
    """

    __slots__ = (
        "taps",
        "padded",
        "method",
        "feed",
        "impulse",
        "carry",
        "drive",
        "power",
        "state",
        "input",
        "read",
        "_scratch",
    )

    def __init__(self, taps, padded, method, feed, buffers):
        self.taps = taps
        self.padded = padded
        self.method = method
        self.feed = feed
        self._scratch = {}

        for name in ("impulse", "carry", "drive", "power", "state", "input", "read"):
            setattr(self, name, buffers[f"_{name}_{taps}"])

    def scratch(self, device, nbatch, nchunk):
        """Reusable chunk state buffers, large enough for one problem size.

        The kernels address these buffers linearly as
        ``(batch * nchunk + chunk) * PADDED + state``, so a flat buffer that is
        merely *big enough* works for every shape.  Keeping one growable pair
        per device instead of one pair per shape bounds the memory to the
        largest input seen -- a per shape cache leaks a buffer for every audio
        length a training loop ever hands over.

        The buffers are private to the plan and every kernel touching them is
        enqueued on the current stream before this call returns, so reusing
        them across calls is safe and saves two allocations per launch.
        """

        needed = nbatch * nchunk * self.padded
        buffers = self._scratch.get(device)

        if buffers is None or buffers[0].numel() < needed:
            buffers = (
                torch.empty(needed, dtype=torch.float32, device=device),
                torch.empty(needed, dtype=torch.float32, device=device),
            )
            self._scratch[device] = buffers

        return buffers


class _IIRFunction(torch.autograd.Function):
    """Autograd wrapper, the backward is the same filter run time reversed.

    With :math:`J` the reversal permutation and :math:`\\mathcal{T}` the causal
    filter, the forward evaluates :math:`\\mathcal{T}` for ``reverse=False`` and
    :math:`J \\mathcal{T} J` for ``reverse=True``.  Because
    :math:`\\mathcal{T}^T = J \\mathcal{T} J` and
    :math:`(J \\mathcal{T} J)^T = \\mathcal{T}`, flipping the flag is exactly
    the adjoint, which also makes the backward itself differentiable.
    """

    @staticmethod
    def forward(ctx, signal, plan, reverse):
        ctx.plan, ctx.reverse = plan, reverse
        return _launch(signal, plan, reverse)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        reverse = not ctx.reverse

        if torch.is_grad_enabled() and grad.requires_grad:
            out = _IIRFunction.apply(grad, ctx.plan, reverse)
        else:
            out = _launch(grad, ctx.plan, reverse)

        return out, None, None


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------


class TritonIIR(torch.nn.Module):
    """Causal IIR filter with zero initial conditions, evaluated on the GPU.

    Numerically equivalent to
    ``torchaudio.functional.lfilter(x, a_coeffs, b_coeffs, clamp=False)``, but
    evaluated with a chunked parallel scan instead of a Python loop over the
    time axis.  The coefficients are compile time constants, no gradient flows
    into them.

    Parameters
    ----------
    b_coeffs : Union[Sequence[float], np.ndarray, torch.Tensor]
        Numerator coefficients of the difference equation
    a_coeffs : Union[Sequence[float], np.ndarray, torch.Tensor]
        Denominator coefficients, everything is normalised by ``a_coeffs[0]``
    chunk_size : Optional[Union[int, str]]
        Length ``L`` of a chunk, a multiple of 64.  ``None`` uses
        :data:`DEFAULT_CHUNK`, ``"tune"`` benchmarks :data:`CHUNK_CANDIDATES`
        once per input shape and caches the winner.
    method : str
        ``"matmul"`` evaluates the chunk local response with ``tl.dot`` on
        ``[chunk, L]`` tiles, ``"scan"`` runs the recursion sequentially over
        the ``L`` steps with one lane per chunk.  ``"auto"`` selects
        ``"matmul"``, which is 2-6x faster on this hardware because the scan
        formulation cannot coalesce its loads.

    Attributes
    ----------
    order : int
        Number of poles ``P`` of the filter
    states : int
        Size of the state vector of the selected realisation, ``order`` for the
        companion and the modal form and ``2 * ceil(order / 2)`` for the
        cascade
    basis : str
        Selected state space realisation, ``"cascade"``, ``"modal"`` or
        ``"companion"``
    padded : int
        State size the constant tables are zero padded to

    Raises
    ------
    RuntimeError
        When no realisation of the coefficients survives float32, see
        :func:`_state_space`

    Notes
    -----
    The float32 accuracy of this module is much better than the one of
    ``torchaudio``'s float32 ``lfilter``: for the order 10 PESQ bandpass the
    peak relative error against a float64 ``lfilter`` is ``2.4e-7`` here and
    ``8.5e-3`` there.  A parity test against the float32 oracle can therefore
    never be tighter than the error of the oracle itself, which is why the test
    suite compares both against a float64 reference.

    Measured on a GTX 1080 Ti with CUDA events, median of 25 calls, order 10
    filter, against ``torchaudio``'s CUDA fallback: ``88us`` versus ``14983us``
    at ``[8, 16000]`` and ``158us`` versus ``62251us`` at ``[8, 64000]``.

    Tile sizes are benchmarked on first use and cached per shape, so results
    are bitwise reproducible within a process but a different machine or a
    differently loaded GPU may settle on another tile size and differ in the
    last float32 digits.  Pass an explicit ``chunk_size`` and warm the cache to
    pin it down completely.  The flip side of the per shape cache is that the
    *first* call at a new ``[batch, sample]`` shape pays roughly 2ms of
    benchmarking; feed the module a stable shape when that matters.

    Only signals with ``batch * sample <= 2**31 - 1`` are supported, the
    kernels index with int32 offsets.
    """

    def __init__(
        self,
        b_coeffs,
        a_coeffs,
        chunk_size: Optional[Union[int, str]] = None,
        method: str = "auto",
    ):
        super(TritonIIR, self).__init__()

        require_triton()

        if method == "auto":
            method = "matmul"
        if method not in METHODS:
            raise ValueError(f"Unknown method `{method}`, pick one of {METHODS}.")

        self.method = method
        self._plans = {}
        self._tuned = {}

        numerator, denominator = _as_coeffs(b_coeffs), _as_coeffs(a_coeffs)

        if chunk_size == "tune":
            candidates = list(CHUNK_CANDIDATES)
        elif chunk_size is None:
            candidates = [DEFAULT_CHUNK]
        else:
            chunk_size = int(chunk_size)
            if chunk_size < 64 or chunk_size % 64 != 0:
                raise ValueError("`chunk_size` has to be a positive multiple of 64.")
            candidates = [chunk_size]

        self.candidates = tuple(candidates)

        state, drive, read, feed, basis = _state_space(
            numerator, denominator, max(candidates)
        )

        # the pole count of the difference equation, which the cascade may
        # realise with one extra (dead) state when the order is odd
        self.order = max(numerator.shape[0], denominator.shape[0]) - 1
        self.states = state.shape[0]
        self.basis = basis
        self.feed = feed
        self.padded = max(16, next_power_of_two(self.states))

        for taps in candidates:
            for name, value in _tables(
                state, drive, read, feed, taps, self.padded
            ).items():
                self.register_buffer(
                    f"_{name}_{taps}",
                    torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32),
                    persistent=False,
                )

    def _apply(self, *args, **kwargs):
        """Drop the cached launch plans, they hold references to the buffers."""

        self._plans.clear()
        return super(TritonIIR, self)._apply(*args, **kwargs)

    def _plan(self, taps: int) -> _Plan:
        """Launch plan for one chunk length, built once."""

        plan = self._plans.get(taps)

        if plan is None:
            plan = _Plan(taps, self.padded, self.method, self.feed, self._buffers)
            self._plans[taps] = plan

        return plan

    def _pick(self, signal: torch.Tensor) -> _Plan:
        """Launch plan for ``signal``, benchmarked once per input shape."""

        if len(self.candidates) == 1:
            return self._plan(self.candidates[0])

        key = tuple(signal.shape)
        taps = self._tuned.get(key)

        if taps is None:
            probe = torch.zeros_like(signal)
            best, score = self.candidates[0], math.inf

            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)

            for candidate in self.candidates:
                plan = self._plan(candidate)
                _launch(probe, plan, False)
                torch.cuda.synchronize()

                start.record()
                for _ in range(8):
                    _launch(probe, plan, False)
                stop.record()
                torch.cuda.synchronize()

                # same hysteresis as `_select`, a longer chunk has to earn it
                elapsed = start.elapsed_time(stop)
                if elapsed < 0.95 * score:
                    best, score = candidate, elapsed

            taps = best
            self._tuned[key] = taps

        return self._plan(taps)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Filter a batch of time signals.

        Parameters
        ----------
        signal : torch.Tensor
            Input time signal with shape ``[batch, sample]``

        Returns
        -------
        torch.Tensor
            Filtered signal with shape ``[batch, sample]``
        """

        signal = check_input(signal, "signal")

        if signal.dim() != 2:
            raise RuntimeError(
                f"The Triton IIR filter needs a [batch, sample] tensor, got "
                f"{tuple(signal.shape)}."
            )
        if signal.shape[1] == 0:
            return signal.clone()

        plan = self._pick(signal)

        if not (torch.is_grad_enabled() and signal.requires_grad):
            return _launch(signal, plan, False)

        return _IIRFunction.apply(signal, plan, False)
