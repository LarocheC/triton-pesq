# Triton backend

`torch_pesq.PesqLossTriton` is a drop-in replacement for `PesqLoss` in which every
stage of the PESQ pipeline — forward *and* backward — is a Triton GPU kernel.

```python
import torch
from torch_pesq import PesqLossTriton

pesq = PesqLossTriton(0.5, sample_rate=16000).cuda()

mos = pesq.mos(reference, degraded)
loss = pesq(reference, degraded)
loss.sum().backward()
```

It subclasses `PesqLoss`, so both share the exact same filter coefficients,
filterbank and loudness parametrization, and only `raw()`, `mos()` and
`forward()` are overridden. Inputs must be float32 CUDA tensors.

## Why it is faster

The PyTorch implementation spends essentially all of its time in
`torchaudio.functional.lfilter`. On CUDA torchaudio falls back to a Python loop
over time samples, so the two level-alignment filters and the two pre-emphasis
filters cost ~43 ms of a ~43 ms forward pass at batch 8 and one second of audio.
Everything after them — spectrogram, Bark filterbank, loudness, disturbance —
takes well under a millisecond in total but is spread over roughly forty
kernel launches on tensors that are only a few hundred kilobytes.

The Triton backend attacks both: the IIR filters become parallel scans instead
of a sequential loop, and the perceptual model is fused into a handful of
kernels that keep a whole frame in registers.

Once the filters are fast the pipeline stops being GPU bound. At batch 8 and one
second of audio a forward and backward pass is only ~0.8 ms of device time but
~5 ms of host time, spent dispatching around forty small kernels. That is why
:meth:`PesqLossTriton.graphed` exists: replaying the whole pass from a CUDA
graph removes the dispatch entirely and is bit exact.

```python
step = pesq.graphed(reference, degraded)   # fixes the input shape
loss = step(reference, degraded)
loss.sum().backward()
```

It is also why further micro-optimisation of individual kernels has little left
to give, and why the benchmark measures synchronised wall clock rather than CUDA
events — an event pair around a host bound workload measures only the device
part and flatters it by several times.

## Stage decomposition

`torch_pesq/triton_ops/reference.py` holds a pure PyTorch decomposition of
`PesqLoss.raw()` into the stages below. It is verified to reproduce the original
bit for bit, gradients included, and is the oracle every kernel is tested
against.

| stage | module | kernel file |
|---|---|---|
| peak normalisation, edge ramp, level alignment, MOS epilogue | functions | `elementwise.py` |
| resampling to 16 kHz | `TritonResample` | `resample.py` |
| IIR filtering | `TritonIIR` | `iir.py` |
| STFT → power → Bark bands | `TritonStftBark` | `spectral.py` |
| perceptual model, disturbance, overlapping sums | `TritonPesqChain` | `chain.py` |

Reference and degraded signal are processed as a single batch of twice the size,
which halves the number of kernel launches for everything up to the Bark
spectrogram.

## IIR filtering as a parallel scan

An IIR filter is a sequential recurrence, but a linear time-invariant one, so it
can be split into independent chunks and stitched back together. With the
direct-form-II-transposed state $s$ of a filter of order $P$,

$$s[n] = A\,s[n-1] + \beta\,x[n], \qquad y[n] = b_0 x[n] + e_1^\top s[n-1],$$

where $A$ is the companion matrix built from the denominator coefficients and
$\beta_j = b_j - a_j b_0$. Because the filter is time invariant, a chunk of
length $L$ has a *constant* transition matrix $M = A^L$, so for chunk $c$

$$y_\text{chunk} = y_\text{zero-state}(x_\text{chunk}) + G\,S_c, \qquad
S_{c+1} = M\,S_c + s_\text{zero-state}(x_\text{chunk}),$$

with $G[t,:]$ the first row of $A^t$. All of $A$, $M$, $G$, the impulse response
$h$ and the input-to-final-state map are precomputed on the host in double
precision. That leaves three kernels: one embarrassingly parallel pass over all
(batch, chunk) pairs for the zero-state responses, a tiny sequential scan over
the chunks, and a second parallel pass that adds the homogeneous correction.

The backward pass needs no new machinery. For a causal LTI filter $y = Tx$ with
$T$ lower-triangular Toeplitz, the vector-Jacobian product is

$$\nabla_x = T^\top \nabla_y = \operatorname{flip}\left(T\operatorname{flip}(\nabla_y)\right),$$

i.e. the same filter run backwards in time, which the kernels implement with a
compile-time index flip.

## Fused spectrogram and Bark filterbank

The filterbank matrix of `BarkScale` is binary, contiguous and non-overlapping —
band $i$ is the sum of a bin range $[\text{start}_i, \text{end}_i)$. The Bark
transform is therefore a segment sum, not a matrix product, and the
`[batch, frame, 257]` power spectrogram never has to exist: one kernel windows
the frame, evaluates the real DFT, squares, and accumulates directly into the
`[batch, frame, bark]` output. The trailing zero padding of the reference
implementation is folded into the masked loads, and bin 0 (which the reference
discards) is skipped entirely.

## The perceptual chain

Everything from silent-frame detection to the overlapping sums is five kernels.
The band count is at most 49, so a whole frame fits in one register tile and the
loudness, deadzone, disturbance, both width-weighted norms and the asymmetric
scaling are computed without a single round trip to memory. Only the three
genuine cross-frame dependencies force a barrier: the per-band average over
frames that produces the band power ratio, the FIR smoothing of the frame power
ratio, and the final overlapping sums.

All reductions are two-stage and deterministic — no atomics — so repeated
evaluation is bitwise reproducible.

## Limitations

* float32 CUDA tensors only; the ops raise a descriptive error otherwise.
* Double backward is not supported. `TritonResample` handles it, the other ops
  raise instead of silently returning something wrong.
* `n_fft` must be a power of two of at least 32, and `nbarks` at most 49 (the
  latter is a limitation of `BarkScale` itself, not of the kernels).
* A captured CUDA graph is valid only for the shape it was captured with.

## Accuracy

The Bark tables of the PyTorch implementation are built with scipy and never
cast down, so its perceptual model accidentally runs in float64, while the
Triton backend is float32 throughout. That sounds like it should make the Triton
backend the less accurate of the two. It does not, because the PyTorch backend
inherits a much larger error from somewhere else.

Measured against a float64 evaluation of the *same* pipeline, on the level
alignment filter alone:

| | relative error vs float64 |
|---|---|
| `torchaudio.functional.lfilter`, float32 | 7.6e-3 |
| `TritonIIR`, float32 | 1.7e-7 |

torchaudio evaluates the order-10 Butterworth bandpass as a direct-form-I
recursion, which is badly conditioned; the Triton kernel evaluates it as a
chunked scan in a modal state basis. End to end this makes the Triton backend
roughly **17x closer to the float64 result** than the PyTorch backend on both
distances. In other words, most of the disagreement between the two backends is
the reference being wrong, not the kernels.

Two places needed explicit numerical care:

* **The modal state basis** (see above). The companion matrix of the order-10
  filter is strongly non-normal — `max|A^t|` reaches 1.1e4 — so the float32
  tables of a naive chunked scan overflow. Diagonalising into rotation-scaling
  blocks keeps every table `O(10)`.
* **The loudness curve near the hearing threshold.** Evaluating
  `(0.5 + 0.5 x/th)^e - 1` directly loses the entire mantissa as `x` approaches
  `th`, and bands sitting just above their threshold are the common case in this
  pipeline. The kernels instead form `0.5 (x - th)/th` (exact by Sterbenz'
  lemma there) and evaluate `expm1(e * log1p(.))`, falling back to the plain
  power form far from the threshold where that is the more accurate of the two.

What remains is genuine float32 round off plus the hard decisions in the chain
(the silent-frame threshold, the asymmetric scaling cut at 3.0), where a
rounding difference can flip a single band. See `tests/test_triton_loss.py` for
the tolerances this justifies and `benchmarks/validate_triton.py` for a
comparison of both backends against the ITU-T P.862 reference implementation.
