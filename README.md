# triton-pesq — a Triton GPU backend for the PESQ loss

> **This is a derivative of [audiolabs/torch-pesq](https://github.com/audiolabs/torch-pesq).**
> The PESQ model, the Bark filterbank, the loudness model and the reference
> PyTorch implementation are the work of **Lorenz Schmidt, Nils Werner and
> Nils Peters** at the International Audio Laboratories Erlangen and FAU
> Erlangen-Nürnberg, published under the MIT license. This project adds a
> Triton GPU backend on top of it and changes nothing about the model itself.
> See [Credits](#credits).

Implementation of the widely used Perceptual Evaluation of Speech Quality (PESQ) score as a torch loss function. The PESQ loss alone performs not good for noise suppression, instead combine with scale invariant [SDR](https://arxiv.org/abs/1811.02508). For more information see [1],[2].

`PesqLoss` is the original PyTorch implementation, unchanged. `PesqLossTriton`
is a drop-in replacement whose every stage, forward and backward, is a Triton
kernel — 60x faster in the forward pass and up to 181x for forward and backward
together, and closer to a float64 evaluation of the same pipeline than the
original is.

## Installation

The import path is deliberately still `torch_pesq`, so this drops into code
written against the original:

```bash
$ git clone https://github.com/LarocheC/triton-pesq && pip install -e triton-pesq
```

The upstream PyTorch-only package remains available as `pip install torch-pesq`.
Do not install both into the same environment, they provide the same module.

## Usage

```python
import torch
from torch_pesq import PesqLoss

pesq = PesqLoss(0.5,
    sample_rate=44100, 
)

mos = pesq.mos(reference, degraded)
loss = pesq(reference, degraded)

print(mos, loss)
loss.backward()
```

## Triton backend

`PesqLossTriton` is a drop-in replacement in which every stage of the pipeline,
forward and backward, is a Triton GPU kernel:

```python
from torch_pesq import PesqLossTriton

pesq = PesqLossTriton(0.5, sample_rate=16000).cuda()
loss = pesq(reference, degraded)   # float32 CUDA tensors
loss.sum().backward()
```

Measured on a GTX 1080 Ti with 16 kHz input, synchronised wall clock, median of
15 runs (`python benchmarks/bench_triton.py`):

| batch | audio | PyTorch | Triton | speedup | Triton + CUDA graph | speedup |
|------:|------:|--------:|-------:|--------:|--------------------:|--------:|
| **forward** | | | | | | |
| 1 | 1 s | 41.6 ms | 0.688 ms | **60x** | | |
| 8 | 8 s | 43.9 ms | 0.714 ms | **62x** | | |
| **forward + backward** | | | | | | |
| 1 | 1 s | 84.1 ms | 5.83 ms | 14x | 0.465 ms | **181x** |
| 8 | 8 s | 87.3 ms | 6.03 ms | 14x | 0.868 ms | **101x** |

Almost all of the PyTorch time is `torchaudio.functional.lfilter`, which has no
CUDA kernel and falls back to a Python loop over time samples; the Triton
backend replaces it with a chunked parallel scan. After that the pipeline is so
fast that it becomes bound by the host side cost of launching its ~40 small
kernels, which is why replaying it from a CUDA graph is worth another 7x on the
training path:

```python
step = pesq.graphed(reference, degraded)   # fixes the input shape
loss = step(reference, degraded)           # bit exact with the eager call
```

The two backends do not agree to float32 round off, and the Triton one is not
the reason. Measured against a float64 evaluation of the same pipeline, over a
sweep of signal types, sample rates and lengths:

| | relative error of the distances vs float64 |
|---|---|
| `PesqLoss`, float32 | 1e-7 … 4e-3, typically ~1e-4 |
| `PesqLossTriton`, float32 | 1e-8 … 6e-6, typically ~1e-7 |

so the two backends differ from *each other* by whatever the first row is,
usually ~1e-4 and up to a few times 1e-3 on signals that stress the level
alignment. Essentially all of it is `torchaudio.functional.lfilter`: it
evaluates the order 10 level alignment filter as a float32 direct-form-I
recursion, which is badly conditioned and off by ~1e-2 on the filtered signal
itself. The Triton backend is the one to trust here, by two to three orders of
magnitude.

One case *does* agree to ~1e-6, and it is worth knowing which: signals whose
per frame distortion sits at the `45.0` clamp — heavily degraded narrowband
speech, or any purely harmonic reference with additive noise. Both distances
are then pinned to the clamp and the two backends cannot do anything else but
agree.

See [docs/triton.md](docs/triton.md) for the design, `benchmarks/bench_triton.py`
for the numbers above and `benchmarks/validate_triton.py` for a comparison of
both backends against the ITU-T P.862 reference.

## Comparison to reference implementation

The following figures uses samples from the VCTK [1] speech and DEMAND [2] noise dataset with varying mixing factors. They illustrate correlation and maximum error between the reference and torch implementation:

![Correlation](https://raw.githubusercontent.com/audiolabs/torch-pesq/main/figures/compare_reference.png)

The difference is a result from missing time alignment implementation and a level alignment done with IIR filtering instead of a frequency weighting. They are minor and should not be significant when used as a loss function. There are two outliers which may degrade results and further investigation is needed to find the source of difference.

## Validation improvements when used as loss function

Validation results for fullband noise suppression:
 - Noise estimator: Recurrent [SRU](https://github.com/asappresearch/sru) with soft masking. 8 layers, width of 512 result in ~1586k parameters of the unpruned model.
 - STFT for signal coding: 512 window length, 50% overlap, hamming window
 - Mel filterbank with 32 Mel features

The baseline system uses L1 time domain loss. Combining the PESQ loss function together with scale invariant [SDR](https://arxiv.org/abs/1811.02508) gives improvement of ~0.1MOS for PESQ and slight improvements in speech distortions, as well as a more stable training progression. Horizontal lines indicate the score of noisy speech.

![Validation comparison](https://raw.githubusercontent.com/audiolabs/torch-pesq/main/figures/validation.svg)

## Credits

This project would not exist without **[audiolabs/torch-pesq](https://github.com/audiolabs/torch-pesq)**,
by **Lorenz Schmidt**, **Nils Werner** and **Nils Peters** (International Audio
Laboratories Erlangen, a joint institution of Friedrich-Alexander-Universität
Erlangen-Nürnberg and Fraunhofer IIS), with a further contribution from
**Moreno La Quatra**. It is MIT licensed and the full upstream git history,
with every original commit and its author, is preserved in this repository.

Everything that makes the score a *score* is theirs: the port of the ITU-T
P.862 model to PyTorch, the Bark filterbank, the loudness model, the
disturbance and asymmetry processing, the parameter tables, and the validation
against the reference implementation. `torch_pesq/bark.py`,
`torch_pesq/loudness.py` and `torch_pesq/loss.py` are their code, unmodified.
`torch_pesq/triton_ops/reference.py` is a rearrangement of their `loss.py` into
per-stage functions, used as the oracle the kernels are tested against.

What this fork adds is `torch_pesq/triton_ops/`: an alternative execution
backend written in Triton. It changes no model behaviour, and every stage is
verified against their code, evaluated in float64, to single precision — see
[Triton backend](#triton-backend) for what that does and does not mean for the
agreement between the two backends.

If you use this in academic work, please cite the original authors' paper:

```bibtex
@article{schmidt2023torchpesq,
  title   = {Torch PESQ - a PyTorch implementation of the Perceptual
             Evaluation of Speech Quality},
  author  = {Schmidt, Lorenz and Werner, Nils and Peters, Nils},
  year    = {2023},
  note    = {International Audio Laboratories Erlangen},
  url     = {https://github.com/audiolabs/torch-pesq}
}
```

The PESQ algorithm itself is ITU-T Recommendation P.862 [3][4]; this is an
independent implementation of it and is not endorsed by or affiliated with the
ITU, the International Audio Laboratories Erlangen, or the original authors.

## Relevant references
1. [End-to-End Multi-Task Denoising for joint SDR and PESQ Optimization](https://arxiv.org/abs/1901.09146)
2. [A Deep Learning Loss Function Based on the Perceptual Evaluation of the Speech Quality](https://ieeexplore.ieee.org/document/8468124)
3. [P.862 : Perceptual evaluation of speech quality (PESQ)](https://www.itu.int/rec/T-REC-P.862)
4. [Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs](https://ieeexplore.ieee.org/document/941023)
5. [CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit](https://datashare.ed.ac.uk/handle/10283/2950)
6. [The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND): A database of multichannel environmental noise recordings](https://asa.scitation.org/doi/abs/10.1121/1.4799597)

[1]: https://arxiv.org/abs/1901.09146
[2]: https://ieeexplore.ieee.org/document/8468124
[3]: https://www.itu.int/rec/T-REC-P.862
[4]: https://ieeexplore.ieee.org/document/941023
[5]: https://datashare.ed.ac.uk/handle/10283/2950
[6]: https://asa.scitation.org/doi/abs/10.1121/1.4799597
