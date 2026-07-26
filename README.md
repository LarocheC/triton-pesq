# Loss function inspired by the PESQ score

![Testing badge](https://github.com/audiolabs/torch-pesq/actions/workflows/test.yaml/badge.svg) 
![Linting badge](https://github.com/audiolabs/torch-pesq/actions/workflows/black.yaml/badge.svg) 
![Docs badge](https://github.com/audiolabs/torch-pesq/actions/workflows/docs.yaml/badge.svg)

Implementation of the widely used Perceptual Evaluation of Speech Quality (PESQ) score as a torch loss function. The PESQ loss alone performs not good for noise suppression, instead combine with scale invariant [SDR](https://arxiv.org/abs/1811.02508). For more information see [1],[2].

## Installation

To install the package just run:
```bash
$ pip install torch-pesq
```

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

The loss agrees with the PyTorch implementation to ~1e-6 relative. It is in fact
the *more* accurate of the two: measured against a float64 evaluation of the
same pipeline it is ~17x closer, because torchaudio's float32 direct-form-I
recursion is poorly conditioned for the order 10 level alignment filter.

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
