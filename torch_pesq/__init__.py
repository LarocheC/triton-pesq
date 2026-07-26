from .bark import *
from .loudness import *
from .loss import PesqLoss

# emit warning for torchaudio < 2.0.0
import torchaudio
from packaging import version
import warnings

if version.parse(torchaudio.__version__) < version.parse("2.0.0"):
    warnings.warn(
        "Your torchaudio seems to be older than version 2.0.0, the energy alignment routine may run slowly. See https://github.com/audiolabs/torch-pesq/issues/3 for further details."
    )


def __getattr__(name):
    """Import the Triton backend lazily, it is optional and needs a GPU."""

    if name == "PesqLossTriton":
        from .triton_ops.loss import PesqLossTriton

        return PesqLossTriton

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PesqLoss",
    "PesqLossTriton",
    "BarkScale",
    "Loudness",
]
