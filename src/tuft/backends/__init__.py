from typing import TYPE_CHECKING

from .sampling_backend import BaseSamplingBackend, DPSamplingBackend, VLLMSamplingBackend
from .training_backend import BaseTrainingBackend, HFTrainingBackend


if TYPE_CHECKING:
    from .fsdp_training_backend import FSDPTrainingBackend

__all__ = [
    "BaseSamplingBackend",
    "DPSamplingBackend",
    "VLLMSamplingBackend",
    "BaseTrainingBackend",
    "HFTrainingBackend",
    "FSDPTrainingBackend",
]


def __getattr__(name: str):
    # Lazy import: the FSDP backend pulls in torch.distributed/peft machinery that
    # only 'fsdp'-backend deployments need; keep it off the default import path.
    if name == "FSDPTrainingBackend":
        from .fsdp_training_backend import FSDPTrainingBackend

        return FSDPTrainingBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
