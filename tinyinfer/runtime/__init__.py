"""Public runtime contracts."""

from tinyinfer.runtime.hf_loader import LoadedModel, ModelLoadError, load_reference_model
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult, StopReason

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "LoadedModel",
    "ModelLoadError",
    "StopReason",
    "load_reference_model",
]
