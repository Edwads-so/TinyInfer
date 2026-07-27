"""Public runtime contracts."""

from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult, StopReason

__all__ = ["GenerationRequest", "GenerationResult", "StopReason"]
