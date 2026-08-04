"""Public runtime contracts."""

from tinyinfer.runtime.generation import (
    DecodeOutput,
    GenerationObserver,
    PrefillOutput,
    decode_one,
    generate_greedy,
    generate_with_owned_cache,
    generate_with_paged_cache,
    prefill,
    reference_generate,
)
from tinyinfer.runtime.hf_loader import LoadedModel, ModelLoadError, load_reference_model
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult, StopReason

__all__ = [
    "DecodeOutput",
    "GenerationObserver",
    "GenerationRequest",
    "GenerationResult",
    "LoadedModel",
    "ModelLoadError",
    "PrefillOutput",
    "StopReason",
    "decode_one",
    "generate_greedy",
    "generate_with_owned_cache",
    "generate_with_paged_cache",
    "load_reference_model",
    "prefill",
    "reference_generate",
]
