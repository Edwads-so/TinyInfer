"""Typed engine state and iteration statistics for the batch decode engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tinyinfer.runtime.request import GenerationRequest


@dataclass(slots=True)
class EngineRequestState:
    """Per-request runtime state owned by the batch engine.

    ``cache`` is the backend's opaque handle for this request's KV storage.
    """

    request: GenerationRequest
    cache: Any
    prompt_token_ids: tuple[int, ...]
    output_token_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class IterationStats:
    """Immutable evidence for one scheduling iteration."""

    iteration_id: int
    admitted_count: int
    decode_batch_size: int
    decode_forward_count: int
    running_count_after_step: int
    finished_count: int
    cancelled_count: int
    request_ids: tuple[str, ...]
    old_cache_lengths: tuple[int, ...]
