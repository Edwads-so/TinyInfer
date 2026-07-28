"""Profiling primitives for the reference inference runtime."""

from tinyinfer.profiling.record import (
    BenchmarkRecord,
    TokenComparison,
    compare_token_ids,
)
from tinyinfer.profiling.timer import InferenceTimer, PhaseTimings

__all__ = [
    "BenchmarkRecord",
    "InferenceTimer",
    "PhaseTimings",
    "TokenComparison",
    "compare_token_ids",
]
