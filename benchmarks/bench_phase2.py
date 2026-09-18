"""G2-P1 workload scheduling and benchmark entrypoint.

The full GPU runner is deliberately built on the pure submission schedule below,
so static and continuous mode semantics are testable without CUDA.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from tinyinfer.profiling.phase2 import Phase2Mode, Phase2Request, Phase2Workload


@dataclass(frozen=True, slots=True)
class RequestMeasurement:
    """One request's host-side lifecycle measurements in milliseconds."""

    ttft_ms: float
    end_to_end_ms: float
    tpot_ms: float


class RequestLifecycleTimings:
    """Clock-backed ``BatchEngineObserver`` used only by the benchmark runner."""

    def __init__(self, *, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self._clock = clock
        self._admitted_ns: dict[str, int] = {}
        self._first_token_ns: dict[str, int] = {}
        self._finished_ns: dict[str, int] = {}

    def request_admitted(self, request_id: str) -> None:
        self._admitted_ns[request_id] = self._clock()

    def first_token_ready(self, request_id: str) -> None:
        self._first_token_ns[request_id] = self._clock()

    def request_finished(self, request_id: str) -> None:
        self._finished_ns[request_id] = self._clock()

    def measurement(self, request_id: str, *, output_tokens: int) -> RequestMeasurement:
        if type(output_tokens) is not int or output_tokens < 1:
            raise ValueError("output_tokens must be positive")
        try:
            admitted = self._admitted_ns[request_id]
            first_token = self._first_token_ns[request_id]
            finished = self._finished_ns[request_id]
        except KeyError as error:
            raise ValueError(f"incomplete lifecycle timestamps for {request_id!r}") from error
        if not admitted <= first_token <= finished:
            raise ValueError(f"non-monotonic lifecycle timestamps for {request_id!r}")
        ttft_ms = (first_token - admitted) / 1_000_000
        end_to_end_ms = (finished - admitted) / 1_000_000
        tpot_ms = (
            0.0
            if output_tokens == 1
            else (finished - first_token) / 1_000_000 / (output_tokens - 1)
        )
        return RequestMeasurement(ttft_ms=ttft_ms, end_to_end_ms=end_to_end_ms, tpot_ms=tpot_ms)


def submission_schedule(
    workload: Phase2Workload, *, mode: Phase2Mode
) -> tuple[tuple[Phase2Request, int], ...]:
    """Return deterministic request submission points for one benchmark mode."""
    ordered = tuple(
        sorted(
            workload.requests, key=lambda request: (request.arrival_iteration, request.request_id)
        )
    )
    if mode == "sequential":
        return tuple((request, index) for index, request in enumerate(ordered))
    if mode == "static_batch":
        if any(request.arrival_iteration != 0 for request in ordered):
            raise ValueError("static_batch requires arrival_iteration 0 for every request")
        return tuple((request, 0) for request in ordered)
    if mode == "continuous_batch":
        return tuple((request, request.arrival_iteration) for request in ordered)
    raise ValueError(f"unsupported mode: {mode}")
