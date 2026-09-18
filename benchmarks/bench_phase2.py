"""G2-P1 workload scheduling and benchmark entrypoint.

The full GPU runner is deliberately built on the pure submission schedule below,
so static and continuous mode semantics are testable without CUDA.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.config import EngineConfig
from tinyinfer.profiling.phase2 import Phase2Mode, Phase2Request, Phase2Workload
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult
from tinyinfer.scheduler.engine import BatchEngine


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


def create_shared_paged_cache(loaded: Any, config: EngineConfig) -> PagedKVCache:
    """Allocate one shared paged pool using the benchmark's declared geometry."""
    import importlib

    model_config = loaded.model.config
    num_kv_heads = int(
        getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
    )
    torch = importlib.import_module("torch")
    return PagedKVCache(
        num_layers=int(model_config.num_hidden_layers),
        num_kv_heads=num_kv_heads,
        block_size=config.kv_block_size,
        head_dim=int(model_config.hidden_size // model_config.num_attention_heads),
        num_physical_blocks=(config.max_sequence_length + config.kv_block_size - 1)
        // config.kv_block_size,
        device=loaded.model.device,
        dtype=torch.bfloat16,
    )


@dataclass(frozen=True, slots=True)
class WorkloadExecution:
    """Results and lifecycle timestamps from one shared-pool workload run."""

    results: tuple[GenerationResult, ...]
    timings: RequestLifecycleTimings
    paged: PagedKVCache


def run_shared_paged_workload(
    loaded: Any,
    config: EngineConfig,
    workload: Phase2Workload,
    *,
    mode: Phase2Mode,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> WorkloadExecution:
    """Execute a workload with one shared pool and verify all blocks return."""
    schedule = submission_schedule(workload, mode=mode)
    timings = RequestLifecycleTimings(clock=clock)
    paged = create_shared_paged_cache(loaded, config)
    max_batch = 1 if mode == "sequential" else len(workload.requests)
    engine = BatchEngine(loaded, paged, max_batch=max_batch, observer=timings)
    pending_index = 0
    iteration = 0
    while pending_index < len(schedule) or not engine.queue.all_done():
        while pending_index < len(schedule) and schedule[pending_index][1] <= iteration:
            request, _ = schedule[pending_index]
            engine.submit(
                GenerationRequest(request.request_id, request.prompt, request.max_new_tokens)
            )
            pending_index += 1
        engine.step()
        iteration += 1
    results = tuple(engine.queue.drain_finished())
    expected_ids = {request.request_id for request in workload.requests}
    if {result.request_id for result in results} != expected_ids or len(results) != len(
        expected_ids
    ):
        raise RuntimeError("workload did not produce exactly one result per request")
    if paged.free_block_count != paged.num_physical_blocks:
        raise RuntimeError("shared paged pool did not fully recover")
    return WorkloadExecution(results=results, timings=timings, paged=paged)


def assert_token_alignment(
    results: Sequence[Any], reference_tokens: Callable[[str], tuple[int, ...]]
) -> dict[str, tuple[int, ...]]:
    """Require every request output to equal its own deterministic reference."""
    references: dict[str, tuple[int, ...]] = {}
    for result in results:
        request_id = result.request_id
        actual = tuple(result.output_token_ids)
        reference = reference_tokens(request_id)
        if actual != reference:
            raise RuntimeError(
                f"token mismatch for {request_id}: TinyInfer={actual}, HuggingFace={reference}"
            )
        references[request_id] = reference
    return references


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
