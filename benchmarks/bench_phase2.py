"""G2-P1 workload scheduling and benchmark entrypoint.

The full GPU runner is deliberately built on the pure submission schedule below,
so static and continuous mode semantics are testable without CUDA.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.config import EngineConfig
from tinyinfer.profiling.phase2 import (
    Phase2Mode,
    Phase2Record,
    Phase2Request,
    Phase2RequestResult,
    Phase2Workload,
)
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


def build_request_results(
    results: Sequence[Any], references: dict[str, tuple[int, ...]]
) -> tuple[Phase2RequestResult, ...]:
    """Convert aligned runtime outputs to immutable raw-record request entries."""
    entries: list[Phase2RequestResult] = []
    for result in results:
        try:
            reference = references[result.request_id]
        except KeyError as error:
            raise RuntimeError(f"missing reference tokens for {result.request_id}") from error
        actual = tuple(result.output_token_ids)
        if actual != reference:
            raise RuntimeError(f"unaligned result for {result.request_id}")
        entries.append(
            Phase2RequestResult(
                request_id=result.request_id,
                output_token_ids=actual,
                reference_token_ids=reference,
                token_agreement=1.0,
                first_mismatch_index=None,
            )
        )
    return tuple(entries)


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    if not values:
        raise ValueError("cannot calculate percentile from no values")
    return sorted(values)[ceil(percentile * len(values) / 100) - 1]


def build_phase2_record(
    *,
    loaded: Any,
    workload_id: str,
    mode: Phase2Mode,
    repetition: int,
    warmup_runs: int,
    command: tuple[str, ...],
    execution: Any,
    references: dict[str, tuple[int, ...]],
    git_revision: str,
    transformers_version: str,
    torch_version: str,
    device_name: str,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> Phase2Record:
    """Build one schema-validated raw benchmark record from a completed workload."""
    request_results = build_request_results(execution.results, references)
    measurements = [
        execution.timings.measurement(item.request_id, output_tokens=len(item.output_token_ids))
        for item in request_results
    ]
    end_to_end_ms = max(item.end_to_end_ms for item in measurements)
    generated_tokens = sum(len(item.output_token_ids) for item in request_results)
    return Phase2Record(
        schema_version=1,
        git_revision=git_revision,
        model_id=loaded.model_id,
        model_revision=loaded.model_revision,
        transformers_version=transformers_version,
        torch_version=torch_version,
        dtype="bfloat16",
        device_name=device_name,
        workload_id=workload_id,
        mode=mode,
        repetition=repetition,
        warmup_runs=warmup_runs,
        command=command,
        block_size=execution.paged.block_size,
        num_physical_blocks=execution.paged.num_physical_blocks,
        request_results=request_results,
        completed_requests=len(request_results),
        generated_tokens=generated_tokens,
        end_to_end_ms=end_to_end_ms,
        requests_per_s=1000 * len(request_results) / end_to_end_ms,
        generated_tokens_per_s=1000 * generated_tokens / end_to_end_ms,
        ttft_p50_ms=_nearest_rank([item.ttft_ms for item in measurements], 50),
        ttft_p95_ms=_nearest_rank([item.ttft_ms for item in measurements], 95),
        tpot_p50_ms=_nearest_rank([item.tpot_ms for item in measurements], 50),
        tpot_p95_ms=_nearest_rank([item.tpot_ms for item in measurements], 95),
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        gpu_utilization_percent=None,
        pool_free_blocks=execution.paged.free_block_count,
        pool_total_blocks=execution.paged.num_physical_blocks,
    )


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
