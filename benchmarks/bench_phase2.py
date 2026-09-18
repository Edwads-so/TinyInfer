"""G2-P1 workload scheduling and benchmark entrypoint.

The full GPU runner is deliberately built on the pure submission schedule below,
so static and continuous mode semantics are testable without CUDA.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.config import EngineConfig
from tinyinfer.profiling.phase2 import (
    Phase2Mode,
    Phase2Record,
    Phase2Request,
    Phase2RequestResult,
    Phase2Workload,
    load_phase2_workloads,
)
from tinyinfer.runtime.generation import reference_generate
from tinyinfer.runtime.hf_loader import load_reference_model
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


def write_phase2_records_atomic(
    records: Sequence[Any], output: Path, *, overwrite: bool = False
) -> None:
    """Write complete JSONL through a sibling temporary file then replace atomically."""
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def select_workload_modes(
    workload: Phase2Workload, modes: Sequence[Phase2Mode]
) -> tuple[Phase2Mode, ...]:
    """Validate that a mode does not rewrite a workload's declared arrivals."""
    selected = tuple(modes)
    if not selected:
        raise ValueError("at least one benchmark mode is required")
    if len(selected) != len(set(selected)):
        raise ValueError("benchmark modes must not contain duplicates")
    if workload.family == "dynamic_mixed" and "static_batch" in selected:
        raise ValueError("dynamic_mixed workload cannot run in static_batch mode")
    return selected


def _git_revision() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _default_modes(workload: Phase2Workload) -> tuple[Phase2Mode, ...]:
    if workload.family == "dynamic_mixed":
        return ("sequential", "continuous_batch")
    return ("sequential", "static_batch", "continuous_batch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the G2-P1 shared-paged-cache benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/qwen2_5_0_5b.yaml"))
    parser.add_argument("--workloads", type=Path, default=Path("benchmarks/phase2_workloads.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("sequential", "static_batch", "continuous_batch"),
        default=None,
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run reproducible G2-P1 evidence collection on an offline CUDA host."""
    from benchmarks.bench_reference import load_engine_config

    args = build_parser().parse_args(argv)
    if args.warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if args.repetitions < 5:
        raise ValueError("repetitions must be at least 5")
    config = load_engine_config(args.config)
    if not config.device.startswith("cuda"):
        raise ValueError("G2-P1 benchmark requires a CUDA configuration")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("G2-P1 benchmark requires an available CUDA device")
    workloads = load_phase2_workloads(args.workloads)
    loaded = load_reference_model(config)
    command = tuple([sys.executable, *sys.argv[1:]])
    records: list[Phase2Record] = []
    for workload in workloads.values():
        modes = (
            select_workload_modes(workload, tuple(args.modes))
            if args.modes is not None
            else _default_modes(workload)
        )
        requests = {
            item.request_id: GenerationRequest(item.request_id, item.prompt, item.max_new_tokens)
            for item in workload.requests
        }
        references = {
            request_id: reference_generate(loaded, request)
            for request_id, request in requests.items()
        }
        for mode in modes:
            for _ in range(args.warmup_runs):
                execution = run_shared_paged_workload(loaded, config, workload, mode=mode)
                assert_token_alignment(execution.results, references.__getitem__)
            for repetition in range(args.repetitions):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                execution = run_shared_paged_workload(loaded, config, workload, mode=mode)
                assert_token_alignment(execution.results, references.__getitem__)
                torch.cuda.synchronize()
                records.append(
                    build_phase2_record(
                        loaded=loaded,
                        workload_id=workload.workload_id,
                        mode=mode,
                        repetition=repetition,
                        warmup_runs=args.warmup_runs,
                        command=command,
                        execution=execution,
                        references=references,
                        git_revision=_git_revision(),
                        transformers_version=importlib.metadata.version("transformers"),
                        torch_version=torch.__version__,
                        device_name=torch.cuda.get_device_name(),
                        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                    )
                )
    write_phase2_records_atomic(records, args.output, overwrite=args.overwrite)
    print(f"G2-P1 BENCHMARK: wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


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
