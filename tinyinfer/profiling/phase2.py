"""Validated workload and raw-record contracts for Phase-2 benchmarks."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tinyinfer._tokens import validate_token_ids

Phase2Family = Literal["fixed", "mixed", "dynamic_mixed"]
Phase2Mode = Literal["hf_sequential", "sequential", "static_batch", "continuous_batch"]
PHASE2_MODES: tuple[Phase2Mode, ...] = (
    "hf_sequential",
    "sequential",
    "static_batch",
    "continuous_batch",
)
SINGLE_REQUEST_MODES: frozenset[Phase2Mode] = frozenset({"hf_sequential", "sequential"})
PHASE2_SCHEMA_VERSION = 2


def _non_blank(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_non_negative(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class Phase2Request:
    request_id: str
    case_id: str
    prompt: str
    max_new_tokens: int
    arrival_iteration: int

    def __post_init__(self) -> None:
        _non_blank("request_id", self.request_id)
        _non_blank("case_id", self.case_id)
        _non_blank("prompt", self.prompt)
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 256:
            raise ValueError("max_new_tokens must be in [1, 256]")
        _non_negative_int("arrival_iteration", self.arrival_iteration)


@dataclass(frozen=True, slots=True)
class Phase2Workload:
    workload_id: str
    family: Phase2Family
    requests: tuple[Phase2Request, ...]

    def __post_init__(self) -> None:
        _non_blank("workload_id", self.workload_id)
        if self.family not in {"fixed", "mixed", "dynamic_mixed"}:
            raise ValueError("family must be fixed, mixed, or dynamic_mixed")
        if not self.requests:
            raise ValueError("requests must not be empty")
        ids = [request.request_id for request in self.requests]
        if len(ids) != len(set(ids)):
            raise ValueError("request_id values must be unique within a workload")
        arrivals = [request.arrival_iteration for request in self.requests]
        if self.family in {"fixed", "mixed"} and any(arrivals):
            raise ValueError(f"{self.family} workloads require arrival_iteration 0")
        if self.family == "dynamic_mixed" and len(set(arrivals)) < 2:
            raise ValueError("dynamic_mixed workloads require staggered arrival_iteration values")


def load_phase2_workloads(path: Path) -> dict[str, Phase2Workload]:
    """Read the strict, versioned Phase-2 workload JSON contract."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Phase-2 workload file: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "workloads"}:
        raise ValueError("workload file must contain only schema_version and workloads")
    if payload["schema_version"] != 1 or not isinstance(payload["workloads"], list):
        raise ValueError("workload file requires schema_version 1 and workloads list")
    workloads: dict[str, Phase2Workload] = {}
    expected = {"workload_id", "family", "requests"}
    request_keys = {"request_id", "case_id", "prompt", "max_new_tokens", "arrival_iteration"}
    for index, raw in enumerate(payload["workloads"]):
        if (
            not isinstance(raw, dict)
            or set(raw) != expected
            or not isinstance(raw["requests"], list)
        ):
            raise ValueError(f"workload {index} has invalid schema")
        requests: list[Phase2Request] = []
        for request_index, item in enumerate(raw["requests"]):
            if not isinstance(item, dict) or set(item) != request_keys:
                raise ValueError(f"workload {index} request {request_index} has invalid schema")
            try:
                requests.append(Phase2Request(**item))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"workload {index} request {request_index} is invalid: {error}"
                ) from error
        try:
            workload = Phase2Workload(
                workload_id=raw["workload_id"], family=raw["family"], requests=tuple(requests)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"workload {index} is invalid: {error}") from error
        if workload.workload_id in workloads:
            raise ValueError("workload_id values must be unique")
        workloads[workload.workload_id] = workload
    if not workloads:
        raise ValueError("workload file must define at least one workload")
    return workloads


def first_mismatch_index(actual: Sequence[int], reference: Sequence[int]) -> int | None:
    """Return the first position where two token sequences differ, or ``None``."""
    for index, (left, right) in enumerate(zip(actual, reference, strict=False)):
        if left != right:
            return index
    if len(actual) != len(reference):
        return min(len(actual), len(reference))
    return None


@dataclass(frozen=True, slots=True)
class Phase2RequestResult:
    """One request's tokens against its B=1 Hugging Face reference.

    An aligned result matches exactly.  A diverged result is allowed only where the
    two tokens were a BF16 tie: ``|divergence_logit_gap| <= divergence_tolerance``,
    the tolerance being one BF16 unit at the reference's top logit.
    ``token_agreement`` is the matching prefix length over the reference length.
    """

    request_id: str
    output_token_ids: tuple[int, ...]
    reference_token_ids: tuple[int, ...]
    token_agreement: float
    first_mismatch_index: int | None
    divergence_logit_gap: float | None
    divergence_tolerance: float | None

    def __post_init__(self) -> None:
        _non_blank("request_id", self.request_id)
        validate_token_ids("output_token_ids", self.output_token_ids, allow_empty=False)
        validate_token_ids("reference_token_ids", self.reference_token_ids, allow_empty=False)
        expected_index = first_mismatch_index(self.output_token_ids, self.reference_token_ids)
        if self.first_mismatch_index != expected_index:
            raise ValueError(f"first_mismatch_index must be {expected_index}")
        if expected_index is None:
            if self.token_agreement != 1.0:
                raise ValueError("token_agreement must be exactly 1.0 for aligned results")
            if self.divergence_logit_gap is not None or self.divergence_tolerance is not None:
                raise ValueError("aligned results carry no divergence fields")
            return
        if expected_index >= min(len(self.output_token_ids), len(self.reference_token_ids)):
            raise ValueError("a length difference is never a BF16 tie divergence")
        if self.token_agreement != expected_index / len(self.reference_token_ids):
            raise ValueError("token_agreement must equal matching prefix over reference length")
        if self.divergence_logit_gap is None or self.divergence_tolerance is None:
            raise ValueError("diverged results require divergence gap and tolerance")
        gap = self.divergence_logit_gap
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap):
            raise ValueError("divergence_logit_gap must be finite")
        tolerance = _finite_non_negative("divergence_tolerance", self.divergence_tolerance)
        if tolerance == 0 or abs(gap) > tolerance:
            raise ValueError("divergence_logit_gap exceeds the BF16 tie tolerance")

    @property
    def diverged(self) -> bool:
        return self.first_mismatch_index is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "output_token_ids": list(self.output_token_ids),
            "reference_token_ids": list(self.reference_token_ids),
            "token_agreement": self.token_agreement,
            "first_mismatch_index": self.first_mismatch_index,
            "divergence_logit_gap": self.divergence_logit_gap,
            "divergence_tolerance": self.divergence_tolerance,
        }


@dataclass(frozen=True, slots=True)
class Phase2Record:
    schema_version: int
    git_revision: str
    model_id: str
    model_revision: str
    transformers_version: str
    torch_version: str
    dtype: str
    device_name: str
    workload_id: str
    mode: Phase2Mode
    repetition: int
    warmup_runs: int
    command: tuple[str, ...]
    block_size: int | None
    num_physical_blocks: int | None
    request_results: tuple[Phase2RequestResult, ...]
    completed_requests: int
    generated_tokens: int
    makespan_ms: float
    requests_per_s: float
    generated_tokens_per_s: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    gpu_utilization_percent: float | None
    pool_free_blocks: int | None
    pool_total_blocks: int | None

    def __post_init__(self) -> None:
        if self.schema_version != PHASE2_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PHASE2_SCHEMA_VERSION}")
        if len(self.git_revision) != 40 or any(
            char not in "0123456789abcdef" for char in self.git_revision
        ):
            raise ValueError("git_revision must be a 40-character lowercase hex SHA")
        for name in (
            "model_id",
            "model_revision",
            "transformers_version",
            "torch_version",
            "dtype",
            "device_name",
            "workload_id",
        ):
            _non_blank(name, getattr(self, name))
        if self.mode not in PHASE2_MODES:
            raise ValueError("mode is invalid")
        if self.mode in SINGLE_REQUEST_MODES and any(
            item.diverged for item in self.request_results
        ):
            raise ValueError(f"{self.mode} records require exact token alignment")
        for name in (
            "repetition",
            "warmup_runs",
            "completed_requests",
            "generated_tokens",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            _non_negative_int(name, getattr(self, name))
        self._validate_pool()
        if not self.command or any(not isinstance(arg, str) or not arg for arg in self.command):
            raise ValueError("command must contain non-blank strings")
        if not self.request_results or self.completed_requests != len(self.request_results):
            raise ValueError("completed_requests must equal non-empty request_results length")
        if self.generated_tokens != sum(
            len(item.output_token_ids) for item in self.request_results
        ):
            raise ValueError("generated_tokens must match request results")
        for name in (
            "makespan_ms",
            "requests_per_s",
            "generated_tokens_per_s",
            "e2e_p50_ms",
            "e2e_p95_ms",
            "ttft_p50_ms",
            "ttft_p95_ms",
            "tpot_p50_ms",
            "tpot_p95_ms",
        ):
            _finite_non_negative(name, getattr(self, name))
        if self.makespan_ms <= 0:
            raise ValueError("makespan_ms must be positive")
        if self.e2e_p95_ms > self.makespan_ms:
            raise ValueError("request end-to-end latency cannot exceed the workload makespan")
        if self.gpu_utilization_percent is not None:
            utilization = _finite_non_negative(
                "gpu_utilization_percent", self.gpu_utilization_percent
            )
            if utilization > 100:
                raise ValueError("gpu_utilization_percent must be at most 100")

    def _validate_pool(self) -> None:
        """Paged modes record their pool; the Hugging Face reference has none."""
        pool = (
            self.block_size,
            self.num_physical_blocks,
            self.pool_free_blocks,
            self.pool_total_blocks,
        )
        if self.mode == "hf_sequential":
            if any(value is not None for value in pool):
                raise ValueError("hf_sequential records must not carry pool geometry")
            return
        if any(value is None for value in pool):
            raise ValueError(f"{self.mode} records require pool geometry")
        block_size, num_physical_blocks, pool_free_blocks, pool_total_blocks = (
            _non_negative_int(name, value)
            for name, value in zip(
                ("block_size", "num_physical_blocks", "pool_free_blocks", "pool_total_blocks"),
                pool,
                strict=True,
            )
        )
        if block_size < 1 or num_physical_blocks < 1 or pool_total_blocks < 1:
            raise ValueError("cache geometry must be positive")
        if pool_free_blocks > pool_total_blocks:
            raise ValueError("pool_free_blocks cannot exceed pool_total_blocks")

    def to_dict(self) -> dict[str, object]:
        return {
            name: (
                list(value)
                if name == "command"
                else [item.to_dict() for item in value]
                if name == "request_results"
                else value
            )
            for name, value in (
                (field, getattr(self, field)) for field in self.__dataclass_fields__
            )
        }

    @classmethod
    def from_dict(cls, payload: object) -> Phase2Record:
        """Reconstruct a strict record from one JSON-compatible payload."""
        if not isinstance(payload, dict):
            raise ValueError("record payload must be an object")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("record payload has unknown or missing keys")
        raw_results = payload["request_results"]
        if not isinstance(raw_results, list):
            raise ValueError("request_results must be a list")
        request_results: list[Phase2RequestResult] = []
        for raw in raw_results:
            if not isinstance(raw, dict) or set(raw) != set(
                Phase2RequestResult.__dataclass_fields__
            ):
                raise ValueError("request result has invalid keys")
            request_results.append(
                Phase2RequestResult(
                    request_id=raw["request_id"],
                    output_token_ids=tuple(raw["output_token_ids"]),
                    reference_token_ids=tuple(raw["reference_token_ids"]),
                    token_agreement=raw["token_agreement"],
                    first_mismatch_index=raw["first_mismatch_index"],
                    divergence_logit_gap=raw["divergence_logit_gap"],
                    divergence_tolerance=raw["divergence_tolerance"],
                )
            )
        values = dict(payload)
        values["command"] = tuple(payload["command"])
        values["request_results"] = tuple(request_results)
        try:
            return cls(**values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid record payload: {error}") from error


@dataclass(frozen=True, slots=True)
class Phase2Summary:
    workload_id: str
    mode: Phase2Mode
    repetitions: int
    makespan_p50_ms: float
    makespan_p95_ms: float
    ttft_p50_ms: float
    tpot_p50_ms: float
    requests_per_s_p50: float
    requests_per_s_p95: float
    generated_tokens_per_s_p50: float
    generated_tokens_per_s_p95: float
    diverged_requests: int


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize no values")
    return ordered[math.ceil(percentile * len(ordered) / 100) - 1]


def summarize_records(records: Sequence[Phase2Record]) -> tuple[Phase2Summary, ...]:
    grouped: dict[tuple[str, Phase2Mode], list[Phase2Record]] = {}
    for record in records:
        grouped.setdefault((record.workload_id, record.mode), []).append(record)
    summaries: list[Phase2Summary] = []
    for (workload_id, mode), group in sorted(grouped.items()):
        summaries.append(
            Phase2Summary(
                workload_id=workload_id,
                mode=mode,
                repetitions=len(group),
                makespan_p50_ms=_nearest_rank([item.makespan_ms for item in group], 50),
                makespan_p95_ms=_nearest_rank([item.makespan_ms for item in group], 95),
                ttft_p50_ms=_nearest_rank([item.ttft_p50_ms for item in group], 50),
                tpot_p50_ms=_nearest_rank([item.tpot_p50_ms for item in group], 50),
                requests_per_s_p50=_nearest_rank([item.requests_per_s for item in group], 50),
                requests_per_s_p95=_nearest_rank([item.requests_per_s for item in group], 95),
                generated_tokens_per_s_p50=_nearest_rank(
                    [item.generated_tokens_per_s for item in group], 50
                ),
                generated_tokens_per_s_p95=_nearest_rank(
                    [item.generated_tokens_per_s for item in group], 95
                ),
                diverged_requests=max(
                    sum(result.diverged for result in item.request_results) for item in group
                ),
            )
        )
    return tuple(summaries)
