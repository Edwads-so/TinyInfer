from typing import Any

import pytest

from tinyinfer.profiling.phase2 import (
    Phase2Record,
    Phase2RequestResult,
    summarize_records,
)


def _request_result() -> Phase2RequestResult:
    return Phase2RequestResult(
        request_id="r0",
        output_token_ids=(1, 2),
        reference_token_ids=(1, 2),
        token_agreement=1.0,
        first_mismatch_index=None,
        divergence_logit_gap=None,
        divergence_tolerance=None,
    )


def _record(
    *,
    repetition: int,
    makespan_ms: float,
    mode: str = "sequential",
    pooled: bool = True,
    results: tuple[Phase2RequestResult, ...] | None = None,
) -> Phase2Record:
    request_results = (_request_result(),) if results is None else results
    return Phase2Record(
        schema_version=2,
        git_revision="a" * 40,
        model_id="Qwen/Qwen2.5-0.5B",
        model_revision="main",
        transformers_version="5.14.1",
        torch_version="2.13.0",
        dtype="bfloat16",
        device_name="test-gpu",
        workload_id="fixed_b1",
        mode=mode,  # type: ignore[arg-type]
        repetition=repetition,
        warmup_runs=3,
        command=("python", "benchmarks/bench_phase2.py"),
        block_size=16 if pooled else None,
        num_physical_blocks=64 if pooled else None,
        request_results=request_results,
        completed_requests=len(request_results),
        generated_tokens=sum(len(item.output_token_ids) for item in request_results),
        makespan_ms=makespan_ms,
        requests_per_s=1000 / makespan_ms,
        generated_tokens_per_s=2000 / makespan_ms,
        e2e_p50_ms=makespan_ms,
        e2e_p95_ms=makespan_ms,
        ttft_p50_ms=makespan_ms / 2,
        ttft_p95_ms=makespan_ms / 2,
        tpot_p50_ms=makespan_ms / 4,
        tpot_p95_ms=makespan_ms / 4,
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
        gpu_utilization_percent=None,
        pool_free_blocks=64 if pooled else None,
        pool_total_blocks=64 if pooled else None,
    )


def test_phase2_record_rejects_divergence_without_tie_evidence() -> None:
    with pytest.raises(ValueError, match="divergence"):
        Phase2RequestResult(
            request_id="r0",
            output_token_ids=(1,),
            reference_token_ids=(2,),
            token_agreement=0.0,
            first_mismatch_index=0,
            divergence_logit_gap=None,
            divergence_tolerance=None,
        )


def test_summary_uses_nearest_rank_percentiles() -> None:
    records = tuple(
        _record(repetition=index, makespan_ms=value)
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 100.0))
    )

    summary = summarize_records(records)[0]

    assert summary.makespan_p50_ms == 3.0
    assert summary.makespan_p95_ms == 100.0
    assert summary.ttft_p50_ms == 1.5
    assert summary.tpot_p50_ms == 0.75


def test_phase2_record_serializes_json_values() -> None:
    payload: dict[str, Any] = _record(repetition=0, makespan_ms=1.0).to_dict()

    assert payload["command"] == ["python", "benchmarks/bench_phase2.py"]
    assert payload["request_results"][0]["output_token_ids"] == [1, 2]
    assert payload["gpu_utilization_percent"] is None


def test_hf_sequential_record_has_no_paged_pool() -> None:
    record = _record(repetition=0, makespan_ms=1.0, mode="hf_sequential", pooled=False)

    assert record.block_size is None
    assert record.to_dict()["pool_total_blocks"] is None


def test_paged_mode_record_requires_pool_geometry() -> None:
    with pytest.raises(ValueError, match="pool geometry"):
        _record(repetition=0, makespan_ms=1.0, mode="continuous_batch", pooled=False)


def test_hf_sequential_record_rejects_pool_geometry() -> None:
    with pytest.raises(ValueError, match="pool geometry"):
        _record(repetition=0, makespan_ms=1.0, mode="hf_sequential", pooled=True)


def test_record_rejects_request_latency_beyond_makespan() -> None:
    payload: dict[str, Any] = _record(repetition=0, makespan_ms=1.0).to_dict()
    payload["e2e_p95_ms"] = 2.0

    with pytest.raises(ValueError, match="makespan"):
        Phase2Record.from_dict(payload)


def _diverged_result(**overrides: Any) -> Phase2RequestResult:
    values: dict[str, Any] = {
        "request_id": "r1",
        "output_token_ids": (1, 2, 9),
        "reference_token_ids": (1, 2, 3),
        "token_agreement": 2 / 3,
        "first_mismatch_index": 2,
        "divergence_logit_gap": 0.0,
        "divergence_tolerance": 0.125,
    }
    values.update(overrides)
    return Phase2RequestResult(**values)


def test_request_result_accepts_bf16_tie_divergence() -> None:
    result = _diverged_result()

    assert result.to_dict()["divergence_tolerance"] == 0.125


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"divergence_logit_gap": 0.25}, "tolerance"),
        ({"first_mismatch_index": 1}, "first_mismatch_index"),
        ({"token_agreement": 1.0}, "token_agreement"),
        ({"divergence_logit_gap": None}, "divergence"),
    ],
)
def test_request_result_rejects_inconsistent_divergence(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _diverged_result(**overrides)


def test_aligned_request_result_carries_no_divergence() -> None:
    with pytest.raises(ValueError, match="divergence"):
        Phase2RequestResult("r0", (1, 2), (1, 2), 1.0, None, 0.0, 0.125)


@pytest.mark.parametrize("mode", ["hf_sequential", "sequential"])
def test_single_request_modes_reject_diverged_results(mode: str) -> None:
    with pytest.raises(ValueError, match="exact"):
        _record(
            repetition=0,
            makespan_ms=1.0,
            mode=mode,
            pooled=mode != "hf_sequential",
            results=(_diverged_result(),),
        )


def test_summary_reports_diverged_request_results() -> None:
    records = tuple(
        _record(
            repetition=index,
            makespan_ms=1.0,
            mode="static_batch",
            results=(_diverged_result(),),
        )
        for index in range(5)
    )

    assert summarize_records(records)[0].diverged_requests == 1
