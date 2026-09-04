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
    )


def _record(*, repetition: int, end_to_end_ms: float) -> Phase2Record:
    return Phase2Record(
        schema_version=1,
        git_revision="a" * 40,
        model_id="Qwen/Qwen2.5-0.5B",
        model_revision="main",
        transformers_version="5.14.1",
        torch_version="2.13.0",
        dtype="bfloat16",
        device_name="test-gpu",
        workload_id="fixed_b1",
        mode="sequential",
        repetition=repetition,
        warmup_runs=3,
        command=("python", "benchmarks/bench_phase2.py"),
        block_size=16,
        num_physical_blocks=64,
        request_results=(_request_result(),),
        completed_requests=1,
        generated_tokens=2,
        end_to_end_ms=end_to_end_ms,
        requests_per_s=1000 / end_to_end_ms,
        generated_tokens_per_s=2000 / end_to_end_ms,
        ttft_p50_ms=end_to_end_ms / 2,
        ttft_p95_ms=end_to_end_ms / 2,
        tpot_p50_ms=end_to_end_ms / 4,
        tpot_p95_ms=end_to_end_ms / 4,
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
        gpu_utilization_percent=None,
        pool_free_blocks=64,
        pool_total_blocks=64,
    )


def test_phase2_record_rejects_nonexact_token_agreement() -> None:
    with pytest.raises(ValueError, match="token_agreement"):
        Phase2RequestResult(
            request_id="r0",
            output_token_ids=(1,),
            reference_token_ids=(2,),
            token_agreement=0.0,
            first_mismatch_index=0,
        )


def test_summary_uses_nearest_rank_percentiles() -> None:
    records = tuple(
        _record(repetition=index, end_to_end_ms=value)
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 100.0))
    )

    summary = summarize_records(records)[0]

    assert summary.end_to_end_p50_ms == 3.0
    assert summary.end_to_end_p95_ms == 100.0


def test_phase2_record_serializes_json_values() -> None:
    payload: dict[str, Any] = _record(repetition=0, end_to_end_ms=1.0).to_dict()

    assert payload["command"] == ["python", "benchmarks/bench_phase2.py"]
    assert payload["request_results"][0]["output_token_ids"] == [1, 2]
    assert payload["gpu_utilization_percent"] is None
