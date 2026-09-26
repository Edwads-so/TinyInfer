import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from benchmarks.bench_phase2 import (
    RequestAlignment,
    RequestLifecycleTimings,
    assert_token_alignment,
    build_phase2_record,
    build_request_results,
    check_pool_capacity,
    create_shared_paged_cache,
    default_modes,
    run_hf_sequential_workload,
    run_shared_paged_workload,
    select_workload_modes,
    submission_schedule,
    write_phase2_records_atomic,
)
from tinyinfer.config import EngineConfig
from tinyinfer.profiling.phase2 import Phase2Record, Phase2Request, Phase2Workload
from tinyinfer.runtime.result import GenerationResult


def test_main_guard_is_the_last_statement() -> None:
    # Running the file as a script executes main() at the guard, so anything defined
    # after it is missing at runtime even though imports in tests see it.
    tree = ast.parse(Path("benchmarks/bench_phase2.py").read_text(encoding="utf-8"))
    last = tree.body[-1]

    assert isinstance(last, ast.If)
    assert ast.unparse(last.test) == "__name__ == '__main__'"


def _fixed_workload() -> Phase2Workload:
    return Phase2Workload(
        workload_id="fixed",
        family="fixed",
        requests=(
            Phase2Request("r0", "a", "one", 2, 0),
            Phase2Request("r1", "b", "two", 3, 0),
        ),
    )


def _dynamic_workload() -> Phase2Workload:
    return Phase2Workload(
        workload_id="dynamic",
        family="dynamic_mixed",
        requests=(
            Phase2Request("r0", "short", "a", 4, 0),
            Phase2Request("r1", "short", "b", 4, 0),
            Phase2Request("r2", "long", "c", 4, 1),
            Phase2Request("r3", "long", "d", 4, 2),
        ),
    )


def test_continuous_schedule_preserves_deterministic_arrival_iterations() -> None:
    schedule = submission_schedule(_dynamic_workload(), mode="continuous_batch")

    assert [(item.request_id, iteration) for item, iteration in schedule] == [
        ("r0", 0),
        ("r1", 0),
        ("r2", 1),
        ("r3", 2),
    ]


def test_static_schedule_rejects_staggered_workload() -> None:
    try:
        submission_schedule(_dynamic_workload(), mode="static_batch")
    except ValueError as error:
        assert "arrival_iteration 0" in str(error)
    else:
        raise AssertionError("static batch must reject staggered arrivals")


def test_sequential_schedule_submits_at_declared_arrival() -> None:
    schedule = submission_schedule(_fixed_workload(), mode="sequential")

    assert [(item.request_id, iteration) for item, iteration in schedule] == [
        ("r0", 0),
        ("r1", 0),
    ]


def test_lifecycle_timings_measure_latency_from_submission() -> None:
    readings = iter((100, 160, 200, 280))
    timings = RequestLifecycleTimings(clock=lambda: next(readings))

    timings.request_submitted("r0")
    timings.request_admitted("r0")
    timings.first_token_ready("r0")
    timings.request_finished("r0")

    measurement = timings.measurement("r0", output_tokens=5)
    assert measurement.ttft_ms == 0.0001
    assert measurement.end_to_end_ms == 0.00018
    assert measurement.tpot_ms == 0.00002


def test_lifecycle_timings_require_submission() -> None:
    readings = iter((100, 140, 220))
    timings = RequestLifecycleTimings(clock=lambda: next(readings))
    timings.request_admitted("r0")
    timings.first_token_ready("r0")
    timings.request_finished("r0")

    with pytest.raises(ValueError, match="incomplete"):
        timings.measurement("r0", output_tokens=2)


def test_makespan_spans_first_submission_to_last_finish() -> None:
    readings = iter((0, 0, 10, 20, 25, 45))
    timings = RequestLifecycleTimings(clock=lambda: next(readings))
    for request_id in ("r0", "r1"):
        timings.request_submitted(request_id)
    for request_id in ("r0", "r1"):
        timings.first_token_ready(request_id)
        timings.request_finished(request_id)

    assert timings.makespan_ms() == 0.000045


def test_shared_paged_cache_uses_engine_geometry() -> None:
    loaded = SimpleNamespace(
        model=SimpleNamespace(
            device=torch.device("cpu"),
            config=SimpleNamespace(
                num_hidden_layers=2,
                num_key_value_heads=2,
                num_attention_heads=2,
                hidden_size=8,
                max_position_embeddings=64,
            ),
        )
    )

    config = EngineConfig(device="cpu", max_sequence_length=32, kv_block_size=8)
    paged = create_shared_paged_cache(loaded, config)
    larger = create_shared_paged_cache(loaded, config, pool_tokens=33)

    assert paged.block_size == 8
    assert paged.num_physical_blocks == 4
    assert paged.free_block_count == 4
    assert larger.num_physical_blocks == 5


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [1] * len(text)


def test_pool_capacity_counts_concurrent_requests_for_batch_modes() -> None:
    loaded = SimpleNamespace(tokenizer=_CharTokenizer())
    # r0: 3 prompt + 2 new = 5 tokens = 2 blocks; r1: 3 + 3 = 6 tokens = 2 blocks.
    check_pool_capacity(loaded, _fixed_workload(), mode="sequential", block_size=4, pool_blocks=2)
    check_pool_capacity(loaded, _fixed_workload(), mode="static_batch", block_size=4, pool_blocks=4)

    with pytest.raises(ValueError, match="kv-pool-tokens"):
        check_pool_capacity(
            loaded, _fixed_workload(), mode="continuous_batch", block_size=4, pool_blocks=3
        )


def test_hf_sequential_runs_each_request_through_reference_path() -> None:
    calls: list[str] = []

    def fake_generate(loaded: Any, request: Any, observer: Any) -> GenerationResult:
        calls.append(request.request_id)
        observer.request_started()
        observer.first_token_ready()
        observer.request_finished()
        return GenerationResult(request.request_id, (1,), (5, 6), "x", "length")

    execution = run_hf_sequential_workload(
        SimpleNamespace(), _fixed_workload(), generate=fake_generate
    )

    assert calls == ["r0", "r1"]
    assert execution.paged is None
    assert execution.timings.measurement("r1", output_tokens=2).ttft_ms >= 0


def test_hf_sequential_rejects_dynamic_arrivals() -> None:
    with pytest.raises(ValueError, match="hf_sequential"):
        run_hf_sequential_workload(SimpleNamespace(), _dynamic_workload())
    with pytest.raises(ValueError, match="hf_sequential"):
        select_workload_modes(_dynamic_workload(), ("hf_sequential",))


def test_default_modes_include_hf_reference_only_without_staggered_arrivals() -> None:
    assert default_modes(_fixed_workload()) == (
        "hf_sequential",
        "sequential",
        "static_batch",
        "continuous_batch",
    )
    assert default_modes(_dynamic_workload()) == ("sequential", "continuous_batch")


def test_shared_paged_workload_returns_results_and_recovers_pool() -> None:
    class Tokenizer:
        def encode(self, text: str) -> list[int]:
            return [1, 2]

        def __call__(self, prompt: str, **kwargs: object) -> dict[str, torch.Tensor]:
            ids = torch.tensor([[1, 2]], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def decode(self, tokens: list[int], **kwargs: object) -> str:
            return "tokens"

    class Model:
        device = torch.device("cpu")
        config = SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=1,
            num_attention_heads=1,
            hidden_size=4,
            max_position_embeddings=64,
        )

        def __call__(self, **kwargs: object) -> object:
            input_ids = kwargs["input_ids"]
            past = kwargs.get("past_key_values")
            batch = int(input_ids.shape[0])
            if past is not None:
                kv = torch.zeros(batch, 1, 1, 4, dtype=torch.bfloat16)
                past.update(kv, kv.clone(), 0)
            return SimpleNamespace(
                logits=torch.zeros(batch, 1, 10),
                past_key_values=past,
            )

    loaded = SimpleNamespace(model=Model(), tokenizer=Tokenizer(), eos_token_id=9, pad_token_id=9)
    workload = Phase2Workload(
        workload_id="fixed",
        family="fixed",
        requests=(
            Phase2Request("r0", "a", "one", 2, 0),
            Phase2Request("r1", "b", "two", 2, 0),
        ),
    )
    execution = run_shared_paged_workload(
        loaded,
        EngineConfig(device="cpu", max_sequence_length=16, kv_block_size=4),
        workload,
        mode="static_batch",
    )

    assert {result.request_id for result in execution.results} == {"r0", "r1"}
    assert execution.paged.free_block_count == execution.paged.num_physical_blocks
    assert execution.timings.measurement("r0", output_tokens=2).ttft_ms >= 0


def test_alignment_rejects_request_specific_reference_mismatch() -> None:
    results = (
        SimpleNamespace(request_id="r0", output_token_ids=(1, 2)),
        SimpleNamespace(request_id="r1", output_token_ids=(3, 4)),
    )

    try:
        assert_token_alignment(results, lambda request_id: (1, 2) if request_id == "r0" else (9,))
    except RuntimeError as error:
        assert "r1" in str(error)
    else:
        raise AssertionError("mismatched request reference must abort the benchmark")


def test_build_request_results_carries_exact_reference_tokens() -> None:
    results = (SimpleNamespace(request_id="r0", output_token_ids=(1, 2)),)

    records = build_request_results(results, {"r0": RequestAlignment((1, 2), None, None, None)})

    assert records[0].request_id == "r0"
    assert records[0].output_token_ids == (1, 2)
    assert records[0].reference_token_ids == (1, 2)
    assert records[0].token_agreement == 1.0
    assert records[0].first_mismatch_index is None


def _two_request_execution(paged: Any) -> SimpleNamespace:
    # r0: submit 0, first 10 ms, finish 30 ms.  r1: submit 0, first 40 ms, finish 60 ms.
    readings = iter((0, 0, 10_000_000, 30_000_000, 40_000_000, 60_000_000))
    timings = RequestLifecycleTimings(clock=lambda: next(readings))
    timings.request_submitted("r0")
    timings.request_submitted("r1")
    timings.first_token_ready("r0")
    timings.request_finished("r0")
    timings.first_token_ready("r1")
    timings.request_finished("r1")
    return SimpleNamespace(
        results=(
            SimpleNamespace(request_id="r0", output_token_ids=(1, 2)),
            SimpleNamespace(request_id="r1", output_token_ids=(3, 4)),
        ),
        timings=timings,
        paged=paged,
    )


def _record_for(execution: Any, mode: Any) -> Phase2Record:
    return build_phase2_record(
        loaded=SimpleNamespace(model_id="fake", model_revision="revision"),
        workload_id="fixed_b2",
        mode=mode,
        repetition=0,
        warmup_runs=3,
        command=("python", "benchmarks/bench_phase2.py"),
        execution=execution,
        alignments={
            "r0": RequestAlignment((1, 2), None, None, None),
            "r1": RequestAlignment((3, 4), None, None, None),
        },
        git_revision="a" * 40,
        transformers_version="5.14.1",
        torch_version="2.13.0",
        device_name="test-gpu",
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
    )


def test_build_phase2_record_uses_makespan_for_throughput() -> None:
    paged = SimpleNamespace(block_size=4, num_physical_blocks=8, free_block_count=8)
    record = _record_for(_two_request_execution(paged), "sequential")

    assert record.completed_requests == 2
    assert record.generated_tokens == 4
    assert record.makespan_ms == 60.0
    assert record.generated_tokens_per_s == 4000 / 60
    assert record.e2e_p50_ms == 30.0
    assert record.e2e_p95_ms == 60.0
    assert record.ttft_p95_ms == 40.0
    assert record.tpot_p50_ms == 20.0


def test_build_phase2_record_omits_pool_for_hf_reference() -> None:
    record = _record_for(_two_request_execution(None), "hf_sequential")

    assert record.block_size is None
    assert record.pool_total_blocks is None


def test_atomic_writer_emits_one_json_object_per_line(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    record = SimpleNamespace(to_dict=lambda: {"schema_version": 1, "value": 2})

    write_phase2_records_atomic((record,), output)

    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [
        {"schema_version": 1, "value": 2}
    ]
    assert not output.with_name("results.jsonl.tmp").exists()


def test_atomic_writer_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    output.write_text("existing\n", encoding="utf-8")

    try:
        write_phase2_records_atomic((), output)
    except FileExistsError as error:
        assert "--overwrite" in str(error)
    else:
        raise AssertionError("existing evidence must require explicit overwrite")


def test_dynamic_workload_rejects_static_batch_mode() -> None:
    try:
        select_workload_modes(_dynamic_workload(), ("static_batch",))
    except ValueError as error:
        assert "static_batch" in str(error)
    else:
        raise AssertionError("dynamic workload must reject static batch mode")
