from types import SimpleNamespace

import torch

from benchmarks.bench_phase2 import (
    RequestLifecycleTimings,
    create_shared_paged_cache,
    run_shared_paged_workload,
    submission_schedule,
)
from tinyinfer.config import EngineConfig
from tinyinfer.profiling.phase2 import Phase2Request, Phase2Workload


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


def test_lifecycle_timings_calculate_ttft_and_end_to_end() -> None:
    readings = iter((100, 140, 220))
    timings = RequestLifecycleTimings(clock=lambda: next(readings))

    timings.request_admitted("r0")
    timings.first_token_ready("r0")
    timings.request_finished("r0")

    measurement = timings.measurement("r0", output_tokens=5)
    assert measurement.ttft_ms == 0.00004
    assert measurement.end_to_end_ms == 0.00012
    assert measurement.tpot_ms == 0.00002


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

    paged = create_shared_paged_cache(
        loaded,
        EngineConfig(device="cpu", max_sequence_length=32, kv_block_size=8),
    )

    assert paged.block_size == 8
    assert paged.num_physical_blocks == 4
    assert paged.free_block_count == 4


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
