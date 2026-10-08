"""Benchmark plumbing for the owned backend and for clean repeated measurements."""

import weakref
from types import SimpleNamespace
from typing import Any

import torch

from benchmarks.bench_phase2 import (
    build_parser,
    create_owned_pool,
    measure_repetitions,
    run_owned_paged_workload,
)
from tinyinfer.config import EngineConfig
from tinyinfer.model.generate import greedy_generate
from tinyinfer.model.qwen2 import Qwen2Model
from tinyinfer.profiling.phase2 import Phase2Request, Phase2Workload
from tinyinfer.runtime.request import GenerationRequest

CPU = torch.device("cpu")


class CharTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        ids = torch.tensor([[ord(char) % 127 + 1 for char in text]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(self, text: str) -> list[int]:
        return [ord(char) % 127 + 1 for char in text]

    def decode(self, tokens: list[int], **_: object) -> str:
        return " ".join(str(token) for token in tokens)


def test_measure_repetitions_releases_each_run_before_the_next() -> None:
    class Execution:
        pass

    alive: list[weakref.ref[Execution]] = []
    events: list[str] = []

    def run() -> Execution:
        assert all(ref() is None for ref in alive), "previous run still alive"
        events.append("run")
        execution = Execution()
        alive.append(weakref.ref(execution))
        return execution

    measured = measure_repetitions(
        run,
        3,
        before_run=lambda: events.append("before"),
        after_run=lambda repetition, execution: repetition,
    )

    assert measured == [0, 1, 2]
    assert events == ["before", "run"] * 3


def test_owned_pool_reports_usable_blocks_without_the_null_block(tiny_qwen2: Any) -> None:
    model = Qwen2Model.from_pretrained(tiny_qwen2.model_dir, device=CPU, dtype=torch.float32)

    pool = create_owned_pool(
        model, EngineConfig(device="cpu", max_sequence_length=32, kv_block_size=8), pool_tokens=33
    )

    assert pool.num_blocks == 6  # 5 usable blocks for 33 tokens, plus the null block
    assert pool.free_block_count == 5


def test_owned_workload_matches_greedy_alone_and_recovers_the_pool(tiny_qwen2: Any) -> None:
    model = Qwen2Model.from_pretrained(tiny_qwen2.model_dir, device=CPU, dtype=torch.float32)
    loaded = SimpleNamespace(tokenizer=CharTokenizer(), eos_token_id=-1)
    workload = Phase2Workload(
        workload_id="fixed",
        family="fixed",
        requests=(
            Phase2Request("r0", "a", "owned forward", 4, 0),
            Phase2Request("r1", "b", "paged", 6, 0),
        ),
    )

    execution = run_owned_paged_workload(
        model,
        loaded,
        EngineConfig(device="cpu", max_sequence_length=64, kv_block_size=4),
        workload,
        mode="static_batch",
    )

    results = {result.request_id: result.output_token_ids for result in execution.results}
    for item in workload.requests:
        request = GenerationRequest(item.request_id, item.prompt, item.max_new_tokens)
        alone = greedy_generate(model, CharTokenizer(), request, eos_token_id=-1)
        assert results[item.request_id] == alone.output_token_ids
    assert execution.paged is not None
    assert execution.paged.free_block_count == execution.paged.num_physical_blocks


def test_parser_selects_the_backend() -> None:
    assert build_parser().parse_args(["--output", "x"]).backend == "hf"
    assert build_parser().parse_args(["--output", "x", "--backend", "owned"]).backend == "owned"
