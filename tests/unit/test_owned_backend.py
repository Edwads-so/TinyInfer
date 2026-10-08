"""BatchEngine driving the owned forward pass on the paged pool."""

from typing import Any

import pytest
import torch

from tinyinfer.model.generate import greedy_generate
from tinyinfer.model.paged import PagedKVPool
from tinyinfer.model.qwen2 import Qwen2Model
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.scheduler.backends import OwnedPagedBackend
from tinyinfer.scheduler.engine import BatchEngine

CPU = torch.device("cpu")
REQUESTS = (
    GenerationRequest("short", "kv", 6),
    GenerationRequest("medium", "paged attention", 3),
    GenerationRequest("long", "continuous batching on one gpu", 5),
)


class CharTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        ids = torch.tensor([[ord(char) % 127 + 1 for char in text]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, tokens: list[int], **_: object) -> str:
        return " ".join(str(token) for token in tokens)


@pytest.fixture
def model(tiny_qwen2: Any) -> Qwen2Model:
    return Qwen2Model.from_pretrained(tiny_qwen2.model_dir, device=CPU, dtype=torch.float32)


def _backend(model: Qwen2Model, num_blocks: int = 32) -> OwnedPagedBackend:
    pool = PagedKVPool.for_spec(
        model.spec, block_size=4, num_blocks=num_blocks, device=CPU, dtype=model.dtype
    )
    pool.buffer[:, :, 1:] = float("nan")
    return OwnedPagedBackend(model, CharTokenizer(), pool, eos_token_id=-1)


def _alone(model: Qwen2Model, request: GenerationRequest) -> tuple[int, ...]:
    return greedy_generate(model, CharTokenizer(), request, eos_token_id=-1).output_token_ids


@pytest.mark.parametrize("max_batch", [1, 2, 3], ids=["sequential", "dynamic", "all-at-once"])
def test_engine_outputs_match_each_request_generated_alone(
    model: Qwen2Model, max_batch: int
) -> None:
    backend = _backend(model)
    engine = BatchEngine.from_backend(backend, max_batch=max_batch)
    for request in REQUESTS:
        engine.submit(request)

    results = {result.request_id: result for result in engine.run()}

    for request in REQUESTS:
        assert results[request.request_id].output_token_ids == _alone(model, request)
    assert backend.pool.free_block_count == backend.pool.num_blocks - 1
    batch_sizes = [stats.decode_batch_size for stats in engine.iteration_history]
    assert max(batch_sizes) == max_batch
    assert all(stats.decode_forward_count <= 1 for stats in engine.iteration_history)


def test_finished_request_frees_its_slot_for_a_waiting_one(model: Qwen2Model) -> None:
    engine = BatchEngine.from_backend(_backend(model), max_batch=2)
    for request in REQUESTS:
        engine.submit(request)

    engine.run()

    batches = [stats.request_ids for stats in engine.iteration_history if stats.request_ids]
    assert ("short", "medium") in batches  # the first two run together
    assert any("long" in batch and "medium" not in batch for batch in batches)


def test_cancel_returns_blocks_to_the_pool(model: Qwen2Model) -> None:
    backend = _backend(model)
    engine = BatchEngine.from_backend(backend, max_batch=3)
    for request in REQUESTS:
        engine.submit(request)
    engine.step()

    engine.cancel("long")
    engine.run()

    assert backend.pool.free_block_count == backend.pool.num_blocks - 1


def test_admission_fails_cleanly_when_the_pool_is_full(model: Qwen2Model) -> None:
    backend = _backend(model, num_blocks=3)  # 2 usable blocks = 8 tokens
    engine = BatchEngine.from_backend(backend, max_batch=3)
    engine.submit(REQUESTS[2])  # 30 prompt tokens: never fits

    with pytest.raises(ValueError, match="blocks"):
        engine.step()
    assert backend.pool.free_block_count == 2
