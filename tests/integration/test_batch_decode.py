"""Real-model path identity and token alignment for heterogeneous batched decode.

Proves:
- a real Qwen forward hook observes a [3, 1] input with the batched cache,
- the matching iteration records one decode forward and 3 distinct old lengths,
- each request's output matches its own HF reference_generate,
- the block pool fully recovers and peak memory stays within tolerance.
"""

from typing import Any

import pytest
import torch

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.config import EngineConfig
from tinyinfer.runtime.generation import reference_generate
from tinyinfer.runtime.hf_loader import LoadedModel, load_reference_model
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.scheduler.engine import BatchEngine

pytestmark = [pytest.mark.gpu, pytest.mark.model]

PROMPTS = [
    "Explain KV cache in one concise paragraph.",
    "Explain prefill and decode. " * 4,
    "A reference runtime separates prompt prefill from one-token decode. " * 8
    + "Explain the engineering lesson.",
]
MAX_NEW = 4


@pytest.fixture(scope="module")
def loaded_model() -> LoadedModel:
    return load_reference_model(EngineConfig())


def _make_paged_cache(loaded: LoadedModel) -> PagedKVCache:
    config = loaded.model.config
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(config.hidden_size // config.num_attention_heads)
    max_seq_len = int(getattr(config, "max_position_embeddings", 4096))
    block_size = 16
    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        num_physical_blocks=(max_seq_len + block_size - 1) // block_size,
        device=loaded.model.device,
        dtype=torch.bfloat16,
    )


def _build_requests() -> list[GenerationRequest]:
    return [GenerationRequest(f"req-{i}", prompt, MAX_NEW) for i, prompt in enumerate(PROMPTS)]


class TestRealBatchPathIdentity:
    def test_hook_observes_b3_and_outputs_align(self, loaded_model: LoadedModel) -> None:
        lens = {len(loaded_model.tokenizer.encode(p)) for p in PROMPTS}
        assert len(lens) == 3, f"prompt lengths not distinct: {lens}"

        records: list[dict[str, Any]] = []
        handle = loaded_model.model.register_forward_hook(
            lambda module, args, kwargs, output: records.append(
                {
                    "input_shape": tuple(kwargs["input_ids"].shape)
                    if "input_ids" in kwargs
                    else None,
                    "position_ids": kwargs.get("position_ids").tolist()
                    if kwargs.get("position_ids") is not None
                    else None,
                    "cache_type": type(kwargs.get("past_key_values")).__name__,
                }
            ),
            with_kwargs=True,
        )
        try:
            paged = _make_paged_cache(loaded_model)
            engine = BatchEngine(loaded_model, paged, max_batch=8)
            for req in _build_requests():
                engine.submit(req)
            results = engine.run()
        finally:
            handle.remove()

        batched = [r for r in records if r["input_shape"] == (3, 1)]
        assert batched, "no [3,1] batched forward observed"
        assert any(r["cache_type"] == "BatchedPagedTinyinferCache" for r in batched)

        assert any(
            stat.decode_batch_size == 3 and stat.decode_forward_count == 1
            for stat in engine.iteration_history
        )
        assert any(
            len(set(stat.old_cache_lengths)) == 3
            for stat in engine.iteration_history
            if stat.decode_batch_size == 3
        )

        assert len(results) == 3
        for req, result in zip(_build_requests(), results, strict=True):
            ref = reference_generate(loaded_model, req)
            assert result.output_token_ids == ref, (
                f"{req.request_id} mismatch: batch={result.output_token_ids}, ref={ref}"
            )


class TestPoolAndMemoryStability:
    """Step 8.4: pool fully recovers and peak memory stays stable."""

    def test_three_runs_pool_recovers_and_memory_stable(self, loaded_model: LoadedModel) -> None:
        paged = _make_paged_cache(loaded_model)
        initial_free = paged.free_block_count

        peaks: list[int] = []
        for _ in range(3):
            torch.cuda.reset_peak_memory_stats()
            engine = BatchEngine(loaded_model, paged, max_batch=8)
            for req in _build_requests():
                engine.submit(req)
            results = engine.run()
            peaks.append(int(torch.cuda.max_memory_allocated()))
            assert len(results) == 3
            # no engine request state remains after run
            assert len(engine._states) == 0
            assert paged.free_block_count == initial_free

        # memory within 8% of first measured peak (warm-up jitter allowed)
        assert peaks[0] > 0
        for peak in peaks[1:]:
            assert peak <= peaks[0] * 1.08, f"peak memory grew: {peaks}"
