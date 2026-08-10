"""BatchEngine GPU token-alignment and memory-leak verification.

Proves::

    HF generate(A) == BatchEngine(A)
    HF generate(B) == BatchEngine(B)
    HF generate(C) == BatchEngine(C)

for 3 requests of different prompt lengths, and verifies the paged block
pool does not accumulate memory across repeated engine runs.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.config import EngineConfig
from tinyinfer.runtime.generation import reference_generate
from tinyinfer.runtime.hf_loader import LoadedModel, load_reference_model
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.scheduler.engine import BatchEngine

pytestmark = [pytest.mark.gpu, pytest.mark.model]

# 3 requests with progressively longer prompts, all generating 16 tokens.
REQUEST_PROMPTS = [
    "Explain KV cache in one concise paragraph.",
    "Explain prefill and decode. " * 4,
    "A reference runtime separates prompt prefill from one-token decode. " * 8
    + "Explain the engineering lesson.",
]

LONG_REPEAT_FACTOR = 4  # ensure "longer" prompt actually tokenizes longer


@pytest.fixture(scope="module")
def loaded_model() -> LoadedModel:
    return load_reference_model(EngineConfig())


def _make_paged_cache(loaded: LoadedModel) -> PagedKVCache:
    """Build a PagedKVCache matching the model's dimensions."""
    config = loaded.model.config
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(config.hidden_size // config.num_attention_heads)
    max_seq_len = int(getattr(config, "max_position_embeddings", 4096))
    block_size = 16
    num_physical = (max_seq_len + block_size - 1) // block_size
    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        num_physical_blocks=num_physical,
        device=loaded.model.device,
        dtype=torch.bfloat16,
    )


def _build_requests() -> list[GenerationRequest]:
    return [GenerationRequest(f"req-{i}", prompt, 16) for i, prompt in enumerate(REQUEST_PROMPTS)]


class TestBatchTokenAlignment:
    """BatchEngine outputs must match HF model.generate() token-for-token."""

    def test_batch_outputs_match_hf_reference(self, loaded_model: LoadedModel) -> None:
        requests = _build_requests()
        paged = _make_paged_cache(loaded_model)
        engine = BatchEngine(loaded_model, paged, max_batch=8)

        for req in requests:
            engine.submit(req)

        results = engine.run()
        assert len(results) == len(requests)

        for req, result in zip(requests, results, strict=True):
            reference = reference_generate(loaded_model, req)
            assert result.output_token_ids == reference, (
                f"{req.request_id} mismatch: batch={result.output_token_ids}, ref={reference}"
            )


class TestBatchMemoryStability:
    """Peak memory must not grow across repeated engine runs."""

    def test_repeated_runs_do_not_accumulate_memory(self, loaded_model: LoadedModel) -> None:
        requests = _build_requests()
        paged = _make_paged_cache(loaded_model)

        peaks: list[int] = []
        for _ in range(3):
            torch.cuda.reset_peak_memory_stats()
            engine = BatchEngine(loaded_model, paged, max_batch=8)
            for req in requests:
                engine.submit(req)
            engine.run()
            peaks.append(int(torch.cuda.max_memory_allocated()))

        # Peak memory must be stable (allow 8% jitter for allocator noise).
        assert peaks[0] > 0
        for peak in peaks[1:]:
            assert peak <= peaks[0] * 1.08, f"peak memory grew across runs: {peaks}"
