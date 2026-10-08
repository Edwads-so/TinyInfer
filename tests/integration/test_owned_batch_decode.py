"""Gate C on the real model: the owned forward pass batching on the paged pool."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from benchmarks.bench_phase2 import check_token_alignment, make_tie_checker, reference_logits_at
from tinyinfer.config import EngineConfig
from tinyinfer.model.paged import PagedKVPool
from tinyinfer.model.qwen2 import Qwen2Model
from tinyinfer.model.weights import resolve_model_dir
from tinyinfer.runtime.generation import reference_generate
from tinyinfer.runtime.hf_loader import MODEL_ID, MODEL_REVISION, LoadedModel, load_reference_model
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult
from tinyinfer.scheduler.backends import OwnedPagedBackend
from tinyinfer.scheduler.engine import BatchEngine

pytestmark = [pytest.mark.gpu, pytest.mark.model]

G1_CASES = json.loads(
    (Path(__file__).parents[2] / "benchmarks" / "prompts.json").read_text(encoding="utf-8")
)["cases"]
# The G2-BD scenario: three prompts of distinct lengths decoded in one [3, 1] forward.
BATCH_PROMPTS = [
    "Explain KV cache in one concise paragraph.",
    "Explain prefill and decode. " * 4,
    "A reference runtime separates prompt prefill from one-token decode. " * 8
    + "Explain the engineering lesson.",
]


@pytest.fixture(scope="module")
def reference() -> LoadedModel:
    return load_reference_model(EngineConfig())


@pytest.fixture(scope="module")
def owned() -> Qwen2Model:
    return Qwen2Model.from_pretrained(
        resolve_model_dir(MODEL_ID, MODEL_REVISION),
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )


def _backend(reference: LoadedModel, owned: Qwen2Model) -> OwnedPagedBackend:
    pool = PagedKVPool.for_spec(
        owned.spec, block_size=16, num_blocks=513, device=owned.device, dtype=owned.dtype
    )
    return OwnedPagedBackend(owned, reference.tokenizer, pool, eos_token_id=reference.eos_token_id)


def _run(
    backend: OwnedPagedBackend, requests: list[GenerationRequest], max_batch: int
) -> tuple[BatchEngine, list[GenerationResult]]:
    engine = BatchEngine.from_backend(backend, max_batch=max_batch)
    for request in requests:
        engine.submit(request)
    return engine, engine.run()


def _check(
    reference: LoadedModel, requests: list[GenerationRequest], results: list[GenerationResult]
) -> None:
    """Exact for one request at a time; batched rows may differ only at a BF16 tie."""
    prompts = {request.request_id: request.prompt for request in requests}
    expected = {request.request_id: reference_generate(reference, request) for request in requests}
    alignments = check_token_alignment(
        results,
        expected.__getitem__,
        mode="static_batch" if len(requests) > 1 else "sequential",
        tie_check=make_tie_checker(
            lambda rid, prefix: reference_logits_at(reference, prompts[rid], prefix)
        ),
    )
    for request_id, alignment in alignments.items():
        if alignment.first_mismatch_index is not None:
            print(f"\n{request_id}: BF16 tie at {alignment.first_mismatch_index}")


@pytest.mark.parametrize("case", G1_CASES, ids=[case["case_id"] for case in G1_CASES])
def test_single_request_matches_hf(
    reference: LoadedModel, owned: Qwen2Model, case: dict[str, object]
) -> None:
    request = GenerationRequest(
        str(case["case_id"]), str(case["prompt"]), int(str(case["max_new_tokens"]))
    )
    _, results = _run(_backend(reference, owned), [request], max_batch=1)

    _check(reference, [request], results)


def test_heterogeneous_batch_decodes_in_one_forward(
    reference: LoadedModel, owned: Qwen2Model
) -> None:
    requests = [GenerationRequest(f"req-{i}", p, 16) for i, p in enumerate(BATCH_PROMPTS)]
    lengths = {len(reference.tokenizer.encode(prompt)) for prompt in BATCH_PROMPTS}
    assert len(lengths) == 3

    engine, results = _run(_backend(reference, owned), requests, max_batch=8)

    assert any(
        stats.decode_batch_size == 3
        and stats.decode_forward_count == 1
        and len(set(stats.old_cache_lengths)) == 3
        for stats in engine.iteration_history
    )
    _check(reference, requests, results)


def test_pool_recovers_and_peak_memory_is_stable(reference: LoadedModel, owned: Qwen2Model) -> None:
    backend = _backend(reference, owned)
    initial_free = backend.pool.free_block_count
    requests = [GenerationRequest(f"req-{i}", p, 8) for i, p in enumerate(BATCH_PROMPTS)]

    peaks = []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats()
        _run(backend, requests, max_batch=8)
        peaks.append(int(torch.cuda.max_memory_allocated()))
        assert backend.pool.free_block_count == initial_free

    assert all(peak <= peaks[0] * 1.08 for peak in peaks[1:]), peaks
