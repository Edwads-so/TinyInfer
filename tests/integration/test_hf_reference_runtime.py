"""Real-model smoke tests for explicit prefill and decode."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tinyinfer.config import EngineConfig
from tinyinfer.runtime.generation import generate_greedy
from tinyinfer.runtime.hf_loader import LoadedModel, load_reference_model
from tinyinfer.runtime.request import GenerationRequest

pytestmark = [pytest.mark.gpu, pytest.mark.model]


@pytest.fixture(scope="module")
def loaded_model() -> LoadedModel:
    return load_reference_model(EngineConfig())


@pytest.mark.parametrize("max_new_tokens", [1, 2, 8, 32])
def test_explicit_runtime_respects_generation_limit(
    loaded_model: LoadedModel,
    max_new_tokens: int,
) -> None:
    result = generate_greedy(
        loaded_model,
        GenerationRequest(
            f"length-{max_new_tokens}",
            "Explain what a GPU cache is.",
            max_new_tokens,
        ),
    )
    assert 1 <= len(result.output_token_ids) <= max_new_tokens
    if result.stop_reason == "length":
        assert len(result.output_token_ids) == max_new_tokens
