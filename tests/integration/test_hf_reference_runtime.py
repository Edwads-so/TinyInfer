"""Real-model smoke tests for explicit prefill and decode."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tinyinfer.config import EngineConfig
from tinyinfer.profiling import InferenceTimer
from tinyinfer.runtime.generation import generate_greedy, reference_generate
from tinyinfer.runtime.hf_loader import LoadedModel, load_reference_model
from tinyinfer.runtime.request import GenerationRequest

pytestmark = [pytest.mark.gpu, pytest.mark.model]

PROMPTS_PATH = Path(__file__).parents[2] / "benchmarks" / "prompts.json"


def _prompt_cases() -> list[dict[str, Any]]:
    payload = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    return payload["cases"]


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


def test_inference_timer_records_real_cuda_metrics(
    loaded_model: LoadedModel,
) -> None:
    timer = InferenceTimer()
    result = generate_greedy(
        loaded_model,
        GenerationRequest("profiled", "Explain what a GPU cache is.", 2),
        observer=timer,
    )
    timings = timer.timings

    assert len(result.output_token_ids) == 2
    float_metrics = (
        timings.ttft_host_ms,
        timings.prefill_host_ms,
        timings.prefill_device_ms,
        timings.decode_host_ms,
        timings.decode_device_ms,
        timings.tpot_host_ms,
        timings.tpot_device_ms,
        timings.output_tokens_per_s,
    )
    assert all(math.isfinite(value) and value >= 0 for value in float_metrics)
    assert timings.output_tokens_per_s > 0
    assert timings.peak_allocated_bytes > 0
    assert timings.peak_reserved_bytes > 0


def test_long_context_has_required_real_token_length(
    loaded_model: LoadedModel,
) -> None:
    case = next(case for case in _prompt_cases() if case["case_id"] == "long_context")
    encoded = loaded_model.tokenizer(
        case["prompt"],
        return_tensors="pt",
        add_special_tokens=True,
    )
    token_count = int(encoded["input_ids"].shape[1])
    assert 512 <= token_count <= 768, (
        f"long_context must tokenize to 512-768 tokens, got {token_count}"
    )


@pytest.mark.parametrize(
    ("case_id", "prompt", "max_new_tokens"),
    [(case["case_id"], case["prompt"], case["max_new_tokens"]) for case in _prompt_cases()],
)
def test_explicit_runtime_matches_hugging_face_generate(
    loaded_model: LoadedModel,
    case_id: str,
    prompt: str,
    max_new_tokens: int,
) -> None:
    request = GenerationRequest(case_id, prompt, max_new_tokens)
    actual = generate_greedy(loaded_model, request).output_token_ids
    reference = reference_generate(loaded_model, request)
    if actual != reference:
        mismatch_index = next(
            (
                index
                for index, (actual_token, reference_token) in enumerate(
                    zip(actual, reference, strict=False)
                )
                if actual_token != reference_token
            ),
            min(len(actual), len(reference)),
        )
        pytest.fail(
            f"{case_id} first mismatch at token {mismatch_index}: "
            f"TinyInfer={actual}, HuggingFace={reference}"
        )
