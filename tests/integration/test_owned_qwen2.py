"""Gate G3 on the real model: the owned forward pass against Hugging Face."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from benchmarks.bench_phase2 import bf16_tie, reference_logits_at
from tinyinfer.config import EngineConfig
from tinyinfer.model.generate import greedy_generate
from tinyinfer.model.qwen2 import Qwen2Model
from tinyinfer.model.spec import QWEN25_05B
from tinyinfer.model.weights import resolve_model_dir
from tinyinfer.profiling.phase2 import first_mismatch_index
from tinyinfer.runtime.generation import reference_generate
from tinyinfer.runtime.hf_loader import MODEL_ID, MODEL_REVISION, LoadedModel, load_reference_model
from tinyinfer.runtime.request import GenerationRequest

pytestmark = [pytest.mark.gpu, pytest.mark.model]

CASES = json.loads(
    (Path(__file__).parents[2] / "benchmarks" / "prompts.json").read_text(encoding="utf-8")
)["cases"]


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


def test_spec_is_qwen25_05b(owned: Qwen2Model) -> None:
    assert owned.spec == QWEN25_05B


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_every_layer_matches_hf(
    reference: LoadedModel, owned: Qwen2Model, case: dict[str, Any]
) -> None:
    encoded = reference.tokenizer(case["prompt"], return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    with torch.inference_mode():
        hf = reference.model(
            input_ids=input_ids,
            attention_mask=encoded["attention_mask"].to("cuda"),
            output_hidden_states=True,
        )
    positions = torch.arange(input_ids.shape[1], device="cuda")[None]
    ours = owned.forward(input_ids, positions, capture_hidden=True)

    assert ours.layer_outputs is not None
    pairs = [
        *zip(ours.layer_outputs[:-1], hf.hidden_states[:-1], strict=True),
        (ours.final_hidden, hf.hidden_states[-1]),
        (ours.logits, hf.logits),
    ]
    report = []
    for index, (actual, expected) in enumerate(pairs):
        a, e = actual.float().flatten(), expected.float().flatten()
        cosine = float(torch.nn.functional.cosine_similarity(a, e, dim=0))
        report.append((index, cosine, float((a - e).abs().max()), torch.equal(actual, expected)))
    print(f"\n{case['case_id']}: (index, cosine, max|diff|, bit-equal)")
    for row in report:
        print(f"  {row[0]:>2} {row[1]:.6f} {row[2]:.3g} {row[3]}")
    assert all(cosine >= 0.999 for _, cosine, _, _ in report)


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_greedy_matches_hf_generate(
    reference: LoadedModel, owned: Qwen2Model, case: dict[str, Any]
) -> None:
    request = GenerationRequest(case["case_id"], case["prompt"], case["max_new_tokens"])
    expected = reference_generate(reference, request)

    actual = greedy_generate(
        owned, reference.tokenizer, request, eos_token_id=reference.eos_token_id
    ).output_token_ids

    index = first_mismatch_index(actual, expected)
    if index is None:
        return
    # Same rule as G2-P1: only a BF16 tie may break the other way.
    assert index < min(len(actual), len(expected)), f"length differs at {index}"
    logits = reference_logits_at(reference, case["prompt"], expected[:index])
    tie = bf16_tie(logits, reference_token=expected[index], actual_token=actual[index])
    assert tie.is_tie, f"diverged at {index}: gap {tie.logit_gap} > {tie.tolerance}"
    print(f"\n{case['case_id']}: BF16 tie at {index}, gap {tie.logit_gap}")
