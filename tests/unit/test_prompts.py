import json
from pathlib import Path

LONG_PARAGRAPH = (
    "A reference runtime separates prompt prefill from one-token decode so that cache "
    "growth, latency, and correctness can be measured independently. Accurate measurements "
    "require fixed inputs, deterministic decoding, explicit synchronization, and recorded "
    "software revisions."
)
EXPECTED_PROMPTS = [
    "Explain KV cache in one concise paragraph.",
    "请用一段简洁的话解释大语言模型中的 KV Cache。",
    "Return exactly three numbered points explaining prefill, decode, and token latency.",
    "\n".join([LONG_PARAGRAPH] * 14 + ["Summarize the engineering lesson in two sentences."]),
]


def test_g1_prompts_have_exact_schema_order_and_limits() -> None:
    payload = json.loads(Path("benchmarks/prompts.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    cases = payload["cases"]
    assert [case["case_id"] for case in cases] == [
        "short_en",
        "short_zh",
        "structured",
        "long_context",
    ]
    assert all(set(case) == {"case_id", "category", "prompt", "max_new_tokens"} for case in cases)
    assert [case["prompt"] for case in cases] == EXPECTED_PROMPTS
    assert all(case["max_new_tokens"] == 32 for case in cases)


def test_long_context_prompt_is_frozen_to_fourteen_paragraphs() -> None:
    payload = json.loads(Path("benchmarks/prompts.json").read_text(encoding="utf-8"))
    long_prompt = payload["cases"][3]["prompt"]
    assert long_prompt.count(LONG_PARAGRAPH) == 14
    assert long_prompt.endswith("Summarize the engineering lesson in two sentences.")
