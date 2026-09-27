from typing import Any

import torch

from tinyinfer.model.generate import greedy_generate
from tinyinfer.model.qwen2 import Qwen2Model
from tinyinfer.runtime.request import GenerationRequest

PROMPT = "owned forward"


class CharTokenizer:
    """Maps characters into the tiny model's 128-token vocabulary."""

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        ids = torch.tensor([[ord(char) % 127 + 1 for char in text]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, tokens: list[int], **_: object) -> str:
        return " ".join(str(token) for token in tokens)


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[str] = []

    def request_started(self) -> None:
        self.events.append("request_started")

    def prefill_started(self) -> None:
        self.events.append("prefill_started")

    def prefill_finished(self) -> None:
        self.events.append("prefill_finished")

    def first_token_ready(self) -> None:
        self.events.append("first_token_ready")

    def decode_started(self) -> None:
        self.events.append("decode_started")

    def decode_finished(self, output_tokens: int) -> None:
        self.events.append(f"decode_finished:{output_tokens}")

    def request_finished(self) -> None:
        self.events.append("request_finished")


def _hf_tokens(tiny_qwen2: Any, max_new_tokens: int) -> tuple[int, ...]:
    input_ids = CharTokenizer()(PROMPT)["input_ids"]
    with torch.no_grad():
        generated = tiny_qwen2.hf_model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=None,
            pad_token_id=0,
        )
    return tuple(int(token) for token in generated[0, input_ids.shape[1] :])


def _model(tiny_qwen2: Any) -> Qwen2Model:
    return Qwen2Model.from_pretrained(
        tiny_qwen2.model_dir, device=torch.device("cpu"), dtype=torch.float32
    )


def test_greedy_tokens_match_hf_generate(tiny_qwen2: Any) -> None:
    expected = _hf_tokens(tiny_qwen2, 16)

    result = greedy_generate(
        _model(tiny_qwen2), CharTokenizer(), GenerationRequest("r0", PROMPT, 16), eos_token_id=-1
    )

    assert result.output_token_ids == expected
    assert result.stop_reason == "length"
    assert result.prompt_token_ids == tuple(CharTokenizer()(PROMPT)["input_ids"][0].tolist())


def test_greedy_stops_at_eos(tiny_qwen2: Any) -> None:
    reference = _hf_tokens(tiny_qwen2, 16)
    # The latest position whose token has not appeared before it, so EOS stops exactly there.
    stop_at = max(index for index, token in enumerate(reference) if token not in reference[:index])

    result = greedy_generate(
        _model(tiny_qwen2),
        CharTokenizer(),
        GenerationRequest("r0", PROMPT, 16),
        eos_token_id=reference[stop_at],
    )

    assert result.output_token_ids == reference[: stop_at + 1]
    assert result.stop_reason == "eos"


def test_observer_sees_the_request_lifecycle_in_order(tiny_qwen2: Any) -> None:
    observer = RecordingObserver()

    greedy_generate(
        _model(tiny_qwen2),
        CharTokenizer(),
        GenerationRequest("r0", PROMPT, 3),
        eos_token_id=-1,
        observer=observer,
    )

    assert observer.events == [
        "request_started",
        "prefill_started",
        "prefill_finished",
        "first_token_ready",
        "decode_started",
        "decode_finished:3",
        "request_finished",
    ]
