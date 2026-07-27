from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from tinyinfer.runtime import generation
from tinyinfer.runtime.hf_loader import LoadedModel
from tinyinfer.runtime.request import GenerationRequest


class FakeTensor:
    def __init__(
        self,
        values: list[Any],
        *,
        device: str = "cuda",
        dtype: str = "int64",
    ) -> None:
        self.values = values
        self.device = device
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        if self.values and isinstance(self.values[0], list):
            return (len(self.values), len(self.values[0]))
        return (len(self.values),)

    def to(self, device: str) -> FakeTensor:
        return FakeTensor(self.values, device=device, dtype=self.dtype)

    def __getitem__(self, index: int) -> FakeTensor:
        return FakeTensor(self.values[index], device=self.device, dtype=self.dtype)

    def tolist(self) -> list[Any]:
        return self.values

    def item(self) -> Any:
        assert len(self.values) == 1
        return self.values[0]


class FakeScores:
    def __init__(self, next_token_id: int) -> None:
        self.next_token_id = next_token_id

    def argmax(self, *, dim: int) -> FakeTensor:
        assert dim == -1
        return FakeTensor([self.next_token_id])


class FakeLogits:
    def __init__(self, next_token_id: int, sequence_length: int) -> None:
        self.next_token_id = next_token_id
        self.shape = (1, sequence_length, 32)

    def __getitem__(self, index: object) -> FakeScores:
        assert index == (slice(None), -1, slice(None))
        return FakeScores(self.next_token_id)


class FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class FakeTorch:
    long = "int64"

    def inference_mode(self) -> Any:
        return nullcontext()

    def tensor(
        self,
        values: list[list[int]],
        *,
        device: str,
        dtype: object,
    ) -> FakeTensor:
        return FakeTensor(values, device=device, dtype=str(dtype))

    def ones(
        self,
        shape: tuple[int, int],
        *,
        device: str,
        dtype: object,
    ) -> FakeTensor:
        assert shape == (1, 1)
        return FakeTensor([[1]], device=device, dtype=str(dtype))

    def cat(self, tensors: list[FakeTensor], *, dim: int) -> FakeTensor:
        assert dim == 1
        left, right = tensors
        return FakeTensor(
            [left.values[0] + right.values[0]],
            device=left.device,
            dtype=left.dtype,
        )

    def arange(self, start: int, end: int, *, device: str) -> FakeTensor:
        return FakeTensor(list(range(start, end)), device=device)


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generation, "_load_torch", FakeTorch)


def test_prefill_uses_full_prompt_and_returns_argmax_cache() -> None:
    cache = FakeCache(3)

    class Model:
        def __call__(self, **kwargs: Any) -> Any:
            assert kwargs["input_ids"].shape == (1, 3)
            assert kwargs["attention_mask"].shape == (1, 3)
            assert kwargs["use_cache"] is True
            assert kwargs["return_dict"] is True
            return SimpleNamespace(
                logits=FakeLogits(next_token_id=7, sequence_length=3),
                past_key_values=cache,
            )

    attention_mask = FakeTensor([[1, 1, 1]])
    result = generation.prefill(
        Model(),
        FakeTensor([[10, 11, 12]]),
        attention_mask,
    )

    assert result.next_token_id == 7
    assert result.cache is cache
    assert result.attention_mask is attention_mask


def test_prefill_rejects_cache_length_mismatch() -> None:
    output = SimpleNamespace(
        logits=FakeLogits(next_token_id=7, sequence_length=3),
        past_key_values=FakeCache(2),
    )

    def model(**_: Any) -> Any:
        return output

    with pytest.raises(RuntimeError, match="prefill cache length"):
        generation.prefill(model, FakeTensor([[1, 2, 3]]), FakeTensor([[1, 1, 1]]))


def test_decode_uses_one_token_extends_mask_and_advances_cache() -> None:
    cache = FakeCache(3)

    class Model:
        def __call__(self, **kwargs: Any) -> Any:
            assert kwargs["input_ids"].values == [[7]]
            assert kwargs["input_ids"].shape == (1, 1)
            assert kwargs["attention_mask"].values == [[1, 1, 1, 1]]
            assert kwargs["cache_position"].values == [3]
            assert kwargs["past_key_values"] is cache
            cache.length += 1
            return SimpleNamespace(
                logits=FakeLogits(next_token_id=8, sequence_length=1),
                past_key_values=cache,
            )

    result = generation.decode_one(
        Model(),
        last_token_id=7,
        attention_mask=FakeTensor([[1, 1, 1]]),
        cache=cache,
    )

    assert result.next_token_id == 8
    assert result.cache.get_seq_length() == 4
    assert result.attention_mask.values == [[1, 1, 1, 1]]


def test_decode_rejects_cache_that_does_not_advance() -> None:
    cache = FakeCache(3)
    output = SimpleNamespace(
        logits=FakeLogits(next_token_id=8, sequence_length=1),
        past_key_values=cache,
    )

    def model(**_: Any) -> Any:
        return output

    with pytest.raises(RuntimeError, match="decode cache length"):
        generation.decode_one(
            model,
            last_token_id=7,
            attention_mask=FakeTensor([[1, 1, 1]]),
            cache=cache,
        )


class FakeTokenizer:
    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, FakeTensor]:
        assert prompt == "hello"
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": True}
        return {
            "input_ids": FakeTensor([[1, 2]]),
            "attention_mask": FakeTensor([[1, 1]]),
        }

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": True}
        return " ".join(str(token_id) for token_id in token_ids)


def install_generation_steps(
    monkeypatch: pytest.MonkeyPatch,
    generated_ids: list[int],
    *,
    eos_token_id: int,
) -> LoadedModel:
    remaining = iter(generated_ids)
    cache = FakeCache(2)

    def fake_prefill(model: Any, input_ids: Any, attention_mask: Any) -> Any:
        return generation.PrefillOutput(next(remaining), cache, attention_mask)

    def fake_decode(
        model: Any,
        last_token_id: int,
        attention_mask: Any,
        old_cache: FakeCache,
    ) -> Any:
        old_cache.length += 1
        return generation.DecodeOutput(next(remaining), old_cache, attention_mask)

    monkeypatch.setattr(generation, "prefill", fake_prefill)
    monkeypatch.setattr(generation, "decode_one", fake_decode)
    return LoadedModel(
        model=SimpleNamespace(device="cuda"),
        tokenizer=FakeTokenizer(),
        model_id="Qwen/Qwen2.5-0.5B",
        model_revision="revision",
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )


@pytest.mark.parametrize(
    ("generated_ids", "limit", "eos_token_id", "expected", "stop_reason"),
    [
        ([7], 1, 99, (7,), "length"),
        ([99], 4, 99, (99,), "eos"),
        ([7, 99, 8], 4, 99, (7, 99), "eos"),
        ([7, 8, 9], 3, 99, (7, 8, 9), "length"),
    ],
)
def test_generate_greedy_stops_correctly(
    monkeypatch: pytest.MonkeyPatch,
    generated_ids: list[int],
    limit: int,
    eos_token_id: int,
    expected: tuple[int, ...],
    stop_reason: str,
) -> None:
    loaded = install_generation_steps(
        monkeypatch,
        generated_ids,
        eos_token_id=eos_token_id,
    )
    result = generation.generate_greedy(
        loaded,
        GenerationRequest("request-1", "hello", limit),
    )
    assert result.prompt_token_ids == (1, 2)
    assert result.output_token_ids == expected
    assert result.stop_reason == stop_reason
    assert result.text == " ".join(str(token_id) for token_id in expected)


class FakeGeneratedTokens:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __getitem__(self, index: tuple[int, slice]) -> FakeTensor:
        row, token_slice = index
        assert row == 0
        return FakeTensor(self.values[token_slice])


def test_reference_generate_uses_greedy_settings_and_returns_only_suffix() -> None:
    tokenizer = FakeTokenizer()

    class Model:
        device = "cuda"

        def generate(self, **kwargs: Any) -> FakeGeneratedTokens:
            assert kwargs["input_ids"].values == [[1, 2]]
            assert kwargs["attention_mask"].values == [[1, 1]]
            assert kwargs["do_sample"] is False
            assert kwargs["num_beams"] == 1
            assert kwargs["use_cache"] is True
            assert kwargs["max_new_tokens"] == 3
            assert kwargs["eos_token_id"] == 99
            assert kwargs["pad_token_id"] == 99
            return FakeGeneratedTokens([1, 2, 7, 8, 9])

    loaded = LoadedModel(
        model=Model(),
        tokenizer=tokenizer,
        model_id="Qwen/Qwen2.5-0.5B",
        model_revision="revision",
        eos_token_id=99,
        pad_token_id=99,
    )

    assert generation.reference_generate(
        loaded,
        GenerationRequest("reference", "hello", 3),
    ) == (7, 8, 9)


@pytest.mark.parametrize(
    ("tokens", "expected_output_count"),
    [([99], 1), ([7, 8, 9], 3)],
)
def test_generate_greedy_emits_all_observer_events_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tokens: list[int],
    expected_output_count: int,
) -> None:
    events: list[object] = []

    class Observer:
        def request_started(self) -> None:
            events.append("request_started")

        def prefill_started(self) -> None:
            events.append("prefill_started")

        def prefill_finished(self) -> None:
            events.append("prefill_finished")

        def first_token_ready(self) -> None:
            events.append("first_token_ready")

        def decode_started(self) -> None:
            events.append("decode_started")

        def decode_finished(self, output_tokens: int) -> None:
            events.append(("decode_finished", output_tokens))

        def request_finished(self) -> None:
            events.append("request_finished")

    result = generation.generate_greedy(
        install_generation_steps(
            monkeypatch,
            tokens,
            eos_token_id=99,
        ),
        GenerationRequest("observed", "hello", 3),
        observer=Observer(),
    )

    assert len(result.output_token_ids) == expected_output_count
    assert events == [
        "request_started",
        "prefill_started",
        "prefill_finished",
        "first_token_ready",
        "decode_started",
        ("decode_finished", expected_output_count),
        "request_finished",
    ]
