from dataclasses import FrozenInstanceError

import pytest

from tinyinfer.runtime.request import GenerationRequest


def test_request_defaults_to_32_new_tokens() -> None:
    request = GenerationRequest("request-1", "hello")
    assert request.max_new_tokens == 32


@pytest.mark.parametrize("value", ["", "   ", None])
def test_request_rejects_blank_or_non_string_id(value: object) -> None:
    with pytest.raises(ValueError, match="request_id"):
        GenerationRequest(value, "hello")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", "   ", None])
def test_request_rejects_blank_or_non_string_prompt(value: object) -> None:
    with pytest.raises(ValueError, match="prompt"):
        GenerationRequest("request-1", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, 257, 1.5, True])
def test_request_rejects_invalid_max_new_tokens(value: object) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationRequest("request-1", "hello", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, 256])
def test_request_accepts_token_bounds(value: int) -> None:
    assert GenerationRequest("request-1", "hello", value).max_new_tokens == value


def test_request_is_frozen_and_slotted() -> None:
    request = GenerationRequest("request-1", "hello")
    assert not hasattr(request, "__dict__")
    with pytest.raises(FrozenInstanceError):
        request.request_id = "changed"  # type: ignore[misc]
