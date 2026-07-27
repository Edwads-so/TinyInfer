from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from tinyinfer.runtime.result import GenerationResult, StopReason


def make_result(
    *,
    request_id: str = "request-1",
    prompt_token_ids: tuple[int, ...] = (1, 2),
    output_token_ids: tuple[int, ...] = (3,),
    stop_reason: StopReason = "length",
) -> GenerationResult:
    return GenerationResult(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        text="output",
        stop_reason=stop_reason,
    )


def test_stop_reason_has_exact_public_values() -> None:
    assert get_args(StopReason) == ("eos", "length")


@pytest.mark.parametrize("value", ["", "   ", None])
def test_result_rejects_blank_or_non_string_request_id(value: object) -> None:
    with pytest.raises(ValueError, match="request_id"):
        make_result(request_id=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_token_ids", ()),
        ("output_token_ids", ()),
        ("prompt_token_ids", [1]),
        ("output_token_ids", [1]),
        ("prompt_token_ids", (-1,)),
        ("output_token_ids", (-1,)),
        ("prompt_token_ids", (1.5,)),
        ("output_token_ids", (True,)),
    ],
)
def test_result_rejects_invalid_token_tuples(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        make_result(**kwargs)  # type: ignore[arg-type]


def test_result_rejects_unknown_stop_reason() -> None:
    with pytest.raises(ValueError, match="stop_reason"):
        make_result(stop_reason="unknown")  # type: ignore[arg-type]


def test_result_is_frozen_slotted_and_serializable_data_only() -> None:
    result = make_result()
    assert not hasattr(result, "__dict__")
    assert result.prompt_token_ids == (1, 2)
    assert result.output_token_ids == (3,)
    with pytest.raises(FrozenInstanceError):
        result.stop_reason = "eos"  # type: ignore[misc]
