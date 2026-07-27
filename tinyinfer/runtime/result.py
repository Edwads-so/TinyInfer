"""Immutable generation results."""

from dataclasses import dataclass
from typing import Literal

StopReason = Literal["eos", "length"]


def _validate_token_ids(name: str, values: object) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(type(token_id) is not int or token_id < 0 for token_id in values):
        raise ValueError(f"{name} must contain only non-negative integers")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Serializable output contract for one completed request."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    text: str
    stop_reason: StopReason

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-blank string")
        _validate_token_ids("prompt_token_ids", self.prompt_token_ids)
        _validate_token_ids("output_token_ids", self.output_token_ids)
        if self.stop_reason not in ("eos", "length"):
            raise ValueError("stop_reason must be 'eos' or 'length'")
