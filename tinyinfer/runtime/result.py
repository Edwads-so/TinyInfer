"""Immutable generation results."""

from dataclasses import dataclass
from typing import Literal

from tinyinfer._tokens import validate_token_ids

StopReason = Literal["eos", "length"]


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
        validate_token_ids("prompt_token_ids", self.prompt_token_ids)
        validate_token_ids("output_token_ids", self.output_token_ids)
        if self.stop_reason not in ("eos", "length"):
            raise ValueError("stop_reason must be 'eos' or 'length'")
