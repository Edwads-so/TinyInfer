"""Validated immutable generation requests."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Input contract for one greedy generation request."""

    request_id: str
    prompt: str
    max_new_tokens: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-blank string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-blank string")
        if type(self.max_new_tokens) is not int:
            raise ValueError("max_new_tokens must be an integer")
        if not 1 <= self.max_new_tokens <= 256:
            raise ValueError("max_new_tokens must be in [1, 256]")
