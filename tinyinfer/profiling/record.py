"""Immutable correctness and benchmark record contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tinyinfer._tokens import validate_token_ids
from tinyinfer.runtime.result import StopReason

_TIMING_FIELDS = (
    "ttft_host_ms",
    "prefill_host_ms",
    "prefill_device_ms",
    "decode_host_ms",
    "decode_device_ms",
    "tpot_host_ms",
    "tpot_device_ms",
    "output_tokens_per_s",
)


@dataclass(frozen=True, slots=True)
class TokenComparison:
    agreement: float
    first_mismatch_index: int | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.agreement, bool)
            or not isinstance(self.agreement, (int, float))
            or not math.isfinite(self.agreement)
            or not 0.0 <= self.agreement <= 1.0
        ):
            raise ValueError("agreement must be finite and between 0.0 and 1.0")
        if self.first_mismatch_index is not None and (
            type(self.first_mismatch_index) is not int or self.first_mismatch_index < 0
        ):
            raise ValueError("first_mismatch_index must be None or a non-negative integer")
        if self.agreement == 1.0 and self.first_mismatch_index is not None:
            raise ValueError("first_mismatch_index must be None for exact agreement")
        if self.agreement < 1.0 and self.first_mismatch_index is None:
            raise ValueError("first_mismatch_index is required when agreement is not exact")


def compare_token_ids(
    actual: tuple[int, ...],
    reference: tuple[int, ...],
) -> TokenComparison:
    """Compare aligned tokens using the longer sequence as denominator."""

    validate_token_ids("actual token IDs", actual, allow_empty=True)
    validate_token_ids("reference token IDs", reference, allow_empty=True)
    denominator = max(len(actual), len(reference))
    if denominator == 0:
        raise ValueError("cannot compare two empty token sequences")

    shared_length = min(len(actual), len(reference))
    matching = sum(actual[index] == reference[index] for index in range(shared_length))
    first_mismatch = next(
        (index for index in range(shared_length) if actual[index] != reference[index]),
        None,
    )
    if first_mismatch is None and len(actual) != len(reference):
        first_mismatch = shared_length
    return TokenComparison(
        agreement=matching / denominator,
        first_mismatch_index=first_mismatch,
    )


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    schema_version: int
    git_revision: str
    model_id: str
    model_revision: str
    transformers_version: str
    torch_version: str
    dtype: str
    device_name: str
    case_id: str
    repetition: int
    warmup_runs: int
    prompt_tokens: int
    output_tokens: int
    output_token_ids: tuple[int, ...]
    stop_reason: StopReason
    token_agreement: float
    first_mismatch_index: int | None
    ttft_host_ms: float
    prefill_host_ms: float
    prefill_device_ms: float
    decode_host_ms: float
    decode_device_ms: float
    tpot_host_ms: float
    tpot_device_ms: float
    output_tokens_per_s: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if (
            not isinstance(self.git_revision, str)
            or len(self.git_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.git_revision)
        ):
            raise ValueError("git_revision must be a 40-character lowercase hex SHA")
        for field_name in (
            "model_id",
            "model_revision",
            "transformers_version",
            "torch_version",
            "dtype",
            "device_name",
            "case_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-blank string")

        for field_name in (
            "repetition",
            "warmup_runs",
            "prompt_tokens",
            "output_tokens",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

        validate_token_ids(
            "output_token_ids",
            self.output_token_ids,
            allow_empty=False,
        )
        if self.output_tokens != len(self.output_token_ids):
            raise ValueError("output_tokens must equal the length of output_token_ids")
        if self.stop_reason not in ("eos", "length"):
            raise ValueError("stop_reason must be 'eos' or 'length'")
        if (
            isinstance(self.token_agreement, bool)
            or not isinstance(self.token_agreement, (int, float))
            or not math.isfinite(self.token_agreement)
            or self.token_agreement != 1.0
        ):
            raise ValueError("token_agreement must be exactly 1.0")
        if self.first_mismatch_index is not None:
            raise ValueError("first_mismatch_index must be None for a canonical record")

        for field_name in _TIMING_FIELDS:
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.output_tokens >= 2 and self.decode_host_ms <= 0:
            raise ValueError("decode_host_ms must be positive for multiple output tokens")
        if self.output_tokens >= 2 and self.decode_device_ms <= 0:
            raise ValueError("decode_device_ms must be positive for multiple output tokens")

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible schema-v1 payload."""

        return {
            "schema_version": self.schema_version,
            "git_revision": self.git_revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "dtype": self.dtype,
            "device_name": self.device_name,
            "case_id": self.case_id,
            "repetition": self.repetition,
            "warmup_runs": self.warmup_runs,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "output_token_ids": list(self.output_token_ids),
            "stop_reason": self.stop_reason,
            "token_agreement": self.token_agreement,
            "first_mismatch_index": self.first_mismatch_index,
            "ttft_host_ms": self.ttft_host_ms,
            "prefill_host_ms": self.prefill_host_ms,
            "prefill_device_ms": self.prefill_device_ms,
            "decode_host_ms": self.decode_host_ms,
            "decode_device_ms": self.decode_device_ms,
            "tpot_host_ms": self.tpot_host_ms,
            "tpot_device_ms": self.tpot_device_ms,
            "output_tokens_per_s": self.output_tokens_per_s,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }
