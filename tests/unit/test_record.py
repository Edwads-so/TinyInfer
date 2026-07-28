import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from tinyinfer.profiling.record import (
    BenchmarkRecord,
    TokenComparison,
    compare_token_ids,
)

SCHEMA_KEYS = (
    "schema_version",
    "git_revision",
    "model_id",
    "model_revision",
    "transformers_version",
    "torch_version",
    "dtype",
    "device_name",
    "case_id",
    "repetition",
    "warmup_runs",
    "prompt_tokens",
    "output_tokens",
    "output_token_ids",
    "stop_reason",
    "token_agreement",
    "first_mismatch_index",
    "ttft_host_ms",
    "prefill_host_ms",
    "prefill_device_ms",
    "decode_host_ms",
    "decode_device_ms",
    "tpot_host_ms",
    "tpot_device_ms",
    "output_tokens_per_s",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)

TIMING_FIELDS = (
    "ttft_host_ms",
    "prefill_host_ms",
    "prefill_device_ms",
    "decode_host_ms",
    "decode_device_ms",
    "tpot_host_ms",
    "tpot_device_ms",
    "output_tokens_per_s",
)


def _valid_record(**changes: Any) -> BenchmarkRecord:
    values: dict[str, Any] = {
        "schema_version": 1,
        "git_revision": "a" * 40,
        "model_id": "Qwen/Qwen2.5-0.5B",
        "model_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "transformers_version": "5.14.1",
        "torch_version": "2.13.0+cu130",
        "dtype": "bfloat16",
        "device_name": "NVIDIA GeForce RTX 4080 SUPER",
        "case_id": "short_en",
        "repetition": 0,
        "warmup_runs": 3,
        "prompt_tokens": 8,
        "output_tokens": 3,
        "output_token_ids": (10, 11, 12),
        "stop_reason": "length",
        "token_agreement": 1.0,
        "first_mismatch_index": None,
        "ttft_host_ms": 4.0,
        "prefill_host_ms": 2.0,
        "prefill_device_ms": 1.5,
        "decode_host_ms": 2.0,
        "decode_device_ms": 1.0,
        "tpot_host_ms": 1.0,
        "tpot_device_ms": 0.5,
        "output_tokens_per_s": 500.0,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }
    values.update(changes)
    return BenchmarkRecord(**values)


def test_compare_token_ids_reports_exact_match() -> None:
    assert compare_token_ids((3, 5, 8), (3, 5, 8)) == TokenComparison(
        agreement=1.0,
        first_mismatch_index=None,
    )


def test_compare_token_ids_uses_max_length_denominator() -> None:
    assert compare_token_ids((3, 9, 8), (3, 5, 8, 13)) == TokenComparison(
        agreement=0.5,
        first_mismatch_index=1,
    )


@pytest.mark.parametrize(
    ("actual", "reference"),
    [
        ((3, 5), (3, 5, 8)),
        ((3, 5, 8), (3, 5)),
    ],
)
def test_compare_token_ids_reports_shared_boundary_for_length_mismatch(
    actual: tuple[int, ...],
    reference: tuple[int, ...],
) -> None:
    comparison = compare_token_ids(actual, reference)

    assert comparison.agreement == pytest.approx(2 / 3)
    assert comparison.first_mismatch_index == 2


def test_compare_token_ids_allows_one_empty_side() -> None:
    assert compare_token_ids((), (7, 8)) == TokenComparison(
        agreement=0.0,
        first_mismatch_index=0,
    )


def test_compare_token_ids_rejects_two_empty_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        compare_token_ids((), ())


@pytest.mark.parametrize("values", [(1, -1), (1, True)])
def test_compare_token_ids_rejects_invalid_tokens(
    values: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="token"):
        compare_token_ids(values, (1, 2))


def test_token_comparison_is_immutable() -> None:
    comparison = TokenComparison(1.0, None)

    with pytest.raises(FrozenInstanceError):
        comparison.agreement = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("agreement", "first_mismatch_index"),
    [
        (float("nan"), 0),
        (float("inf"), 0),
        (-0.1, 0),
        (1.1, None),
        (1.0, 0),
        (0.5, None),
        (0.5, -1),
    ],
)
def test_token_comparison_rejects_inconsistent_values(
    agreement: float,
    first_mismatch_index: int | None,
) -> None:
    with pytest.raises(ValueError):
        TokenComparison(agreement, first_mismatch_index)


def test_benchmark_record_serializes_exact_schema_v1() -> None:
    payload = _valid_record().to_dict()

    assert tuple(payload) == SCHEMA_KEYS
    assert payload["output_token_ids"] == [10, 11, 12]
    assert payload["stop_reason"] == "length"
    assert payload["first_mismatch_index"] is None
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_benchmark_record_is_immutable() -> None:
    record = _valid_record()

    with pytest.raises(FrozenInstanceError):
        record.case_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("field", TIMING_FIELDS)
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_benchmark_record_rejects_invalid_timing(
    field: str,
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        _valid_record(**{field: invalid})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("repetition", -1),
        ("warmup_runs", -1),
        ("prompt_tokens", -1),
        ("output_tokens", -1),
        ("peak_allocated_bytes", -1),
        ("peak_reserved_bytes", -1),
    ],
)
def test_benchmark_record_rejects_negative_counter(
    field: str,
    invalid: int,
) -> None:
    with pytest.raises(ValueError, match=field):
        _valid_record(**{field: invalid})


@pytest.mark.parametrize(
    "git_revision",
    ["abc", "g" * 40, "a" * 41],
)
def test_benchmark_record_rejects_invalid_git_sha(
    git_revision: str,
) -> None:
    with pytest.raises(ValueError, match="git_revision"):
        _valid_record(git_revision=git_revision)


def test_benchmark_record_rejects_inconsistent_output_count() -> None:
    with pytest.raises(ValueError, match="output_tokens"):
        _valid_record(output_tokens=2)


def test_benchmark_record_rejects_noncanonical_agreement() -> None:
    with pytest.raises(ValueError, match="token_agreement"):
        _valid_record(token_agreement=0.5, first_mismatch_index=1)


def test_benchmark_record_rejects_boolean_agreement() -> None:
    with pytest.raises(ValueError, match="token_agreement"):
        _valid_record(token_agreement=True)


def test_benchmark_record_rejects_mismatch_on_canonical_record() -> None:
    with pytest.raises(ValueError, match="first_mismatch_index"):
        _valid_record(first_mismatch_index=0)


def test_benchmark_record_rejects_zero_decode_for_multiple_tokens() -> None:
    with pytest.raises(ValueError, match="decode_host_ms"):
        _valid_record(decode_host_ms=0.0)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("schema_version", 2),
        ("model_id", " "),
        ("output_token_ids", ()),
        ("output_token_ids", (1, -1)),
        ("stop_reason", "cancelled"),
    ],
)
def test_benchmark_record_rejects_invalid_contract_field(
    field: str,
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        _valid_record(**{field: invalid})


def test_benchmark_record_accepts_single_token_with_zero_decode() -> None:
    record = _valid_record(
        output_tokens=1,
        output_token_ids=(10,),
        decode_host_ms=0.0,
        decode_device_ms=0.0,
        tpot_host_ms=0.0,
        tpot_device_ms=0.0,
    )

    assert record.output_tokens == 1


def test_replace_revalidates_immutable_record() -> None:
    record = _valid_record()

    with pytest.raises(ValueError, match="output_tokens"):
        replace(record, output_tokens=99)
