"""Run the canonical TinyInfer Hugging Face reference benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.config import EngineConfig
from tinyinfer.profiling import (
    BenchmarkRecord,
    InferenceTimer,
    compare_token_ids,
)
from tinyinfer.runtime.generation import (
    GenerationObserver,
    generate_greedy,
    reference_generate,
)
from tinyinfer.runtime.hf_loader import LoadedModel, load_reference_model
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult

EXPECTED_CASE_IDS = ("short_en", "short_zh", "structured", "long_context")
CANONICAL_PROMPT_DIGEST = "c341df7fdef5bd3b8d8279bea9f0388f825241e6baa1f683e7910079b0b486db"
CANONICAL_WARMUP_RUNS = 3
CANONICAL_REPETITIONS = 5
CANONICAL_RECORD_COUNT = len(EXPECTED_CASE_IDS) * CANONICAL_REPETITIONS
CONFIG_KEYS = (
    "model_id",
    "dtype",
    "device",
    "max_sequence_length",
    "kv_block_size",
)


class BenchmarkError(RuntimeError):
    """Expected benchmark failure that must not update canonical evidence."""


class AlignmentError(BenchmarkError):
    """Raised when TinyInfer tokens differ from the fixed HF reference."""


class NondeterminismError(BenchmarkError):
    """Raised when measured repetitions produce different token sequences."""


class OutputExistsError(BenchmarkError):
    """Raised when canonical output exists without explicit overwrite."""


@dataclass(frozen=True, slots=True)
class PromptCase:
    case_id: str
    category: str
    prompt: str
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("prompt case_id must be a non-blank string")
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("prompt category must be a non-blank string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt text must be a non-blank string")
        if type(self.max_new_tokens) is not int or self.max_new_tokens != 32:
            raise ValueError("prompt max_new_tokens must be exactly 32")


@dataclass(frozen=True, slots=True)
class BenchmarkEnvironment:
    transformers_version: str
    torch_version: str
    dtype: str
    device_name: str

    def __post_init__(self) -> None:
        for field_name in (
            "transformers_version",
            "torch_version",
            "dtype",
            "device_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-blank string")


@dataclass(frozen=True, slots=True)
class BenchmarkDependencies:
    loader: Callable[[EngineConfig], LoadedModel]
    tinyinfer_runner: Callable[
        [LoadedModel, GenerationRequest, GenerationObserver | None],
        GenerationResult,
    ]
    reference_runner: Callable[
        [LoadedModel, GenerationRequest],
        tuple[int, ...],
    ]
    timer_factory: Callable[[], InferenceTimer]
    record_factory: Callable[..., BenchmarkRecord]
    environment_identity: Callable[[], BenchmarkEnvironment]
    git_revision: Callable[[], str]


def load_engine_config(path: Path) -> EngineConfig:
    """Parse the repository's dependency-free, flat YAML configuration."""

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise ValueError(f"config line {line_number} must be 'key: value'")
        if key in values:
            raise ValueError(f"config contains duplicate key: {key}")
        values[key] = value
    if tuple(values) != CONFIG_KEYS:
        raise ValueError(f"config keys must be exactly {CONFIG_KEYS}, got {tuple(values)}")
    try:
        return EngineConfig(
            model_id=values["model_id"],
            dtype=values["dtype"],
            device=values["device"],
            max_sequence_length=int(values["max_sequence_length"]),
            kv_block_size=int(values["kv_block_size"]),
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid benchmark config: {error}") from error


def load_prompt_cases(path: Path) -> tuple[PromptCase, ...]:
    """Load the strict schema-v1 workload in canonical case order."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prompt file: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "cases",
    }:
        raise ValueError("prompt file must contain only schema_version and cases")
    if payload["schema_version"] != 1 or not isinstance(payload["cases"], list):
        raise ValueError("prompt file requires schema_version 1 and a cases list")
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prompt_digest = hashlib.sha256(canonical_payload).hexdigest()
    if prompt_digest != CANONICAL_PROMPT_DIGEST:
        raise ValueError("prompt content must match the frozen G1 workload")

    cases: list[PromptCase] = []
    expected_keys = {"case_id", "category", "prompt", "max_new_tokens"}
    for index, raw_case in enumerate(payload["cases"]):
        if not isinstance(raw_case, dict) or set(raw_case) != expected_keys:
            raise ValueError(f"prompt case {index} has invalid keys")
        try:
            cases.append(
                PromptCase(
                    case_id=raw_case["case_id"],
                    category=raw_case["category"],
                    prompt=raw_case["prompt"],
                    max_new_tokens=raw_case["max_new_tokens"],
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"prompt case {index} is invalid: {error}") from error
    case_ids = tuple(case.case_id for case in cases)
    if case_ids != EXPECTED_CASE_IDS:
        raise ValueError(f"prompt case order must be {EXPECTED_CASE_IDS}, got {case_ids}")
    return tuple(cases)


def _validate_canonical_inputs(
    config: EngineConfig,
    cases: Sequence[PromptCase],
    *,
    warmup_runs: int,
    repetitions: int,
) -> None:
    if config != EngineConfig():
        raise BenchmarkError("canonical G1 requires the fixed default engine configuration")
    if warmup_runs != CANONICAL_WARMUP_RUNS:
        raise BenchmarkError(f"canonical G1 requires exactly {CANONICAL_WARMUP_RUNS} warm-up runs")
    if repetitions != CANONICAL_REPETITIONS:
        raise BenchmarkError(
            f"canonical G1 requires exactly {CANONICAL_REPETITIONS} measured repetitions"
        )
    if tuple(case.case_id for case in cases) != EXPECTED_CASE_IDS:
        raise BenchmarkError("canonical G1 requires the frozen prompt cases")


def _validate_canonical_records(records: Sequence[BenchmarkRecord]) -> None:
    if len(records) != CANONICAL_RECORD_COUNT:
        raise BenchmarkError(f"canonical G1 requires exactly {CANONICAL_RECORD_COUNT} records")
    expected_order = [
        (case_id, repetition)
        for case_id in EXPECTED_CASE_IDS
        for repetition in range(CANONICAL_REPETITIONS)
    ]
    actual_order = [(record.case_id, record.repetition) for record in records]
    if actual_order != expected_order:
        raise BenchmarkError("canonical G1 records have invalid case or repetition order")
    if any(record.warmup_runs != CANONICAL_WARMUP_RUNS for record in records):
        raise BenchmarkError("canonical G1 records have an invalid warm-up count")
    if any(
        record.token_agreement != 1.0 or record.first_mismatch_index is not None
        for record in records
    ):
        raise BenchmarkError("canonical G1 records require exact token agreement")


def _ensure_output_available(output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise OutputExistsError(f"output already exists: {output}; pass --overwrite to replace it")


def write_records_atomic(
    records: Sequence[BenchmarkRecord],
    output: Path,
    *,
    overwrite: bool,
) -> None:
    """Write complete JSONL through a sibling temporary file and atomic replace."""

    _ensure_output_available(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                line = json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                handle.write(f"{line}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _alignment_error(
    case: PromptCase,
    actual: tuple[int, ...],
    reference: tuple[int, ...],
) -> AlignmentError:
    comparison = compare_token_ids(actual, reference)
    return AlignmentError(
        f"{case.case_id} mismatch at token "
        f"{comparison.first_mismatch_index}: "
        f"TinyInfer={actual}, HuggingFace={reference}"
    )


def run_benchmark(
    *,
    config: EngineConfig,
    cases: Sequence[PromptCase],
    warmup_runs: int,
    repetitions: int,
    dependencies: BenchmarkDependencies,
) -> list[BenchmarkRecord]:
    """Run one load, one reference per case, warmups, and measured repetitions."""

    if type(warmup_runs) is not int or warmup_runs < 0:
        raise ValueError("warmup_runs must be a non-negative integer")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if tuple(case.case_id for case in cases) != EXPECTED_CASE_IDS:
        raise ValueError(f"benchmark cases must be ordered as {EXPECTED_CASE_IDS}")

    loaded = dependencies.loader(config)
    environment = dependencies.environment_identity()
    git_revision = dependencies.git_revision()
    records: list[BenchmarkRecord] = []

    for case in cases:
        reference_request = GenerationRequest(
            f"{case.case_id}-reference",
            case.prompt,
            case.max_new_tokens,
        )
        reference = dependencies.reference_runner(
            loaded,
            reference_request,
        )

        for warmup in range(warmup_runs):
            warmup_request = GenerationRequest(
                f"{case.case_id}-warmup-{warmup}",
                case.prompt,
                case.max_new_tokens,
            )
            result = dependencies.tinyinfer_runner(
                loaded,
                warmup_request,
                None,
            )
            comparison = compare_token_ids(
                result.output_token_ids,
                reference,
            )
            if comparison.agreement != 1.0:
                raise _alignment_error(
                    case,
                    result.output_token_ids,
                    reference,
                )

        deterministic_output: tuple[int, ...] | None = None
        for repetition in range(repetitions):
            timer = dependencies.timer_factory()
            measured_request = GenerationRequest(
                f"{case.case_id}-repetition-{repetition}",
                case.prompt,
                case.max_new_tokens,
            )
            result = dependencies.tinyinfer_runner(
                loaded,
                measured_request,
                timer,
            )
            if deterministic_output is not None and result.output_token_ids != deterministic_output:
                raise NondeterminismError(
                    f"{case.case_id} repetition {repetition} differs from "
                    f"repetition 0: first={deterministic_output}, "
                    f"current={result.output_token_ids}"
                )
            deterministic_output = result.output_token_ids
            comparison = compare_token_ids(
                result.output_token_ids,
                reference,
            )
            if comparison.agreement != 1.0:
                raise _alignment_error(
                    case,
                    result.output_token_ids,
                    reference,
                )
            timings = timer.timings
            records.append(
                dependencies.record_factory(
                    schema_version=1,
                    git_revision=git_revision,
                    model_id=loaded.model_id,
                    model_revision=loaded.model_revision,
                    transformers_version=environment.transformers_version,
                    torch_version=environment.torch_version,
                    dtype=environment.dtype,
                    device_name=environment.device_name,
                    case_id=case.case_id,
                    repetition=repetition,
                    warmup_runs=warmup_runs,
                    prompt_tokens=len(result.prompt_token_ids),
                    output_tokens=len(result.output_token_ids),
                    output_token_ids=result.output_token_ids,
                    stop_reason=result.stop_reason,
                    token_agreement=comparison.agreement,
                    first_mismatch_index=comparison.first_mismatch_index,
                    ttft_host_ms=timings.ttft_host_ms,
                    prefill_host_ms=timings.prefill_host_ms,
                    prefill_device_ms=timings.prefill_device_ms,
                    decode_host_ms=timings.decode_host_ms,
                    decode_device_ms=timings.decode_device_ms,
                    tpot_host_ms=timings.tpot_host_ms,
                    tpot_device_ms=timings.tpot_device_ms,
                    output_tokens_per_s=timings.output_tokens_per_s,
                    peak_allocated_bytes=timings.peak_allocated_bytes,
                    peak_reserved_bytes=timings.peak_reserved_bytes,
                )
            )
    return records


def _default_environment_identity() -> BenchmarkEnvironment:
    torch = importlib.import_module("torch")
    return BenchmarkEnvironment(
        transformers_version=importlib.metadata.version("transformers"),
        torch_version=str(torch.__version__),
        dtype="bfloat16",
        device_name=str(torch.cuda.get_device_name(0)),
    )


def _default_git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _default_tinyinfer_runner(
    loaded: LoadedModel,
    request: GenerationRequest,
    observer: GenerationObserver | None,
) -> GenerationResult:
    return generate_greedy(loaded, request, observer=observer)


def default_dependencies() -> BenchmarkDependencies:
    return BenchmarkDependencies(
        loader=load_reference_model,
        tinyinfer_runner=_default_tinyinfer_runner,
        reference_runner=reference_generate,
        timer_factory=InferenceTimer,
        record_factory=BenchmarkRecord,
        environment_identity=_default_environment_identity,
        git_revision=_default_git_revision,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the TinyInfer Hugging Face reference runtime."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen2_5_0_5b.yaml"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/prompts.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/g1_reference.jsonl"),
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: BenchmarkDependencies | None = None,
) -> int:
    args = _parse_args(argv)
    try:
        _ensure_output_available(args.output, overwrite=args.overwrite)
        config = load_engine_config(args.config)
        cases = load_prompt_cases(args.prompts)
        _validate_canonical_inputs(
            config,
            cases,
            warmup_runs=args.warmup_runs,
            repetitions=args.repetitions,
        )
        records = run_benchmark(
            config=config,
            cases=cases,
            warmup_runs=args.warmup_runs,
            repetitions=args.repetitions,
            dependencies=(dependencies if dependencies is not None else default_dependencies()),
        )
        _validate_canonical_records(records)
        write_records_atomic(
            records,
            args.output,
            overwrite=args.overwrite,
        )
    except (BenchmarkError, ValueError) as error:
        print(str(error), file=sys.stderr)
        print("G1 REFERENCE RUNTIME: FAIL")
        return 1
    print("G1 REFERENCE RUNTIME: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
