from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from benchmarks import bench_reference
from benchmarks.bench_reference import (
    AlignmentError,
    BenchmarkDependencies,
    BenchmarkEnvironment,
    NondeterminismError,
    OutputExistsError,
    PromptCase,
    load_engine_config,
    load_prompt_cases,
    run_benchmark,
    write_records_atomic,
)
from tinyinfer.profiling import BenchmarkRecord, PhaseTimings
from tinyinfer.runtime.hf_loader import LoadedModel
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult

CASE_IDS = ("short_en", "short_zh", "structured", "long_context")

FIXED_TIMINGS = PhaseTimings(
    ttft_host_ms=4.0,
    prefill_host_ms=2.0,
    prefill_device_ms=1.5,
    decode_host_ms=2.0,
    decode_device_ms=1.0,
    tpot_host_ms=2.0,
    tpot_device_ms=1.0,
    output_tokens_per_s=400.0,
    peak_allocated_bytes=100,
    peak_reserved_bytes=200,
)


@dataclass
class FakeTimer:
    timings: PhaseTimings = FIXED_TIMINGS


def _cases() -> tuple[PromptCase, ...]:
    return tuple(
        PromptCase(
            case_id=case_id,
            category=f"category-{index}",
            prompt=f"prompt-{index}",
            max_new_tokens=32,
        )
        for index, case_id in enumerate(CASE_IDS)
    )


def _valid_record(case_id: str = "short_en", repetition: int = 0) -> BenchmarkRecord:
    return BenchmarkRecord(
        schema_version=1,
        git_revision="a" * 40,
        model_id="Qwen/Qwen2.5-0.5B",
        model_revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        transformers_version="5.14.1",
        torch_version="2.13.0+cu130",
        dtype="bfloat16",
        device_name="NVIDIA GeForce RTX 4080 SUPER",
        case_id=case_id,
        repetition=repetition,
        warmup_runs=3,
        prompt_tokens=2,
        output_tokens=2,
        output_token_ids=(7, 8),
        stop_reason="length",
        token_agreement=1.0,
        first_mismatch_index=None,
        ttft_host_ms=4.0,
        prefill_host_ms=2.0,
        prefill_device_ms=1.5,
        decode_host_ms=2.0,
        decode_device_ms=1.0,
        tpot_host_ms=2.0,
        tpot_device_ms=1.0,
        output_tokens_per_s=400.0,
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
    )


def _dependencies(
    calls: list[tuple[str, bool]],
    *,
    mode: str = "match",
) -> BenchmarkDependencies:
    loaded = LoadedModel(
        model=object(),
        tokenizer=object(),
        model_id="Qwen/Qwen2.5-0.5B",
        model_revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        eos_token_id=99,
        pad_token_id=99,
    )

    def loader(_: object) -> LoadedModel:
        calls.append(("loader", False))
        return loaded

    def reference_runner(
        _: LoadedModel,
        request: GenerationRequest,
    ) -> tuple[int, ...]:
        request_id = request.request_id
        case_id = request_id.removesuffix("-reference")
        calls.append((request_id, False))
        token = CASE_IDS.index(case_id) + 7
        return (token, token + 1)

    measured_count: dict[str, int] = {}

    def tinyinfer_runner(
        _: LoadedModel,
        request: GenerationRequest,
        observer: object | None,
    ) -> GenerationResult:
        request_id = request.request_id
        case_id = next(case for case in CASE_IDS if request_id.startswith(case))
        is_measured = observer is not None
        calls.append((request_id, is_measured))
        token = CASE_IDS.index(case_id) + 7
        output = (token, token + 1)
        if mode == "mismatch":
            output = (999, 1000)
        elif mode == "nondeterministic" and is_measured:
            count = measured_count.get(case_id, 0)
            measured_count[case_id] = count + 1
            if count > 0:
                output = (token, 1000)
        return GenerationResult(
            request_id=request_id,
            prompt_token_ids=(1, 2),
            output_token_ids=output,
            text="fake",
            stop_reason="length",
        )

    return BenchmarkDependencies(
        loader=loader,
        tinyinfer_runner=tinyinfer_runner,
        reference_runner=reference_runner,
        timer_factory=FakeTimer,
        record_factory=BenchmarkRecord,
        environment_identity=lambda: BenchmarkEnvironment(
            transformers_version="5.14.1",
            torch_version="2.13.0+cu130",
            dtype="bfloat16",
            device_name="NVIDIA GeForce RTX 4080 SUPER",
        ),
        git_revision=lambda: "b" * 40,
    )


def test_load_prompt_cases_requires_exact_schema_and_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "prompt": case.prompt,
                        "max_new_tokens": case.max_new_tokens,
                    }
                    for case in _cases()
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_prompt_cases(path) == _cases()


@pytest.mark.parametrize(
    "mutation",
    ["schema", "order", "extra-key", "duplicate"],
)
def test_load_prompt_cases_rejects_invalid_workload_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    cases = [
        {
            "case_id": case.case_id,
            "category": case.category,
            "prompt": case.prompt,
            "max_new_tokens": case.max_new_tokens,
        }
        for case in _cases()
    ]
    payload: dict[str, Any] = {"schema_version": 1, "cases": cases}
    if mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "order":
        cases[0], cases[1] = cases[1], cases[0]
    elif mutation == "extra-key":
        cases[0]["extra"] = True
    else:
        cases[1]["case_id"] = "short_en"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt"):
        load_prompt_cases(path)


def test_load_engine_config_parses_repository_yaml_subset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "model_id: Qwen/Qwen2.5-0.5B",
                "dtype: bfloat16",
                "device: cuda",
                "max_sequence_length: 4096",
                "kv_block_size: 16",
            ]
        ),
        encoding="utf-8",
    )

    config = load_engine_config(path)

    assert config.model_id == "Qwen/Qwen2.5-0.5B"
    assert config.dtype == "bfloat16"
    assert config.device == "cuda"
    assert config.max_sequence_length == 4096
    assert config.kv_block_size == 16


def test_script_entrypoint_can_import_checkout_package() -> None:
    completed = subprocess.run(
        [sys.executable, "benchmarks/bench_reference.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--warmup-runs" in completed.stdout


def test_run_benchmark_loads_once_and_orders_twenty_records() -> None:
    calls: list[tuple[str, bool]] = []

    records = run_benchmark(
        config=load_engine_config(Path("configs/qwen2_5_0_5b.yaml")),
        cases=_cases(),
        warmup_runs=3,
        repetitions=5,
        dependencies=_dependencies(calls),
    )

    assert len(records) == 20
    assert [(record.case_id, record.repetition) for record in records] == [
        (case_id, repetition) for case_id in CASE_IDS for repetition in range(5)
    ]
    assert [call for call in calls if call[0] == "loader"] == [("loader", False)]
    assert len([call for call in calls if call[0].endswith("-reference")]) == 4
    assert len([call for call in calls if not call[1] and "warmup" in call[0]]) == 12
    assert len([call for call in calls if call[1]]) == 20
    assert all(record.warmup_runs == 3 for record in records)
    assert all(record.token_agreement == 1.0 for record in records)
    assert all(record.git_revision == "b" * 40 for record in records)
    assert all(record.device_name == "NVIDIA GeForce RTX 4080 SUPER" for record in records)


def test_run_benchmark_aborts_on_alignment_mismatch() -> None:
    with pytest.raises(AlignmentError, match=r"short_en.*mismatch.*0"):
        run_benchmark(
            config=load_engine_config(Path("configs/qwen2_5_0_5b.yaml")),
            cases=_cases(),
            warmup_runs=1,
            repetitions=2,
            dependencies=_dependencies([], mode="mismatch"),
        )


def test_run_benchmark_aborts_on_nondeterministic_measured_output() -> None:
    with pytest.raises(NondeterminismError, match=r"short_en.*repetition 1"):
        run_benchmark(
            config=load_engine_config(Path("configs/qwen2_5_0_5b.yaml")),
            cases=_cases(),
            warmup_runs=1,
            repetitions=2,
            dependencies=_dependencies([], mode="nondeterministic"),
        )


def test_atomic_writer_refuses_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.jsonl"
    output.write_text("original\n", encoding="utf-8")

    with pytest.raises(OutputExistsError, match="exists"):
        write_records_atomic([_valid_record()], output, overwrite=False)

    assert output.read_text(encoding="utf-8") == "original\n"


def test_atomic_writer_replaces_output_and_removes_temporary_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.jsonl"
    output.write_text("original\n", encoding="utf-8")

    write_records_atomic(
        [_valid_record("short_en", 0), _valid_record("short_en", 1)],
        output,
        overwrite=True,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["repetition"] for line in lines] == [0, 1]
    assert not output.with_name(f"{output.name}.tmp").exists()


def test_atomic_writer_cleans_temporary_file_and_preserves_output_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.jsonl"
    output.write_text("original\n", encoding="utf-8")

    def fail_serialization(*_: object, **__: object) -> str:
        raise TypeError("serialization failed")

    monkeypatch.setattr(bench_reference.json, "dumps", fail_serialization)

    with pytest.raises(TypeError, match="serialization failed"):
        write_records_atomic([_valid_record()], output, overwrite=True)

    assert output.read_text(encoding="utf-8") == "original\n"
    assert not output.with_name(f"{output.name}.tmp").exists()


def test_main_reports_mismatch_and_leaves_canonical_output_untouched(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "model_id: Qwen/Qwen2.5-0.5B\n"
        "dtype: bfloat16\n"
        "device: cuda\n"
        "max_sequence_length: 4096\n"
        "kv_block_size: 16\n",
        encoding="utf-8",
    )
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "prompt": case.prompt,
                        "max_new_tokens": case.max_new_tokens,
                    }
                    for case in _cases()
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "canonical.jsonl"
    output.write_text("original\n", encoding="utf-8")

    exit_code = bench_reference.main(
        [
            "--config",
            str(config),
            "--prompts",
            str(prompts),
            "--output",
            str(output),
            "--warmup-runs",
            "1",
            "--repetitions",
            "2",
            "--overwrite",
        ],
        dependencies=_dependencies([], mode="mismatch"),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "short_en" in captured.err
    assert "mismatch" in captured.err
    assert captured.out.rstrip().endswith("G1 REFERENCE RUNTIME: FAIL")
    assert output.read_text(encoding="utf-8") == "original\n"


def test_main_writes_ordered_records_and_reports_pass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "canonical.jsonl"

    exit_code = bench_reference.main(
        [
            "--output",
            str(output),
            "--warmup-runs",
            "1",
            "--repetitions",
            "2",
        ],
        dependencies=_dependencies([]),
    )

    captured = capsys.readouterr()
    payloads = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.rstrip().endswith("G1 REFERENCE RUNTIME: PASS")
    assert [(payload["case_id"], payload["repetition"]) for payload in payloads] == [
        (case_id, repetition) for case_id in CASE_IDS for repetition in range(2)
    ]


def test_main_refuses_existing_output_before_loading_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "canonical.jsonl"
    output.write_text("original\n", encoding="utf-8")
    calls: list[tuple[str, bool]] = []

    exit_code = bench_reference.main(
        ["--output", str(output)],
        dependencies=_dependencies(calls),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "exists" in captured.err
    assert captured.out.rstrip().endswith("G1 REFERENCE RUNTIME: FAIL")
    assert calls == []
