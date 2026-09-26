import json
from pathlib import Path
from types import SimpleNamespace

from benchmarks.summarize_phase2 import (
    read_phase2_records,
    render_markdown,
    validate_complete_groups,
)
from tinyinfer.profiling.phase2 import Phase2Summary


def test_reader_rejects_blank_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text("\n", encoding="utf-8")

    try:
        read_phase2_records(path)
    except ValueError as error:
        assert "blank" in str(error)
    else:
        raise AssertionError("blank raw-record lines must be rejected")


def test_reader_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"not": "a record"}) + "\n", encoding="utf-8")

    try:
        read_phase2_records(path)
    except ValueError as error:
        assert "invalid" in str(error)
    else:
        raise AssertionError("invalid raw record must be rejected")


def test_renderer_includes_workload_mode_and_percentiles() -> None:
    text = render_markdown(
        (
            Phase2Summary(
                workload_id="fixed_b4",
                mode="continuous_batch",
                repetitions=5,
                makespan_p50_ms=12.0,
                makespan_p95_ms=14.0,
                ttft_p50_ms=3.0,
                tpot_p50_ms=1.5,
                requests_per_s_p50=300.0,
                requests_per_s_p95=280.0,
                generated_tokens_per_s_p50=900.0,
                generated_tokens_per_s_p95=850.0,
                diverged_requests=1,
            ),
        )
    )

    assert "fixed_b4" in text
    assert "continuous_batch" in text
    assert "300.000" in text
    assert "3.000" in text
    assert "| 1 |" in text
    assert "BF16 tie" in text


def test_complete_groups_require_exactly_five_unique_repetitions() -> None:
    records = [
        SimpleNamespace(workload_id="fixed_b1", mode="sequential", repetition=index)
        for index in range(4)
    ]

    try:
        validate_complete_groups(records)
    except ValueError as error:
        assert "five" in str(error)
    else:
        raise AssertionError("incomplete repetition group must be rejected")
