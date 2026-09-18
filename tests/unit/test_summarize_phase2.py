import json
from pathlib import Path

from benchmarks.summarize_phase2 import read_phase2_records


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
