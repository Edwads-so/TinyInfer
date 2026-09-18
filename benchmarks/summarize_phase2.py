"""Read and summarize immutable G2-P1 raw JSONL evidence."""

from __future__ import annotations

import json
from pathlib import Path

from tinyinfer.profiling.phase2 import Phase2Record


def read_phase2_records(path: Path) -> list[Phase2Record]:
    """Read strict JSONL; reject blank lines and invalid record payloads."""
    records: list[Phase2Record] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read raw records: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line {line_number}")
        try:
            payload = json.loads(line)
            records.append(Phase2Record.from_dict(payload))
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            raise ValueError(f"invalid raw record at line {line_number}: {error}") from error
    if not records:
        raise ValueError("raw record file must not be empty")
    return records
