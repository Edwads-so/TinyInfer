"""Read and summarize immutable G2-P1 raw JSONL evidence."""

from __future__ import annotations

import json
from pathlib import Path

from tinyinfer.profiling.phase2 import Phase2Record, Phase2Summary


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


def render_markdown(summaries: tuple[Phase2Summary, ...]) -> str:
    """Render deterministic, compact throughput and latency summary table."""
    lines = [
        "# G2-P1 Performance Summary",
        "",
        "| Workload | Mode | Repetitions | Req/s p50 | Req/s p95 | Tok/s p50 | Tok/s p95 "
        "| E2E p50 ms | E2E p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.workload_id} | {summary.mode} | {summary.repetitions} | "
            f"{summary.requests_per_s_p50:.3f} | {summary.requests_per_s_p95:.3f} | "
            f"{summary.generated_tokens_per_s_p50:.3f} | "
            f"{summary.generated_tokens_per_s_p95:.3f} | "
            f"{summary.end_to_end_p50_ms:.3f} | {summary.end_to_end_p95_ms:.3f} |"
        )
    return "\n".join(lines) + "\n"
