from pathlib import Path

import pytest

from tinyinfer.profiling.phase2 import load_phase2_workloads


def test_dynamic_workload_has_staggered_arrivals() -> None:
    workloads = load_phase2_workloads(Path("benchmarks/phase2_workloads.json"))

    assert [request.arrival_iteration for request in workloads["dynamic_mixed_b4"].requests] == [
        0,
        0,
        1,
        2,
    ]


def test_workload_loader_rejects_fixed_workload_with_staggered_arrivals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        """{
  "schema_version": 1,
  "workloads": [{
    "workload_id": "bad",
    "family": "fixed",
    "requests": [{
      "request_id": "r0",
      "case_id": "short",
      "prompt": "hello",
      "max_new_tokens": 4,
      "arrival_iteration": 1
    }]
  }]
}""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="arrival_iteration 0"):
        load_phase2_workloads(path)
