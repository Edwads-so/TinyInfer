import json
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


def test_workloads_sweep_batch_sizes_for_short_and_long_prompts() -> None:
    workloads = load_phase2_workloads(Path("benchmarks/phase2_workloads.json"))

    for prefix, sizes in (("fixed_b", (1, 2, 4, 8, 16)), ("long_b", (1, 2, 4, 8))):
        for size in sizes:
            workload = workloads[f"{prefix}{size}"]
            assert workload.family == "fixed"
            assert len(workload.requests) == size
            assert len({request.max_new_tokens for request in workload.requests}) == 1


def test_long_workloads_reuse_the_frozen_g1_long_context_prompt() -> None:
    workloads = load_phase2_workloads(Path("benchmarks/phase2_workloads.json"))
    cases = json.loads(Path("benchmarks/prompts.json").read_text(encoding="utf-8"))["cases"]
    long_prompt = next(case["prompt"] for case in cases if case["case_id"] == "long_context")

    assert all(request.prompt == long_prompt for request in workloads["long_b8"].requests)


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
