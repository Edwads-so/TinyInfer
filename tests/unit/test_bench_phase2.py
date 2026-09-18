from benchmarks.bench_phase2 import RequestLifecycleTimings, submission_schedule
from tinyinfer.profiling.phase2 import Phase2Request, Phase2Workload


def _dynamic_workload() -> Phase2Workload:
    return Phase2Workload(
        workload_id="dynamic",
        family="dynamic_mixed",
        requests=(
            Phase2Request("r0", "short", "a", 4, 0),
            Phase2Request("r1", "short", "b", 4, 0),
            Phase2Request("r2", "long", "c", 4, 1),
            Phase2Request("r3", "long", "d", 4, 2),
        ),
    )


def test_continuous_schedule_preserves_deterministic_arrival_iterations() -> None:
    schedule = submission_schedule(_dynamic_workload(), mode="continuous_batch")

    assert [(item.request_id, iteration) for item, iteration in schedule] == [
        ("r0", 0),
        ("r1", 0),
        ("r2", 1),
        ("r3", 2),
    ]


def test_static_schedule_rejects_staggered_workload() -> None:
    try:
        submission_schedule(_dynamic_workload(), mode="static_batch")
    except ValueError as error:
        assert "arrival_iteration 0" in str(error)
    else:
        raise AssertionError("static batch must reject staggered arrivals")


def test_lifecycle_timings_calculate_ttft_and_end_to_end() -> None:
    readings = iter((100, 140, 220))
    timings = RequestLifecycleTimings(clock=lambda: next(readings))

    timings.request_admitted("r0")
    timings.first_token_ready("r0")
    timings.request_finished("r0")

    measurement = timings.measurement("r0", output_tokens=5)
    assert measurement.ttft_ms == 0.00004
    assert measurement.end_to_end_ms == 0.00012
    assert measurement.tpot_ms == 0.00002
