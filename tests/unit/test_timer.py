from collections.abc import Iterator

import pytest

from tinyinfer.profiling.timer import InferenceTimer


class FakeEvent:
    def __init__(self, elapsed_ms: float) -> None:
        self.elapsed_ms = elapsed_ms
        self.recorded = False

    def record(self) -> None:
        self.recorded = True

    def elapsed_time(self, end_event: object) -> float:
        assert self.recorded
        assert isinstance(end_event, FakeEvent)
        assert end_event.recorded
        return self.elapsed_ms


class FakeCuda:
    def __init__(
        self,
        event_elapsed_ms: list[float],
        *,
        peak_allocated: int = 123,
        peak_reserved: int = 456,
    ) -> None:
        self._event_elapsed_ms = iter(event_elapsed_ms)
        self.peak_allocated = peak_allocated
        self.peak_reserved = peak_reserved
        self.reset_calls = 0
        self.synchronize_calls = 0

    def event(self) -> FakeEvent:
        return FakeEvent(next(self._event_elapsed_ms))

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def max_memory_allocated(self) -> int:
        return self.peak_allocated

    def max_memory_reserved(self) -> int:
        return self.peak_reserved


def _clock(values: list[int]) -> tuple[Iterator[int], object]:
    iterator = iter(values)
    return iterator, lambda: next(iterator)


def test_timer_computes_exact_host_device_and_memory_metrics() -> None:
    _, clock = _clock(
        [
            0,
            10_000_000,
            30_000_000,
            40_000_000,
            45_000_000,
            65_000_000,
            70_000_000,
        ]
    )
    cuda = FakeCuda([12.5, 0.0, 8.0, 0.0])
    timer = InferenceTimer(clock=clock, cuda=cuda)

    timer.request_started()
    timer.prefill_started()
    timer.prefill_finished()
    timer.first_token_ready()
    timer.decode_started()
    timer.decode_finished(3)
    timer.request_finished()

    assert timer.timings.ttft_host_ms == 40.0
    assert timer.timings.prefill_host_ms == 20.0
    assert timer.timings.prefill_device_ms == 12.5
    assert timer.timings.decode_host_ms == 20.0
    assert timer.timings.decode_device_ms == 8.0
    assert timer.timings.tpot_host_ms == 10.0
    assert timer.timings.tpot_device_ms == 4.0
    assert timer.timings.output_tokens_per_s == pytest.approx(3000 / 70)
    assert timer.timings.peak_allocated_bytes == 123
    assert timer.timings.peak_reserved_bytes == 456
    assert cuda.reset_calls == 1
    assert cuda.synchronize_calls == 2


def test_timer_immediate_eos_has_zero_decode_and_tpot() -> None:
    _, clock = _clock([0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000])
    timer = InferenceTimer(clock=clock, cuda=FakeCuda([0.5, 0.0, 99.0, 0.0]))

    timer.request_started()
    timer.prefill_started()
    timer.prefill_finished()
    timer.first_token_ready()
    timer.decode_started()
    timer.decode_finished(1)
    timer.request_finished()

    assert timer.timings.decode_host_ms == 0.0
    assert timer.timings.decode_device_ms == 0.0
    assert timer.timings.tpot_host_ms == 0.0
    assert timer.timings.tpot_device_ms == 0.0


def test_timer_rejects_duplicate_request_start() -> None:
    _, clock = _clock([0])
    timer = InferenceTimer(clock=clock, cuda=FakeCuda([]))
    timer.request_started()

    with pytest.raises(RuntimeError, match="request"):
        timer.request_started()


def test_timer_rejects_finish_before_request_is_complete() -> None:
    _, clock = _clock([0])
    timer = InferenceTimer(clock=clock, cuda=FakeCuda([]))
    timer.request_started()

    with pytest.raises(RuntimeError, match="request"):
        timer.request_finished()


def test_timer_rejects_missing_prefill_start() -> None:
    _, clock = _clock([0])
    timer = InferenceTimer(clock=clock, cuda=FakeCuda([]))
    timer.request_started()

    with pytest.raises(RuntimeError, match="prefill"):
        timer.prefill_finished()


def test_timer_rejects_negative_prefill_host_duration() -> None:
    _, clock = _clock([10, 5, 4])
    timer = InferenceTimer(clock=clock, cuda=FakeCuda([1.0, 0.0]))
    timer.request_started()
    timer.prefill_started()

    with pytest.raises(ValueError, match="prefill"):
        timer.prefill_finished()


def test_timer_rejects_negative_prefill_device_duration() -> None:
    _, clock = _clock([0, 1, 2, 3, 4, 5, 6])
    timer = InferenceTimer(
        clock=clock,
        cuda=FakeCuda([-1.0, 0.0, 1.0, 0.0]),
    )
    timer.request_started()
    timer.prefill_started()
    timer.prefill_finished()
    timer.first_token_ready()
    timer.decode_started()
    timer.decode_finished(2)

    with pytest.raises(ValueError, match="prefill"):
        timer.request_finished()


def test_timings_are_unavailable_before_request_finish() -> None:
    _, clock = _clock([])
    timer = InferenceTimer(clock=clock, cuda=FakeCuda([]))

    with pytest.raises(RuntimeError, match="request"):
        _ = timer.timings
