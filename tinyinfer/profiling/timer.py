"""Observer-based host, CUDA, and memory measurements."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast


class CudaEvent(Protocol):
    def record(self) -> None: ...

    def elapsed_time(self, end_event: Any) -> float: ...


class CudaFacade(Protocol):
    def event(self) -> CudaEvent: ...

    def reset_peak_memory_stats(self) -> None: ...

    def synchronize(self) -> None: ...

    def max_memory_allocated(self) -> int: ...

    def max_memory_reserved(self) -> int: ...


class _TorchCudaFacade:
    def __init__(self) -> None:
        self._cuda = importlib.import_module("torch").cuda

    def event(self) -> CudaEvent:
        return cast(CudaEvent, self._cuda.Event(enable_timing=True))

    def reset_peak_memory_stats(self) -> None:
        self._cuda.reset_peak_memory_stats()

    def synchronize(self) -> None:
        self._cuda.synchronize()

    def max_memory_allocated(self) -> int:
        return int(self._cuda.max_memory_allocated())

    def max_memory_reserved(self) -> int:
        return int(self._cuda.max_memory_reserved())


@dataclass(frozen=True, slots=True)
class PhaseTimings:
    ttft_host_ms: float
    prefill_host_ms: float
    prefill_device_ms: float
    decode_host_ms: float
    decode_device_ms: float
    tpot_host_ms: float
    tpot_device_ms: float
    output_tokens_per_s: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


class InferenceTimer:
    """Measure one request through the generation observer callbacks."""

    def __init__(
        self,
        *,
        clock: Callable[[], int] = time.perf_counter_ns,
        cuda: CudaFacade | None = None,
    ) -> None:
        self._clock = clock
        self._cuda = cuda if cuda is not None else _TorchCudaFacade()
        self._state = "new"
        self._request_start_ns = 0
        self._prefill_start_ns = 0
        self._prefill_end_ns = 0
        self._first_token_ns = 0
        self._decode_start_ns = 0
        self._decode_end_ns = 0
        self._request_end_ns = 0
        self._output_tokens = 0
        self._prefill_start_event: CudaEvent | None = None
        self._prefill_end_event: CudaEvent | None = None
        self._decode_start_event: CudaEvent | None = None
        self._decode_end_event: CudaEvent | None = None
        self._timings: PhaseTimings | None = None

    def _require_state(self, expected: str, phase: str) -> None:
        if self._state != expected:
            raise RuntimeError(f"{phase} callback requires state {expected}, got {self._state}")

    @staticmethod
    def _duration_ms(start_ns: int, end_ns: int, phase: str) -> float:
        duration = (end_ns - start_ns) / 1_000_000
        if duration < 0:
            raise ValueError(f"{phase} host duration must be non-negative")
        return duration

    @staticmethod
    def _device_duration_ms(
        start_event: CudaEvent | None,
        end_event: CudaEvent | None,
        phase: str,
    ) -> float:
        if start_event is None or end_event is None:
            raise RuntimeError(f"{phase} CUDA event is missing")
        duration = float(start_event.elapsed_time(end_event))
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"{phase} device duration must be finite and non-negative")
        return duration

    def request_started(self) -> None:
        self._require_state("new", "request")
        self._cuda.reset_peak_memory_stats()
        self._request_start_ns = self._clock()
        self._state = "request_started"

    def prefill_started(self) -> None:
        self._require_state("request_started", "prefill")
        self._prefill_start_ns = self._clock()
        self._prefill_start_event = self._cuda.event()
        self._prefill_start_event.record()
        self._state = "prefill_started"

    def prefill_finished(self) -> None:
        self._require_state("prefill_started", "prefill")
        self._prefill_end_event = self._cuda.event()
        self._prefill_end_event.record()
        self._prefill_end_ns = self._clock()
        self._duration_ms(
            self._prefill_start_ns,
            self._prefill_end_ns,
            "prefill",
        )
        self._state = "prefill_finished"

    def first_token_ready(self) -> None:
        self._require_state("prefill_finished", "first token")
        self._cuda.synchronize()
        self._first_token_ns = self._clock()
        self._duration_ms(
            self._request_start_ns,
            self._first_token_ns,
            "first token",
        )
        self._state = "first_token_ready"

    def decode_started(self) -> None:
        self._require_state("first_token_ready", "decode")
        self._decode_start_ns = self._clock()
        self._decode_start_event = self._cuda.event()
        self._decode_start_event.record()
        self._state = "decode_started"

    def decode_finished(self, output_tokens: int) -> None:
        self._require_state("decode_started", "decode")
        if type(output_tokens) is not int or output_tokens < 1:
            raise ValueError("decode output_tokens must be a positive integer")
        self._decode_end_event = self._cuda.event()
        self._decode_end_event.record()
        self._decode_end_ns = self._clock()
        self._duration_ms(
            self._decode_start_ns,
            self._decode_end_ns,
            "decode",
        )
        self._output_tokens = output_tokens
        self._state = "decode_finished"

    def request_finished(self) -> None:
        self._require_state("decode_finished", "request")
        self._cuda.synchronize()
        self._request_end_ns = self._clock()
        end_to_end_ms = self._duration_ms(
            self._request_start_ns,
            self._request_end_ns,
            "request",
        )
        if end_to_end_ms == 0:
            raise ValueError("request host duration must be positive")

        ttft_host_ms = self._duration_ms(
            self._request_start_ns,
            self._first_token_ns,
            "first token",
        )
        prefill_host_ms = self._duration_ms(
            self._prefill_start_ns,
            self._prefill_end_ns,
            "prefill",
        )
        prefill_device_ms = self._device_duration_ms(
            self._prefill_start_event,
            self._prefill_end_event,
            "prefill",
        )
        decode_host_ms = self._duration_ms(
            self._decode_start_ns,
            self._decode_end_ns,
            "decode",
        )
        decode_device_ms = self._device_duration_ms(
            self._decode_start_event,
            self._decode_end_event,
            "decode",
        )
        decode_tokens = self._output_tokens - 1
        if decode_tokens == 0:
            decode_host_ms = 0.0
            decode_device_ms = 0.0
            tpot_host_ms = 0.0
            tpot_device_ms = 0.0
        else:
            tpot_host_ms = decode_host_ms / decode_tokens
            tpot_device_ms = decode_device_ms / decode_tokens

        peak_allocated = self._cuda.max_memory_allocated()
        peak_reserved = self._cuda.max_memory_reserved()
        if peak_allocated < 0 or peak_reserved < 0:
            raise ValueError("request peak memory values must be non-negative")
        self._timings = PhaseTimings(
            ttft_host_ms=ttft_host_ms,
            prefill_host_ms=prefill_host_ms,
            prefill_device_ms=prefill_device_ms,
            decode_host_ms=decode_host_ms,
            decode_device_ms=decode_device_ms,
            tpot_host_ms=tpot_host_ms,
            tpot_device_ms=tpot_device_ms,
            output_tokens_per_s=1000 * self._output_tokens / end_to_end_ms,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
        self._state = "request_finished"

    @property
    def timings(self) -> PhaseTimings:
        if self._timings is None:
            raise RuntimeError("request timings are unavailable before request_finished")
        return self._timings
