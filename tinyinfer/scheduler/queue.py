"""FCFS RequestQueue with waiting/running/finished states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult

RequestState = Literal["waiting", "running", "finished"]


@dataclass
class _RequestEntry:
    request: GenerationRequest
    state: RequestState = "waiting"
    result: GenerationResult | None = None


@dataclass
class BatchResult:
    """Aggregated results from one batched decode iteration."""

    finished: list[GenerationResult] = field(default_factory=list)
    running_count: int = 0
    waiting_count: int = 0


class RequestQueue:
    """FCFS (First-Come, First-Served) request queue with explicit states.

    Usage::

        queue = RequestQueue()
        queue.enqueue(GenerationRequest("a", "Hello", 32))
        queue.enqueue(GenerationRequest("b", "Hi", 16))

        running = queue.admit(max_batch=4)
        # running = [request_a, request_b]

        # ... after one decode step ...
        queue.mark_finished("a", result_a)
        queue.mark_finished("b", result_b)

        batch_status = queue.snapshot()
        # batch_status.finished = [result_a, result_b]
        # batch_status.running_count = 0
        # batch_status.waiting_count = 0
    """

    def __init__(self) -> None:
        self._entries: list[_RequestEntry] = []

    def enqueue(self, request: GenerationRequest) -> None:
        """Add a request to the waiting queue."""
        self._entries.append(_RequestEntry(request=request, state="waiting"))

    def admit(self, max_batch: int) -> list[GenerationRequest]:
        """Promote up to *max_batch* waiting requests to running.

        Returns the list of newly running requests.
        """
        admitted: list[GenerationRequest] = []
        for entry in self._entries:
            if entry.state == "waiting" and len(admitted) < max_batch:
                entry.state = "running"
                admitted.append(entry.request)
        return admitted

    def mark_finished(self, request_id: str, result: GenerationResult) -> None:
        """Transition a running request to finished."""
        for entry in self._entries:
            if entry.request.request_id == request_id and entry.state == "running":
                entry.state = "finished"
                entry.result = result
                return
        raise KeyError(f"no running request with id {request_id!r}")

    @property
    def running_requests(self) -> list[GenerationRequest]:
        return [entry.request for entry in self._entries if entry.state == "running"]

    @property
    def waiting_count(self) -> int:
        return sum(1 for entry in self._entries if entry.state == "waiting")

    @property
    def running_count(self) -> int:
        return sum(1 for entry in self._entries if entry.state == "running")

    @property
    def finished_count(self) -> int:
        return sum(1 for entry in self._entries if entry.state == "finished")

    def snapshot(self) -> BatchResult:
        """Return a point-in-time summary of the queue."""
        return BatchResult(
            finished=[entry.result for entry in self._entries if entry.state == "finished" and entry.result is not None],
            running_count=self.running_count,
            waiting_count=self.waiting_count,
        )

    def all_done(self) -> bool:
        """Return True when no waiting or running requests remain."""
        return self.waiting_count == 0 and self.running_count == 0

    def drain_finished(self) -> list[GenerationResult]:
        """Remove and return all finished results."""
        results: list[GenerationResult] = []
        remaining: list[_RequestEntry] = []
        for entry in self._entries:
            if entry.state == "finished" and entry.result is not None:
                results.append(entry.result)
            else:
                remaining.append(entry)
        self._entries = remaining
        return results