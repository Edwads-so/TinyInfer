"""Unit tests for RequestQueue (CPU-safe)."""

import pytest

from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult
from tinyinfer.scheduler.queue import BatchResult, RequestQueue


def make_request(req_id: str, prompt: str = "hello", max_tokens: int = 32) -> GenerationRequest:
    return GenerationRequest(req_id, prompt, max_tokens)


def make_result(req_id: str, tokens: tuple[int, ...] = (7,)) -> GenerationResult:
    return GenerationResult(
        request_id=req_id,
        prompt_token_ids=(1, 2),
        output_token_ids=tokens,
        text="ok",
        stop_reason="length",
    )


class TestRequestQueue:
    """RequestQueue FCFS state machine tests."""

    def test_empty_queue_has_zero_counts(self) -> None:
        q = RequestQueue()
        assert q.waiting_count == 0
        assert q.running_count == 0
        assert q.finished_count == 0
        assert q.all_done()

    def test_enqueue_adds_waiting(self) -> None:
        q = RequestQueue()
        q.enqueue(make_request("a"))
        assert q.waiting_count == 1
        assert q.running_count == 0

    def test_admit_promotes_to_running(self) -> None:
        q = RequestQueue()
        q.enqueue(make_request("a"))
        q.enqueue(make_request("b"))

        admitted = q.admit(max_batch=2)
        assert len(admitted) == 2
        assert q.waiting_count == 0
        assert q.running_count == 2

    def test_admit_respects_max_batch(self) -> None:
        q = RequestQueue()
        for i in range(5):
            q.enqueue(make_request(str(i)))

        admitted = q.admit(max_batch=2)
        assert len(admitted) == 2
        assert q.waiting_count == 3
        assert q.running_count == 2

    def test_mark_finished_transitions_state(self) -> None:
        q = RequestQueue()
        q.enqueue(make_request("a"))
        q.admit(max_batch=1)
        q.mark_finished("a", make_result("a"))

        assert q.running_count == 0
        assert q.finished_count == 1

    def test_mark_finished_unknown_raises_keyerror(self) -> None:
        q = RequestQueue()
        with pytest.raises(KeyError):
            q.mark_finished("nonexistent", make_result("nonexistent"))

    def test_snapshot_reflects_current_state(self) -> None:
        q = RequestQueue()
        q.enqueue(make_request("a"))
        q.enqueue(make_request("b"))
        q.admit(max_batch=1)

        snap = q.snapshot()
        assert snap.running_count == 1
        assert snap.waiting_count == 1
        assert len(snap.finished) == 0

    def test_drain_finished_removes_and_returns_results(self) -> None:
        q = RequestQueue()
        q.enqueue(make_request("a"))
        q.admit(max_batch=1)
        q.mark_finished("a", make_result("a"))

        results = q.drain_finished()
        assert len(results) == 1
        assert results[0].request_id == "a"
        assert q.finished_count == 0

    def test_all_done_after_all_finished(self) -> None:
        q = RequestQueue()
        q.enqueue(make_request("a"))
        q.admit(max_batch=1)
        q.mark_finished("a", make_result("a"))
        q.drain_finished()

        assert q.all_done()

    def test_running_requests_returns_only_running(self) -> None:
        q = RequestQueue()
        q.enqueue(make_request("a"))
        q.enqueue(make_request("b"))
        q.admit(max_batch=1)

        running = q.running_requests
        assert len(running) == 1
        assert running[0].request_id == "a"


class TestBatchResult:
    def test_defaults(self) -> None:
        br = BatchResult()
        assert br.finished == []
        assert br.running_count == 0
        assert br.waiting_count == 0