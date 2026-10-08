"""FCFS scheduling engine with one batched decode forward per iteration."""

from __future__ import annotations

from typing import Any, Protocol

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.runtime.hf_loader import LoadedModel
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult, StopReason
from tinyinfer.scheduler.backends import DecodeBackend, HFPagedBackend
from tinyinfer.scheduler.queue import RequestQueue
from tinyinfer.scheduler.state import EngineRequestState, IterationStats


class BatchEngineObserver(Protocol):
    """Receives request lifecycle events without affecting scheduling."""

    def request_admitted(self, request_id: str) -> None: ...

    def first_token_ready(self, request_id: str) -> None: ...

    def request_finished(self, request_id: str) -> None: ...


class _NoOpBatchEngineObserver:
    def request_admitted(self, request_id: str) -> None:
        pass

    def first_token_ready(self, request_id: str) -> None:
        pass

    def request_finished(self, request_id: str) -> None:
        pass


_NO_OP_OBSERVER = _NoOpBatchEngineObserver()


class BatchEngine:
    """FCFS scheduler over a ``DecodeBackend``.

    One ``step()`` iteration:

    1. Admit waiting requests (reserve KV storage + prefill serially).
    2. Run one batched decode forward over all running requests.
    3. Complete finished requests and release their KV storage.

    ``BatchEngine(loaded, paged)`` uses the Hugging Face paged backend (the G2-P1
    baseline); ``BatchEngine.from_backend`` accepts any backend.
    """

    def __init__(
        self,
        loaded: LoadedModel,
        paged: PagedKVCache,
        *,
        max_batch: int = 8,
        observer: BatchEngineObserver | None = None,
    ) -> None:
        self._loaded = loaded
        self._paged = paged
        self._setup(HFPagedBackend(loaded, paged), max_batch, observer)

    @classmethod
    def from_backend(
        cls,
        backend: DecodeBackend,
        *,
        max_batch: int = 8,
        observer: BatchEngineObserver | None = None,
    ) -> BatchEngine:
        engine = cls.__new__(cls)
        engine._setup(backend, max_batch, observer)
        return engine

    def _setup(
        self, backend: DecodeBackend, max_batch: int, observer: BatchEngineObserver | None
    ) -> None:
        self._backend = backend
        self._max_batch = max_batch
        self._queue = RequestQueue()
        self._requests: dict[str, GenerationRequest] = {}
        self._states: dict[str, EngineRequestState] = {}
        self._iteration = 0
        self._history: tuple[IterationStats, ...] = ()
        self._observer = observer if observer is not None else _NO_OP_OBSERVER

    @property
    def queue(self) -> RequestQueue:
        return self._queue

    @property
    def last_iteration(self) -> IterationStats | None:
        return self._history[-1] if self._history else None

    @property
    def iteration_history(self) -> tuple[IterationStats, ...]:
        return self._history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: GenerationRequest) -> None:
        """Enqueue one request."""
        self._requests[request.request_id] = request
        self._queue.enqueue(request)

    def cancel(self, request_id: str) -> None:
        """Cancel a waiting/running request and release its allocated blocks."""
        self._queue.cancel(request_id)
        self._release_state(request_id)

    def step(self) -> int:
        """Run one scheduling iteration.  Returns finished count.

        Any request that already satisfies its stop condition (EOS, or
        ``max_new_tokens`` reached) is finalized *before* the batched decode
        forward, so no request can ever exceed ``max_new_tokens``.  A failure
        during prefill or decode aborts every in-flight request and releases
        all blocks, then re-raises.
        """
        finished = 0
        cancelled = 0
        admitted: list[Any] = []
        try:
            admitted = self._queue.admit(self._max_batch - self._queue.running_count)
            for req in admitted:
                self._init_request_state(req.request_id)
                self._observer.request_admitted(req.request_id)
            # Serial prefill for newly admitted requests.
            for req in admitted:
                self._prefill_request(req.request_id)
                self._observer.first_token_ready(req.request_id)

            # Finalize requests that are already done (e.g. EOS at prefill or
            # max_new == 1) BEFORE decoding, so they never enter decode.
            finished += self._finalize_finished()

            running = list(self._queue.running_requests)
            running_ids = [r.request_id for r in running]
            old_lengths = tuple(
                self._backend.seq_length(self._states[rid].cache) for rid in running_ids
            )

            if running_ids:
                self._decode_batch(running_ids)
        except Exception:
            # Any failure leaks nothing: abort every in-flight request, cancel
            # queue entries, return all blocks, and re-raise.
            cancelled = self._abort_all()
            self._record_stats(
                admitted_count=len(admitted),
                decode_forward_count=0,
                finished_count=finished,
                cancelled_count=cancelled,
                request_ids=(),
                old_cache_lengths=(),
            )
            raise

        finished += self._finalize_finished()

        self._record_stats(
            admitted_count=len(admitted),
            decode_forward_count=1 if running_ids else 0,
            finished_count=finished,
            cancelled_count=cancelled,
            request_ids=tuple(running_ids),
            old_cache_lengths=old_lengths,
        )
        return finished

    def run(self) -> list[GenerationResult]:
        """Run steps until all requests finish.  Returns all results."""
        while not self._queue.all_done():
            self.step()
        return self._queue.drain_finished()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finalize_finished(self) -> int:
        """Finish every request that satisfies its stop condition.

        Returns the number of requests finalized (blocks released, states
        dropped, queue entries marked finished).
        """
        count = 0
        for rid in list(self._states):
            if self._is_done(rid):
                result = self._build_result(rid)
                self._queue.mark_finished(rid, result)
                self._observer.request_finished(rid)
                self._release_state(rid)
                count += 1
        return count

    def _abort_all(self) -> int:
        """Cancel every in-flight request and release its blocks.

        Used on prefill/decode/admission failure.  Cancels every running
        queue entry (including ones admitted but whose init failed before
        their state was tracked) and releases any engine-tracked blocks.
        Returns the number of cancelled requests.
        """
        import contextlib

        count = 0
        for req in list(self._queue.running_requests):
            with contextlib.suppress(KeyError):
                self._queue.cancel(req.request_id)
            self._release_state(req.request_id)
            count += 1
        for rid in list(self._states):
            self._release_state(rid)
        return count

    def _record_stats(
        self,
        *,
        admitted_count: int,
        decode_forward_count: int,
        finished_count: int,
        cancelled_count: int,
        request_ids: tuple[str, ...],
        old_cache_lengths: tuple[int, ...],
    ) -> None:
        stats = IterationStats(
            iteration_id=self._iteration,
            admitted_count=admitted_count,
            decode_batch_size=len(request_ids),
            decode_forward_count=decode_forward_count,
            running_count_after_step=len([r for r in self._queue.running_requests]),
            finished_count=finished_count,
            cancelled_count=cancelled_count,
            request_ids=request_ids,
            old_cache_lengths=old_cache_lengths,
        )
        self._history = (*self._history, stats)
        self._iteration += 1

    def _init_request_state(self, req_id: str) -> None:
        req = self._requests[req_id]
        self._states[req_id] = EngineRequestState(
            request=req,
            cache=self._backend.open(req),
            prompt_token_ids=(),
        )

    def _prefill_request(self, req_id: str) -> None:
        st = self._states[req_id]
        st.prompt_token_ids, first_token = self._backend.prefill(st.request, st.cache)
        st.output_token_ids = [first_token]

    def _decode_batch(self, running_ids: list[str]) -> None:
        """Run exactly one forward over all running requests."""
        states = [self._states[rid] for rid in running_ids]
        next_tokens = self._backend.decode(
            [st.cache for st in states], [st.output_token_ids[-1] for st in states]
        )
        for st, token in zip(states, next_tokens, strict=True):
            st.output_token_ids.append(token)

    def _release_state(self, req_id: str) -> None:
        """Release any reserved KV storage for *req_id* and drop state."""
        st = self._states.pop(req_id, None)
        if st is not None:
            self._backend.release(st.cache)
        self._requests.pop(req_id, None)

    def _is_done(self, req_id: str) -> bool:
        req = self._requests[req_id]
        tokens = self._states[req_id].output_token_ids
        if not tokens:
            return False
        return tokens[-1] == self._backend.eos_token_id or len(tokens) >= req.max_new_tokens

    def _build_result(self, req_id: str) -> GenerationResult:
        st = self._states[req_id]
        tokens = st.output_token_ids
        stop_reason: StopReason = "eos" if tokens[-1] == self._backend.eos_token_id else "length"
        text = self._backend.decode_text(tokens)
        return GenerationResult(
            request_id=req_id,
            prompt_token_ids=st.prompt_token_ids,
            output_token_ids=tuple(tokens),
            text=text,
            stop_reason=stop_reason,
        )
