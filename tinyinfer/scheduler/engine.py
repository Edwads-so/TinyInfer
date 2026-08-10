"""FCFS scheduling engine with one batched decode forward per iteration."""

from __future__ import annotations

import importlib

from tinyinfer.cache.batched_paged_tinyinfer_cache import BatchedPagedTinyinferCache
from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.cache.paged_tinyinfer_cache import PagedTinyinferCache
from tinyinfer.cache.request_state import PagedRequestState
from tinyinfer.runtime.generation import _select_next_token, _select_next_tokens
from tinyinfer.runtime.hf_loader import LoadedModel
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult, StopReason
from tinyinfer.scheduler.queue import RequestQueue
from tinyinfer.scheduler.state import EngineRequestState, IterationStats


class BatchEngine:
    """FCFS scheduler + batched paged KV decode.

    One ``step()`` iteration:

    1. Admit waiting requests (allocate block tables + prefill serially).
    2. Build one batched decode forward over all running requests.
    3. Complete finished requests and release their blocks.
    """

    def __init__(
        self,
        loaded: LoadedModel,
        paged: PagedKVCache,
        *,
        max_batch: int = 8,
    ) -> None:
        self._loaded = loaded
        self._paged = paged
        self._max_batch = max_batch
        self._queue = RequestQueue()
        self._requests: dict[str, GenerationRequest] = {}
        self._states: dict[str, EngineRequestState] = {}
        self._torch = importlib.import_module("torch")
        self._iteration = 0
        self._history: tuple[IterationStats, ...] = ()

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
        admitted = self._queue.admit(self._max_batch - self._queue.running_count)
        for req in admitted:
            self._init_request_state(req.request_id)

        finished = 0
        cancelled = 0
        try:
            # Serial prefill for newly admitted requests.
            for req in admitted:
                self._prefill_request(req.request_id)

            # Finalize requests that are already done (e.g. EOS at prefill or
            # max_new == 1) BEFORE decoding, so they never enter decode.
            finished += self._finalize_finished()

            running = list(self._queue.running_requests)
            running_ids = [r.request_id for r in running]
            old_lengths = tuple(self._states[rid].cache.seq_length for rid in running_ids)

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
                self._release_state(rid)
                count += 1
        return count

    def _abort_all(self) -> int:
        """Cancel every tracked request and release its blocks.

        Used on prefill/decode failure.  Returns the number of cancelled
        requests.
        """
        count = 0
        import contextlib

        for rid in list(self._states):
            st = self._states[rid]
            with contextlib.suppress(KeyError):
                self._queue.cancel(rid)
            count += 1
            self._paged.release_request(st.cache)
        self._states.clear()
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
        max_tokens = len(self._loaded.tokenizer.encode(req.prompt)) + req.max_new_tokens
        block_table = self._paged.create_block_table(max_tokens)
        blocks_needed = block_table.logical_blocks_used(max_tokens, self._paged.block_size)
        self._paged.allocate_blocks(block_table, blocks_needed)
        paged_state = PagedRequestState(
            request_id=req_id,
            block_table=block_table,
            seq_length=0,
            blocks_allocated=blocks_needed,
        )
        self._states[req_id] = EngineRequestState(
            request=req,
            cache=paged_state,
            prompt_token_ids=(),
        )

    def _prefill_request(self, req_id: str) -> None:
        req = self._requests[req_id]
        st = self._states[req_id]
        encoded = self._loaded.tokenizer(req.prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self._loaded.model.device)
        attention_mask = encoded["attention_mask"].to(self._loaded.model.device)
        cache = PagedTinyinferCache(self._paged, st.cache)
        with self._torch.inference_mode():
            outputs = self._loaded.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        st.prompt_token_ids = tuple(int(t) for t in input_ids[0].tolist())
        st.output_token_ids = [_select_next_token(outputs.logits)]

    def _decode_batch(self, running_ids: list[str]) -> None:
        """Run exactly one forward over all running requests."""
        states = [self._states[rid] for rid in running_ids]
        cache_states = tuple(st.cache for st in states)
        old_lengths = tuple(st.cache.seq_length for st in states)
        batch = len(states)

        input_ids = self._torch.tensor(
            [[st.output_token_ids[-1]] for st in states],
            device=self._loaded.model.device,
            dtype=self._torch.long,
        )
        position_ids = self._torch.tensor(
            [[n] for n in old_lengths],
            device=self._loaded.model.device,
            dtype=self._torch.long,
        )
        max_old = max(old_lengths) if old_lengths else 0
        attention_mask = self._torch.zeros(
            (batch, max_old + 1),
            device=self._loaded.model.device,
            dtype=self._torch.long,
        )
        for row, n in enumerate(old_lengths):
            attention_mask[row, max_old - n :] = 1

        batched_cache = BatchedPagedTinyinferCache(self._paged, cache_states)
        with self._torch.inference_mode():
            outputs = self._loaded.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=batched_cache,
                use_cache=True,
                return_dict=True,
            )
        next_tokens = _select_next_tokens(outputs.logits, batch)
        for rid, token in zip(running_ids, next_tokens, strict=True):
            self._states[rid].output_token_ids.append(token)

    def _release_state(self, req_id: str) -> None:
        """Release any allocated cache blocks for *req_id* and drop state."""
        st = self._states.pop(req_id, None)
        if st is not None:
            self._paged.release_request(st.cache)

    def _is_done(self, req_id: str) -> bool:
        req = self._requests[req_id]
        tokens = self._states[req_id].output_token_ids
        if not tokens:
            return False
        return tokens[-1] == self._loaded.eos_token_id or len(tokens) >= req.max_new_tokens

    def _build_result(self, req_id: str) -> GenerationResult:
        st = self._states[req_id]
        tokens = st.output_token_ids
        stop_reason: StopReason = "eos" if tokens[-1] == self._loaded.eos_token_id else "length"
        text = self._loaded.tokenizer.decode(tokens, skip_special_tokens=True)
        return GenerationResult(
            request_id=req_id,
            prompt_token_ids=st.prompt_token_ids,
            output_token_ids=tuple(tokens),
            text=text,
            stop_reason=stop_reason,
        )
