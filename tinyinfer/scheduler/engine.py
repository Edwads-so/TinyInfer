"""FCFS scheduling engine with per-request paged KV cache lifecycle."""

from __future__ import annotations

import importlib
from typing import Any

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.cache.paged_tinyinfer_cache import PagedTinyinferCache
from tinyinfer.runtime.generation import _select_next_token
from tinyinfer.runtime.hf_loader import LoadedModel
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult, StopReason
from tinyinfer.scheduler.queue import RequestQueue

_StateEntry = dict[str, Any]


class BatchEngine:
    """FCFS scheduler + paged KV cache lifecycle for multiple requests.

    Each request gets a private ``BlockTable`` and shares the global
    ``PagedKVCache`` block pool.  One ``step()`` runs:

    1. Admit waiting requests (allocate block tables).
    2. Prefill newly admitted requests.
    3. Decode all running requests.
    4. Complete finished requests (free blocks).
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
        self._states: dict[str, _StateEntry] = {}
        self._torch = importlib.import_module("torch")

    @property
    def queue(self) -> RequestQueue:
        return self._queue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: GenerationRequest) -> None:
        """Enqueue one request."""
        self._requests[request.request_id] = request
        self._queue.enqueue(request)

    def step(self) -> int:
        """Run one scheduling iteration.  Returns finished count."""
        admitted = self._queue.admit(self._max_batch - self._queue.running_count)
        for req in admitted:
            self._init_request_state(req.request_id)

        for req in admitted:
            self._prefill_request(req.request_id)

        for req in self._queue.running_requests:
            self._decode_request(req.request_id)

        finished = 0
        for req in list(self._queue.running_requests):
            rid = req.request_id
            state = self._states[rid]
            if self._is_done(rid):
                result = self._build_result(rid)
                self._queue.mark_finished(rid, result)
                self._paged.free_blocks(state["block_table"], state["blocks_allocated"])
                del self._states[rid]
                finished += 1
        return finished

    def run(self) -> list[GenerationResult]:
        """Run steps until all requests finish.  Returns all results."""
        while not self._queue.all_done():
            self.step()
        return self._queue.drain_finished()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_request_state(self, req_id: str) -> None:
        req = self._requests[req_id]
        max_tokens = len(self._loaded.tokenizer.encode(req.prompt)) + req.max_new_tokens
        block_table = self._paged.create_block_table(max_tokens)
        blocks_needed = block_table.logical_blocks_used(max_tokens, self._paged.block_size)
        self._paged.allocate_blocks(block_table, blocks_needed)
        self._states[req_id] = {
            "block_table": block_table,
            "blocks_allocated": blocks_needed,
            "tinfer_cache": PagedTinyinferCache(self._paged, block_table),
            "attention_mask": None,
            "output_token_ids": [],
        }

    def _prefill_request(self, req_id: str) -> None:
        req = self._requests[req_id]
        state = self._states[req_id]
        encoded = self._loaded.tokenizer(
            req.prompt, return_tensors="pt", add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(self._loaded.model.device)
        attention_mask = encoded["attention_mask"].to(self._loaded.model.device)

        with self._torch.inference_mode():
            outputs = self._loaded.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=state["tinfer_cache"],
                use_cache=True,
                return_dict=True,
            )
        state["output_token_ids"] = [_select_next_token(outputs.logits)]
        state["attention_mask"] = attention_mask

    def _decode_request(self, req_id: str) -> None:
        state = self._states[req_id]
        if self._is_done(req_id):
            return
        tokens = state["output_token_ids"]
        last_token = tokens[-1]
        old_length = state["tinfer_cache"].length
        attn_mask = state["attention_mask"]

        input_ids_step = self._torch.tensor(
            [[last_token]], device=self._loaded.model.device, dtype=self._torch.long
        )
        extended_mask = self._torch.cat(
            [
                attn_mask,
                self._torch.ones((1, 1), device=attn_mask.device, dtype=attn_mask.dtype),
            ],
            dim=1,
        )
        cache_position = self._torch.arange(
            old_length, old_length + 1, device=self._loaded.model.device
        )

        with self._torch.inference_mode():
            outputs = self._loaded.model(
                input_ids=input_ids_step,
                attention_mask=extended_mask,
                past_key_values=state["tinfer_cache"],
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        state["output_token_ids"].append(_select_next_token(outputs.logits))
        state["attention_mask"] = extended_mask

    def _is_done(self, req_id: str) -> bool:
        req = self._requests[req_id]
        tokens = self._states[req_id]["output_token_ids"]
        if not tokens:
            return False
        return tokens[-1] == self._loaded.eos_token_id or len(tokens) >= req.max_new_tokens

    def _build_result(self, req_id: str) -> GenerationResult:
        req = self._requests[req_id]
        state = self._states[req_id]
        tokens = state["output_token_ids"]
        stop_reason: StopReason = (
            "eos" if tokens[-1] == self._loaded.eos_token_id else "length"
        )
        encoded = self._loaded.tokenizer(
            req.prompt, return_tensors="pt", add_special_tokens=True
        )
        prompt_ids = tuple(int(t) for t in encoded["input_ids"][0].tolist())
        text = self._loaded.tokenizer.decode(tokens, skip_special_tokens=True)
        return GenerationResult(
            request_id=req_id,
            prompt_token_ids=prompt_ids,
            output_token_ids=tuple(tokens),
            text=text,
            stop_reason=stop_reason,
        )