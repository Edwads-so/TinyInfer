"""Model backends for ``BatchEngine``: how a prefill and a batched decode actually run.

The engine owns scheduling (admission, stop conditions, cancellation, cleanup,
statistics); a backend owns the model and its KV storage.  Each request gets an
opaque handle from ``open`` and returns it through ``release``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Protocol

from tinyinfer.cache.batched_paged_tinyinfer_cache import BatchedPagedTinyinferCache
from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.cache.paged_tinyinfer_cache import PagedTinyinferCache
from tinyinfer.cache.request_state import PagedRequestState
from tinyinfer.runtime.generation import _select_next_token, _select_next_tokens
from tinyinfer.runtime.hf_loader import LoadedModel
from tinyinfer.runtime.request import GenerationRequest


class DecodeBackend(Protocol):
    """What ``BatchEngine`` needs from a model and its KV cache."""

    @property
    def eos_token_id(self) -> int: ...

    def open(self, request: GenerationRequest) -> Any:
        """Reserve KV storage for ``request``; raise without side effects if it cannot."""
        ...

    def prefill(self, request: GenerationRequest, handle: Any) -> tuple[tuple[int, ...], int]:
        """Run the prompt; return its token ids and the first generated token."""
        ...

    def decode(self, handles: Sequence[Any], last_tokens: Sequence[int]) -> list[int]:
        """Run one forward over all ``handles``; return one next token per handle, in order."""
        ...

    def seq_length(self, handle: Any) -> int: ...

    def release(self, handle: Any) -> None:
        """Return the handle's KV storage; repeated calls are no-ops."""
        ...

    def decode_text(self, tokens: Sequence[int]) -> str: ...


class HFPagedBackend:
    """Hugging Face Qwen with TinyInfer's dense-gather paged adapter (the G2-P1 baseline path)."""

    def __init__(self, loaded: LoadedModel, paged: PagedKVCache) -> None:
        self.loaded = loaded
        self.paged = paged
        self._torch = importlib.import_module("torch")

    @property
    def eos_token_id(self) -> int:
        return self.loaded.eos_token_id

    def open(self, request: GenerationRequest) -> PagedRequestState:
        max_tokens = len(self.loaded.tokenizer.encode(request.prompt)) + request.max_new_tokens
        block_table = self.paged.create_block_table(max_tokens)
        blocks_needed = block_table.logical_blocks_used(max_tokens, self.paged.block_size)
        self.paged.allocate_blocks(block_table, blocks_needed)
        return PagedRequestState(
            request_id=request.request_id,
            block_table=block_table,
            seq_length=0,
            blocks_allocated=blocks_needed,
        )

    def prefill(
        self, request: GenerationRequest, handle: PagedRequestState
    ) -> tuple[tuple[int, ...], int]:
        model = self.loaded.model
        encoded = self.loaded.tokenizer(
            request.prompt, return_tensors="pt", add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        cache = PagedTinyinferCache(self.paged, handle)
        with self._torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        prompt_ids = tuple(int(t) for t in input_ids[0].tolist())
        return prompt_ids, _select_next_token(outputs.logits)

    def decode(self, handles: Sequence[PagedRequestState], last_tokens: Sequence[int]) -> list[int]:
        torch, model = self._torch, self.loaded.model
        old_lengths = tuple(state.seq_length for state in handles)
        batch = len(handles)
        input_ids = torch.tensor(
            [[token] for token in last_tokens], device=model.device, dtype=torch.long
        )
        position_ids = torch.tensor(
            [[n] for n in old_lengths], device=model.device, dtype=torch.long
        )
        # Left padding: every row's newest token sits in the last column.
        max_old = max(old_lengths)
        attention_mask = torch.zeros((batch, max_old + 1), device=model.device, dtype=torch.long)
        for row, n in enumerate(old_lengths):
            attention_mask[row, max_old - n :] = 1
        batched_cache = BatchedPagedTinyinferCache(self.paged, tuple(handles))
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=batched_cache,
                use_cache=True,
                return_dict=True,
            )
        return list(_select_next_tokens(outputs.logits, batch))

    def seq_length(self, handle: PagedRequestState) -> int:
        return handle.seq_length

    def release(self, handle: PagedRequestState) -> None:
        self.paged.release_request(handle)

    def decode_text(self, tokens: Sequence[int]) -> str:
        return str(self.loaded.tokenizer.decode(list(tokens), skip_special_tokens=True))
