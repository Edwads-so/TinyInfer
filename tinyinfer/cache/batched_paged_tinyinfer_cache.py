"""BatchedPagedTinyinferCache — ephemeral heterogeneous HF cache view."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.cache.request_state import PagedRequestState


class BatchedPagedTinyinferCache:
    """HF-compatible cache view over several request states (one decode step).

    Frozen membership at construction.  Each ``update()`` writes the one decode
    KV token per request into that request's block table, then returns a
    left-padded dense gather for the whole batch.  Committed lengths advance
    only after the final layer's gather succeeds.
    """

    def __init__(self, paged: PagedKVCache, states: Sequence[PagedRequestState]) -> None:
        if not states:
            raise ValueError("at least one request state is required")
        self._paged = paged
        self._states: tuple[PagedRequestState, ...] = tuple(states)
        self._ids = tuple(s.request_id for s in self._states)
        if len(set(self._ids)) != len(self._ids):
            raise ValueError("duplicate request ids in batch")
        expected_geometry = (
            paged.num_layers,
            paged.num_kv_heads,
            paged.block_size,
            paged.head_dim,
        )
        for s in self._states:
            if s.released:
                raise ValueError(f"request {s.request_id!r} cache state is released")
            geom = getattr(s.block_table, "pool_geometry", None)
            if geom != expected_geometry:
                raise ValueError(
                    f"request {s.request_id!r} block table geometry {geom} "
                    f"does not match pool geometry {expected_geometry}"
                )
        self._old_lengths: tuple[int, ...] = tuple(s.seq_length for s in self._states)
        self._layer = -1
        self.key_cache: list[Any] = []
        self.value_cache: list[Any] = []

    @property
    def request_ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def old_lengths(self) -> tuple[int, ...]:
        return self._old_lengths

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: Any = None,
    ) -> tuple[Any, Any]:
        # Enforce strict layer ordering.
        if layer_idx != self._layer + 1:
            raise RuntimeError(
                f"layer update out of order: got {layer_idx}, expected {self._layer + 1}"
            )
        if layer_idx < 0 or layer_idx >= self._paged.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range")
        if key_states.shape[2] != 1:
            raise ValueError("batched decode query length must be 1")

        old_lengths = self._old_lengths
        tables = [s.block_table for s in self._states]
        self._paged.write_decode_batch(
            key_states, value_states, layer_idx, tables, list(old_lengths)
        )
        gathered, gathered_v = self._paged.gather_batch(
            layer_idx, tables, [n + 1 for n in old_lengths]
        )
        if layer_idx == self._paged.num_layers - 1:
            for s, old in zip(self._states, old_lengths, strict=True):
                s.seq_length = old + 1
        self._layer = layer_idx
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(gathered)
            self.value_cache.append(gathered_v)
        else:
            self.key_cache[layer_idx] = gathered
            self.value_cache[layer_idx] = gathered_v
        return gathered, gathered_v

    def __getitem__(self, layer_idx: int) -> tuple[Any, Any]:
        return self._paged.gather_batch(
            layer_idx,
            [s.block_table for s in self._states],
            [s.seq_length for s in self._states],
        )

    def get_seq_length(self) -> int:
        return max(s.seq_length for s in self._states)

    def get_query_offset(self, _layer_idx: int = 0) -> int:
        return self.get_seq_length()

    def get_mask_sizes(self, q_length: int, _layer_idx: int = 0) -> tuple[int, int]:
        return (self.get_seq_length() + q_length, 0)
