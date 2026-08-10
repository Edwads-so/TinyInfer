"""PagedTinyinferCache — HF Cache protocol backed by PagedKVCache."""

from __future__ import annotations

from typing import Any

from tinyinfer.cache.block import BlockTable
from tinyinfer.cache.paged import PagedKVCache


class PagedTinyinferCache:
    """HF-compatible cache backed by TinyInfer paged block pool + block table.

    Implements the ``update()`` / ``__getitem__()`` / ``get_seq_length()``
    protocol expected by HF attention layers.
    """

    def __init__(self, paged: PagedKVCache, block_table: BlockTable) -> None:
        self._paged = paged
        self._block_table = block_table
        self._length = 0
        self._has_seen_prefill = False
        self.key_cache: list[Any] = []
        self.value_cache: list[Any] = []

    # ------------------------------------------------------------------
    # HF Cache protocol
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: Any = None,
    ) -> tuple[Any, Any]:
        """Called by HF attention after computing K/V for this layer."""
        if not self._has_seen_prefill:
            prompt_len = int(key_states.shape[2])
            self._paged.write_prefill(
                key_states, value_states, layer_idx, self._block_table, prompt_len
            )
            # All layers read with prompt_len, but only the last one
            # commits it.
            full_key, full_value = self._paged.read(layer_idx, self._block_table, prompt_len)
            if layer_idx == self._paged.num_layers - 1:
                self._has_seen_prefill = True
                self._length = prompt_len
        else:
            self._paged.write_decode(
                key_states, value_states, layer_idx, self._block_table, self._length
            )
            full_key, full_value = self._paged.read(layer_idx, self._block_table, self._length + 1)
            if layer_idx == self._paged.num_layers - 1:
                self._length += 1

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(full_key)
            self.value_cache.append(full_value)
        else:
            self.key_cache[layer_idx] = full_key
            self.value_cache[layer_idx] = full_value
        return full_key, full_value

    def __getitem__(self, layer_idx: int) -> tuple[Any, Any]:
        """Return (key, value) for *layer_idx*."""
        return self._paged.read(layer_idx, self._block_table, self._length)

    def get_seq_length(self) -> int:
        """Return the number of tokens currently cached."""
        return self._length

    def get_query_offset(self, _layer_idx: int = 0) -> int:
        """Return the query offset for causal mask creation (Transformers 5.x+)."""
        return 0

    def get_mask_sizes(self, q_length: int, _layer_idx: int = 0) -> tuple[int, int]:
        """Return (kv_length + q_length, 0) for mask sizing (Transformers 5.x+)."""
        return (self._length + q_length, 0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def paged(self) -> PagedKVCache:
        return self._paged

    @property
    def block_table(self) -> BlockTable:
        return self._block_table

    @property
    def length(self) -> int:
        return self._length
