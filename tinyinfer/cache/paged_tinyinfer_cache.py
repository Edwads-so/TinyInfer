"""PagedTinyinferCache — HF Cache protocol backed by PagedKVCache + request state."""

from __future__ import annotations

from typing import Any

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.cache.request_state import PagedRequestState


class PagedTinyinferCache:
    """HF-compatible cache backed by TinyInfer paged block pool + request state.

    Implements the ``update()`` / ``__getitem__()`` / ``get_seq_length()``
    protocol expected by HF attention layers.  All length/table ownership is
    delegated to the shared ``PagedRequestState``.
    """

    def __init__(self, paged: PagedKVCache, state: PagedRequestState) -> None:
        state.require_active()
        self._paged = paged
        self._state = state
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
        self._state.require_active()
        if not self._has_seen_prefill:
            prompt_len = int(key_states.shape[2])
            self._paged.write_prefill(
                key_states, value_states, layer_idx, self._state.block_table, prompt_len
            )
            full_key, full_value = self._paged.read(layer_idx, self._state.block_table, prompt_len)
            if layer_idx == self._paged.num_layers - 1:
                self._has_seen_prefill = True
                self._state.seq_length = prompt_len
        else:
            self._paged.write_decode(
                key_states, value_states, layer_idx, self._state.block_table, self._state.seq_length
            )
            full_key, full_value = self._paged.read(
                layer_idx, self._state.block_table, self._state.seq_length + 1
            )
            if layer_idx == self._paged.num_layers - 1:
                self._state.seq_length += 1

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(full_key)
            self.value_cache.append(full_value)
        else:
            self.key_cache[layer_idx] = full_key
            self.value_cache[layer_idx] = full_value
        return full_key, full_value

    def __getitem__(self, layer_idx: int) -> tuple[Any, Any]:
        """Return (key, value) for *layer_idx*."""
        self._state.require_active()
        return self._paged.read(layer_idx, self._state.block_table, self._state.seq_length)

    def get_seq_length(self) -> int:
        """Return the number of tokens currently cached."""
        self._state.require_active()
        return self._state.seq_length

    def get_query_offset(self, _layer_idx: int = 0) -> int:
        """Return the committed sequence length as query offset (Transformers 5.x+).

        For incremental decode the query starts after the cached prefix, so the
        offset equals the committed length.
        """
        self._state.require_active()
        return self._state.seq_length

    def get_mask_sizes(self, q_length: int, _layer_idx: int = 0) -> tuple[int, int]:
        """Return (kv_length + q_length, 0) for mask sizing (Transformers 5.x+)."""
        self._state.require_active()
        return (self._state.seq_length + q_length, 0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def paged(self) -> PagedKVCache:
        return self._paged

    @property
    def state(self) -> PagedRequestState:
        return self._state

    @property
    def block_table(self) -> Any:
        return self._state.block_table

    @property
    def length(self) -> int:
        self._state.require_active()
        return self._state.seq_length
