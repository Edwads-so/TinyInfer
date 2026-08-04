"""TinyinferCache — HF Cache protocol backed by TinyInfer ContiguousKVCache."""

from __future__ import annotations

import importlib
from typing import Any

from tinyinfer.cache.contiguous import ContiguousKVCache, allocate_contiguous_cache


class TinyinferCache:
    """HF-compatible cache backed by TinyInfer-owned contiguous KV buffer.

    Implements the ``transformers.Cache`` update / __getitem__ / get_seq_length
    protocol so that HF attention layers read from and write to the TinyInfer
    contiguous buffer directly — no DynamicCache rebuild required.

    We do *not* inherit from ``transformers.Cache`` to avoid pulling in
    ``transformers`` at import time.  The object is duck-type compatible.
    """

    def __init__(self, contiguous: ContiguousKVCache) -> None:
        self._contiguous = contiguous
        self._has_seen_prefill = False
        # Qwen2 attention checks ``hasattr(past_key_value, "key_cache")`` in some
        # code paths.  Expose lazy list-like views.
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
        """Called by HF attention after computing K/V for this layer.

        Returns the *full* cached key and value for this layer so the
        attention module can use them in the forward pass.
        """
        if not self._has_seen_prefill:
            # Prefill: write the complete prompt KV.
            self._contiguous.write_prefill(key_states, value_states, layer_idx)
            if layer_idx == self._contiguous.num_layers - 1:
                self._has_seen_prefill = True
        else:
            # Decode: append one token.
            self._contiguous.write_decode(key_states, value_states, layer_idx)

        full_key, full_value = self._contiguous.read(layer_idx)
        # Keep key_cache / value_cache in sync for code paths that access them.
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(full_key)
            self.value_cache.append(full_value)
        else:
            self.key_cache[layer_idx] = full_key
            self.value_cache[layer_idx] = full_value
        return full_key, full_value

    def __getitem__(self, layer_idx: int) -> tuple[Any, Any]:
        """Return (key, value) for *layer_idx*."""
        return self._contiguous.read(layer_idx)

    def get_seq_length(self) -> int:
        """Return the number of tokens currently cached."""
        return self._contiguous.length

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def contiguous(self) -> ContiguousKVCache:
        return self._contiguous


def allocate_tinyinfer_cache(
    *,
    num_layers: int,
    num_kv_heads: int,
    max_seq_len: int,
    head_dim: int,
    device: Any,
    dtype: Any,
) -> TinyinferCache:
    """Shortcut: allocate a contiguous buffer and wrap in TinyinferCache."""
    return TinyinferCache(
        allocate_contiguous_cache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
    )