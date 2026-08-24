"""Adapter between Hugging Face DynamicCache and TinyInfer ContiguousKVCache."""

from __future__ import annotations

import importlib
from typing import Any

from tinyinfer.cache.contiguous import ContiguousKVCache


class CacheAdapter:
    """Bidirectional bridge between HF DynamicCache and TinyInfer ContiguousKVCache."""

    def __init__(self, contiguous: ContiguousKVCache) -> None:
        self._contiguous = contiguous

    def import_from_dynamic_cache(self, dynamic_cache: Any) -> None:
        """Copy every layer's KV from a HF DynamicCache into the contiguous buffer."""
        transformers = importlib.import_module("transformers")
        Cache = transformers.Cache

        if not isinstance(dynamic_cache, Cache):
            raise TypeError(
                f"expected a transformers Cache instance, got {type(dynamic_cache).__name__}"
            )
        for layer_idx in range(self._contiguous.num_layers):
            key, value = dynamic_cache[layer_idx]  # calls __getitem__
            self._contiguous.write_prefill(key, value, layer_idx)

    def export_to_dynamic_cache(self, *, maximum_length: int = 0) -> Any:
        """Build a fresh HF DynamicCache populated from the contiguous buffer."""
        transformers = importlib.import_module("transformers")
        DynamicCache = transformers.DynamicCache

        cache = DynamicCache()
        for layer_idx in range(self._contiguous.num_layers):
            key, value = self._contiguous.read(layer_idx)
            cache.update(key, value, layer_idx)
        if maximum_length and hasattr(cache, "max_cache_length"):
            cache.max_cache_length = maximum_length
        return cache

    def append_decode(
        self,
        dynamic_cache: Any,
    ) -> None:
        """Extract the newly appended KV from DynamicCache and write into contiguous.

        Assumes the contiguous buffer already holds ``length - 1`` tokens, and the
        DynamicCache was constructed with those tokens before the decode forward.
        After the forward, DynamicCache holds one extra token per layer at position
        ``contiguous.length``.
        """
        for layer_idx in range(self._contiguous.num_layers):
            key, value = dynamic_cache[layer_idx]
            # The last position is the new decode token.
            new_key = key[:, :, -1:, :]
            new_value = value[:, :, -1:, :]
            assert int(new_key.shape[2]) == 1
            assert int(new_value.shape[2]) == 1
            self._contiguous.write_decode(new_key, new_value, layer_idx)

    @property
    def contiguous(self) -> ContiguousKVCache:
        return self._contiguous
