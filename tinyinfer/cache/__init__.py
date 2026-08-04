"""TinyInfer KV cache implementations."""

from tinyinfer.cache.contiguous import ContiguousKVCache, allocate_contiguous_cache
from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.cache.tinyinfer_cache import TinyinferCache, allocate_tinyinfer_cache

__all__ = [
    "ContiguousKVCache",
    "PagedKVCache",
    "TinyinferCache",
    "allocate_contiguous_cache",
    "allocate_tinyinfer_cache",
]
