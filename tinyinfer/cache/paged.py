"""PagedKVCache — block-level paged KV storage for continuous batching."""

from __future__ import annotations

import importlib
from typing import Any

from tinyinfer.cache.block import BlockAllocator, BlockTable


class PagedKVCache:
    """Physical block pool + per-request block table for paged KV storage.

    Physical layout:
      [num_layers, 2, num_physical_blocks, num_kv_heads, block_size, head_dim]
      where dim 1 index 0 = key, 1 = value.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        num_physical_blocks: int,
        device: Any,
        dtype: Any,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if num_kv_heads < 1:
            raise ValueError("num_kv_heads must be positive")
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        if num_physical_blocks < 1:
            raise ValueError("num_physical_blocks must be positive")

        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._block_size = block_size
        self._head_dim = head_dim
        self._num_physical_blocks = num_physical_blocks

        torch = importlib.import_module("torch")
        self._buffer = torch.zeros(
            num_layers,
            2,
            num_physical_blocks,
            num_kv_heads,
            block_size,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self._allocator = BlockAllocator(num_physical_blocks)
        self._device = device
        self._dtype = dtype

    # -- properties -------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def num_physical_blocks(self) -> int:
        return self._num_physical_blocks

    @property
    def device(self) -> Any:
        return self._device

    @property
    def free_block_count(self) -> int:
        return self._allocator.free_count

    # -- block table ------------------------------------------------------

    def create_block_table(self, max_tokens: int) -> BlockTable:
        """Allocate a block table for one request."""
        max_logical = (max_tokens + self._block_size - 1) // self._block_size
        return BlockTable(self._num_layers, max_logical)

    def allocate_blocks(self, block_table: BlockTable, num_blocks: int) -> list[int]:
        """Atomically allocate *num_blocks* physical blocks and map them.

        If the pool cannot satisfy the full request, any blocks acquired by
        this call are returned to the pool and every mapping written by this
        call is cleared before re-raising.
        """
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if num_blocks > self._allocator.free_count:
            raise IndexError(f"need {num_blocks} blocks but only {self._allocator.free_count} free")
        physical: list[int] = []
        try:
            for _ in range(num_blocks):
                p = self._allocator.allocate()
                physical.append(p)
                for layer in range(self._num_layers):
                    block_table.set(layer, len(physical) - 1, p)
        except Exception:
            # Roll back any blocks acquired so far and clear this call's mappings.
            import contextlib

            for p in physical:
                with contextlib.suppress(ValueError):
                    self._allocator.free(p)
            for layer in range(self._num_layers):
                for logical in range(len(physical)):
                    block_table.clear(layer, logical)
            raise
        return physical

    def free_blocks(self, block_table: BlockTable, num_blocks: int) -> None:
        """Free the physical blocks mapped in logical blocks 0..num_blocks-1.

        Each mapped physical block is freed once and its mapping cleared.  A
        repeated pool-level free (already -1) is a no-op.
        """
        if num_blocks < 1:
            return
        if num_blocks > block_table.max_logical_blocks:
            raise ValueError("num_blocks exceeds block table capacity")
        # Collect the unique physical blocks mapped across all layers so each
        # is released exactly once (an instance is shared across layers).
        to_free: set[int] = set()
        for layer in range(self._num_layers):
            for logical in range(num_blocks):
                p = block_table.get(layer, logical)
                if p >= 0:
                    to_free.add(p)
        for p in to_free:
            self._allocator.free(p)
        # Clear the mappings in every layer after releasing the pool block.
        for layer in range(self._num_layers):
            for logical in range(num_blocks):
                block_table.clear(layer, logical)

    # -- KV I/O -----------------------------------------------------------

    def write_prefill(
        self,
        key: Any,
        value: Any,
        layer_idx: int,
        block_table: BlockTable,
        tokens: int,
    ) -> None:
        """Write prefill KV for one layer using block table.

        ``key`` / ``value``: ``[1, num_kv_heads, tokens, head_dim]``
        """
        prompt_len = int(key.shape[2])
        if prompt_len < 1:
            raise ValueError("prefill key must have at least one token")
        blocks_needed = (prompt_len + self._block_size - 1) // self._block_size
        for logical in range(blocks_needed):
            p = block_table.get(layer_idx, logical)
            if p < 0:
                raise RuntimeError(f"block table missing physical block for logical {logical}")
            start = logical * self._block_size
            end = min(start + self._block_size, prompt_len)
            span = end - start
            self._buffer[layer_idx, 0, p, :, :span, :].copy_(key[0, :, start:end, :])
            self._buffer[layer_idx, 1, p, :, :span, :].copy_(value[0, :, start:end, :])

    def write_decode(
        self,
        key: Any,
        value: Any,
        layer_idx: int,
        block_table: BlockTable,
        position: int,
    ) -> None:
        """Append one decode token KV using block table."""
        if key.shape[2] != 1 or value.shape[2] != 1:
            raise ValueError("decode key/value must have exactly one token")
        logical = position // self._block_size
        offset = position % self._block_size
        p = block_table.get(layer_idx, logical)
        if p < 0:
            raise RuntimeError(f"block table missing physical block for position {position}")
        self._buffer[layer_idx, 0, p, :, offset, :].copy_(key[0, :, 0, :])
        self._buffer[layer_idx, 1, p, :, offset, :].copy_(value[0, :, 0, :])

    def read(
        self,
        layer_idx: int,
        block_table: BlockTable,
        tokens: int,
    ) -> tuple[Any, Any]:
        """Read full contiguous KV for one layer by gathering blocks.

        Returns ``(key, value)`` shapes ``[1, num_kv_heads, tokens, head_dim]``.
        """
        torch = importlib.import_module("torch")
        key_out = torch.empty(
            1,
            self._num_kv_heads,
            tokens,
            self._head_dim,
            device=self._device,
            dtype=self._dtype,
        )
        value_out = torch.empty_like(key_out)
        blocks_needed = (tokens + self._block_size - 1) // self._block_size
        for logical in range(blocks_needed):
            p = block_table.get(layer_idx, logical)
            if p < 0:
                raise RuntimeError(f"block table missing physical block for logical {logical}")
            start = logical * self._block_size
            end = min(start + self._block_size, tokens)
            span = end - start
            key_out[0, :, start:end, :] = self._buffer[layer_idx, 0, p, :, :span, :]
            value_out[0, :, start:end, :] = self._buffer[layer_idx, 1, p, :, :span, :]
        return key_out, value_out
