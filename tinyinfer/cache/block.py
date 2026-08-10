"""Block, BlockTable, and BlockAllocator for paged KV cache."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class BlockTable:
    """Per-request, per-layer logical-to-physical block mapping.

    ``table[layer_idx][logical_block_idx] = physical_block_idx``
    """

    num_layers: int
    max_logical_blocks: int

    def __post_init__(self) -> None:
        self._table: list[list[int]] = [
            [-1] * self.max_logical_blocks for _ in range(self.num_layers)
        ]

    def get(self, layer_idx: int, logical_block: int) -> int:
        """Return the physical block index, or -1 if not allocated."""
        return self._table[layer_idx][logical_block]

    def set(self, layer_idx: int, logical_block: int, physical_block: int) -> None:
        """Map a logical block to a physical block."""
        self._table[layer_idx][logical_block] = physical_block

    def clear(self, layer_idx: int, logical_block: int) -> None:
        """Unmap one logical block entry (set to -1)."""
        self._table[layer_idx][logical_block] = -1

    def clear_all(self) -> None:
        """Unmap every logical block across all layers."""
        for layer in self._table:
            for i in range(len(layer)):
                layer[i] = -1

    def logical_blocks_used(self, tokens: int, block_size: int) -> int:
        """Number of logical blocks needed for *tokens*."""
        if tokens == 0:
            return 0
        return (tokens + block_size - 1) // block_size


class BlockAllocator:
    """Free-list based physical block allocator with ownership tracking.

    Tracks which blocks are currently allocated so freeing an unknown or
    already-free block raises ``ValueError`` instead of corrupting the pool.
    """

    def __init__(self, total_blocks: int) -> None:
        if total_blocks < 1:
            raise ValueError("total_blocks must be positive")
        self._total_blocks = total_blocks
        self._free: deque[int] = deque(range(total_blocks))
        self._allocated: set[int] = set()

    def allocate(self) -> int:
        """Reserve one physical block.  Raises IndexError if none available."""
        if not self._free:
            raise IndexError("no free blocks available")
        block_idx = self._free.popleft()
        self._allocated.add(block_idx)
        return block_idx

    def free(self, block_idx: int) -> None:
        """Return a physical block to the free list.

        Raises ``ValueError`` when the block is out of range or not currently
        allocated (double-free).
        """
        if block_idx < 0 or block_idx >= self._total_blocks:
            raise ValueError(f"block_idx {block_idx} out of range [0, {self._total_blocks})")
        if block_idx not in self._allocated:
            raise ValueError(f"block {block_idx} is not allocated")
        self._allocated.remove(block_idx)
        self._free.append(block_idx)

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def allocated_count(self) -> int:
        return len(self._allocated)

    @property
    def total_blocks(self) -> int:
        return self._total_blocks
