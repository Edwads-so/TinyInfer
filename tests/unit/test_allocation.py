"""Randomized BlockAllocator + PagedKVCache invariant tests.

These tests verify the most critical property of the paged KV cache:
the block allocator never double-allocates and never leaks blocks,
even under random allocate/free churn.
"""

import random

import pytest

from tinyinfer.cache.block import BlockAllocator, BlockTable


class TestBlockAllocatorInvariants:
    """Deterministic randomized allocation invariants (seed-controlled)."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 1234, 9999])
    def test_random_churn_never_leaks_or_double_allocates(self, seed: int) -> None:
        rng = random.Random(seed)
        total = 64
        allocator = BlockAllocator(total)
        allocated: set[int] = set()

        for _ in range(10_000):
            if rng.random() < 0.6 and allocator.free_count > 0:
                # allocate
                block = allocator.allocate()
                assert block not in allocated, (
                    f"double allocation of block {block} at seed {seed}"
                )
                allocated.add(block)
            else:
                # free an existing block
                if allocated:
                    block = allocated.pop()
                    allocator.free(block)

            # Invariant: no leaks, no lost blocks
            assert (
                allocator.free_count + len(allocated)
            ) == total, (
                f"leak/double: free={allocator.free_count}, "
                f"used={len(allocated)}, total={total} (seed {seed})"
            )

    def test_exhaustion_raises_index_error(self) -> None:
        allocator = BlockAllocator(8)
        for _ in range(8):
            allocator.allocate()
        with pytest.raises(IndexError, match="no free blocks"):
            allocator.allocate()

    def test_freed_block_is_reusable(self) -> None:
        allocator = BlockAllocator(4)
        a, b, c, d = (allocator.allocate() for _ in range(4))
        assert {a, b, c, d} == {0, 1, 2, 3}

        allocator.free(b)
        assert allocator.free_count == 1
        new_block = allocator.allocate()
        assert new_block == b  # FIFO: returns the most recently freed

    def test_allocator_counts_reflect_state(self) -> None:
        allocator = BlockAllocator(10)
        assert allocator.free_count == 10
        assert allocator.allocated_count == 0

        blocks = [allocator.allocate() for _ in range(4)]
        assert allocator.free_count == 6
        assert allocator.allocated_count == 4

        allocator.free(blocks[0])
        allocator.free(blocks[1])
        assert allocator.allocated_count == 2

    def test_free_invalid_negative_raises(self) -> None:
        allocator = BlockAllocator(4)
        with pytest.raises(ValueError):
            allocator.free(-1)


class TestBlockTableInvariants:
    def test_logical_block_mapping_roundtrip(self) -> None:
        table = BlockTable(num_layers=3, max_logical_blocks=8)
        assert table.get(0, 0) == -1

        table.set(0, 0, 42)
        assert table.get(0, 0) == 42
        assert table.get(1, 0) == -1  # other layers unaffected

    def test_logical_blocks_used_math(self) -> None:
        table = BlockTable(num_layers=2, max_logical_blocks=16)
        assert table.logical_blocks_used(0, block_size=16) == 0
        assert table.logical_blocks_used(1, block_size=16) == 1
        assert table.logical_blocks_used(16, block_size=16) == 1
        assert table.logical_blocks_used(17, block_size=16) == 2
        assert table.logical_blocks_used(32, block_size=16) == 2
        assert table.logical_blocks_used(33, block_size=16) == 3


class TestPagedKVCacheAllocationBalance:
    """End-to-end block accounting through PagedKVCache (no torch data)."""

    def test_multiple_block_tables_share_pool_without_leak(self) -> None:
        from tinyinfer.cache.paged import PagedKVCache

        # Use a tiny fake torch via monkeypatch-free shape check: construct with
        # the real torch import but on CPU with tiny dims.  This only checks
        # allocator bookkeeping, not tensor data (uses torch.zeros which is
        # available even on CPU-only environments).
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")

        paged = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            block_size=4,
            head_dim=8,
            num_physical_blocks=32,
            device="cpu",
            dtype="bfloat16",
        )
        assert paged.free_block_count == 32

        # request 1: needs 5 blocks across all layers
        t1 = paged.create_block_table(max_tokens=20)
        b1 = paged.allocate_blocks(t1, 5)
        # all 5 blocks are distinct and mapped to every layer
        assert len(set(b1)) == 5
        for layer in range(2):
            assert t1.get(layer, 0) == b1[0]

        assert paged.free_block_count == 27

        # request 2: needs 3 blocks
        t2 = paged.create_block_table(max_tokens=12)
        b2 = paged.allocate_blocks(t2, 3)
        assert len(set(b2)) == 3
        # no overlap with request 1
        assert set(b1).isdisjoint(b2)
        assert paged.free_block_count == 24

        # free request 2 → blocks return to pool
        paged.free_blocks(t2, 3)
        assert paged.free_block_count == 27

        # free request 1
        paged.free_blocks(t1, 5)
        assert paged.free_block_count == 32

    def test_chained_requests_do_not_accumulate_allocations(self) -> None:
        from tinyinfer.cache.paged import PagedKVCache

        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")

        paged = PagedKVCache(
            num_layers=1,
            num_kv_heads=1,
            block_size=4,
            head_dim=4,
            num_physical_blocks=16,
            device="cpu",
            dtype="bfloat16",
        )

        for _ in range(50):
            table = paged.create_block_table(max_tokens=16)
            blocks = paged.allocate_blocks(table, 4)
            assert len(blocks) == 4
            paged.free_blocks(table, 4)
            assert paged.free_block_count == 16  # no accumulation
