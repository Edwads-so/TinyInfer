"""Paged KV storage for the owned forward pass, addressed by token slots.

Layout ``[layers, 2, num_blocks, block_size, kv_heads, head_dim]`` keeps each
token's K (or V) for all heads together, so one layer's K is a flat table of
``num_blocks * block_size`` slots and a whole batch's history is one indexed
gather.  Physical block 0 is never allocated: it stays zero and backs every
padding position, so masked-out attention never multiplies by garbage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from tinyinfer.cache.block import BlockAllocator
from tinyinfer.model.spec import Qwen2Spec

Tensor = torch.Tensor


@dataclass(slots=True)
class PagedSequence:
    """One request's physical blocks and committed token count."""

    request_id: str
    blocks: list[int]
    block_size: int
    length: int = 0
    released: bool = field(default=False)

    @property
    def capacity(self) -> int:
        return len(self.blocks) * self.block_size


class PagedKVPool:
    """Shared GPU block pool plus allocator; block 0 is the reserved null block."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        device: Any,
        dtype: Any,
    ) -> None:
        if min(num_layers, num_kv_heads, head_dim, block_size) < 1 or num_blocks < 2:
            raise ValueError("pool dimensions must be positive and num_blocks at least 2")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.buffer = torch.zeros(
            num_layers,
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self._allocator = BlockAllocator(num_blocks)
        if self._allocator.allocate() != 0:  # reserve the null block
            raise RuntimeError("expected the allocator to hand out block 0 first")

    @classmethod
    def for_spec(
        cls, spec: Qwen2Spec, *, block_size: int, num_blocks: int, device: Any, dtype: Any
    ) -> PagedKVPool:
        return cls(
            num_layers=spec.num_layers,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
            dtype=dtype,
        )

    @property
    def null_slot(self) -> int:
        return 0

    @property
    def free_block_count(self) -> int:
        return self._allocator.free_count

    @property
    def device(self) -> Any:
        return self.buffer.device

    def table(self, layer: int, kind: int) -> Tensor:
        """Layer ``layer``'s K (``kind=0``) or V (``kind=1``) as ``[slots, kv_heads, head_dim]``."""
        _, _, blocks, size, heads, dim = self.buffer.shape
        return self.buffer[layer, kind].view(blocks * size, heads, dim)

    def open(self, request_id: str, *, max_tokens: int) -> PagedSequence:
        """Reserve blocks for ``max_tokens`` tokens, all or nothing."""
        needed = -(-max_tokens // self.block_size)
        if needed < 1:
            raise ValueError("max_tokens must be positive")
        if needed > self.free_block_count:
            raise ValueError(
                f"{request_id} needs {needed} blocks but only {self.free_block_count} are free"
            )
        blocks = [self._allocator.allocate() for _ in range(needed)]
        return PagedSequence(request_id=request_id, blocks=blocks, block_size=self.block_size)

    def release(self, sequence: PagedSequence) -> None:
        """Return the sequence's blocks; repeated calls are no-ops."""
        if sequence.released:
            return
        for block in sequence.blocks:
            self._allocator.free(block)
        sequence.released = True

    def slots(self, sequence: PagedSequence, start: int, end: int) -> list[int]:
        """Slot numbers for token positions ``start .. end - 1`` of ``sequence``."""
        if sequence.released:
            raise RuntimeError(f"{sequence.request_id} has been released")
        if end > sequence.capacity:
            raise ValueError(
                f"{sequence.request_id} position {end - 1} exceeds capacity {sequence.capacity}"
            )
        size = self.block_size
        return [sequence.blocks[p // size] * size + p % size for p in range(start, end)]

    def batch(self, sequences: Sequence[PagedSequence], *, new_tokens: int) -> PagedBatchView:
        """A ``KVCache`` view for one forward that adds ``new_tokens`` to every sequence."""
        return PagedBatchView(self, sequences, new_tokens)


class PagedBatchView:
    """Implements ``KVCache`` for one forward over a fixed ordering of sequences.

    Two forms are supported: a fresh prefill (no sequence has history yet) and a
    one-token decode step over sequences of any lengths.  The slot indices are
    built once here and reused by every layer, so each layer costs one indexed
    write and one indexed gather for K and for V, however many blocks there are.
    """

    def __init__(
        self, pool: PagedKVPool, sequences: Sequence[PagedSequence], new_tokens: int
    ) -> None:
        if not sequences:
            raise ValueError("a batch needs at least one sequence")
        ids = [sequence.request_id for sequence in sequences]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate sequences in batch: {ids}")
        for sequence in sequences:
            if sequence.released:
                raise RuntimeError(f"{sequence.request_id} has been released")
        if new_tokens < 1:
            raise ValueError("new_tokens must be positive")
        self._fresh = all(sequence.length == 0 for sequence in sequences)
        if not self._fresh and new_tokens != 1:
            raise NotImplementedError(
                "chunked prefill (several tokens after history) is unsupported"
            )
        for sequence in sequences:
            if sequence.length + new_tokens > sequence.capacity:
                raise ValueError(
                    f"{sequence.request_id} would reach {sequence.length + new_tokens} tokens, "
                    f"over its capacity {sequence.capacity}"
                )

        self._pool = pool
        self._sequences = tuple(sequences)
        self._new_tokens = new_tokens
        self._next_layer = 0
        device = pool.device
        old = [sequence.length for sequence in sequences]
        self.position_ids = torch.tensor(
            [[length + offset for offset in range(new_tokens)] for length in old], device=device
        )
        self._write = torch.tensor(
            [
                slot
                for sequence, length in zip(sequences, old, strict=True)
                for slot in pool.slots(sequence, length, length + new_tokens)
            ],
            device=device,
        )
        self._read: Tensor | None = None
        self.mask: Tensor | None = None
        if not self._fresh:
            totals = [length + new_tokens for length in old]
            width = max(totals)
            rows = [
                pool.slots(sequence, 0, total) + [pool.null_slot] * (width - total)
                for sequence, total in zip(sequences, totals, strict=True)
            ]
            self._read = torch.tensor(rows, device=device)  # [B, S]
            if len(sequences) > 1:
                valid = (
                    torch.arange(width, device=device)[None, :]
                    < torch.tensor(totals, device=device)[:, None]
                )
                self.mask = valid[:, None, None, :]  # [B, 1, 1, S]

    @property
    def length(self) -> int:
        """Longest committed history in the batch (for ``KVCache`` compatibility)."""
        return max(sequence.length for sequence in self._sequences)

    def update(self, layer: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Write this step's ``[B, Hkv, T, D]`` K/V into their slots; return each row's history."""
        if layer != self._next_layer:
            raise RuntimeError(
                f"layer update out of order: got {layer}, expected {self._next_layer}"
            )
        batch, heads, tokens, dim = key.shape
        expected = (len(self._sequences), heads, self._new_tokens, dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"key/value shape {tuple(key.shape)} does not match {expected}")
        keys, values = self._pool.table(layer, 0), self._pool.table(layer, 1)
        keys[self._write] = key.transpose(1, 2).reshape(batch * tokens, heads, dim)
        values[self._write] = value.transpose(1, 2).reshape(batch * tokens, heads, dim)
        self._next_layer = layer + 1
        if self._read is None:
            return key, value  # fresh prefill: the history is exactly this step
        # [B, S, Hkv, D] -> [B, Hkv, S, D]
        return keys[self._read].transpose(1, 2), values[self._read].transpose(1, 2)

    def commit(self, new_tokens: int) -> None:
        """Advance every sequence once all layers have written ``new_tokens`` tokens."""
        layers = self._pool.buffer.shape[0]
        if self._next_layer != layers:
            raise RuntimeError(f"only {self._next_layer} of {layers} layers were updated")
        if new_tokens != self._new_tokens:
            raise RuntimeError(
                f"commit of {new_tokens} tokens, but the view adds {self._new_tokens}"
            )
        for sequence in self._sequences:
            sequence.length += new_tokens
        self._next_layer = 0
