"""Per-request paged KV cache state shared by scheduler and adapters."""

from __future__ import annotations

from dataclasses import dataclass

from tinyinfer.cache.block import BlockTable


@dataclass(slots=True)
class PagedRequestState:
    """Single source of truth for one request's paged cache lifecycle.

    Carries the block table, committed sequence length, allocation count,
    and released flag so scheduler and cache adapters share one instance.
    """

    request_id: str
    block_table: BlockTable
    seq_length: int
    blocks_allocated: int
    released: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-blank string")
        if type(self.seq_length) is not int or self.seq_length < 0:
            raise ValueError("seq_length must be a non-negative integer")
        if type(self.blocks_allocated) is not int or self.blocks_allocated < 0:
            raise ValueError("blocks_allocated must be a non-negative integer")

    def require_active(self) -> None:
        """Raise if this state has been released."""
        if self.released:
            raise RuntimeError(f"request {self.request_id!r} cache state is released")

    def commit_tokens(self, count: int) -> None:
        """Advance the committed sequence length by a positive token count."""
        self.require_active()
        if type(count) is not int or count <= 0:
            raise ValueError("commit_tokens count must be a positive integer")
        self.seq_length += count

    def mark_released(self) -> None:
        """Idempotently mark this state released."""
        self.released = True
