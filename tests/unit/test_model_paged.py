import pytest
import torch

from tinyinfer.model.paged import PagedKVPool

CPU = torch.device("cpu")


def _pool(num_blocks: int = 8, block_size: int = 4) -> PagedKVPool:
    return PagedKVPool(
        num_layers=2,
        num_kv_heads=2,
        head_dim=3,
        block_size=block_size,
        num_blocks=num_blocks,
        device=CPU,
        dtype=torch.float32,
    )


def test_block_zero_is_reserved_as_the_null_block() -> None:
    pool = _pool(num_blocks=4)

    sequence = pool.open("r0", max_tokens=12)

    assert pool.free_block_count == 0
    assert 0 not in sequence.blocks
    assert pool.null_slot == 0
    assert torch.count_nonzero(pool.buffer[:, :, 0]) == 0


def test_open_reserves_whole_blocks_and_release_returns_them_once() -> None:
    pool = _pool(num_blocks=8)

    sequence = pool.open("r0", max_tokens=9)  # 9 tokens / 4 per block -> 3 blocks
    assert len(sequence.blocks) == 3
    assert sequence.capacity == 12
    assert pool.free_block_count == 4

    pool.release(sequence)
    pool.release(sequence)
    assert pool.free_block_count == 7
    assert sequence.released


def test_open_fails_atomically_when_the_pool_is_short() -> None:
    pool = _pool(num_blocks=4)
    pool.open("r0", max_tokens=8)

    with pytest.raises(ValueError, match="blocks"):
        pool.open("r1", max_tokens=8)
    assert pool.free_block_count == 1


def test_slots_follow_the_block_list_across_block_boundaries() -> None:
    pool = _pool(num_blocks=8, block_size=4)
    first = pool.open("r0", max_tokens=4)  # takes block 1
    second = pool.open("r1", max_tokens=6)  # takes blocks 2 and 3
    pool.release(first)
    second_blocks = second.blocks

    assert second_blocks == [2, 3]
    assert pool.slots(second, 0, 6) == [8, 9, 10, 11, 12, 13]


def _kv(batch: int, tokens: int, fill: float) -> torch.Tensor:
    """Distinct values per (row, head, token) so misplaced slots show up."""
    base = torch.arange(batch * 2 * tokens * 3, dtype=torch.float32).view(batch, 2, tokens, 3)
    return base + fill


def _forward(pool: PagedKVPool, view: object, keys: torch.Tensor) -> list[torch.Tensor]:
    """Drive a view like the model does: every layer, in order, then commit."""
    outputs = [view.update(layer, keys + layer, -(keys + layer))[0] for layer in range(2)]  # type: ignore[attr-defined]
    view.commit(keys.shape[2])  # type: ignore[attr-defined]
    return outputs


def test_prefill_view_returns_new_kv_directly_and_writes_slots() -> None:
    pool = _pool(block_size=4)
    sequence = pool.open("r0", max_tokens=8)
    view = pool.batch([sequence], new_tokens=5)
    keys = _kv(1, 5, 100.0)

    assert view.mask is None
    assert view.position_ids.tolist() == [[0, 1, 2, 3, 4]]
    first = view.update(0, keys, -keys)
    assert first[0] is keys
    view.update(1, keys + 1, -(keys + 1))
    view.commit(5)

    slots = pool.slots(sequence, 0, 5)
    stored = pool.table(0, 0)[slots]  # [5, heads, dim]
    assert torch.equal(stored, keys[0].transpose(0, 1))
    assert torch.equal(pool.table(1, 1)[slots], -(keys + 1)[0].transpose(0, 1))
    assert sequence.length == 5


def test_decode_view_gathers_each_rows_history_and_pads_with_the_null_slot() -> None:
    pool = _pool(num_blocks=8, block_size=4)
    short = pool.open("short", max_tokens=8)
    long = pool.open("long", max_tokens=8)
    short_prompt, long_prompt = _kv(1, 2, 10.0), _kv(1, 6, 50.0)
    _forward(pool, pool.batch([short], new_tokens=2), short_prompt)
    _forward(pool, pool.batch([long], new_tokens=6), long_prompt)

    view = pool.batch([short, long], new_tokens=1)
    step = _kv(2, 1, 900.0)
    history = _forward(pool, view, step)[0]  # layer 0 keys: [2, heads, 7, dim]

    assert view.position_ids.tolist() == [[2], [6]]
    assert history.shape == (2, 2, 7, 3)
    assert torch.equal(history[0, :, :2], short_prompt[0])
    assert torch.equal(history[0, :, 2], step[0, :, 0])
    assert torch.count_nonzero(history[0, :, 3:]) == 0  # padding read from block 0
    assert torch.equal(history[1, :, :6], long_prompt[0])
    assert torch.equal(history[1, :, 6], step[1, :, 0])
    assert view.mask is not None
    assert view.mask.tolist() == [[[[True] * 3 + [False] * 4]], [[[True] * 7]]]
    assert (short.length, long.length) == (3, 7)


def test_single_sequence_decode_needs_no_mask() -> None:
    pool = _pool()
    sequence = pool.open("r0", max_tokens=8)
    _forward(pool, pool.batch([sequence], new_tokens=3), _kv(1, 3, 0.0))

    assert pool.batch([sequence], new_tokens=1).mask is None


def test_view_rejects_misuse_before_touching_the_pool() -> None:
    pool = _pool()
    sequence = pool.open("r0", max_tokens=4)
    _forward(pool, pool.batch([sequence], new_tokens=2), _kv(1, 2, 0.0))

    with pytest.raises(NotImplementedError, match="chunked"):
        pool.batch([sequence], new_tokens=2)
    with pytest.raises(ValueError, match="capacity"):
        pool.batch([pool.open("r1", max_tokens=4)], new_tokens=5)
    with pytest.raises(ValueError, match="duplicate"):
        pool.batch([sequence, sequence], new_tokens=1)
    view = pool.batch([sequence], new_tokens=1)
    with pytest.raises(RuntimeError, match="order"):
        view.update(1, _kv(1, 1, 0.0), _kv(1, 1, 0.0))
    view.update(0, _kv(1, 1, 0.0), _kv(1, 1, 0.0))
    with pytest.raises(RuntimeError, match="layers"):
        view.commit(1)
    assert sequence.length == 2
    pool.release(sequence)
    with pytest.raises(RuntimeError, match="released"):
        pool.batch([sequence], new_tokens=1)


def test_slots_reject_positions_beyond_capacity() -> None:
    pool = _pool()
    sequence = pool.open("r0", max_tokens=4)

    with pytest.raises(ValueError, match="capacity"):
        pool.slots(sequence, 0, 5)
