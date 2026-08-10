"""BatchedPagedTinyinferCache protocol and layer-transaction tests."""

import pytest

torch = pytest.importorskip("torch")


def make_paged(num_layers: int = 2) -> object:
    from tinyinfer.cache.paged import PagedKVCache

    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=2,
        block_size=4,
        head_dim=4,
        num_physical_blocks=32,
        device="cpu",
        dtype=torch.bfloat16,
    )


def make_states(paged: object, lengths: list[int]) -> list:
    from tinyinfer.cache.request_state import PagedRequestState

    states = []
    for i, length in enumerate(lengths):
        table = paged.create_block_table(max_tokens=length + 4)
        blocks = (length + 4 + paged.block_size - 1) // paged.block_size
        paged.allocate_blocks(table, blocks)
        states.append(
            PagedRequestState(
                request_id=f"r{i}",
                block_table=table,
                seq_length=length,
                blocks_allocated=blocks,
            )
        )
    # write prefill KV so gather sees real content
    for row, state in enumerate(states):
        n = state.seq_length
        key = (
            (torch.arange(1, n + 1, dtype=torch.bfloat16) * (row + 1))
            .reshape(1, 1, n, 1)
            .expand(1, 2, n, 4)
            .contiguous()
        )
        paged.write_prefill(key, key + 100, 0, state.block_table, n)
        for layer in range(1, paged.num_layers):
            paged.write_prefill(key, key + 100, layer, state.block_table, n)
    return states


class TestMembership:
    def test_order_and_old_lengths(self) -> None:
        paged = make_paged()
        states = make_states(paged, [2, 3, 5])
        from tinyinfer.cache.batched_paged_tinyinfer_cache import (
            BatchedPagedTinyinferCache,
        )

        cache = BatchedPagedTinyinferCache(paged, states)
        assert cache.request_ids == ("r0", "r1", "r2")
        assert cache.old_lengths == (2, 3, 5)
        assert cache.get_seq_length() == 5
        assert cache.get_query_offset() == 5
        assert cache.get_mask_sizes(1, 0) == (6, 0)

    def test_rejects_empty(self) -> None:
        from tinyinfer.cache.batched_paged_tinyinfer_cache import (
            BatchedPagedTinyinferCache,
        )

        with pytest.raises(ValueError, match="at least one"):
            BatchedPagedTinyinferCache(make_paged(), [])

    def test_rejects_duplicate_ids(self) -> None:
        from tinyinfer.cache.batched_paged_tinyinfer_cache import (
            BatchedPagedTinyinferCache,
        )

        paged = make_paged()
        st = make_states(paged, [1, 1])
        st[1].request_id = "r0"  # force duplicate
        with pytest.raises(ValueError, match="duplicate"):
            BatchedPagedTinyinferCache(paged, st)

    def test_rejects_released_state(self) -> None:
        from tinyinfer.cache.batched_paged_tinyinfer_cache import (
            BatchedPagedTinyinferCache,
        )

        paged = make_paged()
        st = make_states(paged, [1])
        st[0].mark_released()
        with pytest.raises(ValueError, match="released"):
            BatchedPagedTinyinferCache(paged, st)


class TestLayerTransaction:
    def test_layer0_does_not_commit_final_commits(self) -> None:
        paged = make_paged(num_layers=2)
        states = make_states(paged, [2, 3, 5])
        from tinyinfer.cache.batched_paged_tinyinfer_cache import (
            BatchedPagedTinyinferCache,
        )

        cache = BatchedPagedTinyinferCache(paged, states)
        # layer 0 update
        kv = torch.randn(3, 2, 1, 4, dtype=torch.bfloat16)
        k0, _v0 = cache.update(kv, kv.clone(), 0)
        # committed lengths unchanged after layer 0
        assert [s.seq_length for s in states] == [2, 3, 5]
        # output shape [B, Hkv, max+1, D]
        assert k0.shape == (3, 2, 6, 4)

        # layer 1 (final): commits
        k1, _v1 = cache.update(kv, kv.clone(), 1)
        assert [s.seq_length for s in states] == [3, 4, 6]
        assert k1.shape == (3, 2, 6, 4)

    def test_rejects_out_of_order_layers(self) -> None:
        from tinyinfer.cache.batched_paged_tinyinfer_cache import (
            BatchedPagedTinyinferCache,
        )

        paged = make_paged(num_layers=2)
        states = make_states(paged, [2, 3])
        cache = BatchedPagedTinyinferCache(paged, states)
        kv = torch.randn(2, 2, 1, 4, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="out of order"):
            cache.update(kv, kv.clone(), 1)  # should be 0 first

    def test_rejects_query_length_not_1(self) -> None:
        from tinyinfer.cache.batched_paged_tinyinfer_cache import (
            BatchedPagedTinyinferCache,
        )

        paged = make_paged(num_layers=1)
        states = make_states(paged, [1])
        cache = BatchedPagedTinyinferCache(paged, states)
        kv = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="query length"):
            cache.update(kv, kv.clone(), 0)


class TestGatherContent:
    def test_left_padded_right_aligned_content(self) -> None:
        paged = make_paged(num_layers=2)
        states = make_states(paged, [2, 3, 5])
        from tinyinfer.cache.batched_paged_tinyinfer_cache import (
            BatchedPagedTinyinferCache,
        )

        cache = BatchedPagedTinyinferCache(paged, states)
        # after a full decode pass the old content prefixed with new token
        kv = torch.randn(3, 2, 1, 4, dtype=torch.bfloat16)
        cache.update(kv, kv.clone(), 0)
        cache.update(kv, kv.clone(), 1)
        # rows are right-aligned; row 0 (len=3) has 3 leading pad zero
        k, _v = cache[1]
        assert k.shape == (3, 2, 6, 4)
        for row, old_len in enumerate([2, 3, 5]):
            new_len = old_len + 1
            lead = 6 - new_len
            if lead > 0:
                assert torch.count_nonzero(k[row, :, :lead, :]) == 0
            # non-padded region non-zero (prefill used 1..old_len and decode token)
            assert torch.count_nonzero(k[row, :, lead:, :]) > 0
