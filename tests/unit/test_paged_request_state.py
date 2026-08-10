"""PagedRequestState contract tests: state lifecycle and HF cache protocol."""

import pytest

# ---------------------------------------------------------------------------
# State construction validation
# ---------------------------------------------------------------------------


class TestPagedRequestStateConstruction:
    def test_rejects_blank_request_id(self) -> None:
        from tinyinfer.cache.request_state import PagedRequestState

        with pytest.raises(ValueError, match="request_id"):
            PagedRequestState(
                request_id="   ",
                block_table=object(),
                seq_length=0,
                blocks_allocated=0,
            )

    def test_rejects_negative_seq_length(self) -> None:
        from tinyinfer.cache.request_state import PagedRequestState

        with pytest.raises(ValueError, match="seq_length"):
            PagedRequestState(
                request_id="a",
                block_table=object(),
                seq_length=-1,
                blocks_allocated=0,
            )

    def test_rejects_negative_blocks_allocated(self) -> None:
        from tinyinfer.cache.request_state import PagedRequestState

        with pytest.raises(ValueError, match="blocks_allocated"):
            PagedRequestState(
                request_id="a",
                block_table=object(),
                seq_length=0,
                blocks_allocated=-1,
            )

    def test_rejects_bool_as_int_seq_length(self) -> None:
        from tinyinfer.cache.request_state import PagedRequestState

        with pytest.raises(ValueError, match="seq_length"):
            PagedRequestState(
                request_id="a",
                block_table=object(),
                seq_length=True,
                blocks_allocated=0,
            )


# ---------------------------------------------------------------------------
# State lifecycle
# ---------------------------------------------------------------------------


def make_state(*, seq_length: int = 0, blocks_allocated: int = 0) -> object:
    from tinyinfer.cache.block import BlockTable
    from tinyinfer.cache.request_state import PagedRequestState

    return PagedRequestState(
        request_id="req-1",
        block_table=BlockTable(num_layers=1, max_logical_blocks=8),
        seq_length=seq_length,
        blocks_allocated=blocks_allocated,
    )


class TestPagedRequestStateLifecycle:
    def test_active_positive_commit_advances_length(self) -> None:
        state = make_state(seq_length=5)
        state.commit_tokens(2)
        assert state.seq_length == 7

    def test_idempotent_release(self) -> None:
        state = make_state()
        state.mark_released()
        state.mark_released()  # no-op, must not raise
        assert state.released is True

    def test_released_access_rejected(self) -> None:
        state = make_state()
        state.mark_released()
        with pytest.raises(RuntimeError, match="req-1"):
            state.require_active()

    def test_released_commit_rejected(self) -> None:
        state = make_state()
        state.mark_released()
        with pytest.raises(RuntimeError, match="req-1"):
            state.commit_tokens(1)

    def test_nonpositive_commit_rejected(self) -> None:
        state = make_state(seq_length=3)
        with pytest.raises(ValueError, match="positive"):
            state.commit_tokens(0)
        with pytest.raises(ValueError, match="positive"):
            state.commit_tokens(-1)


# ---------------------------------------------------------------------------
# HF cache protocol via shared state
# ---------------------------------------------------------------------------


class TestPagedTinyinferCacheProtocol:
    def test_single_cache_protocol_contract(self) -> None:
        torch = pytest.importorskip("torch")
        from tinyinfer.cache.paged import PagedKVCache
        from tinyinfer.cache.paged_tinyinfer_cache import PagedTinyinferCache

        state = make_state(seq_length=7, blocks_allocated=1)
        paged = PagedKVCache(
            num_layers=1,
            num_kv_heads=1,
            block_size=16,
            head_dim=4,
            num_physical_blocks=4,
            device="cpu",
            dtype=torch.bfloat16,
        )
        cache = PagedTinyinferCache(paged, state)
        assert cache.get_seq_length() == 7
        assert cache.get_query_offset() == 7
        assert cache.get_mask_sizes(1, 0) == (8, 0)
