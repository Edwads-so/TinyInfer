"""Real-tensor batched paged KV IO tests (write_decode_batch / gather_batch)."""

import pytest

torch = pytest.importorskip("torch")


def make_paged(*, num_layers: int = 2, num_blocks: int = 16) -> object:
    from tinyinfer.cache.paged import PagedKVCache

    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=2,
        block_size=2,
        head_dim=4,
        num_physical_blocks=num_blocks,
        device="cpu",
        dtype=torch.bfloat16,
    )


def make_tables(paged: object, lengths: list[int]) -> list:
    tables = []
    for length in lengths:
        max_tokens = length + 4  # room for one extra decode token
        table = paged.create_block_table(max_tokens=max_tokens)
        # allocate for the full max_tokens so decode at position `length`
        # (which requires logical block length//block_size) is mapped.
        blocks = (max_tokens + paged.block_size - 1) // paged.block_size
        paged.allocate_blocks(table, blocks)
        tables.append(table)
    return tables


def write_prefill_all(paged: object, tables: list, lengths: list[int]) -> None:
    """Write distinct per-request KV so gather results are request-specific."""
    for row, (table, length) in enumerate(zip(tables, lengths, strict=True)):
        # key/value: [1, Hkv, length, D]; value = key + 100 for easy diff
        key = (
            torch.arange(1, length + 1, dtype=torch.bfloat16)
            .reshape(1, 1, length, 1)
            .expand(1, 2, length, 4)
            .contiguous()
        )
        key = key.clone() * (row + 1)
        value = (key + 100).clone()
        paged.write_prefill(key, value, 0, table, length)


class TestWriteDecodeBatch:
    def test_b3_shapes_and_values(self) -> None:
        paged = make_paged()
        lengths = [2, 3, 5]
        tables = make_tables(paged, lengths)
        write_prefill_all(paged, tables, lengths)

        # decode one token per request at current positions
        positions = [n for n in lengths]
        key = torch.randn(3, 2, 1, 4, dtype=torch.bfloat16) * (
            torch.arange(3, dtype=torch.bfloat16).reshape(3, 1, 1, 1) + 1
        )
        value = (key + 200).clone()

        paged.write_decode_batch(key, value, 0, tables, positions)

        # re-read the last written token from each table
        for row, (table, length) in enumerate(zip(tables, lengths, strict=True)):
            k, v = paged.read(0, table, length + 1)
            torch.testing.assert_close(k[0, :, -1:, :], key[row, :, :, :])
            torch.testing.assert_close(v[0, :, -1:, :], value[row, :, :, :])

    def test_validation_before_mutation(self) -> None:
        """Failing input must leave the pool buffer byte-for-byte unchanged."""
        paged = make_paged()
        lengths = [2, 3]
        tables = make_tables(paged, lengths)
        write_prefill_all(paged, tables, lengths)

        before = paged._buffer.clone()

        # negative position in row 1
        key = torch.randn(2, 2, 1, 4, dtype=torch.bfloat16)
        value = key.clone()
        with pytest.raises(ValueError):
            paged.write_decode_batch(key, value, 0, tables, [0, -1])
        assert torch.equal(paged._buffer, before)

    def test_wrong_batch_rows_rejected_unchanged(self) -> None:
        paged = make_paged()
        lengths = [2, 3]
        tables = make_tables(paged, lengths)
        write_prefill_all(paged, tables, lengths)
        before = paged._buffer.clone()

        key = torch.randn(2, 2, 1, 4, dtype=torch.bfloat16)
        value = key.clone()
        with pytest.raises(ValueError):
            paged.write_decode_batch(key, value, 0, tables, [0])  # 1 pos for B=2
        assert torch.equal(paged._buffer, before)


class TestGatherBatch:
    def test_b3_left_padded_gather(self) -> None:
        paged = make_paged()
        lengths = [2, 3, 5]
        tables = make_tables(paged, lengths)
        write_prefill_all(paged, tables, lengths)

        key_out, value_out = paged.gather_batch(0, tables, lengths)

        assert key_out.shape == (3, 2, 5, 4)
        assert value_out.shape == (3, 2, 5, 4)

        # right-aligned: each row's valid tokens occupy the last `length` slots
        for row, length in enumerate(lengths):
            want = (
                (torch.arange(1, length + 1, dtype=torch.bfloat16) * (row + 1))
                .reshape(length, 1)
                .expand(length, 4)
                .contiguous()
            )
            torch.testing.assert_close(
                key_out[row, :, 5 - length :, :], want.unsqueeze(0).expand(2, length, 4)
            )
            torch.testing.assert_close(
                value_out[row, :, 5 - length :, :], (want + 100).unsqueeze(0).expand(2, length, 4)
            )
            # leading padding is zero
            if 5 - length > 0:
                assert torch.count_nonzero(key_out[row, :, : 5 - length, :]) == 0
                assert torch.count_nonzero(value_out[row, :, : 5 - length, :]) == 0

    def test_b1_equals_read(self) -> None:
        paged = make_paged()
        tables = make_tables(paged, [4])
        write_prefill_all(paged, tables, [4])

        k_g, v_g = paged.gather_batch(0, tables, [4])
        k_r, v_r = paged.read(0, tables[0], 4)
        torch.testing.assert_close(k_g[0], k_r[0])
        torch.testing.assert_close(v_g[0], v_r[0])

    def test_empty_batch_returns_empty(self) -> None:
        paged = make_paged()
        k, v = paged.gather_batch(0, [], [])
        assert k.shape == (0, 2, 0, 4)
        assert v.shape == (0, 2, 0, 4)

    def test_validation_does_not_mutate(self) -> None:
        paged = make_paged()
        tables = make_tables(paged, [3])
        write_prefill_all(paged, tables, [3])
        before = paged._buffer.clone()

        # gather with a negative length fails before any mutation
        with pytest.raises(ValueError):
            paged.gather_batch(0, tables, [-1])
        assert torch.equal(paged._buffer, before)

    def test_isolation_across_requests(self) -> None:
        paged = make_paged()
        # group A: single table of length 3 (row 0 multiplier 1)
        t_a = make_tables(paged, [3])
        write_prefill_all(paged, t_a, [3])
        # group B: single table of length 2 (row 0 multiplier 1) but distinct
        # value base, written as a separate batch so tables don't share blocks
        t_b = make_tables(paged, [2])
        key_b = (
            (torch.arange(1, 3, dtype=torch.bfloat16) * 100)
            .reshape(1, 1, 2, 1)
            .expand(1, 2, 2, 4)
            .contiguous()
        )
        paged.write_prefill(key_b, key_b + 100, 0, t_b[0], 2)

        k_a, _ = paged.gather_batch(0, t_a, [3])
        k_b, _ = paged.gather_batch(0, t_b, [2])
        # distinct token magnitudes prove no cross-request contamination
        assert k_a[0, 0, 0, 0] != k_b[0, 0, 0, 0]


class TestBatchValidationMetadata:
    """Audit fix: batched decode must reject mismatched dtype/device unchanged."""

    def test_wrong_dtype_rejected(self) -> None:
        paged = make_paged()
        lengths = [2, 3]
        tables = make_tables(paged, lengths)
        write_prefill_all(paged, tables, lengths)
        before = paged._buffer.clone()

        key = torch.randn(2, 2, 1, 4, dtype=torch.float32)  # wrong dtype
        value = key.clone()
        with pytest.raises(ValueError, match="dtype"):
            paged.write_decode_batch(key, value, 0, tables, list(lengths))
        assert torch.equal(paged._buffer, before)

    def test_wrong_device_rejected(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        paged = make_paged()
        lengths = [2, 3]
        tables = make_tables(paged, lengths)
        write_prefill_all(paged, tables, lengths)
        before = paged._buffer.clone()

        key = torch.randn(2, 2, 1, 4, dtype=torch.bfloat16, device="cuda")
        value = key.clone()
        with pytest.raises(ValueError, match="device"):
            paged.write_decode_batch(key, value, 0, tables, list(lengths))
        assert torch.equal(paged._buffer, before)
