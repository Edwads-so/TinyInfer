"""Unit tests for ContiguousKVCache without requiring torch."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


class SimpleTensor:
    """Minimal tensor stand-in that supports shape, device, dtype, __getitem__, copy_, and unsqueeze."""

    def __init__(self, *shape: int, device: str = "cpu", dtype: str = "bfloat16") -> None:
        self._shape = shape
        self._device = device
        self._dtype = dtype
        self._storage: list[float] = [0.0] * _prod(shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> str:
        return self._dtype

    def __getitem__(self, idx: object) -> SimpleTensor:
        """Return a view-like slice."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        return SimpleTensor(*self.shape, device=self._device, dtype=self._dtype)

    def copy_(self, src: SimpleTensor) -> SimpleTensor:
        return self

    def unsqueeze(self, dim: int) -> SimpleTensor:
        new_shape = list(self._shape)
        insert_at = dim if dim >= 0 else len(new_shape) + dim + 1
        new_shape.insert(insert_at, 1)
        t = SimpleTensor(*new_shape, device=self._device, dtype=self._dtype)
        return t


def _prod(shape: tuple[int, ...]) -> int:
    result = 1
    for s in shape:
        result *= s
    return result


class FakeTorch:
    bfloat16 = "bfloat16"
    long = "int64"

    @staticmethod
    def zeros(*args: int, device: str = "cpu", dtype: str = "bfloat16") -> SimpleTensor:
        return SimpleTensor(*args, device=device, dtype=dtype)

    @staticmethod
    def device(device_str: str) -> str:
        return device_str


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_allocate_contiguous_cache_returns_valid_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from tinyinfer.cache.contiguous import allocate_contiguous_cache

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorch)

    cache = allocate_contiguous_cache(
        num_layers=24,
        num_kv_heads=4,
        max_seq_len=4096,
        head_dim=64,
        device="cpu",
        dtype="bfloat16",
    )
    assert cache.num_layers == 24
    assert cache.num_kv_heads == 4
    assert cache.max_seq_len == 4096
    assert cache.head_dim == 64
    assert cache.length == 0
    assert cache.device == "cpu"


def test_contiguous_cache_rejects_invalid_shape() -> None:
    from tinyinfer.cache.contiguous import ContiguousKVCache

    with pytest.raises(ValueError, match="buffer must have shape"):
        ContiguousKVCache(buffer=SimpleTensor(1, 2, 3))


def test_contiguous_cache_rejects_negative_length() -> None:
    from tinyinfer.cache.contiguous import ContiguousKVCache

    with pytest.raises(ValueError, match="length must be a non-negative integer"):
        ContiguousKVCache(buffer=SimpleTensor(24, 2, 4, 4096, 64), length=-1)


def test_contiguous_cache_is_frozen() -> None:
    from tinyinfer.cache.contiguous import ContiguousKVCache

    cache = ContiguousKVCache(SimpleTensor(24, 2, 4, 4096, 64))
    with pytest.raises(FrozenInstanceError):
        cache.length = 10  # type: ignore[misc]


def test_allocate_contiguous_cache_rejects_zero_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from tinyinfer.cache.contiguous import allocate_contiguous_cache

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorch)

    with pytest.raises(ValueError, match="num_layers must be positive"):
        allocate_contiguous_cache(
            num_layers=0,
            num_kv_heads=4,
            max_seq_len=4096,
            head_dim=64,
            device="cpu",
            dtype="bfloat16",
        )


def test_read_at_zero_length_returns_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from tinyinfer.cache.contiguous import allocate_contiguous_cache

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorch)

    cache = allocate_contiguous_cache(
        num_layers=2,
        num_kv_heads=2,
        max_seq_len=8,
        head_dim=4,
        device="cpu",
        dtype="bfloat16",
    )
    key, value = cache.read(0)
    # read() returns two tensors (FakeTensor-safe assertion on dtype/device)
    assert key.dtype == "bfloat16"
    assert value.dtype == "bfloat16"
    assert key.device == "cpu"


def test_write_prefill_sets_length(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from tinyinfer.cache.contiguous import allocate_contiguous_cache

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorch)

    cache = allocate_contiguous_cache(
        num_layers=2,
        num_kv_heads=2,
        max_seq_len=8,
        head_dim=4,
        device="cpu",
        dtype="bfloat16",
    )
    # key shape: [1, num_kv_heads, prompt_len, head_dim]
    key = SimpleTensor(1, 2, 3, 4)
    value = SimpleTensor(1, 2, 3, 4)
    cache.write_prefill(key, value, layer_idx=0)
    assert cache.length == 3


def test_write_prefill_rejects_zero_length_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from tinyinfer.cache.contiguous import allocate_contiguous_cache

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorch)

    cache = allocate_contiguous_cache(
        num_layers=2,
        num_kv_heads=2,
        max_seq_len=8,
        head_dim=4,
        device="cpu",
        dtype="bfloat16",
    )
    key = SimpleTensor(1, 2, 0, 4)
    value = SimpleTensor(1, 2, 0, 4)
    with pytest.raises(ValueError, match="prefill key must have at least one token"):
        cache.write_prefill(key, value, layer_idx=0)


def test_write_decode_increments_length(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from tinyinfer.cache.contiguous import allocate_contiguous_cache

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorch)

    cache = allocate_contiguous_cache(
        num_layers=2,
        num_kv_heads=2,
        max_seq_len=8,
        head_dim=4,
        device="cpu",
        dtype="bfloat16",
    )
    key_prefill = SimpleTensor(1, 2, 3, 4)
    value_prefill = SimpleTensor(1, 2, 3, 4)
    cache.write_prefill(key_prefill, value_prefill, layer_idx=0)
    assert cache.length == 3

    key_decode = SimpleTensor(1, 2, 1, 4)
    value_decode = SimpleTensor(1, 2, 1, 4)
    cache.write_decode(key_decode, value_decode, layer_idx=0)
    assert cache.length == 4


def test_write_decode_rejects_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from tinyinfer.cache.contiguous import allocate_contiguous_cache

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorch)

    cache = allocate_contiguous_cache(
        num_layers=1,
        num_kv_heads=1,
        max_seq_len=2,
        head_dim=1,
        device="cpu",
        dtype="bfloat16",
    )
    cache.write_prefill(SimpleTensor(1, 1, 2, 1), SimpleTensor(1, 1, 2, 1), 0)
    with pytest.raises(ValueError, match="already at max_seq_len"):
        cache.write_decode(SimpleTensor(1, 1, 1, 1), SimpleTensor(1, 1, 1, 1), 0)