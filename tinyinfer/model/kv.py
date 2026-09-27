"""KV cache protocol used by the owned forward pass, and a contiguous implementation."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from tinyinfer.model.spec import Qwen2Spec

Tensor = torch.Tensor


class KVCache(Protocol):
    """Per-layer K/V storage for one forward pass at a time.

    Within one forward, every layer calls ``update`` once in order with the new
    tokens' K/V and gets back that layer's full history; ``commit`` then advances
    the length once, so a failed forward never exposes a half-written step.
    """

    @property
    def length(self) -> int: ...

    def update(self, layer: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]: ...

    def commit(self, new_tokens: int) -> None: ...


class ContiguousKV:
    """``[num_layers, 2, B, Hkv, max_tokens, D]`` buffer shared by an equal-length batch."""

    def __init__(
        self,
        *,
        num_layers: int,
        batch: int,
        num_kv_heads: int,
        head_dim: int,
        max_tokens: int,
        device: Any,
        dtype: Any,
    ) -> None:
        if min(num_layers, batch, num_kv_heads, head_dim, max_tokens) < 1:
            raise ValueError("ContiguousKV dimensions must be positive")
        self._buffer = torch.zeros(
            num_layers, 2, batch, num_kv_heads, max_tokens, head_dim, device=device, dtype=dtype
        )
        self._length = 0
        self._pending: int | None = None
        self._next_layer = 0

    @classmethod
    def for_spec(
        cls, spec: Qwen2Spec, *, batch: int, max_tokens: int, device: Any, dtype: Any
    ) -> ContiguousKV:
        """Size the buffer for ``max_tokens`` (prompt plus generated), not the model maximum."""
        return cls(
            num_layers=spec.num_layers,
            batch=batch,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            max_tokens=max_tokens,
            device=device,
            dtype=dtype,
        )

    @property
    def length(self) -> int:
        return self._length

    def update(self, layer: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Write ``key``/``value`` ``[B, Hkv, T, D]`` after the committed length; return history."""
        if layer != self._next_layer:
            raise RuntimeError(
                f"layer update out of order: got {layer}, expected {self._next_layer}"
            )
        _, _, batch, heads, capacity, head_dim = self._buffer.shape
        expected = (batch, heads, key.shape[2], head_dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"key/value shape {tuple(key.shape)} does not match {expected}")
        tokens = int(key.shape[2])
        if self._pending is not None and tokens != self._pending:
            raise RuntimeError(f"layer {layer} wrote {tokens} tokens, expected {self._pending}")
        end = self._length + tokens
        if end > capacity:
            raise ValueError(f"{end} tokens exceed KV capacity {capacity}")
        self._buffer[layer, 0, :, :, self._length : end] = key
        self._buffer[layer, 1, :, :, self._length : end] = value
        self._pending = tokens
        self._next_layer = layer + 1
        return self._buffer[layer, 0, :, :, :end], self._buffer[layer, 1, :, :, :end]

    def commit(self, new_tokens: int) -> None:
        """Advance the length after every layer has written ``new_tokens`` tokens."""
        num_layers = self._buffer.shape[0]
        if self._next_layer != num_layers:
            raise RuntimeError(f"only {self._next_layer} of {num_layers} layers were updated")
        if new_tokens != self._pending:
            raise RuntimeError(f"commit of {new_tokens} tokens, but layers wrote {self._pending}")
        self._length += new_tokens
        self._pending = None
        self._next_layer = 0
