"""TinyInfer-owned contiguous KV cache for one request."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ContiguousKVCache:
    """Pre-allocated contiguous KV buffer for a single request.

    Layout: ``[num_layers, 2, num_kv_heads, max_seq_len, head_dim]``
    where dim 1 index 0 = key, 1 = value.
    """

    buffer: Any  # torch.Tensor of shape [num_layers, 2, num_kv_heads, max_seq_len, head_dim]
    length: int = 0

    def __post_init__(self) -> None:
        shape = tuple(self.buffer.shape)
        if len(shape) != 5 or shape[1] != 2:
            raise ValueError(
                f"buffer must have shape [num_layers, 2, num_kv_heads, max_seq_len, head_dim], "
                f"got {shape}"
            )
        if not hasattr(self.buffer, "device"):
            raise ValueError("buffer must be a torch.Tensor")
        if type(self.length) is not int or self.length < 0:
            raise ValueError("length must be a non-negative integer")

    @property
    def num_layers(self) -> int:
        return int(self.buffer.shape[0])

    @property
    def num_kv_heads(self) -> int:
        return int(self.buffer.shape[2])

    @property
    def max_seq_len(self) -> int:
        return int(self.buffer.shape[3])

    @property
    def head_dim(self) -> int:
        return int(self.buffer.shape[4])

    @property
    def device(self) -> Any:
        return self.buffer.device

    def write_prefill(
        self,
        key: Any,
        value: Any,
        layer_idx: int,
    ) -> None:
        """Write prefill KV for a single layer into the cache.

        ``key`` / ``value`` shapes: ``[1, num_kv_heads, prompt_len, head_dim]``
        """
        prompt_len = int(key.shape[2])
        if prompt_len < 1:
            raise ValueError("prefill key must have at least one token")
        if prompt_len > self.max_seq_len:
            raise ValueError(f"prefill length {prompt_len} exceeds max_seq_len {self.max_seq_len}")
        self.buffer[layer_idx, 0, :, :prompt_len, :].copy_(key[0])
        self.buffer[layer_idx, 1, :, :prompt_len, :].copy_(value[0])
        object.__setattr__(self, "length", prompt_len)

    def write_decode(
        self,
        key: Any,
        value: Any,
        layer_idx: int,
    ) -> None:
        """Append one decode-step KV for a single layer.

        ``key`` / ``value`` shapes: ``[1, num_kv_heads, 1, head_dim]``

        Does NOT increment ``length`` — the caller (TinyinferCache) is
        responsible for advancing the sequence cursor once after all
        layers have been written.
        """
        if key.shape[2] != 1 or value.shape[2] != 1:
            raise ValueError("decode key/value must have exactly one token")
        pos = self.length
        if pos >= self.max_seq_len:
            raise ValueError(f"cache length {pos} already at max_seq_len {self.max_seq_len}")
        self.buffer[layer_idx, 0, :, pos, :].copy_(key[0, :, 0, :])
        self.buffer[layer_idx, 1, :, pos, :].copy_(value[0, :, 0, :])

    def advance_length(self) -> None:
        """Increment length by one after all layers have written a decode step."""
        object.__setattr__(self, "length", self.length + 1)

    def read(self, layer_idx: int, *, tokens: int | None = None) -> tuple[Any, Any]:
        """Return (key, value) tensors for the cached sequence.

        Shapes: ``[1, num_kv_heads, tokens, head_dim]``
        """
        n = tokens if tokens is not None else self.length
        return (
            self.buffer[layer_idx, 0, :, :n, :].unsqueeze(0),
            self.buffer[layer_idx, 1, :, :n, :].unsqueeze(0),
        )


def allocate_contiguous_cache(
    *,
    num_layers: int,
    num_kv_heads: int,
    max_seq_len: int,
    head_dim: int,
    device: Any,
    dtype: Any,
) -> ContiguousKVCache:
    """Allocate a zero-filled contiguous KV buffer."""
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if num_kv_heads < 1:
        raise ValueError("num_kv_heads must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if head_dim < 1:
        raise ValueError("head_dim must be positive")
    torch = importlib.import_module("torch")
    buffer = torch.zeros(
        num_layers,
        2,
        num_kv_heads,
        max_seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )
    return ContiguousKVCache(buffer=buffer, length=0)
