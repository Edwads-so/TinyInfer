"""Validated engine configuration shared by TinyInfer runtime components."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_DTYPES = frozenset({"float32", "float16", "bfloat16"})


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Stable, immutable settings for one TinyInfer engine instance."""

    model_id: str = "Qwen/Qwen2.5-0.5B"
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_sequence_length: int = 4096
    kv_block_size: int = 16

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.kv_block_size <= 0:
            raise ValueError("kv_block_size must be positive")
        if self.kv_block_size > self.max_sequence_length:
            raise ValueError("kv_block_size cannot exceed max_sequence_length")
