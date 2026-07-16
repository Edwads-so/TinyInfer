from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tinyinfer.config import EngineConfig


def test_default_config_targets_small_qwen() -> None:
    config = EngineConfig()
    assert config.model_id == "Qwen/Qwen2.5-0.5B"
    assert config.dtype == "bfloat16"
    assert config.device == "cuda"
    assert config.max_sequence_length == 4096
    assert config.kv_block_size == 16


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "", "model_id must not be empty"),
        ("dtype", "int8", "unsupported dtype"),
        ("device", "mps", "device must be 'cpu' or a CUDA device"),
        ("max_sequence_length", 0, "max_sequence_length must be positive"),
        ("kv_block_size", 0, "kv_block_size must be positive"),
        ("kv_block_size", 8192, "kv_block_size cannot exceed max_sequence_length"),
    ],
)
def test_invalid_config_is_rejected(field: str, value: object, message: str) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        EngineConfig(**kwargs)  # type: ignore[arg-type]


def test_config_is_immutable() -> None:
    config = EngineConfig()
    with pytest.raises(FrozenInstanceError):
        config.dtype = "float32"  # type: ignore[misc]
