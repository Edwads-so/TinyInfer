"""Qwen2 structural parameters read from a model directory's ``config.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Qwen2Spec:
    """The shapes and constants the forward pass needs, and nothing else."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @property
    def num_kv_groups(self) -> int:
        """Query heads served by each KV head (GQA group size)."""
        return self.num_heads // self.num_kv_heads


QWEN25_05B = Qwen2Spec(
    vocab_size=151936,
    hidden_size=896,
    intermediate_size=4864,
    num_layers=24,
    num_heads=14,
    num_kv_heads=2,
    head_dim=64,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    max_position_embeddings=32768,
    tie_word_embeddings=True,
)
"""Qwen/Qwen2.5-0.5B as published; the GPU test checks it against the cached config."""


def _rope_theta(config: dict[str, Any]) -> float:
    # Transformers 5.x writes ``rope_parameters``; older hub files use top-level keys.
    parameters = config.get("rope_parameters")
    if parameters is not None:
        if parameters.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported rope_parameters: {parameters}")
        return float(parameters["rope_theta"])
    if config.get("rope_scaling") is not None:
        raise ValueError(f"unsupported rope_scaling: {config['rope_scaling']}")
    return float(config["rope_theta"])


def load_qwen2_spec(model_dir: Path) -> Qwen2Spec:
    """Parse and validate ``model_dir/config.json``; reject anything not implemented."""
    path = Path(model_dir) / "config.json"
    try:
        config: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error

    if config.get("model_type") != "qwen2":
        raise ValueError(f"model_type must be 'qwen2', got {config.get('model_type')!r}")
    if config.get("hidden_act") != "silu":
        raise ValueError(f"hidden_act must be 'silu', got {config.get('hidden_act')!r}")
    if config.get("use_sliding_window", False):
        raise ValueError("sliding window attention is not supported")
    layer_types = config.get("layer_types")
    if layer_types is not None and any(kind != "full_attention" for kind in layer_types):
        raise ValueError(f"only full_attention layer_types are supported, got {layer_types}")

    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size {hidden_size} is not divisible by {num_heads} heads")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_attention_heads {num_heads} is not a multiple of "
            f"num_key_value_heads {num_kv_heads}"
        )
    return Qwen2Spec(
        vocab_size=int(config["vocab_size"]),
        hidden_size=hidden_size,
        intermediate_size=int(config["intermediate_size"]),
        num_layers=int(config["num_hidden_layers"]),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=int(config.get("head_dim") or hidden_size // num_heads),
        rms_norm_eps=float(config["rms_norm_eps"]),
        rope_theta=_rope_theta(config),
        max_position_embeddings=int(config["max_position_embeddings"]),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
    )
