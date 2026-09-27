"""Load Qwen2 weights from safetensors into plain tensor dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinyinfer.model.spec import Qwen2Spec

# Checkpoint suffix -> (field name, shape builder) for one decoder layer.
_LAYER_TENSORS: dict[str, tuple[str, Any]] = {
    "input_layernorm.weight": ("input_norm", lambda s: (s.hidden_size,)),
    "self_attn.q_proj.weight": ("q_proj", lambda s: (s.num_heads * s.head_dim, s.hidden_size)),
    "self_attn.q_proj.bias": ("q_bias", lambda s: (s.num_heads * s.head_dim,)),
    "self_attn.k_proj.weight": ("k_proj", lambda s: (s.num_kv_heads * s.head_dim, s.hidden_size)),
    "self_attn.k_proj.bias": ("k_bias", lambda s: (s.num_kv_heads * s.head_dim,)),
    "self_attn.v_proj.weight": ("v_proj", lambda s: (s.num_kv_heads * s.head_dim, s.hidden_size)),
    "self_attn.v_proj.bias": ("v_bias", lambda s: (s.num_kv_heads * s.head_dim,)),
    "self_attn.o_proj.weight": ("o_proj", lambda s: (s.hidden_size, s.num_heads * s.head_dim)),
    "post_attention_layernorm.weight": ("post_norm", lambda s: (s.hidden_size,)),
    "mlp.gate_proj.weight": ("gate_proj", lambda s: (s.intermediate_size, s.hidden_size)),
    "mlp.up_proj.weight": ("up_proj", lambda s: (s.intermediate_size, s.hidden_size)),
    "mlp.down_proj.weight": ("down_proj", lambda s: (s.hidden_size, s.intermediate_size)),
}


@dataclass(frozen=True, slots=True)
class Qwen2LayerWeights:
    """One decoder layer.  Linear weights are ``[out_features, in_features]``."""

    input_norm: Any
    q_proj: Any
    q_bias: Any
    k_proj: Any
    k_bias: Any
    v_proj: Any
    v_bias: Any
    o_proj: Any
    post_norm: Any
    gate_proj: Any
    up_proj: Any
    down_proj: Any


@dataclass(frozen=True, slots=True)
class Qwen2Weights:
    """All model weights.  ``lm_head`` is ``embed`` itself when embeddings are tied."""

    embed: Any
    layers: tuple[Qwen2LayerWeights, ...]
    final_norm: Any
    lm_head: Any


def resolve_model_dir(model_id: str, revision: str) -> Path:
    """Return the cached snapshot directory for ``model_id@revision`` without downloading."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    except Exception as error:
        raise ValueError(
            f"{model_id}@{revision} is not in the local Hugging Face cache ({type(error).__name__})"
        ) from error


def _expected_shapes(spec: Qwen2Spec) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (spec.vocab_size, spec.hidden_size),
        "model.norm.weight": (spec.hidden_size,),
    }
    if not spec.tie_word_embeddings:
        shapes["lm_head.weight"] = (spec.vocab_size, spec.hidden_size)
    for layer in range(spec.num_layers):
        for suffix, (_, shape) in _LAYER_TENSORS.items():
            shapes[f"model.layers.{layer}.{suffix}"] = shape(spec)
    return shapes


def _checkpoint_files(model_dir: Path) -> dict[str, Path]:
    """Map every tensor name to the safetensors file that holds it."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return {name: model_dir / shard for name, shard in weight_map.items()}
    from safetensors import safe_open

    single = model_dir / "model.safetensors"
    if not single.exists():
        raise ValueError(f"no model.safetensors or index in {model_dir}")
    with safe_open(str(single), framework="pt") as handle:
        return {name: single for name in handle.keys()}  # noqa: SIM118 - not a dict


def load_qwen2_weights(
    model_dir: Path, spec: Qwen2Spec, *, device: Any, dtype: Any
) -> Qwen2Weights:
    """Read, validate, and place every tensor; reject missing, extra, or misshapen keys."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    files = _checkpoint_files(model_dir)
    expected = _expected_shapes(spec)
    missing = sorted(set(expected) - set(files))
    unexpected = sorted(set(files) - set(expected))
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unexpected:
            hint = " (tie_word_embeddings is true)" if "lm_head.weight" in unexpected else ""
            detail.append(f"unexpected {unexpected}{hint}")
        raise ValueError(f"checkpoint keys do not match Qwen2Spec: {'; '.join(detail)}")

    tensors: dict[str, Any] = {}
    for path in sorted(set(files.values())):
        with safe_open(str(path), framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118 - not a dict
                tensor = handle.get_tensor(name)
                if tuple(tensor.shape) != expected[name]:
                    raise ValueError(
                        f"shape of {name} is {tuple(tensor.shape)}, expected {expected[name]}"
                    )
                tensors[name] = tensor.to(device=device, dtype=dtype)

    layers = tuple(
        Qwen2LayerWeights(
            **{
                field: tensors[f"model.layers.{layer}.{suffix}"]
                for suffix, (field, _) in _LAYER_TENSORS.items()
            }
        )
        for layer in range(spec.num_layers)
    )
    embed = tensors["model.embed_tokens.weight"]
    return Qwen2Weights(
        embed=embed,
        layers=layers,
        final_norm=tensors["model.norm.weight"],
        lm_head=embed if spec.tie_word_embeddings else tensors["lm_head.weight"],
    )
