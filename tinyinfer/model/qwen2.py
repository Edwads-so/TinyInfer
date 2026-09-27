"""The Qwen2 decoder forward pass, written out layer by layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tinyinfer.model.kv import KVCache
from tinyinfer.model.ops import (
    apply_rope,
    attention,
    rms_norm,
    rope_inv_freq,
    rope_tables,
    swiglu_mlp,
)
from tinyinfer.model.spec import Qwen2Spec, load_qwen2_spec
from tinyinfer.model.weights import Qwen2Weights, load_qwen2_weights

Tensor = torch.Tensor
Trace = Callable[[str, Tensor], object]


@dataclass(frozen=True, slots=True)
class ForwardOutput:
    """``logits [B, T, V]``; with ``capture_hidden``, the per-layer hidden states.

    ``layer_outputs`` holds the embeddings followed by every decoder layer's
    output (``num_layers + 1`` tensors of ``[B, T, C]``); ``final_hidden`` is the
    last layer's output after the final RMSNorm.
    """

    logits: Tensor
    final_hidden: Tensor
    layer_outputs: tuple[Tensor, ...] | None


class Qwen2Model:
    """Owned Qwen2 forward pass over plain tensors."""

    def __init__(self, spec: Qwen2Spec, weights: Qwen2Weights) -> None:
        self.spec = spec
        self.weights = weights
        self._inv_freq = rope_inv_freq(spec.head_dim, spec.rope_theta, device=weights.embed.device)

    @classmethod
    def from_pretrained(cls, model_dir: Path, *, device: Any, dtype: Any) -> Qwen2Model:
        spec = load_qwen2_spec(model_dir)
        return cls(spec, load_qwen2_weights(model_dir, spec, device=device, dtype=dtype))

    @property
    def device(self) -> Any:
        return self.weights.embed.device

    @property
    def dtype(self) -> Any:
        return self.weights.embed.dtype

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        *,
        kv: KVCache | None = None,
        attention_mask: Tensor | None = None,
        capture_hidden: bool = False,
        trace: Trace | None = None,
    ) -> ForwardOutput:
        """Run ``input_ids [B, T]`` at explicit ``position_ids [B, T]``.

        ``kv`` supplies earlier tokens' K/V and receives this step's.  ``attention_mask``
        is ``None`` for an unpadded causal pass, otherwise boolean ``[B, 1, T, S]``
        over the ``S`` tokens attention will see (history plus this step).  ``trace``
        is called with a step name and tensor at every stage (``layer.*`` once per layer).
        """
        if tuple(position_ids.shape) != tuple(input_ids.shape):
            raise ValueError(
                f"position_ids shape {tuple(position_ids.shape)} must match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )

        def note(name: str, tensor: Tensor) -> Tensor:
            if trace is not None:
                trace(name, tensor)
            return tensor

        spec, weights = self.spec, self.weights
        batch, tokens = input_ids.shape

        hidden = note("embedding", F.embedding(input_ids, weights.embed))  # [B, T, C]
        captured = [hidden] if capture_hidden else None
        cos, sin = rope_tables(position_ids, self._inv_freq, hidden.dtype)  # [B, T, D] each
        note("rope.cos", cos)
        note("rope.sin", sin)

        for index, layer in enumerate(weights.layers):
            # Attention block: x + o_proj(attention(rope(q), rope(k), v)) on the normed input.
            residual = hidden
            x = note("layer.input_norm", rms_norm(hidden, layer.input_norm, spec.rms_norm_eps))
            q = F.linear(x, layer.q_proj, layer.q_bias).view(batch, tokens, -1, spec.head_dim)
            k = F.linear(x, layer.k_proj, layer.k_bias).view(batch, tokens, -1, spec.head_dim)
            v = F.linear(x, layer.v_proj, layer.v_bias).view(batch, tokens, -1, spec.head_dim)
            q = note("layer.q", q.transpose(1, 2))  # [B, H, T, D]
            k = note("layer.k", k.transpose(1, 2))  # [B, Hkv, T, D]
            v = note("layer.v", v.transpose(1, 2))
            q, k = apply_rope(q, k, cos, sin)
            if kv is not None:
                k, v = kv.update(index, k, v)  # [B, Hkv, S, D] including history
            note("layer.k_history", k)
            attended = note(
                "layer.attention",
                attention(q, k, v, num_kv_groups=spec.num_kv_groups, mask=attention_mask),
            )  # [B, T, H * D]
            hidden = residual + note("layer.o_proj", F.linear(attended, layer.o_proj))

            # MLP block: x + swiglu(normed x).
            residual = hidden
            x = note("layer.post_norm", rms_norm(hidden, layer.post_norm, spec.rms_norm_eps))
            mlp = swiglu_mlp(x, layer.gate_proj, layer.up_proj, layer.down_proj)
            hidden = note("layer.output", residual + note("layer.mlp", mlp))
            if captured is not None:
                captured.append(hidden)

        if kv is not None:
            kv.commit(tokens)
        final_hidden = note("final_norm", rms_norm(hidden, weights.final_norm, spec.rms_norm_eps))
        return ForwardOutput(
            logits=note("logits", F.linear(final_hidden, weights.lm_head)),
            final_hidden=final_hidden,
            layer_outputs=None if captured is None else tuple(captured),
        )
