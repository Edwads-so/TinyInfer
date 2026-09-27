"""Qwen2 building blocks as plain tensor functions.

Each function reproduces the operation order of Transformers 5.14.1
(``models/qwen2/modeling_qwen2.py`` and ``integrations/sdpa_attention.py``), so
BF16 results match Hugging Face bit for bit, not just within a tolerance.

Shape names: B batch, T query tokens, S key/value tokens, H query heads,
Hkv key/value heads, D head_dim, C hidden size, I intermediate size.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``[..., C] -> [..., C]``.  Normalize in FP32, cast back, then scale."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def rope_inv_freq(head_dim: int, theta: float, device: Any = None) -> Tensor:
    """``[D/2]`` FP32 inverse frequencies ``1 / theta^(2i/D)``."""
    exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
    return 1.0 / (theta ** (exponents / head_dim))


def rope_tables(
    position_ids: Tensor, inv_freq: Tensor, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """``position_ids [B, T] -> (cos, sin)`` each ``[B, T, D]``, computed in FP32."""
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    positions = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.to(positions.device) @ positions).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate ``q [B, H, T, D]`` and ``k [B, Hkv, T, D]`` by their token positions."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(x: Tensor, groups: int) -> Tensor:
    """``[B, Hkv, S, D] -> [B, Hkv * groups, S, D]``; each KV head serves ``groups`` query heads."""
    if groups == 1:
        return x
    batch, kv_heads, length, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, groups, length, head_dim)
    return x.reshape(batch, kv_heads * groups, length, head_dim)


def attention(
    q: Tensor, k: Tensor, v: Tensor, *, num_kv_groups: int, mask: Tensor | None
) -> Tensor:
    """Scaled dot-product attention with GQA: ``q [B, H, T, D]`` -> ``[B, T, H * D]``.

    ``mask`` is ``None`` or boolean ``[B, 1, T, S]`` (True = may attend).  With no
    mask the query must be the whole sequence (prefill, causal) or one token
    (decode, attends to everything); anything else needs an explicit mask.
    Dispatch mirrors Hugging Face: GQA goes through ``enable_gqa`` only when there
    is no mask, otherwise KV heads are repeated first.
    """
    q_len, kv_len = q.shape[2], k.shape[2]
    if mask is None and 1 < q_len < kv_len:
        raise ValueError(
            f"a {q_len}-token query over {kv_len} cached tokens needs an explicit mask"
        )
    extra: dict[str, bool] = {}
    if num_kv_groups > 1:
        if mask is None and k.shape[-1] <= 256:
            extra["enable_gqa"] = True
        else:
            k = _repeat_kv(k, num_kv_groups)
            v = _repeat_kv(v, num_kv_groups)
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        scale=q.shape[-1] ** -0.5,
        is_causal=q_len > 1 and mask is None,
        **extra,
    )
    output = output.transpose(1, 2).contiguous()
    return output.reshape(q.shape[0], q_len, -1)


def swiglu_mlp(x: Tensor, gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Tensor:
    """``[..., C] -> [..., C]`` via ``down(silu(gate(x)) * up(x))``.

    ``gate_proj`` and ``up_proj`` are ``[I, C]``; ``down_proj`` is ``[C, I]``.
    """
    return F.linear(F.silu(F.linear(x, gate_proj)) * F.linear(x, up_proj), down_proj)
