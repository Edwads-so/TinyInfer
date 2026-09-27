from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from tinyinfer.model.ops import (
    apply_rope,
    attention,
    rms_norm,
    rope_inv_freq,
    rope_tables,
    swiglu_mlp,
)

DTYPES = [torch.float32, torch.bfloat16]


def _same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """FP32 within the plan's tolerance; BF16 bit-for-bit (same op order as HF)."""
    if actual.dtype == torch.bfloat16:
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_matches_hf(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    reference = Qwen2RMSNorm(64, eps=1e-6)
    with torch.no_grad():
        reference.weight.copy_(torch.randn(64))
    reference = reference.to(dtype)
    x = torch.randn(2, 5, 64, dtype=dtype)

    _same(rms_norm(x, reference.weight.detach(), 1e-6), reference(x).detach())


@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_matches_hf(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    config = Qwen2Config(
        hidden_size=64,
        num_attention_heads=4,
        max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )
    reference = Qwen2RotaryEmbedding(config)
    position_ids = torch.tensor([[0, 1, 2, 3], [7, 8, 9, 10]])
    x = torch.randn(2, 4, 64, dtype=dtype)
    q = torch.randn(2, 4, 4, 16, dtype=dtype)
    k = torch.randn(2, 2, 4, 16, dtype=dtype)

    hf_cos, hf_sin = reference(x, position_ids)
    cos, sin = rope_tables(position_ids, rope_inv_freq(16, 1000000.0), dtype)
    _same(cos, hf_cos)
    _same(sin, hf_sin)
    for actual, expected in zip(
        apply_rope(q, k, cos, sin), apply_rotary_pos_emb(q, k, hf_cos, hf_sin), strict=True
    ):
        _same(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_mlp_matches_hf(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    reference = Qwen2MLP(Qwen2Config(hidden_size=64, intermediate_size=160)).to(dtype)
    x = torch.randn(2, 3, 64, dtype=dtype)

    actual = swiglu_mlp(
        x,
        reference.gate_proj.weight.detach(),
        reference.up_proj.weight.detach(),
        reference.down_proj.weight.detach(),
    )
    _same(actual, reference(x).detach())


def _hf_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    module = SimpleNamespace(num_key_value_groups=q.shape[1] // k.shape[1], is_causal=True)
    output, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=q.shape[-1] ** -0.5)
    return output.reshape(q.shape[0], q.shape[2], -1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("q_len", "kv_len", "masked"),
    [(5, 5, False), (1, 6, False), (1, 6, True), (3, 3, True)],
    ids=["prefill", "decode", "decode-masked", "prefill-masked"],
)
def test_attention_matches_hf_sdpa_dispatch(
    dtype: torch.dtype, q_len: int, kv_len: int, masked: bool
) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 4, q_len, 16, dtype=dtype)
    k = torch.randn(2, 2, kv_len, 16, dtype=dtype)
    v = torch.randn(2, 2, kv_len, 16, dtype=dtype)
    mask = None
    if masked:
        causal = torch.ones(q_len, kv_len, dtype=torch.bool).tril(kv_len - q_len)
        mask = causal.expand(2, 1, q_len, kv_len).clone()
        mask[1, :, :, 0] = False  # left padding on row 1

    actual = attention(q, k, v, num_kv_groups=2, mask=mask)
    _same(actual, _hf_attention(q, k, v, mask))
    assert actual.shape == (2, q_len, 64)


def test_attention_rejects_ambiguous_unmasked_chunk() -> None:
    q = torch.randn(1, 4, 2, 16)
    k = torch.randn(1, 2, 5, 16)

    with pytest.raises(ValueError, match="mask"):
        attention(q, k, k, num_kv_groups=2, mask=None)
