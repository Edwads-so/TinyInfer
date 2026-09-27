from typing import Any

import pytest
import torch
from transformers import Qwen2ForCausalLM

from tinyinfer.model.kv import ContiguousKV
from tinyinfer.model.qwen2 import Qwen2Model

CPU = torch.device("cpu")
PROMPT = torch.tensor([[5, 17, 42, 99, 3, 64, 8]])


def _owned(tiny_qwen2: Any, dtype: torch.dtype = torch.float32) -> Qwen2Model:
    return Qwen2Model.from_pretrained(tiny_qwen2.model_dir, device=CPU, dtype=dtype)


def _positions(input_ids: torch.Tensor, start: int = 0) -> torch.Tensor:
    return torch.arange(start, start + input_ids.shape[1]).expand(input_ids.shape[0], -1)


def _kv(model: Qwen2Model, batch: int, dtype: torch.dtype = torch.float32) -> ContiguousKV:
    return ContiguousKV.for_spec(model.spec, batch=batch, max_tokens=32, device=CPU, dtype=dtype)


def test_every_layer_and_logits_match_hf(tiny_qwen2: Any) -> None:
    model = _owned(tiny_qwen2)
    with torch.no_grad():
        hf = tiny_qwen2.hf_model(input_ids=PROMPT, output_hidden_states=True)

    output = model.forward(PROMPT, _positions(PROMPT), capture_hidden=True)

    assert output.layer_outputs is not None and len(output.layer_outputs) == 3
    # HF returns embeddings, the outputs of all but the last layer, then the final norm.
    for actual, expected in zip(output.layer_outputs[:-1], hf.hidden_states[:-1], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(output.final_hidden, hf.hidden_states[-1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(output.logits, hf.logits, atol=1e-5, rtol=1e-4)


def test_bfloat16_logits_are_bit_identical_to_hf(tiny_qwen2: Any) -> None:
    # Load BF16 the way hf_loader does: from_pretrained keeps RoPE's inv_freq in FP32,
    # whereas model.to(bfloat16) would also round that buffer.
    model = _owned(tiny_qwen2, torch.bfloat16)
    hf_model = Qwen2ForCausalLM.from_pretrained(
        tiny_qwen2.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    with torch.no_grad():
        expected = hf_model(input_ids=PROMPT).logits

    assert torch.equal(model.forward(PROMPT, _positions(PROMPT)).logits, expected)


def test_cached_decode_matches_full_recompute(tiny_qwen2: Any) -> None:
    model = _owned(tiny_qwen2)
    full = model.forward(PROMPT, _positions(PROMPT)).logits
    kv = _kv(model, batch=1)

    prefill = model.forward(PROMPT[:, :4], _positions(PROMPT[:, :4]), kv=kv).logits
    steps = [
        model.forward(PROMPT[:, t : t + 1], _positions(PROMPT[:, :1], start=t), kv=kv).logits
        for t in range(4, PROMPT.shape[1])
    ]

    assert kv.length == PROMPT.shape[1]
    torch.testing.assert_close(torch.cat([prefill, *steps], dim=1), full, atol=1e-5, rtol=1e-4)


def test_equal_length_batch_matches_single_requests(tiny_qwen2: Any) -> None:
    model = _owned(tiny_qwen2)
    batch = torch.stack([PROMPT[0], PROMPT[0].flip(0)])

    together = model.forward(batch, _positions(batch)).logits
    for row in range(2):
        alone = model.forward(batch[row : row + 1], _positions(batch[row : row + 1])).logits
        torch.testing.assert_close(together[row : row + 1], alone, atol=1e-5, rtol=1e-4)


def test_left_padded_row_matches_unpadded_request(tiny_qwen2: Any) -> None:
    model = _owned(tiny_qwen2)
    short = PROMPT[:, :4]
    padded = torch.cat([torch.zeros(1, 3, dtype=torch.long), short], dim=1)
    batch = torch.cat([padded, PROMPT], dim=0)
    valid = torch.tensor([[False] * 3 + [True] * 4, [True] * 7])
    positions = torch.stack([torch.tensor([0, 0, 0, 0, 1, 2, 3]), torch.arange(7)])
    causal = torch.ones(7, 7, dtype=torch.bool).tril()
    mask = causal[None, None] & valid[:, None, None, :]

    logits = model.forward(batch, positions, attention_mask=mask).logits

    alone = model.forward(short, _positions(short)).logits
    torch.testing.assert_close(logits[0:1, 3:], alone, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        logits[1:2], model.forward(PROMPT, _positions(PROMPT)).logits, atol=1e-5, rtol=1e-4
    )


def test_forward_rejects_mismatched_positions(tiny_qwen2: Any) -> None:
    model = _owned(tiny_qwen2)

    with pytest.raises(ValueError, match="position_ids"):
        model.forward(PROMPT, torch.arange(3)[None])
