"""The owned forward pass on the paged pool must equal the contiguous path."""

from typing import Any

import torch

from tinyinfer.model.kv import ContiguousKV
from tinyinfer.model.paged import PagedKVPool, PagedSequence
from tinyinfer.model.qwen2 import Qwen2Model

CPU = torch.device("cpu")
PROMPTS = ([5, 17, 42], [9, 33, 61, 2, 77], [1, 2, 3, 4, 5, 6, 7])
STEPS = (11, 22, 33, 44)  # fixed decode inputs, so logits compare without argmax


def _model(tiny_qwen2: Any, dtype: torch.dtype) -> Qwen2Model:
    return Qwen2Model.from_pretrained(tiny_qwen2.model_dir, device=CPU, dtype=dtype)


def _poisoned_pool(model: Qwen2Model) -> PagedKVPool:
    pool = PagedKVPool.for_spec(
        model.spec, block_size=4, num_blocks=16, device=CPU, dtype=model.dtype
    )
    pool.buffer[:, :, 1:] = float("nan")  # every block except the null block
    return pool


def _contiguous_logits(model: Qwen2Model, prompt: list[int]) -> list[torch.Tensor]:
    """Logits of the prefill's last position, then of each fixed decode step."""
    kv = ContiguousKV.for_spec(model.spec, batch=1, max_tokens=16, device=CPU, dtype=model.dtype)
    ids = torch.tensor([prompt])
    logits = [model.forward(ids, torch.arange(len(prompt))[None], kv=kv).logits[0, -1]]
    for token in STEPS:
        position = torch.tensor([[kv.length]])
        logits.append(model.forward(torch.tensor([[token]]), position, kv=kv).logits[0, -1])
    return logits


def _paged_prefill(model: Qwen2Model, pool: PagedKVPool, prompt: list[int], name: str) -> Any:
    sequence = pool.open(name, max_tokens=len(prompt) + len(STEPS))
    view = pool.batch([sequence], new_tokens=len(prompt))
    output = model.forward(
        torch.tensor([prompt]), view.position_ids, kv=view, attention_mask=view.mask
    )
    return sequence, output.logits[0, -1]


def _paged_decode(
    model: Qwen2Model, pool: PagedKVPool, sequences: list[PagedSequence], token: int
) -> torch.Tensor:
    view = pool.batch(sequences, new_tokens=1)
    ids = torch.full((len(sequences), 1), token)
    return model.forward(ids, view.position_ids, kv=view, attention_mask=view.mask).logits[:, -1]


def test_single_request_paged_matches_contiguous_bit_for_bit_in_bf16(tiny_qwen2: Any) -> None:
    model = _model(tiny_qwen2, torch.bfloat16)
    pool = _poisoned_pool(model)
    prompt = PROMPTS[2]

    sequence, first = _paged_prefill(model, pool, prompt, "r0")
    paged = [first] + [_paged_decode(model, pool, [sequence], token)[0] for token in STEPS]

    for actual, expected in zip(paged, _contiguous_logits(model, prompt), strict=True):
        assert torch.equal(actual, expected)


def test_heterogeneous_batch_matches_each_request_alone(tiny_qwen2: Any) -> None:
    model = _model(tiny_qwen2, torch.float32)
    pool = _poisoned_pool(model)
    started = [_paged_prefill(model, pool, list(p), f"r{i}") for i, p in enumerate(PROMPTS)]
    sequences = [sequence for sequence, _ in started]

    batched = [torch.stack([first for _, first in started])]
    for token in STEPS:
        batched.append(_paged_decode(model, pool, sequences, token))

    assert all(torch.isfinite(step).all() for step in batched)
    for row, prompt in enumerate(PROMPTS):
        alone = _contiguous_logits(model, list(prompt))
        for step, expected in enumerate(alone):
            torch.testing.assert_close(batched[step][row], expected, atol=1e-5, rtol=1e-4)
    assert [sequence.length for sequence in sequences] == [len(p) + len(STEPS) for p in PROMPTS]


def test_released_blocks_are_reused_without_leaking_old_tokens(tiny_qwen2: Any) -> None:
    model = _model(tiny_qwen2, torch.float32)
    pool = _poisoned_pool(model)
    old, _ = _paged_prefill(model, pool, PROMPTS[2], "old")
    pool.release(old)

    fresh, first = _paged_prefill(model, pool, PROMPTS[0], "fresh")
    later = _paged_decode(model, pool, [fresh], STEPS[0])[0]

    expected = _contiguous_logits(model, PROMPTS[0])
    torch.testing.assert_close(first, expected[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(later, expected[1], atol=1e-5, rtol=1e-4)
