"""Greedy generation on the owned Qwen2 forward pass."""

from __future__ import annotations

from typing import Any

import torch

from tinyinfer.model.kv import ContiguousKV
from tinyinfer.model.qwen2 import Qwen2Model
from tinyinfer.runtime.generation import GenerationObserver, _NoOpGenerationObserver
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult, StopReason


def greedy_generate(
    model: Qwen2Model,
    tokenizer: Any,
    request: GenerationRequest,
    *,
    eos_token_id: int,
    observer: GenerationObserver | None = None,
) -> GenerationResult:
    """Prefill the prompt once, then feed one token per step until EOS or the length cap.

    Mirrors ``tinyinfer.runtime.generation.generate_greedy`` (the G1 reference loop),
    with the owned forward pass and a KV cache sized to this request.
    """
    events = observer if observer is not None else _NoOpGenerationObserver()
    events.request_started()
    input_ids = tokenizer(request.prompt, return_tensors="pt", add_special_tokens=True)[
        "input_ids"
    ].to(model.device)
    prompt_tokens = int(input_ids.shape[1])
    kv = ContiguousKV.for_spec(
        model.spec,
        batch=1,
        max_tokens=prompt_tokens + request.max_new_tokens,
        device=model.device,
        dtype=model.dtype,
    )

    events.prefill_started()
    positions = torch.arange(prompt_tokens, device=model.device)[None]
    logits = model.forward(input_ids, positions, kv=kv).logits
    output_token_ids = [int(logits[0, -1].argmax())]
    events.prefill_finished()
    events.first_token_ready()

    events.decode_started()
    while output_token_ids[-1] != eos_token_id and len(output_token_ids) < request.max_new_tokens:
        step = torch.tensor([[output_token_ids[-1]]], device=model.device)
        position = torch.tensor([[kv.length]], device=model.device)
        logits = model.forward(step, position, kv=kv).logits
        output_token_ids.append(int(logits[0, -1].argmax()))
    events.decode_finished(len(output_token_ids))

    stop_reason: StopReason = "eos" if output_token_ids[-1] == eos_token_id else "length"
    result = GenerationResult(
        request_id=request.request_id,
        prompt_token_ids=tuple(int(token) for token in input_ids[0].tolist()),
        output_token_ids=tuple(output_token_ids),
        text=tokenizer.decode(output_token_ids, skip_special_tokens=True),
        stop_reason=stop_reason,
    )
    events.request_finished()
    return result
