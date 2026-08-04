"""Explicit single-request greedy prefill and one-token decode."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

from tinyinfer.cache.block import BlockTable
from tinyinfer.cache.contiguous import ContiguousKVCache, allocate_contiguous_cache
from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.cache.paged_tinyinfer_cache import PagedTinyinferCache
from tinyinfer.cache.tinyinfer_cache import TinyinferCache
from tinyinfer.runtime.hf_loader import LoadedModel
from tinyinfer.runtime.request import GenerationRequest
from tinyinfer.runtime.result import GenerationResult, StopReason


@dataclass(frozen=True, slots=True)
class PrefillOutput:
    next_token_id: int
    cache: Any
    attention_mask: Any


@dataclass(frozen=True, slots=True)
class DecodeOutput:
    next_token_id: int
    cache: Any
    attention_mask: Any


class GenerationObserver(Protocol):
    def request_started(self) -> None: ...

    def prefill_started(self) -> None: ...

    def prefill_finished(self) -> None: ...

    def first_token_ready(self) -> None: ...

    def decode_started(self) -> None: ...

    def decode_finished(self, output_tokens: int) -> None: ...

    def request_finished(self) -> None: ...


class _NoOpGenerationObserver:
    def request_started(self) -> None:
        pass

    def prefill_started(self) -> None:
        pass

    def prefill_finished(self) -> None:
        pass

    def first_token_ready(self) -> None:
        pass

    def decode_started(self) -> None:
        pass

    def decode_finished(self, output_tokens: int) -> None:
        pass

    def request_finished(self) -> None:
        pass


_NO_OP_OBSERVER = _NoOpGenerationObserver()


def _get_torch() -> Any:
    """Return the torch module (lazily imported and cached by importlib)."""
    return importlib.import_module("torch")


def _cache_length(cache: Any) -> int:
    if cache is None or not callable(getattr(cache, "get_seq_length", None)):
        raise RuntimeError("model output must expose a public Cache with get_seq_length()")
    length = cache.get_seq_length()
    if type(length) is not int or length < 0:
        raise RuntimeError("cache length must be a non-negative integer")
    return length


def _select_next_token(logits: Any) -> int:
    shape = tuple(logits.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] < 1:
        raise RuntimeError(f"invalid logits shape: {shape}")
    return int(logits[:, -1, :].argmax(dim=-1).item())


def _model_cache_dimensions(model: Any) -> dict[str, int]:
    """Extract cache dimensions from an HF model config."""
    config = model.config
    return {
        "num_layers": int(config.num_hidden_layers),
        "num_kv_heads": int(
            getattr(config, "num_key_value_heads", config.num_attention_heads)
        ),
        "head_dim": int(config.hidden_size // config.num_attention_heads),
        "max_seq_len": int(getattr(config, "max_position_embeddings", 4096)),
    }


def prefill(model: Any, input_ids: Any, attention_mask: Any) -> PrefillOutput:
    """Run the full prompt once and return the first greedy token."""

    if tuple(input_ids.shape)[:1] != (1,) or len(tuple(input_ids.shape)) != 2:
        raise ValueError("prefill input_ids must have shape [1, prompt_tokens]")
    prompt_tokens = int(input_ids.shape[1])
    if prompt_tokens < 1 or tuple(attention_mask.shape) != (1, prompt_tokens):
        raise ValueError("prefill attention_mask must match the non-empty prompt")
    torch = _get_torch()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    cache = outputs.past_key_values
    cache_length = _cache_length(cache)
    if cache_length != prompt_tokens:
        raise RuntimeError(f"prefill cache length must be {prompt_tokens}, got {cache_length}")
    return PrefillOutput(
        next_token_id=_select_next_token(outputs.logits),
        cache=cache,
        attention_mask=attention_mask,
    )


def decode_one(
    model: Any,
    last_token_id: int,
    attention_mask: Any,
    cache: Any,
) -> DecodeOutput:
    """Feed exactly one generated token and advance the public Cache once."""

    if type(last_token_id) is not int or last_token_id < 0:
        raise ValueError("last_token_id must be a non-negative integer")
    old_cache_length = _cache_length(cache)
    if tuple(attention_mask.shape) != (1, old_cache_length):
        raise RuntimeError("attention mask length must equal cache length before decode")
    torch = _get_torch()
    input_ids = torch.tensor(
        [[last_token_id]],
        device=attention_mask.device,
        dtype=torch.long,
    )
    next_mask_value = torch.ones(
        (1, 1),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    extended_attention_mask = torch.cat(
        [attention_mask, next_mask_value],
        dim=1,
    )
    cache_position = torch.arange(
        old_cache_length,
        old_cache_length + 1,
        device=attention_mask.device,
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=extended_attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
    updated_cache = outputs.past_key_values
    new_cache_length = _cache_length(updated_cache)
    if new_cache_length != old_cache_length + 1:
        raise RuntimeError(
            f"decode cache length must advance from {old_cache_length} "
            f"to {old_cache_length + 1}, got {new_cache_length}"
        )
    return DecodeOutput(
        next_token_id=_select_next_token(outputs.logits),
        cache=updated_cache,
        attention_mask=extended_attention_mask,
    )


def generate_greedy(
    loaded: LoadedModel,
    request: GenerationRequest,
    observer: GenerationObserver | None = None,
) -> GenerationResult:
    """Generate one request with explicit prefill and one-token decode."""

    active_observer = observer if observer is not None else _NO_OP_OBSERVER
    active_observer.request_started()
    encoded = loaded.tokenizer(
        request.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(loaded.model.device)
    attention_mask = encoded["attention_mask"].to(loaded.model.device)
    prompt_token_ids = tuple(int(token_id) for token_id in input_ids[0].tolist())
    active_observer.prefill_started()
    first = prefill(loaded.model, input_ids, attention_mask)
    active_observer.prefill_finished()
    output_token_ids = [first.next_token_id]
    active_observer.first_token_ready()
    cache = first.cache
    attention_mask = first.attention_mask

    active_observer.decode_started()
    while (
        output_token_ids[-1] != loaded.eos_token_id
        and len(output_token_ids) < request.max_new_tokens
    ):
        decoded = decode_one(
            loaded.model,
            output_token_ids[-1],
            attention_mask,
            cache,
        )
        output_token_ids.append(decoded.next_token_id)
        cache = decoded.cache
        attention_mask = decoded.attention_mask

    active_observer.decode_finished(len(output_token_ids))
    stop_reason: StopReason = "eos" if output_token_ids[-1] == loaded.eos_token_id else "length"
    text = loaded.tokenizer.decode(output_token_ids, skip_special_tokens=True)
    result = GenerationResult(
        request_id=request.request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=tuple(output_token_ids),
        text=text,
        stop_reason=stop_reason,
    )
    active_observer.request_finished()
    return result


def reference_generate(
    loaded: LoadedModel,
    request: GenerationRequest,
) -> tuple[int, ...]:
    """Return only the generated suffix from Hugging Face ``generate``."""

    encoded = loaded.tokenizer(
        request.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(loaded.model.device)
    attention_mask = encoded["attention_mask"].to(loaded.model.device)
    prompt_tokens = int(input_ids.shape[1])
    torch = _get_torch()
    with torch.inference_mode():
        generated = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            max_new_tokens=request.max_new_tokens,
            eos_token_id=loaded.eos_token_id,
            pad_token_id=loaded.pad_token_id,
        )
    suffix = tuple(int(token_id) for token_id in generated[0, prompt_tokens:].tolist())
    if not suffix:
        raise RuntimeError("Hugging Face reference returned no generated tokens")
    return suffix


# ---------------------------------------------------------------------------
# TinyInfer-owned cache generation (Phase 1.5 — contiguous)
# ---------------------------------------------------------------------------


def generate_with_owned_cache(
    loaded: LoadedModel,
    request: GenerationRequest,
    observer: GenerationObserver | None = None,
) -> GenerationResult:
    """Generate with TinyInfer-owned contiguous KV cache — no DynamicCache rebuild.

    A ``TinyinferCache`` is passed as ``past_key_values`` directly into the HF
    forward.  HF attention layers call ``update()`` / ``__getitem__()`` on it,
    which read from and write to the TinyInfer ContiguousKVCache buffer.
    """
    active_observer = observer if observer is not None else _NO_OP_OBSERVER
    active_observer.request_started()

    encoded = loaded.tokenizer(
        request.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(loaded.model.device)
    attention_mask = encoded["attention_mask"].to(loaded.model.device)
    prompt_token_ids = tuple(int(token_id) for token_id in input_ids[0].tolist())
    torch = _get_torch()
    dims = _model_cache_dimensions(loaded.model)

    active_observer.prefill_started()

    contiguous = allocate_contiguous_cache(
        num_layers=dims["num_layers"],
        num_kv_heads=dims["num_kv_heads"],
        max_seq_len=dims["max_seq_len"],
        head_dim=dims["head_dim"],
        device=loaded.model.device,
        dtype=torch.bfloat16,
    )
    tinfer_cache = TinyinferCache(contiguous)

    with torch.inference_mode():
        outputs = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=tinfer_cache,
            use_cache=True,
            return_dict=True,
        )
    first_token_id = _select_next_token(outputs.logits)
    output_token_ids = [first_token_id]
    active_observer.prefill_finished()
    active_observer.first_token_ready()

    active_observer.decode_started()
    while (
        output_token_ids[-1] != loaded.eos_token_id
        and len(output_token_ids) < request.max_new_tokens
    ):
        last_token = output_token_ids[-1]
        old_length = contiguous.length

        input_ids_step = torch.tensor(
            [[last_token]],
            device=loaded.model.device,
            dtype=torch.long,
        )
        extended_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )
        cache_position = torch.arange(
            old_length,
            old_length + 1,
            device=loaded.model.device,
        )
        with torch.inference_mode():
            decode_outputs = loaded.model(
                input_ids=input_ids_step,
                attention_mask=extended_mask,
                past_key_values=tinfer_cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        next_token_id = _select_next_token(decode_outputs.logits)
        output_token_ids.append(next_token_id)
        attention_mask = extended_mask

    active_observer.decode_finished(len(output_token_ids))
    stop_reason: StopReason = (
        "eos" if output_token_ids[-1] == loaded.eos_token_id else "length"
    )
    text = loaded.tokenizer.decode(output_token_ids, skip_special_tokens=True)
    result = GenerationResult(
        request_id=request.request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=tuple(output_token_ids),
        text=text,
        stop_reason=stop_reason,
    )
    active_observer.request_finished()
    return result


# ---------------------------------------------------------------------------
# Paged KV cache generation (Phase 2)
# ---------------------------------------------------------------------------


def generate_with_paged_cache(
    loaded: LoadedModel,
    request: GenerationRequest,
    observer: GenerationObserver | None = None,
) -> GenerationResult:
    """Generate with TinyInfer paged KV cache — block-level allocation.

    A ``PagedTinyinferCache`` is passed as ``past_key_values``.  HF attention
    reads/writes through the paged block table into the physical block pool.
    """
    active_observer = observer if observer is not None else _NO_OP_OBSERVER
    active_observer.request_started()

    encoded = loaded.tokenizer(
        request.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(loaded.model.device)
    attention_mask = encoded["attention_mask"].to(loaded.model.device)
    prompt_token_ids = tuple(int(token_id) for token_id in input_ids[0].tolist())
    torch = _get_torch()
    dims = _model_cache_dimensions(loaded.model)

    active_observer.prefill_started()

    block_size = 16
    max_tokens = dims["max_seq_len"]
    num_physical = (max_tokens + block_size - 1) // block_size

    paged = PagedKVCache(
        num_layers=dims["num_layers"],
        num_kv_heads=dims["num_kv_heads"],
        block_size=block_size,
        head_dim=dims["head_dim"],
        num_physical_blocks=num_physical,
        device=loaded.model.device,
        dtype=torch.bfloat16,
    )
    block_table = paged.create_block_table(max_tokens)
    prompt_len = int(input_ids.shape[1])
    total_blocks = (prompt_len + request.max_new_tokens + block_size - 1) // block_size
    paged.allocate_blocks(block_table, total_blocks)

    tinfer_cache = PagedTinyinferCache(paged, block_table)

    with torch.inference_mode():
        outputs = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=tinfer_cache,
            use_cache=True,
            return_dict=True,
        )
    first_token_id = _select_next_token(outputs.logits)
    output_token_ids = [first_token_id]
    active_observer.prefill_finished()
    active_observer.first_token_ready()

    active_observer.decode_started()
    while (
        output_token_ids[-1] != loaded.eos_token_id
        and len(output_token_ids) < request.max_new_tokens
    ):
        last_token = output_token_ids[-1]
        old_length = tinfer_cache.length

        input_ids_step = torch.tensor(
            [[last_token]],
            device=loaded.model.device,
            dtype=torch.long,
        )
        extended_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )
        cache_position = torch.arange(
            old_length,
            old_length + 1,
            device=loaded.model.device,
        )
        with torch.inference_mode():
            decode_outputs = loaded.model(
                input_ids=input_ids_step,
                attention_mask=extended_mask,
                past_key_values=tinfer_cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        next_token_id = _select_next_token(decode_outputs.logits)
        output_token_ids.append(next_token_id)
        attention_mask = extended_mask

    active_observer.decode_finished(len(output_token_ids))
    stop_reason: StopReason = (
        "eos" if output_token_ids[-1] == loaded.eos_token_id else "length"
    )
    text = loaded.tokenizer.decode(output_token_ids, skip_special_tokens=True)
    result = GenerationResult(
        request_id=request.request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=tuple(output_token_ids),
        text=text,
        stop_reason=stop_reason,
    )
    active_observer.request_finished()
    return result