"""Fake-model tests proving one heterogeneous batched decode forward."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")


class FakeDenseModel:
    """Dense-tensor model double that records decode calls and exercises cache."""

    def __init__(self, num_layers: int = 2) -> None:
        self.num_layers = num_layers
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_key_value_heads=2,
            num_attention_heads=2,
            hidden_size=4,
            max_position_embeddings=64,
        )
        self.decode_calls: list[dict[str, Any]] = []
        self.eos_token_id = 999

    def __call__(self, **kwargs: Any) -> Any:
        """Return row-distinct logits; capture decode call evidence."""
        input_ids = kwargs["input_ids"]
        past = kwargs.get("past_key_values")
        batch = int(input_ids.shape[0])
        # Classify decode call by cache type name
        if past is not None:
            self.decode_calls.append(
                {
                    "input_shape": tuple(input_ids.shape),
                    "position_ids": kwargs.get("position_ids").tolist()
                    if kwargs.get("position_ids") is not None
                    else None,
                    "cache_type": type(past).__name__,
                    "attention_mask_shape": tuple(kwargs["attention_mask"].shape),
                }
            )
            # Exercise the cache: simulate layer updates so lengths advance
            for layer in range(self.num_layers):
                kv = torch.zeros(batch, 2, 1, 4)
                past.update(kv, kv.clone(), layer)
        logits = torch.randn(batch, 1, 10)
        return SimpleNamespace(logits=logits, past_key_values=past)


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        # stable length per prompt
        return [7] * (len(text) % 5 + 1)

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        n = len(prompt) % 5 + 1
        input_ids = torch.tensor([[10 + i for i in range(n)]])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)


def make_engine() -> Any:
    from tinyinfer.cache.paged import PagedKVCache
    from tinyinfer.scheduler.engine import BatchEngine

    model = FakeDenseModel(num_layers=2)
    paged = PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        block_size=4,
        head_dim=4,
        num_physical_blocks=64,
        device="cpu",
        dtype=torch.bfloat16,
    )
    loaded = SimpleNamespace(
        model=model,
        tokenizer=DummyTokenizer(),
        eos_token_id=999,
        model_id="fake",
        model_revision="rev",
    )
    return BatchEngine(loaded, paged, max_batch=8)


def make_req(rid: str, prompt: str, max_new: int = 3) -> Any:
    from tinyinfer.runtime.request import GenerationRequest

    return GenerationRequest(rid, prompt, max_new)


class TestOneBatchedForward:
    def test_b3_one_call_shape_position(self) -> None:
        from tinyinfer.runtime.generation import _select_next_tokens

        # direct unit check of the batch selector
        logits = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 0.5]]])
        tokens = _select_next_tokens(logits, 3)
        assert tokens == (1, 0, 1)

        engine = make_engine()
        # max_new=1 => exactly one decode iteration, guaranteeing a single
        # batched (3,1) forward (prefill calls are batch=1).
        engine.submit(make_req("a", "aaaa", 1))
        engine.submit(make_req("b", "bbbbbb", 1))
        engine.submit(make_req("c", "cccccccc", 1))
        engine.run()

        decode_calls = engine._loaded.model.decode_calls
        # prefill calls are (1, n) so they never match (3, 1)
        b3 = [c for c in decode_calls if c["input_shape"] == (3, 1)]
        assert len(b3) == 1
        call = b3[0]
        assert call["cache_type"] == "BatchedPagedTinyinferCache"
        assert len(call["position_ids"]) == 3
        # position_ids is [[len_a],[len_b],[len_c]]; mask width = max_len + 1
        max_old = max(p[0] for p in call["position_ids"])
        assert call["attention_mask_shape"] == (3, max_old + 1)

    def test_batch_token_selection_rejects_mismatch(self) -> None:
        from tinyinfer.runtime.generation import _select_next_tokens

        with pytest.raises(RuntimeError, match="expected"):
            _select_next_tokens(torch.randn(2, 1, 5), 3)


class TestSelectNextTokens:
    def test_selects_one_per_row(self) -> None:
        from tinyinfer.runtime.generation import _select_next_tokens

        logits = torch.tensor([[[0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]])
        assert _select_next_tokens(logits, 2) == (1, 0)

    def test_rejects_flat(self) -> None:
        from tinyinfer.runtime.generation import _select_next_tokens

        with pytest.raises(RuntimeError, match="invalid logits shape"):
            _select_next_tokens(torch.randn(5), 1)
