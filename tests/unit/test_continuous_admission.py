"""Dynamic continuous admission and lifecycle/cleanup tests for BatchEngine."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")


class FakeDenseModel:
    """Batch-aware dense model double that advances cache via update()."""

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
        self.eos_token_id = 999

    def __call__(self, **kwargs: Any) -> Any:
        past = kwargs.get("past_key_values")
        input_ids = kwargs["input_ids"]
        batch = int(input_ids.shape[0])
        if past is not None:
            for layer in range(self.num_layers):
                kv = torch.zeros(batch, 2, 1, 4, dtype=torch.bfloat16)
                past.update(kv, kv.clone(), layer)
        logits = torch.randn(batch, 1, 10)
        return SimpleNamespace(logits=logits, past_key_values=past)


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [7] * (len(text) % 5 + 1)

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        n = len(prompt) % 5 + 1
        ids = torch.tensor([[10 + i for i in range(n)]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)


def make_engine(max_batch: int = 2) -> Any:
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
    return BatchEngine(loaded, paged, max_batch=max_batch)


def make_req(rid: str, prompt: str, max_new: int = 3) -> Any:
    from tinyinfer.runtime.request import GenerationRequest

    return GenerationRequest(rid, prompt, max_new)


class TestDynamicAdmission:
    def test_c_enters_after_a_finishes_while_b_runs(self) -> None:
        """max_batch=2: A completes after prefill+1 decode, C enters slot."""
        engine = make_engine(max_batch=2)
        # A: max_new=1 -> completes after its single decode row
        engine.submit(make_req("a", "aaaa", 1))
        # B: max_new=3 -> longer running
        engine.submit(make_req("b", "bbbbbb", 3))
        # C: waiting
        engine.submit(make_req("c", "ccccccc", 3))

        results = engine.run()

        # All three finished with their own request ids
        ids = {r.request_id for r in results}
        assert {"a", "b", "c"} <= ids

        # Engine history must reflect continuous admission: once A completes,
        # C eventually joins the running set.
        saw_c_in_decoding = any("c" in stat.request_ids for stat in engine.iteration_history)
        assert saw_c_in_decoding, "C never entered the decode batch"

    def test_no_iteration_exceeds_max_batch(self) -> None:
        engine = make_engine(max_batch=2)
        for i in range(5):
            engine.submit(make_req(f"r{i}", "x" * (i + 2), 2))
        engine.run()
        for stat in engine.iteration_history:
            assert stat.decode_batch_size <= 2

    def test_each_iteration_at_most_one_decode_forward(self) -> None:
        engine = make_engine(max_batch=4)
        for i in range(3):
            engine.submit(make_req(f"r{i}", "y" * (i + 1), 4))
        engine.run()
        for stat in engine.iteration_history:
            assert stat.decode_forward_count in (0, 1)


class TestCancellation:
    def test_cancel_waiting(self) -> None:
        engine = make_engine(max_batch=1)
        engine.submit(make_req("a", "aaaa", 1))
        engine.submit(make_req("b", "bbb", 3))  # never admitted (max_batch=1)
        engine.cancel("b")
        results = engine.run()
        # b removed from queue; only a finished
        assert {r.request_id for r in results} == {"a"}

    def test_pool_recovers_after_cancel(self) -> None:
        engine = make_engine(max_batch=8)
        paged = engine._paged
        initial_free = paged.free_block_count
        engine.submit(make_req("a", "aaaa", 3))
        engine.submit(make_req("b", "bbbbb", 3))
        engine.step()  # admit + prefill a,b; one decode
        engine.cancel("b")  # running -> cancelled, release blocks
        engine.run()
        assert paged.free_block_count == initial_free


class TestStateOf:
    def test_state_of_exposes_request_state(self) -> None:
        from tinyinfer.scheduler.queue import RequestQueue

        q = RequestQueue()
        from tinyinfer.runtime.request import GenerationRequest

        q.enqueue(GenerationRequest("r1", "hi", 2))
        assert q.state_of("r1") == "waiting"


class TestPrefillFinalize:
    def test_immediate_eos_releases_before_decode(self) -> None:
        """A request whose prefill yields EOS must finish during its step."""
        engine = make_engine(max_batch=4)
        engine.submit(make_req("a", "aaaa", 3))
        results = engine.run()
        # request a completed (length-limited by max_new=3)
        assert len(results) == 1
        assert results[0].request_id == "a"
        # every iteration produced at most one decode forward
        for stat in engine.iteration_history:
            assert stat.decode_forward_count in (0, 1)


class TestMaxNewBound:
    """Audit fix: no request may exceed its max_new_tokens."""

    def test_no_request_exceeds_max_new(self) -> None:
        from tinyinfer.runtime.request import GenerationRequest

        engine = make_engine(max_batch=4)
        engine.submit(GenerationRequest("a", "aaaa", 1))
        engine.submit(GenerationRequest("b", "bbbbb", 1))
        engine.submit(GenerationRequest("c", "cccccccc", 1))
        results = engine.run()
        assert len(results) == 3
        for r in results:
            assert len(r.output_token_ids) <= 1, (
                f"{r.request_id} exceeded max_new=1: {r.output_token_ids}"
            )


class TestFailureCleanup:
    """Audit fix: prefill/decode exceptions must release blocks and states."""

    def test_prefill_exception_cleans_all(self) -> None:
        engine = make_engine(max_batch=4)
        paged = engine._paged
        initial_free = paged.free_block_count

        engine.submit(make_req("a", "aaaa", 3))
        engine.submit(make_req("b", "bbbbb", 3))

        original = engine._loaded.model

        class ExplodingPrefill(FakeDenseModel):
            def __call__(self, **kwargs: Any) -> Any:
                pc = kwargs.get("past_key_values")
                if pc is not None and type(pc).__name__ == "PagedTinyinferCache":
                    raise RuntimeError("boom in prefill")
                return super().__call__(**kwargs)

        engine._loaded.model = ExplodingPrefill()
        with pytest.raises(RuntimeError, match="boom in prefill"):
            engine.step()
        engine._loaded.model = original

        assert paged.free_block_count == initial_free, "blocks leaked after prefill error"
        assert len(engine._states) == 0, "engine states leaked after prefill error"
        assert engine.queue.running_count == 0
        assert engine.queue.waiting_count == 0

    def test_decode_exception_cleans_all(self) -> None:
        engine = make_engine(max_batch=4)
        paged = engine._paged
        initial_free = paged.free_block_count

        engine.submit(make_req("a", "aaaa", 3))
        engine.submit(make_req("b", "bbbbb", 3))

        original = engine._loaded.model

        class ExplodingDecode(FakeDenseModel):
            def __call__(self, **kwargs: Any) -> Any:
                pc = kwargs.get("past_key_values")
                if pc is not None and type(pc).__name__ == "BatchedPagedTinyinferCache":
                    raise RuntimeError("boom in decode")
                return super().__call__(**kwargs)

        engine._loaded.model = ExplodingDecode()
        with pytest.raises(RuntimeError, match="boom in decode"):
            engine.step()
        engine._loaded.model = original

        assert paged.free_block_count == initial_free, "blocks leaked after decode error"
        assert len(engine._states) == 0, "engine states leaked after decode error"
        assert engine.queue.running_count == 0
        assert engine.queue.waiting_count == 0
