from types import SimpleNamespace

import pytest
import torch

from benchmarks.bench_phase2 import (
    RequestAlignment,
    TieCheck,
    bf16_tie,
    bf16_ulp,
    build_request_results,
    check_token_alignment,
    make_tie_checker,
)


def _results(**outputs: tuple[int, ...]) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(request_id=request_id, output_token_ids=tokens)
        for request_id, tokens in outputs.items()
    )


def test_bf16_ulp_matches_bfloat16_spacing() -> None:
    assert bf16_ulp(21.0) == 0.125
    assert bf16_ulp(-21.0) == 0.125
    assert bf16_ulp(1.0) == 2.0**-7
    for value in (0.3, 7.0, 21.0, 100.0):
        exact = torch.tensor(value, dtype=torch.bfloat16)
        above = torch.nextafter(exact, torch.tensor(1e9, dtype=torch.bfloat16))
        assert float(above - exact) == bf16_ulp(float(exact))


def test_bf16_tie_accepts_tokens_within_one_ulp_of_the_top() -> None:
    logits = torch.tensor([21.0, 21.0, 20.75, 19.0], dtype=torch.bfloat16)

    tie = bf16_tie(logits, reference_token=0, actual_token=1)

    assert tie == TieCheck(is_tie=True, logit_gap=0.0, tolerance=0.125)


def test_bf16_tie_rejects_a_token_beyond_one_ulp() -> None:
    logits = torch.tensor([21.0, 21.0, 20.75, 19.0], dtype=torch.bfloat16)

    tie = bf16_tie(logits, reference_token=0, actual_token=2)

    assert not tie.is_tie
    assert tie.logit_gap == 0.25


def test_tie_checker_recomputes_each_distinct_divergence_once() -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    def logits_at(request_id: str, prefix: tuple[int, ...]) -> torch.Tensor:
        calls.append((request_id, prefix))
        return torch.tensor([0.0, 21.0, 21.0], dtype=torch.bfloat16)

    tie_check = make_tie_checker(logits_at)
    for _ in range(3):
        assert tie_check("r1", (7, 1), (7, 2), 1).is_tie

    assert calls == [("r1", (7,))]


def test_batched_mode_accepts_bf16_tie_divergence() -> None:
    calls: list[tuple[str, int]] = []

    def tie_check(
        request_id: str, reference: tuple[int, ...], actual: tuple[int, ...], index: int
    ) -> TieCheck:
        calls.append((request_id, index))
        return TieCheck(is_tie=True, logit_gap=0.0, tolerance=0.125)

    alignments = check_token_alignment(
        _results(r0=(1, 2, 3), r1=(1, 2, 9)),
        {"r0": (1, 2, 3), "r1": (1, 2, 3)}.__getitem__,
        mode="static_batch",
        tie_check=tie_check,
    )

    assert calls == [("r1", 2)]
    assert alignments["r0"] == RequestAlignment((1, 2, 3), None, None, None)
    assert alignments["r1"] == RequestAlignment((1, 2, 3), 2, 0.0, 0.125)


def test_batched_mode_rejects_divergence_that_is_not_a_tie() -> None:
    with pytest.raises(RuntimeError, match=r"fixed_b2/continuous_batch.*r1.*index 1.*gap 0.25"):
        check_token_alignment(
            _results(r1=(1, 9)),
            lambda _: (1, 2),
            mode="continuous_batch",
            tie_check=lambda *_: TieCheck(is_tie=False, logit_gap=0.25, tolerance=0.125),
            context="fixed_b2/continuous_batch",
        )


@pytest.mark.parametrize("mode", ["hf_sequential", "sequential"])
def test_single_request_modes_require_exact_alignment(mode: str) -> None:
    def tie_check(*_: object) -> TieCheck:
        raise AssertionError("B=1 modes must not consult the tie rule")

    with pytest.raises(RuntimeError, match="index 1"):
        check_token_alignment(
            _results(r1=(1, 9)),
            lambda _: (1, 2),
            mode=mode,  # type: ignore[arg-type]
            tie_check=tie_check,
        )


def test_length_divergence_is_never_a_tie() -> None:
    with pytest.raises(RuntimeError, match="length"):
        check_token_alignment(
            _results(r1=(1, 2)),
            lambda _: (1, 2, 3),
            mode="static_batch",
            tie_check=lambda *_: TieCheck(is_tie=True, logit_gap=0.0, tolerance=0.125),
        )


def test_build_request_results_records_divergence() -> None:
    entries = build_request_results(
        _results(r1=(1, 2, 9)),
        {"r1": RequestAlignment((1, 2, 3), 2, 0.0, 0.125)},
    )

    assert entries[0].first_mismatch_index == 2
    assert entries[0].token_agreement == 2 / 3
    assert entries[0].divergence_logit_gap == 0.0
    assert entries[0].divergence_tolerance == 0.125
