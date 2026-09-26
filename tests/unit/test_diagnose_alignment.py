import pytest
import torch

from benchmarks.bench_phase2 import assert_token_alignment
from benchmarks.diagnose_alignment import divergence_at, first_mismatch_index


def test_first_mismatch_index_finds_first_differing_position() -> None:
    assert first_mismatch_index((1, 2, 3), (1, 2, 3)) is None
    assert first_mismatch_index((1, 2, 3), (1, 9, 3)) == 1
    assert first_mismatch_index((1, 2), (1, 2, 3)) == 2


def test_divergence_reports_gap_between_reference_and_actual_token() -> None:
    logits = torch.tensor([0.0, 5.0, 4.9, 1.0])

    divergence = divergence_at(logits, reference_token=1, actual_token=2)

    assert divergence.reference_token == 1
    assert divergence.actual_token == 2
    assert divergence.actual_rank == 2
    assert divergence.logit_gap == pytest.approx(0.1)
    assert divergence.top_token_ids == (1, 2, 3, 0)


def test_divergence_rejects_reference_that_is_not_the_argmax() -> None:
    with pytest.raises(ValueError, match="argmax"):
        divergence_at(torch.tensor([3.0, 1.0]), reference_token=1, actual_token=0)


def test_alignment_error_names_workload_mode_and_position() -> None:
    results = (type("R", (), {"request_id": "r1", "output_token_ids": (1, 2, 7)})(),)

    with pytest.raises(RuntimeError, match=r"fixed_b2/static_batch.*r1.*index 2"):
        assert_token_alignment(results, lambda _: (1, 2, 3), context="fixed_b2/static_batch")
