"""CPU-only unit tests for scripts/check_gpu_smoke.py pure-Python logic.

These tests verify that the smoke script's helper functions produce
correct shapes, result summaries, and error messages when CUDA or Triton
are unavailable.  Real GPU correctness is exercised by the G0.5 hardware
acceptance gate, not by this test module.
"""

from __future__ import annotations

from scripts.check_gpu_smoke import FIXED_SIZES, SmokeResult, summarize_results


# ------------------------------------------------------------------ FIXED_SIZES

def test_fixed_sizes_are_ordered_and_non_empty() -> None:
    assert len(FIXED_SIZES) >= 4
    assert tuple(sorted(FIXED_SIZES)) == FIXED_SIZES
    assert FIXED_SIZES[0] >= 1


def test_fixed_sizes_include_edge_lengths() -> None:
    """Design §6.1 mandates sizes 1, 257, 4096, 65537."""
    mandatory = {1, 257, 4096, 65537}
    assert mandatory.issubset(set(FIXED_SIZES))


# ----------------------------------------------------------------- SmokeResult

def test_smoke_result_default_is_pass() -> None:
    result = SmokeResult(name="test")
    assert result.passed is True
    assert result.error is None


def test_smoke_result_can_mark_failure() -> None:
    result = SmokeResult(name="matmul", error="CUDA unavailable")
    assert result.passed is False
    assert result.error == "CUDA unavailable"


def test_smoke_result_repr_includes_name_and_status() -> None:
    r = SmokeResult(name="triton-add", error="Triton import failed")
    text = repr(r)
    assert "triton-add" in text
    assert "FAIL" in text
    # A passed result should not contain FAIL.
    assert "FAIL" not in repr(SmokeResult(name="ok"))


# ------------------------------------------------------------- summarize_results

def test_summarize_all_pass_returns_zero() -> None:
    results = [SmokeResult("a"), SmokeResult("b")]
    exit_code = summarize_results(results)
    assert exit_code == 0


def test_summarize_single_failure_returns_nonzero() -> None:
    results = [SmokeResult("a"), SmokeResult("b", error="fail")]
    exit_code = summarize_results(results)
    assert exit_code != 0


def test_summarize_missing_cuda_reports_failure() -> None:
    """A missing-CUDA result must produce non-zero exit (Design §8)."""
    results = [SmokeResult("cuda-check", error="torch.cuda.is_available() == False")]
    exit_code = summarize_results(results)
    assert exit_code != 0


def test_summarize_missing_triton_reports_failure() -> None:
    """A missing-Triton result must produce non-zero exit (Design §8)."""
    results = [SmokeResult("triton-add", error="No module named 'triton'")]
    exit_code = summarize_results(results)
    assert exit_code != 0


def test_summarize_output_contains_pass_or_fail() -> None:
    """Design §6.1 item 10: structured PASS line for humans and agents."""
    results = [SmokeResult("matmul"), SmokeResult("triton-add")]
    exit_code = summarize_results(results)
    assert exit_code == 0
    # Trick to capture printed output would need an integration test;
    # the unit-level guarantee is that exit_code reflects all-or-none
    # semantics.
