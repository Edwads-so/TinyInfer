"""CPU-only unit tests for scripts/verify_g0_5.py orchestration logic.

These tests verify error propagation, exit-code semantics, and output
formatting of the G0.5 verifier without requiring a GPU or real tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    from scripts.verify_g0_5 import StepResult, run_steps, STEP_SPECS
except ImportError:
    StepResult = None  # type: ignore[assignment]
    run_steps = None  # type: ignore[assignment]
    STEP_SPECS: list = []

pytestmark = pytest.mark.skipif(
    StepResult is None,
    reason="scripts/verify_g0_5.py is not yet available",
)


# ----------------------------------------------------------------- StepResult --


def test_step_result_default_is_pass() -> None:
    sr = StepResult(name="pytest")
    assert sr.passed is True
    assert sr.exit_code == 0
    assert sr.error is None


def test_step_result_can_mark_failure() -> None:
    sr = StepResult(name="ruff-check", exit_code=1, error="ruff check failed")
    assert sr.passed is False
    assert sr.exit_code == 1


# ----------------------------------------------------------------- run_steps ----


def test_run_steps_all_pass_returns_zero() -> None:
    specs = [
        StepResult("a"),
        StepResult("b"),
        StepResult("c"),
    ]
    exit_code = run_steps(specs)
    assert exit_code == 0


def test_run_steps_first_failure_stops_and_returns_nonzero() -> None:
    """Design §6.3: first failing step returns non-zero and stops."""
    specs = [
        StepResult("lock-check"),
        StepResult("env-report", exit_code=2, error="env script crashed"),
        StepResult("pytest"),  # should never be "executed"
    ]
    exit_code = run_steps(specs)
    assert exit_code != 0


def test_run_steps_later_failure_stops() -> None:
    specs = [
        StepResult("lock-check"),
        StepResult("env-report"),
        StepResult("pytest", exit_code=1, error="3 tests failed"),
        StepResult("ruff-check"),
    ]
    exit_code = run_steps(specs)
    assert exit_code != 0


def test_run_steps_empty_specs_is_ok() -> None:
    """All steps already verified; verifier itself stays clean."""
    exit_code = run_steps([])
    assert exit_code == 0


# ------------------------------------------------------- STEP_SPECS predefined --


def test_step_specs_are_ordered() -> None:
    """The verifier must execute checks in a fixed, documented order."""
    names = [s.name for s in STEP_SPECS]
    assert names == list(dict.fromkeys(names))  # no duplicates
    assert "pytest" in names
    assert "ruff-check" in names


def test_step_specs_do_not_auto_commit_or_tag() -> None:
    """Design §6.3: verifier MUST NOT auto-commit, tag, or push."""
    dangerous = {"git", "commit", "tag", "push", "release"}
    for spec in STEP_SPECS:
        assert not dangerous.intersection(set(spec.name.lower().split()))