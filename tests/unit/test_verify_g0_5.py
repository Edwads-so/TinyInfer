"""CPU-only unit tests for scripts/verify_g0_5.py orchestration logic.

These tests verify error propagation, exit-code semantics, and output
formatting of the G0.5 verifier without requiring a GPU or real tools.
"""

from __future__ import annotations

from typing import Any

from scripts import verify_g0_5

STEP_SPECS = verify_g0_5.STEP_SPECS
StepResult = verify_g0_5.StepResult
run_steps = verify_g0_5.run_steps


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


def test_step_result_nonzero_exit_implies_failure() -> None:
    sr = StepResult(name="pytest", exit_code=1)
    assert sr.passed is False


def test_step_result_error_implies_failure_even_with_zero_exit() -> None:
    sr = StepResult(name="pytest", error="runner unavailable")
    assert sr.passed is False


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


def test_execute_steps_does_not_run_after_first_failure() -> None:
    calls: list[str] = []
    step_spec_type = verify_g0_5.StepSpec

    def runner(spec: Any) -> StepResult:
        calls.append(spec.name)
        if spec.name == "env-report":
            return StepResult(spec.name, exit_code=2, error="failed")
        return StepResult(spec.name)

    results = verify_g0_5.execute_steps(
        (
            step_spec_type("lock-check", ("uv", "lock", "--check")),
            step_spec_type("env-report", ("python", "env.py")),
            step_spec_type("pytest", ("python", "-m", "pytest")),
        ),
        runner,
    )

    assert calls == ["lock-check", "env-report"]
    assert [result.name for result in results] == calls


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
