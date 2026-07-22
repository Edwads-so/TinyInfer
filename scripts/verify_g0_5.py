#!/usr/bin/env python3
"""G0.5 verification orchestrator -- Design §6.3.

Executes each quality gate in a fixed order and stops at the first
failure.  Prints a unified PASS/FAIL summary.  Does NOT auto-commit,
tag, push, download models, install dependencies, or write secrets.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data structures shared with tests/unit/test_verify_g0_5.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepResult:
    """Single verification step outcome."""

    name: str
    exit_code: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Return true only for a zero exit code without a runner error."""
        return self.exit_code == 0 and self.error is None


@dataclass(frozen=True, slots=True)
class StepSpec:
    """One command in the ordered G0.5 verification gate."""

    name: str
    command: tuple[str, ...]


def build_step_specs(
    python: str = sys.executable,
    uv: str = "uv",
) -> tuple[StepSpec, ...]:
    """Build the fixed verification order for the active interpreter."""
    return (
        StepSpec("lock-check", (uv, "lock", "--check")),
        StepSpec("env-report", (python, "scripts/check_environment.py", "--json")),
        StepSpec("pytest", (python, "-m", "pytest")),
        StepSpec("ruff-check", (python, "-m", "ruff", "check", ".")),
        StepSpec("ruff-format", (python, "-m", "ruff", "format", "--check", ".")),
        StepSpec("mypy", (python, "-m", "mypy")),
        StepSpec("gpu-smoke", (python, "scripts/check_gpu_smoke.py")),
    )


STEP_SPECS = build_step_specs()


def run_steps(results: Sequence[StepResult]) -> int:
    """Walk through results; return 0 only when ALL passed."""
    for r in results:
        if not r.passed:
            return r.exit_code if r.exit_code != 0 else 1
    return 0


def execute_steps(
    specs: Sequence[StepSpec],
    runner: Callable[[StepSpec], StepResult],
) -> list[StepResult]:
    """Execute ordered steps and stop immediately after the first failure."""
    results: list[StepResult] = []
    for spec in specs:
        result = runner(spec)
        results.append(result)
        if not result.passed:
            break
    return results


# ---------------------------------------------------------------------------
# Real execution helpers
# ---------------------------------------------------------------------------


def _run_cmd(spec: StepSpec) -> StepResult:
    """Run a subprocess and wrap the outcome."""
    print(f"\n=== {spec.name} ===  {' '.join(spec.command)}")
    try:
        proc = subprocess.run(spec.command, check=False, capture_output=False)
    except FileNotFoundError as exc:
        return StepResult(name=spec.name, exit_code=127, error=str(exc))
    if proc.returncode == 0:
        print(f"--- {spec.name}: PASS ---")
        return StepResult(name=spec.name)
    return StepResult(
        name=spec.name,
        exit_code=proc.returncode,
        error=f"exit code {proc.returncode}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full G0.5 verification gate."""
    results = execute_steps(STEP_SPECS, _run_cmd)

    # --- Summary ---
    print()
    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"  [{marker}] {r.name}")
    print()
    exit_code = run_steps(results)
    if exit_code == 0 and len(results) == len(STEP_SPECS):
        print("G0.5 VERIFICATION: PASS")
        return 0
    print("G0.5 VERIFICATION: FAIL")
    return exit_code if exit_code != 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
