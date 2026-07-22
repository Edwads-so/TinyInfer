#!/usr/bin/env python3
"""G0.5 verification orchestrator -- Design §6.3.

Executes each quality gate in a fixed order and stops at the first
failure.  Prints a unified PASS/FAIL summary.  Does NOT auto-commit,
tag, push, download models, install dependencies, or write secrets.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
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


# Predefined step specifications.
# Each entry is (short_name, command_list, needs_gpu, description).
STEP_SPECS: list[StepResult] = [
    StepResult(name="lock-check"),
    StepResult(name="env-report"),
    StepResult(name="pytest"),
    StepResult(name="ruff-check"),
    StepResult(name="ruff-format"),
    StepResult(name="mypy"),
    StepResult(name="gpu-smoke"),
]


def run_steps(results: Sequence[StepResult]) -> int:
    """Walk through results; return 0 only when ALL passed."""
    for r in results:
        if not r.passed:
            return r.exit_code if r.exit_code != 0 else 1
    return 0


# ---------------------------------------------------------------------------
# Real execution helpers
# ---------------------------------------------------------------------------


def _run_cmd(name: str, cmd: list[str]) -> StepResult:
    """Run a subprocess and wrap the outcome."""
    print(f"\n=== {name} ===  {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=False, capture_output=False)
    except FileNotFoundError as exc:
        return StepResult(name=name, exit_code=127, error=str(exc))
    if proc.returncode == 0:
        print(f"--- {name}: PASS ---")
        return StepResult(name=name)
    return StepResult(
        name=name,
        exit_code=proc.returncode,
        error=f"exit code {proc.returncode}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full G0.5 verification gate."""
    results: list[StepResult] = []

    # 1. Lock file consistency
    results.append(_run_cmd("lock-check", ["uv", "lock", "--check"]))

    # 2. Environment report
    results.append(
        _run_cmd("env-report", [sys.executable, "scripts/check_environment.py", "--json"])
    )

    # 3. pytest
    results.append(_run_cmd("pytest", [sys.executable, "-m", "pytest"]))

    # 4. Ruff check
    results.append(_run_cmd("ruff-check", [sys.executable, "-m", "ruff", "check", "."]))

    # 5. Ruff format check
    results.append(
        _run_cmd("ruff-format", [sys.executable, "-m", "ruff", "format", "--check", "."])
    )

    # 6. Mypy
    results.append(_run_cmd("mypy", [sys.executable, "-m", "mypy"]))

    # 7. GPU smoke test
    results.append(
        _run_cmd("gpu-smoke", [sys.executable, "scripts/check_gpu_smoke.py"])
    )

    # --- Summary ---
    print()
    all_ok = True
    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"  [{marker}] {r.name}")
        if not r.passed:
            all_ok = False
    print()
    if all_ok:
        print("G0.5 VERIFICATION: PASS")
        return 0
    print("G0.5 VERIFICATION: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
