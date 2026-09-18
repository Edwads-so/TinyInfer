#!/usr/bin/env python3
"""Fail-fast local verification gate for G2-P1 evidence tooling."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

PY = sys.executable
UV = shutil.which("uv") or "uv"


@dataclass(frozen=True, slots=True)
class StepResult:
    name: str
    exit_code: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and self.error is None


@dataclass(frozen=True, slots=True)
class StepSpec:
    name: str
    command: tuple[str, ...]
    timeout: int


def build_step_specs(python: str = PY) -> tuple[StepSpec, ...]:
    return (
        StepSpec("lock-check", (UV, "lock", "--check"), 120),
        StepSpec("pytest-unit", (python, "-m", "pytest", "tests/unit", "-q"), 900),
        StepSpec(
            "pytest-gpu-phase2",
            (
                python,
                "-m",
                "pytest",
                "tests/integration/test_phase2_benchmark.py",
                "-m",
                "gpu and model",
                "-q",
            ),
            900,
        ),
        StepSpec("ruff-check", (python, "-m", "ruff", "check", "."), 120),
        StepSpec("ruff-format", (python, "-m", "ruff", "format", "--check", "."), 120),
        StepSpec("mypy", (python, "-m", "mypy"), 120),
        StepSpec("git-diff-check", ("git", "diff", "--check"), 120),
    )


def execute_steps(
    specs: Sequence[StepSpec], runner: Callable[[StepSpec], StepResult]
) -> list[StepResult]:
    results: list[StepResult] = []
    for spec in specs:
        result = runner(spec)
        results.append(result)
        if not result.passed:
            break
    return results


def run_cmd(spec: StepSpec) -> StepResult:
    print(f"=== {spec.name} ===  {' '.join(spec.command)}")
    try:
        result = subprocess.run(spec.command, capture_output=True, text=True, timeout=spec.timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return StepResult(spec.name, error=str(error))
    if result.returncode == 0:
        return StepResult(spec.name)
    return StepResult(
        spec.name, exit_code=result.returncode, error=(result.stderr or result.stdout)[-500:]
    )


def main() -> int:
    specs = build_step_specs()
    results = execute_steps(specs, run_cmd)
    passed = len(results) == len(specs) and all(result.passed for result in results)
    print(f"G2-P1 VERIFICATION: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else results[-1].exit_code


if __name__ == "__main__":
    raise SystemExit(main())
