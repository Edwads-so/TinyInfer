#!/usr/bin/env python3
"""G2-BD local verification gate.

Fixed order: lock-check, pytest-unit, pytest-gpu-batch-decode,
pytest-gpu-integration, ruff-check, ruff-format, mypy, git-diff-check.

Fails fast, prints command/exit code, installs/downloads nothing, and
mutates no Git state.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PY = sys.executable

# `uv` is not guaranteed on PATH in non-login shells; resolve via which() and
# fall back to the well-known WSL install location.
UV = shutil.which("uv") or "/home/administrator/.local/bin/uv"


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
            "pytest-gpu-batch-decode",
            (
                python,
                "-m",
                "pytest",
                "tests/integration/test_batch_decode.py",
                "-m",
                "gpu and model",
                "-q",
            ),
            900,
        ),
        StepSpec(
            "pytest-gpu-integration",
            (
                python,
                "-m",
                "pytest",
                "tests/integration",
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


STEP_SPECS = build_step_specs()


def execute_steps(
    specs: Sequence[StepSpec],
    runner: Callable[[StepSpec], StepResult],
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
        proc = subprocess.run(
            spec.command,
            capture_output=True,
            text=True,
            timeout=spec.timeout,
        )
    except subprocess.TimeoutExpired:
        return StepResult(name=spec.name, error=f"{spec.name} timed out")
    except FileNotFoundError as exc:
        return StepResult(name=spec.name, error=str(exc))
    if proc.returncode == 0:
        print(f"[PASS] {spec.name}")
        return StepResult(name=spec.name)
    return StepResult(
        name=spec.name,
        exit_code=proc.returncode,
        error=(proc.stderr or proc.stdout)[-500:],
    )


def main() -> int:
    results = execute_steps(STEP_SPECS, run_cmd)
    print()
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}")
    print()
    if all(r.passed for r in results) and len(results) == len(STEP_SPECS):
        print("G2-BD VERIFICATION: PASS")
        return 0
    print("G2-BD VERIFICATION: FAIL")
    return results[-1].exit_code if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
