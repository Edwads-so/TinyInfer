#!/usr/bin/env python3
"""G1 reference-runtime verification orchestrator.

Executes CPU unit tests, GPU integration tests, and canonical JSONL
validation in a fixed order.  Stops at the first failure.  Prints a
unified PASS/FAIL summary.  Does NOT download models, install
dependencies, or write canonical evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_JSONL = ROOT / "docs" / "results" / "g1_reference.jsonl"
EXPECTED_CASE_IDS = ("short_en", "short_zh", "structured", "long_context")
EXPECTED_RECORDS = len(EXPECTED_CASE_IDS) * 5  # 5 repetitions per case

# ---------------------------------------------------------------------------
# Data structures
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
    """One command in the ordered G1 verification gate."""

    name: str
    command: tuple[str, ...]


def build_step_specs(python: str = sys.executable) -> tuple[StepSpec, ...]:
    """Build the fixed verification order for the active interpreter."""
    return (
        StepSpec("pytest-cpu", (python, "-m", "pytest", "-m", "not gpu and not model and not slow")),
        StepSpec(
            "pytest-gpu",
            (python, "-m", "pytest", "tests/integration", "-m", "gpu and model", "-v"),
        ),
        StepSpec("validate-canonical", ()),  # handled by _validate_canonical_jsonl
    )


STEP_SPECS = build_step_specs()


# ---------------------------------------------------------------------------
# Canonical JSONL validation (pure Python, no subprocess)
# ---------------------------------------------------------------------------


def _validate_canonical_jsonl() -> StepResult:
    """Inspect the canonical G1 JSONL file without running the benchmark."""
    if not CANONICAL_JSONL.exists():
        return StepResult(
            "validate-canonical",
            exit_code=1,
            error=f"canonical JSONL not found: {CANONICAL_JSONL}",
        )
    try:
        lines = CANONICAL_JSONL.read_text(encoding="utf-8").strip().split("\n")
    except OSError as exc:
        return StepResult("validate-canonical", exit_code=1, error=str(exc))
    if len(lines) != EXPECTED_RECORDS:
        return StepResult(
            "validate-canonical",
            exit_code=1,
            error=f"expected {EXPECTED_RECORDS} records, got {len(lines)}",
        )
    case_counts: dict[str, int] = {}
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            return StepResult(
                "validate-canonical",
                exit_code=1,
                error=f"line {index + 1} is invalid JSON: {exc}",
            )
        if record.get("schema_version") != 1:
            return StepResult(
                "validate-canonical",
                exit_code=1,
                error=f"line {index + 1} schema_version must be 1",
            )
        if record.get("token_agreement") != 1.0:
            return StepResult(
                "validate-canonical",
                exit_code=1,
                error=f"line {index + 1} token_agreement must be 1.0",
            )
        case_id = record.get("case_id")
        if case_id not in EXPECTED_CASE_IDS:
            return StepResult(
                "validate-canonical",
                exit_code=1,
                error=f"line {index + 1} has unknown case_id: {case_id!r}",
            )
        case_counts[case_id] = case_counts.get(case_id, 0) + 1
    for case_id in EXPECTED_CASE_IDS:
        if case_counts.get(case_id, 0) != 5:
            return StepResult(
                "validate-canonical",
                exit_code=1,
                error=f"{case_id} appears {case_counts.get(case_id, 0)} times, expected 5",
            )
    return StepResult("validate-canonical")


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _run_cmd(spec: StepSpec) -> StepResult:
    """Run a subprocess and wrap the outcome."""
    print(f"\n=== {spec.name} ===  {' '.join(spec.command)}")
    try:
        proc = subprocess.run(spec.command, check=False)
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


def execute_steps(
    specs: Sequence[StepSpec],
    runner: Callable[[StepSpec], StepResult],
) -> list[StepResult]:
    """Execute ordered steps and stop immediately after the first failure."""
    results: list[StepResult] = []
    for spec in specs:
        if spec.name == "validate-canonical":
            result = _validate_canonical_jsonl()
        else:
            result = runner(spec)
        results.append(result)
        if not result.passed:
            break
    return results


def run_steps(results: Sequence[StepResult]) -> int:
    """Walk through results; return 0 only when ALL passed."""
    for r in results:
        if not r.passed:
            return r.exit_code if r.exit_code != 0 else 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full G1 verification gate."""
    results = execute_steps(STEP_SPECS, _run_cmd)

    print()
    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"  [{marker}] {r.name}")
    print()
    exit_code = run_steps(results)
    if exit_code == 0 and len(results) == len(STEP_SPECS):
        print("G1 VERIFICATION: PASS")
        return 0
    print("G1 VERIFICATION: FAIL")
    return exit_code if exit_code != 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())