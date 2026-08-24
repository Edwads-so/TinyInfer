"""Verifier behavior tests for scripts/verify_g2_bd.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import verify_g2_bd as verifier


def _ok(name: str) -> Any:
    return verifier.StepResult(name=name)


def _fail(name: str, code: int = 1) -> Any:
    return verifier.StepResult(name=name, exit_code=code, error="boom")


class TestStepOrder:
    def test_exact_order(self) -> None:
        specs = verifier.build_step_specs(python="python")
        assert [s.name for s in specs] == [
            "lock-check",
            "pytest-unit",
            "pytest-gpu-batch-decode",
            "pytest-gpu-integration",
            "ruff-check",
            "ruff-format",
            "mypy",
            "git-diff-check",
        ]

    def test_timeouts_assigned(self) -> None:
        for spec in verifier.STEP_SPECS:
            assert spec.timeout > 0


class TestFailFast:
    def test_stops_on_first_failure(self) -> None:
        results = verifier.execute_steps(
            [verifier.StepSpec("a", (), 1), verifier.StepSpec("b", (), 1)],
            lambda s: _fail(s.name),
        )
        assert len(results) == 1
        assert results[0].name == "a"


class TestExit:
    def test_all_pass_returns_zero(self) -> None:
        results = verifier.execute_steps(
            verifier.STEP_SPECS,
            lambda s: _ok(s.name),
        )
        assert all(r.passed for r in results)
        assert len(results) == len(verifier.STEP_SPECS)


class TestNoMutation:
    def test_source_has_no_forbidden_commands(self) -> None:
        text = (Path(__file__).resolve().parents[2] / "scripts" / "verify_g2_bd.py").read_text(
            encoding="utf-8"
        )
        assert "pip install" not in text
        assert "uv sync" not in text
        assert "git add" not in text
        assert "git commit" not in text
        assert "git push" not in text
        assert "/mnt/d/" not in text
