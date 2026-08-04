from __future__ import annotations

import subprocess
import sys

import tinyinfer


def test_package_exposes_version() -> None:
    assert tinyinfer.__version__ == "0.2.0"


def test_module_entrypoint_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tinyinfer"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "TinyInfer 0.2.0" in result.stdout
