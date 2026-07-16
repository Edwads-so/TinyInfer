from __future__ import annotations

import json

import pytest

from tinyinfer.environment import (
    EnvironmentReport,
    PackageStatus,
    inspect_environment,
    main,
    render_human_report,
)


def test_inspection_reports_missing_optional_package() -> None:
    report = inspect_environment(("a_package_that_does_not_exist_tinyinfer",))
    assert len(report.packages) == 1
    assert report.packages[0].installed is False
    assert report.cuda_available is False


def test_human_report_contains_capabilities() -> None:
    report = EnvironmentReport(
        python_version="3.11.7",
        platform="test-platform",
        packages=(PackageStatus("torch", False, None),),
        cuda_available=False,
        gpu_name=None,
        bf16_supported=False,
        diagnostics=("PyTorch is not installed; CUDA was not inspected.",),
    )
    rendered = render_human_report(report)
    assert "Python: 3.11.7" in rendered
    assert "torch: not installed" in rendered
    assert "CUDA available: no" in rendered


def test_json_cli_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--json", "--packages", "a_package_that_does_not_exist_tinyinfer"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["packages"][0]["installed"] is False
