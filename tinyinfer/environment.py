"""Structured, dependency-tolerant environment readiness inspection."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import platform as platform_module
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

OPTIONAL_PACKAGES = ("torch", "transformers", "safetensors", "triton")


@dataclass(frozen=True, slots=True)
class PackageStatus:
    """Installation state for one optional package."""

    name: str
    installed: bool
    version: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Machine-readable TinyInfer capability report."""

    python_version: str
    platform: str
    packages: tuple[PackageStatus, ...]
    cuda_available: bool
    gpu_name: str | None
    bf16_supported: bool
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to JSON-serializable primitives."""
        return asdict(self)


def _inspect_package(name: str) -> PackageStatus:
    try:
        if importlib.util.find_spec(name) is None:
            return PackageStatus(name=name, installed=False, version=None)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return PackageStatus(name=name, installed=True, version=version)
    except (ImportError, ValueError) as exc:
        return PackageStatus(name=name, installed=False, version=None, error=str(exc))


def inspect_environment(
    package_names: tuple[str, ...] = OPTIONAL_PACKAGES,
) -> EnvironmentReport:
    """Inspect Python packages and CUDA without requiring optional dependencies."""
    packages = tuple(_inspect_package(name) for name in package_names)
    diagnostics: list[str] = []
    cuda_available = False
    gpu_name: str | None = None
    bf16_supported = False
    torch_status = next((item for item in packages if item.name == "torch"), None)

    if torch_status is None or not torch_status.installed:
        diagnostics.append("PyTorch is not installed; CUDA was not inspected.")
    else:
        try:
            torch = importlib.import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_name = str(torch.cuda.get_device_name(0))
                bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception as exc:  # Third-party runtime failures become diagnostics.
            diagnostics.append(f"PyTorch CUDA inspection failed: {exc}")

    return EnvironmentReport(
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        platform=platform_module.platform(),
        packages=packages,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        bf16_supported=bf16_supported,
        diagnostics=tuple(diagnostics),
    )


def render_human_report(report: EnvironmentReport) -> str:
    """Render a concise report for developers."""
    lines = [f"Python: {report.python_version}", f"Platform: {report.platform}", "Packages:"]
    for package in report.packages:
        state = package.version if package.installed else "not installed"
        lines.append(f"  {package.name}: {state}")
    lines.extend(
        (
            f"CUDA available: {'yes' if report.cuda_available else 'no'}",
            f"GPU: {report.gpu_name or 'unavailable'}",
            f"BF16 supported: {'yes' if report.bf16_supported else 'no'}",
        )
    )
    lines.extend(f"Diagnostic: {message}" for message in report.diagnostics)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the environment inspection CLI."""
    parser = argparse.ArgumentParser(description="Inspect TinyInfer environment readiness.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--packages", nargs="+", default=list(OPTIONAL_PACKAGES))
    args = parser.parse_args(argv)
    report = inspect_environment(tuple(args.packages))
    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_human_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
