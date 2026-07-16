# TinyInfer Runnable Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable, executable, documented, and CPU-testable Python foundation for TinyInfer without downloading models or requiring CUDA.

**Architecture:** The `tinyinfer` package exposes a small immutable engine configuration and a dependency-tolerant environment inspector. Two thin command-line entry points render the same structured environment report. Packaging, tests, CI, and documentation establish a reproducible boundary for the later Hugging Face runtime.

**Tech Stack:** Python 3.11, standard-library dataclasses/argparse/importlib, pytest, Ruff, mypy, GitHub Actions.

## Global Constraints

- Support Python `>=3.11,<3.12`.
- The base package must import and run without PyTorch, CUDA, Transformers, Safetensors, or Triton.
- Do not download models or install GPU dependencies during scaffold initialization.
- CI must run on CPU-only Ubuntu and Python 3.11.
- The initial runtime target is `Qwen/Qwen2.5-0.5B`, BF16, single GPU, greedy decoding, but inference is outside this plan.
- Use the MIT License and the `main` default branch.

---

## File Structure

- `pyproject.toml`: package metadata and tool configuration.
- `tinyinfer/__init__.py`: public version constant.
- `tinyinfer/__main__.py`: package CLI.
- `tinyinfer/config.py`: immutable `EngineConfig` and validation.
- `tinyinfer/environment.py`: structured optional dependency and CUDA inspection.
- `scripts/check_environment.py`: standalone wrapper around the environment module.
- `tests/unit/test_package.py`: import and package CLI tests.
- `tests/unit/test_config.py`: configuration validation tests.
- `tests/unit/test_environment.py`: environment report and rendering tests.
- `configs/qwen2_5_0_5b.yaml`: documented future runtime defaults; no loader yet.
- `.github/workflows/ci.yml`: CPU-only quality checks.
- `.gitignore`: Python, editor, model, and benchmark artifact exclusions.
- `README.md`: truthful setup, commands, architecture boundary, and reduced roadmap.
- `docs/PROJECT_PLAN.md`: reduced application-oriented project plan.
- `LICENSE`: MIT License.

### Task 1: Package Metadata and Executable Package

**Files:**
- Create: `pyproject.toml`
- Create: `tinyinfer/__init__.py`
- Create: `tinyinfer/__main__.py`
- Create: `tests/unit/test_package.py`

**Interfaces:**
- Produces: `tinyinfer.__version__: str`
- Produces: `tinyinfer.__main__.main(argv: Sequence[str] | None = None) -> int`
- Consumes later: `tinyinfer.environment.inspect_environment` and `render_human_report`

- [ ] **Step 1: Write the failing package tests**

```python
from __future__ import annotations

import subprocess
import sys

import tinyinfer


def test_package_exposes_version() -> None:
    assert tinyinfer.__version__ == "0.1.0"


def test_module_entrypoint_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tinyinfer"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "TinyInfer 0.1.0" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_package.py -v`

Expected: collection fails because `tinyinfer` does not exist.

- [ ] **Step 3: Add package metadata and the smallest package implementation**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "tinyinfer"
version = "0.1.0"
description = "A compact experimental runtime for studying LLM KV-cache and scheduling behavior."
readme = "README.md"
requires-python = ">=3.11,<3.12"
license = { text = "MIT" }
authors = [{ name = "Edwads-so" }]
dependencies = []

[project.optional-dependencies]
dev = [
  "mypy>=1.14,<2",
  "pytest>=8.3,<9",
  "pytest-cov>=6,<7",
  "ruff>=0.9,<1",
]

[tool.setuptools.packages.find]
include = ["tinyinfer*"]

[tool.pytest.ini_options]
addopts = "-ra --strict-config --strict-markers"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
files = ["tinyinfer", "scripts"]
```

`tinyinfer/__init__.py`:

```python
"""TinyInfer public package interface."""

__version__ = "0.1.0"
```

`tinyinfer/__main__.py` initially:

```python
from __future__ import annotations

from collections.abc import Sequence

from tinyinfer import __version__


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(f"TinyInfer {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_package.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit the package foundation**

```bash
git add pyproject.toml tinyinfer/__init__.py tinyinfer/__main__.py tests/unit/test_package.py
git commit -m "Initialize TinyInfer Python package"
```

### Task 2: Immutable Engine Configuration

**Files:**
- Create: `tinyinfer/config.py`
- Create: `tests/unit/test_config.py`
- Create: `configs/qwen2_5_0_5b.yaml`

**Interfaces:**
- Produces: `EngineConfig(model_id: str, dtype: str, device: str, max_sequence_length: int, kv_block_size: int)`
- Produces: `SUPPORTED_DTYPES: frozenset[str]`

- [ ] **Step 1: Write failing validation tests**

```python
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tinyinfer.config import EngineConfig


def test_default_config_targets_small_qwen() -> None:
    config = EngineConfig()
    assert config.model_id == "Qwen/Qwen2.5-0.5B"
    assert config.dtype == "bfloat16"
    assert config.device == "cuda"
    assert config.max_sequence_length == 4096
    assert config.kv_block_size == 16


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "", "model_id must not be empty"),
        ("dtype", "int8", "unsupported dtype"),
        ("device", "mps", "device must be 'cpu' or a CUDA device"),
        ("max_sequence_length", 0, "max_sequence_length must be positive"),
        ("kv_block_size", 0, "kv_block_size must be positive"),
        ("kv_block_size", 8192, "kv_block_size cannot exceed max_sequence_length"),
    ],
)
def test_invalid_config_is_rejected(field: str, value: object, message: str) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        EngineConfig(**kwargs)  # type: ignore[arg-type]


def test_config_is_immutable() -> None:
    config = EngineConfig()
    with pytest.raises(FrozenInstanceError):
        config.dtype = "float32"  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_config.py -v`

Expected: collection fails because `tinyinfer.config` does not exist.

- [ ] **Step 3: Implement configuration validation**

```python
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_DTYPES = frozenset({"float32", "float16", "bfloat16"})


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_sequence_length: int = 4096
    kv_block_size: int = 16

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.kv_block_size <= 0:
            raise ValueError("kv_block_size must be positive")
        if self.kv_block_size > self.max_sequence_length:
            raise ValueError("kv_block_size cannot exceed max_sequence_length")
```

`configs/qwen2_5_0_5b.yaml`:

```yaml
model_id: Qwen/Qwen2.5-0.5B
dtype: bfloat16
device: cuda
max_sequence_length: 4096
kv_block_size: 16
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_config.py -v`

Expected: 8 passed.

- [ ] **Step 5: Commit configuration**

```bash
git add tinyinfer/config.py tests/unit/test_config.py configs/qwen2_5_0_5b.yaml
git commit -m "Add validated engine configuration"
```

### Task 3: Structured Environment Inspection

**Files:**
- Create: `tinyinfer/environment.py`
- Modify: `tinyinfer/__main__.py`
- Create: `scripts/check_environment.py`
- Create: `tests/unit/test_environment.py`
- Modify: `tests/unit/test_package.py`

**Interfaces:**
- Produces: `PackageStatus(name: str, installed: bool, version: str | None, error: str | None)`
- Produces: `EnvironmentReport(python_version: str, platform: str, packages: tuple[PackageStatus, ...], cuda_available: bool, gpu_name: str | None, bf16_supported: bool, diagnostics: tuple[str, ...])`
- Produces: `inspect_environment(package_names: tuple[str, ...] = OPTIONAL_PACKAGES) -> EnvironmentReport`
- Produces: `render_human_report(report: EnvironmentReport) -> str`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing environment tests**

```python
from __future__ import annotations

import json

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
        packages=(PackageStatus("torch", False, None, None),),
        cuda_available=False,
        gpu_name=None,
        bf16_supported=False,
        diagnostics=("PyTorch is not installed; CUDA was not inspected.",),
    )
    rendered = render_human_report(report)
    assert "Python: 3.11.7" in rendered
    assert "torch: not installed" in rendered
    assert "CUDA available: no" in rendered


def test_json_cli_is_machine_readable(capsys: object) -> None:
    exit_code = main(["--json", "--packages", "a_package_that_does_not_exist_tinyinfer"])
    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload["packages"][0]["installed"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_environment.py -v`

Expected: collection fails because `tinyinfer.environment` does not exist.

- [ ] **Step 3: Implement dependency-tolerant inspection and CLI rendering**

`tinyinfer/environment.py`:

```python
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
    name: str
    installed: bool
    version: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    python_version: str
    platform: str
    packages: tuple[PackageStatus, ...]
    cuda_available: bool
    gpu_name: str | None
    bf16_supported: bool
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
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
        except Exception as exc:  # third-party runtime errors become diagnostics
            diagnostics.append(f"PyTorch CUDA inspection failed: {exc}")

    return EnvironmentReport(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform_module.platform(),
        packages=packages,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        bf16_supported=bf16_supported,
        diagnostics=tuple(diagnostics),
    )


def render_human_report(report: EnvironmentReport) -> str:
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
```

Replace `tinyinfer/__main__.py` with:

```python
from __future__ import annotations

from collections.abc import Sequence

from tinyinfer import __version__
from tinyinfer.environment import inspect_environment, render_human_report


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(f"TinyInfer {__version__}")
    print(render_human_report(inspect_environment()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`scripts/check_environment.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinyinfer.environment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/unit/test_environment.py tests/unit/test_package.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit environment diagnostics**

```bash
git add tinyinfer/environment.py tinyinfer/__main__.py scripts/check_environment.py tests/unit/test_environment.py tests/unit/test_package.py
git commit -m "Add environment readiness diagnostics"
```

### Task 4: Documentation, CI, and Reduced Project Scope

**Files:**
- Create: `.gitignore`
- Create: `.gitattributes`
- Create: `.github/workflows/ci.yml`
- Create: `LICENSE`
- Create: `README.md`
- Move and rewrite: `PROJECT_PLAN.md` to `docs/PROJECT_PLAN.md`

**Interfaces:**
- Documents commands implemented by Tasks 1-3.
- CI consumes the `dev` optional dependency group from `pyproject.toml`.

- [ ] **Step 1: Add repository hygiene and CI**

`.gitignore`:

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
build/
dist/
.venv/
.idea/
.vscode/
*.safetensors
*.bin
*.pt
*.pth
models/
benchmarks/results/
reports/figures/generated/
```

`.gitattributes`:

```gitattributes
* text=auto eol=lf
*.bat text eol=crlf
```

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: python -m pytest
      - run: python -m ruff check .
      - run: python -m ruff format --check .
      - run: python -m mypy
```

- [ ] **Step 2: Write the README and reduced project plan**

Create `README.md` with this content:

````markdown
# TinyInfer

TinyInfer is a compact experimental runtime for studying how KV-cache layout and iteration-level scheduling affect LLM inference latency, throughput, and GPU memory efficiency.

## Status

The repository currently provides the engineering scaffold: packaging, validated configuration, environment diagnostics, tests, linting, typing, CI, and documentation. Model inference, continuous batching, paged KV cache, and Triton kernels are planned and are not implemented yet.

## Architecture Boundary

Hugging Face will provide Qwen2.5 model structure, tokenizer, weights, and the correctness baseline. TinyInfer will own request state, the explicit prefill/decode loop, scheduling, profiling, cache allocation, block tables, and the Qwen attention cache adapter.

## Requirements

- Python 3.11
- Linux or WSL2 for later CUDA/Triton work
- NVIDIA GPU with BF16 support for inference phases

The scaffold and CI run without a GPU.

## Install

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Check the Environment

```bash
python -m tinyinfer
python scripts/check_environment.py --json
```

Missing optional GPU packages are reported as capabilities, not import failures.

## Quality Checks

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

## Reduced Roadmap

1. Engineering foundation.
2. Hugging Face Qwen2.5-0.5B reference runtime with explicit prefill/decode profiling.
3. Contiguous KV cache, continuous batching, paged KV cache, and a Qwen attention adapter.
4. One profile-guided Triton kernel, reproducible benchmarks, technical report, and demo.

See [the project plan](docs/PROJECT_PLAN.md) for gates and non-goals.

## Correctness Policy

Every optimized path needs a deterministic reference test before performance measurement. Results must record the code revision, hardware, software versions, workload, warm-up, repeated measurements, and raw data.

## Non-goals

V1 does not include a from-scratch Qwen implementation, HTTP serving, broad model support, distributed inference, quantization, speculative decoding, production authentication, or a general-purpose vLLM replacement.

## License

MIT
````

Move `PROJECT_PLAN.md` to `docs/PROJECT_PLAN.md` and replace it with this reduced structure:

```markdown
# TinyInfer Project Plan

> Version 0.2 — application-focused scope, 2026-07-16

## Objective

TinyInfer demonstrates an explainable understanding of LLM inference and GPU optimization. It is an experimental runtime, not a small vLLM clone.

The fixed V1 target is Qwen2.5-0.5B, BF16, greedy decoding, a single GPU, an offline runtime, and representative workloads. Hugging Face supplies model structure, tokenizer, weights, and the correctness baseline. TinyInfer owns request state, generation control, profiling, scheduling, KV allocation, block tables, and the attention cache adapter.

## Phase 0: Engineering Foundation

- Installable Python 3.11 package.
- Immutable validated configuration.
- Dependency-tolerant environment report.
- pytest, Ruff, mypy, CPU-only CI, README, and experiment documentation.

Gate G0: clean installation; tests, lint, formatting, and typing pass; environment commands emit human and JSON reports without requiring CUDA.

## Phase 1: Hugging Face Reference Runtime

- Load Qwen2.5-0.5B through Transformers.
- Implement explicit request state and greedy generation loop.
- Separate and measure prefill and decode.
- Use Hugging Face cache behavior as the initial reference.
- Record TTFT, TPOT, token throughput, and peak memory.

Gate G1: fixed prompts produce the same greedy tokens as `model.generate()`; raw profiling records include environment metadata and the Git revision.

## Phase 2: KV Cache and Continuous Batching

- Add a contiguous cache adapter.
- Implement waiting, running, finished, and cancelled request states.
- Implement iteration-level FCFS scheduling and batched decode.
- Add a fixed-size KV block pool, free list, block tables, and paged allocation.
- Replace only the Qwen attention KV path with a paged adapter; do not reimplement the complete model.
- Compare latency, throughput, memory, concurrency, utilization, and fragmentation.

Gate G2: deterministic tokens align across reference, contiguous, batched, and paged paths; randomized allocation tests show no leaks or duplicate blocks; representative concurrency workloads complete without memory growth.

## Phase 3: Profile-Guided Kernel and Final Report

- Profile the paged runtime before choosing a kernel.
- Prefer a paged KV gather kernel; attempt paged decode attention only if measurements justify it.
- Compare the Triton kernel with a PyTorch reference across predefined shapes.
- Produce reproducible benchmark data, two or three core figures, a 4–6 page English technical report, and a CLI or notebook demo.

Gate G3: the kernel passes correctness tests and its effective and ineffective workload regions are reported. The project remains complete if the kernel produces a well-explained negative performance result.

## Correctness and Measurement Rules

- Deterministic greedy generation is the system-level oracle.
- Compare intermediate cache contents and logits before comparing text.
- Save raw results with hardware, software versions, seeds, commands, workload, warm-up, repetitions, and Git revision.
- Report latency and throughput together; report OOM rather than changing the workload silently.
- Performance claims require at least five repetitions on identical inputs.

## Non-goals

- From-scratch Qwen implementation.
- HTTP or OpenAI-compatible serving in V1.
- Multiple model families or model sizes in the core evaluation.
- Sampling beyond greedy in the core evaluation.
- Quantization, distributed inference, speculative decoding, prefix caching, LoRA, multimodality, or production serving features.

## Deliverables

- Publicly readable GitHub repository after local initialization is verified.
- Reproducible environment and benchmark commands.
- Correctness and resource-management tests.
- Architecture diagram and experiment figures.
- English technical report and short demo.
```

- [ ] **Step 3: Add the MIT License**

`LICENSE`:

```text
MIT License

Copyright (c) 2026 Edwads-so

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Install development dependencies and run all checks**

Run:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m tinyinfer
python scripts/check_environment.py --json
```

Expected: installation succeeds; all tests, lint, formatting, and type checks pass; both commands exit 0; the JSON command emits one valid JSON object.

- [ ] **Step 5: Commit the completed scaffold**

```bash
git add .gitignore .gitattributes .github/workflows/ci.yml LICENSE README.md docs/PROJECT_PLAN.md PROJECT_PLAN.md
git commit -m "Complete runnable project scaffold"
```

### Task 5: Final Repository Verification

**Files:**
- Modify only files required to fix verification failures.

**Interfaces:**
- Validates all acceptance criteria from the design specification.

- [ ] **Step 1: Verify repository isolation and branch**

Run: `git rev-parse --show-toplevel && git branch --show-current`

Expected: repository root is `D:/project/TinyInfer` and branch is `main`.

- [ ] **Step 2: Verify clean installation from declared metadata**

Run: `python -m pip check`

Expected: `No broken requirements found.`

- [ ] **Step 3: Re-run the complete quality gate**

Run:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 4: Inspect final history and worktree**

Run: `git status --short --branch && git log --oneline --decorate -6`

Expected: `main` has the design, plan, and scaffold commits; no intended source or documentation file is untracked.
