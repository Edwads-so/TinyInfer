# TinyInfer Runnable Scaffold Design

## Goal

Create a small, executable, testable foundation for TinyInfer before implementing model inference. The scaffold must establish repository isolation, packaging, configuration, environment diagnostics, quality checks, and documentation without downloading model weights or requiring a working GPU in CI.

## Scope

The initialization includes:

- an independent Git repository whose default branch is `main`;
- a Python 3.11 package named `tinyinfer`;
- an executable `python -m tinyinfer` entry point;
- immutable configuration objects with basic validation;
- a structured environment inspection command;
- pytest, Ruff, and static type-checking configuration;
- CPU-only unit tests and GitHub Actions;
- project documentation, license, and repository hygiene files.

It does not include model downloads, Qwen model code, CUDA kernels, Triton exercises, serving endpoints, schedulers, or KV cache implementations.

## Repository Layout

```text
TinyInfer/
├── .github/workflows/ci.yml
├── configs/qwen2_5_0_5b.yaml
├── docs/
│   ├── PROJECT_PLAN.md
│   └── superpowers/specs/
├── scripts/check_environment.py
├── tests/unit/
├── tinyinfer/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   └── environment.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

Empty future-facing directories will not be created merely to mirror the final architecture. New engine, model, cache, kernel, sampling, server, benchmark, and report directories will be introduced with their first implementation.

## Runtime Interfaces

`python -m tinyinfer` prints the package version and a concise readiness summary. `python scripts/check_environment.py` emits a human-readable environment report and supports JSON output for later benchmark metadata collection.

The environment inspection code must tolerate absent optional packages and unavailable CUDA. Missing Triton or Transformers is reported as capability state, not treated as an import failure.

The initial configuration object covers stable engine-wide values such as model identifier, dtype, device, maximum sequence length, and KV block size. Validation rejects invalid sequence lengths, unsupported dtypes, and non-positive block sizes early.

## Dependencies

The base package uses the Python standard library and remains installable without CUDA. Development dependencies include pytest, Ruff, and a type checker. PyTorch, Transformers, Safetensors, and Triton are optional dependency groups because their compatible versions must be fixed after the WSL2/CUDA environment is established.

No lock file is generated during this initialization: locking GPU dependencies before selecting and verifying the Linux CUDA environment would record a misleading Windows setup. Locking is a Phase 0 follow-up requirement.

## Testing and CI

Unit tests verify:

- package import and version exposure;
- valid and invalid configuration construction;
- stable environment report keys;
- graceful behavior when optional GPU dependencies are unavailable.

Ruff checks formatting and common correctness issues. Static type checking covers `tinyinfer` and `scripts`. GitHub Actions runs on a current Ubuntu Python 3.11 image and installs only development dependencies, so CI does not require a GPU.

## Documentation

The README presents the project positioning, current implementation status, supported Python version, local installation, environment inspection, test and lint commands, roadmap link, and explicit non-goals. It must not claim inference features that have not been implemented.

The existing project plan moves to `docs/PROJECT_PLAN.md` without content changes. The repository uses the MIT License.

## Error Handling

Configuration failures raise `ValueError` with actionable field-specific messages. Environment inspection converts optional import and GPU detection failures into structured status fields while preserving a diagnostic message. Unexpected failures in the CLI produce a non-zero exit status.

## Acceptance Criteria

The scaffold is complete when:

1. `TinyInfer` is recognized as an independent repository on branch `main`.
2. The package installs in editable mode with development dependencies.
3. `python -m tinyinfer` and the environment script run without requiring CUDA.
4. Unit tests, Ruff, and static type checks pass.
5. The README commands match the implemented interfaces.
6. The initial scaffold is committed locally with no unrelated parent-repository files included.

GitHub repository creation and pushing are deliberately separate. They will be performed after GitHub CLI is installed and authenticated, using a private `TinyInfer` repository unless the user requests public visibility.
