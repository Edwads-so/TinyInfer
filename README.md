# TinyInfer

TinyInfer is a compact experimental runtime for studying how KV-cache layout and
iteration-level scheduling affect LLM inference latency, throughput, and GPU memory
efficiency.

## Status

The repository currently provides the engineering scaffold: packaging, validated
configuration, environment diagnostics, tests, linting, typing, CI, and documentation.
Model inference, continuous batching, paged KV cache, and Triton kernels are planned and
are not implemented yet.

## Architecture Boundary

Hugging Face will provide the Qwen2.5 model structure, tokenizer, weights, and correctness
baseline. TinyInfer will own request state, the explicit prefill/decode loop, scheduling,
profiling, cache allocation, block tables, and the Qwen attention cache adapter.

## Requirements

- Python 3.11
- Linux or WSL2 for later CUDA and Triton work
- An NVIDIA GPU with BF16 support for inference phases

The scaffold and CI run without a GPU.

## Install

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

On Windows PowerShell, commands can be run directly with `.venv\Scripts\python.exe` before
activating the environment. Later CUDA development will use Linux or WSL2.

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

See [the project plan](docs/PROJECT_PLAN.md) for phase gates and non-goals. The detailed
[Phase 1 workflow](docs/PHASE1_WORKFLOW.md) covers WSL2 environment readiness, the Hugging
Face reference runtime, correctness gates, and profiling outputs.

## Correctness Policy

Every optimized path needs a deterministic reference test before performance measurement.
Results must record the code revision, hardware, software versions, workload, warm-up,
repeated measurements, and raw data.

## Non-goals

V1 does not include a from-scratch Qwen implementation, HTTP serving, broad model support,
distributed inference, quantization, speculative decoding, production authentication, or a
general-purpose vLLM replacement.

## License

MIT
