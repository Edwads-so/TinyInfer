# TinyInfer

TinyInfer is a compact experimental runtime for studying how KV-cache layout and
iteration-level scheduling affect LLM inference latency, throughput, and GPU memory
efficiency.

## Status

**G0.5 Linux/GPU Environment Baseline** is complete. Native WSL acceptance passed at
tested revision `05b9160`, including two GPU smoke runs and the full verifier. The release is
published from the documentation-only release revision identified by tag `v0.1.0-g0.5`.

**G1 Hugging Face Reference Runtime** is verified. CPU, WSL GPU/model, token-alignment,
canonical benchmark, code-review, feature CI, fast-forward merge, and main CI gates passed.
The 20 canonical records are committed in `docs/results/g1_reference.jsonl`; the
review-hardened runtime was revalidated on `3b3fae1`.

The repository currently provides:

- A reproducible WSL2 Ubuntu-22.04 environment with Python 3.11, PyTorch CUDA,
  Transformers, and Triton.
- A GPU correctness smoke test (BF16 matmul + Triton vector-add), verified in native WSL.
- A fail-fast G0.5 verification entrypoint (`scripts/verify_g0_5.py`).
- Fixed-revision Qwen2.5-0.5B loading.
- A TinyInfer-controlled greedy prefill and one-token decode loop.
- Four-case exact token alignment against Hugging Face `model.generate()`.
- Host/CUDA phase timing, throughput, and peak-memory observation.
- An atomic benchmark CLI and 20 schema-v1 canonical records.
- Engineering scaffold: packaging, validated configuration, environment diagnostics,
  tests, linting, typing, CI, and documentation.

## Phase 2 Status

Phase 2 capabilities are under active development and are NOT production-ready:

- `contiguous KV cache`: implemented prototype
- `paged KV cache`: correctness prototype
- `FCFS scheduler`: iteration-interleaving prototype
- `true batched decode`: local candidate; native-WSL correctness and audit remediation passed (remote CI status must be checked on the declared commit)
- `block-native paged attention`: not implemented
- `G2-P1` batching-throughput evidence: verified locally (baseline; see `docs/G2_P1_REPORT.md`)
- `G2-P2` paged-KV memory-efficiency evidence: not started
- `full G2`: not passed

See `docs/PHASE2_STATUS.md` and `docs/G2_BD_REPORT.md` for the completion boundary.

## Architecture Boundary

Hugging Face provides the Qwen2.5 model structure, tokenizer, weights, correctness
baseline, and the public reference `Cache`. TinyInfer currently owns request/result
contracts, the explicit prefill/decode loop, correctness comparison, profiling,
benchmark records, and Phase-2 prototype scheduling/cache allocation/block-table/Qwen
attention-cache-adapter paths. These prototypes are not production-ready and do not
constitute full G2.

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

## G0.5 Verification (GPU required)

```bash
# Full G0.5 gate: lock check, env report, pytest, ruff, mypy, GPU smoke
python scripts/verify_g0_5.py

# GPU correctness smoke test only
python scripts/check_gpu_smoke.py
```

The G0.5 verifier does not download models, install dependencies, or modify Git history.

## G1 Reference Runtime (GPU required)

Run correctness and the benchmark from a native Linux/WSL2 checkout with the fixed model
already cached:

```bash
uv sync --locked --extra dev --extra inference --extra kernels

HF_HUB_OFFLINE=1 python -m pytest tests/integration -m "gpu and model" -v

HF_HUB_OFFLINE=1 python benchmarks/bench_reference.py \
  --config configs/qwen2_5_0_5b.yaml \
  --prompts benchmarks/prompts.json \
  --output /tmp/tinyinfer-g1/reference.jsonl \
  --warmup-runs 3 \
  --repetitions 5 \
  --overwrite
```

Canonical inputs and evidence:

- [model configuration](configs/qwen2_5_0_5b.yaml)
- [frozen prompts](benchmarks/prompts.json)
- [benchmark CLI](benchmarks/bench_reference.py)
- [20-record canonical JSONL](docs/results/g1_reference.jsonl)
- [G1 reference runtime report](docs/G1_REFERENCE_RUNTIME_REPORT.md)

## Reduced Roadmap

**Completed:**
1. Engineering foundation.
2. Reproducible G0.5 Linux/GPU environment baseline, verification gate, and release.
3. Local G1 reference-runtime implementation, exact token alignment, profiling, and
   canonical benchmark evidence.

**In progress / planned:**
4. G1 remote CI/release gate.
5. Contiguous KV cache, continuous batching, paged KV cache, and a Qwen attention adapter.
6. One profile-guided Triton kernel, reproducible comparative benchmarks, technical
   report, and demo.

See [the project plan](docs/PROJECT_PLAN.md) for phase gates and non-goals. The detailed
[Phase 1 workflow](docs/PHASE1_WORKFLOW.md) covers the completed G0.5 environment baseline,
the Hugging Face reference runtime, correctness gates, and profiling outputs.
For a code-oriented explanation of the implemented Task 1–7 runtime, read the
[Phase 1 learning guide](docs/PHASE1_LEARNING_GUIDE.md).

For agent-executable workflow standards, see
[Workflow Authoring Standard](docs/WORKFLOW_AUTHORING_STANDARD.md) and the
[template](docs/templates/WORKFLOW_TEMPLATE.md).

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
