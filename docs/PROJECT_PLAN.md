# TinyInfer Project Plan

> Version 0.3 — owned-model scope, 2026-09-26.
> Supersedes v0.2 (`docs/archive/PROJECT_PLAN_v0.2.md`).

## Objective

TinyInfer demonstrates an explainable understanding of LLM inference and GPU optimization.
It is an experimental runtime, not a small vLLM clone.

The fixed V1 target is Qwen2.5-0.5B, BF16, greedy decoding, a single GPU, an offline runtime,
and representative workloads. Hugging Face supplies the tokenizer, weights, and the
correctness and performance reference. TinyInfer owns the Qwen2 forward pass, request state,
generation control, profiling, scheduling, KV allocation, block tables, and attention.

## Changes from v0.2

- **A from-scratch Qwen2 forward pass moves from non-goal to core scope.** Writing the model
  is the most direct evidence of understanding. Block-native paged attention also requires
  TinyInfer to own the attention call; the Hugging Face `Cache` protocol can only hand
  dense KV back to Hugging Face attention.
- **G2-P2 requires on-demand block allocation.** Reserving `prompt + max_new_tokens` blocks
  at admission is memory-equivalent to a max-length contiguous reservation, so a
  paged-vs-contiguous comparison would measure nothing.
- **Performance is reported as a before/after series** on identical workloads:
  Hugging Face reference → Hugging Face model with dense-gather paged adapter (current)
  → owned model with vectorized gather → owned model with a Triton paged kernel.

## Status (2026-09-28)

| Checkpoint | Status |
|---|---|
| G0 / G0.5 engineering and GPU environment | passed |
| G1 Hugging Face reference runtime | passed |
| G2-BD true heterogeneous batched decode | local candidate |
| G2-P1 batching-throughput evidence | verified locally (2026-09-27, `docs/G2_P1_REPORT.md`) |
| G3 owned Qwen2 forward | verified locally (2026-09-28: 9/9 GPU gate tests; 85 of 104 checkpoints bit-identical) |
| G2-P2 paged memory evidence | not started |
| G4 Triton kernel | not started |

## Phase 0: Engineering Foundation — complete

Installable Python 3.11 package, validated configuration, environment report, pytest, Ruff,
mypy, CPU-only CI, and a reproducible WSL2 GPU environment.

## Phase 1: Hugging Face Reference Runtime — complete

Explicit greedy prefill/decode loop, exact token alignment with `model.generate()`, and
canonical TTFT/TPOT/throughput/memory records (`docs/results/g1_reference.jsonl`).

## Phase 2: KV Cache and Continuous Batching — in progress

Done: contiguous KV cache, shared paged block pool with block tables, FCFS iteration-level
scheduling, and one `[B, 1]` decode forward over heterogeneous requests (G2-BD).

Remaining — **G2-P1 baseline ("before")**:

- Run the G2-P1 benchmark on the current path per `docs/G2_P1_RUNBOOK.md`.
- Workloads cover B = 1, 2, 4, 8, 16 for fixed requests, mixed-length and dynamic-arrival
  families, and a long-prompt family (≥ 512 prompt tokens) that exposes gather cost.
- Include Hugging Face `generate_greedy` rows as the absolute reference.
- Report achieved weight-read bandwidth against the GPU peak for B = 1.

Gate G2-P1: as defined in `docs/superpowers/specs/2026-09-04-g2-performance-design.md`.

## Phase 3: TinyInfer-owned Qwen2 Forward

- Load fixed-revision safetensors weights; implement embedding, RMSNorm, RoPE, GQA attention,
  SwiGLU MLP, decoder layers, final norm, and the tied LM head.
- Compare hidden states with Hugging Face layer by layer before comparing tokens.
- Attention reads TinyInfer KV directly (`PagedKVCache`, block tables, per-row lengths),
  replacing the Hugging Face `Cache` adapters.
- A vectorized gather (precomputed slot indices) plus PyTorch SDPA is the paged baseline.
- Re-run the G2-P1 workloads on this path.

Gate G3: FP32 operator tests pass at `atol=1e-5, rtol=1e-4`; BF16 per-layer hidden-state
cosine similarity is at least 0.999; the four G1 canonical cases and the G2-BD batched
integration case produce the same greedy tokens as Hugging Face.

## Phase 4: On-demand Paging and Memory Evidence (G2-P2)

- Allocate a block only when a sequence crosses a block boundary.
- Admit requests according to free blocks; when the pool is exhausted, preempt by
  recomputation rather than corrupting a running request.
- Randomized allocation tests show no leaks or duplicate blocks.
- Run G2-P2 as defined in the 2026-09-04 performance design: max-length reservation versus
  on-demand paging under a fixed memory budget.

Full Gate G2: deterministic tokens align across reference, contiguous, batched, and paged
paths; allocation integrity holds; representative concurrency workloads complete without
memory growth; G2-BD, G2-P1, and G2-P2 pass.

## Phase 5: Profile-Guided Triton Kernel

- Profile the owned paged runtime before choosing the kernel.
- Prefer a paged GQA decode-attention kernel that reads block tables directly; fall back to
  a paged KV gather kernel if measurements or effort do not justify attention.
- Compare against vectorized gather plus SDPA across context length, batch size, and block
  size.

Gate G4: the kernel passes correctness tests, and both effective and ineffective workload
regions are reported. A well-explained negative result completes the phase.

## Phase 6: Report and Demo

- Figure 1: throughput and latency versus batch size for each path in the before/after
  series, with bandwidth utilization.
- Figure 2: maximum concurrency and KV waste, reservation versus on-demand paging.
- Figure 3: kernel speedup regions.
- A 4–6 page English technical report, an architecture diagram, and a CLI or notebook demo.

## Phase 7 (optional): One Research Extension

Choose exactly one direction from measured bottlenecks: kernel specialization by block size
and context distribution, memory-pressure-aware admission, or chunked prefill. Do not frame
a result as a contribution if the gain appears only on one hand-picked shape, lacks a
same-input baseline, fails to reproduce across five runs, or comes from disabling baseline
features.

## Correctness and Measurement Rules

- Deterministic greedy generation is the system-level oracle.
- Compare intermediate hidden states, cache contents, and logits before comparing text.
- Save raw results with hardware, software versions, seeds, commands, workload, warm-up,
  repetitions, and Git revision.
- Report latency and throughput together; report OOM rather than changing the workload silently.
- Performance claims require at least five repetitions on identical inputs.

## Non-goals

- HTTP or OpenAI-compatible serving in V1.
- Multiple model families or model sizes in the core evaluation.
- Sampling beyond greedy in the core evaluation.
- Quantization, distributed inference, speculative decoding, prefix caching, LoRA,
  multimodality, or production serving features.

## Deliverables

- Publicly readable GitHub repository with reproducible environment and benchmark commands.
- Correctness and resource-management tests.
- Architecture diagram, experiment figures, English technical report, and short demo.
- Public-facing documentation limited to the README, design documents, the report, and
  learning notes; process documents move to `docs/archive/`.
