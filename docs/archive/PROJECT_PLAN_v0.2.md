# TinyInfer Project Plan

> Version 0.2 — application-focused scope, 2026-07-16

## Objective

TinyInfer demonstrates an explainable understanding of LLM inference and GPU optimization.
It is an experimental runtime, not a small vLLM clone.

The fixed V1 target is Qwen2.5-0.5B, BF16, greedy decoding, a single GPU, an offline runtime,
and representative workloads. Hugging Face supplies model structure, tokenizer, weights, and
the correctness baseline. TinyInfer owns request state, generation control, profiling,
scheduling, KV allocation, block tables, and the attention cache adapter.

## Phase 0: Engineering Foundation

- Installable Python 3.11 package.
- Immutable validated configuration.
- Dependency-tolerant environment report.
- pytest, Ruff, mypy, CPU-only CI, README, and experiment documentation.

Gate G0: clean installation; tests, lint, formatting, and typing pass; environment commands
emit human and JSON reports without requiring CUDA.

## Phase 1: Hugging Face Reference Runtime

- Load Qwen2.5-0.5B through Transformers.
- Implement explicit request state and a greedy generation loop.
- Separate and measure prefill and decode.
- Use Hugging Face cache behavior as the initial reference.
- Record TTFT, TPOT, token throughput, and peak memory.

Gate G1: fixed prompts produce the same greedy tokens as `model.generate()`; raw profiling
records include environment metadata and the Git revision.

## Phase 2: KV Cache and Continuous Batching

- Add a contiguous cache adapter.
- Implement waiting, running, finished, and cancelled request states.
- Implement iteration-level FCFS scheduling and batched decode.
- Add a fixed-size KV block pool, free list, block tables, and paged allocation.
- Replace only the Qwen attention KV path with a paged adapter; do not reimplement the model.
- Compare latency, throughput, memory, concurrency, utilization, and fragmentation.

Gate G2: deterministic tokens align across reference, contiguous, batched, and paged paths;
randomized allocation tests show no leaks or duplicate blocks; representative concurrency
workloads complete without memory growth.

## Phase 3: Profile-Guided Kernel and Final Report

- Profile the paged runtime before choosing a kernel.
- Prefer a paged KV gather kernel; attempt paged decode attention only if measurements justify it.
- Compare the Triton kernel with a PyTorch reference across predefined shapes.
- Produce reproducible benchmark data, two or three core figures, a 4–6 page English technical
  report, and a CLI or notebook demo.

Gate G3: the kernel passes correctness tests and its effective and ineffective workload regions
are reported. The project remains complete if the kernel produces a well-explained negative
performance result.

## Correctness and Measurement Rules

- Deterministic greedy generation is the system-level oracle.
- Compare intermediate cache contents and logits before comparing text.
- Save raw results with hardware, software versions, seeds, commands, workload, warm-up,
  repetitions, and Git revision.
- Report latency and throughput together; report OOM rather than changing the workload silently.
- Performance claims require at least five repetitions on identical inputs.

## Non-goals

- From-scratch Qwen implementation.
- HTTP or OpenAI-compatible serving in V1.
- Multiple model families or model sizes in the core evaluation.
- Sampling beyond greedy in the core evaluation.
- Quantization, distributed inference, speculative decoding, prefix caching, LoRA,
  multimodality, or production serving features.

## Deliverables

- Publicly readable GitHub repository after local initialization is verified.
- Reproducible environment and benchmark commands.
- Correctness and resource-management tests.
- Architecture diagram and experiment figures.
- English technical report and short demo.
