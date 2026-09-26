# G2 Performance Evidence Design

> Status: proposed — requires implementation plan and fresh GPU evidence.
> Scope: Phase 2 performance checkpoints only. This document does not claim full G2.

## Purpose

Measure two independent Phase 2 hypotheses without conflating scheduler effects
with KV-cache-layout effects:

1. **G2-P1 — batching throughput.** Iteration-level continuous batching improves
   serving throughput over sequential single-request execution on representative
   concurrent workloads, while reporting its latency trade-offs.
2. **G2-P2 — paged-KV memory efficiency.** Under heterogeneous, dynamic workloads,
   paged KV uses a fixed GPU-memory budget more efficiently than contiguous KV.

G2-P1 is the first implementation priority. G2-P2 follows only after G2-P1 has
reproducible evidence. Neither checkpoint changes the existing full-G2 gate in
`docs/PROJECT_PLAN.md`.

## Status vocabulary and boundaries

- **G2-BD:** local correctness candidate. The shared-pool heterogeneous decode path
  passed the native-WSL verifier and two audit-remediation passes. Its remote-CI
  state must be checked against the declared commit before calling it remotely
  verified.
- **G2-P1 / G2-P2:** not started until their benchmark commands, raw records, and
  fresh GPU runs exist.
- **Full G2:** not passed until G2-BD, G2-P1, G2-P2, and every original Gate G2
  condition pass. In particular, dense left-padded gather is not block-native
  paged attention and must not be described as such.

## G2-P1: batching throughput

### Comparisons

All runs use the fixed Qwen2.5-0.5B BF16 model, greedy decoding, one GPU, a
fixed model revision, and offline cached weights. Compare:

| Mode | Purpose |
|---|---|
| Sequential single-request | Serving baseline: one request runs to completion before the next begins. |
| Static batch | Separates ordinary same-arrival batching from dynamic scheduling. |
| Iteration-level continuous batching | Candidate path: scheduler admits and retires heterogeneous requests across decode iterations. |

The cache mode, block-pool geometry, prompt corpus, output caps, warmup count,
and repetition count must be recorded for every result. P1 must not attribute a
speedup to block-native attention.

### Workload matrix

Run each mode for at least five measured repetitions after warmup on these
workload families:

| Family | Prompt / generation shape | Arrival pattern | Primary purpose |
|---|---|---|---|
| Fixed | Same prompt length and `max_new_tokens` | All requests available at time zero | Establish the clean throughput curve. |
| Mixed-length | Short, medium, and long prompts and output caps | All requests available at time zero | Measure padding and length-skew effects. |
| Dynamic mixed-length | Same mixed corpus | Deterministic staggered arrivals and departures | Exercise continuous admission. |

The implementation plan will select exact prompt IDs, concurrency levels, and
arrival offsets. It must include at least one B=1 baseline and representative
concurrency above one, bounded by the tested hardware and pool capacity.

### Required records and metrics

Each raw JSONL record includes: Git revision, timestamp, hardware and software
environment, model/config/prompt revisions, mode, cache geometry, seed,
warmup/repetition identifiers, workload parameters, and command line.

Report the distribution and raw values for:

- completed requests/s and generated tokens/s;
- end-to-end request latency and TTFT (at least median and p95);
- TPOT (at least median and p95);
- peak allocated/reserved CUDA memory; and
- observed GPU utilization when a supported sampler is available; otherwise
  record `unavailable` rather than inferring utilization.

### Decision rule

The primary outcome is the throughput ratio of continuous batching to sequential
execution on each workload. A result is publishable when raw records are
complete, five repeated measurements are available, token correctness passes,
and latency distributions are reported beside throughput. No universal speedup
threshold is imposed: an unchanged or negative result is valid evidence if it
is reproducible and explained. A claimed improvement must identify its workload
and must not hide an accompanying TTFT regression.

## G2-P2: paged-KV memory efficiency

### Comparisons and workloads

Hold model, decode/scheduler policy, concurrency target, and completion policy
constant. Compare contiguous KV with paged KV across deterministic mixed-length
workloads that include staggered arrivals, early completions, and long/short
request co-residency. Run at least five measured repetitions after warmup.

### Required metrics

- peak allocated and reserved CUDA memory;
- KV capacity reserved, KV capacity used, and allocated/used ratio;
- internal fragmentation, defined as allocated KV capacity minus live logical
  KV capacity, reported both as bytes/tokens and a ratio;
- external fragmentation, defined explicitly from the allocator/block-pool
  state before collection begins (not estimated from CUDA peak memory);
- completed workload size and the highest successful active-request concurrency
  under an explicitly stated memory budget; and
- leaks, duplicate blocks, or unreleased allocations, which are correctness
  failures rather than performance measurements.

### Decision rule

P2 can conclude that paged KV is more memory-efficient only for a named workload
and memory budget where it either supports more active requests or lowers a
defined waste metric without correctness/lifecycle failure. A null or negative
result remains a valid documented outcome.

## Deliverables and verification

Implementation must provide a benchmark CLI, immutable workload definitions,
schema-validated JSONL raw records, a deterministic summarizer, and tests for
the record schema, workload construction, metric calculations, and command
fail-fast behavior. GPU commands must not install packages, download models, or
silently reduce a failed workload.

The final report must state exact commands, revisions, hardware, raw-data paths,
figures/tables derived from those records, limitations, and whether each G2-P
sub-gate passed. It must retain the boundary that full G2 is not passed without
the original Gate G2 requirements.

## Non-goals

- Block-native paged attention or a Triton kernel.
- New model families, sampling, distributed inference, or production serving.
- Combining P1 and P2 into one causal performance claim.
- A G2 release tag before full-G2 conditions are satisfied.

## Amendment (2026-09-26): measurement fixes before first GPU run

No G2-P1 records existed when these changes were made, so no evidence is reinterpreted.

- **Latency from submission.** Schema v1 measured TTFT and end-to-end latency from
  admission, which hid queueing time in sequential mode.
- **Throughput from makespan.** Schema v1 divided tokens by the slowest single
  request's latency; in sequential mode that overstated throughput by roughly the batch
  size. Record schema v2 stores `makespan_ms` (first submission to last completion)
  and per-request `e2e_p50_ms` / `e2e_p95_ms`.
- **Absolute reference.** A fourth mode, `hf_sequential`, runs each request through the
  G1 explicit greedy loop on Hugging Face's own cache, so the cost of the paged adapter
  is visible, not only the relative benefit of batching.
- **Workload sweep.** `fixed_b1/2/4/8/16` give the batch-size curve; `long_b1/2/4/8`
  reuse the frozen G1 `long_context` prompt to expose dense-gather cost at long context.
- **Pool capacity.** `--kv-pool-tokens` sizes the shared pool and every workload/mode
  is checked against it before any run.
