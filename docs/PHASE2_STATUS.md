# Phase 2 Status

> Current capability summary for Phase 2 (KV Cache & Continuous Batching).
> Historical G1 evidence (`docs/G1_REFERENCE_RUNTIME_REPORT.md`,
> `docs/results/g1_reference.jsonl`) is immutable and not reinterpreted here.

## Current Capabilities

- contiguous KV cache: implemented prototype
- paged KV cache: correctness prototype
- FCFS scheduler: iteration-interleaving prototype
- true batched decode: local candidate
- block-native paged attention: not implemented
- G2-P1: not started
- G2-P2: not started
- full G2: not passed

## Capability Table

| Capability | Status |
|---|---|
| contiguous KV cache | `implemented prototype` |
| paged KV cache | `correctness prototype` |
| FCFS scheduler | `iteration-interleaving prototype` |
| true batched decode | `local candidate; native-WSL correctness and audit remediation passed (remote CI status must be checked on the declared commit)` |
| block-native paged attention | `not implemented` |
| G2-P1: batching-throughput evidence | `not started` |
| G2-P2: paged-KV memory-efficiency evidence | `not started` |
| full G2 | `not passed` |

## Gate Boundary

- **G2-BD: local candidate** (true batched decode correctness). Native-WSL
  verification and audit remediation passed; remote CI status must be checked on the
  declared commit. Its implementation is tracked by
  `docs/superpowers/plans/2026-08-10-true-batched-decode.md` and
  `docs/superpowers/specs/2026-08-10-true-batched-decode-design.md`.
- **G2-P1: not started** — batching-throughput evidence, defined in
  `docs/superpowers/specs/2026-09-04-g2-performance-design.md`; implementation
  and GPU handoff are in `docs/G2_P1_RUNBOOK.md`.
- **G2-P2: not started** — paged-KV memory-efficiency evidence, after P1.
- **full G2: not passed** until G2-BD, G2-P1, G2-P2, and the original
  `docs/PROJECT_PLAN.md` Gate G2 conditions all pass.

## Boundary Notes

- True heterogeneous batched decode is proven correct against the HF
  reference (GPU, `tests/integration/test_batch_decode.py`); two independent
  audit passes found and we fixed seven boundary/lifecycle defects, then
  re-ran the full `scripts/verify_g2_bd.py` gate green after each pass.
- Paged KV uses dense left-padded gather, not block-native paged attention.
