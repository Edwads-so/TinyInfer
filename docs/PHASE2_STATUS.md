# Phase 2 Status

> Current capability summary for Phase 2 (KV Cache & Continuous Batching).
> Historical G1 evidence (`docs/G1_REFERENCE_RUNTIME_REPORT.md`,
> `docs/results/g1_reference.jsonl`) is immutable and not reinterpreted here.

## Current Capabilities

- contiguous KV cache: implemented prototype
- paged KV cache: correctness prototype
- FCFS scheduler: iteration-interleaving prototype
- true batched decode: not verified
- block-native paged attention: not implemented
- G2: not passed

## Capability Table

| Capability | Status |
|---|---|
| contiguous KV cache | `implemented prototype` |
| paged KV cache | `correctness prototype` |
| FCFS scheduler | `iteration-interleaving prototype` |
| true batched decode | `verified locally (G2-BD, pending remote CI)` |
| block-native paged attention | `not implemented` |
| G2 | `not passed` |

## Gate Boundary

- **G2-BD** (true batched decode correctness): not yet verified; tracked by
  `docs/superpowers/plans/2026-08-10-true-batched-decode.md` and
  `docs/superpowers/specs/2026-08-10-true-batched-decode-design.md`.
- **G2-P** (performance evidence): not started.
- **full G2**: not passed until G2-BD, G2-P, and the original
  `docs/PROJECT_PLAN.md` Gate G2 conditions all pass.

## Boundary Notes

- Only single-request decode is currently proven correct against the HF
  reference; batched decode is still a scheduler prototype.
- Paged KV uses dense left-padded gather, not block-native paged attention.
