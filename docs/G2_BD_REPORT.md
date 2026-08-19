# G2-BD Verification Report

> Status: `local candidate` — verified locally (remote CI not yet green).

## Summary

G2-BD (true heterogeneous batched decode over TinyInfer's shared paged KV pool)
is verified on native WSL2:

- one real Qwen forward observes `[3, 1]` input with `BatchedPagedTinyinferCache`,
- the matching iteration records one decode forward and three distinct old lengths,
- each request's output equals its own HF `reference_generate` token-for-token,
- the block pool fully recovers after repeated runs,
- peak CUDA memory stays within 8% of the first measured peak.

## Boundary

- G2-BD: verified locally
- G2-P: not started
- full G2: not passed

## Evidence

- verifier: `scripts/verify_g2_bd.py` (8-step fail-fast gate)
- implementation/review history: branch `agent/true-batched-decode`
- runtime: Python 3.11.15, Torch 2.13.0+cu130, Transformers 5.14.1, RTX 4080 SUPER

## Dense-gather limitation

The batched cache adapter uses dense left-padded gather; block-native paged
attention is not implemented. Performance claims for B=1/2/4/8 belong to G2-P.


## Audit Remediation (2026-08-10)

An independent audit found five boundary/lifecycle defects in the initial
verification claim, and the `verified locally` wording was premature:

| # | Defect | Fix |
|---|--------|-----|
| T2 | `allocate_blocks` overflowed table capacity leaving dirty mappings | pre-check `num_blocks > max_logical_blocks`; rollback via `clear_all()` |
| T4 | `write_decode_batch` accepted wrong dtype/device | dtype + device validation before mutation |
| T5 | batched adapter accepted foreign-geometry states | `BlockTable.pool_geometry` stamp + constructor validation |
| T7 | `step()` decoded past `max_new_tokens` | finalize-before-decode in `step()` |
| T7 | prefill/decode exceptions leaked blocks/states/queue entries | `_abort_all()` cleanup + re-raise |

Each defect now has a RED regression test (`TestCapacityOverflow`,
`TestBatchValidationMetadata`, `TestGeometryValidation`, `TestMaxNewBound`,
`TestFailureCleanup`) and the full verifier was re-run green:

- `verify_g2_bd.py` 8-step gate: **PASS**
- unit suite: 279 passed (was 272)
- GPU batched decode integration: 2 passed (real Qwen `[3,1]` path identity +
  token alignment retained)

## Second Audit Remediation (2026-08-11)

A follow-up audit found two remaining boundary/lifecycle defects not covered
by the first pass:

| # | Defect | Fix |
|---|--------|-----|
| T3 | released `PagedTinyinferCache` still served reads | `require_active()` guard on all five read paths (`__getitem__`, `get_seq_length`, `get_query_offset`, `get_mask_sizes`, `length`) |
| T7 | partial admission leaked allocated blocks on mid-init failure | move `admit` + `_init_request_state` into the `step()` try; `_abort_all()` cancels every running queue entry before releasing blocks |

Each defect now has a RED regression test (`TestReleasedCacheReads`,
`TestAdmissionPartialFailure`) and the full verifier was re-run green:

- `verify_g2_bd.py` 8-step gate: **PASS**
- unit suite: 282 passed (was 279)
- ruff check/format and mypy: all green
