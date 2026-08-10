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
