# G2-P1 GPU Runbook

> Status: `not yet executed` — this document is an execution handoff, not performance evidence.

## Purpose

Run the G2-P1 batching-throughput experiment on a native Linux/WSL CUDA host.
The experiment compares sequential, static-batch, and continuous-batching runs
without changing a workload's declared arrival pattern. It produces raw JSONL
records only after each request matches its own Hugging Face greedy reference.

## Preconditions

- Native Linux or WSL2 checkout of branch `codex/g2-p1-throughput` (or its merged
  equivalent), with a clean worktree.
- NVIDIA CUDA device visible to PyTorch; `python -c "import torch; assert torch.cuda.is_available()"`
  must exit zero.
- Cached fixed Qwen2.5-0.5B weights and tokenizer; do not allow downloads.
- Python 3.11 with locked dev and inference extras:

```bash
uv sync --locked --extra dev --extra inference
```

## Verification and collection

Run from the repository root:

```bash
HF_HUB_OFFLINE=1 python scripts/verify_g2_p1.py

HF_HUB_OFFLINE=1 python benchmarks/bench_phase2.py \
  --config configs/qwen2_5_0_5b.yaml \
  --workloads benchmarks/phase2_workloads.json \
  --output docs/results/g2_p1_raw.jsonl \
  --warmup-runs 3 \
  --repetitions 5

python benchmarks/summarize_phase2.py \
  --input docs/results/g2_p1_raw.jsonl \
  --output docs/results/g2_p1_summary.md
```

The verifier must print `G2-P1 VERIFICATION: PASS`. The benchmark must report
the number of records written and must not overwrite evidence without an explicit
`--overwrite`. A failed verifier, token mismatch, unreleased block pool, missing
CUDA device, or invalid workload/mode combination is a failed run; do not commit
partial JSONL as evidence.

## Expected workload/mode matrix

| Workload family | Sequential | Static batch | Continuous batch |
|---|---:|---:|---:|
| fixed | yes | yes | yes |
| mixed | yes | yes | yes |
| dynamic mixed arrival | yes | no | yes |

Static batch is intentionally invalid for a dynamic-arrival workload, because
silently changing arrivals would invalidate the comparison.

## Evidence required before a status change

Only after the commands above pass on the declared commit may a report state
`G2-P1: verified locally`. The report must include the Git SHA, hardware,
Torch/Transformers versions, exact commands, raw JSONL and summary links,
throughput/TTFT/TPOT/peak-memory results, and limitations. It must retain:

```text
G2-P2: not started
full G2: not passed
block-native paged attention: not implemented
```

Remote CI must then verify the same declared history before any broader release
claim. G2-P1 does not demonstrate paged-KV memory efficiency; that remains G2-P2.
