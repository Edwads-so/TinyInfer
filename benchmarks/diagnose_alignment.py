"""Explain G2-P1 token divergences from the Hugging Face B=1 reference.

For every workload/mode this runs the workload more than once and reports, per
request, where its tokens first differ from ``reference_generate`` and how close
the reference's top choice was to the token TinyInfer picked at that step.  A BF16
tie (both tokens within one BF16 unit of the top logit) flipped under a different
batch shape is the accepted case; anything else points to a logic error.  This
tool only diagnoses; it never writes evidence.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_phase2 import (
    WorkloadExecution,
    bf16_tie,
    check_pool_capacity,
    default_modes,
    reference_logits_at,
    run_hf_sequential_workload,
    run_shared_paged_workload,
)
from tinyinfer.profiling.phase2 import (
    Phase2Mode,
    Phase2Workload,
    first_mismatch_index,
    load_phase2_workloads,
)
from tinyinfer.runtime.generation import reference_generate
from tinyinfer.runtime.hf_loader import load_reference_model
from tinyinfer.runtime.request import GenerationRequest

__all__ = ["Divergence", "divergence_at", "first_mismatch_index", "main"]


@dataclass(frozen=True, slots=True)
class Divergence:
    """Reference logits at the first step where TinyInfer chose a different token."""

    reference_token: int
    actual_token: int
    actual_rank: int
    logit_gap: float
    top_token_ids: tuple[int, ...]
    top_logits: tuple[float, ...]


def divergence_at(
    logits: Any, *, reference_token: int, actual_token: int, top_k: int = 5
) -> Divergence:
    """Summarize one vocabulary logit row at a divergence step.

    ``actual_rank`` is 1-based; ``logit_gap`` is reference minus actual logit.
    """
    row = logits.float()
    if int(row.argmax()) != reference_token:
        raise ValueError("reference token is not the argmax of the supplied logits")
    order = row.argsort(descending=True)
    top = order[: min(top_k, int(row.shape[0]))]
    return Divergence(
        reference_token=reference_token,
        actual_token=actual_token,
        actual_rank=int((order == actual_token).nonzero()[0, 0]) + 1,
        logit_gap=float(row[reference_token] - row[actual_token]),
        top_token_ids=tuple(int(token) for token in top),
        top_logits=tuple(float(row[token]) for token in top),
    )


def _describe(loaded: Any, prompt: str, actual: tuple[int, ...], reference: tuple[int, ...]) -> str:
    index = first_mismatch_index(actual, reference)
    if index is None:
        return "aligned"
    if index >= min(len(actual), len(reference)):
        return f"diverges@{index}: length differs ({len(actual)} vs {len(reference)} tokens)"
    logits = reference_logits_at(loaded, prompt, reference[:index])
    try:
        divergence = divergence_at(
            logits, reference_token=reference[index], actual_token=actual[index]
        )
    except ValueError:
        return (
            f"diverges@{index}: full-recompute argmax is {int(logits.argmax())}, "
            f"not reference {reference[index]} (reference itself sits on a near-tie)"
        )
    tie = bf16_tie(logits, reference_token=reference[index], actual_token=actual[index])
    verdict = (
        f"BF16 tie, tolerance {tie.tolerance}"
        if tie.is_tie
        else f"NOT A BF16 TIE (tolerance {tie.tolerance}) - investigate"
    )
    top = ", ".join(
        f"{token}:{value:.3f}"
        for token, value in zip(divergence.top_token_ids, divergence.top_logits, strict=True)
    )
    return (
        f"diverges@{index}: ref={divergence.reference_token} got={divergence.actual_token} "
        f"rank={divergence.actual_rank} gap={divergence.logit_gap:.4f} [{verdict}] top=[{top}]"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/qwen2_5_0_5b.yaml"))
    parser.add_argument("--workloads", type=Path, default=Path("benchmarks/phase2_workloads.json"))
    parser.add_argument("--workload-id", action="append", default=None)
    parser.add_argument("--kv-pool-tokens", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=2, help="runs per workload/mode")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from benchmarks.bench_reference import load_engine_config

    args = build_parser().parse_args(argv)
    if args.runs < 1:
        raise ValueError("runs must be positive")
    config = load_engine_config(args.config)
    loaded = load_reference_model(config)
    workloads = load_phase2_workloads(args.workloads)
    selected: list[Phase2Workload] = (
        [workloads[workload_id] for workload_id in args.workload_id]
        if args.workload_id
        else list(workloads.values())
    )
    pool_blocks = (args.kv_pool_tokens + config.kv_block_size - 1) // config.kv_block_size

    def run(workload: Phase2Workload, mode: Phase2Mode) -> WorkloadExecution:
        if mode == "hf_sequential":
            return run_hf_sequential_workload(loaded, workload)
        return run_shared_paged_workload(
            loaded, config, workload, mode=mode, pool_tokens=args.kv_pool_tokens
        )

    diverged = 0
    nondeterministic = 0
    for workload in selected:
        prompts = {item.request_id: item.prompt for item in workload.requests}
        references = {
            item.request_id: reference_generate(
                loaded, GenerationRequest(item.request_id, item.prompt, item.max_new_tokens)
            )
            for item in workload.requests
        }
        for mode in default_modes(workload):
            if mode != "hf_sequential":
                check_pool_capacity(
                    loaded,
                    workload,
                    mode=mode,
                    block_size=config.kv_block_size,
                    pool_blocks=pool_blocks,
                )
            runs = [
                {result.request_id: tuple(result.output_token_ids) for result in run_.results}
                for run_ in (run(workload, mode) for _ in range(args.runs))
            ]
            for request_id in sorted(references):
                outputs = {run_[request_id] for run_ in runs}
                deterministic = len(outputs) == 1
                nondeterministic += not deterministic
                actual = runs[0][request_id]
                description = _describe(loaded, prompts[request_id], actual, references[request_id])
                diverged += description != "aligned"
                print(
                    f"{workload.workload_id}/{mode} {request_id}: {description}; "
                    f"runs {'identical' if deterministic else 'DIFFER'}",
                    flush=True,
                )
    print(
        f"DIAGNOSE ALIGNMENT: {diverged} diverged request results, "
        f"{nondeterministic} nondeterministic across runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
