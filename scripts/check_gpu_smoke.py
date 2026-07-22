#!/usr/bin/env python3
"""Reproducible GPU correctness smoke test for TinyInfer G0.5 gate.

Design §6.1 -- ten explicit requirements:
1.  Confirm PyTorch CUDA is available.
2.  Execute BF16 1024x1024 matmul on CUDA.
3.  Explicit torch.cuda.synchronize() after the matmul.
4.  Verify output shape, dtype, device, and finiteness.
5.  Compile and run Triton vector-add kernel.
6.  Cover lengths 1, 257, 4096, 65537.
7.  Use a mask for partial-block tails.
8.  Compare against x + y with torch.testing.assert_close.
9.  Return non-zero exit on any failure.
10. On success, print a structured PASS line for humans and agents.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without GPU) -- §6.2
# ---------------------------------------------------------------------------

FIXED_SIZES: tuple[int, ...] = (1, 257, 4096, 65537)

TRITON_BLOCK_SIZE = 1024


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Single smoke check outcome."""

    name: str
    passed: bool = True
    error: str | None = None

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        suffix = f" -- {self.error}" if self.error else ""
        return f"[{status}] {self.name}{suffix}"


def summarize_results(results: Sequence[SmokeResult]) -> int:
    """Print a summary and return 0 when every result passed."""
    all_ok = True
    for r in results:
        print(repr(r))
        if not r.passed:
            all_ok = False
    print()
    if all_ok:
        print("G0.5 GPU SMOKE: PASS")
        return 0
    print("G0.5 GPU SMOKE: FAIL")
    return 1


# ---------------------------------------------------------------------------
# GPU-dependent checks
# ---------------------------------------------------------------------------


def _check_cuda_matmul() -> SmokeResult:
    """Design §6.1 items 1-4."""
    import torch

    if not torch.cuda.is_available():
        return SmokeResult("cuda-available", False, "torch.cuda.is_available() == False")

    if not torch.cuda.is_bf16_supported():
        return SmokeResult("bf16-supported", False, "BF16 not supported on this GPU")

    try:
        device = torch.device("cuda")
        x = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()

        if y.shape != (1024, 1024):
            return SmokeResult("matmul-shape", False, f"got {y.shape}")
        if y.dtype != torch.bfloat16:
            return SmokeResult("matmul-dtype", False, f"got {y.dtype}")
        if y.device.type != "cuda":
            return SmokeResult("matmul-device", False, f"got {y.device}")
        if not torch.isfinite(y).all():
            return SmokeResult("matmul-finite", False, "output contains inf/nan")
    except Exception as exc:
        return SmokeResult("cuda-matmul", False, str(exc))

    return SmokeResult("cuda-matmul")


def _triton_vector_add_kernel():
    """Inline Triton kernel -- avoids import-time dependency."""
    import triton
    import triton.language as tl

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    return _add_kernel


def _check_triton_vector_add() -> SmokeResult:
    """Design §6.1 items 5-8."""
    import torch

    try:
        import triton  # noqa: F401
    except ImportError as exc:
        return SmokeResult("triton-import", False, str(exc))

    try:
        add_kernel = _triton_vector_add_kernel()
    except Exception as exc:
        return SmokeResult("triton-compile", False, str(exc))

    device = torch.device("cuda")
    for n in FIXED_SIZES:
        x = torch.randn(n, device=device, dtype=torch.float32)
        y = torch.randn(n, device=device, dtype=torch.float32)
        out = torch.empty_like(x)

        grid = lambda meta: (  # noqa: E731
            (n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        )
        try:
            add_kernel[grid](x, y, out, n, BLOCK_SIZE=TRITON_BLOCK_SIZE)
        except Exception as exc:
            return SmokeResult(f"triton-add-{n}", False, str(exc))

        torch.cuda.synchronize()
        try:
            torch.testing.assert_close(out, x + y)
        except AssertionError as exc:
            return SmokeResult(f"triton-add-{n}", False, str(exc))

    return SmokeResult("triton-add")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full GPU smoke suite.  Exit 0 only when every check passes."""
    results: list[SmokeResult] = []
    results.append(_check_cuda_matmul())

    if results[-1].passed:
        results.append(_check_triton_vector_add())
    else:
        results.append(
            SmokeResult("triton-add", False, "skipped: CUDA matmul did not pass")
        )

    return summarize_results(results)


if __name__ == "__main__":
    raise SystemExit(main())