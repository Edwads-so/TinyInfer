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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without GPU) -- §6.2
# ---------------------------------------------------------------------------

FIXED_SIZES: tuple[int, ...] = (1, 257, 4096, 65537)

TRITON_BLOCK_SIZE = 1024


def triton_result_name(size: int) -> str:
    """Return the stable evidence label for one vector-add size."""
    return f"triton-add-{size}"


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Single smoke check outcome."""

    name: str
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Return true only when the check recorded no error."""
        return self.error is None

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
        return SmokeResult("cuda-available", "torch.cuda.is_available() == False")

    if not torch.cuda.is_bf16_supported():
        return SmokeResult("bf16-supported", "BF16 not supported on this GPU")

    try:
        device = torch.device("cuda")
        x = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()

        if y.shape != (1024, 1024):
            return SmokeResult("matmul-shape", f"got {y.shape}")
        if y.dtype != torch.bfloat16:
            return SmokeResult("matmul-dtype", f"got {y.dtype}")
        if y.device.type != "cuda":
            return SmokeResult("matmul-device", f"got {y.device}")
        if not torch.isfinite(y).all():
            return SmokeResult("matmul-finite", "output contains inf/nan")
    except Exception as exc:
        return SmokeResult("cuda-matmul", str(exc))

    return SmokeResult("cuda-matmul")


def _check_triton_vector_add() -> list[SmokeResult]:
    """Design §6.1 items 5-8."""
    import torch

    try:
        from scripts.triton_vector_add import launch_vector_add
    except ImportError as exc:
        return [SmokeResult("triton-import", str(exc))]

    device = torch.device("cuda")
    results: list[SmokeResult] = []
    for n in FIXED_SIZES:
        x = torch.randn(n, device=device, dtype=torch.float32)
        y = torch.randn(n, device=device, dtype=torch.float32)
        out = torch.empty_like(x)
        name = triton_result_name(n)

        try:
            launch_vector_add(x, y, out, n, TRITON_BLOCK_SIZE)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, x + y)
        except Exception as exc:
            results.append(SmokeResult(name, str(exc)))
            break
        results.append(SmokeResult(name))

    return results


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full GPU smoke suite.  Exit 0 only when every check passes."""
    results: list[SmokeResult] = []
    results.append(_check_cuda_matmul())

    if results[-1].passed:
        results.extend(_check_triton_vector_add())
    else:
        results.append(SmokeResult("triton-add", "skipped: CUDA matmul did not pass"))

    return summarize_results(results)


if __name__ == "__main__":
    raise SystemExit(main())
