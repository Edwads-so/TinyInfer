# mypy: ignore-errors
"""Dynamic Triton boundary for the G0.5 vector-add smoke test.

Triton's JIT decorator and kernel pointer parameters do not expose complete
static typing metadata. Keeping that boundary in this module preserves strict
typing for the smoke-test orchestration code.
"""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def launch_vector_add(
    x: Any,
    y: Any,
    output: Any,
    n_elements: int,
    block_size: int,
) -> None:
    """Launch vector-add with a stable tuple grid."""
    grid = (triton.cdiv(n_elements, block_size),)
    _vector_add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
    )
