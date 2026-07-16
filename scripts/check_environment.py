"""Run TinyInfer environment inspection from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run() -> int:
    """Load TinyInfer after making the source checkout importable."""
    from tinyinfer.environment import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
