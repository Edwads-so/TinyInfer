from __future__ import annotations

from collections.abc import Sequence

from tinyinfer import __version__
from tinyinfer.environment import inspect_environment, render_human_report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TinyInfer readiness command."""
    del argv
    print(f"TinyInfer {__version__}")
    print(render_human_report(inspect_environment()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
