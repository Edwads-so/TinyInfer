from __future__ import annotations

from collections.abc import Sequence

from tinyinfer import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TinyInfer readiness command."""
    del argv
    print(f"TinyInfer {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
