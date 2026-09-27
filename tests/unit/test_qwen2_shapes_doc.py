from pathlib import Path

from scripts.render_qwen2_shapes import render

DOC = Path(__file__).resolve().parents[2] / "docs" / "QWEN2_SHAPES.md"


def test_render_reports_hand_checked_qwen25_05b_shapes() -> None:
    text = render()

    assert "| `layer.q` |" in text
    assert "`[1, 14, 9, 64]`" in text  # prefill query heads
    assert "`[1, 2, 10, 64]`" in text  # decode K history: 9 prompt tokens + 1 new
    assert "`[1, 9, 151936]`" in text  # prefill logits
    assert "494,032,768" in text  # 136,134,656 embedding + 24 x 14,912,384 + 896
    assert "12,288" in text  # BF16 KV bytes per token: 2 x 24 x 2 x 64 x 2


def test_shapes_doc_matches_the_code() -> None:
    assert DOC.read_text(encoding="utf-8") == render(), (
        "docs/QWEN2_SHAPES.md is stale; run python scripts/render_qwen2_shapes.py"
    )
