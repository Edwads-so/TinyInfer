from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_does_not_claim_all_phase2_is_absent() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    stale = "Runtime-owned KV cache, continuous batching, paged KV cache"
    assert stale not in text


def test_phase2_status_preserves_gate_boundary() -> None:
    text = (ROOT / "docs" / "PHASE2_STATUS.md").read_text(encoding="utf-8")
    assert "G2-BD" in text
    assert "G2-P" in text
    assert "full G2" in text
    assert "block-native paged attention" in text
    assert "not implemented" in text
    assert "G2: not passed" in text


def test_phase2_docs_state_checkpoint_boundaries() -> None:
    status = (ROOT / "docs" / "PHASE2_STATUS.md").read_text(encoding="utf-8")
    beginner = (ROOT / "docs" / "BEGINNER_GUIDE.md").read_text(encoding="utf-8")

    assert "G2-BD: local candidate" in status
    assert "G2-P1: not started" in status
    assert "G2-P2: not started" in status
    assert "full G2: not passed" in status
    assert "true batched decode: not verified" not in beginner
