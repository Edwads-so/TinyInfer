"""Shared fixtures: a tiny random Qwen2 saved like a real checkpoint."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class TinyQwen2:
    """A saved tiny checkpoint plus the FP32 Hugging Face model it came from."""

    model_dir: Path
    hf_model: Any


@pytest.fixture(scope="session")
def tiny_qwen2(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TinyQwen2]:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    # Random init leaves norms at 1 and biases at 0; perturb them so a wrong
    # mapping or a skipped bias changes the output.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("bias"):
                parameter.add_(torch.randn_like(parameter) * 0.1)
    model_dir = tmp_path_factory.mktemp("tiny_qwen2")
    model.save_pretrained(model_dir)
    yield TinyQwen2(model_dir=model_dir, hf_model=model)
