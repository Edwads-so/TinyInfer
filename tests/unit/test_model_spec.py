import json
from pathlib import Path
from typing import Any

import pytest

from tinyinfer.model.spec import QWEN25_05B, load_qwen2_spec

# Values as published in the Qwen2.5-0.5B config.json (pre-5.x layout).
HUB_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_act": "silu",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "max_position_embeddings": 32768,
    "max_window_layers": 24,
    "model_type": "qwen2",
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def _write(tmp_path: Path, config: dict[str, Any]) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_hub_layout_parses_qwen25_05b_shapes(tmp_path: Path) -> None:
    spec = load_qwen2_spec(_write(tmp_path, HUB_CONFIG))

    assert spec == QWEN25_05B
    assert spec.num_kv_groups == 7


def test_transformers5_layout_reads_rope_parameters(tmp_path: Path) -> None:
    config = {
        key: value for key, value in HUB_CONFIG.items() if key not in {"rope_theta", "rope_scaling"}
    }
    config["rope_parameters"] = {"rope_theta": 10000.0, "rope_type": "default"}
    config["layer_types"] = ["full_attention"] * 24

    assert load_qwen2_spec(_write(tmp_path, config)).rope_theta == 10000.0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model_type": "llama"}, "model_type"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"use_sliding_window": True}, "sliding"),
        ({"rope_scaling": {"type": "yarn", "factor": 4.0}}, "rope"),
        ({"layer_types": ["full_attention", "sliding_attention"]}, "layer_types"),
        ({"hidden_size": 900}, "divisible"),
        ({"num_key_value_heads": 3}, "num_key_value_heads"),
        ({"rope_parameters": {"rope_theta": 1.0, "rope_type": "linear"}}, "rope"),
    ],
)
def test_unsupported_configs_fail_loudly(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    config = {**HUB_CONFIG, **override}
    if "rope_parameters" in override:
        config.pop("rope_theta")
        config.pop("rope_scaling")

    with pytest.raises(ValueError, match=message):
        load_qwen2_spec(_write(tmp_path, config))


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"config\.json"):
        load_qwen2_spec(tmp_path)
