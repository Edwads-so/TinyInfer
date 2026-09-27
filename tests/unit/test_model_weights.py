import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinyinfer.model.spec import load_qwen2_spec
from tinyinfer.model.weights import load_qwen2_weights, resolve_model_dir


def _copy_checkpoint(source: Path, target: Path, tensors: dict[str, torch.Tensor]) -> Path:
    target.mkdir()
    (target / "config.json").write_text(
        (source / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    save_file(tensors, str(target / "model.safetensors"))
    return target


def test_weights_match_checkpoint_tensors(tiny_qwen2: Any) -> None:
    spec = load_qwen2_spec(tiny_qwen2.model_dir)
    weights = load_qwen2_weights(
        tiny_qwen2.model_dir, spec, device=torch.device("cpu"), dtype=torch.float32
    )
    hf = tiny_qwen2.hf_model.state_dict()

    assert len(weights.layers) == 2
    torch.testing.assert_close(weights.embed, hf["model.embed_tokens.weight"])
    torch.testing.assert_close(weights.final_norm, hf["model.norm.weight"])
    layer = weights.layers[1]
    prefix = "model.layers.1."
    torch.testing.assert_close(layer.q_proj, hf[prefix + "self_attn.q_proj.weight"])
    torch.testing.assert_close(layer.k_bias, hf[prefix + "self_attn.k_proj.bias"])
    torch.testing.assert_close(layer.o_proj, hf[prefix + "self_attn.o_proj.weight"])
    torch.testing.assert_close(layer.post_norm, hf[prefix + "post_attention_layernorm.weight"])
    torch.testing.assert_close(layer.down_proj, hf[prefix + "mlp.down_proj.weight"])
    assert weights.lm_head is weights.embed


def test_weights_convert_dtype(tiny_qwen2: Any) -> None:
    spec = load_qwen2_spec(tiny_qwen2.model_dir)
    weights = load_qwen2_weights(
        tiny_qwen2.model_dir, spec, device=torch.device("cpu"), dtype=torch.bfloat16
    )

    assert weights.embed.dtype == torch.bfloat16
    assert weights.layers[0].q_bias.dtype == torch.bfloat16


def _tensors(tiny_qwen2: Any) -> dict[str, torch.Tensor]:
    return load_file(str(tiny_qwen2.model_dir / "model.safetensors"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: t.pop("model.layers.0.self_attn.q_proj.bias"), "missing.*q_proj.bias"),
        (lambda t: t.update({"model.extra.weight": torch.zeros(1)}), "unexpected.*model.extra"),
        (
            lambda t: t.update({"model.norm.weight": torch.zeros(3)}),
            "shape.*model.norm.weight",
        ),
        (lambda t: t.update({"lm_head.weight": torch.zeros(128, 64)}), "lm_head"),
    ],
)
def test_checkpoint_mismatches_fail_loudly(
    tiny_qwen2: Any, tmp_path: Path, mutate: object, message: str
) -> None:
    tensors = _tensors(tiny_qwen2)
    mutate(tensors)  # type: ignore[operator]
    model_dir = _copy_checkpoint(tiny_qwen2.model_dir, tmp_path / "bad", tensors)

    with pytest.raises(ValueError, match=message):
        load_qwen2_weights(
            model_dir,
            load_qwen2_spec(model_dir),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_sharded_checkpoint_loads(tiny_qwen2: Any, tmp_path: Path) -> None:
    tensors = _tensors(tiny_qwen2)
    names = sorted(tensors)
    shards = {"a.safetensors": names[: len(names) // 2], "b.safetensors": names[len(names) // 2 :]}
    model_dir = tmp_path / "sharded"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        (tiny_qwen2.model_dir / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    for shard, keys in shards.items():
        save_file({key: tensors[key] for key in keys}, str(model_dir / shard))
    index = {"weight_map": {key: shard for shard, keys in shards.items() for key in keys}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    weights = load_qwen2_weights(
        model_dir, load_qwen2_spec(model_dir), device=torch.device("cpu"), dtype=torch.float32
    )

    torch.testing.assert_close(weights.embed, tensors["model.embed_tokens.weight"])


def test_resolve_model_dir_never_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    with pytest.raises(ValueError, match="not in the local Hugging Face cache"):
        resolve_model_dir("tinyinfer/does-not-exist", "0" * 40)
