# The rendered document is Chinese; its full-width punctuation is intentional.
# ruff: noqa: RUF001
"""Render docs/QWEN2_SHAPES.md by tracing the owned forward pass on the meta device.

The meta device computes shapes and dtypes without memory or arithmetic, so the
table comes from the real code path at full Qwen2.5-0.5B size on any machine.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tinyinfer.model.kv import ContiguousKV
from tinyinfer.model.qwen2 import Qwen2Model
from tinyinfer.model.spec import QWEN25_05B
from tinyinfer.model.weights import checkpoint_shapes, empty_qwen2_weights

DOC = Path(__file__).resolve().parents[1] / "docs" / "QWEN2_SHAPES.md"
PROMPT_TOKENS = 9  # the G1 short_en prompt
META = torch.device("meta")

MEANINGS = {
    "embedding": "token id 查表，每个 token 得到一个 hidden 向量",
    "rope.cos": "每个位置的旋转角余弦（FP32 计算后转 BF16）",
    "rope.sin": "每个位置的旋转角正弦",
    "layer.input_norm": "attention 前的 RMSNorm",
    "layer.q": "query，14 个 head",
    "layer.k": "本步新 token 的 key，只有 2 个 KV head（GQA）",
    "layer.v": "本步新 token 的 value",
    "layer.k_history": "attention 看到的全部 key：cache 历史 + 本步",
    "layer.attention": "14 个 head 的输出拼接",
    "layer.o_proj": "投影回 hidden 维度，随后与残差相加",
    "layer.post_norm": "MLP 前的 RMSNorm",
    "layer.mlp": "SwiGLU MLP 输出（中间维度见参数表）",
    "layer.output": "本层输出（两次残差之后）",
    "final_norm": "最后一层之后的 RMSNorm",
    "logits": "每个位置对整个词表的打分",
}


def _trace(model: Qwen2Model, kv: ContiguousKV, tokens: int, start: int) -> dict[str, torch.Tensor]:
    seen: dict[str, torch.Tensor] = {}

    def record(name: str, tensor: torch.Tensor) -> None:
        seen.setdefault(name, tensor)  # layer.* repeats per layer; keep the first

    input_ids = torch.zeros(1, tokens, dtype=torch.long, device=META)
    positions = torch.arange(start, start + tokens, device=META)[None]
    model.forward(input_ids, positions, kv=kv, trace=record)
    return seen


def _shape(tensor: torch.Tensor) -> str:
    return f"`{list(tensor.shape)}`"


def render() -> str:
    spec = QWEN25_05B
    model = Qwen2Model(spec, empty_qwen2_weights(spec, device=META, dtype=torch.bfloat16))
    kv = ContiguousKV.for_spec(
        spec, batch=1, max_tokens=PROMPT_TOKENS + 1, device=META, dtype=torch.bfloat16
    )
    prefill = _trace(model, kv, PROMPT_TOKENS, 0)
    decode = _trace(model, kv, 1, PROMPT_TOKENS)  # reuses the prefill's cached K/V

    shapes = checkpoint_shapes(spec)
    layer_prefix = "model.layers.0."
    per_layer = sum(math.prod(s) for n, s in shapes.items() if n.startswith(layer_prefix))
    total = sum(math.prod(s) for s in shapes.values())
    kv_bytes = 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * 2

    lines = [
        "# Qwen2.5-0.5B 张量形状表",
        "",
        "> 由 `scripts/render_qwen2_shapes.py` 在 meta device 上运行 TinyInfer 自己的前向生成，",
        "> `tests/unit/test_qwen2_shapes_doc.py` 保证它与代码一致。请勿手工修改。",
        "",
        "## 模型常数",
        "",
        "| 符号 | 含义 | 值 |",
        "|---|---|---:|",
        f"| V | 词表大小 | {spec.vocab_size} |",
        f"| C | hidden 维度 | {spec.hidden_size} |",
        f"| I | MLP 中间维度 | {spec.intermediate_size} |",
        f"| L | decoder 层数 | {spec.num_layers} |",
        f"| H | query head 数 | {spec.num_heads} |",
        f"| Hkv | KV head 数（每个服务 {spec.num_kv_groups} 个 query head） "
        f"| {spec.num_kv_heads} |",
        f"| D | head 维度 | {spec.head_dim} |",
        f"| θ | RoPE base | {spec.rope_theta:g} |",
        "",
        "## 前向每一步",
        "",
        f"B = 1。prefill 输入 T = {PROMPT_TOKENS} 个 token（G1 `short_en` 的长度）；"
        f"decode 输入 T = 1，attention 看到 S = {PROMPT_TOKENS + 1} 个 token。"
        "`layer.*` 每层重复一次，共 L 次。",
        "",
        "| 步骤 | 含义 | prefill | decode | dtype |",
        "|---|---|---|---|---|",
    ]
    for name, meaning in MEANINGS.items():
        dtype = str(prefill[name].dtype).removeprefix("torch.")
        lines.append(
            f"| `{name}` | {meaning} | {_shape(prefill[name])} | {_shape(decode[name])} | {dtype} |"
        )
    lines += [
        "",
        "## 参数",
        "",
        "| 张量 | 形状 | 参数量 |",
        "|---|---|---:|",
    ]
    for name, shape in shapes.items():
        if name.startswith("model.layers.") and not name.startswith(layer_prefix):
            continue
        lines.append(f"| `{name}` | `{list(shape)}` | {math.prod(shape):,} |")
    lines += [
        "",
        f"- 每层参数：{per_layer:,}；共 {spec.num_layers} 层。",
        f"- 总参数：{total:,}（`lm_head` 与 embedding 共享，只算一次）；BF16 约 "
        f"{total * 2 / 1e9:.2f} GB，decode 每步都要读一遍。",
        f"- KV cache：每个 token 占 2（K 和 V）× {spec.num_layers} 层 × {spec.num_kv_heads} × "
        f"{spec.head_dim} × 2 字节 = {kv_bytes:,} 字节。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    DOC.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
