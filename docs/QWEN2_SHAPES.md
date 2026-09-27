# Qwen2.5-0.5B 张量形状表

> 由 `scripts/render_qwen2_shapes.py` 在 meta device 上运行 TinyInfer 自己的前向生成，
> `tests/unit/test_qwen2_shapes_doc.py` 保证它与代码一致。请勿手工修改。

## 模型常数

| 符号 | 含义 | 值 |
|---|---|---:|
| V | 词表大小 | 151936 |
| C | hidden 维度 | 896 |
| I | MLP 中间维度 | 4864 |
| L | decoder 层数 | 24 |
| H | query head 数 | 14 |
| Hkv | KV head 数（每个服务 7 个 query head） | 2 |
| D | head 维度 | 64 |
| θ | RoPE base | 1e+06 |

## 前向每一步

B = 1。prefill 输入 T = 9 个 token（G1 `short_en` 的长度）；decode 输入 T = 1，attention 看到 S = 10 个 token。`layer.*` 每层重复一次，共 L 次。

| 步骤 | 含义 | prefill | decode | dtype |
|---|---|---|---|---|
| `embedding` | token id 查表，每个 token 得到一个 hidden 向量 | `[1, 9, 896]` | `[1, 1, 896]` | bfloat16 |
| `rope.cos` | 每个位置的旋转角余弦（FP32 计算后转 BF16） | `[1, 9, 64]` | `[1, 1, 64]` | bfloat16 |
| `rope.sin` | 每个位置的旋转角正弦 | `[1, 9, 64]` | `[1, 1, 64]` | bfloat16 |
| `layer.input_norm` | attention 前的 RMSNorm | `[1, 9, 896]` | `[1, 1, 896]` | bfloat16 |
| `layer.q` | query，14 个 head | `[1, 14, 9, 64]` | `[1, 14, 1, 64]` | bfloat16 |
| `layer.k` | 本步新 token 的 key，只有 2 个 KV head（GQA） | `[1, 2, 9, 64]` | `[1, 2, 1, 64]` | bfloat16 |
| `layer.v` | 本步新 token 的 value | `[1, 2, 9, 64]` | `[1, 2, 1, 64]` | bfloat16 |
| `layer.k_history` | attention 看到的全部 key：cache 历史 + 本步 | `[1, 2, 9, 64]` | `[1, 2, 10, 64]` | bfloat16 |
| `layer.attention` | 14 个 head 的输出拼接 | `[1, 9, 896]` | `[1, 1, 896]` | bfloat16 |
| `layer.o_proj` | 投影回 hidden 维度，随后与残差相加 | `[1, 9, 896]` | `[1, 1, 896]` | bfloat16 |
| `layer.post_norm` | MLP 前的 RMSNorm | `[1, 9, 896]` | `[1, 1, 896]` | bfloat16 |
| `layer.mlp` | SwiGLU MLP 输出（中间维度见参数表） | `[1, 9, 896]` | `[1, 1, 896]` | bfloat16 |
| `layer.output` | 本层输出（两次残差之后） | `[1, 9, 896]` | `[1, 1, 896]` | bfloat16 |
| `final_norm` | 最后一层之后的 RMSNorm | `[1, 9, 896]` | `[1, 1, 896]` | bfloat16 |
| `logits` | 每个位置对整个词表的打分 | `[1, 9, 151936]` | `[1, 1, 151936]` | bfloat16 |

## 参数

| 张量 | 形状 | 参数量 |
|---|---|---:|
| `model.embed_tokens.weight` | `[151936, 896]` | 136,134,656 |
| `model.norm.weight` | `[896]` | 896 |
| `model.layers.0.input_layernorm.weight` | `[896]` | 896 |
| `model.layers.0.self_attn.q_proj.weight` | `[896, 896]` | 802,816 |
| `model.layers.0.self_attn.q_proj.bias` | `[896]` | 896 |
| `model.layers.0.self_attn.k_proj.weight` | `[128, 896]` | 114,688 |
| `model.layers.0.self_attn.k_proj.bias` | `[128]` | 128 |
| `model.layers.0.self_attn.v_proj.weight` | `[128, 896]` | 114,688 |
| `model.layers.0.self_attn.v_proj.bias` | `[128]` | 128 |
| `model.layers.0.self_attn.o_proj.weight` | `[896, 896]` | 802,816 |
| `model.layers.0.post_attention_layernorm.weight` | `[896]` | 896 |
| `model.layers.0.mlp.gate_proj.weight` | `[4864, 896]` | 4,358,144 |
| `model.layers.0.mlp.up_proj.weight` | `[4864, 896]` | 4,358,144 |
| `model.layers.0.mlp.down_proj.weight` | `[896, 4864]` | 4,358,144 |

- 每层参数：14,912,384；共 24 层。
- 总参数：494,032,768（`lm_head` 与 embedding 共享，只算一次）；BF16 约 0.99 GB，decode 每步都要读一遍。
- KV cache：每个 token 占 2（K 和 V）× 24 层 × 2 × 64 × 2 字节 = 12,288 字节。
