# TinyInfer Owned Qwen2 Forward Design

> 日期：2026-09-27
>
> 状态：proposed design，等待评审
>
> 设计基线：`main` at `221be43`（G2-P1 verified locally）
>
> 目标阶段：v0.3 Phase 3 / Gate G3
>
> 适用模型：Qwen/Qwen2.5-0.5B @ `060db6499f32faf8b98477b0a26969ef7d8b9987`，Transformers 5.14.1 作为参照

## 1. Objective

用 TinyInfer 自己的代码实现 Qwen2 前向传播，Hugging Face 只提供 tokenizer、权重文件和对照结果。

完成后必须能证明：

1. 每个算子、每一层 hidden state、最终 logits 都与 Hugging Face 对齐；
2. 带 KV cache 的 greedy 生成与 Hugging Face `generate()` 逐 token 一致；
3. 模型的 attention 调用由 TinyInfer 控制，为 Phase 3 后半段（attention 直接读 paged KV）和
   Phase 5（Triton kernel）留出接口。

这一步主要服务"理解推理"：每个张量的形状、dtype 和计算顺序都写在 TinyInfer 的代码里，而不是
藏在 `transformers` 内部。

## 2. Non-goals

- 不修改 `BatchEngine`、paged cache 或 G2-P1 benchmark；接入它们属于阶段 C。
- 不实现异长 batch、paged attention、Triton kernel。
- 不支持其他模型族、其他尺寸、sliding window、RoPE scaling、sampling。
- 不做性能声明；本阶段只证明正确性。

## 3. Qwen2 数值细节（来自 Transformers 5.14.1 源码）

对齐成败取决于这些细节，实现必须逐条照做：

| 组件 | 行为 |
|---|---|
| Embedding | `embed_tokens[input_ids]`，不缩放 |
| RMSNorm | 先转 fp32，`x * rsqrt(mean(x²) + eps)`，转回输入 dtype 后**再**乘 `weight` |
| q/k/v 投影 | 带 bias；o 投影不带 bias |
| RoPE | `inv_freq = 1 / theta^(arange(0, d, 2) / d)` 在 fp32 计算；`cos/sin` 在 fp32 计算后转成输入 dtype；`rotate_half` 取前后两半 |
| Attention | SDPA，`scale = head_dim^-0.5` |
| GQA 分发 | 无 mask 时用 `enable_gqa=True`，`q_len > 1` 时 `is_causal=True`；有 mask 时先 `repeat_kv` 再传 bool mask |
| MLP | `down(silu(gate(x)) * up(x))`，无 bias |
| Decoder 层 | pre-norm 残差：`x + attn(norm1(x))`，再 `x + mlp(norm2(x))` |
| 输出 | 最后一层后 `norm`，`lm_head` 与 `embed_tokens` 共享权重 |

Qwen2.5-0.5B 形状：vocab 151936，hidden 896，intermediate 4864，24 层，14 个 query head，
2 个 KV head（每个 KV head 服务 7 个 query head），head_dim 64，`rope_theta` 1e6，`rms_norm_eps` 1e-6。

为了在 BF16 下尽量逐位一致，实现要**复刻上表的运算顺序和 SDPA 分发条件**，而不是写一个数学上
等价但顺序不同的版本。

## 4. Architecture

```text
tinyinfer/model/
├── spec.py       Qwen2Spec：从 config.json 读取并校验结构参数
├── weights.py    从 safetensors 读取权重 → Qwen2Weights（纯张量 dataclass）
├── ops.py        纯函数：rms_norm, rope_tables, apply_rope, attention, swiglu_mlp
├── kv.py         KVCache 协议 + 连续实现（复用 tinyinfer/cache/contiguous.py）
├── qwen2.py      Qwen2Model.forward：embedding → 24 层 → norm → logits
└── generate.py   greedy prefill/decode 循环，返回现有的 GenerationResult
```

### 4.1 风格：纯函数 + 权重 dataclass，不用 `nn.Module`

每一层写成显式函数调用，权重是普通张量字段。好处：

- 读代码就能看到完整数据流，没有 hook、注册、自动广播之类的隐藏行为；
- 单元测试可以对单个函数喂张量，直接和 Hugging Face 对应模块对比；
- 阶段 C 替换 attention 时只改一个函数参数。

### 4.2 `Qwen2Spec`

从 `config.json` 构造的不可变 dataclass。遇到不支持的配置立即报错，而不是悄悄算错：
`hidden_act != "silu"`、`use_sliding_window`、`rope_scaling` 非空、`layer_types` 含非 full attention、
`hidden_size % num_attention_heads != 0`。

### 4.3 权重加载

- 用 `huggingface_hub.snapshot_download(model_id, revision=..., local_files_only=True)` 找到本地缓存目录，
  不联网；CPU 测试直接传入 `save_pretrained()` 生成的临时目录。
- 用 `safetensors` 逐个读取张量，按名字映射到 `Qwen2Weights`：

```text
model.embed_tokens.weight                         → embed
model.layers.{i}.input_layernorm.weight           → layers[i].input_norm
model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.post_attention_layernorm.weight  → layers[i].post_norm
model.layers.{i}.mlp.{gate,up,down}_proj.weight
model.norm.weight                                 → final_norm
lm_head.weight                                    → 仅当 tie_word_embeddings=False
```

- **严格校验**：缺少的键、多余的键、形状不符、tie 设置与文件不一致，都立即报错并列出键名。
- 加载时直接放到目标 device 和 dtype（BF16），不经过 `nn.Module`。

### 4.4 `ops.py`

```python
def rms_norm(x, weight, eps) -> Tensor
def rope_tables(position_ids, inv_freq, dtype) -> tuple[Tensor, Tensor]   # cos, sin
def apply_rope(q, k, cos, sin) -> tuple[Tensor, Tensor]
def attention(q, k, v, *, num_kv_groups, mask) -> Tensor                    # SDPA，按 §3 分发
def swiglu_mlp(x, gate_w, up_w, down_w) -> Tensor
```

每个函数的 docstring 写明输入输出形状，例如 `q: [B, 14, T, 64]`，`k: [B, 2, S, 64]`。

### 4.5 `KVCache` 协议

```python
class KVCache(Protocol):
    def length(self) -> int: ...
    def update(self, layer: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]: ...
    def commit(self, new_tokens: int) -> None: ...
```

- `update` 写入本层新 K/V，返回该层完整 K/V；`commit` 在最后一层之后统一推进长度，与现有
  `ContiguousKVCache` 的"所有层写完才提交"语义一致。
- 连续实现 `ContiguousKV` 按请求实际需要的长度分配，不再按 `max_position_embeddings`（32768）
  预分配。（实现时改为新写，而非包装 `tinyinfer/cache/contiguous.py`：后者只支持 B=1，且 prefill
  时逐层就修改长度，与"所有层写完才提交"不一致。）
- 阶段 C 提供 paged 实现，模型代码不变。

### 4.6 `Qwen2Model.forward`

```python
def forward(
    input_ids: Tensor,          # [B, T]
    position_ids: Tensor,       # [B, T]，调用方显式给出
    kv: KVCache | None,
    attention_mask: Tensor | None,   # None 表示无 padding 的因果注意力
    capture_hidden: bool = False,
) -> ForwardOutput:             # logits [B, T, V]；可选每层 hidden states
```

- `capture_hidden=True` 时返回 embedding 输出与每一层输出，共 25 个张量，另有经过最终 norm 的
  `final_hidden`，用于逐层对齐。Hugging Face `output_hidden_states=True` 返回 25 个张量，但最后一个
  是最终 norm 之后的结果，测试按此对应。
- `trace(name, tensor)` 回调在每一步记录张量，默认关闭、不改变计算；`docs/QWEN2_SHAPES.md` 由它在
  meta device 上生成。
- 本阶段只测 B=1 与等长 B>1；接口已为异长 batch 保留 `position_ids` 与 `attention_mask`。

### 4.7 `generate.py`

`greedy_generate(model, tokenizer, request) -> GenerationResult`：prefill 一次，之后每步输入一个 token，
EOS 或 `max_new_tokens` 停止。复用 `GenerationRequest`、`GenerationResult` 与 observer 协议，这样
G1 benchmark 以后可以直接加一个 owned 路径。

## 5. Testing Strategy

按 TDD 推进：每个行为先有失败的测试，再实现。

### 5.1 CPU 单元测试（不需要 GPU 和真实模型）

用随机初始化的小 `Qwen2ForCausalLM`（例如 2 层、hidden 64、4 个 query head、2 个 KV head），
`save_pretrained()` 到临时目录，FP32 运行：

| 测试 | 标准 |
|---|---|
| spec 解析与不支持配置报错 | 精确 |
| 权重加载：键映射、缺键/多键/形状错/tie 错报错 | 精确 |
| 每个算子对比 HF 对应模块 | `atol=1e-5, rtol=1e-4` |
| 每层 hidden state 与 logits 对比 HF | `atol=1e-5, rtol=1e-4` |
| 有 cache 的逐步 decode = 无 cache 的完整重算 | `atol=1e-5, rtol=1e-4` |
| greedy 16 个 token 与 HF `generate()` | 逐 token 一致 |
| 等长 B=2 与两次 B=1 | logits 一致 |

### 5.2 GPU 集成测试（`gpu and model`，真实 Qwen2.5-0.5B，BF16）

| 测试 | 标准 |
|---|---|
| G1 四个 canonical prompt 的逐层 hidden state | 每层 cosine ≥ 0.999，并报告最大绝对误差 |
| G1 四个 canonical case 的 greedy 输出 | 与 HF `generate()` 逐 token 一致 |

因为 §3 要求复刻运算顺序和 SDPA 分发，预期 BF16 下结果逐位一致。若出现分歧，沿用 G2-P1 已确立的
规则：只有两个 token 的 logit 都在最高值 1 个 BF16 单位以内才接受，并记录位置和差值；其他分歧都算失败。

### 5.3 学习产出

`docs/QWEN2_SHAPES.md`：一页张量形状表，从 `input_ids` 到 logits，每一步写出形状、dtype 和对应代码行。
由测试从真实前向中打印形状生成，保证与代码一致。

## 6. Gate G3（对应 plan v0.3）

- 5.1 与 5.2 全部通过；
- pytest、Ruff check、Ruff format、Mypy 全绿，CI 通过；
- `docs/QWEN2_SHAPES.md` 完成；
- 不修改任何历史证据文件。

G3 通过后允许声明：

> TinyInfer runs Qwen2.5-0.5B with its own forward pass and matches Hugging Face token-for-token.

仍禁止声明：

> TinyInfer's owned forward is faster than Hugging Face, or supports paged/batched attention.

## 7. Risks

| 风险 | 应对 |
|---|---|
| BF16 运算顺序与 HF 不同，导致平局翻转 | 严格复刻 §3；先在 FP32 CPU 上逐层对齐，再上 GPU |
| SDPA 在有无 mask 时选不同 kernel | 复刻 HF 的分发条件；B=1 无 padding 时传 `mask=None` |
| safetensors 键名与假设不符 | 严格键校验，报错时列出实际键名 |
| 同时加载 HF 与 owned 两份权重占显存 | 0.5B BF16 各约 1 GB，16 GB 显存足够；只在对齐测试中同时加载 |
| 范围扩张到 engine 集成 | 本阶段不改 `BatchEngine`，接入留给阶段 C |

## 8. Review Decisions

1. 是否同意用纯函数 + 权重 dataclass，而不是 `nn.Module`？
2. 是否同意自己从 safetensors 读取权重，而不是借用已加载 HF 模型的张量？
3. 是否同意本阶段不改 `BatchEngine`，把接入留给阶段 C？
