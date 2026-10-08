# TinyInfer Owned Paged Attention Design（阶段 C）

> 日期：2026-10-08
>
> 状态：proposed design，等待评审
>
> 设计基线：`main` at `e636ea4`（G3 verified locally）
>
> 目标阶段：v0.3 Phase 3 后半段（"Attention reads paged KV directly"）

## 1. Objective

让 TinyInfer 自己的 Qwen2 前向直接在共享的 paged KV 池上运行，并接入现有的合批引擎。

G2-P1 测出的问题是：当前 paged 路径在每个 decode 步、每一层，用 Python 循环把每个 block
逐个拷进连续张量。每个 block 每步约 0.64 ms，导致 612-token prompt 的 decode 比 Hugging Face
慢 2.7 倍，长 prompt 合批甚至慢于串行。

阶段 C 完成后：

1. 每层每步读取 KV 只需**固定次数**的 GPU 操作，与 batch 大小和上下文长度无关；
2. 合批引擎可以选择后端：现有的 HF 适配路径（G2-P1 基线）或 owned 路径，调度逻辑完全共享；
3. owned 路径的输出与 Hugging Face 一致：B=1 逐 token 相同，批处理沿用 BF16 平局规则。

性能是否真的改善，由阶段 D 在同一套 G2-P1 workload 上重测决定；本阶段只证明正确，不声明提速。

## 2. Non-goals

- 不写 Triton 或 CUDA kernel（阶段 F）。
- 不做按需分配、准入控制、抢占（G2-P2）；仍在请求进入时预留全部 block。
- 不做批量 prefill 或 chunked prefill；prefill 仍逐个请求串行执行。
- 不修改 `tinyinfer/cache/` 下现有的 paged 实现，保证 G2-P1 基线可复现。

## 3. 关键设计

### 3.1 新池子的内存布局：按 token 排列

现有池子布局是 `[layers, 2, blocks, kv_heads, block_size, head_dim]`。一个 block 内，同一个 token
的不同 head 不相邻，因此无法用"一个 token 一个下标"的方式整体取出。

新池子改为：

```text
[layers, 2, num_blocks, block_size, kv_heads, head_dim]
```

这样每层的 K（或 V）可以看成一张扁平的表 `[num_blocks × block_size, kv_heads, head_dim]`，
其中每一行是一个 **slot**（槽位），存一个 token 的全部 KV head。第 `p` 号物理 block 中第 `o` 个
位置的 slot 编号是 `p × block_size + o`。这和 vLLM 的 slot mapping 思路一致。

### 3.2 向量化 gather 与写入

每个 decode 步开始时，按本轮 batch 的顺序一次性构建两个下标张量（放在 GPU 上，24 层共用）：

```text
read_slots  [B, S_max]   每行是该请求全部历史 token 的 slot 编号，短的行用 null slot 补齐
write_slots [B × T]      本步新 token 要写入的 slot 编号
```

每层只需要：

```python
pool_k[layer][write_slots] = new_k        # 1 次写入
k = pool_k[layer][read_slots]             # 1 次 gather → [B, S_max, Hkv, D]
```

K 和 V 各一次，所以每层每步固定 4 个索引操作，不再随 block 数增长。

### 3.3 Null block：padding 必须读到有限值

batch 中较短的行要补齐到 `S_max`。补齐位置虽然会被 attention mask 屏蔽，但 SDPA 仍会计算
`0 × V[padding]`。如果 padding 读到的是未初始化或残留的 NaN，`0 × NaN = NaN`，会污染整行输出。

因此**保留物理 block 0 作为 null block**，内容恒为 0，永不分配；所有 padding 下标都指向它。
测试会用 NaN 填满所有空闲 block，以证明 padding 从不读到它们。

### 3.4 `PagedKV` 实现现有 `KVCache` 协议

owned 前向已经通过 `KVCache` 协议（`update` / `commit` / `length`）读写 KV，因此模型代码
**不需要修改**，只需要新增一个实现：

```python
class PagedKVPool:        # 长期存在：显存池 + 复用现有 BlockAllocator（block 0 预留）
class PagedSequence:      # 每个请求：物理 block 列表 + 已提交长度
class PagedBatchView:     # 每个 forward 临时创建：本轮 batch 的 read/write slot 与 mask，实现 KVCache
```

- 对新请求的 prefill（没有历史）：`update` 直接返回本步的 K/V，不经过 gather。这样 prefill 的
  attention 与 contiguous 路径逐位一致。
- `commit` 只在最后一层之后推进所有序列的长度，沿用"所有层写完才提交"的语义。

### 3.5 Mask 与 SDPA 分发

- batch 中只有 1 个请求时，不构造 mask（`mask=None`），走与 Hugging Face 相同的 SDPA 路径，
  预期逐位一致。
- 多个请求时，构造布尔 mask `[B, 1, 1, S_max]`，补齐位置为 False。此时 SDPA 走 `repeat_kv` +
  mask 路径，与 B=1 的 kernel 不同，按 BF16 平局规则验收。

### 3.6 引擎：抽出后端接口，调度逻辑共享

现有 `BatchEngine` 的调度部分（接纳、结束判定、取消、异常清理、统计、observer）与"怎么跑模型"
混在一起。抽出一个后端协议：

```python
class DecodeBackend(Protocol):
    def open(self, request_id: str, prompt_ids: Sequence[int], max_new_tokens: int) -> Handle: ...
    def prefill(self, handle: Handle) -> int: ...                      # 返回第 1 个 token
    def decode(self, handles: Sequence[Handle], last_tokens: Sequence[int]) -> list[int]: ...
    def seq_length(self, handle: Handle) -> int: ...
    def release(self, handle: Handle) -> None: ...                    # 幂等
```

- `HFPagedBackend`：把现有 `_init_request_state` / `_prefill_request` / `_decode_batch` 原样搬进来，
  行为不变，现有测试全部保留。
- `OwnedPagedBackend`：用 `Qwen2Model` + `PagedKVPool`。
- `BatchEngine(backend=...)` 只负责调度。G2-P1 benchmark 增加 `--backend` 维度，供阶段 D 使用。

## 4. 顺带修复

- **显存峰值重复计入**：G2-P1 报告指出，benchmark 在旧运行对象仍存活时分配新池。阶段 D 重测前，
  在每次运行前释放上一次的执行对象，并加测试锁定。
- **G3 中 19 个非逐位一致的对照点**：在 GPU 上补一次分层诊断，确认差异来源（预期为 SDPA kernel
  选择差异）。只记录，不阻塞本阶段。

## 5. Testing Strategy

按 TDD 推进。

### 5.1 CPU 单元测试（tiny random Qwen2）

| 测试 | 标准 |
|---|---|
| slot 计算：跨 block 边界、多请求、补齐指向 null slot | 精确 |
| `PagedBatchView` 写入后 gather 回来的 K/V 与写入值相同 | 精确 |
| null block 不被分配；空闲 block 填 NaN 时输出全部有限 | 精确 |
| 单请求 paged 路径 vs contiguous 路径的 logits（prefill + 多步 decode，block_size=4 以覆盖跨块） | FP32 `atol=1e-5`；BF16 逐位一致 |
| 3 个长度不同的请求合批 decode vs 各自单独运行 | FP32 `atol=1e-5` |
| `HFPagedBackend` 重构后，现有引擎测试全部通过 | 不改测试 |
| `OwnedPagedBackend` 驱动 `BatchEngine`：动态到达、提前结束、取消后 block 全部回收 | 精确 |

### 5.2 GPU 集成测试（真实 Qwen2.5-0.5B）

| 测试 | 标准 |
|---|---|
| owned 后端跑 G1 四个 case（B=1） | 与 HF `generate()` 逐 token 一致 |
| owned 后端跑 G2-BD 三个异长请求（真实 `[3, 1]` decode） | 每个请求与 HF 一致，或在 BF16 平局处分歧并记录 |
| 重复运行后池子完全回收 | 精确 |

## 6. Gate C

- 5.1 与 5.2 全部通过；pytest、Ruff、Mypy、CI 全绿；
- 每层每步的 KV 读写为固定次数的索引操作（代码审查确认，无按 block 的 Python 循环）；
- `tinyinfer/cache/` 现有实现与 G2-P1 证据文件未被修改。

通过后允许声明：

> TinyInfer's owned forward pass runs continuous batching on a shared paged KV pool with
> vectorized slot gather, matching Hugging Face.

仍禁止声明任何性能提升，直到阶段 D 的重测数据支持。

## 7. Risks

| 风险 | 应对 |
|---|---|
| gather 产生 `[B, S_max, Hkv, D]` 的临时张量，长上下文时显存和带宽开销仍不小 | 这正是阶段 F（block-native kernel）要解决的；阶段 D 会量化它 |
| 补齐到 `S_max` 时，长短差异大的 batch 浪费计算 | 记录在阶段 D 的分析中，不在本阶段优化 |
| 引擎重构破坏现有行为 | 先搬代码、不改测试，全部通过后再加 owned 后端 |
| 新旧两套 paged 实现并存，增加理解成本 | 文档明确：旧实现仅用于复现 G2-P1 基线，新功能只加在新实现上 |

## 8. Review Decisions

1. 是否同意新建按 token 排列的池子，而不修改 `tinyinfer/cache/` 的旧实现？
2. 是否同意抽出 `DecodeBackend` 接口，让新旧两种后端共享同一个调度器？
3. 是否同意保留 block 0 作为 null block？
