# TinyInfer True Batched Decode Design

> 日期：2026-08-10
>
> 状态：approved design，等待实现计划审阅
>
> 设计基线：`main` at `0c46afc05d790bfe0c78f8e51250ee93d3bd8542`
>
> 目标阶段：Phase 2 / G2
> 适用模型：固定 revision 的 Qwen2.5-0.5B，Transformers 5.14.1

## 1. Objective

将当前“一个调度轮中逐请求执行多个 `B=1` decode forward”的原型，升级为真正的
iteration-level continuous batching：每个调度轮把所有可 decode 的 running request 合并成
一个 `[B, 1]` 输入，只执行一次 `model(...)` forward，同时继续使用 TinyInfer 管理的共享
paged KV 物理块池和逐请求 block table。

完成后，TinyInfer 必须能用三个历史长度不同的请求证明：

1. 同一轮 decode 的实际模型输入 batch size 大于 1；
2. 同一轮只有一次 decode forward，而不是 Python 循环中的多个单请求 forward；
3. 每个请求的输出分别与 Hugging Face reference 逐 token 完全一致；
4. 请求完成、取消和新请求进入时，batch 可动态缩放且 KV block 不泄漏；
5. 文档只声明已有证据能够支持的能力，不把 dense-gather 原型描述为 block-native paged
   attention。

## 2. Current State and Problem Statement

当前代码已经具备以下真实能力：

- `PagedKVCache` 提供共享物理块池、free list 和逐请求 `BlockTable`；
- `PagedTinyinferCache` 能通过 Hugging Face Cache 协议拦截 Qwen attention 的 KV 更新；
- `RequestQueue` 能管理 waiting、running、finished 和 cancelled 状态；
- `BatchEngine` 能在一个调度循环内推进多个请求；
- GPU integration test 能验证三个请求各自与 reference token 对齐。

但当前 [`BatchEngine.step()`](../../../tinyinfer/scheduler/engine.py) 仍遍历
`running_requests`，每个请求调用一次 `_decode_request()`，因此实际执行的是：

```text
request A -> model(input=[1, 1])
request B -> model(input=[1, 1])
request C -> model(input=[1, 1])
```

现有输出对齐测试只能证明多个请求都能被正确推进，不能证明存在真正的 batched model
path。当前能力应称为 **iteration interleaving prototype**，而不是 verified continuous
batching。

此外，2026-08-10 的独立基线审查还发现：

- CPU unit suite 有 2 个失败；
- Ruff check 有 21 个错误，format check 有 15 个文件不合格；
- Mypy 有 3 个错误；
- 最新 main CI 因 Ruff 失败；
- allocator 的部分分配失败缺少 rollback，取消 running request 可能遗留资源；
- 当前 GPU batch test 没有记录 forward 次数和实际 batch shape。

这些是进入完整 G2 Gate 前必须清零的已知基线问题。这里记录的是设计输入，不是新的
Gate 通过声明；实现阶段必须在目标 revision 上重新生成证据。

## 3. Non-goals

本设计明确不包含：

- 不实现 Triton paged attention kernel；
- 不实现 block-native attention；
- 不重写 Qwen decoder layer、MLP、RoPE 或完整 attention；
- 不实现 chunked prefill 或 batched prefill；
- 不实现 prefix caching、beam search、sampling 或 speculative decoding；
- 不实现优先级、抢占、swap 或分布式调度；
- 不实现 API server；
- 不以一次输出对齐代替性能、资源生命周期和 forward-count 证据；
- 不在没有 benchmark 数据时声明吞吐已经提升；
- 不把 G2 batched correctness checkpoint 自动等同于完整 G2 verified。

## 4. Transformers Constraint

Transformers 5.14.1 中，Qwen attention 在每层执行：

```python
key_states, value_states = past_key_values.update(
    key_states,
    value_states,
    layer_idx,
)
```

输入 KV 天然带 batch 维度：

```text
[batch_size, num_kv_heads, query_length, head_dim]
```

但是 Cache mask 协议只提供 batch 共享的标量：

```python
get_seq_length() -> int
get_query_offset(layer_idx) -> int
get_mask_sizes(query_length, layer_idx) -> tuple[int, int]
```

因此，不能仅仅把现有单请求 `PagedTinyinferCache.update()` 改成接受 `B > 1` 就宣称支持
异长 batch。设计必须同时定义：

- 每一行写入哪个 request 的 block table；
- 每一行使用哪个逻辑 token position；
- gather 后不同长度的 KV 如何对齐；
- padding slot 如何被 mask；
- 每一行 RoPE 使用哪个 `position_id`；
- Cache 协议需要返回什么共享标量。

## 5. Considered Approaches

### 5.1 Length Bucketing

把历史长度相同的请求放入同一 batch。实现较简单，因为 Cache 的标量长度对整个 batch
有效。

拒绝原因：三个不同长度的请求会被拆成多个 forward，不能满足本阶段“异长请求同一
decode forward”的验收目标，也会在真实 continuous batching 中严重削弱合批机会。

### 5.2 Custom Block-native Qwen Attention

让 Qwen attention 直接消费物理 KV pool、block tables 和逐行长度，不再 gather dense KV。

暂缓原因：这会把当前工作扩展到自定义 attention 或 Triton kernel，明显超过本阶段范围。
它应在 Phase 3 由 profiler 数据驱动，而不是作为 continuous batching 的前置条件。

### 5.3 Left-padded Dense Gather — Selected

共享 paged pool 继续作为 KV 的 source of truth。每层 attention 前，按本轮 active batch 的
request state gather KV，并左填充为统一长度的 dense tensor，交给 Hugging Face 原生
attention。

优点：

- 能在一次真实 model forward 中处理异长请求；
- 不修改 Qwen 模型主体；
- 保持逐请求 block table 和共享物理池；
- correctness 可独立验证；
- 后续可用 Triton gather kernel 替换 Python/PyTorch gather，而不改变 scheduler contract。

限制：

- attention 仍读取 dense KV；
- gather 会产生临时 tensor 和复制开销；
- 这不是 vLLM 式 block-native paged attention；
- 是否带来吞吐提升必须由 benchmark 决定。

## 6. Architecture

```text
                         long-lived
                 +-----------------------+
                 |     PagedKVCache      |
                 | physical block pool   |
                 | allocator / free list |
                 +-----------+-----------+
                             |
              +--------------+---------------+
              |              |               |
      +-------v------+ +-----v--------+ +----v---------+
      | Request A    | | Request B    | | Request C    |
      | block_table  | | block_table  | | block_table  |
      | seq_length=5 | | seq_length=8 | | seq_length=11|
      +-------+------+ +-----+--------+ +----+---------+
              |              |               |
              +--------------+---------------+
                             |
                         per iteration
                 +-----------v----------------+
                 | BatchedPagedTinyinferCache |
                 | active request state view  |
                 | batched write + left gather|
                 +-----------+----------------+
                             |
                 key/value [B, Hkv, maxL+1, D]
                             |
                 +-----------v-----------+
                 | one Qwen decode call  |
                 | input_ids [B, 1]      |
                 +-----------------------+
```

设计包含三个不同生命周期的对象。

### 6.1 `PagedKVCache`: Long-lived Shared Pool

物理 layout 保持：

```text
[num_layers, 2, num_physical_blocks, num_kv_heads, block_size, head_dim]
```

这里的第三维是 physical block，不是 request batch。不能给物理池增加固定 batch 维，因为
batch membership 会在每轮变化。

低层新增或重构为 batch-aware I/O：

```python
def write_decode_batch(
    key: Tensor,                       # [B, Hkv, 1, D]
    value: Tensor,                     # [B, Hkv, 1, D]
    layer_idx: int,
    block_tables: Sequence[BlockTable],
    positions: Sequence[int],
) -> None: ...

def gather_batch(
    layer_idx: int,
    block_tables: Sequence[BlockTable],
    lengths: Sequence[int],
    *,
    padding_side: Literal["left"] = "left",
) -> tuple[Tensor, Tensor]: ...
```

首版允许在 `write_decode_batch()` 和 `gather_batch()` 内部按 batch row 使用 Python 循环。
本阶段“真正 batched”的判据是一次 model forward，不是假装已有向量化 paged gather。
后续 profiling 若表明 gather 是瓶颈，再用 Triton 替换低层实现。

所有方法必须先完整验证 batch size、shape、dtype、device、table 数量、position 和映射，再
开始写入。不能在第 `i` 行失败后留下不可解释的部分状态。

### 6.2 `PagedRequestState`: Per-request Logical State

单请求的 block table 和长度必须从单请求 HF adapter 中抽离，成为 scheduler 可持有的明确
状态：

```python
@dataclass(slots=True)
class PagedRequestState:
    request_id: str
    block_table: BlockTable
    seq_length: int
    blocks_allocated: int
    released: bool = False
```

约束：

- `seq_length` 表示已经在所有模型层提交完成的 token 数；
- 一个 decode forward 开始时的写入位置等于该请求的旧 `seq_length`；
- 只有最后一层 update 成功后才能提交新的 `seq_length`；
- `released=True` 后任何 read/write 必须失败；
- release 必须幂等，不能 double-free；
- scheduler、单请求 prefill adapter 和 batched decode adapter 共享同一个 state 对象。

### 6.3 `PagedTinyinferCache`: Single-request Prefill Adapter

现有 adapter 保留单请求职责，用于 prefill 和单请求 correctness path，但不再私有持有一份
无法共享的 `_length`。它改为引用 `PagedRequestState`。

这样 serial prefill 完成后，batched adapter 能直接读取同一 state 的 `block_table` 和
`seq_length`，不需要复制 cache 或从另一个 adapter 猜测长度。

其 Cache 协议必须修正为：

```python
get_seq_length() == state.seq_length
get_query_offset(layer_idx) == state.seq_length
get_mask_sizes(q_length, layer_idx) == (state.seq_length + q_length, 0)
```

`get_query_offset()` 不得继续无条件返回 `0`。

### 6.4 `BatchedPagedTinyinferCache`: Ephemeral Forward View

该对象只代表本轮 decode 的 active request ordering：

```python
BatchedPagedTinyinferCache(
    paged: PagedKVCache,
    states: Sequence[PagedRequestState],
)
```

它不拥有物理池、不分配 block，也不长期注册所有请求。下一轮 batch membership 改变时创建
新的 view，避免在 Cache 对象内部复制 scheduler 状态。

构造时必须冻结本轮顺序，并计算：

```text
old_lengths = [state.seq_length for state in states]
max_past_length = max(old_lengths)
```

Cache 协议返回：

```python
get_seq_length() == max_past_length
get_query_offset(layer_idx) == max_past_length
get_mask_sizes(1, layer_idx) == (max_past_length + 1, 0)
```

首版只接受 decode query length `1`。如果 `update()` 收到 `query_length != 1`，必须明确抛出
异常，而不是静默产生错误 mask。

每层 `update()` 执行：

1. 验证输入为 `[B, Hkv, 1, D]` 且 `B == len(states)`；
2. 用每行对应的 block table 写到 `old_lengths[row]`；
3. 按 `new_lengths = old_lengths + 1` gather 每行 KV；
4. 左填充为 `[B, Hkv, max_past_length + 1, D]`；
5. 返回 dense key/value 给 Hugging Face attention；
6. 只有最后一层成功时，为所有 state 提交 `seq_length += 1`；
7. 防止同一 adapter 的同一 layer 被重复 update 或 layer 顺序异常。

如果中间层 forward 失败，state 的 committed length 保持不变。已经写到目标 slot 的 KV 可以
在重试时被同一 position 覆盖，但不得把请求暴露成“部分层已经增长”的状态。

## 7. Variable-length Layout and Mask Semantics

假设旧历史长度为：

```text
A = 5, B = 8, C = 11, max_past_length = 11
```

本轮 decode 写入后，gather 结果的 sequence 维长度统一为 12：

```text
A KV: [pad x6][A0 A1 A2 A3 A4 A5]
B KV: [pad x3][B0 B1 B2 B3 B4 B5 B6 B7 B8]
C KV: [C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11]
```

这里 `A5/B8/C11` 是本轮新 token 的 key/value，始终位于 dense tensor 最后一列。

Engine 同时构造：

```python
input_ids.shape == (3, 1)

position_ids = tensor([
    [5],
    [8],
    [11],
])

attention_mask = tensor([
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])
```

正确性依据：

- Qwen RoPE 使用显式逐行 `position_ids`，因此每个请求的新 query/key 使用自己的真实逻辑
  position；
- 历史 key 在生成时已经使用各自真实 position 做过 RoPE；
- batch 共享的 `q_offset=max_past_length` 只负责 dense causal coordinate；
- 左 padding 让所有新 query 都位于 dense KV 的最后一列；
- 逐行 attention mask 排除左侧 padding slot；
- 因此每行只关注自己的有效历史和当前 token，不会读取别的 request，也不会读取 padding。

Engine 必须显式传入 `position_ids`。`cache_position` 在本固定 Transformers 版本的 causal mask
入口中已 deprecated/unused；batched path 不依赖它表达逐行长度。

## 8. Scheduler Semantics

`BatchEngine.step()` 的目标顺序：

1. 回收已经被取消但仍持有资源的请求；
2. 按 FCFS 和 `max_batch` 接纳 waiting request；
3. 为新请求原子分配 blocks，失败时 rollback；
4. 对新请求执行 serial prefill，产生第一个输出 token；
5. 立即完成已达到 EOS 或 `max_new_tokens` 的请求；
6. 按稳定的 request ordering 收集仍需继续生成的 active requests；
7. 构造一次 `[B, 1]` decode 输入、逐行 `position_ids` 和左 padding mask；
8. 创建一个 `BatchedPagedTinyinferCache`；
9. 在 `torch.inference_mode()` 中只调用一次 `model(...)`；
10. 按 batch row 抽取 logits，并映射回对应 request；
11. 分别追加 token、判断停止条件并释放完成请求；
12. 返回本轮完成数量和可观测统计。

新接纳请求在 prefill 后可以参加同一 `step()` 的 decode，延续当前行为；但如果它已经因为
EOS 或 token limit 完成，则不能进入 batched decode。

本阶段允许 serial prefill。设计不得用“prefill 仍串行”否定 batched decode，也不得把它
描述成已经实现 batched/chunked prefill。

## 9. Resource Ownership and Failure Atomicity

### 9.1 Allocation

`allocate_blocks()` 必须先确认容量或记录已分配 blocks。任何中途失败都要：

- 将本次已经取得的 physical blocks 归还 allocator；
- 清除本次写入 block table 的映射；
- 保持 `free_block_count` 与调用前相同；
- 不创建可被 scheduler 看见的半初始化 request state。

### 9.2 Release

所有终止路径最终进入同一个 release helper：

- normal length completion；
- EOS completion；
- waiting cancellation；
- running cancellation；
- prefill failure；
- decode failure 后的 engine cleanup。

release 后：

- physical blocks 只归还一次；
- block table 映射清空；
- request state 标记 `released=True`；
- engine 删除对 adapter/state 的长期引用；
- repeated release 是 no-op 或受控异常，但不能 double-free。

### 9.3 Forward Failure

model forward 异常时，不得将任何 request 错误标记为生成了新 token。长度只有最后一层完成
后提交。Engine 可以终止该轮并向上抛出异常；是否取消整批请求由调用方处理，但资源必须有
明确的 cleanup 路径。

## 10. Public and Internal Interfaces

对外仍保留：

```python
BatchEngine.submit(request)
BatchEngine.step()
BatchEngine.run()
```

建议新增：

```python
BatchEngine.cancel(request_id: str) -> None
```

取消必须由 engine 协调 queue 状态和 cache lifecycle，不能只修改 `RequestQueue` 状态。

内部建议将当前无类型 `_StateEntry = dict[str, Any]` 替换为明确 dataclass，例如：

```python
@dataclass(slots=True)
class EngineRequestState:
    request: GenerationRequest
    cache: PagedRequestState
    attention_mask: Tensor
    output_token_ids: list[int]
```

具体命名可以在实现计划中调整，但不能继续依赖字符串 key 和 `Any` 隐式表达 cache ownership。

## 11. Observability

为了让测试和 benchmark 能证明真实 batching，Engine 至少暴露只读 iteration metrics：

```text
iteration_id
admitted_count
decode_batch_size
decode_forward_count
running_count_after_step
finished_count
cancelled_count
```

这些指标可以通过 observer、返回值扩展或内部 trace record 实现。不能依赖测试 monkeypatch
一个完全私有且不稳定的细节作为唯一证据。

生产代码不应为了测试保存完整 logits 或 KV 副本。

## 12. Testing Strategy

实现必须采用 TDD。每个行为先出现能够因缺失行为而失败的 RED test，再实现 GREEN。

### 12.1 CPU Unit Tests

#### Batched storage correctness

- `B=1` 和 `B=3` write/read shape 正确；
- 三个 request 使用不同 block table，不发生跨请求污染；
- 三个不同 position 写入后逐元素等于输入；
- 左 padding gather 的有效 token、padding 位置和输出 shape 正确；
- batch/table/position 数量不一致时 fail fast；
- dtype、device、head 数、head_dim、query length 不匹配时 fail fast；
- 已释放 state 禁止访问。

这些测试必须使用真实 tensor 或能验证复制内容、slice 和 identity 的可靠 test double。现有
`getitem()` 每次产生新 tensor、`copy_()` no-op 的 fake tensor 不能作为 KV 内容正确性的
唯一证据。

#### Cache protocol

- single cache 的 query offset 等于真实 seq length；
- batched cache 的 seq length/query offset 等于 batch max past length；
- mask sizes 等于 `max_past_length + 1`；
- 每层 update 不提前提交长度；
- 最后一层成功后所有 state 长度恰好增加 1；
- layer 重复、乱序或 query length 非 1 时受控失败；
- forward 中途失败后 committed lengths 不变。

#### Allocator and lifecycle

- partial exhaustion 自动 rollback；
- double release 不造成 duplicate free block；
- running cancellation 释放全部 blocks；
- block table 在释放后清空；
- randomized allocate/free 后无重复 active block、无泄漏。

#### Engine batching

- 三个 active request 只触发一次 decode model call；
- call recorder 看到 `input_ids.shape == (3, 1)`；
- `position_ids` 精确等于三个请求各自旧长度；
- attention mask 的左 padding 精确符合设计；
- logits row 按稳定顺序回写正确 request；
- 某请求完成后下一轮 batch size 缩小；
- 槽位释放后 waiting request 被接纳；
- `B=0` 时不执行空 forward；
- `B=1` 继续工作。

### 12.2 GPU Integration Tests

新增真实模型测试 `tests/integration/test_batch_decode.py`，固定三个 tokenized 长度不同的
prompt。测试必须同时记录 correctness 和 path identity：

1. reference 为每个请求独立执行固定 greedy generation；
2. TinyInfer 通过共享 pool 和一个 BatchEngine 执行；
3. 每个请求的 output token tuple 与对应 reference 100% 相等；
4. 至少一个 decode iteration 的实际 `input_ids` 是 `[3, 1]`；
5. 该 iteration 的 `decode_forward_count == 1`；
6. 记录的三个旧 cache lengths 互不相同；
7. repeated runs 后 pool 的 free block count 回到初始值；
8. CUDA peak allocated memory 在预定义容差内不随重复次数单调增长。

测试不能仅凭类名、函数名或三个最终结果推断“batched path 已执行”。

### 12.3 Dynamic Continuous Admission Test

至少一个测试使用不同 `max_new_tokens`，使请求在不同 iteration 完成，并设置 waiting request
数量大于 `max_batch`。证据必须显示：

```text
iteration N:   batch contains A, B
iteration N+1: A finishes, block resources released
iteration N+2: batch contains B, C
```

这用于区分静态 batching 和 continuous batching。

### 12.4 Performance Benchmark

同一固定环境比较：

- serial per-request decode baseline；
- true batched decode，至少 `B=1, 2, 4, 8`；
- 固定 prompt/output length 和并发 workload；
- warm-up 与至少 5 次正式重复；
- 记录 median/P95 latency、aggregate tokens/s、forward count、peak allocated/reserved
  memory、pool utilization 和 fragmentation。

benchmark 必须输出包含环境、模型 revision、Git SHA 和 workload identity 的原始 JSONL。

如果 true batch 的 median aggregate throughput 没有超过 serial baseline，correctness checkpoint
仍可通过，但不得声明“吞吐优化完成”；必须用 profiler 解释 gather、padding 或 model overhead。

## 13. Documentation Status Correction

实现代码前先修正文档状态，但保留历史报告的语义：

- `README.md`：从“KV cache/continuous batching/paged KV 均未实现”更新为逐项状态；
- `docs/BEGINNER_GUIDE.md`：明确当前 scheduler 是 serial-forward prototype，并在实现后解释
  true batch path；
- 新增或更新 Phase 2 status/report，记录证据 revision；
- `docs/PROJECT_PLAN.md` 的原始 G2 定义保留，不因为 checkpoint 降低标准；
- `docs/G1_REFERENCE_RUNTIME_REPORT.md` 是历史 G1 revision 的报告，不应被重写成当前 Phase 2
  状态。它说 G1 当时未证明 KV/continuous batching 是正确的历史边界；当前状态应由 README
  和 Phase 2 report 表达。

实现前状态词：

```text
contiguous KV cache: implemented prototype
paged KV cache: correctness prototype
FCFS scheduler: iteration-interleaving prototype
true batched decode: in progress / not yet implemented
continuous batching: not verified
block-native paged attention: not implemented
G2: not passed
```

只有新证据通过后才能更新相应状态。

## 14. Gate Structure

为避免再次把局部进展误报为完整阶段完成，G2 分成内部 checkpoint 和最终 Gate。

### 14.1 G2-BD: True Batched Decode Correctness Checkpoint

必须全部满足：

- unit tests 证明 batched storage、mask、position 和 forward-count contract；
- GPU 上三个异长请求逐 token 100% 对齐；
- 至少一次真实 `[3, 1]` decode forward；
- 对应 iteration 只有一次 model decode call；
- dynamic admission test 通过；
- completion/cancellation 后 block pool 恢复；
- full pytest、Ruff check、Ruff format 和 Mypy 全部退出码 0；
- CI 在目标 commit 上通过；
- 文档状态与证据一致。

G2-BD 通过后允许声明：

> TinyInfer implements and verifies a true heterogeneous batched decode path using a
> shared paged KV pool and a left-padded dense gather adapter.

仍禁止声明：

> TinyInfer implements block-native paged attention or has completed all G2 performance
> work.

### 14.2 G2-P: Performance Evidence Checkpoint

必须全部满足：

- serial 与 batch `B=1/2/4/8` benchmark 可复现；
- 原始记录、汇总表和图表提交；
- profiler 能解释主要耗时；
- 结果如实报告 speedup 或 regression；
- 若要声明“吞吐优化”，固定 workload 的 median aggregate throughput 必须大于 serial
  baseline。

### 14.3 Full G2 Gate

完整 G2 仍遵循 [`docs/PROJECT_PLAN.md`](../../PROJECT_PLAN.md)：

- reference、contiguous、batched 和 paged paths deterministic token alignment；
- randomized allocator 无泄漏、无 duplicate active blocks；
- representative concurrency workloads 无持续 memory growth；
- latency、throughput、memory、concurrency、utilization 和 fragmentation 均有可复现比较；
- G2-BD 与 G2-P 均通过；
- 当前 revision 的本地 Gate 和 remote CI 均为绿色。

## 15. Definition of Done

本设计对应的实现只有在以下全部成立时才完成：

### Functionality

- 每个 decode iteration 对所有 eligible running request 至多执行一个 model forward；
- 异长 batch 使用显式 per-row position IDs 和左 padding mask；
- shared paged pool 仍是持久 KV source of truth；
- batch membership 可随请求进入、结束和取消而变化。

### Correctness

- CPU unit、GPU alignment、dynamic admission、allocator randomized 和 memory stability
  tests 全部通过；
- 每个 token 与固定 HF reference 完全一致；
- forward shape/count 有直接证据。

### Engineering Quality

- pytest、Ruff check、Ruff format、Mypy 和 CI 全绿；
- 不遗留无类型 state dict 作为新 batched path 的核心 contract；
- 错误消息能指出 request row、layer、position 或 shape。

### Documentation and Evidence

- README、beginner guide、Phase 2 report 与实际状态一致；
- benchmark 原始记录包含环境、revision 和 workload identity；
- checkpoint 与完整 G2 的状态没有混用。

### Version Control

- 使用独立 worktree/feature branch；
- 每个 task 有独立、可审查 commit；
- 最终 worktree 干净；
- 未修改或伪造历史 G1 benchmark evidence；
- 不在 Gate 通过前创建 G2 release tag。

## 16. Risks and Mitigations

### Padding overhead hides batching benefit

异长差距过大会导致 dense padding 和 gather 开销。先用固定 workload 测量，再考虑 length-aware
batch selection 或 Triton gather；不能在 correctness 实现中提前加入复杂启发式。

### Scalar Cache protocol causes mask mismatch

使用左 padding保证当前 query 在所有行的 dense 最后一列，显式传 per-row position IDs 和 2D
mask，并用真实 Transformers 版本的 GPU test 锁定行为。

### Layer update partially succeeds

长度只在最后一层提交，目标 slot 可安全覆盖；所有输入、映射和容量在 forward 前预检。

### Dynamic batch order corrupts row mapping

每轮创建不可变 active ordering，输入、cache states、logits 和结果回写共用同一 ordering，并用
不同 sentinel logits 的 unit test 验证。

### Test passes while model still runs serially

验收同时检查 model call count 和真实 input shape；最终 token equality 不是 path identity 证据。

### Scope expands into vLLM clone

保持 serial prefill 和 dense attention，把 block-native kernel、chunked prefill、抢占和高级调度
留到 profile 驱动的后续阶段。

## 17. Review Decision Required Before Planning

本设计确认后，下一步使用 `superpowers:writing-plans` 编写可执行工作流。计划必须：

- 按 TDD 拆分 task；
- 为每个 task 列出 Inputs / Actions / Outputs / Acceptance Criteria；
- 明确 RED -> GREEN 证据；
- 明确 WSL/GPU 命令和超时；
- 明确 worktree、commit、handoff 和 stop conditions；
- 不允许 agent 在 G2-BD 通过后直接把完整 G2 标为 verified；
- 不允许绕过当前 baseline 的 pytest/Ruff/Mypy/CI 失败。
