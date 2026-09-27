# TinyInfer 完全入门指南

> 写给对 LLM 推理只有一点了解的初学者，带你从零理解整个项目。

---

## 第一部分：背景知识

### 1. 什么是 LLM 推理？

LLM（大语言模型，如 Qwen2.5-0.5B）是一个"下一个 token 预测器"。你给它一段文字（prompt），它预测下一个词（token），然后把新词拼接上，再预测下一个…… 如此循环直到说出"结束"标记或达到上限。

```
用户输入: "解释什么是 KV Cache"
       ↓
模型内部处理：
  Step 1: 预测 token A → "KV"
  Step 2: 预测 token B → "Cache"
  Step 3: 预测 token C → "是"
  Step 4: 预测 token D → "一种"
  ...
  Step N: 预测 token Z → "<eos>" (end of sequence, 停止)
       ↓
最终输出: "KV Cache 是一种..."
```

这个过程叫做**自回归生成**（autoregressive generation）。

### 1.1 为什么需要优化？

每次预测都需要经过模型的所有层（Transformer layers）。Qwen2.5-0.5B 有 24 层。如果不做优化，步骤 2 预测 token B 时，需要重新计算 token A 的结果；步骤 3 需要重新计算 token A+B 的结果... 计算量随生成长度**平方增长**。

这就引出了 KV Cache 的概念。

---

### 2. 三个核心概念

#### 2.1 Prefill（预填充）

Prefill 是生成过程的**第一步**：

```
输入: "解释什么是 KV Cache"  (完整 prompt, 8 个 token)
       ↓
[模型 24 层 Transformer，逐层计算]
       ↓
输出: "KV"  (第一个预测 token)
       ↓
同时产生: KV Cache (8 个 token 的 Key/Value 中间结果)
```

Prefill 的特点：
- **一次处理完整 prompt**（所有 token 同时进入模型）
- 产生第一个输出 token
- 产生 KV Cache

#### 2.2 Decode（解码）

Decode 是生成过程的**后续步骤**：

```
Step 2:
  输入: "KV"  (只有 1 个 token!)
  已有: KV Cache (前 8 个 token 的 K/V)
       ↓
  [模型利用 KV Cache，只需计算新 token]
       ↓
  输出: "Cache"
  追加 KV Cache (现在是 9 个 token 的 K/V)

Step 3:
  输入: "Cache"  (只有 1 个 token)
  已有: KV Cache (前 9 个 token 的 K/V)
       ↓
  输出: "是"
  追加 KV Cache

... 重复直到 <eos> ...
```

Decode 的特点：
- **每次只处理 1 个 token**
- 复用之前的 KV Cache，避免重复计算
- 每次递增加一个 token 的 K/V

#### 2.3 KV Cache（Key-Value 缓存）

KV Cache 是 Transformer 注意力机制的加速核心：

```
想象你在读一本书，每读一个新句子都需要回顾之前所有的句子来理解上下文。
KV Cache = 你对之前句子的"记忆笔记"，不用每次都从头重读。

在模型内部：
  每层 attention 需要计算 Query × Key → attention scores → × Value
  把之前 token 的 Key 和 Value 存下来 = KV Cache
  新 token 只需要计算自己的 K/V，然后和缓存拼接即可
```

---

### 3. 三种 Cache 实现

TinyInfer 的核心任务就是**自己管理 KV Cache**，从借用 HF 的实现到完全接管：

#### 3.1 HF DynamicCache（原 G1 — `generate_greedy()`）

```
HF 内部:
  past_key_values = DynamicCache()
  # 每层是一个 list: [key_tensor, value_tensor]
  # 每步 append 新 K/V
  # 动态增长，碎片化
```

TinyInfer 之前的 `generate_greedy()` **只是透传**这个 HF 自己的 Cache。

```
HF forward → HF DynamicCache → HF forward → HF DynamicCache → ...
  ↑                                                              ↑
  └──── TinyInfer 不做任何管理，只是传递 ──────────────────────────┘
```

#### 3.2 TinyInfer ContiguousKVCache（Phase 1.5 — `generate_with_owned_cache()`）

TinyInfer 自己预分配一块连续 tensor：

```
TinyinferCache 对象:
  内部持有的 ContiguousKVCache:
    ┌───────────────────────────────────────────┐
    │ [24 layers, 2 (K/V), 4 heads, 4096 seq,  head_dim]  │
    │ 预分配的连续内存，一次性创建，反复使用    │
    └───────────────────────────────────────────┘
```

**但怎么写和读这块 buffer 呢？** 我们让 `TinyinferCache` 实现了 HF 期望的接口：

```
HF attention 内部调用:
  past_key_value.update(key, value, layer_idx)  → 写入 TinyInfer buffer
  key, value = past_key_value[layer_idx]        → 从 TinyInfer buffer 读
  
TinyinferCache 实现:
  update()  → ContiguousKVCache.write_decode()
  __getitem__() → ContiguousKVCache.read()
```

**关键洞察**：HF 不是"写到一个独立的 DynamicCache 再让我们复制"，而是**直接写到 TinyInfer 的 buffer**。TinyInfer 拥有 KV 的所有权。

#### 3.3 TinyInfer PagedKVCache（Phase 2 — `generate_with_paged_cache()`）

更进一步，按固定大小分块：

```
PagedKVCache:
  物理块池 (Physical Block Pool):
    [24 layers, 2 (K/V), B blocks, 4 heads, 16 block_size, head_dim]
    block 0: [...][...][...][...]
    block 1: [...][...][...][...]
    block 2: [...][...][...][...]
    ...

  BlockTable (每请求一个):
    [layer 0]: logical_block_0 → physical block 3
               logical_block_1 → physical block 7
               logical_block_2 → physical block 1
    [layer 1]: logical_block_0 → physical block 3
               ...
```

```
逻辑视图 (连续的 token):
  [t0 t1 ... t15] [t16 t17 ... t31] [t32 t33 ...]
  ←── block 0 ──→ ←── block 1 ───→ ←── block 2 ─→

物理存储 (通过 BlockTable 映射):
  block 0 → physical block 3
  block 1 → physical block 7
  block 2 → physical block 1
```

**好处**：物理块可以复用，多个请求可以共享块池，不会因为一个请求用完所有显存。

---

## 第二部分：项目结构索引导读

```
tinyinfer/                         # Python 包（= 你的代码）
│
├── __init__.py                    # 版本号 v0.2.0
├── __main__.py                    # python -m tinyinfer 入口
├── _tokens.py                     # token ID 验证工具（shared）
├── config.py                      # EngineConfig 不可变配置
├── environment.py                 # 环境检查（Python/包/CUDA/GPU）
│
├── runtime/                       # 🏃 核心运行时
│   ├── request.py                 # GenerationRequest 输入合约
│   ├── result.py                  # GenerationResult 输出合约
│   ├── hf_loader.py               # 固定 revision 加载 Qwen2.5-0.5B
│   └── generation.py              # 三条生成路径 ⭐
│
├── cache/                         # 🗄️ KV Cache 实现
│   ├── contiguous.py              # ContiguousKVCache 连续缓存
│   ├── tinynfer_cache.py          # 实现 HF Cache 协议（contiguous 版）
│   ├── block.py                   # BlockTable + BlockAllocator
│   ├── paged.py                   # PagedKVCache 分页缓存
│   ├── paged_tinyinfer_cache.py   # 实现 HF Cache 协议（paged 版）
│   └── adapter.py                 # HF ↔ TinyInfer 桥（旧方案，保留）
│
└── profiling/                     # 📊 性能测量
    ├── timer.py                   # InferenceTimer 状态机计时器
    └── record.py                  # BenchmarkRecord 记录契约

scripts/                           # 🔧 工具脚本
├── check_environment.py           # 环境检查 CLI
├── check_gpu_smoke.py             # GPU 冒烟测试
├── triton_vector_add.py           # Triton 向量加 kernel
├── verify_g0_5.py                 # G0.5 验证门
└── verify_g1.py                   # G1 验证门

benchmarks/                        # 📈 基准测试
├── bench_reference.py             # Benchmark CLI（507行）
└── prompts.json                   # 4 个冻结 prompt

configs/
└── qwen2_5_0_5b.yaml             # 模型配置

tests/
├── unit/                          # 14 个单元测试（Fake 层，CPU 可跑）
└── integration/                   # 真实 GPU 集成测试
```

---

## 第三部分：逐模块深入讲解

### 6. 配置层

#### 6.1 EngineConfig (`config.py`)

```python
@dataclass(frozen=True, slots=True)
class EngineConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    dtype: str = "bfloat16"       # 精度
    device: str = "cuda"          # CPU 或 cuda
    max_sequence_length: int = 4096  # 最大序列长度
    kv_block_size: int = 16       # 分页块大小
```

**为什么用 `frozen=True, slots=True`？**
- `frozen=True`：创建后不可修改。防止配置在运行中被意外改动。
- `slots=True`：节省内存，禁止动态添加属性。

**验证在构造时完成**（`__post_init__`），不合格的配置**立即报错**，而不是在运行到一半才发现问题。

#### 6.2 EnvironmentReport (`environment.py`)

```python
@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    python_version: str
    platform: str
    packages: tuple[PackageStatus, ...]  # torch/transformers/safetensors/triton
    cuda_available: bool
    gpu_name: str | None
    bf16_supported: bool
    diagnostics: tuple[str, ...]
```

运行 `python -m tinyinfer` 即可看到完整报告。它**不需要 GPU**也能运行——缺少的包被标记为 "not installed" 而不是报错。

---

### 7. 合约层 — 输入输出

#### 7.1 GenerationRequest (`request.py`)

```python
@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str             # 请求ID，如 "short_en"
    prompt: str                 # 用户输入的文本
    max_new_tokens: int = 32    # 最多生成多少 token
```

约束：
- `request_id` / `prompt` 不能为空
- `max_new_tokens` 在 1~256 之间

#### 7.2 GenerationResult (`result.py`)

```python
@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    prompt_token_ids: tuple[int, ...]     # 输入的 token ID 序列
    output_token_ids: tuple[int, ...]     # 输出的 token ID 序列
    text: str                             # 解码后的文本
    stop_reason: StopReason               # "eos" 或 "length"
```

`StopReason`：
- `"eos"`：模型自己生成了结束标记（`<eos>`）
- `"length"`：达到 `max_new_tokens` 限制

---

### 8. 加载层 (`hf_loader.py`)

```python
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
```

**为什么锁定 revision？**
不同版本的模型权重可能不同。锁定 hash 保证**跨时间、跨机器、跨人的可复现性**。

```python
def load_reference_model(config: EngineConfig) -> LoadedModel:
    # 1. 验证配置（必须是 Qwen2.5-0.5B + bfloat16 + cuda）
    # 2. 加载 tokenizer（固定 revision）
    # 3. 加载模型（固定 revision, bfloat16, SDPA attention）
    # 4. 放置到 GPU，eval 模式
    # 5. 解析 EOS/PAD token ID
    # 6. 返回 LoadedModel
```

---

### 9. 生成层 — `generation.py` ⭐

这是整个项目的核心文件。里面有**三条**生成路径，从简单到复杂。

#### 9.1 路径 1：`generate_greedy()` — 透传 HF Cache（G1）

```
拿 HF DynamicCache
     ↓
prefill: HF forward (use_cache=True) → 拿到 DynamicCache
     ↓
decode: HF forward (past_key_values=DynamicCache) → 更新 DynamicCache
     ↓
decode: HF forward (past_key_values=DynamicCache) → 更新 DynamicCache
     ↓
... 重复 ...
```

```python
# 伪代码
cache = None  # prefill 时不传 past_key_values
while 没到 eos 且 没到 max_new_tokens:
    outputs = model(
        input_ids=新token,
        past_key_values=cache,   # ← 透传 HF 自己的 Cache
        use_cache=True,
    )
    cache = outputs.past_key_values  # ← 拿回来
```

#### 9.2 路径 2：`generate_with_owned_cache()` — TinyInfer 接管（Phase 1.5）

```
创建 TinyinferCache(内部是 ContiguousKVCache)
     ↓
prefill: HF forward (past_key_values=tinyinfer_cache)
     ↓   HF attention.update() → 写入 TinyInfer buffer
     ↓   HF attention[key] → 从 TinyInfer buffer 读
     ↓
decode: HF forward (past_key_values=同一个 tinynfer_cache 对象!)
     ↓   HF attention.update() → 追加到 TinyInfer buffer
     ↓
decode: HF forward (past_key_values=同一个 tinynfer_cache 对象!)
     ↓
... 重复 ...
```

**关键区别**：不需要每步重建 DynamicCache。同一个 `TinyinferCache` 对象全程复用。

```python
# 关键代码
contiguous = allocate_contiguous_cache(...)  # 预分配连续 tensor
tinfer_cache = TinyinferCache(contiguous)    # 包装为 HF Cache 协议

# prefill — 同一个对象
outputs = model(..., past_key_values=tinfer_cache, ...)

# decode — 仍然是同一个对象!
decode_outputs = model(..., past_key_values=tinfer_cache, ...)
```

#### 9.3 路径 3：`generate_with_paged_cache()` — 分页 KV（Phase 2）

```
PagedKVCache:
  物理块池 [L, 2, B_blocks, H, 16_block_size, D]
  BlockAllocator: free list [0, 1, 2, ..., 255]
     ↓
创建 BlockTable (per request)
  分配物理块: block_table[layer][0] = 3
              block_table[layer][1] = 7
              block_table[layer][2] = 1
     ↓
PagedTinyinferCache(paged, block_table)
     ↓
HF forward (past_key_values=paged_tinyinfer_cache)
     ↓ HF attention.update() → PagedKVCache.write_prefill()
     ↓   → 按 block_table 映射写入物理块
     ↓ HF attention[key] → PagedKVCache.read()
     ↓   → 按 block_table 从物理块 gather 成连续 tensor
     ↓
decode: 同上，每次追加 1 token 到对应位置的物理块
```

```python
# 预分配
paged = PagedKVCache(num_layers=24, num_kv_heads=4, block_size=16, ...)
block_table = paged.create_block_table(max_tokens=4096)
paged.allocate_blocks(block_table, total_blocks=256)

# 同一个对象全程复用
tinfer_cache = PagedTinyinferCache(paged, block_table)
outputs = model(..., past_key_values=tinfer_cache, ...)
decode_outputs = model(..., past_key_values=tinfer_cache, ...)
```

---

### 10. Cache 层详解

#### 10.1 ContiguousKVCache (`contiguous.py`)

```python
@dataclass(frozen=True, slots=True)
class ContiguousKVCache:
    buffer: Any    # [num_layers, 2, num_kv_heads, max_seq_len, head_dim]
    length: int = 0

    def write_prefill(self, key, value, layer_idx):
        # key: [1, num_kv_heads, prompt_len, head_dim]
        # 写入 buffer[layer_idx, 0, :, :prompt_len, :] ← key[0]
        #       buffer[layer_idx, 1, :, :prompt_len, :] ← value[0]
        # 设置 length = prompt_len

    def write_decode(self, key, value, layer_idx):
        # key: [1, num_kv_heads, 1, head_dim]
        # 写入 buffer[layer_idx, 0, :, position, :] ← key
        # length += 1

    def read(self, layer_idx):
        # 返回 buffer[layer_idx, 0, :, :length, :]  (key)
        #       buffer[layer_idx, 1, :, :length, :]  (value)
```

**为什么用 `object.__setattr__` 修改 frozen dataclass 的 `length`？**

因为 `ContiguousKVCache` 是 `frozen=True` 的，不能直接 `self.length = 3`。但 KV Cache 的本质决定了 length 需要变化。我们用 `object.__setattr__(self, "length", new_value)` 绕过 frozen 限制——这是 Python 中公认的 "mutable field inside frozen dataclass" 模式。

#### 10.2 TinyinferCache (`tinyinfer_cache.py`)

```python
class TinyinferCache:
    """HF Cache 协议实现"""
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """HF 每层 attention 计算完 K/V 后调用此方法"""
        if 第一次:
            self._contiguous.write_prefill(key_states, value_states, layer_idx)
            设为已经 prefill
        else:
            self._contiguous.write_decode(key_states, value_states, layer_idx)
        return self._contiguous.read(layer_idx)  # 返回完整 KV

    def __getitem__(self, layer_idx):
        """HF attention 读取历史 KV 时调用"""
        return self._contiguous.read(layer_idx)

    def get_seq_length(self):
        return self._contiguous.length
```

**为什么不需要继承 `transformers.Cache`？**

因为 Python 的 duck typing——只要对象有 `update()`、`__getitem__()`、`get_seq_length()` 方法，HF 代码就能用。继承会增加对 `transformers` 的 import 依赖，破坏 CPU 测试的隔离性。

#### 10.3 BlockTable + BlockAllocator (`block.py`)

```python
class BlockTable:
    """逻辑块 → 物理块的映射"""
    # _table[layer_idx][logical_block] = physical_block
    # 未分配时 = -1

class BlockAllocator:
    """空闲块管理器（FIFO 队列）"""
    # _free: deque([0, 1, 2, ..., N-1])
    # allocate(): pop left
    # free(block_idx): append right
```

```
BlockAllocator 示例:
  创建时: _free = [0, 1, 2, 3, 4, 5, 6, 7]
  
  分配 3 个块:
    allocate() → 0, _free = [1, 2, 3, 4, 5, 6, 7]
    allocate() → 1, _free = [2, 3, 4, 5, 6, 7]
    allocate() → 2, _free = [3, 4, 5, 6, 7]
  
  释放块 1:
    free(1) → _free = [3, 4, 5, 6, 7, 1]
```

#### 10.4 PagedKVCache (`paged.py`)

```python
class PagedKVCache:
    """分页 KV 缓存池 + 块表管理"""
    
    def __init__(self, *, num_layers, num_kv_heads, block_size, head_dim, ...):
        # 物理 buffer: [L, 2, B, H, block_size, D]
        self._buffer = torch.zeros(...)
        self._allocator = BlockAllocator(num_physical_blocks)

    def write_prefill(self, key, value, layer_idx, block_table, tokens):
        # 按块拆分写入:
        #   logical 0 → block_table[layer_idx][0] → physical block
        #   logical 1 → block_table[layer_idx][1] → physical block
        #   每个块写最多 block_size 个 token

    def read(self, layer_idx, block_table, tokens):
        # 从物理块 gather 成连续 tensor:
        #   遍历 logical blocks
        #   从对应 physical block copy 到输出 tensor
```

#### 10.5 PagedTinyinferCache (`paged_tinyinfer_cache.py`)

```python
class PagedTinyinferCache:
    """实现 HF Cache 协议的分页版本"""
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if 第一次:
            self._paged.write_prefill(key_states, value_states, layer_idx,
                                       self._block_table, prompt_len)
        else:
            self._paged.write_decode(key_states, value_states, layer_idx,
                                      self._block_table, 当前位置)

    def __getitem__(self, layer_idx):
        return self._paged.read(layer_idx, self._block_table, self._length)
```

---

### 11. Profiling 层

#### 11.1 InferenceTimer (`timer.py`)

`InferenceTimer` 实现了 `GenerationObserver` 协议，在 7 个时间点记录状态：

```
request_started()        ─→ 记录开始时间，重置显存峰值
prefill_started()        ─→ 记录 prefill 开始（Host + CUDA event）
prefill_finished()       ─→ 记录 prefill 结束（Host + CUDA event）
first_token_ready()      ─→ CUDA synchronize，计算 TTFT
decode_started()         ─→ 记录 decode 开始（Host + CUDA event）
decode_finished(n_tokens)─→ 记录 decode 结束（Host + CUDA event）
request_finished()       ─→ CUDA synchronize，计算所有指标
```

最终产生 `PhaseTimings`：
```python
@dataclass(frozen=True, slots=True)
class PhaseTimings:
    ttft_host_ms: float          # Time To First Token (Host)
    prefill_host_ms: float
    prefill_device_ms: float     # CUDA event 计时
    decode_host_ms: float
    decode_device_ms: float
    tpot_host_ms: float          # Time Per Output Token
    tpot_device_ms: float
    output_tokens_per_s: float   # 吞吐
    peak_allocated_bytes: int    # 峰值显存
    peak_reserved_bytes: int
```

#### 11.2 BenchmarkRecord (`record.py`)

`BenchmarkRecord` 有 20+ 字段，记录一次 benchmark 运行的所有信息：
- **环境**：Git revision × model revision × PyTorch/Torch 版本 × dtype × GPU 名称
- **输入**：case_id, prompt_tokens
- **输出**：output_token_ids, stop_reason
- **正确性**：token_agreement (0.0~1.0, 必须是 1.0), first_mismatch_index
- **性能**：所有 PhaseTimings 字段

每条记录序列化为 `schema-v1` JSONL 行：
```jsonl
{"schema_version":1,"git_revision":"b5225f8e...","case_id":"short_en","output_token_ids":[271,1045,...],"token_agreement":1.0,...}
```

---

## 第四部分：动手实践

### 12. 环境搭建

```bash
# 创建虚拟环境
python -m venv .venv

# 安装 dev 依赖（pytest, ruff, mypy）
pip install -e ".[dev]"

# 如果需要 GPU 推理（在 WSL2/Linux 上）
pip install -e ".[dev,inference,kernels]"
```

### 13. 跑第一个测试

```bash
# 打印版本和环境
python -m tinyinfer

# 输出示例：
# TinyInfer 0.2.0
# Python: 3.11.15
# Platform: Linux-...
# Packages:
#   torch: 2.13.0+cu130
#   transformers: 5.14.1
#   ...
```

```bash
# 运行 CPU 单元测试（不需要 GPU）
python -m pytest tests/unit -m "not gpu and not model and not slow"
```

### 14. 运行三种生成路径

```python
# 在 WSL2/GPU 环境下：
from tinyinfer.config import EngineConfig
from tinyinfer.runtime import (
    load_reference_model,
    GenerationRequest,
    generate_greedy,           # 路径 1: 透传 HF Cache
    generate_with_owned_cache, # 路径 2: TinyInfer 接管 Cache
    generate_with_paged_cache, # 路径 3: 分页 Cache
    reference_generate,        # HF 官方 generate (正确性基准)
)

# 加载模型
loaded = load_reference_model(EngineConfig())

# 创建请求
req = GenerationRequest("test", "Explain KV cache.", max_new_tokens=16)

# 路径 1 — 透传 HF Cache
result1 = generate_greedy(loaded, req)

# 路径 2 — TinyInfer 接管
result2 = generate_with_owned_cache(loaded, req)

# 路径 3 — 分页
result3 = generate_with_paged_cache(loaded, req)

# HF 官方基准
reference = reference_generate(loaded, req)

# 验证正确性
print("路径 1 == HF:", result1.output_token_ids == reference)
print("路径 2 == HF:", result2.output_token_ids == reference)
print("路径 3 == HF:", result3.output_token_ids == reference)
```

### 15. 运行 Benchmark

```bash
# 在 WSL2/GPU 环境下：
HF_HUB_OFFLINE=1 python benchmarks/bench_reference.py \
  --config configs/qwen2_5_0_5b.yaml \
  --prompts benchmarks/prompts.json \
  --output /tmp/tinyinfer-g1/reference.jsonl \
  --warmup-runs 3 \
  --repetitions 5 \
  --overwrite
```

成功输出：`G1 REFERENCE RUNTIME: PASS`

这会为 4 个 prompt（short_en, short_zh, structured, long_context）各生成 5 次 repetition，共 20 条记录。

---

## 第五部分：Agent 视角 QA

### Q: 为什么所有数据结构都用 `@dataclass(frozen=True, slots=True)`？

**A**: 因为这是一个推理引擎项目，正确性优先。不可变数据类的好处：
- 创建后不能被意外修改，避免"某处改了配置导致调试三天"的情况
- 构造时验证（`__post_init__`），错误尽早暴露
- `slots=True` 节省内存，对于大量 request/result 对象有意义

### Q: 为什么 `ContiguousKVCache` 用 `object.__setattr__(self, "length", ...)` 修改 frozen dataclass？

**A**: `ContiguousKVCache` 是 `frozen=True` 的，但 `length` 需要在 prefill/decode 过程中变化。我们选择保持 dataclass 不变、用底层方法绕过冻结——因为 `length` 的语义就是"当前已缓存的 token 数"，它不是配置，是运行时状态。

### Q: 为什么 `TinyinferCache` 不继承 `transformers.Cache`？

**A**: 避免 import 依赖。`__init__.py` import `generation.py` → `generation.py` import `tinyinfer_cache.py`。如果 `tinyinfer_cache.py` import `transformers`，整个 `tinyinfer` 包都依赖 `transformers`，CPU 单元测试就跑不了了。

Python 的 duck typing —— "如果它走路像鸭子，叫起来像鸭子，那它就是鸭子" —— 让这件事成为可能。

### Q: Phase 1.5 的 `generate_with_owned_cache()` 和之前的 `generate_with_tinyinfer_cache()` 有什么区别？

**A**: 本质区别在于 **KV 所有权**：

```
旧 (generate_with_tinyinfer_cache):
  TinyInfer buffer → export → HF DynamicCache → HF forward → import → TinyInfer buffer
  (每步都要重建 DynamicCache，所有权在两边传递)

新 (generate_with_owned_cache):
  TinyinferCache ← HF attention.update() ← 直接写入 TinyInfer buffer
  TinyinferCache → HF attention[key] → 直接从 TinyInfer buffer 读取
  (同一个对象全程复用，所有权完全在 TinyInfer)
```

### Q: PagedKVCache 的 block_size=16 是怎么定的？

**A**: 这是 Qwen2.5 模型 `EngineConfig` 的默认值，也是 vLLM 等生产系统的常用值。太小的 block 增加映射开销，太大的 block 浪费显存（一个 token 就占 16 个位置）。16 是一个经验平衡点。

### Q: 为什么 benchmark 要跑 5 次 repetition + 3 次 warmup？

**A**: GPU 的第一次运行通常比后续慢（kernel 编译缓存、CUDA 初始化）。3 次 warmup 消除"冷启动"效应。5 次 repetition 提供足够的样本计算 median/p95，同时验证输出是确定性的（5 次必须完全一致）。

### Q: 这个项目的测试可以在没有 GPU 的机器上跑吗？

**A**: 可以！CPU 单元测试（`pytest -m "not gpu"`）使用 Fake 层模拟 torch、Tensor、Cache 的行为，完全不依赖 GPU。集成测试（`pytest -m "gpu and model"`）需要 GPU 和已下载的模型。

### Q: 整个项目的"难度曲线"是什么样的？

**A**: 
```
先理解这三个概念：
  ① Prefill vs Decode
  ② KV Cache 是什么
  ③ BlockTable 的逻辑块到物理块映射

然后按路径读代码：
  ① generate_greedy()       → 最简单，只是透传 HF Cache
  ② generate_with_owned_cache()  → 理解 TinyinferCache 如何"骗过" HF
  ③ generate_with_paged_cache()  → 理解分块 + 映射

最后看测试：
  ① test_generation.py      → Fake 层是怎么模拟的
  ② test_contiguous_cache.py → Cache 层的边界条件
```

---

## 附录：术语表

| 术语 | 含义 |
|------|------|
| **LLM** | Large Language Model（大语言模型） |
| **Token** | 文本的最小单位（可能是词、字、子词） |
| **Prefill** | 一次性处理完整 prompt |
| **Decode** | 每次处理一个 token |
| **KV Cache** | 注意力层的 Key-Value 中间结果缓存 |
| **Greedy Decoding** | 每次选概率最高的 token（确定性） |
| **EOS** | End of Sequence（结束标记） |
| **TTFT** | Time To First Token（首个 token 延迟） |
| **TPOT** | Time Per Output Token（每个输出 token 平均延迟） |
| **Block Table** | 逻辑块 → 物理块的映射表 |
| **Free List** | 空闲物理块队列 |
| **Contiguous** | 连续内存布局 |
| **Paged** | 分页内存布局 |
| **SDPA** | Scaled Dot-Product Attention（PyTorch 内置实现） |
| **GQA** | Grouped Query Attention（Qwen2.5 使用的注意力变体） |
| **BF16** | Brain Floating Point 16（半精度浮点格式） |

---

> 文档版本：2026-08-04  
> 对应代码 revision：`13ff82d`  
> 作者：TinyInfer Project


---

## 附录 C：Phase 2 状态边界（2026-08-10）

> 本文档描述的三条生成路径中：
>
> - `generate_greedy`（HF DynamicCache）是已发布的 G1 参考实现；
> - `generate_with_owned_cache`（ContiguousKVCache）与
>   `generate_with_paged_cache`（PagedKVCache）当前为 **prototype**；
> - FCFS scheduler 是 **iteration-interleaving 原型**，并且 true batched decode
>   为 **G2-BD local candidate**：原生 WSL correctness 与两轮 audit remediation
>   已通过，仍需按声明提交核验 remote CI 状态；
> - `G2-P1: verified locally`（batching throughput evidence，见 `docs/G2_P1_REPORT.md`）；
> - `G2-P2: not started`（paged-KV memory-efficiency evidence）；
> - `block-native paged attention: not implemented`（当前用 dense 左填充 gather）；
> - `full G2: not passed`。

具体状态请以 `docs/PHASE2_STATUS.md` 为准。
