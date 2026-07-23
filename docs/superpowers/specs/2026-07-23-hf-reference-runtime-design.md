# TinyInfer R4–R7 + G1 Hugging Face Reference Runtime 设计规格

> 版本：v1.0
> 日期：2026-07-23
> 状态：design approved，implementation not started
> 基线：`v0.1.0-g0.5` / `d8e55651186b19eae2af97b42cc17d1df1bbf7e5`
> 目标平台：WSL2 Ubuntu-22.04、Python 3.11、NVIDIA GeForce RTX 4080 SUPER

## 1. 背景

TinyInfer 已完成 G0.5 Linux/GPU 环境基线，但尚未实现模型推理。申请导向的下一项有效证据不是
继续扩展基础设施，而是在真实 Qwen 模型上展示以下能力：

1. 理解 causal language model 的 prefill 与逐 token decode 数据流。
2. 正确使用 Hugging Face 提供的 KV cache，而不是把 `model.generate()` 包装成自己的 runtime。
3. 用 token-level agreement 证明实现正确。
4. 用可复现实验记录 TTFT、TPOT、吞吐量和显存，为后续自定义 KV cache 与调度器提供基线。

本规格把 R4–R7 与 G1 定义为一个可独立运行、独立验收的工作包。工作包结束时应产生第一个
真正具有 LLM systems 展示价值的 TinyInfer 版本。

## 2. Objective

基于固定 revision 的 `Qwen/Qwen2.5-0.5B`，实现单请求、greedy、显式 prefill/decode 的
reference runtime；在四类固定 prompt 上与 Hugging Face `model.generate()` 达到 100% token
agreement，并输出可复现的性能 JSONL 与阶段报告。

## 3. Non-goals

本工作包明确不实现：

- 自定义 contiguous KV cache。
- Request 状态机、iteration-level scheduler 或 continuous batching。
- Paged KV cache、block pool、block table 或 Qwen attention adapter。
- API Server、流式 HTTP/SSE、鉴权或并发服务。
- 自定义 Triton kernel、paged attention 或 attention 性能优化。
- Sampling、temperature、top-k、top-p、beam search 或 repetition penalty。
- 多 GPU、tensor parallel、CPU offload、量化或混合模型。
- 长上下文压力测试；G1 的长 prompt 只要求不小于 512 tokens。
- 与 vLLM、TensorRT-LLM 等外部 runtime 做性能结论。

上述内容只能在 G1 通过后分别设计。任何 agent 不得以“方便后续”为由提前加入抽象层。

## 4. 固定依赖与模型身份

### 4.1 软件版本

实现与验收使用 G0.5 锁定环境：

| 组件 | 固定值 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.14.1 |
| Safetensors | 0.8.0 |
| Triton | 3.7.1；本工作包不编写新 kernel |
| dtype | `torch.bfloat16` |
| device | `cuda` |
| attention implementation | `sdpa` |

### 4.2 模型 revision

模型身份固定为：

```text
model_id: Qwen/Qwen2.5-0.5B
model_revision: 060db6499f32faf8b98477b0a26969ef7d8b9987
```

该 SHA 于 2026-07-23 从 Hugging Face 模型 API 解析。模型与 tokenizer 必须同时传入该
revision，禁止依赖可变化的默认 `main`。

### 4.3 Transformers Cache API

Transformers 5.14.1 中 `Qwen2ForCausalLM.forward` 接受
`past_key_values: transformers.cache_utils.Cache | None`。实现必须使用此公开 Cache API，
通过 `get_seq_length()` 检查 cache 长度；不得依赖 legacy tuple 的内部层布局。

## 5. 系统边界与目录结构

新增或修改以下文件：

```text
tinyinfer/
├── runtime/
│   ├── __init__.py
│   ├── request.py
│   ├── result.py
│   ├── hf_loader.py
│   └── generation.py
├── profiling/
│   ├── __init__.py
│   ├── timer.py
│   └── record.py
benchmarks/
├── __init__.py
├── prompts.json
└── bench_reference.py
tests/
├── unit/
│   ├── test_request.py
│   ├── test_result.py
│   ├── test_hf_loader.py
│   ├── test_generation.py
│   ├── test_prompts.py
│   ├── test_timer.py
│   ├── test_record.py
│   └── test_bench_reference.py
└── integration/
    ├── test_hf_loader.py
    └── test_hf_reference_runtime.py
docs/
├── R4_R7_G1_WORKFLOW.md
├── G1_REFERENCE_RUNTIME_REPORT.md
└── results/
    └── g1_reference.jsonl
```

文件职责：

| 文件 | 唯一职责 |
|---|---|
| `request.py` | 校验不可变的单请求生成输入 |
| `result.py` | 表示不可变的生成结果与停止原因 |
| `hf_loader.py` | 固定 revision 加载 tokenizer/model，并建立设备与 token 配置 |
| `generation.py` | 显式 prefill、单步 decode 和 greedy 循环 |
| `timer.py` | Host clock、CUDA event、同步和 peak memory 采样 |
| `record.py` | 将正确性、计时和环境身份转换为 schema v1 记录 |
| `prompts.json` | 固定四类 G1 correctness/benchmark workload |
| `bench_reference.py` | CLI、warm-up、重复运行、HF 对齐和 JSONL 写入 |
| `docs/results/g1_reference.jsonl` | G1 实机验收的小型规范结果集，可提交 |

`.venv`、Hugging Face cache、模型权重、临时日志、Python cache 和未筛选的大型 profiler 输出
不得提交。

## 6. 公开数据模型

### 6.1 GenerationRequest

```python
@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int = 32
```

约束：

- `request_id.strip()` 非空。
- `prompt.strip()` 非空。
- `1 <= max_new_tokens <= 256`。
- 对象创建后不可修改。
- 本阶段一次只执行一个 request；对象不包含调度状态。

### 6.2 StopReason

```python
StopReason = Literal["eos", "length"]
```

模型加载错误、CUDA OOM、无效输入与 runtime invariant 失败必须抛异常，不能伪装成正常停止。

### 6.3 GenerationResult

```python
@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    text: str
    stop_reason: StopReason
```

约束：

- prompt 与 output token 均非空。
- token ID 均为非负整数。
- `text` 只用于展示；正确性比较使用 `output_token_ids`。
- result 不保存 model、tokenizer、Tensor 或 cache，保证可序列化与低耦合。

### 6.4 LoadedModel

```python
@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_id: str
    model_revision: str
    eos_token_id: int
    pad_token_id: int
```

`LoadedModel` 是加载边界的返回值。其字段只描述单模型实例，不承担生命周期管理、并发或缓存池
职责。

## 7. 模型加载设计

`load_reference_model(config: EngineConfig) -> LoadedModel` 执行以下固定步骤：

1. 校验 `config.model_id == "Qwen/Qwen2.5-0.5B"`、dtype 为 `bfloat16`、device 以
   `cuda` 开头。
2. tokenizer 使用固定 `model_revision` 和 `trust_remote_code=False` 加载。
3. model 使用同一 revision、`torch_dtype=torch.bfloat16`、
   `attn_implementation="sdpa"`、`trust_remote_code=False` 加载。
4. model 显式移动到目标 CUDA device，并调用 `eval()`。
5. 若 tokenizer 无 PAD token，则只在内存中令 `pad_token_id = eos_token_id`；不得修改远端资产。
6. 将 `model.config.use_cache` 设为 `True`。
7. 返回 resolved model identity、EOS/PAD ID 与对象引用。

禁止使用 `device_map="auto"`，因为它会引入 accelerate 与隐式设备放置。模型加载失败时抛出
`ModelLoadError`，错误文本必须包含 model ID、revision 和原始异常类型，但不得包含 token、代理或
本机敏感路径。

## 8. 显式生成数据流

```text
GenerationRequest
  → tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  → input_ids + attention_mask 移至 CUDA
  → prefill(full input, use_cache=True)
  → logits[:, -1, :] → argmax → first output token
  → Cache(seq_length == prompt_tokens)
  → decode_one(last token, full attention mask, cache_position)
  → updated Cache + argmax token
  → EOS 或 max_new_tokens
  → tokenizer.decode(output ids, skip_special_tokens=True)
  → GenerationResult
```

所有 forward 必须位于 `torch.inference_mode()` 内，不得创建梯度。

`generate_greedy()` 接受可选的 `GenerationObserver` 协议。Observer 只接收 request 开始、
prefill 开始/结束、第一个 token 可读、decode 开始/结束和 request 结束事件；它不能改变 token、
Cache 或停止条件。未传 observer 时使用 no-op 路径。`InferenceTimer` 实现该协议，从而保证生成
循环只有一份，profiling 不复制 runtime 控制流。

### 8.1 Prefill

`prefill(...)` 一次输入完整 prompt。返回第一个 greedy token、更新后的公开 Cache 和必要 logits
元数据。第一个 token 使用 `outputs.logits[:, -1, :].argmax(dim=-1)` 选择。

Prefill 后必须满足：

- batch size 为 1。
- logits batch 维与 vocab 维有效。
- `past_key_values` 非空。
- `past_key_values.get_seq_length() == prompt_tokens`。

### 8.2 Decode one step

`decode_one(...)` 每轮只输入形状 `[1, 1]` 的上一个生成 token，同时：

- attention mask 长度扩展到 `prompt_tokens + 已输入的生成 token 数`。
- `cache_position` 等于调用前 Cache 的 `get_seq_length()`。
- 更新后的 Cache 长度恰好增加 1。
- 从本轮最后 logits 位置选择下一个 argmax token。

生成 N 个 output tokens 后，最终 Cache 长度应为
`prompt_tokens + max(N - 1, 0)`；最后一个预测 token 尚未再次送入模型，因此不计入 Cache。

### 8.3 Stop behavior

- 新 token 等于 `eos_token_id`：将 EOS token 保留在 `output_token_ids` 中，停止原因是 `eos`。
- output 数达到 `max_new_tokens`：停止原因是 `length`。
- G1 固定 workload 以 32 为上限，预期四类 prompt 均产生 32 tokens；若固定 prompt 提前 EOS，
  必须先确认 HF 与 TinyInfer 都如此，再在报告中记录，不能静默更换 prompt。

## 9. Hugging Face 正确性基线

同一个 `LoadedModel` 上运行 `model.generate()`，参数固定为：

```python
do_sample=False
num_beams=1
use_cache=True
max_new_tokens=request.max_new_tokens
eos_token_id=loaded.eos_token_id
pad_token_id=loaded.pad_token_id
```

比较时只取生成序列中 prompt 后的 suffix。定义：

```text
token_agreement = matching_positions / max(reference_output_length, tinyinfer_output_length)
```

若长度不同，缺失位置视为不匹配。G1 只接受 reference 和 TinyInfer token tuple 完全相等，故
`token_agreement == 1.0`。报告必须记录首个分歧位置；不得以最终文本相同替代 token 相同。

对齐失败的排查顺序固定为：

1. tokenizer 输入 token IDs。
2. EOS/PAD ID。
3. Prefill 第一个 token。
4. attention mask 与 cache position。
5. 每步 Cache 长度。
6. 首个分歧步骤的 logits argmax。

## 10. 固定 Workloads

`benchmarks/prompts.json` 包含且只包含四个 G1 case：

| case_id | 类别 | 要求 |
|---|---|---|
| `short_en` | 短英文事实问题 | 约 8–32 prompt tokens |
| `short_zh` | 短中文解释问题 | 约 8–64 prompt tokens |
| `structured` | 多行结构化输入 | 至少 3 行，包含键值或列表 |
| `long_context` | 长上下文摘要/续写 | tokenizer 后不少于 512 tokens，不超过 768 tokens |

JSON 中保存 prompt 原文、`max_new_tokens: 32` 与类别，不保存预计算 token IDs。实际 token 长度由
固定 tokenizer 在验收时生成并写入结果，防止人工估算冒充实测。

## 11. Profiling 设计

### 11.1 测量边界

- 模型与 tokenizer 加载不计入推理指标。
- 每个 workload 先 warm-up 3 次，不写入正式 JSONL。
- 每个 workload 正式重复 5 次，共生成 20 条测量记录。
- 每次正式运行前调用 `torch.cuda.reset_peak_memory_stats()`。
- Host 使用 `time.perf_counter_ns()`。
- Device 使用 `torch.cuda.Event(enable_timing=True)`。
- CUDA event 读取前只在 `timer.py` 集中调用 synchronize。
- `InferenceTimer` 通过 `GenerationObserver` 回调记录边界，不在 benchmark 中复制生成循环。
- 首次下载、首次加载和首次 CUDA 初始化不计入 steady-state。

### 11.2 指标语义

| 字段 | 精确定义 |
|---|---|
| `ttft_host_ms` | 开始 tokenizer 到第一个输出 token 在 host 可读的时间 |
| `prefill_host_ms` | prefill forward 调用边界的 host elapsed |
| `prefill_device_ms` | prefill forward 对应 CUDA event elapsed |
| `decode_host_ms` | 第二个至最后一个 output token 的 decode host 总时间 |
| `decode_device_ms` | 同一区间 CUDA event 总时间 |
| `tpot_host_ms` | `decode_host_ms / max(output_tokens - 1, 1)` |
| `tpot_device_ms` | `decode_device_ms / max(output_tokens - 1, 1)` |
| `output_tokens_per_s` | `1000 * output_tokens / end_to_end_host_ms` |
| `peak_allocated_bytes` | 本轮 `torch.cuda.max_memory_allocated()` |
| `peak_reserved_bytes` | 本轮 `torch.cuda.max_memory_reserved()` |

Host 与 device 指标必须分字段保存，不得用一个无来源的 `prefill_ms` 混合两种口径。

### 11.3 JSONL schema v1

每条记录至少包含：

```json
{
  "schema_version": 1,
  "git_revision": "40-char commit SHA",
  "model_id": "Qwen/Qwen2.5-0.5B",
  "model_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
  "transformers_version": "5.14.1",
  "torch_version": "2.13.0+cu130",
  "dtype": "bfloat16",
  "device_name": "NVIDIA GeForce RTX 4080 SUPER",
  "case_id": "short_en",
  "repetition": 0,
  "warmup_runs": 3,
  "prompt_tokens": 0,
  "output_tokens": 0,
  "output_token_ids": [],
  "stop_reason": "length",
  "token_agreement": 1.0,
  "first_mismatch_index": null,
  "ttft_host_ms": 0.0,
  "prefill_host_ms": 0.0,
  "prefill_device_ms": 0.0,
  "decode_host_ms": 0.0,
  "decode_device_ms": 0.0,
  "tpot_host_ms": 0.0,
  "tpot_device_ms": 0.0,
  "output_tokens_per_s": 0.0,
  "peak_allocated_bytes": 0,
  "peak_reserved_bytes": 0
}
```

示例中的零值只用于说明类型，不是验收期望。正式记录中时间、吞吐量和显存必须为有限非负数；
对于实际执行了至少两个 output tokens 的 workload，decode 时间必须大于零。

## 12. CLI 设计

执行入口：

```bash
python benchmarks/bench_reference.py \
  --config configs/qwen2_5_0_5b.yaml \
  --prompts benchmarks/prompts.json \
  --output docs/results/g1_reference.jsonl \
  --warmup-runs 3 \
  --repetitions 5
```

CLI 行为：

1. 输出文件已存在时默认拒绝覆盖；显式 `--overwrite` 才允许替换。
2. 任何 workload agreement 小于 1.0 时退出码非零，在 stderr 输出 case ID、两侧 token IDs 与首个
   mismatch index；不得创建或替换 canonical JSONL。
3. 记录写入采用临时文件加原子 rename，避免中断后把半行 JSON 当作有效结果。
4. 成功时最后一行输出 `G1 REFERENCE RUNTIME: PASS`。
5. 失败时最后一行输出 `G1 REFERENCE RUNTIME: FAIL`，并在其前输出 case ID 与失败类别。

## 13. 错误模型与停止条件

### 13.1 明确异常

| 异常 | 使用场景 |
|---|---|
| `ValueError` | request/config/prompt schema 输入非法 |
| `ModelLoadError` | tokenizer/model/revision/device 加载失败 |
| `RuntimeInvariantError` | shape、Cache 长度、batch size 或 token 流不满足设计 |
| `AlignmentError` | TinyInfer 与 HF token tuple 不一致 |

CUDA OOM 保留原始 `torch.cuda.OutOfMemoryError` 类型并增加 case ID 上下文，不转换成 `length`。

### 13.2 必须停止

出现以下任一情况时停止 G1，不进入 profiling 结论或下一阶段：

- 实际模型 revision 与固定 SHA 不一致。
- 任一固定 prompt token agreement 小于 1.0。
- 同一 case 五次正式运行生成 token tuple 不一致。
- Prefill/Decode 未分离，或 decode 输入 token 数不等于 1。
- Cache 长度不满足精确 invariant。
- 模型加载被计入 steady-state 指标。
- JSONL 缺少 Git SHA、模型 SHA、环境身份或 Host/Device 计时来源。
- GPU OOM、NaN/Inf 指标、测试失败或 dirty worktree 无法解释。

## 14. 测试策略

### 14.1 Unit tests（默认 CPU CI）

不下载模型、不要求 CUDA：

- Request/Result 的空值、上下界、不可变性与序列化。
- 使用最小 fake tokenizer/model/cache 验证 prefill 和 decode 控制流。
- 验证 decode 每轮输入 `[1, 1]`、Cache 每轮增长 1。
- EOS 与 length 两种停止路径。
- fake host clock/CUDA timer adapter 的公式与除数边界。
- JSONL schema、有限数检查、mismatch index 和原子输出策略。

Fake 只模拟公开接口，不复制 Qwen 数学；真实正确性由 integration tests 负责。

### 14.2 Integration tests（WSL GPU + 本地模型）

使用 pytest markers：

```text
gpu: 需要 CUDA/BF16
model: 需要固定 Qwen 模型 revision
slow: 执行四类 prompt 或多次生成
```

分层执行：

1. Loader smoke：身份、dtype、device、eval、EOS/PAD。
2. 单 prompt prefill：第一个 token 与直接 forward argmax 一致。
3. 1/2/8/32 tokens decode：shape、Cache invariant、无梯度。
4. 四类 prompt HF agreement。
5. 相同 case 连续运行确定性。
6. 完整 benchmark JSONL schema 与数量。

默认 GitHub CPU CI 只执行 unit tests；G1 发布前必须在 WSL GPU 上运行全部 markers，并在报告中保存
命令、退出码和测试数量。

## 15. G1 Acceptance Criteria

G1 只有在以下条件全部产生当前提交上的新鲜证据后才能标记 `verified`：

1. `pytest` 默认套件退出码为 0。
2. GPU/model integration suite 退出码为 0。
3. Ruff check、Ruff format check、Mypy 退出码均为 0。
4. 模型与 tokenizer resolved revision 均为固定 SHA。
5. 四类固定 prompt 均执行，long case 的实际 prompt tokens 在 512–768 范围内。
6. 每类 `max_new_tokens=32`，reference 与 TinyInfer token tuple 完全相同。
7. 20 条正式记录的 `token_agreement` 全部为 `1.0`。
8. 每个 case 五次正式运行输出 token tuple 一致。
9. JSONL 恰有 20 条 schema v1 记录，逐行可由标准 JSON parser 解析。
10. Host/Device 计时、吞吐量、allocated/reserved VRAM 均存在且满足数值约束。
11. `docs/G1_REFERENCE_RUNTIME_REPORT.md` 记录硬件、软件、命令、Git SHA、模型 SHA、结果摘要与限制。
12. README 提供从已缓存模型运行 correctness/benchmark 的一组命令。
13. worktree 干净，所有变更位于独立分支，远端 CI 成功。

## 16. Definition of Done

- **功能**：固定 Qwen 模型可完成显式 prefill 与逐 token decode。
- **正确性**：四类 prompt 与 HF generate 达到 100% token agreement，结果可重复。
- **可观测性**：20 条规范 JSONL 含正确性、Host/Device 时间、吞吐量和显存。
- **工程质量**：TDD 提交历史清晰；pytest、Ruff、format、Mypy 与 CI 全绿。
- **文档**：workflow、README、G1 report 与 JSONL 字段一致，无旧状态矛盾。
- **安全**：仓库不含 Hugging Face token、代理凭据、模型权重或敏感绝对路径。
- **可复现**：model revision、Git revision、版本、GPU、warm-up 和 repetitions 已记录。
- **版本控制**：独立 worktree/branch；禁止强推；合并前新鲜验证。
- **完成边界**：只允许将 R4–R7 和 G1 标记完成；Phase 2、KV cache、batching、paged KV、API 和 Triton 优化仍为 `not started`。

## 17. 后续阶段入口

G1 verified 后，下一份独立设计才允许覆盖：

1. Runtime-owned contiguous KV cache adapter。
2. Request 状态机与 iteration-level continuous batching。
3. Paged block pool 与 block table。
4. 根据 G1/Phase 2 profiling 选择一个 Triton kernel。

G1 的性能数据是基线和瓶颈证据，不是性能优越性声明。
