# TinyInfer Phase 1 学习指南

> 面向当前项目代码的解释文档
>
> 当前覆盖范围：G0.5 环境基线，以及 Phase 1 Task 1–7
>
> 当前分支：`agent/hf-runtime-fix`
>
> 文档基线提交：`ebd9a00`

## 1. 这个项目现在究竟在做什么

TinyInfer 不是要重新制造一个完整的 vLLM。当前目标更小，也更适合作为硕士申请项目：

1. 理解一次 LLM 请求怎样经过 tokenizer、prefill 和逐 token decode。
2. 使用公开的 Hugging Face 模型建立可信的正确性基线。
3. 明确观察 KV Cache 如何随 decode 增长。
4. 用严格测试证明自己的显式生成循环与 Hugging Face 完全一致。
5. 建立后续 KV Cache、batching 和 GPU 优化所需的 profiling 基础。

当前 Phase 1 仍是“参考运行时”。它使用 Hugging Face 加载 Qwen 模型，也使用模型自身公开的
Cache 对象，但生成控制流由 TinyInfer 自己编排。

尚未实现的内容包括：

- 自己管理的 contiguous KV Cache；
- continuous batching 和 scheduler；
- paged KV Cache；
- API Server；
- 自定义 Triton 性能 kernel；
- sampling、量化和多 GPU。

这条边界非常重要。现在项目能够证明“理解并控制了推理过程”，但还不能声称已经实现了
vLLM 类运行时。

## 2. 当前进度地图

| 阶段 | 结果 | 对应提交 |
|---|---|---|
| G0.5 | WSL2、CUDA、PyTorch、Transformers、Triton smoke 基线 | `v0.1.0-g0.5` 基线 |
| Task 1 | 请求和结果数据契约 | `df84ed6` |
| Task 2 | 固定模型 revision 的 Hugging Face loader | `abd74b4` |
| Task 3 | 显式 prefill、decode 和 greedy generation | `feb3f99` |
| 修复 | 隔离缺少 GPU 依赖时的 pytest module import | `f1abbbb` |
| Task 4 | 冻结 workload，并证明四类 prompt token 完全一致 | `fe5ccdc` |
| Task 5 | Host/CUDA 分阶段计时和显存峰值观测 | `815af33` |
| Task 6 | token agreement 与 benchmark schema-v1 | `84d8572` |
| Task 7 | benchmark CLI 与 20 条正式记录 | `b5225f8` / `ebd9a00` |
| Task 8 | G1 报告与发布 Gate | 本地报告完成，等待远端 CI |

因此准确状态是：**Task 1–7 与 G1 本地 Gate 已实现；最终 `verified` 等待远端 CI。**

## 3. 一次文本生成的完整路径

当前运行时的数据流可以写成：

```text
GenerationRequest
  -> tokenizer(prompt)
  -> input_ids + attention_mask
  -> prefill(完整 prompt)
  -> 第一个 output token + KV Cache
  -> decode_one(上一个 token, Cache)
  -> 重复 decode，直到 EOS 或 max_new_tokens
  -> tokenizer.decode(output_token_ids)
  -> GenerationResult
```

这里最容易混淆的是：模型并不是一次返回完整答案。Causal LM 每次只预测“下一个 token”的概率。
生成循环选中一个 token，再把它作为下一轮输入的一部分。

### 3.1 Tokenizer

Tokenizer 把字符串转换成整数序列：

```text
"Explain KV cache."
  -> [token_0, token_1, token_2, ...]
```

模型只处理 token ID，不直接处理自然语言字符串。不同模型可能使用不同词表和切分规则，因此
long prompt 的长度必须用目标模型的真实 tokenizer 测量，不能按单词数估计。

Task 4 最初使用 24 段长文本，真实 Qwen tokenizer 得到 1042 tokens，超过验收要求的
512–768。经批准后改为 14 段，并重新通过真实 tokenizer Gate。这说明 benchmark 输入也属于
需要验证的工程资产。

### 3.2 Prefill

Prefill 把完整 prompt 一次送入模型：

```text
input shape: [batch=1, prompt_tokens]
```

它完成两件事：

1. 计算 prompt 最后位置的 logits，从中选择第一个输出 token；
2. 为 prompt 中所有位置建立 KV Cache。

当前代码要求：

```text
cache.get_seq_length() == prompt_tokens
```

如果不满足，说明模型返回的 cache 与输入长度不一致，运行时会立即失败，而不是继续产生可能
错误的 token。

### 3.3 Decode

Prefill 之后，每轮 decode 只输入一个新 token：

```text
input shape: [1, 1]
```

与此同时，运行时传入此前积累的 KV Cache。这样模型无需重新计算整个 prompt。

每轮 decode 都必须满足：

- attention mask 比上一轮增加一个位置；
- `cache_position` 指向调用前的 cache 末尾；
- 更新后的 cache 长度恰好增加 1；
- 只从当前轮最后位置的 logits 选择下一个 token。

生成 `N` 个 output tokens 后，最终 cache 长度是：

```text
prompt_tokens + max(N - 1, 0)
```

原因是第一个 output token 来自 prefill；最后预测出的 token 尚未再次输入模型，因此不会立即
写入 cache。

### 3.4 Greedy decoding

当前固定使用 greedy decoding：

```python
next_token_id = logits[:, -1, :].argmax(dim=-1)
```

也就是每次选择概率最大的 token。它没有温度、top-k 或 top-p 随机性，适合做确定性正确性
对齐。相同模型、revision、输入和数值环境应产生相同 token 序列。

## 4. KV Cache 到底缓存了什么

Transformer attention 会为每个历史 token 计算 Key 和 Value。若每次生成新 token 都重新计算
全部历史位置，计算量会不断增长。

KV Cache 保存历史 token 的 Key/Value：

```text
Prefill:
  prompt token 0 ... token P-1
  -> cache length P

Decode:
  输入 output token 0
  -> cache length P+1

下一轮:
  输入 output token 1
  -> cache length P+2
```

当前模型的公开 Cache 由 Transformers 管理。TinyInfer 负责验证它的长度和调用方式，但还没有
负责显存块分配、回收或分页。Phase 2 才会把这部分变成自己的系统组件。

## 5. 为什么还要和 Hugging Face 对齐

自己写出一个“能生成文本”的循环并不能证明它正确。attention mask、cache position、EOS 或
prompt/output 切片出现偏差时，输出仍可能看起来像自然语言。

Task 4 使用两条独立路径：

```text
TinyInfer:
  generate_greedy -> prefill -> decode_one -> token tuple

Reference:
  Hugging Face model.generate -> generated suffix token tuple
```

固定 workload 有四类：

- 短英文；
- 短中文；
- 结构化指令；
- 512–768 token 的长上下文。

验收比较的是整数 token tuple，而不是肉眼比较文本。文本解码可能隐藏特殊 token 或空格差异，
逐 token equality 才是强正确性证据。

当前四类 prompt 均达到：

```text
TinyInfer output_token_ids == Hugging Face output_token_ids
```

这证明在当前固定模型和 greedy 条件下，显式生成循环的控制流与参考实现一致。

## 6. Task 1–5 的代码结构

### `tinyinfer/runtime/request.py`

定义不可变的 `GenerationRequest`：

- `request_id`：请求身份；
- `prompt`：输入文本；
- `max_new_tokens`：最多生成多少新 token。

构造时拒绝空 ID、空 prompt 和非法 token 上限。`frozen=True` 防止请求执行过程中被修改，
`slots=True` 减少动态属性和意外拼写错误。

### `tinyinfer/runtime/result.py`

定义不可变的 `GenerationResult` 和 `StopReason`：

- 保存 prompt token；
- 保存 output token；
- 保存解码文本；
- 保存停止原因 `eos` 或 `length`。

prompt token 与 output token 被明确分开。后续正确性和 benchmark 只比较 output 部分。

### `tinyinfer/runtime/hf_loader.py`

负责加载固定的 Hugging Face 模型和 tokenizer。核心原则是：

- 模型 ID 固定；
- model revision 固定到完整 commit SHA；
- dtype 和 device 来自配置；
- tokenizer 与 model 使用同一 revision；
- EOS/PAD ID 在加载时验证并保存。

固定 revision 避免上游仓库更新后，同一份 TinyInfer 代码悄悄加载不同权重或配置。

### `tinyinfer/runtime/generation.py`

这是当前推理控制流核心，包含：

- `prefill`：完整 prompt forward；
- `decode_one`：单 token forward；
- `generate_greedy`：唯一的生成循环；
- `reference_generate`：调用 Hugging Face `generate()` 的正确性参考；
- `GenerationObserver`：不侵入生成逻辑的观测接口。

`prefill` 和 `decode_one` 都使用 `torch.inference_mode()`，避免建立梯度图。Observer 只接收
事件，没有返回值，因此不能替换 token、cache 或停止条件。

### `benchmarks/prompts.json`

保存 schema version 和四个冻结 workload。正式 benchmark 必须读取这里的输入，不能在运行时
为了得到“更好看”的结果临时换 prompt。

### `tinyinfer/profiling/timer.py`

`InferenceTimer` 实现七个 observer 回调：

1. `request_started`
2. `prefill_started`
3. `prefill_finished`
4. `first_token_ready`
5. `decode_started`
6. `decode_finished(output_tokens)`
7. `request_finished`

它是一个严格状态机。重复开始、跳过阶段或提前结束都会报错，从而防止产生边界不明的性能
数据。

## 7. Host 时间与 CUDA 时间为什么必须分开

GPU kernel launch 通常是异步的。CPU 发出命令后可能立刻继续执行，因此简单地在 Python 调用
前后计时，测到的不一定是 GPU 真正执行时间。

当前 profiler 同时保存：

- Host：`time.perf_counter_ns()`；
- Device：`torch.cuda.Event(enable_timing=True)`。

只在第一个 token 可读和请求结束两个集中边界执行 CUDA synchronize，避免到处同步破坏真实
执行行为。

### 7.1 十个指标

| 指标 | 含义 |
|---|---|
| `ttft_host_ms` | 从请求/tokenizer 开始到第一个 token 在 host 可读 |
| `prefill_host_ms` | prefill Python 调用边界的 host 时间 |
| `prefill_device_ms` | prefill CUDA Event 时间 |
| `decode_host_ms` | 第二个到最后一个输出 token 的 host 总时间 |
| `decode_device_ms` | 同一区间的 CUDA Event 总时间 |
| `tpot_host_ms` | `decode_host_ms / (output_tokens - 1)` |
| `tpot_device_ms` | `decode_device_ms / (output_tokens - 1)` |
| `output_tokens_per_s` | 全部 output tokens / 请求总 host 时间 |
| `peak_allocated_bytes` | 本请求峰值已分配显存 |
| `peak_reserved_bytes` | PyTorch allocator 本请求峰值保留显存 |

当请求只生成一个 token 时，没有发生 decode，所以 decode duration 和 TPOT 必须严格为 `0.0`，
不能用除以 1 的技巧制造一个看似有效的数。

### 7.2 TTFT 与 TPOT

TTFT（Time To First Token）主要受 tokenizer、prompt 长度和 prefill 影响。

TPOT（Time Per Output Token）描述首 token 之后的平均生成速度。对交互式应用：

- TTFT 决定用户多久看到回答开始；
- TPOT 决定回答随后“吐字”多快。

二者不能混成一个 latency，因为它们对应不同执行阶段和不同优化方向。

## 8. 为什么使用 Observer

最危险的 profiling 写法是复制一份生成循环，在复制版本里加 timer。这样项目会存在：

```text
正常生成循环
profiling 生成循环
```

两份逻辑迟早会在 EOS、cache 或 mask 上产生差异。当前设计只有一份 `generate_greedy`，timer
通过回调观察边界：

```text
generate_greedy
  -> observer.prefill_started()
  -> 原始 prefill
  -> observer.prefill_finished()
```

未传 observer 时使用 no-op observer，避免在核心循环里散布 `if observer is not None`。

## 9. 测试体系在证明什么

### Unit tests

使用 fake tensor、fake cache、fake clock 和 fake CUDA facade：

- 验证 cache/mask/position 不变量；
- 验证 EOS 和 length 停止；
- 验证七个 observer 事件顺序；
- 用手算数字验证 TTFT、TPOT 和 throughput；
- 验证错误回调顺序会失败。

Fake 的目的不是假装 GPU 正确，而是让边界条件得到精确、快速、确定性的测试。

### Integration tests

在 WSL2 + RTX 4080 SUPER 上加载真实 Qwen：

- 验证固定 loader；
- 验证 1/2/8/32 token 生成；
- 验证四类 workload 与 Hugging Face token 对齐；
- 验证真实 CUDA Event、同步和显存采样。

Unit test 证明控制逻辑，integration test 证明真实依赖能够共同工作，两者不能互相替代。

## 10. Windows 与 WSL2 的职责

Windows 适合：

- PyCharm、文件浏览和普通 Git 操作；
- 不需要 torch/CUDA 的 CPU unit tests；
- 文档维护。

WSL2 Ubuntu 负责：

- Linux 版 PyTorch CUDA；
- 官方 Triton；
- 真实模型加载和 GPU integration tests；
- 后续 benchmark 和 kernel 工作。

Windows Conda 环境没有安装 pytest/ruff 并不代表项目损坏，只代表调用了不属于本项目的
解释器。当前可用的 CPU 工具环境是 `reactos`；GPU 验收使用 WSL 项目的 `.venv`。

## 11. Task 6：把结果变成可审计记录

Task 6 不实现新推理算法，而是把正确性和性能结果变成了严格的数据契约：

- `compare_token_ids`：计算 token agreement 和首个 mismatch；
- `TokenComparison`：不可变的比较结果；
- `BenchmarkRecord`：不可变的 schema-v1 正式记录；
- `to_dict()`：生成可直接写入 JSONL 的数据。

Agreement 的分母使用两侧序列的较长长度：

```text
agreement = 相同位置且相等的 token 数 / max(actual 长度, reference 长度)
```

因此长度不一致会自然降低 agreement。如果共同前缀完全相同，但一侧更长，首个 mismatch index
就是共同长度边界。例如：

```text
actual:    [3, 5]
reference: [3, 5, 8]

agreement = 2 / 3
first_mismatch_index = 2
```

`TokenComparison` 可以表示一般的成功或失败比较；`BenchmarkRecord` 则只允许进入正式 canonical
JSONL 的成功记录，所以它强制：

- `token_agreement == 1.0`；
- `first_mismatch_index is None`；
- Git revision 是 40 位小写十六进制 SHA；
- output token 数与 token tuple 长度一致；
- 所有时间有限且非负；
- 多 token 请求的 decode host/device 时间大于零；
- 所有计数器非负；
- `to_dict()` 的 key 与 schema-v1 完全一致，并把 tuple 转换成 JSON array。

这种区分可以防止失败的对齐结果被误写成正式 benchmark 证据。结构化记录能够被测试、汇总、
复查，并在技术报告中追溯到模型 SHA、Git SHA、prompt、token 和环境。

## 12. Task 7：从一次运行到 canonical evidence

`benchmarks/bench_reference.py` 把前面的 runtime、reference、timer 和 record 串成一个可复现
实验：

```text
加载模型一次
  -> 每个 case 计算一次 HF reference
  -> warm-up 3 次（不记录）
  -> 正式运行 5 次（每次独立 timer）
  -> 每次检查 agreement
  -> 检查同一 case 五次 token tuple 一致
  -> 构造 20 条 BenchmarkRecord
  -> 写入 sibling .tmp
  -> fsync + os.replace
```

CLI 在开始模型加载前拒绝意外覆盖已有文件。只有显式 `--overwrite` 才允许替换；即便允许覆盖，
alignment 或序列化中途失败时，旧 canonical 文件仍保持不变。

`BenchmarkDependencies` 把 loader、TinyInfer runner、HF runner、timer、record factory、环境身份和
Git SHA provider 集中注入。CPU unit tests 因此可以精确验证 3 次 warm-up、5 次正式运行、输出
顺序和失败行为，而真实 GPU Gate 使用默认依赖。

代码和结果采用两个提交也是有意设计：

- `b5225f8` 包含实际被测试和运行的 CLI 代码；
- `ebd9a00` 只增加记录 `b5225f8` 完整 SHA 的 JSONL。

如果代码和“记录当前提交 SHA”的 JSONL 强行放在同一个提交里，就会形成无法实现的
self-reference：运行 benchmark 时最终提交 SHA 还不存在。

Canonical benchmark 在原生 WSL 文件系统上运行，而不是 `/mnt/d`。结果是四类各 5 条、
agreement 全部 1.0；long case prompt 为 612 tokens，并与 HF 一致地在 26 个 output tokens
产生 EOS。

## 13. 建议的学习顺序

1. 阅读 `request.py` 和 `result.py`，理解为什么先冻结输入输出契约。
2. 阅读 `generation.py::prefill`，画出 input/logits/cache 的 shape。
3. 阅读 `decode_one`，手算三轮 cache length 和 attention mask length。
4. 阅读 `generate_greedy`，找出 EOS 与 length 两条停止路径。
5. 对照 `reference_generate`，解释为什么只切 generated suffix。
6. 阅读 `test_generation.py`，将每个 fake 对象对应到真实 PyTorch 对象。
7. 阅读 `timer.py`，画出七个 callback 的状态转换。
8. 手算 `test_timer.py` 中的 TTFT、TPOT 和 throughput，再运行测试核对。
9. 阅读 `record.py`，解释为什么失败比较不能进入 canonical record。
10. 阅读 `bench_reference.py`，追踪一个 case 从 reference 到五条 JSONL 的过程。
11. 最后阅读 integration tests 和 G1 report，理解哪些结论必须由真实 GPU 才能证明。

## 14. 你现在应该能够回答的问题

- prefill 和 decode 的输入 shape 为什么不同？
- 为什么第一个 output token 来自 prefill？
- 为什么最终 cache 不包含最后一个刚预测出的 token？
- 为什么比较文本不如比较 token tuple 严格？
- 为什么模型 revision 必须固定到 SHA？
- 为什么 CUDA 性能不能只用普通 Python timer？
- TTFT 和 TPOT 分别对应什么用户体验？
- Observer 为什么能避免复制生成循环？
- 为什么 canonical JSONL 与被测试代码必须分成两个提交？
- 原子 `.tmp` + `os.replace` 防止了什么损坏？
- 当前项目为什么还不能声称实现了自己的 KV Cache？

如果这些问题能够用自己的话回答，并能指出对应代码和测试，Phase 1 的学习目标就不仅是
“代码跑通”，而是形成了可解释、可验证的系统理解。
