# TinyInfer 项目总计划：研究、学习与实现路线

> 项目全名：**TinyInfer: A Hackable Single-GPU LLM Inference Engine with Paged KV Cache and Triton-Optimized Decoding**  
> 文档版本：v0.1  
> 制定日期：2026-07-15  
> 目标周期：20 周（每周约 12–18 小时）

## 1. 项目定位

TinyInfer 是一个面向学习、研究和作品集展示的单 GPU 大语言模型推理引擎。它不追求覆盖 vLLM/SGLang 的全部生产能力，而是用一套规模可控、接口清晰、可逐层验证的实现，回答三个问题：

1. 一次自回归 LLM 请求如何从 tokenizer 经过 prefill、decode、KV Cache 和 sampling 生成文本？
2. continuous batching 与 paged KV cache 如何影响吞吐、延迟和显存利用率？
3. 在 RTX 4080 SUPER 16GB 这类消费级 GPU 上，能否通过针对 decode 场景的 Triton kernel 或运行时策略，获得可复现的性能收益？

项目首先是一项高质量系统工程，其次才是研究项目。只有当标准功能全部正确、benchmark 可复现并且 profiling 显示稳定瓶颈后，才确定唯一的论文扩展方向。

### 1.1 最终成果

- 一个可读、可测试的单 GPU LLM inference engine。
- 支持 Qwen2.5 Dense 模型家族的基础推理。
- 支持 prefill/decode 分离、continuous batching、paged KV cache 和流式生成。
- 至少一个自己实现并通过正确性验证的 Triton kernel。
- 一个 OpenAI-compatible 的最小 HTTP 服务。
- 一套与 Hugging Face Transformers、nano-vLLM 和 vLLM 对比的 benchmark。
- 一篇完整的英文 technical report，以及一篇适合作品集展示的技术博客。
- 若研究扩展产生稳定结论，再独立整理 workshop 或 systems venue 投稿。

### 1.2 明确不做的事情

TinyInfer V1 不支持：

- Agent、工具调用或多 Agent runtime。
- 分布式推理、tensor parallel、pipeline parallel。
- MoE 和多模态模型。
- speculative decoding。
- prefix caching、LoRA serving 和复杂 beam search。
- 多种量化权重格式。
- CPU/NVMe KV offload。
- 对所有 Hugging Face 模型的通用兼容。
- 生产级鉴权、路由、租户隔离和故障恢复。

这些能力均不进入前 20 周主线，避免项目退化为功能清单。

## 2. 项目价值与投稿判断

### 2.1 对申请的价值

该项目能够同时展示：

- 对 Transformer inference 数据流的理解。
- 对 GPU memory hierarchy、算子和 profiling 的理解。
- 系统模块划分、测试、benchmark 和可复现工程能力。
- 从基线复现到提出研究问题的完整研究过程。

对美硕申请而言，完成度高、实验可信、文档清晰的 TinyInfer，比一个功能很多但无法解释性能来源的“大型 serving 平台”更有说服力。

### 2.2 投稿定位

TinyInfer 本身属于 ML systems / inference systems，而不是典型 ACL 主会工作。仅实现简化版 vLLM 不构成论文创新。合理路线是：

1. 先把 V1 做成强作品集和技术报告。
2. 从真实 profiling 中提出一个窄而明确的系统研究问题。
3. 若结果具有稳定收益和可迁移结论，优先考虑 ML systems、systems for ML 或相关 workshop，而不是强行包装成 NLP 方法论文。

## 3. 技术路线选择

### 3.1 三条可选路线

| 路线 | 核心内容 | 优点 | 风险 | 结论 |
|---|---|---|---|---|
| 教学引擎 | PyTorch reference + KV cache + continuous batching + paged KV | 最容易完成，结构清楚 | 与 nano-vLLM 重合，研究性弱 | 作为必做基础 |
| Kernel-first | 尽早实现 Triton paged attention/decode kernel | GPU 技术信号强 | 容易因正确性和调优困难停滞 | 不作为起点 |
| Profile-guided | 先完成正确引擎，再依据 profiling 选择 kernel、调度或内存扩展 | 风险最低，研究叙事真实 | 需要严格执行阶段门槛 | **采用** |

### 3.2 推荐主线

采用 **Profile-guided TinyInfer**：

```text
正确性基线
  → 朴素 KV Cache
  → Continuous Batching
  → Paged KV Cache
  → 标准化 Benchmark + Profiling
  → 选择唯一研究扩展
  → 技术报告 / 投稿判断
```

研究扩展的默认优先级为：

1. **Triton-optimized GQA paged decode kernel**：最能体现 GPU 能力。
2. **Memory-pressure-aware admission scheduling**：在 16GB 显存约束下优化 goodput/p95。
3. **Chunked prefill policy**：若长 prompt 明显阻塞 decode，再研究 prefill chunk 大小与调度。

最终只能选择其中一项作为论文主张；其他项目最多作为工程功能或消融对照。

## 4. 系统架构

```text
Client / Benchmark Driver
          │
          ▼
OpenAI-compatible API + Streaming
          │
          ▼
Request Queue ──► Scheduler ──► Running Batch
                         │
                         ▼
                    LLM Engine
                ┌────────┴────────┐
                ▼                 ▼
             Prefill            Decode
                │                 │
                └────────┬────────┘
                         ▼
                  Model Executor
             ┌───────────┼───────────┐
             ▼           ▼           ▼
         Model Layers  KV Manager   Sampler
                           │
                  Block Table + GPU Pool
                           │
                           ▼
             Reference / Triton Attention
```

### 4.1 模块职责

| 模块 | 单一职责 | 关键输出 |
|---|---|---|
| Config | 模型、显存、block 和调度参数校验 | 不可变配置对象 |
| Model Loader | 加载 safetensors 并映射 Qwen2.5 权重 | TinyInfer 模型实例 |
| Model Executor | 执行 prefill 或单步 decode | logits 与更新后的 KV |
| Sequence | 保存请求状态、token 和 block table | 可调度序列状态 |
| Scheduler | 决定本轮 prefill/decode 请求集合 | iteration batch |
| KV Cache Manager | 分配、回收和查询物理 KV blocks | block table 与利用率指标 |
| Attention Backend | reference、paged 和 Triton 后端统一接口 | attention output |
| Sampler | greedy、temperature、top-k、top-p | next token |
| Engine | 驱动迭代级执行与生命周期 | token events |
| API Server | HTTP 请求、取消与流式响应 | OpenAI-style response |
| Profiler/Benchmark | 生成 workload 并采集性能数据 | JSONL/CSV 原始结果 |

### 4.2 建议仓库结构

```text
TinyInfer/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   └── qwen2_5_1_5b.yaml
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── ARCHITECTURE.md
│   ├── RESEARCH_LOG.md
│   └── BENCHMARK_PROTOCOL.md
├── tinyinfer/
│   ├── config.py
│   ├── engine/
│   │   ├── engine.py
│   │   ├── request.py
│   │   ├── scheduler.py
│   │   └── sequence.py
│   ├── model/
│   │   ├── loader.py
│   │   ├── qwen2.py
│   │   └── layers.py
│   ├── cache/
│   │   ├── interface.py
│   │   ├── contiguous.py
│   │   └── paged.py
│   ├── kernels/
│   │   ├── reference.py
│   │   └── paged_decode.py
│   ├── sampling/
│   │   └── sampler.py
│   └── server/
│       ├── app.py
│       └── schemas.py
├── benchmarks/
│   ├── workloads.py
│   ├── bench_offline.py
│   ├── bench_serving.py
│   └── analyze.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── kernels/
│   └── correctness/
└── reports/
    ├── figures/
    └── technical_report.md
```

每个文件只承担一种职责。模型代码、缓存管理、调度和 kernel 必须通过接口隔离，保证替换 paged cache 或 Triton backend 时无需改动 API 层。

## 5. 全局技术约束

- 主开发环境：Linux 或 WSL2；不以原生 Windows CUDA/Triton 兼容性为研究内容。
- Python：3.11。
- GPU：本地 RTX 4080 SUPER 16GB；4090/A100 只用于最后的跨硬件验证。
- 模型：第一阶段固定 `Qwen/Qwen2.5-0.5B`，第二阶段固定 `Qwen/Qwen2.5-1.5B`；3B 仅在核心功能稳定后加入。
- 数值类型：初始使用 BF16；不在 V1 主线加入权重量化。
- 推理模式：单 GPU、decoder-only、batch-first、无梯度。
- 测试优先：任何优化都必须先有 reference 实现和数值对照。
- Benchmark 必须保存原始数据、软件版本、GPU 信息、随机种子和完整命令。
- 不复制 nano-vLLM/vLLM 核心实现；可以阅读论文和接口，在自己的设计完成后再进行源码对照。
- 不宣称性能领先，除非在相同模型、相同输入、相同硬件和相同 SLA 下得到可重复结果。

## 6. 学习路线

### 6.1 第一层：Transformer 推理基础

必须掌握：

- decoder-only Transformer 数据流。
- RMSNorm、RoPE、SwiGLU、GQA。
- causal self-attention 的形状变化。
- prefill 与 decode 的计算差异。
- KV Cache 的张量布局和显存公式。
- greedy、temperature、top-k 和 top-p sampling。

学习产出：

- 手写一页 Qwen2.5 单层前向传播张量形状表。
- 写出 KV Cache 显存估算公式，并用 0.5B/1.5B 模型参数验证。
- 用 Hugging Face hook 或 profiler 记录一次 prefill 和 32 步 decode 的算子时间。

### 6.2 第二层：GPU 与 Triton 基础

必须掌握：

- GPU 的 thread/block/warp 基本模型。
- global memory、shared memory、register 和 cache 的差异。
- coalesced memory access、occupancy 和 roofline 思维。
- kernel launch overhead 与算子融合。
- Triton 的 program id、block pointer、mask、reduction 和 autotune。

学习顺序：

1. Triton vector addition。
2. Fused softmax。
3. Matrix multiplication。
4. LayerNorm/RMSNorm。
5. Fused attention。
6. 自己实现 KV gather/append microkernel。

每个练习都要同时提交：正确性测试、benchmark、输入形状说明和 profiler 截图或导出结果。

### 6.3 第三层：Serving Systems

必须掌握：

- request-level batching 与 iteration-level scheduling。
- continuous batching。
- TTFT、TPOT、ITL、E2E latency、throughput 和 goodput。
- KV 内部碎片、外部碎片与 block table。
- admission control、抢占和重计算的权衡。
- prefill 的 compute-bound 特征与 decode 的 memory-bound 特征。

### 6.4 推荐阅读顺序

1. [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)：先理解 iteration-level scheduling。
2. [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)：理解 block-based KV memory management。
3. [FlashAttention](https://arxiv.org/abs/2205.14135) 与 [FlashAttention-2](https://arxiv.org/abs/2307.08691)：理解 IO-aware attention 和 GPU work partition。
4. [Sarathi-Serve](https://arxiv.org/abs/2403.02310)：理解 chunked prefill 和 latency-throughput trade-off。
5. [vAttention](https://arxiv.org/abs/2405.04437)：理解 PagedAttention 之外的虚拟内存方案。
6. [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)：按教程顺序完成基础 kernel。
7. [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)：在自己的 scheduler/cache 设计完成后阅读，用于查漏补缺而不是照抄。
8. [Qwen2.5-1.5B 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)：核对 GQA、RoPE、层数和权重结构。

阅读每篇论文时只回答五个问题：

1. 它解决的瓶颈是什么？
2. 它控制了哪个系统变量？
3. 性能收益来自算法、内存、调度还是 kernel？
4. 它的 baseline 和 workload 是否公平？
5. 哪个假设在 4080 16GB 环境下可能不成立？

## 7. 二十周路线图

### Phase 0：环境与测量基础（第 1–2 周）

目标：建立之后所有实验可复现的开发环境和测量方式。

- [ ] 在 WSL2/Linux 中验证 PyTorch、CUDA、Triton 和 Nsight Systems。
- [ ] 固定 Python 与依赖版本，生成 lock file。
- [ ] 记录 GPU 型号、显存、驱动、CUDA、PyTorch 和 Triton 版本。
- [ ] 下载 Qwen2.5-0.5B 和 tokenizer，运行 Hugging Face greedy baseline。
- [ ] 建立 pytest、ruff、类型检查和 benchmark 输出目录。
- [ ] 完成 Triton vector add、softmax、matmul 三个练习。
- [ ] 编写第一版 `BENCHMARK_PROTOCOL.md`。

阶段门槛 G0：同一输入连续运行三次可得到一致的 greedy token；benchmark 能输出 JSONL 原始记录。

### Phase 1：Reference Model（第 3–5 周）

目标：不使用 KV Cache，独立实现 Qwen2.5 forward 和 autoregressive generation。

- [ ] 解析模型 config 和 safetensors 权重。
- [ ] 实现 RMSNorm、RoPE、GQA attention、SwiGLU 和 decoder layer。
- [ ] 实现 causal mask 和完整序列 forward。
- [ ] 实现 greedy sampler。
- [ ] 逐层与 Hugging Face 对齐 hidden states 和最终 logits。
- [ ] 建立 1、8、32、128 token 输入的数值测试。

阶段门槛 G1：FP32 算子测试使用 `atol=1e-5, rtol=1e-4`；BF16 模型级 logits cosine similarity 不低于 `0.999`，固定测试集中 top-1 token agreement 为 `100%`，且每个 prompt 连续 greedy 生成 32 tokens 与 Hugging Face 一致。若未通过，不进入 KV Cache。

### Phase 2：Contiguous KV Cache（第 6–7 周）

目标：理解并验证 prefill/decode 分离。

- [ ] 定义统一 `AttentionBackend` 与 `KVCache` 接口。
- [ ] 实现每请求连续 KV tensor。
- [ ] 实现 prefill 写入与 decode append。
- [ ] 验证 cached decode 与每步 full recomputation 的 logits 一致。
- [ ] 测量不同 context length 下的 TPOT 和峰值显存。
- [ ] 完成 KV Cache 理论显存与实测显存对照。

阶段门槛 G2：cache 模式的 greedy 输出与非 cache 模式一致；decode TPOT 相比 full recomputation 有稳定改善。

### Phase 3：Continuous Batching（第 8–10 周）

目标：实现迭代级请求调度，而不是一次处理完整请求。

- [ ] 定义 waiting、running、finished、cancelled 状态机。
- [ ] 实现 iteration-level FCFS scheduler。
- [ ] 支持每轮加入新请求和移除完成请求。
- [ ] 支持不同 prompt/output length 的 ragged batch 元数据。
- [ ] 支持请求取消和资源回收。
- [ ] 实现 synthetic arrival workload 和并发 benchmark。
- [ ] 采集 TTFT、TPOT、E2E、吞吐和队列等待时间。

阶段门槛 G3：并发 1、4、8、16 下无状态泄漏；完成顺序和 token 结果可解释；持续运行 30 分钟无显存增长。

### Phase 4：Paged KV Cache（第 11–13 周）

目标：实现逻辑 token 到物理 block 的映射和安全资源管理。

- [ ] 实现固定大小 GPU KV block pool。
- [ ] 实现 free list、allocate、append、free 和 block table。
- [ ] 实现跨 block 边界的 KV gather/reference attention。
- [ ] 对 block 生命周期建立属性测试。
- [ ] 测量 block size 为 8、16、32、64 时的碎片和性能。
- [ ] 对比 contiguous 与 paged cache 的可服务并发数和峰值显存。
- [ ] 增加 OOM admission rejection，禁止静默破坏正在运行的请求。

阶段门槛 G4：随机请求到达/结束序列下无重复分配和泄漏；paged 与 contiguous logits 对齐；碎片指标能够解释 block size 差异。

### Phase 5：Triton Paged Decode（第 14–16 周）

目标：为 GQA decode 场景实现自己的 Triton kernel。

- [ ] 先实现独立 KV append kernel，并与 PyTorch reference 对齐。
- [ ] 实现 paged KV gather microbenchmark。
- [ ] 实现单 query 的 paged decode attention prototype。
- [ ] 覆盖不同 head dimension、context length、block size 和 batch size。
- [ ] 测试非整 block、空闲 block 和最大 context 边界。
- [ ] 使用 Nsight Systems/Torch Profiler 定位 launch 和 memory bottleneck。
- [ ] 只针对实测常见形状设置 autotune configs，避免无边界调参。

阶段门槛 G5：kernel 在所有 correctness cases 下通过；只有在至少一个预先定义的 workload 区间稳定优于 reference backend 后，才写入性能结论。未提速也要保留负结果分析。

### Phase 6：服务接口与完整 Benchmark（第 17–18 周）

目标：把内部引擎变成可演示、可比较的系统。

- [ ] 实现最小 `/v1/completions` 或 `/v1/chat/completions` 接口。
- [ ] 实现 Server-Sent Events 流式 token。
- [ ] 正确处理 disconnect、cancel 和 engine exception。
- [ ] 建立固定 prompt/output 长度与真实文本两类 workload。
- [ ] 对比 Transformers、nano-vLLM、vLLM 和 TinyInfer 各阶段版本。
- [ ] 输出带置信区间或至少 5 次重复实验的图表。

阶段门槛 G6：一条命令启动服务、一条命令复现实验；README 中的结果能够由保存的原始数据重新生成。

### Phase 7：研究扩展与报告（第 19–20 周及以后）

目标：依据数据选择研究方向，而不是先写结论。

- [ ] 汇总各 workload 的 GPU 利用率、kernel time、KV 碎片和 scheduler waiting。
- [ ] 判断主要瓶颈属于 kernel、内存容量还是调度。
- [ ] 从第 3.2 节候选中选择唯一扩展。
- [ ] 写清研究问题、假设、控制变量、baseline 和反例 workload。
- [ ] 完成消融实验和跨 4080/4090 验证。
- [ ] 完成英文 technical report 和 8–10 分钟演示视频。
- [ ] 根据新颖性与结果强度决定只做作品集、投 workshop，或继续扩展。

## 8. 正确性协议

性能优化之前必须通过以下四层验证：

1. **算子级**：RMSNorm、RoPE、attention、MLP 和 sampler 对齐 PyTorch reference。
2. **层级**：同权重、同输入下对齐每层 hidden state。
3. **模型级**：对齐最终 logits 和 greedy tokens。
4. **系统级**：single request、batched、continuous、contiguous KV、paged KV 和 Triton backend 在确定性设置下输出一致。

推荐记录：

- max absolute error。
- mean absolute error。
- cosine similarity。
- top-1 token agreement。
- 连续生成 N 个 token 的首个分歧位置。

统一验收阈值沿用 G1：FP32 算子级 `atol=1e-5, rtol=1e-4`；BF16 模型级 logits cosine similarity `>=0.999`。若个别 fused kernel 因归约顺序无法达到该阈值，必须在实验记录中给出误差分布，并继续满足固定测试集 top-1 token agreement `100%`，不得直接放宽全局阈值。

任何性能表必须标注所用 backend 和对应 correctness test commit。

## 9. Benchmark 设计

### 9.1 Baselines

- Hugging Face Transformers：正确性与朴素 serving 基线。
- PyTorch SDPA：优化 attention reference。
- nano-vLLM：轻量引擎工程基线。
- vLLM：成熟 serving 系统基线。
- TinyInfer ablations：无 KV、contiguous KV、paged KV、Triton backend。

### 9.2 Workload 矩阵

| 维度 | 建议取值 |
|---|---|
| 模型 | Qwen2.5 0.5B、1.5B；稳定后增加 3B |
| Prompt tokens | 128、512、2048、4096 |
| Output tokens | 32、128、512 |
| 并发 | 1、4、8、16、32；以不 OOM 为上限 |
| Block size | 8、16、32、64 tokens |
| 到达模式 | 全部同时、固定速率、Poisson arrival |
| Sampling | greedy 为主；temperature/top-p 只做功能验证 |

不需要穷举笛卡尔积。先用小规模实验筛出短 prompt、长 prompt、decode-heavy 和 mixed 四种代表性 workload，再进行正式重复实验。

### 9.3 核心指标

- TTFT：time to first token。
- TPOT：time per output token。
- ITL：inter-token latency。
- E2E latency：请求端到端延迟。
- Throughput：output tokens/s、total tokens/s、requests/s。
- Goodput：满足预设 TTFT/TPOT SLO 的请求率。
- Peak allocated/reserved VRAM。
- KV block utilization 与 internal fragmentation。
- Scheduler queue time。
- GPU utilization、kernel duration 和 memory throughput。

vLLM 也将 TTFT、TPOT 和 inter-token latency 作为核心 serving 指标，可用其[指标文档](https://docs.vllm.ai/en/v0.10.1/design/metrics.html)统一术语。

### 9.4 公平性规则

- 相同 GPU、模型权重、dtype、prompt tokens 和 output tokens。
- 相同的 warm-up 次数与测量时段。
- 关闭随机 sampling 或固定随机种子。
- 区分模型加载时间和 steady-state 推理时间。
- 报告 OOM，不通过缩短输入隐藏失败。
- 至少重复 5 次并报告均值、标准差或置信区间。
- 性能比较同时报告 latency 和 throughput，禁止只选择有利指标。

## 10. 研究扩展选择协议

到达 G6 后，用下面规则选择方向：

### 10.1 若 decode kernel 占主要时间

选择 **Triton GQA Paged Decode**：研究不同 context distribution、block size 和 GQA ratio 下的 kernel specialization。核心问题是固定配置与 shape/workload-aware 配置之间的 TPOT 和 goodput 差异。

### 10.2 若显存碎片限制并发

选择 **KV Memory Policy**：研究 block size、请求长度估计与 admission policy 的联合影响。主指标为可服务并发、KV utilization、p95 latency 和 goodput。

### 10.3 若长 prefill 阻塞 decode

选择 **Chunked Prefill Scheduling**：研究 chunk size 与 decode batch 的组合策略。Sarathi-Serve 已提出 chunked prefill，因此新工作不能只复现该机制，必须聚焦消费级 GPU 或特定 workload 下的自适应策略。

### 10.4 停止条件

以下任一情况发生时，不包装成论文创新：

- 收益只在一个手工挑选的 shape 上出现。
- 与成熟 baseline 不在相同输入或 SLA 下比较。
- 结果无法跨 5 次运行稳定复现。
- 改进来自关闭 baseline 功能或降低数值精度，却未明确说明。
- 主要贡献只是代码更少或接口更简单。

## 11. 风险与应对

| 风险 | 早期信号 | 应对策略 |
|---|---|---|
| 范围失控 | 不断加入量化、TP、prefix cache | V1 非目标清单视为硬约束 |
| Triton 学习停滞 | 两周仍不能写正确 softmax/matmul | 回到 microkernel，不直接写 attention |
| 数值错误难定位 | 只比较最终文本 | 增加逐层 hidden state 对齐 |
| benchmark 不可信 | 每次结果波动大、只跑一次 | warm-up、重复实验、记录频率与后台负载 |
| 无法超过 vLLM | 成熟系统在所有指标都更快 | 强调可解释性与负结果，缩小研究问题 |
| 16GB OOM | 直接从 7B BF16 起步 | 0.5B 验证、1.5B 正式实验、3B 扩展 |
| Windows/Triton 环境问题 | kernel 安装和编译反复失败 | 主环境固定 WSL2/Linux |
| “复刻”缺乏创新 | 功能与 nano-vLLM 高度一致 | 以 profiler 驱动的窄研究问题作为扩展 |

## 12. 每周研究工作流

每周固定完成一个闭环：

1. 周初写下本周唯一问题和验收标准。
2. 先补 reference test，再实现功能或优化。
3. 保存原始 benchmark 和 profiler 数据。
4. 写 300–500 字研究日志：做了什么、观察到什么、不能说明什么。
5. 更新一张核心图或一张架构图。
6. 周末做 go/no-go 判断，未过门槛不进入下一阶段。

研究日志模板：

```markdown
## YYYY-MM-DD / Experiment ID

- Question:
- Hypothesis:
- Hardware and software:
- Model and workload:
- Changed variable:
- Controlled variables:
- Result:
- Interpretation:
- Alternative explanation:
- Next decision:
```

日志中的空字段仅是每次实验的填写模板，不代表项目需求未确定。

## 13. 前十四天行动清单

### 第 1 周：建立测量能力

- [ ] 确认 WSL2/Linux 能识别 RTX 4080 SUPER。
- [ ] 建立 Python 3.11 环境并安装兼容的 PyTorch/Triton。
- [ ] 将最终依赖版本写入 `pyproject.toml` 和 lock file。
- [ ] 运行 Qwen2.5-0.5B Hugging Face greedy generation。
- [ ] 使用 profiler 区分一次 prefill 和后续 decode。
- [ ] 完成 Triton vector add 与 fused softmax。
- [ ] 写出 Qwen2.5 一层的所有张量形状。

### 第 2 周：建立推理数学基础

- [ ] 手写 RMSNorm、RoPE 和 GQA reference functions。
- [ ] 分别与 PyTorch/Hugging Face 中间结果对齐。
- [ ] 计算 0.5B、1.5B、3B 在不同 context 下的 KV 理论显存。
- [ ] 阅读 Orca 与 PagedAttention，分别写一页笔记。
- [ ] 画出 request state machine 和 block table 示例。
- [ ] 冻结 Phase 1 的模块接口，随后开始 reference model。

十四天交付物：可复现环境、三份 Triton/算子练习、第一版 profiler 结果、两篇论文笔记、模型张量形状表和 KV 显存表。

## 14. 最终作品集结构

README 首页只呈现六件事：

1. 一句话定位。
2. 一张系统架构图。
3. 一条最小运行命令。
4. correctness 说明。
5. 一张核心性能图。
6. 一张 roadmap/feature matrix。

技术报告建议结构：

1. Motivation。
2. Background: prefill/decode、continuous batching、KV cache。
3. TinyInfer architecture。
4. Paged KV design。
5. Triton decode design 或最终研究扩展。
6. Experimental methodology。
7. Results and ablations。
8. Limitations and negative results。
9. Related work。
10. Conclusion。

## 15. 项目成功标准

### 最低成功

- 独立实现 Qwen2.5-0.5B/1.5B 正确生成。
- contiguous KV、continuous batching 和 paged KV 全部通过测试。
- 有完整 benchmark、架构文档和技术报告。

### 理想成功

- Triton paged decode kernel 在明确 workload 区间获得稳定收益。
- TinyInfer 在 4080 上展示清晰的吞吐—延迟—显存权衡。
- 结果可在 4090 或另一张 GPU 上复现趋势。
- 仓库可被不了解项目的人在一天内读懂并运行。

### 研究级成功

- 形成一个不是“简化 vLLM”的独立研究问题。
- 相对强 baseline 获得可重复的 goodput/p95 或显存收益。
- 消融实验解释收益来源，并清楚报告无效区间。
- 具备向 ML systems 相关 workshop 或更高层次 venue 迭代的基础。

## 16. 当前最重要的原则

> 先证明正确，再证明更快；先测量瓶颈，再提出优化；先完成一个清晰的单 GPU 引擎，再讨论完整 serving system。

TinyInfer 的竞争力不来自功能数量，而来自你能够解释每一层设计、验证每一项优化，并诚实说明它在什么条件下有效。
