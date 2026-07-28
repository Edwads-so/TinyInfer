# TinyInfer G1 Reference Runtime Report

> 报告日期：2026-07-28
>
> 状态：`verified` — 本地 CPU、GPU、correctness、benchmark、代码评审、feature/main CI
> 与 fast-forward merge Gate 全部通过
>
> tested revision：`b5225f8e7c70d981c0c19015513594f28f163063`
>
> canonical evidence commit：`ebd9a00`
>
> review-hardened runtime revision：`3b3fae1`

## 1. 结论

TinyInfer 已完成 Phase 1 的 Hugging Face reference runtime 本地候选：

- 固定 revision 加载 Qwen2.5-0.5B；
- TinyInfer 自己控制单请求 greedy prefill/decode 循环；
- 四类冻结 prompt 与 Hugging Face `model.generate()` 逐 token 对齐；
- 使用 observer 记录 Host/CUDA 分阶段时间和显存峰值；
- 在原生 WSL2/Linux worktree 中生成 20 条 schema-v1 canonical JSONL。

20 条正式记录的 `token_agreement` 全部为 `1.0`。每个 case 的五次正式输出 token tuple
完全一致。

这证明当前显式生成控制流是正确、确定且可测量的 reference baseline。它**不证明** TinyInfer
已经实现自己的 KV Cache、continuous batching、paged KV、API server 或 Triton 性能 kernel。

## 2. Revision 与证据链

报告使用两个 revision，避免“结果文件记录包含自身的提交 SHA”这一不可能的自引用：

- `tested revision` `b5225f8...`：包含 benchmark CLI 和全部可执行代码；真实 GPU benchmark
  在这个干净提交上运行。
- `canonical evidence commit` `ebd9a00`：只增加由上述代码生成的
  `docs/results/g1_reference.jsonl`，不修改已测试代码。

JSONL 中每条 `git_revision` 都严格等于完整的 tested revision：

```text
b5225f8e7c70d981c0c19015513594f28f163063
```

## 3. 环境身份

Benchmark 记录中的运行身份：

| 项目 | 实测值 |
|---|---|
| OS | Ubuntu 22.04 / WSL2 |
| Python | 3.11.15 |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| dtype | bfloat16 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.14.1 |
| Model | Qwen/Qwen2.5-0.5B |
| Model revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Attention backend | PyTorch SDPA |
| Warm-up | 每个 case 3 次，不写入 JSONL |
| Formal repetitions | 每个 case 5 次 |

Python、OS 和 GPU 基线也记录于 `docs/environment_report.json` 与 G0.5 报告。正式 JSONL
逐条保存模型 revision、Git revision、Torch/Transformers 版本、dtype 和 GPU 名称。

## 4. Workload 与正确性

| Case | 类别 | Prompt tokens | Output tokens | Stop reason | Agreement | 五次确定 |
|---|---|---:|---:|---|---:|---|
| `short_en` | 短英文 | 9 | 32 | `length` | 1.0 | 是 |
| `short_zh` | 短中文 | 13 | 32 | `length` | 1.0 | 是 |
| `structured` | 结构化指令 | 15 | 32 | `length` | 1.0 | 是 |
| `long_context` | 长上下文 | 612 | 26 | `eos` | 1.0 | 是 |

所有 case 的 `max_new_tokens` 都是 32。`long_context` 在第 26 个 output token 生成 EOS；
TinyInfer 和 Hugging Face 都在相同 token 停止，因此这是正确的 EOS 行为，不是长度或正确性失败。

长 prompt 使用固定 Qwen tokenizer 实测为 612 tokens，满足 512–768 token 验收区间。

## 5. 统计方法

以下数字全部由 committed JSONL 计算，没有从日志手工抄写：

- 每个 case 使用 5 条正式记录；
- median 使用标准中位数；
- p95 使用排序后位置 `(n - 1) × 0.95` 的线性插值；
- 显存从 bytes 转换为 MiB，除数为 `1024²`；
- Host 与 CUDA Device 指标始终分列，不混合来源。

样本量只有 5，因此 p95 主要用于保持报告口径和后续阶段一致，不应被解释为生产服务的
尾延迟结论。

## 6. TTFT、TPOT 与吞吐

表格单元格格式均为 `median / p95`。

| Case | Prompt tokens | Output tokens | Stop | TTFT host (ms) | TPOT host (ms) | TPOT device (ms) | Throughput (tok/s) |
|---|---:|---:|---|---:|---:|---:|---:|
| `short_en` | 9 | 32 | `length` | 16.322 / 16.699 | 14.746 / 15.063 | 14.722 / 15.076 | 67.594 / 68.088 |
| `short_zh` | 13 | 32 | `length` | 15.645 / 16.931 | 14.668 / 14.935 | 14.862 / 15.121 | 68.115 / 69.904 |
| `structured` | 15 | 32 | `length` | 16.307 / 20.381 | 15.322 / 15.847 | 15.381 / 15.974 | 65.076 / 66.896 |
| `long_context` | 612 | 26 | `eos` | 19.042 / 20.354 | 15.340 / 16.240 | 15.304 / 16.238 | 64.684 / 65.508 |

## 7. Prefill、Decode 与显存

表格单元格格式均为 `median / p95`。

| Case | Prefill host (ms) | Prefill device (ms) | Decode host (ms) | Decode device (ms) | Peak allocated (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| `short_en` | 15.916 / 16.209 | 15.767 / 16.206 | 457.130 / 466.965 | 456.390 / 467.360 | 961.801 / 961.801 | 1014.000 / 1014.000 |
| `short_zh` | 15.232 / 16.481 | 15.431 / 16.627 | 454.723 / 462.974 | 460.729 / 468.757 | 962.247 / 962.247 | 1014.000 / 1014.000 |
| `structured` | 15.897 / 19.916 | 16.012 / 20.078 | 474.989 / 491.258 | 476.810 / 495.207 | 962.854 / 962.854 | 1014.000 / 1014.000 |
| `long_context` | 17.656 / 18.751 | 17.581 / 18.719 | 383.504 / 405.994 | 382.597 / 405.956 | 1144.757 / 1144.757 | 1220.000 / 1220.000 |

长上下文的峰值显存高于三个短 prompt，符合更多 prompt KV 状态需要更多显存的预期。当前数据
只建立 reference baseline；没有对其他引擎、cache layout 或 kernel 做比较，因此不能据此声明
性能优势。

## 8. Prefill 与 Decode 的含义

Prefill 一次处理完整 prompt，并产生：

- 第一个 greedy output token；
- prompt 对应的 KV Cache。

Decode 在首 token 后每轮只输入一个 token，并复用已有 cache。TTFT 主要覆盖 tokenizer 和
prefill 到首 token 可读的路径；TPOT 只覆盖首 token 之后的 decode token。

当前 cache 是 Transformers 模型返回的公开 `Cache`，TinyInfer 只验证其长度并把它传回模型。
它不是 TinyInfer 自己分配和管理的 contiguous/paged KV Cache，也没有 block allocator 或
block table。Phase 2 才会单独设计 runtime-owned cache。

## 9. 复现命令

在原生 WSL2 worktree、模型已缓存的前提下：

```bash
uv sync --locked --extra dev --extra inference --extra kernels

python -m pytest -m "not gpu and not model and not slow"
python -m pytest tests/integration -m "gpu and model" -v

HF_HUB_OFFLINE=1 python benchmarks/bench_reference.py \
  --config configs/qwen2_5_0_5b.yaml \
  --prompts benchmarks/prompts.json \
  --output /tmp/tinyinfer-g1/reference.jsonl \
  --warmup-runs 3 \
  --repetitions 5 \
  --overwrite
```

成功时 CLI 最后一行必须是：

```text
G1 REFERENCE RUNTIME: PASS
```

## 10. Canonical artifacts

- 配置：`configs/qwen2_5_0_5b.yaml`
- 冻结 prompts：`benchmarks/prompts.json`
- Benchmark CLI：`benchmarks/bench_reference.py`
- Canonical JSONL：`docs/results/g1_reference.jsonl`
- 实施学习指南：`docs/PHASE1_LEARNING_GUIDE.md`

## 11. Gate 状态

本地已经取得以下新鲜证据：

- Full G0.5 verifier：191 passed；lock、environment、Ruff、Format、Mypy、CUDA matmul 和
  4 个 Triton vector-add sizes 全部 PASS；
- CPU suite：178 passed，GPU/model tests 在 CPU selector 下 deselected；
- WSL GPU/model integration：13 passed；
- Task 7 定向 suite：87 passed；
- canonical benchmark：`G1 REFERENCE RUNTIME: PASS`；
- JSONL 独立验证：20 records、4 cases × 5、agreement 1.0、每 case 单一 output tuple；
- 最终 candidate benchmark 重新运行 PASS，且四个 case 的 token tuples 与 canonical 完全一致；
- Ruff check、Ruff format、Mypy 和 `git diff --check`：通过。
- Feature CI：提交 `727faed` 的 GitHub Actions run `30369261505` 全部步骤成功；
- 评审后 canonical Gate：冻结 prompt 语义哈希、固定 config、3 warm-ups、5 repetitions、
  20 条有序 records 与 exact agreement 均由 CLI 强制执行；CPU suite 更新为 185 passed。
- 最终原生 WSL Gate（`3b3fae1`）：198 passed；Ruff、Mypy、CUDA matmul 与 4 个 Triton
  vector-add sizes 全部通过；
- 最终 GPU canonical 重跑：20 records、agreement 1.0、4/4 case 确定性，token tuples 与已提交
  canonical 完全一致；
- 最终 feature CI run `30370421007` 与 main CI run `30371391709` 均为 success；
- 独立代码复审未发现 Critical 或 Important 遗留问题，结论为可合并。

G1 已达到 `verified`。Annotated tag `v0.2.0-g1` 只允许指向包含本 closeout 报告的最终提交，
并在该提交的 main CI 成功后发布。

## 12. 限制与下一阶段边界

G1 不包含：

- runtime-owned contiguous KV Cache；
- continuous batching 或 scheduler；
- paged KV Cache；
- API Server；
- 自定义 Triton 性能 kernel；
- sampling、量化、多 GPU 或生产服务能力。

下一份独立设计文档现在可以进入 Phase 2，但仍必须遵守上述 Non-goals 与独立 Gate。
