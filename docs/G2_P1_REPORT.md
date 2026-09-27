# TinyInfer G2-P1 Batching-Throughput Report

> 报告日期：2026-09-27
>
> 状态：`verified locally` — 同一 revision 上 `verify_g2_p1.py` PASS、210 条记录全部通过正确性门禁、
> 该 revision 的 remote CI（run `36296736503`）success
>
> tested revision：`31e2953bc788f03eea60c69c49800ca6de1e99d7`
>
> evidence commit：`8fa4323`（只新增 `docs/results/g2_p1_raw.jsonl` 与 `docs/results/g2_p1_summary.md`）

## 1. 结论

这是 v0.3 路线中的 "before" 测量：Hugging Face 模型 + TinyInfer paged 适配层（dense 左填充 gather）。

- **Hugging Face B=1 参照约 64 tok/s**（TPOT 约 15.5 ms），与上下文长度无关，与 G1 一致。按每步读一遍
  约 0.99 GB 权重估算，只用到 RTX 4080 SUPER 标称带宽的约 9%：单请求 decode 受 CPU/kernel 启动开销
  限制，而非显存带宽。
- **paged 路径的单请求开销随上下文线性增长**：短 prompt 比参照慢 22%，612-token 长 prompt 慢 2.7 倍。
  估算每个 KV block 每步多约 0.64 ms，与"每 block 每步 48 次小切片拷贝"的启动开销量级吻合。
- **短 prompt 合批有效但扩展受限**：B=16 达到 275 tok/s（参照的 4.3 倍），但 TPOT 每多一行增加约
  2.4 ms，说明逐行 gather 而非 GPU 计算决定了步长。
- **长 prompt 合批输给参照串行**：`long_b8` 合批 35.7 tok/s，低于 Hugging Face 逐个串行的 59.1 tok/s。
  dense gather 把合批收益完全抵消。
- **合批显著降低排队带来的 TTFT**：`fixed_b16` 串行 TTFT 4194 ms，合批 140 ms。
- **正确性**：所有 B=1 路径与参照逐 token 一致；批处理路径仅在 `short_zh` 第 23 步出现 BF16 平局分歧
  （两个候选 logit 均为 21.000），符合批处理门禁规则。

这些结果**不证明** TinyInfer 已有性能优势，也不涉及 paged KV 的显存效率（属于 G2-P2）。

## 2. Revision 与证据链

- `tested revision` `31e2953…`：包含全部可执行代码；GPU 运行在此干净提交上完成。
- `evidence commit` `8fa4323`：只增加由上述代码生成的原始记录与汇总，不修改代码。
- 210 条记录的 `git_revision` 全部等于 tested revision，命令行一致：

```text
bench_phase2.py --config configs/qwen2_5_0_5b.yaml --workloads benchmarks/phase2_workloads.json
  --output /tmp/g2p1/raw.jsonl --kv-pool-tokens 8192 --warmup-runs 3 --repetitions 5 --overwrite
```

## 3. 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| OS | Ubuntu 22.04 / WSL2 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.14.1 |
| Model | Qwen/Qwen2.5-0.5B @ `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| dtype | bfloat16 |
| KV pool | 512 blocks × 16 tokens |
| Warm-up / repetitions | 3 / 5（每个 workload × mode） |
| Record schema | v2 |

## 4. Workload 与模式

| Workload | 请求 | 输出上限 |
|---|---|---:|
| `fixed_b1/2/4/8/16` | 4 条相近长度短 prompt 循环使用 | 32 |
| `long_b1/2/4/8` | G1 冻结的 `long_context`（612 prompt tokens，第 26 个 token 处 EOS） | 32 |
| `mixed_b4` | 短/中/长混合 | 8–64 |
| `dynamic_mixed_b4` | 同上，第 0/0/1/2 轮到达 | 8–64 |

- `hf_sequential`：G1 显式 greedy 循环 + Hugging Face 自带 cache，逐个请求，作为绝对参照。
- `sequential`：TinyInfer paged 路径，`max_batch=1`。
- `static_batch` / `continuous_batch`：TinyInfer paged 路径，一次 `[B, 1]` decode forward。

延迟从**提交**时刻计；吞吐按 makespan（首个提交到最后完成）计。

## 5. 正确性

- `hf_sequential` 与 `sequential`：全部 workload 与 Hugging Face B=1 `generate()` 逐 token 一致。
- 批处理模式：`fixed_b2/4/8/16` 中每个 `short_zh` 请求在第 23 步分歧，分歧数分别为 1/1/2/4，
  其余 workload 全部一致。分歧处参照 top-1（`101098`）与 TinyInfer 选择（`105653`）的 logit 均为
  21.000，差 0，容差 0.125（1 个 BF16 单位）。两次独立运行结果相同。
- 原因：不同 batch 形状使用不同 GEMM kernel，累加顺序不同，平局按不同方向打破。规则与诊断过程见
  `docs/G2_P1_RUNBOOK.md` 与 `docs/superpowers/specs/2026-09-04-g2-performance-design.md` 的修订节。
- 每次运行后共享 block 池全部回收。

## 6. 结果

以下数字均由 `docs/results/g2_p1_raw.jsonl` 计算，为 5 次重复的中位数。完整表见
`docs/results/g2_p1_summary.md`。

### 6.1 单请求开销（B=1）

| Workload | HF 参照 tok/s | Paged tok/s | HF TPOT ms | Paged TPOT ms | Paged TTFT ms | HF TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_b1` | 63.5 | 52.7 | 15.61 | 19.01 | 17.8 | 17.7 |
| `long_b1` | 63.0 | 22.9 | 15.64 | 42.60 | 71.4 | 20.2 |

### 6.2 短 prompt 的 batch 扫描（`static_batch`）

| B | Tok/s | 对 HF 参照 | 对 paged 串行 | TPOT ms | TTFT ms | 串行 TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 53.0 | 0.83× | 1.01× | 18.83 | 18.4 | 17.8 |
| 2 | 84.8 | 1.24× | 1.56× | 23.18 | 17.3 | 17.2 |
| 4 | 142.3 | 2.09× | 2.51× | 27.39 | 34.4 | 571.4 |
| 8 | 198.8 | 2.96× | 3.72× | 38.69 | 70.5 | 1843.2 |
| 16 | 274.6 | 4.31× | 5.13× | 54.67 | 139.5 | 4193.9 |

`continuous_batch` 在同时到达的 workload 上与 `static_batch` 相差在 6% 以内（`fixed_b8`：187.5 对 198.8）。

### 6.3 长 prompt 的 batch 扫描（`static_batch`）

| B | Tok/s | HF 串行 tok/s | TPOT ms |
|---:|---:|---:|---:|
| 1 | 22.9 | 63.0 | 42.47 |
| 2 | 28.8 | 63.4 | 66.63 |
| 4 | 32.0 | 63.3 | 121.13 |
| 8 | 35.7 | 59.1 | 219.43 |

### 6.4 混合与动态到达

| Workload | Mode | Tok/s | TTFT ms | TPOT ms |
|---|---|---:|---:|---:|
| `mixed_b4` | hf_sequential | 68.7 | 133.6 | 14.45 |
| `mixed_b4` | sequential | 55.6 | 155.0 | 17.48 |
| `mixed_b4` | static_batch | 95.1 | 35.6 | 24.65 |
| `mixed_b4` | continuous_batch | 92.7 | 34.7 | 25.12 |
| `dynamic_mixed_b4` | sequential | 55.4 | 149.5 | 17.30 |
| `dynamic_mixed_b4` | continuous_batch | 92.9 | 17.0 | 24.64 |

## 7. 分析

以下是基于记录的推算，不是独立测量。

- **带宽利用率**：Qwen2.5-0.5B 约 4.94 亿参数，BF16 约 0.99 GB。`fixed_b1/hf_sequential` 的 TPOT 为
  15.61 ms，折合约 63 GB/s，约为 4080 SUPER 标称 736 GB/s 的 8.6%。
- **每 block 开销**：`long_b1` 与 `fixed_b1` 的 paged TPOT 差 23.6 ms；两者 decode 期间平均 KV 长度约
  625 与 25 token，即约 39 与 2 个 block，折合每 block 每步约 0.64 ms。dense gather 对每个 block 每步做 24 层 ×
  K/V = 48 次切片拷贝，平均每次约 13 µs，与一次 Python 索引加一次 CUDA kernel 启动的量级相符。
- **batch 扩展**：短 prompt 下 TPOT 从 B=1 到 B=16 每行增加约 2.4 ms，与每行约 2 个 block 的 gather 加
  逐行写入的启动开销相符。若 decode 受带宽限制，TPOT 应接近常数。
- **长 prompt 倒挂**：`long_b8` 每步约 8 × 41 × 48 ≈ 1.6 万次小拷贝，TPOT 219 ms，吞吐低于参照串行。

**可检验的预测（阶段 C/D 验证，不作为本报告结论）**：用预计算 slot 索引把每层 gather 合并为一次
向量化操作后，每步启动次数将与 B 和上下文长度无关，长 prompt 的 TPOT 应接近参照，批处理吞吐的
上限将明显提高。

## 8. 限制

- **显存峰值可能重复计入 block 池**：paged 模式峰值比参照高 96–192 MiB，而池本身约 96 MiB。基准循环
  在旧运行对象仍存活时分配新池，峰值可能包含两个池。未在 GPU 上确认；本报告不据此讨论显存。
- 只有 host 侧计时，没有 CUDA event 分段计时。
- prefill 串行执行；长 prompt 的 paged TTFT（71 ms 对 20 ms）同样受逐 block 写入与读回影响。
- 每组 5 次重复，p95 仅作口径保持，不代表尾延迟结论。
- B=1 时 `static_batch`/`continuous_batch` 与 `sequential` 走相同代码路径，差异属于噪声。
- 平局判定使用无 cache 的完整重算 logits，与带 cache 的参照数值可能略有差别；本次分歧处两者一致。

## 9. 复现命令

在原生 WSL2 checkout（`31e2953`）、模型已缓存的前提下：

```bash
uv sync --locked --extra dev --extra inference
HF_HUB_OFFLINE=1 python scripts/verify_g2_p1.py
HF_HUB_OFFLINE=1 python benchmarks/bench_phase2.py \
  --config configs/qwen2_5_0_5b.yaml \
  --workloads benchmarks/phase2_workloads.json \
  --output /tmp/g2p1/raw.jsonl \
  --kv-pool-tokens 8192 --warmup-runs 3 --repetitions 5
python benchmarks/summarize_phase2.py --input /tmp/g2p1/raw.jsonl --output /tmp/g2p1/summary.md
```

分歧解释：`HF_HUB_OFFLINE=1 python benchmarks/diagnose_alignment.py`。

## 10. Gate 状态

```text
G2-BD: local candidate
G2-P1: verified locally
G2-P2: not started
full G2: not passed
block-native paged attention: not implemented
```
