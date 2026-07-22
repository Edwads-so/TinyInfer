# TinyInfer G0.5 阶段报告

> 生成日期：2026-07-22
> candidate_revision：4d189af（审计失败的旧候选）
> tested_revision：05b9160dd1337ddf54727c66c3223e1b0cd0293c
> release_revision：尚未产生
> 状态：in progress — 原生 WSL 本地 Gate 已通过，网络与发布 Gate 尚未完成

---

## 1. 阶段概述

G0.5 目标：在 WSL2 Ubuntu-22.04 中建立可复现的 Linux、Python、CUDA 和 Triton 环境基线，
通过独立 GPU smoke test 验证真实计算正确性。

**G0.5 是环境基线门，不是完整 Phase 1。完整 Phase 1 以 G1 为退出门。**

发布 Gate 完成后的下一步是 R4 Reference Runtime 数据模型开发。

在 G0.5 Gate 通过前，上述“下一步”保持锁定，不允许开始 R4。

### 2026-07-22 审计结果

候选代码已存在于 Windows 隔离 worktree，但尚未在 WSL 原生 checkout 上运行正式 GPU smoke
和完整 verifier。重新执行 CPU 质量门得到：

| 检查 | 新鲜结果 |
|---|---|
| pytest | 23 passed, 8 failed |
| Ruff check | 7 errors |
| Ruff format | 4 files would be reformatted |
| Mypy | 7 errors |
| Native WSL GPU smoke | not run |
| Full G0.5 verifier | not run |

因此旧候选中的 PASS 与“全部通过”声明无效。该问题已由后续修复提交纠正。

### 2026-07-22 修复后本地验收

修复后的提交 `05b9160` 已由 Git bundle 转入以下原生 Linux worktree：

```text
/home/administrator/projects/TinyInfer/.worktrees/g0-5-closeout
```

使用 `uv sync --offline --locked --extra dev --extra inference --extra kernels` 建立全新 `.venv`，
离线安装 64 个受控包。随后连续两次执行 GPU smoke，并执行一次完整 verifier，三次命令退出码均为 0。

### Git Revision 说明

本文档使用两个 revision 概念解决"报告无法在自身提交中引用自身 SHA"的循环问题：

- `tested_revision`：包含全部可执行代码、且实际运行了完整 GPU/质量验收的提交。
- `release_revision`：只增加最终证据报告和快照的提交，由 `v0.1.0-g0.5` 标签指向。
  `release_revision` 相比 `tested_revision` **只修改**证据文档和快照，不得修改可执行代码。

---

## 2. 环境冻结版本

以下版本来自 WSL2 最终环境实测输出，不是预期值或旧快照。

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04.5 LTS (WSL2) |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| NVIDIA Driver | 591.86 |
| GPU | NVIDIA GeForce RTX 4080 SUPER (16 GB) |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| CUDA Build | 13.0 |
| Triton | 3.7.1 |
| Transformers | 5.14.1 |
| Safetensors | 0.8.0 |
| Flash-Attn | 未安装（可选，SDPA fallback） |
| pytest | 8.4.2 |
| ruff | 0.15.22 |
| mypy | 1.20.2 |

---

## 3. 验证结果

### 3.1 Capability Inspection（能力检查）

由 `scripts/check_environment.py` 提供。该脚本仅检查包安装、CUDA 可见性、GPU 名称和 BF16
支持状态，不执行任何 GPU 计算。

| 检查项 | 结果 |
|--------|------|
| torch 安装状态 | ✅ installed |
| transformers 安装状态 | ✅ installed |
| safetensors 安装状态 | ✅ installed |
| triton 安装状态 | ✅ installed |
| CUDA 可用 | ✅ yes |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| BF16 支持 | ✅ yes |

### 3.2 GPU Correctness Smoke Test

由独立脚本 `scripts/check_gpu_smoke.py` 提供。该脚本在真实 GPU 上执行计算并验证结果，
与能力检查职责完全分离。

| 测试 | 结果 |
|------|------|
| CUDA BF16 matmul (1024×1024) | ✅ PASS × 2 standalone runs + verifier run |
| Triton vector-add (1) | ✅ PASS × 3 |
| Triton vector-add (257) | ✅ PASS × 3 |
| Triton vector-add (4096) | ✅ PASS × 3 |
| Triton vector-add (65537) | ✅ PASS × 3 |

### 3.3 工程质量检查

| 检查 | 结果 |
|------|------|
| Unit tests | ✅ 35 passed |
| Ruff check | ✅ All checks passed |
| Ruff format | ✅ 14 files already formatted |
| Mypy type check | ✅ No issues in 9 source files |

---

## 4. 验收清单 (G0.5)

- [x] WSL2 Ubuntu-22.04 状态、kernel 和 GPU 信息已记录
- [x] Linux checkout 位于 `/home/administrator/projects/TinyInfer`
- [x] Python 3.11 项目环境已在全新原生 WSL worktree 中从 lock file 离线重建
- [x] PyTorch CUDA、BF16、Transformers、Safetensors 导入通过
- [x] 独立 GPU smoke test（`scripts/check_gpu_smoke.py`）连续两次通过并返回零退出码
- [x] G0.5 验收入口（`scripts/verify_g0_5.py`）编排全部质量检查并返回零退出码
- [x] 环境报告不包含秘密信息
- [x] 35 项单元测试、Ruff、14 文件 format check 和 9 源文件 mypy 通过
- [x] 工作流编写标准与模板已固化
- [ ] R1 网络和私有仓库访问稳定
- [ ] release revision、远端 main 与 `v0.1.0-g0.5` 标签已发布

**本地计算与工程 Gate 已通过；G0.5 发布状态仍为 in progress。**

---

## 5. 交付物

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 流程文档 | `docs/PHASE1_WORKFLOW.md` | G0.5 状态 + R4-R7 规划 |
| 环境 lock file | `uv.lock` | 可复现的依赖版本集 |
| 环境 JSON 报告 | `docs/environment_report.json` | 机器可读取的能力检查结果 |
| pip list 快照 | `docs/environment_pip_list.txt` | 完整包列表 |
| G0.5 阶段报告 | `docs/G0_5_REPORT.md` | 本文件 |
| 环境诊断脚本 | `scripts/check_environment.py` | 轻量级能力检查 |
| GPU smoke test | `scripts/check_gpu_smoke.py` | 独立 GPU correctness 验证 |
| G0.5 验收入口 | `scripts/verify_g0_5.py` | 编排全部质量门 |
| 工作流标准 | `docs/WORKFLOW_AUTHORING_STANDARD.md` | agent 工作流编写规范 |
| 工作流模板 | `docs/templates/WORKFLOW_TEMPLATE.md` | 可复用的工作流起点 |
| G0.5 设计规格 | `docs/superpowers/specs/2026-07-22-g0-5-closeout-design.md` | 本交付的设计文档 |

以下文件存在但因包含固定机器路径或会修改依赖声明，不纳入 G0.5 正式交付物：

- `scripts/r2_linux_env.sh`（未跟踪，暂缓处理）
- `scripts/r2_3_dev_deps.sh`（未跟踪，暂缓处理）
- `scripts/r3_gpu_deps.sh`（未跟踪，暂缓处理）

---

## 6. 下一步

进入 R4：Reference Runtime 数据模型开发。

具体任务：
- R4: Request 与 GenerationResult 数据模型
- R5: 显式 Prefill/Decode Loop
- R6: Hugging Face 正确性对齐
- R7: Profiling 与 JSONL 结果
- G1: Runtime 验收
