# TinyInfer G0.5 环境基线收尾设计

> 日期：2026-07-22  
> 状态：待用户审阅  
> 实施分支：`agent/g0-5-closeout`  
> 基线提交：`48e0abba0eebba3df0d5d0fea7474bc13b9e8e48`

## 1. 背景

TinyInfer 当前已经完成 WSL2、Python 3.11、PyTorch CUDA、Transformers 和 Triton 的环境搭建，并在 RTX 4080 SUPER 上手工通过 CUDA BF16 矩阵乘及 Triton vector-add。当前记录存在以下不一致：

- `v0.1.0-phase1` 把环境门 G0.5 误称为完整 Phase 1；
- Phase 1 的 Reference Runtime、prefill/decode、token agreement 和 profiling 尚未实现；
- 阶段报告记录 Triton 3.7.0，实际环境为 3.7.1；
- 阶段报告记录 Mypy 检查 4 个源文件，当前实际为 5 个；
- 阶段报告声称 Triton smoke test 内置于 `scripts/check_environment.py`，实际环境检查器只检测包、CUDA、GPU 和 BF16 能力；
- 阶段报告头部的 Git revision 与包含报告的提交不一致；
- 报告列出的 GPU 安装脚本未被提交；
- WSL、Windows 和 GitHub 远端的提交状态尚未统一。

本设计将当前成果准确收敛为 **G0.5 Linux/GPU Environment Baseline**，不把它描述成完整 Phase 1。

## 2. 目标

1. 建立可重复执行、返回明确退出码的 CUDA/Triton GPU smoke test。
2. 让环境报告、阶段报告、workflow、实际依赖和 Git 历史互相一致。
3. 建立一个明确的 G0.5 验收入口和证据清单。
4. 把工作文档的编写规则固化到仓库，使其他 agent 能按文档执行并留下可审计证据。
5. 在完整验收后创建准确的 `v0.1.0-g0.5` 标签并发布到 GitHub。

## 3. 非目标

- 不实现 Request 或 GenerationResult 数据模型；
- 不下载或运行 Qwen2.5-0.5B；
- 不实现 prefill/decode generation loop；
- 不实现 KV Cache、scheduler、continuous batching 或 paged KV；
- 不编写 TinyInfer 性能优化 kernel；
- 不引入 API Server；
- 不对现有 Python 包进行无关重构。

这些内容属于 R4 之后的 Reference Runtime 或后续 Phase，必须使用独立分支、规格和验收门。

## 4. 里程碑与标签语义

当前里程碑名称固定为：

```text
G0.5 Linux/GPU Environment Baseline
```

最终标签固定为：

```text
v0.1.0-g0.5
```

本地未发布标签 `v0.1.0-phase1` 必须删除。远端当前没有该标签，因此不需要改写远端标签历史。

完整 Phase 1 仍以 G1 为退出门。只有 Reference Runtime、四类 prompt 的 100% token agreement 和 profiling 交付物全部通过后，才能创建 Phase 1 完成标签。

## 5. 实施边界与工作区策略

- 实施工作在 `agent/g0-5-closeout` 隔离分支进行；
- Windows 原始工作区中的未提交学习文档和旧脚本不得被覆盖或清理；
- WSL `main` 保持在 `48e0abb`，直到收尾分支验收完成；
- GPU 验收必须在 WSL2 Ubuntu-22.04 中运行；
- 正式命令以 Linux checkout 或 Linux 可执行环境为准；
- 分支通过所有 Gate 后才允许合并、打标签和推送；
- 同步 Windows checkout 前必须先处理其未提交文件，禁止使用 `reset --hard` 或覆盖式 checkout。

## 6. 组件设计

### 6.1 GPU smoke test

新增独立 Python 脚本 `scripts/check_gpu_smoke.py`，职责仅限真实 GPU correctness 验证：

1. 确认 PyTorch CUDA 可用；
2. 在 CUDA 上执行 BF16 `1024 × 1024` 矩阵乘；
3. 显式 `torch.cuda.synchronize()`；
4. 检查输出 shape、dtype、device 和有限值；
5. 编译并执行 Triton vector-add；
6. 覆盖长度 `1`、`257`、`4096`、`65537`；
7. 使用 mask 处理非整 block 尾部；
8. 使用 `torch.testing.assert_close` 与 `x + y` 比较；
9. 任何失败返回非零退出码；
10. 成功时输出结构稳定、可供人工和 agent 识别的 PASS 行。

环境检查器 `scripts/check_environment.py` 保持轻量和依赖容错。它负责描述能力，不伪装成 GPU correctness test。

### 6.2 CPU 单元测试

GPU kernel 不进入 CPU-only CI。为 smoke script 中可独立验证的纯 Python 逻辑编写 CPU 单元测试，例如：

- 固定 smoke size 集合；
- 结果摘要结构；
- 缺少 CUDA/Triton 时给出明确错误而不是静默通过。

真实 GPU correctness 由 G0.5 本机验收命令负责。CI 与本机 Gate 的职责必须在文档中明确区分。

### 6.3 G0.5 验收入口

新增 `scripts/verify_g0_5.py` 作为编排入口，按固定顺序执行：

- lock file 一致性检查；
- 环境 JSON 生成；
- pytest；
- Ruff check；
- Ruff format check；
- Mypy；
- GPU smoke test。

该入口不得自动提交、打标签或推送。验证与发布分开，以防测试脚本意外改变 Git 历史。

验证器需要：

- 打印每一步名称和命令；
- 保留子命令 stdout/stderr；
- 第一个失败步骤返回非零；
- 最终打印统一 G0.5 PASS/FAIL 摘要；
- 不下载模型；
- 不安装或升级依赖；
- 不写入任何 token、代理凭据或主机秘密。

### 6.4 环境快照与阶段报告

以下记录必须由最终环境重新生成或根据最终命令输出修正：

- `docs/environment_report.json`；
- `docs/environment_pip_list.txt`；
- 将 `docs/PHASE1_REPORT.md` 重命名为 `docs/G0_5_REPORT.md`；
- `docs/PHASE1_WORKFLOW.md` 中的 G0.5 状态和后续 R4 边界；
- README 中的当前状态与验收命令。

报告不把预计值写成实测值。为避免 Git 提交无法在自身内容中记录自身 SHA 的循环问题，报告使用两个明确概念：

- `tested_revision`：包含全部可执行代码、且实际运行完整 GPU/质量验收的提交；
- `release_revision`：只增加最终证据报告的提交，由 `v0.1.0-g0.5` 指向。

`release_revision` 相比 `tested_revision` 只能修改证据文档和快照，不得修改可执行代码。标签创建前在 `release_revision` 上再次运行 pytest、Ruff、format、Mypy、lock check 和文档检查；GPU smoke 的输出记录对应 `tested_revision`，并确认两个提交间 GPU 脚本内容未变化。

### 6.5 环境脚本治理

以下 Windows 未跟踪脚本均不纳入 G0.5 正式交付：

- `scripts/r2_linux_env.sh`；
- `scripts/r2_3_dev_deps.sh`；
- `scripts/r3_gpu_deps.sh`。

它们包含固定机器路径，其中 GPU 脚本还会再次执行 `uv add` 并修改依赖声明，不适合作为可重复验收入口。安装步骤保留为 workflow 中的显式命令；验证由 `scripts/verify_g0_5.py` 和 `scripts/check_gpu_smoke.py` 承担。Windows 原始未跟踪文件保持不变，等待后续由用户单独决定清理。报告中的交付物清单不得列出这三个脚本。

### 6.6 工作流编写标准

新增 `docs/WORKFLOW_AUTHORING_STANDARD.md` 与 `docs/templates/WORKFLOW_TEMPLATE.md`。

所有以后交给其他 agent 执行的 workflow 必须包含：

1. **Objective**：这份工作最终改变什么；
2. **Non-goals**：明确禁止顺手加入的内容；
3. **Preconditions**：开始前必须成立的环境、分支、依赖和权限；
4. **Baseline Evidence**：开始前运行哪些命令并保存什么结果；
5. **Artifact Manifest**：创建、修改、生成、禁止提交的文件；
6. **Detailed Workflow**：严格编号的执行顺序；
7. **Per-task Inputs/Actions/Outputs**：每个 Task 的输入、操作和输出；
8. **Acceptance Criteria**：每个 Task 的可观察通过条件；
9. **Gate / Exit Criteria**：允许进入下一阶段的证据；
10. **Definition of Done**：代码、测试、文档、Git 和交付物的完整完成定义；
11. **Verification Commands**：可复制命令与预期退出码/关键输出；
12. **Stop Conditions**：遇到哪些情况必须停止而不是猜测；
13. **Failure Handling**：失败证据、清理和恢复方式；
14. **Git/Worktree Rules**：基线、分支、提交边界、合并和标签规则；
15. **Agent Handoff Contract**：agent 开始前要读什么，结束时必须报告什么；
16. **Status Vocabulary**：`not started`、`in progress`、`blocked`、`verified` 的精确定义。

禁止仅用勾选框或“已完成”作为验收证据。每一个完成声明必须对应命令、退出码、测试数量、产物路径或可检查的 Git revision。

## 7. 数据与控制流

```text
pyproject.toml + uv.lock
          |
          v
  WSL project environment
          |
          +--> lightweight environment inspection
          |        -> environment_report.json
          |
          +--> CUDA/Triton smoke correctness
          |        -> explicit PASS/FAIL evidence
          |
          +--> pytest / Ruff / Mypy / lock check
                   -> G0.5 verification summary
                            |
                            v
              corrected report + final commit
                            |
                            v
                     v0.1.0-g0.5
```

环境报告与 smoke test 是两个独立证据源：前者说明能力可见，后者说明真实计算正确。

## 8. 错误处理

- CUDA 不可用：立即失败并打印 `torch.cuda.is_available()` 状态；
- BF16 不支持：立即失败，不偷偷降级 FP16/FP32；
- Triton 导入或编译失败：保留异常和非零退出码；
- vector-add 不一致：报告失败 size，不减少测试范围；
- lock 不一致：停止验收，不自动重写 lock；
- lint/type/test 失败：停止发布，不通过修改报告掩盖；
- Windows/WSL 分支出现未知提交：停止同步并重新比较 graph；
- GitHub 推送失败：保留本地提交和标签，不强推；
- 任意敏感信息出现在快照：删除快照中的敏感字段并重新生成，禁止提交。

## 9. Acceptance Criteria

### AC-1：里程碑语义准确

- 仓库中不再把 G0.5 描述为完整 Phase 1；
- `v0.1.0-phase1` 不存在于最终本地和远端标签；
- G1 清单保持未完成状态；
- 下一步明确为 R4 Reference Runtime 数据模型。

### AC-2：GPU smoke test 可重复

- 在 WSL2 RTX 4080 SUPER 上连续运行两次均成功；
- CUDA BF16 `1024 × 1024` matmul 通过；
- Triton vector-add 的四个固定长度全部通过；
- 第二次运行能够在已有 Triton cache 情况下通过；
- 任意 assertion 或依赖错误导致非零退出码。

### AC-3：工程质量通过

- `uv lock --check` 返回 0；
- pytest 全部通过，报告准确测试数量；
- `ruff check .` 返回 0；
- `ruff format --check .` 返回 0；
- Mypy 返回 0，报告准确源文件数量；
- CPU-only CI 不依赖 GPU。

### AC-4：报告与环境一致

- Triton、PyTorch、Transformers、Python 等版本来自最终环境；
- 环境 JSON 不含用户名、token、代理凭据或秘密；
- 阶段报告的命令、结果、文件清单与仓库实际一致；
- 报告清楚区分 capability inspection 与 correctness smoke test；
- 报告引用可解析的最终 Git revision。

### AC-5：工作流可以安全交接

- 编写标准和模板均存在并由 README 或 docs 索引链接；
- 模板包含本设计第 6.6 节全部 16 项；
- 示例 Gate 使用明确命令、退出码和产物，不只使用勾选框；
- agent handoff 要求报告修改文件、验证命令、结果、未决问题和 Git 状态。

## 10. G0.5 Gate / Exit Criteria

只有以下证据同时存在，G0.5 才能标记 `verified`：

1. 最终提交上的完整 G0.5 验证命令退出码为 0；
2. 两次真实 GPU smoke test 均通过；
3. 环境 JSON 和 pip list 已由该环境生成；
4. 所有 Acceptance Criteria 已逐条复核；
5. 分支工作树干净；
6. 最终提交已推送到 GitHub；
7. annotated tag `v0.1.0-g0.5` 指向受验收提交并已推送；
8. 远端 `main` 与预期最终提交一致；
9. Windows checkout 的未提交文件清单已记录，并明确标记为“暂缓同步”；G0.5 发布不覆盖或删除这些文件；
10. G1/R4 未被误标为完成。

## 11. Definition of Done

G0.5 的“完成”定义为：

- 功能：GPU smoke 和统一验收入口按设计工作；
- 正确性：CUDA/Triton 真实计算通过固定 workload；
- 工程质量：测试、lint、format、typing、lock 检查全部通过；
- 文档：报告、workflow、README、工作流标准和模板一致；
- 安全：快照不含秘密，Git 同步不覆盖未提交工作；
- 可复现：命令、依赖版本、硬件和 Git revision 均已记录；
- 版本控制：改动经独立提交审查并合入 main；
- 发布：`v0.1.0-g0.5` 指向最终验收提交并存在于远端；
- 边界：R4/G1 仍保持未完成并使用后续独立计划。

未满足任意一项时，只能报告 `in progress` 或明确的 `blocked`，不能报告“Phase 1 完成”。

## 12. 验证策略

实施采用 TDD：

1. 先为验证编排与错误传播编写失败测试；
2. 确认测试因缺少行为而失败；
3. 实现最小代码使其通过；
4. 运行单元测试和全量 CPU 质量检查；
5. 在 WSL GPU 上运行真实 smoke；
6. 故意使用不满足条件的执行路径验证失败退出码；
7. 在最终提交上重新执行完整 Gate。

文档改动使用结构检查、链接检查、`git diff --check` 和逐条 Acceptance Criteria 复核。文档中的动态数字必须来自最终命令输出。

## 13. 提交与发布设计

建议提交边界：

1. `docs: define G0.5 closeout and workflow authoring standard`
2. `test: specify G0.5 verification behavior`
3. `feat: add reproducible GPU smoke and G0.5 verifier`
4. `docs: record verified G0.5 environment baseline`

发布顺序：

```text
verify branch
  -> merge/fast-forward main
  -> verify main commit again
  -> delete local v0.1.0-phase1
  -> create annotated v0.1.0-g0.5
  -> push main
  -> push v0.1.0-g0.5
  -> verify remote refs
```

不使用强推。若远端在实施期间出现新提交，停止发布并重新执行合并与验收。

## 14. Agent Handoff Contract

执行本设计的 agent 开始前必须：

- 阅读本设计、项目计划和 Phase 1 workflow；
- 报告当前分支、HEAD、origin/main、worktree 和 dirty files；
- 运行基线测试并记录结果；
- 确认没有把 G0.5 当成完整 G1。

每个 Task 完成时必须报告：

- 修改或新增的文件；
- 实际运行的验证命令；
- 退出码、通过数量和关键输出；
- 与 Acceptance Criteria 的对应关系；
- 仍未完成或被阻塞的事项；
- 当前 Git status 和 commit SHA。

Agent 不得根据旧报告、截图或另一 agent 的口头结论直接勾选 Gate。完成声明必须来自当前提交上的新鲜验证证据。
