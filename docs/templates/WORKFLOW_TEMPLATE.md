# [WORKFLOW TITLE]

> 版本：v1.0
> 制定日期：YYYY-MM-DD
> 适用范围：[描述适用阶段或子系统]
> 状态：not started

## 1. Objective

[一句话说明该工作流完成后仓库或环境将处于什么状态。]

## 2. Non-goals

- [不实现 X。]
- [不修改 Y。]
- [不引入 Z。]

## 3. Preconditions

- **操作系统**：[e.g., WSL2 Ubuntu-22.04]
- **Python**：[e.g., 3.11]
- **系统工具**：[e.g., git, uv, curl]
- **网络**：[e.g., GitHub, PyPI, Hugging Face 可访问]
- **路径**：[e.g., /home/user/projects/TinyInfer]
- **Git**：[e.g., main 与 origin/main 一致, worktree 干净]

## 4. Baseline Evidence

```bash
git status --short --branch
git log -3 --oneline --decorate
git rev-parse HEAD
git rev-parse origin/main
python -m pytest  # 基线测试
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

保存基线输出到 `[path/to/baseline.log]`。

## 5. Artifact Manifest

| 路径 | 类型 | 变动 | 说明 |
|------|------|------|------|
| `[path]` | code/test/docs/snapshot/config | 新增/修改/重命名/删除 | [说明] |

**禁止提交**：[列出禁止加入版本控制的文件或目录]

## 6. Detailed Workflow

1. **[Task 1 名称]** — [一句话描述]
2. **[Task 2 名称]** — [一句话描述]
   ...

## 7. Tasks

### Task 1: [名称]

#### Inputs

- [前置文件和条件]

#### Actions

1. [具体步骤]

#### Outputs

- [可检查产物]

#### Acceptance Criteria

- [ ] [可测量条件]

### Task 2: [名称]

#### Inputs

- [前置文件和条件]

#### Actions

1. [具体步骤]

#### Outputs

- [可检查产物]

#### Acceptance Criteria

- [ ] [可测量条件]

## 8. Gate / Exit Criteria

允许进入下一阶段的证据：

1. 命令 `[command]` 退出码为 0。
2. 文件 `[path]` 存在且内容满足 `[条件]`。
3. Git worktree 干净（`git status --porcelain` 为空）。
4. [其他条件]

## 9. Definition of Done

- **功能**：[描述]
- **正确性**：[描述]
- **工程质量**：pytest / ruff check / ruff format / mypy 全部通过。
- **文档**：[相关文档已更新且一致]
- **安全**：[无敏感信息]
- **可复现**：[命令、版本、硬件已记录]
- **版本控制**：[改动已提交至独立分支]
- **发布**：[标签指向正确提交（如适用）]
- **边界**：[后续阶段未被误标为完成]

## 10. Verification Commands

```bash
# 在工作目录 [path] 中执行
[command]  # 预期退出码: 0 | 关键输出: [regex or text]
```

## 11. Stop Conditions

出现以下任一情况时立即停止：

- [条件 1]
- [条件 2]

## 12. Failure Handling

1. 收集失败日志：`[命令] > failure.log 2>&1`。
2. 如需回滚：`git checkout -- [文件]`。
3. 如需恢复 stash：`git stash pop`。
4. [可重试 / 需人工介入的判断标准]

## 13. Git / Worktree Rules

- **分支**：[branch name]
- **Worktree**：[path 或 "直接 checkout"]
- **提交格式**：[e.g., `type: description`]
- **合并策略**：[fast-forward / merge commit / squash]
- **标签**：[命名规则和类型]
- **强推**：禁止。

## 14. Agent Handoff Contract

### 开始前报告

- 当前分支和 HEAD SHA。
- origin/main SHA。
- worktree 路径和 dirty files。
- 确认未将本门误标为后续阶段。

### 结束时报告

- 修改或新增的每个文件（含路径和变动类型）。
- 实际运行的每条验证命令及其退出码和关键输出。
- 与每个 Acceptance Criteria 的对应关系。
- 仍未完成或被阻塞的事项。
- 当前 Git status 和最终 commit SHA。

## 15. Status Vocabulary

| 状态 | 定义 |
|------|------|
| `not started` | 前置条件尚未确认，或没有任何可观察的进展。 |
| `in progress` | 至少一个 Task 的 Actions 已开始执行但 AC 未全部满足。 |
| `blocked` | 遇到 Stop Condition 或外部依赖不满足，无法继续。 |
| `verified` | 所有 AC 逐条通过，所有 Gate 命令在当前提交上通过。 |