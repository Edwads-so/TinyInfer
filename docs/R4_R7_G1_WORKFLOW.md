# TinyInfer R4–R7 + G1 Reference Runtime 工作流

> 版本：v1.0
> 制定日期：2026-07-23
> 适用范围：Qwen2.5-0.5B 单请求 Hugging Face Reference Runtime
> 状态：not started
> 设计规格：`docs/superpowers/specs/2026-07-23-hf-reference-runtime-design.md`
> 逐步实施计划：`docs/superpowers/plans/2026-07-23-hf-reference-runtime.md`

## 1. Objective

在原生 WSL2 worktree 中实现固定 Qwen2.5-0.5B revision 的单请求 greedy reference runtime，
显式区分 prefill 与逐 token decode，在四类固定 prompt 上与 Hugging Face 达到 100% token
agreement，并生成含 Host/CUDA 时间、吞吐量和显存的 20 条规范 JSONL 记录。

## 2. Non-goals

本工作流禁止实现或顺手加入：

- 自定义 contiguous KV cache。
- Request 调度状态机、continuous batching 或 iteration-level scheduler。
- Paged KV cache、block pool、block table 或 Qwen attention adapter。
- API Server、HTTP/SSE、并发服务或鉴权。
- 自定义 Triton kernel 或 attention 优化。
- Sampling、beam search、量化、多 GPU 或 CPU offload。
- 与 vLLM/TensorRT-LLM 的性能优劣结论。

Agent 发现上述需求时必须记录为后续事项，不能扩大当前分支范围。

## 3. Preconditions

开始 Task R4.1 前必须同时满足：

- **操作系统**：WSL2 Ubuntu-22.04，Linux checkout 位于
  `/home/administrator/projects/TinyInfer`。
- **Python**：3.11.15。
- **GPU**：NVIDIA GeForce RTX 4080 SUPER，CUDA 可用，BF16 支持。
- **依赖**：PyTorch 2.13.0+cu130、Transformers 5.14.1、Safetensors 0.8.0。
- **模型**：`Qwen/Qwen2.5-0.5B` revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987` 可下载或已缓存。
- **工具**：`git`、`uv`、`curl`、GitHub CLI 已安装。
- **网络**：GitHub、PyPI、Hugging Face 可访问；若使用代理，只在 shell 环境配置，禁止写入仓库。
- **磁盘**：Linux home 所在文件系统至少有 8 GiB 可用空间。
- **Git**：`main == origin/main`，worktree 干净，`v0.1.0-g0.5` 是当前 HEAD 的祖先。
- **隔离**：执行分支为 `agent/hf-reference-runtime`，worktree 为
  `/home/administrator/projects/TinyInfer/.worktrees/hf-reference-runtime`。
- **技能**：执行 agent 必须使用 `superpowers:subagent-driven-development` 或
  `superpowers:executing-plans`；代码任务必须使用 TDD。

任一前置条件不成立时状态为 `blocked`，不得开始写 runtime。

## 4. Baseline Evidence

在 Linux 主 checkout 中执行：

```bash
cd /home/administrator/projects/TinyInfer
git fetch origin main --tags
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor v0.1.0-g0.5 HEAD
git worktree list
df -h /home/administrator/projects
```

预期：

- `git status` 不显示修改或未跟踪文件。
- HEAD 与 `origin/main` 输出相同 40 字符 SHA。
- `merge-base --is-ancestor` 退出码为 0。
- 可用磁盘空间不少于 8 GiB。

建立隔离 worktree 后执行：

```bash
cd /home/administrator/projects/TinyInfer/.worktrees/hf-reference-runtime
uv sync --locked --extra dev --extra inference --extra kernels
python scripts/verify_g0_5.py
```

预期退出码均为 0，最终关键行为 `G0.5 VERIFICATION: PASS`。基线失败必须先修复或报告，禁止把
已存在失败带入 R4。

## 5. Artifact Manifest

| 路径 | 类型 | 变动 | 说明 |
|---|---|---|---|
| `tinyinfer/runtime/__init__.py` | code | 新增 | Runtime 公开导出 |
| `tinyinfer/runtime/request.py` | code | 新增 | `GenerationRequest` |
| `tinyinfer/runtime/result.py` | code | 新增 | `StopReason`、`GenerationResult` |
| `tinyinfer/runtime/hf_loader.py` | code | 新增 | 固定 revision 模型加载 |
| `tinyinfer/runtime/generation.py` | code | 新增 | Prefill、decode、greedy loop |
| `tinyinfer/profiling/__init__.py` | code | 新增 | Profiling 公开导出 |
| `tinyinfer/profiling/timer.py` | code | 新增 | Host/CUDA timer 与显存采样 |
| `tinyinfer/profiling/record.py` | code | 新增 | Schema v1 与 agreement |
| `benchmarks/__init__.py` | code | 新增 | Benchmark package marker |
| `benchmarks/prompts.json` | data | 新增 | 四类固定 workload |
| `benchmarks/bench_reference.py` | code | 新增 | G1 CLI 与原子 JSONL 输出 |
| `docs/results/g1_reference.jsonl` | evidence | 新增 | 20 条规范实测记录 |
| `tests/unit/test_request.py` | test | 新增 | Request 约束 |
| `tests/unit/test_result.py` | test | 新增 | Result 约束 |
| `tests/unit/test_hf_loader.py` | test | 新增 | Loader optional-dependency 边界 |
| `tests/unit/test_generation.py` | test | 新增 | Fake model/cache 控制流 |
| `tests/unit/test_prompts.py` | test | 新增 | 固定 workload schema |
| `tests/unit/test_timer.py` | test | 新增 | 计时公式与边界 |
| `tests/unit/test_record.py` | test | 新增 | Agreement/schema/JSONL |
| `tests/unit/test_bench_reference.py` | test | 新增 | CLI orchestration 与原子写入 |
| `tests/integration/test_hf_loader.py` | test | 新增 | GPU/model loader smoke |
| `tests/integration/test_hf_reference_runtime.py` | test | 新增 | 真模型生成与对齐 |
| `pyproject.toml` | config | 修改 | pytest markers 与 Mypy 边界 |
| `README.md` | docs | 修改 | G1 使用命令和状态 |
| `docs/PHASE1_WORKFLOW.md` | docs | 修改 | G1 最终状态与证据链接 |
| `docs/G1_REFERENCE_RUNTIME_REPORT.md` | docs | 新增 | G1 阶段报告 |

**禁止提交：** `.venv/`、`models/`、Hugging Face cache、`*.safetensors`、`*.bin`、token、代理
配置、`benchmarks/results/`、`/tmp/tinyinfer-g1-evidence/`、Python cache 和大型原始日志。

## 6. Detailed Workflow

严格按以下顺序执行，不允许并行修改共享接口：

1. **W0：隔离与基线** — 创建 Linux worktree，重建锁定环境，运行 G0.5 Gate。
2. **R4.1：Request/Result contracts** — 建立不可变数据模型与 CPU tests。
3. **R4.2：HF loader** — 固定 model revision、dtype、device、EOS/PAD 行为。
4. **R5：Explicit generation** — 实现 prefill、单步 decode、greedy loop 与 Cache invariants。
5. **R6：Correctness alignment** — 固定 prompts，与 `model.generate()` 做逐 token 对齐。
6. **R7.1：Timing/record schema** — 实现 Host/CUDA 计时与 schema v1。
7. **R7.2：Benchmark CLI** — warm-up、5 次重复、原子写入 20 条 JSONL。
8. **G1：Fresh acceptance** — 全套测试、真模型 benchmark、报告与发布检查。

每个任务必须遵循：写失败测试 → 确认预期失败 → 最小实现 → 定向测试通过 → 全套 CPU Gate →
小提交。GPU integration tests 只能补充 CPU TDD，不能替代它。

## 7. Tasks：Inputs / Actions / Outputs

### W0：隔离与基线

#### Inputs

- 第 3 节全部 Preconditions。
- `main` 与 `origin/main` 一致且干净。

#### Actions

1. 在主 checkout 执行：

   ```bash
   git worktree add .worktrees/hf-reference-runtime -b agent/hf-reference-runtime
   ```

2. 进入新 worktree，执行第 4 节环境与 G0.5 验证命令。
3. 报告分支、HEAD、origin/main、worktree 路径、dirty files 和测试结果。

#### Outputs

- 独立 Linux worktree 和分支。
- 锁定 `.venv`，仅存在于 worktree，不提交。
- 新鲜 G0.5 PASS 输出。

### R4.1：Request 与 Result contracts

#### Inputs

- W0 通过。
- 设计规格第 6 节。

#### Actions

1. 按实施计划 Task 1 先写 request/result 失败测试。
2. 实现 `GenerationRequest`、`StopReason` 和 `GenerationResult`。
3. 只导出规格定义的公开类型。
4. 运行定向测试与全部 CPU Gate。
5. 提交 `feat: add generation request and result contracts`。

#### Outputs

- 三个不可变公开 contract。
- 非空、token ID、max token 上下界测试。

### R4.2：固定 Hugging Face loader

#### Inputs

- R4.1 提交已存在。
- 模型 revision 与依赖版本和设计规格一致。

#### Actions

1. 使用 monkeypatch 写 loader 调用参数与错误包装的失败测试。
2. 实现 `LoadedModel`、`ModelLoadError` 和 `load_reference_model()`。
3. 禁止 `device_map="auto"`；显式 `.to(config.device)` 与 `.eval()`。
4. 添加 `gpu`、`model` marker 的 loader integration test。
5. CPU tests 通过后，在 WSL GPU 上执行 loader integration test。
6. 提交 `feat: add fixed-revision Hugging Face loader`。

#### Outputs

- 固定 SHA 的 tokenizer/model loader。
- 可审计的 dtype、device、eval、use_cache、EOS/PAD 状态。

### R5：显式 Prefill/Decode Runtime

#### Inputs

- R4.2 公开 `LoadedModel`。
- Transformers 5.14.1 公开 `Cache.get_seq_length()`。

#### Actions

1. 以 fake model/tokenizer/cache 写 prefill、decode、EOS、length 失败测试。
2. 实现 `RuntimeInvariantError`、`PrefillOutput`、`prefill()`、`decode_one()` 和
   `generate_greedy()`。
3. 每轮 decode 断言输入 shape 为 `[1, 1]`，Cache 长度恰好增加 1。
4. 添加真模型 1/2/8/32 token integration tests。
5. 运行 CPU Gate 和定向 GPU suite。
6. 提交 `feat: add explicit greedy prefill and decode loop`。

#### Outputs

- 不调用 `model.generate()` 的 TinyInfer greedy runtime。
- 可观察的 Prefill/Decode 与 Cache 长度证据。

### R6：HF Token Alignment

#### Inputs

- R5 真模型 32-token decode 通过。
- 四类 prompt schema 已由实施计划固定。

#### Actions

1. 写 prompt loader 与 agreement 失败测试。
2. 新增 `benchmarks/prompts.json`，只包含 `short_en`、`short_zh`、`structured`、
   `long_context`。
3. 实现 `reference_generate()`、`compare_token_ids()` 与 `AlignmentError`。
4. 运行四类 prompt 对齐；出现分歧按设计规格第 9 节顺序定位，禁止改文本后处理。
5. 验证 long prompt 实际 token 数为 512–768。
6. 提交 `test: add Hugging Face token alignment gate`。

#### Outputs

- 四类稳定 workload。
- 每类 TinyInfer 与 HF output token tuple 相同的 integration evidence。

### R7.1：Host/CUDA Timer 与 Record Schema

#### Inputs

- R6 correctness Gate 通过。
- 设计规格第 11 节字段定义。

#### Actions

1. 用 fake clock/event 写计时公式失败测试。
2. 实现 `PhaseTimings`、`GenerationObserver`、`InferenceTimer` 与集中 synchronize。
3. 写 agreement、mismatch index、finite values、schema keys 失败测试。
4. 实现 `BenchmarkRecord`、`compare_token_ids()` 与 JSON dict 转换。
5. 运行全部 CPU Gate。
6. 提交 `feat: add reference runtime profiling records`。

#### Outputs

- 分离的 Host/CUDA 计时结果。
- 完整 schema v1 数据模型。

### R7.2：Benchmark CLI 与 JSONL

#### Inputs

- R7.1 类型和字段冻结。
- 模型已缓存或网络稳定。

#### Actions

1. 写 CLI 参数、拒绝覆盖、原子写入和失败退出码测试。
2. 实现 `benchmarks/bench_reference.py`。
3. 对每个 case warm-up 3 次，正式运行 5 次。
4. 每次正式运行比较 HF token，采集显存并构造 record。
5. 写入 `docs/results/g1_reference.jsonl`，成功时恰有 20 条。
6. 提交 `feat: add reproducible reference benchmark`。

#### Outputs

- 可复制 CLI。
- 20 条逐行有效、可提交的实测 JSONL。

### G1：验收、报告与发布候选

#### Inputs

- W0–R7.2 全部提交存在。
- worktree 只允许报告文件有预期修改。

#### Actions

1. 在当前 HEAD 上重新运行第 10 节所有验证命令。
2. 从实测 JSONL 生成摘要，不手填性能数字。
3. 新增 `docs/G1_REFERENCE_RUNTIME_REPORT.md`，更新 README 与 Phase 1 状态。
4. 扫描 secrets、绝对路径、旧状态和禁止范围文件。
5. 提交 `docs: record verified G1 reference runtime`。
6. 在最终文档提交上重新跑 CPU Gate；可执行代码相对 GPU-tested commit 必须无差异。
7. 推送分支并等待 GitHub CI 成功。

#### Outputs

- G1 report、README 使用命令和最终 JSONL。
- 当前 commit 的新鲜 Gate 证据。
- 可评审、可快进合并的分支。

## 8. Per-task Acceptance Criteria

### W0 AC

- [ ] `main == origin/main`，worktree 初始干净。
- [ ] `python scripts/verify_g0_5.py` 退出码 0。
- [ ] 当前分支与 worktree 路径严格匹配第 3 节。

### R4.1 AC

- [ ] 三种空值/非法上界均抛 `ValueError`。
- [ ] 两个 dataclass 为 frozen、slots，修改字段抛异常。
- [ ] token tuple 含负数或为空时拒绝。
- [ ] 定向测试和全部 CPU Gate 退出码 0。

### R4.2 AC

- [ ] tokenizer/model 都收到固定 revision 与 `trust_remote_code=False`。
- [ ] model dtype 为 BF16、device 为 CUDA、training 为 False、use_cache 为 True。
- [ ] PAD token 行为可解释且与 EOS ID 一致或保留模型原值。
- [ ] loader integration test 退出码 0。

### R5 AC

- [ ] Prefill 输入完整 prompt，Cache 长度等于 prompt tokens。
- [ ] 每轮 decode 输入形状 `[1, 1]`，Cache 长度增加 1。
- [ ] 生成 N tokens 后 Cache 长度为 `prompt_tokens + N - 1`。
- [ ] EOS/length 停止原因正确，不分配梯度。
- [ ] 1/2/8/32 token 真模型 tests 退出码 0。

### R6 AC

- [ ] prompts JSON 恰有四个唯一 case ID。
- [ ] long prompt 实测 512–768 tokens。
- [ ] 四类 prompt 的 token tuple 与 HF 完全相同。
- [ ] 首个 mismatch index 能被单元测试精确验证。

### R7.1 AC

- [ ] Host 与 CUDA 字段不混用，TPOT 分母符合规格。
- [ ] NaN/Inf/负指标被拒绝。
- [ ] schema v1 必需字段全集由测试固定。
- [ ] synchronize 只由 `timer.py` 管理。

### R7.2 AC

- [ ] CLI 默认拒绝覆盖现有输出。
- [ ] 任一 agreement 小于 1.0 时退出非零并打印失败 case。
- [ ] 成功文件恰有 20 行，每行是独立 JSON object。
- [ ] 四个 case 各 5 条，repetition 为 0–4，warmup_runs 恒为 3。
- [ ] 成功最后一行为 `G1 REFERENCE RUNTIME: PASS`。

### G1 AC

- [ ] 设计规格第 15 节 13 条 Acceptance Criteria 全部逐项有新鲜证据。
- [ ] GitHub CI 对最终 commit 为 success。
- [ ] `git status --porcelain` 为空。
- [ ] 后续 KV/batching/paged/API/Triton 阶段仍标记 `not started`。

## 9. Gate / Exit Criteria

允许将 G1 标记 `verified` 且进入下一阶段的唯一 Gate：

1. 第 10 节所有 CPU/GPU/benchmark 命令退出码为 0。
2. `docs/results/g1_reference.jsonl` 存在且恰有 20 条有效记录。
3. 20 条 `token_agreement == 1.0`。
4. 每个 case 五次 output token tuple 一致。
5. 固定模型 SHA、Git SHA、环境版本、GPU 身份均写入结果与报告。
6. G1 report、README、Phase 1 workflow 状态一致。
7. worktree 干净，最终 commit 已推送，GitHub CI 成功。
8. 不存在 Non-goals 中的实现文件或完成声明。

缺少任一证据时状态只能是 `in progress` 或 `blocked`，不得使用 `verified`。

## 10. Verification Commands

工作目录统一为：

```bash
cd /home/administrator/projects/TinyInfer/.worktrees/hf-reference-runtime
```

### 10.1 CPU quality Gate

```bash
uv lock --check
python -m pytest -m "not gpu and not model and not slow"
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
```

预期：每条退出码 0；pytest 的精确通过数量以实现后收集值为准，不得在实施前伪造。

### 10.2 GPU/model integration Gate

```bash
python -m pytest tests/integration/test_hf_loader.py -m "gpu and model" -v
python -m pytest tests/integration/test_hf_reference_runtime.py -m "gpu and model" -v
```

预期退出码 0；输出必须显示实际执行而非 skip。若模型未缓存，首次运行允许下载但不得把下载时间写入
steady-state 指标。

### 10.3 G1 benchmark Gate

```bash
python benchmarks/bench_reference.py \
  --config configs/qwen2_5_0_5b.yaml \
  --prompts benchmarks/prompts.json \
  --output docs/results/g1_reference.jsonl \
  --warmup-runs 3 \
  --repetitions 5 \
  --overwrite
```

预期退出码 0；最后一行严格为 `G1 REFERENCE RUNTIME: PASS`。

### 10.4 Evidence validation

```bash
python - <<'PY'
import json
from collections import Counter, defaultdict
from pathlib import Path

path = Path("docs/results/g1_reference.jsonl")
records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
assert len(records) == 20
assert {record["case_id"] for record in records} == {
    "short_en", "short_zh", "structured", "long_context"
}
assert all(record["schema_version"] == 1 for record in records)
assert all(record["token_agreement"] == 1.0 for record in records)
assert set(Counter(record["case_id"] for record in records).values()) == {5}
grouped = defaultdict(set)
for record in records:
    grouped[record["case_id"]].add(tuple(record["output_token_ids"]))
assert all(len(outputs) == 1 for outputs in grouped.values())
print("G1_EVIDENCE_RECORDS=20")
print("G1_TOKEN_AGREEMENT=1.0")
print("G1_DETERMINISTIC_CASES=4")
PY
```

预期退出码 0，并输出三行指定证据。

### 10.5 Git/secret/boundary Gate

```bash
git status --short --branch
git diff --check
git grep -nE 'hf_[A-Za-z0-9]{20,}|github_pat_|ghp_|172\.31\.[0-9]+\.[0-9]+:7890' -- . \
  ':(exclude)docs/R4_R7_G1_WORKFLOW.md' \
  ':(exclude)docs/superpowers/plans/*' || true
git ls-files | grep -E '\.(safetensors|bin|pt|pth)$' && exit 1 || true
git grep -nE 'continuous batching.*(complete|verified)|paged KV.*(complete|verified)' -- . && exit 1 || true
```

预期：无 secret/model/boundary 命中；最终 `git status --porcelain` 为空。

## 11. Definition of Done

- **功能**：固定 Qwen 模型执行显式 prefill 与逐 token decode。
- **正确性**：四类 prompt 与 HF generate token tuple 完全相同，五次结果一致。
- **工程质量**：CPU/GPU pytest、Ruff、format、Mypy、diff-check、GitHub CI 全部成功。
- **可观测性**：20 条 JSONL 含 Host/CUDA 时间、吞吐和 allocated/reserved VRAM。
- **文档**：README、G1 report、Phase 1 workflow、设计和实现相互一致。
- **安全**：无 token、代理凭据、模型权重、cache 或敏感绝对路径。
- **可复现**：固定模型 SHA、Git SHA、依赖、硬件、warm-up=3、repetitions=5。
- **版本控制**：小提交、独立分支、无强推、最终 worktree 干净。
- **发布**：分支推送且 CI 成功；是否合并/打 G1 标签由完成分支流程单独决定。
- **边界**：只有 R4–R7 + G1 可标记 verified，Phase 2 及以后仍未开始。

## 12. Stop Conditions

立即停止并保留证据，不得猜测或跳过：

- 基线 Gate、lock check 或任一当前测试失败。
- Python/torch/transformers/GPU/model SHA 不匹配固定值。
- 网络不可用且模型未缓存。
- model loader 需要 `trust_remote_code=True` 或 `device_map="auto"` 才能运行。
- CUDA/BF16 不可用、OOM 或输出出现 NaN/Inf。
- token agreement 小于 1.0 或五次输出不一致。
- Cache length、decode input shape、EOS/PAD 行为无法解释。
- long prompt 不在 512–768 token 范围。
- benchmark 把模型加载/下载计入 steady-state。
- worktree 出现来源不明的修改、secret 或模型权重。
- 同一技术修复连续三次失败；此时必须重新审查设计而不是尝试第四个补丁。

## 13. Failure Handling

1. 创建本机临时证据目录：

   ```bash
   mkdir -p /tmp/tinyinfer-g1-evidence
   ```

2. 失败命令使用 `tee` 保存完整日志，同时保留退出码：

   ```bash
   set -o pipefail
   python -m pytest tests/integration/test_hf_reference_runtime.py::test_tinyinfer_tokens_match_hf -vv 2>&1 \
     | tee /tmp/tinyinfer-g1-evidence/pytest-failure.log
   ```

3. CUDA OOM 后只执行：

   ```bash
   nvidia-smi
   python scripts/check_environment.py --json
   ```

   不缩短 workload 或降低精度来伪造通过。

4. Alignment 失败时保存 request ID、input IDs、两侧 output IDs、首个 mismatch index、每步 Cache
   长度；禁止保存环境变量或 token。
5. 不使用 `git reset --hard`、`git checkout -- .` 或强推。若需撤销 agent 自己的单文件未提交变更，
   使用 `git diff -- <path>` 审查后通过补丁恢复。
6. 网络超时可重试一次；模型 SHA、测试、agreement、invariant 和 dirty worktree 问题不可通过重试
   绕过。
7. 阻塞时报告：失败命令、退出码、日志路径、当前 SHA、dirty files、已尝试次数和下一项最小调查。

## 14. Git / Worktree Rules

- **基线**：包含本 workflow 与 implementation plan 的最新 `origin/main`。
- **分支**：`agent/hf-reference-runtime`。
- **Worktree**：`/home/administrator/projects/TinyInfer/.worktrees/hf-reference-runtime`。
- **提交边界**：R4.1、R4.2、R5、R6、R7.1、R7.2、G1 docs 各自独立提交。
- **提交格式**：使用计划中给出的 `feat:`、`test:`、`docs:` 消息。
- **合并策略**：验证后优先 fast-forward；若 main 前进，先审查并重新验证。
- **标签**：本工作流不自动创建。完成分支流程获批后可创建 annotated `v0.2.0-g1`。
- **强推**：禁止。
- **清理**：只有合并成功并在 main 上重新验证后，才删除 worktree/branch。

## 15. Agent Handoff Contract

### 15.1 开始前必须报告

- 当前分支名与 HEAD SHA。
- `origin/main` SHA 与 `v0.1.0-g0.5^{}` SHA。
- worktree 绝对路径。
- `git status --short` 完整输出。
- Python、torch、transformers、CUDA、GPU 与 BF16 状态。
- 当前 Task 编号、输入依赖和明确 Non-goals。
- 确认没有根据旧截图/旧报告跳过新鲜 Gate。

### 15.2 每个 Task 结束必须报告

- 修改/新增文件路径和变动类型。
- RED 命令、预期失败原因与实际退出码。
- GREEN 命令、通过数量与实际退出码。
- 全套回归命令和结果。
- Acceptance Criteria 逐条映射。
- commit SHA 与 `git status`。
- 未完成、阻塞或风险事项。

### 15.3 G1 结束必须额外报告

- 四个 case 的 prompt/output tokens、agreement 与五次确定性摘要。
- JSONL 行数、schema、模型 SHA、Git SHA、环境身份。
- TTFT/TPOT/吞吐/显存的统计摘要及口径。
- GitHub CI URL 与 conclusion。
- 明确声明 Phase 2/KV/batching/paged/API/Triton 未实现。

Agent 不得仅提交勾选框、截图或“全部通过”文字作为证据。

## 16. Status Vocabulary

| 状态 | 精确定义 |
|---|---|
| `not started` | W0 尚未开始，或 Preconditions 未确认。 |
| `in progress` | W0–G1 至少一个 Action 已开始，但 Gate 未全部满足。 |
| `blocked` | Stop Condition 成立且当前 agent 无法安全继续，必须附失败证据。 |
| `verified` | 第 8–10 节全部在当前提交上产生新鲜 PASS，GitHub CI 成功且 worktree 干净。 |

`complete` 只可作为自然语言摘要；正式 Gate 状态使用以上四个词。
