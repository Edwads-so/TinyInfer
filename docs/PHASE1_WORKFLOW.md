# TinyInfer Phase 1 工作流程

> 文档版本：v1.1
> 制定日期：2026-07-20
> 最后修订：2026-07-22
> 适用范围：WSL2 环境基线与 Hugging Face Reference Runtime
> 状态：G0.5 环境基线修复与原生 WSL 验收进行中，R4 禁止开始

## 1. 目标与工作量判断

本阶段属于中等规模工程，不是大型推理系统开发。工作被严格限制为两部分：

1. **G0.5 Linux/GPU Environment Baseline**（进行中）：在现有 WSL2 Ubuntu 中建立可复现的
   Linux、Python、CUDA 和模型依赖环境，通过 GPU smoke test 验证。
2. **R4-R7 + G1 Reference Runtime**（待开始）：基于 Hugging Face Qwen2.5-0.5B 实现单请求、
   greedy、显式 prefill/decode 的 reference runtime。

G0.5 是环境基线门，不是完整 Phase 1。完整 Phase 1 以 G1 为退出门。

预计投入为 1–2 周兼职开发时间，新增约 8–12 个源码和测试文件，核心 Python 代码约 500–1000 行。工作量的主要风险来自环境、模型下载、正确性定义和计时方法，而不是代码行数。

本阶段不实现：

- 自定义 KV Cache。
- Continuous batching。
- Paged KV Cache。
- Scheduler。
- Triton 性能 kernel。
- HTTP 或 OpenAI-compatible API。
- 多模型、量化、随机 sampling、分布式推理。

这些功能必须在 G1（完整 Phase 1）验收通过后分别设计，禁止在 G0.5 或 R4-R7 中顺手加入。

## 2. 已验证的当前状态

以下状态已于 2026-07-20 实机检查，不是预期值：

| 项目 | 当前状态 | 判断 |
|---|---|---|
| Windows GPU | RTX 4080 SUPER，16376 MiB | 满足 V1 硬件目标 |
| Windows NVIDIA Driver | 591.86 | WSL CUDA 可见 |
| Windows PyTorch | 2.5.1+cu121 | 仅作为现状参考，不直接决定 Linux 版本 |
| Windows CUDA availability | `True` | 通过 |
| Windows BF16 | `True` | 通过 |
| WSL | 2.6.3.0 | 已安装，无需重新安装 WSL |
| WSL Kernel | 6.6.87.2-microsoft-standard-WSL2 | 通过 |
| 默认发行版 | Ubuntu-22.04，WSL version 2 | 通过 |
| 其他发行版 | docker-desktop，WSL version 2 | 不用于 TinyInfer 开发 |
| Ubuntu 用户 | `administrator` | 后续 Linux 路径以此为准 |
| Ubuntu Python | 3.10.12 | 不满足项目 Python 3.11 约束 |
| Ubuntu GPU | RTX 4080 SUPER 可见 | 通过 |
| Ubuntu `libcuda.so.1` | 存在 | 通过 |
| Ubuntu 根文件系统 | 约 954 GiB 可用 | 足够存放源码、模型和缓存 |
| 当前 Windows checkout | `D:\project\TinyInfer` | 保留，但不作为 Linux benchmark 工作目录 |
| WSL 访问当前目录 | `/mnt/d/project/TinyInfer` | 只用于检查，不用于正式 benchmark |
| GitHub HTTPS | WSL 内返回 HTTP 200 | 可用 |
| Hugging Face DNS | 可解析 | HTTPS 下载需在模型下载前再次验证 |
| Windows Git 代理 | `http://127.0.0.1:7890` | Windows 有效，WSL 不自动继承 |
| Git 状态 | `main` 跟踪 `origin/main` | 正常 |
| 未跟踪文件 | 根目录旧版 `PROJECT_PLAN.md` | 执行前归档 |

WSL 启动时出现 localhost 代理无法自动映射到 NAT 的警告。由于 GitHub 目前可以直接访问，本阶段不预先强制设置 WSL 代理；只有 Hugging Face 或 Python 包下载失败时才进入代理故障处理。

## 3. 固定设计决策

### 3.1 环境边界

- Windows 负责桌面工具、GitHub 登录、文档查看和必要的轻量检查。
- WSL2 Ubuntu-22.04 负责 Python、CUDA、Transformers、Triton、模型执行和 benchmark。
- 不在 Ubuntu 内安装 NVIDIA Linux 显卡驱动。WSL 使用 Windows NVIDIA 驱动提供的 CUDA 接口。
- 初始环境不安装完整 CUDA Toolkit；PyTorch/Triton smoke test 明确需要 `nvcc` 时再单独评估。

### 3.2 文件系统边界

Linux 主 checkout 固定为：

```text
/home/administrator/projects/TinyInfer
```

模型和编译缓存固定在 Linux 文件系统：

```text
/home/administrator/.cache/huggingface
/home/administrator/.cache/torch
/home/administrator/.triton
```

不得把正式 benchmark 的源码、模型或 Triton cache 放在 `/mnt/c` 或 `/mnt/d`。Windows checkout 与 WSL checkout 通过 GitHub 同步，不手工双向复制 `.git` 或虚拟环境。

### 3.3 Python 与依赖边界

- 项目 Python 固定为 3.11，与 `pyproject.toml` 的 `>=3.11,<3.12` 一致。
- 使用项目内 `.venv`，不修改 Ubuntu 系统 Python。
- 使用 `uv` 管理 Python、虚拟环境和 lock file。
- GPU 依赖版本必须通过 smoke test 后写入 lock file，不沿用 Windows 全局环境作为事实标准。
- Phase 1 只需要 PyTorch、Transformers、Safetensors、tokenizer 相关依赖和开发工具。
- Triton 只验证官方 Linux 包可导入并运行 vector-add smoke test，不在本阶段优化 attention。

### 3.4 模型与生成边界

- 模型固定为 `Qwen/Qwen2.5-0.5B`。
- dtype 固定为 BF16。
- device 固定为单张 CUDA GPU。
- sampling 固定为 greedy：`do_sample=False`。
- 每个请求固定一个 prompt 和 `max_new_tokens`。
- 不实现 streaming、batching、beam search、temperature、top-k 或 top-p。

## 4. 总体阶段与门槛

```text
R0 仓库清理
  → R1 WSL 网络与存储
  → R2 Python/Linux GPU 环境
  → R3 依赖锁定与 smoke tests
  → R4 Reference Runtime 数据模型
  → R5 显式 Prefill/Decode Loop
  → R6 Hugging Face 正确性对齐
  → R7 Profiling 与 JSONL 结果
  → G1 验收
```

任何阶段未过门槛时，不进入后面的功能。尤其是 correctness 未通过时，不开始 KV Cache 或性能优化。

## 5. R0：仓库清理与基线冻结

### 5.1 归档旧计划

Windows 根目录存在未跟踪的旧版 `PROJECT_PLAN.md`。执行前将其归档为：

```text
docs/archive/PROJECT_PLAN_v0.1.md
```

新版正式计划继续使用：

```text
docs/PROJECT_PLAN.md
```

归档提交只包含计划移动和说明，不混入环境或 runtime 代码。

### 5.2 确认远程基线

在 Windows checkout 中执行：

```powershell
git status --short --branch
git log -5 --oneline --decorate
git remote -v
git fetch origin
git rev-parse main
git rev-parse origin/main
```

预期结果：

- 除待归档旧计划外没有未知修改。
- `main` 与 `origin/main` 指向相同提交。
- `origin` 为 `https://github.com/Edwads-so/TinyInfer.git`。

### 5.3 基线测试

如果 Windows checkout 没有 `.venv`，先建立仅包含开发依赖的临时环境；不要在这里安装 GPU runtime。执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m mypy
```

R0 门槛：仓库状态可解释，现有 13 项测试和所有质量检查通过。

## 6. R1：WSL 网络、GitHub 与 Linux checkout

### 6.1 启动并确认发行版

在 PowerShell 中执行：

```powershell
wsl --status
wsl --list --verbose
wsl -d Ubuntu-22.04
```

在 Ubuntu 中执行：

```bash
whoami
cat /etc/os-release
uname -r
nvidia-smi
test -e /usr/lib/wsl/lib/libcuda.so.1
```

预期用户为 `administrator`，发行版为 Ubuntu 22.04，GPU 为 RTX 4080 SUPER。

### 6.2 网络探测顺序

依次执行：

```bash
curl -I -L --max-time 15 https://github.com
curl -I -L --max-time 15 https://pypi.org
curl -I -L --max-time 15 https://huggingface.co
```

只有实际失败时才配置代理。

若 WSL NAT 无法直接访问，而 Windows 代理端口为 7890：

```bash
HOST_IP=$(ip route show default | awk '{print $3}')
curl -I --max-time 15 -x "http://${HOST_IP}:7890" https://github.com
```

该测试只有在 Windows 代理允许局域网访问、监听 WSL 可达地址时才会成功。不得为了省事把代理服务暴露到不可信网络。另一种方案是在 Windows 11 的 WSL 配置中使用 mirrored networking 与 autoProxy；这会改变全局 WSL 网络，应单独确认并在 `wsl --shutdown` 后验证，不在本流程中自动修改。

### 6.3 安装基础工具

```bash
sudo apt update
sudo apt install -y git git-lfs curl ca-certificates build-essential
git lfs install
```

GitHub 仓库是私有仓库，WSL 需要独立认证。优先安装 GitHub CLI 后使用设备授权：

```bash
sudo apt install -y gh
gh auth login --hostname github.com --git-protocol https --web
gh auth status
```

不得把 GitHub token 写入仓库、shell history、`.env` 或 workflow 文档。

### 6.4 Clone 到 Linux 文件系统

```bash
mkdir -p /home/administrator/projects
cd /home/administrator/projects
gh repo clone Edwads-so/TinyInfer
cd TinyInfer
git status --short --branch
git remote -v
```

预期路径为：

```text
/home/administrator/projects/TinyInfer
```

R1 门槛：GitHub、PyPI、Hugging Face 可访问；Linux checkout 位于 `/home`；私有仓库 fetch/pull 正常。

## 7. R2：Python 3.11 与项目虚拟环境

### 7.1 安装 uv

先下载官方安装脚本到临时文件并人工检查，再执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
sed -n '1,220p' /tmp/uv-install.sh
sh /tmp/uv-install.sh
source "$HOME/.local/bin/env"
uv --version
```

不要使用 `sudo pip install`，不要替换 `/usr/bin/python3`。

### 7.2 创建 Python 3.11 环境

```bash
cd /home/administrator/projects/TinyInfer
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
python --version
python -c "import sys; print(sys.executable)"
```

预期：

- Python 为 3.11.x。
- executable 位于项目 `.venv/bin/python`。

### 7.3 安装现有开发依赖

```bash
uv pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

R2 门槛：Linux 环境中现有测试和质量检查全部通过。

## 8. R3：GPU 依赖锁定与 Smoke Tests

### 8.1 版本选择原则

执行环境建设时选择一组相互兼容的稳定版本，并记录：

- Python。
- PyTorch。
- PyTorch CUDA build。
- Transformers。
- Safetensors。
- Triton。
- NVIDIA driver。
- WSL 与 kernel。

版本选择后必须生成并提交 `uv.lock`。在 smoke test 通过前，不把未经验证的版本写进 README 性能环境。

### 8.2 安装顺序

```bash
source /home/administrator/projects/TinyInfer/.venv/bin/activate
cd /home/administrator/projects/TinyInfer
uv add --optional inference torch transformers safetensors
uv add --optional kernels triton
uv lock
uv sync --extra dev --extra inference --extra kernels
```

若 PyTorch 官方要求专用 wheel index，应按官方 Linux CUDA 安装指令配置 uv index，再重新 lock；不得同时混用多个来源中同名但不同 CUDA build 的 PyTorch wheel。

### 8.3 PyTorch CUDA smoke test

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("bf16:", torch.cuda.is_bf16_supported())

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("result:", y.shape, y.dtype, y.device)
PY
```

要求：CUDA、4080 SUPER、BF16 和矩阵乘法全部成功。

### 8.4 Transformers smoke test

先只验证导入与版本，不下载模型：

```bash
python - <<'PY'
import safetensors
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("safetensors", safetensors.__version__)
PY
```

### 8.5 Triton smoke test

实现一个独立 vector-add smoke test，与 `torch.add` 比较：

- 输入长度覆盖 1、257、4096、65537。
- 覆盖非整 block mask。
- 使用 `torch.testing.assert_close`。
- 首次运行记录编译时间，但不将其当作 steady-state kernel 时间。
- 第二次运行确认 cache 命中并输出 kernel 时间。

该测试只证明官方 Linux Triton 可工作，不作为 TinyInfer 的性能贡献。

### 8.6 环境快照

G0.5 候选分支已提供 GPU smoke test 脚本（`scripts/check_gpu_smoke.py`）和环境检查器
（`scripts/check_environment.py`）。两者职责分离：

- `scripts/check_environment.py`：轻量级能力检查，报告包、CUDA、GPU 和 BF16 状态。
- `scripts/check_gpu_smoke.py`：真实 GPU correctness 验证，包含 BF16 matmul 和
  Triton vector-add。

环境快照保存为 JSON，不包含用户名、token、代理凭据或其他秘密。

R3 门槛：PyTorch、Transformers 和 Triton smoke tests 全部通过；`uv.lock` 可以在全新
`.venv` 中重建环境；GPU smoke test 返回零退出码。

## 9. Phase 1 代码结构

建议新增：

```text
tinyinfer/
├── runtime/
│   ├── __init__.py
│   ├── request.py
│   ├── hf_loader.py
│   ├── generation.py
│   └── result.py
├── profiling/
│   ├── __init__.py
│   ├── timer.py
│   └── record.py
benchmarks/
├── prompts.json
└── bench_reference.py
tests/
├── unit/
│   ├── test_request.py
│   ├── test_result.py
│   └── test_record.py
└── integration/
    └── test_hf_reference_runtime.py
```

职责边界：

| 文件 | 单一职责 |
|---|---|
| `request.py` | 保存 prompt token、生成上限和请求标识，不执行模型 |
| `hf_loader.py` | 加载 tokenizer/model，设置 dtype、device 和 eval 状态 |
| `generation.py` | 显式执行 prefill、argmax 和逐 token decode |
| `result.py` | 保存生成 token、停止原因和文本 |
| `timer.py` | CUDA event 与 host clock 计时，集中处理 synchronize |
| `record.py` | 构造可序列化 benchmark 记录 |
| `bench_reference.py` | CLI、warm-up、重复运行和 JSONL 输出 |

不得把请求状态、模型加载、生成循环、计时和 JSON 输出全部写进一个文件。

## 10. Reference Runtime 数据流

```text
prompt text
  → tokenizer
  → input_ids / attention_mask
  → prefill forward(use_cache=True)
  → logits[:, -1, :]
  → argmax next token
  → past_key_values
  → decode forward(one token + cache)
  → repeated argmax
  → EOS or max_new_tokens
  → GenerationResult
```

### 10.1 Prefill 定义

- 输入完整 prompt。
- 使用 `torch.inference_mode()`。
- `use_cache=True`。
- 保存返回的 `past_key_values`。
- 从最后一个位置 logits 选择第一个输出 token。

### 10.2 Decode 定义

- 每轮只输入上轮生成的一个 token。
- 传入并更新 `past_key_values`。
- 使用 greedy argmax。
- 遇到 EOS 或达到 `max_new_tokens` 时终止。

### 10.3 停止条件

`GenerationResult.stop_reason` 只允许：

```text
eos
length
```

模型异常、CUDA OOM 和输入非法是错误，不伪装成正常停止原因。

## 11. TDD 实施顺序

每个任务遵循：失败测试 → 确认失败原因 → 最小实现 → 全套测试 → 小提交。

### Task 1：Request 与 Result

先测试：

- 空请求标识被拒绝。
- 空 prompt token 被拒绝。
- `max_new_tokens <= 0` 被拒绝。
- generated token append 顺序稳定。
- 结果对象不可变或只暴露明确的状态更新方法。

不加载模型，不需要 GPU。

### Task 2：模型加载器

先测试配置映射和错误信息，再做 GPU integration test：

- 正确模型 ID。
- BF16。
- CUDA device。
- `eval()`。
- tokenizer 的 EOS/PAD 处理明确。
- 加载错误包含模型 ID 和原始异常类型。

### Task 3：单次 Prefill

先验证：

- output logits shape。
- `past_key_values` 非空。
- 第一个 token 与直接模型 forward 的 argmax 一致。

### Task 4：显式 Decode Loop

测试 1、2、8、32 个输出 token，验证：

- 每轮只追加一个 token。
- cache 随上下文增长。
- EOS 与 length 停止原因正确。
- 不发生梯度分配。

### Task 5：与 `model.generate()` 对齐

固定参数：

```text
do_sample=False
num_beams=1
use_cache=True
相同 eos_token_id
相同 pad_token_id
相同 max_new_tokens
```

逐 token 比较，不只比较最终字符串。

### Task 6：计时与结果记录

先用 CPU fake clock 测试公式，再接 CUDA event。计时模块不能散落调用 `time.time()` 或 `torch.cuda.synchronize()`。

## 12. Correctness 协议

固定至少四类 prompt：

1. 短英文事实问题。
2. 短中文问题。
3. 多行结构化文本。
4. 512 token 以上长 prompt。

每个 prompt 生成 32 tokens，记录：

- prompt token IDs。
- reference output token IDs。
- TinyInfer output token IDs。
- token agreement。
- 首个分歧位置。
- 最终文本，仅用于人工阅读。

系统级验收要求：

- 所有固定 prompt 的 token agreement 为 100%。
- 相同环境连续运行三次结果一致。
- 出现分歧时先比较 tokenizer 输入、EOS/PAD 配置、第一步 logits 和 cache 长度，不通过修改文本后处理掩盖差异。

模型和 tokenizer 应固定具体 revision，并将 revision 写入 benchmark 记录，避免 Hugging Face 默认分支变化导致结果漂移。

## 13. Profiling 与指标定义

### 13.1 两层计时

同时记录：

- Host end-to-end：包含 Python、tokenizer、调度与 CUDA 调用。
- CUDA device time：只表示 GPU 工作时间。

两者不能混为一个指标。

### 13.2 指标

| 指标 | 定义 |
|---|---|
| `prompt_tokens` | tokenizer 产生的输入 token 数 |
| `output_tokens` | 实际生成 token 数 |
| `prefill_ms` | 完整 prompt forward 到第一个 logits 可用的 GPU/host 时间 |
| `ttft_ms` | benchmark 请求开始到第一个输出 token 可用的 host 时间 |
| `decode_total_ms` | 第 2 个输出 token到最后一个输出 token的总时间 |
| `tpot_ms` | `decode_total_ms / max(output_tokens - 1, 1)` |
| `output_tokens_per_s` | steady-state 输出 token throughput |
| `peak_allocated_bytes` | `torch.cuda.max_memory_allocated()` |
| `peak_reserved_bytes` | `torch.cuda.max_memory_reserved()` |

### 13.3 测量规则

- 模型加载不计入 steady-state inference。
- 正式记录前至少 warm-up 3 次。
- 每个 workload 至少重复 5 次。
- 每轮测量前重置 peak memory stats。
- CUDA event 读取前必须 synchronize。
- 分开记录首轮编译、首次模型加载和 steady-state。
- 不在一次结果中混合不同 prompt token 长度。

### 13.4 JSONL 最小字段

```json
{
  "schema_version": 1,
  "git_revision": "commit-sha",
  "model_id": "Qwen/Qwen2.5-0.5B",
  "model_revision": "resolved-revision",
  "dtype": "bfloat16",
  "device": "NVIDIA GeForce RTX 4080 SUPER",
  "prompt_tokens": 128,
  "output_tokens": 32,
  "prefill_ms": 0.0,
  "ttft_ms": 0.0,
  "decode_total_ms": 0.0,
  "tpot_ms": 0.0,
  "peak_allocated_bytes": 0,
  "peak_reserved_bytes": 0,
  "token_agreement": 1.0,
  "warmup_runs": 3,
  "repetition": 0
}
```

示例中的 `0.0` 和 `0` 是 schema 类型示例，不是预期性能结果。

## 14. 模型下载与缓存

```bash
export HF_HOME=/home/administrator/.cache/huggingface
mkdir -p "$HF_HOME"
```

Qwen2.5-0.5B 是公开模型，正常情况下不需要把 Hugging Face token 写入项目。若网络失败：

1. 先验证 DNS。
2. 再验证 HTTPS。
3. 再验证代理。
4. 最后检查 Hugging Face 服务和本地缓存。

下载完成后记录模型实际 revision。不要把模型权重复制到仓库或提交到 Git LFS，除非之后明确改变资产策略。

## 15. Git、Worktree 与提交工作流

所有 Phase 1 代码在 Linux checkout 中开发：

```bash
cd /home/administrator/projects/TinyInfer
git switch main
git pull --ff-only
```

实现前建立隔离 worktree 和分支：

```text
branch: agent/hf-reference-runtime
worktree: .worktrees/hf-reference-runtime
```

建议提交边界：

```text
Archive legacy project plan
Lock Linux inference environment
Add request and generation result models
Add Hugging Face model loader
Add explicit greedy prefill and decode loop
Add Hugging Face correctness comparison
Add reference runtime profiling records
Document Phase 1 results
```

每次提交只包含一个可独立评审的行为。模型权重、`.venv`、cache、token、原始 profiler 巨型文件不得提交。

当前 Codex workspace 指向 Windows 路径。Linux checkout 建立后，应把工作区切换到 WSL 路径或从 WSL 环境启动开发工具。不要一边让 Codex 修改 Windows checkout，一边在 Linux checkout 手工修改相同文件。

## 16. 错误处理与停止条件

### 环境停止条件

出现以下任一问题时，停止写 runtime：

- WSL 无法稳定访问 GitHub、PyPI 或 Hugging Face。
- Python 不是 3.11。
- `torch.cuda.is_available()` 为 `False`。
- GPU 名称不是预期设备。
- BF16 不可用。
- Triton vector-add correctness 失败。
- lock file 无法从空环境重建。

### Runtime 停止条件

出现以下任一问题时，停止 profiling：

- 与 `model.generate()` token 不一致。
- 未明确 EOS/PAD 行为。
- prefill 与 decode 没有分离。
- 计时包含模型加载却未标注。
- benchmark 没有保存环境和 Git revision。

### 明确不采用的补救方式

- 不通过降低数值精度掩盖错误。
- 不通过更换 prompt 隐藏 token 分歧。
- 不通过缩短 workload 隐藏 OOM。
- 不在 Windows Triton fork 上绕过 Linux 官方 Triton 问题。
- 不因 Phase 1 性能较慢而提前实现 paged attention。

## 17. Phase 1 验收清单

### G0.5 Linux/GPU Environment Baseline（进行中）

- [x] WSL2 Ubuntu-22.04 状态、kernel 和 GPU 信息已记录。
- [x] Linux checkout 位于 `/home/administrator/projects/TinyInfer`。
- [ ] Python 3.11 项目环境已在全新原生 WSL worktree 中从 lock file 重建。
- [x] PyTorch CUDA、BF16、Transformers、Safetensors 导入通过。
- [ ] 候选脚本在原生 WSL 中连续两次通过 Triton vector-add correctness。
- [x] 环境报告不包含秘密信息。
- [ ] 候选分支的单元测试、Ruff、format 和 mypy 通过。
- [ ] 独立 GPU smoke test 可重复执行并返回明确退出码。
- [ ] G0.5 验收入口（`scripts/verify_g0_5.py`）编排全部质量检查。
- [x] 工作流编写标准与模板已固化（`docs/WORKFLOW_AUTHORING_STANDARD.md`）。

### Runtime G1（待开始）

- [ ] Qwen2.5-0.5B tokenizer 和模型从固定 revision 加载。
- [ ] 显式 prefill 产生第一个 greedy token 和 cache。
- [ ] 显式 decode 每轮只输入一个 token并更新 cache。
- [ ] 四类固定 prompt 连续生成 32 tokens。
- [ ] TinyInfer 与 `model.generate()` token agreement 为 100%。
- [ ] 相同环境连续三次输出一致。
- [ ] TTFT、prefill、decode、TPOT、吞吐、allocated/reserved VRAM 被记录。
- [ ] 每个正式 workload warm-up 3 次并重复至少 5 次。
- [ ] JSONL 包含环境、model revision 和 Git revision。
- [ ] README 能用一组命令复现 correctness 和 reference benchmark。

## 18. G0.5 当前决策

G0.5 尚未验收通过。以下为候选命令，必须在修复质量门并转移到原生 WSL worktree 后执行：

```bash
# 完整 G0.5 验证（在 WSL2 环境中）
python scripts/verify_g0_5.py

# 仅 GPU correctness smoke test
python scripts/check_gpu_smoke.py

# 环境能力检查
python scripts/check_environment.py --json
```

只有上述命令和本节验收清单全部产生新鲜 PASS 证据后，才允许进入 R4。具体见本流程第 9-13 节。

## 19. G1 完成后的下一决策

只有 G1 全部通过后，才为 Phase 2 单独设计以下内容：

1. 统一 Cache 接口。
2. Contiguous KV Cache adapter。
3. Request 状态机与 iteration-level scheduler。
4. Continuous batching。
5. Paged block pool、block table 和 Qwen attention adapter。

Phase 1 的 profiling 数据只用于建立 reference 和发现瓶颈，不直接得出 Triton kernel 方向。Triton 性能工作必须等待 paged runtime 正确后再由 profiling 选择。

## 20. 本阶段交付物

### G0.5 候选交付物（待验收）

- 本流程文档。
- WSL2/Linux 环境 lock file（`uv.lock`）。
- 可机器读取的环境报告（`docs/environment_report.json`）。
- pip list 快照（`docs/environment_pip_list.txt`）。
- 独立 GPU smoke test（`scripts/check_gpu_smoke.py`）。
- G0.5 验收入口（`scripts/verify_g0_5.py`）。
- G0.5 阶段报告（`docs/G0_5_REPORT.md`）。
- 工作流编写标准与模板（`docs/WORKFLOW_AUTHORING_STANDARD.md`、
  `docs/templates/WORKFLOW_TEMPLATE.md`）。
- 验收后创建的 `v0.1.0-g0.5` 标签（当前不存在）。

### G1 后续交付物（待 R4-R7 实现）

- 单请求 Hugging Face reference runtime。
- correctness 固定 prompt 集。
- JSONL reference benchmark。
- Phase 1 完成报告。
- `v0.1.0-phase1`（或等价标签）。
- 一次经过测试和代码评审的 Git 分支合并。

完成全部交付物后，TinyInfer 将从"工程骨架"进入"可运行的 LLM inference reference"，
同时仍保持范围小、可解释和可验证。
