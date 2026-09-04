# G2-P1 Batching Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible, correctness-gated GPU evidence comparing sequential, static-batch, and iteration-level continuous-batching throughput for TinyInfer's paged runtime.

**Architecture:** Add a Phase-2-specific workload contract and immutable per-repetition JSONL record rather than changing the frozen G1 record schema. A benchmark entrypoint will submit deterministic request schedules to the existing `BatchEngine`, collect request and aggregate timing/memory measurements, reject token mismatches, and atomically write raw records. A separate dependency-free summarizer will calculate median/p95 metrics from the raw JSONL; a fail-fast verifier will run unit checks, GPU correctness, benchmark, and summary without package installation or downloads.

**Tech Stack:** Python 3.11, PyTorch CUDA, Transformers, pytest, Ruff, mypy, JSONL, existing Qwen2.5-0.5B offline runtime.

**Spec:** `docs/superpowers/specs/2026-09-04-g2-performance-design.md`

## Global Constraints

- Use the fixed Qwen2.5-0.5B BF16 model, greedy decode, one GPU, and offline cached weights.
- Preserve `docs/results/g1_reference.jsonl`, `BenchmarkRecord`, and `benchmarks/bench_reference.py` as immutable G1 evidence.
- Run at least five measured repetitions after warmup; records contain revision, environment, workload, cache geometry, command, and per-request correctness evidence.
- P1 compares sequential single-request, all-at-once static batch, and deterministic staggered-arrival continuous batching. It does not claim block-native paged attention.
- A benchmark must fail before writing output on a token mismatch, missing completion, leak, duplicate request id, unsupported GPU, or invalid workload/record schema.
- GPU commands install no packages, download no model, do not silently shrink workloads, and use `HF_HUB_OFFLINE=1`.
- A null or negative result is valid evidence; report latency distributions beside throughput and never claim full G2 from P1.

---

## File structure

| File | Responsibility |
|---|---|
| `benchmarks/phase2_workloads.json` | Versioned, deterministic P1 workload definitions: request templates, mode-specific arrival iteration, and fixed measured matrix. |
| `tinyinfer/profiling/phase2.py` | Validated P1 record, quantile/aggregate calculations, and JSON-compatible serialization. |
| `benchmarks/bench_phase2.py` | GPU benchmark CLI, workload loader, mode runners, correctness gate, atomic JSONL writer, and dependency injection seams for unit tests. |
| `benchmarks/summarize_phase2.py` | Read-only raw-JSONL validator and deterministic Markdown/JSON summary CLI. |
| `scripts/verify_g2_p1.py` | Fixed-order, fail-fast local evidence gate. |
| `tests/unit/test_phase2_workloads.py` | Workload schema and deterministic-schedule tests. |
| `tests/unit/test_phase2_profiling.py` | Record validation, percentile, aggregate, and serialization tests. |
| `tests/unit/test_bench_phase2.py` | Runner, alignment, atomic-output, no-leak, and CLI failure tests using fakes. |
| `tests/unit/test_summarize_phase2.py` | Summary validation and deterministic output tests. |
| `tests/unit/test_verify_g2_p1.py` | Verifier order, fail-fast, timeout, and no-install/download tests. |
| `tests/integration/test_phase2_benchmark.py` | Native GPU/model P1 correctness: all modes agree with HF and release the pool. |
| `README.md`, `docs/PHASE2_STATUS.md`, `docs/BEGINNER_GUIDE.md` | Consistent capability and checkpoint boundary. |
| `docs/G2_P1_REPORT.md` | Final, evidence-backed P1 report; created only after fresh GPU runs. |

### Task 1: Normalize Phase-2 status documentation

**Files:**
- Modify: `README.md:35-44`
- Modify: `docs/PHASE2_STATUS.md:7-38`
- Modify: `docs/BEGINNER_GUIDE.md:800-812`
- Test: `tests/unit/test_phase2_docs.py`

**Interfaces:**
- Consumes: G2-BD evidence in `docs/G2_BD_REPORT.md` and approved G2-P design spec.
- Produces: One status vocabulary: G2-BD is a local candidate with remote-CI status to be checked; G2-P1/P2 are not started; full G2 is not passed.

- [ ] **Step 1: Add assertions for the current checkpoint vocabulary**

```python
def test_phase2_docs_state_checkpoint_boundaries() -> None:
    status = Path("docs/PHASE2_STATUS.md").read_text(encoding="utf-8")
    beginner = Path("docs/BEGINNER_GUIDE.md").read_text(encoding="utf-8")
    assert "G2-BD: local candidate" in status
    assert "G2-P1: not started" in status
    assert "G2-P2: not started" in status
    assert "full G2: not passed" in status
    assert "true batched decode: not verified" not in beginner
```

- [ ] **Step 2: Run the focused documentation test and confirm it fails**

Run: `python -m pytest tests/unit/test_phase2_docs.py -v`

Expected: FAIL because the old `not verified` wording is still present.

- [ ] **Step 3: Replace contradictory prose with the checkpoint table**

Use these exact semantic states in all three documents:

```text
G2-BD: local candidate — native-WSL correctness and audit remediation passed;
remote CI must be checked on the declared commit.
G2-P1: not started — batching-throughput evidence.
G2-P2: not started — paged-KV memory-efficiency evidence.
full G2: not passed.
```

Keep the dense left-padded gather limitation and link to the design spec. Do not alter the historical G1 claims.

- [ ] **Step 4: Re-run docs tests and check prose consistency**

Run: `python -m pytest tests/unit/test_phase2_docs.py -v && rg -n "true batched decode: not verified|G2-P: not started" README.md docs/PHASE2_STATUS.md docs/BEGINNER_GUIDE.md`

Expected: pytest PASS; `rg` returns no stale exact status phrase.

- [ ] **Step 5: Commit the documentation normalization**

```bash
git add README.md docs/PHASE2_STATUS.md docs/BEGINNER_GUIDE.md tests/unit/test_phase2_docs.py
git commit -m "docs: normalize Phase 2 checkpoint status"
```

### Task 2: Define deterministic Phase-2 workloads and measurement contracts

**Files:**
- Create: `benchmarks/phase2_workloads.json`
- Create: `tinyinfer/profiling/phase2.py`
- Modify: `tinyinfer/profiling/__init__.py`
- Test: `tests/unit/test_phase2_workloads.py`
- Test: `tests/unit/test_phase2_profiling.py`

**Interfaces:**
- Produces `load_phase2_workloads(path: Path) -> dict[str, Phase2Workload]`, `Phase2Request(request_id: str, case_id: str, prompt: str, max_new_tokens: int, arrival_iteration: int)`, and `Phase2Workload(workload_id: str, family: Literal["fixed", "mixed", "dynamic_mixed"], requests: tuple[Phase2Request, ...])`.
- Produces `Phase2Record` with `schema_version=1`, `mode: Literal["sequential", "static_batch", "continuous_batch"]`, one repetition's aggregate timing/memory fields, and a tuple of exact per-request token comparisons.
- Produces `summarize_records(records: Sequence[Phase2Record]) -> tuple[Phase2Summary, ...]`, grouping by `(workload_id, mode)` and calculating nearest-rank p50/p95 deterministically.

- [ ] **Step 1: Write failing workload and record-contract tests**

```python
def test_dynamic_workload_has_staggered_arrivals() -> None:
    workload = load_phase2_workloads(Path("benchmarks/phase2_workloads.json"))["dynamic_mixed_b4"]
    assert [request.arrival_iteration for request in workload.requests] == [0, 0, 1, 2]

def test_phase2_record_rejects_nonexact_token_agreement() -> None:
    with pytest.raises(ValueError, match="token_agreement"):
        Phase2RequestResult("r-0", (1,), (2,), 0.0, 0)

def test_summary_uses_nearest_rank_percentiles() -> None:
    summary = summarize_records(make_records(latencies_ms=[1.0, 2.0, 3.0, 4.0, 100.0]))[0]
    assert summary.end_to_end_p50_ms == 3.0
    assert summary.end_to_end_p95_ms == 100.0
```

- [ ] **Step 2: Run the two focused test modules and confirm failure**

Run: `python -m pytest tests/unit/test_phase2_workloads.py tests/unit/test_phase2_profiling.py -v`

Expected: FAIL because the workload loader and Phase-2 profiling contract do not exist.

- [ ] **Step 3: Add the strict workload JSON and validated types**

Use JSON schema version `1` with a `workloads` array. Include exactly these initial IDs: `fixed_b1`, `fixed_b4`, `mixed_b4`, and `dynamic_mixed_b4`; use request IDs unique within each workload, G1 frozen prompt text or copied fixed strings, `max_new_tokens` in `[1, 256]`, and non-negative arrival iterations. Require `fixed`/`mixed` requests all arrive at iteration zero; require `dynamic_mixed` to contain at least two distinct arrival iterations.

Implement the contract with immutable dataclasses and strict validation:

```python
@dataclass(frozen=True, slots=True)
class Phase2RequestResult:
    request_id: str
    output_token_ids: tuple[int, ...]
    reference_token_ids: tuple[int, ...]
    token_agreement: float
    first_mismatch_index: int | None

@dataclass(frozen=True, slots=True)
class Phase2Record:
    schema_version: int
    git_revision: str
    model_id: str
    model_revision: str
    transformers_version: str
    torch_version: str
    dtype: str
    device_name: str
    workload_id: str
    mode: Literal["sequential", "static_batch", "continuous_batch"]
    repetition: int
    warmup_runs: int
    command: tuple[str, ...]
    block_size: int
    num_physical_blocks: int
    request_results: tuple[Phase2RequestResult, ...]
    completed_requests: int
    generated_tokens: int
    end_to_end_ms: float
    requests_per_s: float
    generated_tokens_per_s: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    gpu_utilization_percent: float | None
    pool_free_blocks: int
    pool_total_blocks: int
```

`to_dict()` must serialize token tuples as lists and preserve a stable field order. `Phase2Summary` adds five-repetition p50/p95 throughput and latency values, and records `gpu_utilization_percent=None` as `null`.

- [ ] **Step 4: Run contract tests, formatting, and typing**

Run: `python -m pytest tests/unit/test_phase2_workloads.py tests/unit/test_phase2_profiling.py -v && python -m ruff check tinyinfer/profiling benchmarks/phase2_workloads.json && python -m mypy tinyinfer/profiling`

Expected: all commands PASS.

- [ ] **Step 5: Commit the workload and contract layer**

```bash
git add benchmarks/phase2_workloads.json tinyinfer/profiling/phase2.py tinyinfer/profiling/__init__.py tests/unit/test_phase2_workloads.py tests/unit/test_phase2_profiling.py
git commit -m "feat: add Phase 2 performance record contracts"
```

### Task 3: Implement correctness-gated P1 benchmark execution

**Files:**
- Create: `benchmarks/bench_phase2.py`
- Test: `tests/unit/test_bench_phase2.py`
- Test: `tests/integration/test_phase2_benchmark.py`

**Interfaces:**
- Consumes `Phase2Workload`, `Phase2Record`, `BatchEngine`, `GenerationRequest`, `reference_generate`, and model/cache dimensions.
- Produces `run_phase2_benchmark(config, workloads, modes, warmup_runs, repetitions, dependencies) -> list[Phase2Record]` and `write_phase2_records_atomic(records, output, overwrite) -> None`.
- `Phase2Dependencies` injects model loading, clock, CUDA facade, Git revision, and GPU-utilization sampler; unit tests never load Torch models.

- [ ] **Step 1: Write failing runner tests**

```python
def test_continuous_runner_submits_only_arrived_requests(fake_dependencies: Phase2Dependencies) -> None:
    record = run_one_workload("continuous_batch", dynamic_workload(), fake_dependencies)
    assert fake_dependencies.submitted_iterations == [("r0", 0), ("r1", 0), ("r2", 1), ("r3", 2)]

def test_runner_aborts_before_output_on_reference_mismatch(fake_dependencies: Phase2Dependencies) -> None:
    fake_dependencies.reference_tokens["r2"] = (99,)
    with pytest.raises(AlignmentError, match="r2"):
        run_one_workload("static_batch", mixed_workload(), fake_dependencies)

def test_atomic_writer_leaves_no_partial_output(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        write_phase2_records_atomic([valid_record()], tmp_path / "result.jsonl", False, failing_replace)
    assert not (tmp_path / "result.jsonl.tmp").exists()
```

- [ ] **Step 2: Run the unit benchmark module and confirm failure**

Run: `python -m pytest tests/unit/test_bench_phase2.py -v`

Expected: FAIL because `bench_phase2` and its public runner functions do not exist.

- [ ] **Step 3: Implement the three modes with explicit scheduling semantics**

Implement each measured repetition as follows:

```python
def run_sequential(workload: Phase2Workload, loaded: LoadedModel) -> dict[str, GenerationResult]:
    return {
        item.request_id: generate_with_paged_cache(loaded, to_generation_request(item))
        for item in workload.requests
    }

def run_engine(workload: Phase2Workload, *, continuous: bool, engine: BatchEngine) -> dict[str, GenerationResult]:
    pending = iter(sorted(workload.requests, key=lambda item: (item.arrival_iteration, item.request_id)))
    next_item = next(pending, None)
    iteration = 0
    while next_item is not None or not engine.queue.all_done():
        while next_item is not None and (not continuous or next_item.arrival_iteration <= iteration):
            engine.submit(to_generation_request(next_item))
            next_item = next(pending, None)
        engine.step()
        iteration += 1
    return {result.request_id: result for result in engine.queue.drain_finished()}
```

For `static_batch`, reject any nonzero arrival iteration and submit every request before the first `step()`. For continuous mode, use the workload's recorded arrival iterations. Before every measured run reset and synchronize CUDA peak-memory statistics; time the entire workload with a monotonic clock; record request TTFT/TPOT through an engine observer or explicit per-request timestamps. Load once per benchmark invocation, generate HF references once per request, require token-for-token agreement for every measured result, assert exactly one result per request, and assert `paged.free_block_count == paged.num_physical_blocks` after each run.

Expose CLI flags: `--config`, `--workloads`, `--output`, `--modes`, `--warmup-runs`, `--repetitions`, and `--overwrite`. Defaults are Phase-2 workload JSON, three warmups, and five repetitions. Reject duplicate modes, unsupported modes, output overwrite without `--overwrite`, a non-CUDA device, and `repetitions < 5`.

- [ ] **Step 4: Add and run the native GPU correctness integration test**

```python
@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("mode", ["sequential", "static_batch", "continuous_batch"])
def test_phase2_modes_match_hf_and_recover_pool(loaded_model: LoadedModel, mode: str) -> None:
    record = run_one_workload(mode, fixed_b4_workload(), real_dependencies(loaded_model))
    assert all(item.token_agreement == 1.0 for item in record.request_results)
    assert record.completed_requests == 4
    assert record.pool_free_blocks == record.pool_total_blocks
```

Run: `HF_HUB_OFFLINE=1 python -m pytest tests/unit/test_bench_phase2.py tests/integration/test_phase2_benchmark.py -m "gpu and model" -v`

Expected: unit tests PASS; native GPU integration reports 3 passed.

- [ ] **Step 5: Commit the P1 benchmark runner**

```bash
git add benchmarks/bench_phase2.py tests/unit/test_bench_phase2.py tests/integration/test_phase2_benchmark.py
git commit -m "feat: add G2-P1 benchmark runner"
```

### Task 4: Add deterministic summarization and a fail-fast evidence gate

**Files:**
- Create: `benchmarks/summarize_phase2.py`
- Create: `scripts/verify_g2_p1.py`
- Test: `tests/unit/test_summarize_phase2.py`
- Test: `tests/unit/test_verify_g2_p1.py`

**Interfaces:**
- Consumes `Phase2Record.from_dict(payload)`, `summarize_records`, and raw JSONL generated by Task 3.
- Produces `read_phase2_records(path) -> list[Phase2Record]`, `render_markdown(summaries) -> str`, and verifier `build_step_specs(python: str = PY) -> tuple[StepSpec, ...]`.

- [ ] **Step 1: Write failing summary and verifier tests**

```python
def test_summary_rejects_incomplete_repetitions(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [valid_record(repetition=index) for index in range(4)])
    with pytest.raises(ValueError, match="five measured repetitions"):
        summarize_path(path)

def test_verifier_stops_after_first_failure() -> None:
    results = execute_steps(STEP_SPECS, runner_returning_failure("pytest-unit"))
    assert [result.name for result in results] == ["lock-check", "pytest-unit"]
```

- [ ] **Step 2: Run focused modules and confirm failure**

Run: `python -m pytest tests/unit/test_summarize_phase2.py tests/unit/test_verify_g2_p1.py -v`

Expected: FAIL because the summarizer and verifier do not exist.

- [ ] **Step 3: Implement strict summarization and fixed gate order**

The summarizer must reject blank lines, malformed JSON, unknown keys, a schema-version mismatch, duplicate `(workload_id, mode, repetition)` tuples, non-exact agreement, and any group other than exactly five repetitions. Emit a Markdown table grouped by workload/mode, containing p50/p95 requests/s, tokens/s, end-to-end latency, TTFT, TPOT, and peak memory. Render missing utilization as `unavailable`.

Model `verify_g2_p1.py` on `scripts/verify_g2_bd.py`. Its exact order is:

```python
return (
    StepSpec("lock-check", (UV, "lock", "--check"), 120),
    StepSpec("pytest-unit", (python, "-m", "pytest", "tests/unit", "-q"), 900),
    StepSpec("pytest-gpu-phase2", (python, "-m", "pytest", "tests/integration/test_phase2_benchmark.py", "-m", "gpu and model", "-q"), 900),
    StepSpec("ruff-check", (python, "-m", "ruff", "check", "."), 120),
    StepSpec("ruff-format", (python, "-m", "ruff", "format", "--check", "."), 120),
    StepSpec("mypy", (python, "-m", "mypy"), 120),
    StepSpec("git-diff-check", ("git", "diff", "--check"), 120),
)
```

The verifier prints `G2-P1 VERIFICATION: PASS` only when every step passes. Unit tests assert its source contains none of `pip install`, `uv sync`, `git add`, `git commit`, a download command, or hard-coded machine paths.

- [ ] **Step 4: Run tests and the non-GPU quality checks**

Run: `python -m pytest tests/unit/test_summarize_phase2.py tests/unit/test_verify_g2_p1.py -v && python -m ruff check benchmarks/summarize_phase2.py scripts/verify_g2_p1.py && python -m ruff format --check benchmarks/summarize_phase2.py scripts/verify_g2_p1.py && python -m mypy`

Expected: all commands PASS.

- [ ] **Step 5: Commit summarization and the verifier**

```bash
git add benchmarks/summarize_phase2.py scripts/verify_g2_p1.py tests/unit/test_summarize_phase2.py tests/unit/test_verify_g2_p1.py
git commit -m "feat: add G2-P1 evidence gate"
```

### Task 5: Produce fresh evidence and close only G2-P1

**Files:**
- Create: `docs/results/g2_p1_raw.jsonl`
- Create: `docs/results/g2_p1_summary.md`
- Create: `docs/G2_P1_REPORT.md`
- Modify: `README.md:35-44`
- Modify: `docs/PHASE2_STATUS.md:7-38`
- Test: `tests/unit/test_phase2_docs.py`

**Interfaces:**
- Consumes the verifier, raw schema, summary CLI, fixed workload JSON, and native GPU environment.
- Produces evidence that can mark only `G2-P1` as locally verified; `G2-P2` and full G2 remain not passed.

- [ ] **Step 1: Add failing status-report assertions**

```python
def test_phase2_docs_link_p1_evidence_only_after_report_exists() -> None:
    text = Path("docs/PHASE2_STATUS.md").read_text(encoding="utf-8")
    assert "G2-P1: verified locally" in text
    assert "G2-P2: not started" in text
    assert "full G2: not passed" in text
    assert "docs/G2_P1_REPORT.md" in text
```

- [ ] **Step 2: Run the assertion and confirm failure before fresh evidence exists**

Run: `python -m pytest tests/unit/test_phase2_docs.py::test_phase2_docs_link_p1_evidence_only_after_report_exists -v`

Expected: FAIL because P1 is not yet evidenced and the report is absent.

- [ ] **Step 3: Run the gate and benchmark on native WSL/Linux GPU**

Run:

```bash
HF_HUB_OFFLINE=1 python scripts/verify_g2_p1.py
HF_HUB_OFFLINE=1 python benchmarks/bench_phase2.py \
  --config configs/qwen2_5_0_5b.yaml \
  --workloads benchmarks/phase2_workloads.json \
  --output docs/results/g2_p1_raw.jsonl \
  --warmup-runs 3 \
  --repetitions 5 \
  --overwrite
python benchmarks/summarize_phase2.py \
  --input docs/results/g2_p1_raw.jsonl \
  --output docs/results/g2_p1_summary.md
```

Expected: verifier PASS, exactly 60 raw records (4 workloads × 3 modes × 5 repetitions), and a summary that reports every workload/mode group. If any run is invalid, preserve the failure output outside committed evidence and do not change checkpoint status.

- [ ] **Step 4: Write an evidence-only report and update status**

`docs/G2_P1_REPORT.md` must name the tested implementation SHA, model/config/workload digests, OS/GPU/Torch/Transformers versions, exact commands, all mode results, token-alignment result, memory values, TTFT/TPOT distributions, utilization availability, and limitations. Link raw JSONL and summary. State `G2-P1: verified locally` only if Step 3 passed on the declared SHA; otherwise retain `not started`/`blocked` with the failure fact. In either case retain:

```text
G2-P2: not started
full G2: not passed
block-native paged attention: not implemented
```

- [ ] **Step 5: Run final verification and commit evidence**

Run: `HF_HUB_OFFLINE=1 python scripts/verify_g2_p1.py && python -m pytest tests/unit/test_phase2_docs.py -v && git diff --check`

Expected: all commands PASS before an evidence claim.

```bash
git add docs/results/g2_p1_raw.jsonl docs/results/g2_p1_summary.md docs/G2_P1_REPORT.md README.md docs/PHASE2_STATUS.md tests/unit/test_phase2_docs.py
git commit -m "docs: record G2-P1 performance evidence"
```

## Plan self-review

- **Spec coverage:** Task 1 resolves status ambiguity. Tasks 2–4 implement P1's deterministic workloads, five-repetition raw JSONL, correctness gate, aggregate latency/throughput/memory metrics, and fail-fast behavior. Task 5 creates evidence without overstating full G2. G2-P2 is intentionally deferred because the approved spec makes it a second checkpoint requiring a distinct contiguous-vs-paged concurrent-cache design after P1 evidence.
- **No placeholders:** All task actions name exact files, contracts, commands, expected failures/passes, and commit boundaries; no incomplete implementation markers remain.
- **Type consistency:** `Phase2Request`, `Phase2RequestResult`, `Phase2Record`, `Phase2Summary`, `Phase2Dependencies`, `run_phase2_benchmark`, and `write_phase2_records_atomic` are introduced once and consumed under those names in later tasks.
