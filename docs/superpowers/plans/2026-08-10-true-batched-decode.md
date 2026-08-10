# True Batched Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify one-forward heterogeneous batched decode over TinyInfer's shared paged KV pool without claiming block-native attention or full G2 performance completion.

**Architecture:** Keep `PagedKVCache` as the long-lived physical block pool, move logical length and ownership into one `PagedRequestState` per request, and create an ephemeral `BatchedPagedTinyinferCache` for each decode iteration. The adapter writes `[B, Hkv, 1, D]` KV rows into per-request block tables, gathers left-padded dense KV, and lets one Hugging Face Qwen forward process all eligible running requests.

**Tech Stack:** Python 3.11.15, PyTorch 2.13.0+cu130, Transformers 5.14.1, pytest 8.x, Ruff 0.9.x, Mypy 1.14+, uv, WSL2 Ubuntu 22.04, NVIDIA RTX 4080 SUPER, BF16.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-10-true-batched-decode-design.md`; an implementer may not change its architecture without controller approval.
- The fixed model is `Qwen/Qwen2.5-0.5B` at revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Model/GPU verification runs in native WSL2, with `HF_HUB_OFFLINE=1` after confirming the fixed model is cached.
- Use TDD for every behavior change: record a failing RED command before implementation and a passing GREEN command afterward.
- Preserve serial prefill; only decode is batched in this workflow.
- Dense left-padded gather is the selected adapter; do not implement block-native attention or a Triton kernel.
- Do not lower or rewrite the full G2 definition in `docs/PROJECT_PLAN.md`.
- G2-BD may become `verified` only after Task 9; G2-P and full G2 remain incomplete.
- The controller creates the worktree. Task implementers must not create, delete, move, or repoint worktrees.
- Do not push, merge, tag, or force-push unless the controller starts the branch-finishing workflow.

---

## 1. Objective

At the end of this workflow, one scheduling iteration passes `[B, 1]` input IDs for all eligible running requests to exactly one Qwen model forward. Every row uses its own block table and logical position in a shared paged KV pool, and three heterogeneous requests match their own Hugging Face references token-for-token.

The deliverable is the **G2-BD correctness checkpoint**, not G2-P performance evidence and not the full G2 release.

## 2. Non-goals

- Do not batch or chunk prefill.
- Do not implement paged attention, FlashAttention integration, Triton gather, or custom Qwen attention.
- Do not add sampling, beam search, speculative decoding, prefix caching, preemption, swap, distributed inference, or an API server.
- Do not add length-bucketing heuristics.
- Do not make `B=1/2/4/8` performance claims; those belong to a later G2-P plan.
- Do not modify canonical G1 JSONL or reinterpret the historical G1 report as a Phase 2 report.
- Do not create a G2 tag.
- Do not accept final-token equality as the only proof that the batched path ran.

## 3. Preconditions

- **Baseline:** local `main` contains approved design commit `1c1333a` and this plan commit.
- **Worktree:** controller-created branch `agent/true-batched-decode` at `/home/administrator/projects/TinyInfer/.worktrees/true-batched-decode`.
- **OS:** WSL2 Ubuntu 22.04; Windows PowerShell is only the controller shell.
- **Python:** 3.11.15 from the locked `.venv`.
- **Runtime:** Torch 2.13.0+cu130, Transformers 5.14.1, Triton 3.7.1.
- **GPU:** RTX 4080 SUPER; CUDA and BF16 support are true.
- **Tools:** `git`, `uv`, `python`, and locked dev/inference dependencies are available.
- **Model cache:** the fixed Qwen revision loads with `HF_HUB_OFFLINE=1`.
- **Disk:** at least 20 GiB free in WSL.
- **Git:** worktree starts clean at the controller-declared SHA.

## 4. Baseline Evidence

The controller runs and reports these commands before delegating Task 0:

```bash
pwd
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
git status --short
git log -5 --oneline --decorate
git merge-base --is-ancestor 1c1333a HEAD
uv lock --check
.venv/bin/python - <<'PY'
import torch
import transformers
print(torch.__version__)
print(transformers.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
PY
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
git diff --check
```

Known design-time failures are 2 unit failures, 21 Ruff errors, 15 files requiring formatting, and 3 Mypy errors. Fresh output is authoritative; a materially different baseline is a stop condition.

## 5. Artifact Manifest

| Path | Change | Responsibility |
|---|---|---|
| `scripts/verify_g1.py` | modify | Repair existing Ruff findings only. |
| `scripts/verify_gpu_paths.py` | modify | Remove hard-coded path; repair typing/formatting. |
| `tinyinfer/profiling/timer.py` | modify | Type the CUDA event return. |
| `tinyinfer/cache/adapter.py` | modify | Repair existing lint violations. |
| `tinyinfer/cache/block.py` | modify | Track allocation, reject double-free, clear mappings. |
| `tinyinfer/cache/paged.py` | modify | Atomic lifecycle, batch writes, left-padded gather. |
| `tinyinfer/cache/request_state.py` | create | Per-request table, committed length, release state. |
| `tinyinfer/cache/paged_tinyinfer_cache.py` | modify | Share request state; correct HF offsets. |
| `tinyinfer/cache/batched_paged_tinyinfer_cache.py` | create | Ephemeral heterogeneous HF Cache view. |
| `tinyinfer/cache/__init__.py` | modify | Export new cache contracts. |
| `tinyinfer/cache/tinyinfer_cache.py` | modify | Existing import/offset quality repair. |
| `tinyinfer/runtime/generation.py` | modify | Shared state and batch-safe token selection. |
| `tinyinfer/scheduler/state.py` | create | Typed engine state and iteration statistics. |
| `tinyinfer/scheduler/queue.py` | modify | Quality repair and state query. |
| `tinyinfer/scheduler/engine.py` | modify | One-forward decode, cancellation, cleanup, metrics. |
| `tinyinfer/scheduler/__init__.py` | modify | Export engine statistics. |
| `tests/unit/test_allocation.py` | modify | Real dtype and lifecycle invariants. |
| `tests/unit/test_contiguous_cache.py` | modify | Existing Ruff repair only. |
| `tests/unit/test_phase2_docs.py` | create | Guard truthful capability status. |
| `tests/unit/test_paged_request_state.py` | create | State and single-cache protocol. |
| `tests/unit/test_paged_batch_io.py` | create | Real-tensor batch I/O content. |
| `tests/unit/test_batched_paged_cache.py` | create | HF protocol and layer transaction. |
| `tests/unit/test_batch_engine.py` | create | One-forward shape/mask/row mapping. |
| `tests/unit/test_continuous_admission.py` | create | Dynamic membership and cleanup. |
| `tests/integration/test_batch_engine.py` | rename | Replace the old serial-forward proof. |
| `tests/integration/test_batch_decode.py` | create | Real Qwen path identity and alignment. |
| `scripts/verify_g2_bd.py` | create | Fail-fast local G2-BD gate. |
| `tests/unit/test_verify_g2_bd.py` | create | Verifier behavior. |
| `README.md` | modify | Truthful per-capability status. |
| `docs/BEGINNER_GUIDE.md` | modify | Explain prototype and dense-gather boundary. |
| `docs/PHASE2_STATUS.md` | create | Current Phase 2 status. |
| `docs/G2_BD_REPORT.md` | create | Final evidence and remaining boundary. |

Do not commit `.venv/`, model/cache files, Python/tool caches, logs, profiler dumps, screenshots, G2-P figures, changes to `docs/results/g1_reference.jsonl`, or changes to `uv.lock`.

## 6. Detailed Workflow

1. **Task 0:** Restore the existing quality baseline.
2. **Task 1:** Correct Phase 2 status documentation.
3. **Task 2:** Make block allocation and release atomic.
4. **Task 3:** Introduce shared per-request paged state.
5. **Task 4:** Add batched paged KV write and left-padded gather.
6. **Task 5:** Implement the ephemeral batched HF Cache adapter.
7. **Task 6:** Replace serial decode calls with one batched model forward.
8. **Task 7:** Verify cancellation and dynamic continuous admission.
9. **Task 8:** Prove real Qwen path identity and token agreement on GPU.
10. **Task 9:** Add the verifier, report evidence, and close G2-BD only.

Tasks execute strictly in order. Each starts from the previous clean commit and ends with one reviewable clean commit.

## 7. Tasks

### Task 0: Restore the Existing Quality Baseline

**Files:** modify only the existing files listed under Task 0 in the Artifact Manifest.

**Interfaces:** consumes current behavior; produces a green pre-feature baseline and no new runtime API.

#### Inputs

- Clean feature worktree and fresh Section 4 evidence.
- No true-batching implementation has started.

#### Actions

- [ ] **Step 0.1: Reproduce the dtype failures.**

```bash
.venv/bin/python -m pytest \
  tests/unit/test_allocation.py::TestPagedKVCacheAllocationBalance -q
```

Expected RED: two failures because real `torch.zeros` receives dtype string `"bfloat16"`.

- [ ] **Step 0.2: Correct the tests, not production Torch semantics.**

Use `torch = pytest.importorskip("torch")` and pass `dtype=torch.bfloat16` in both pool constructions. Run the same command; expected GREEN is `2 passed`.

- [ ] **Step 0.3: Repair the exact Ruff findings.**

Make these mechanical changes:

```text
scripts/verify_g1.py: wrap StepSpec; use the Ruff-requested conditional expression
scripts/verify_gpu_paths.py: remove sys.path hard-code; sort imports; wrap prompts;
  expand one-line if statements; annotate main() -> int; raise SystemExit(main())
tests/integration/test_batch_engine.py: sort imports; zip(..., strict=True)
tests/unit/test_contiguous_cache.py: wrap the 104-character docstring
tinyinfer/cache/adapter.py: remove two unused noqa directives and old_length
tinyinfer/cache/block.py: remove unused Any
tinyinfer/cache/tinyinfer_cache.py: remove unused importlib
tinyinfer/runtime/generation.py: remove unused BlockTable and ContiguousKVCache imports
tinyinfer/scheduler/queue.py: remove unused Any and wrap the finished comprehension
```

Use safe `ruff check --fix` only after inspecting its diff; do not use unsafe fixes.

- [ ] **Step 0.4: Repair the three Mypy findings.**

In `timer.py`, import `cast` and return:

```python
return cast(CudaEvent, self._cuda.Event(enable_timing=True))
```

Ensure `verify_gpu_paths.main() -> int` returns 0/1 and its module exit uses `SystemExit`.

- [ ] **Step 0.5: Format and run the complete pre-feature gate.**

```bash
.venv/bin/python -m ruff format .
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
uv lock --check
git diff --check
```

Expected: zero test failures, zero Ruff/Mypy errors, no lock change.

- [ ] **Step 0.6: Commit.**

```bash
git add scripts/verify_g1.py scripts/verify_gpu_paths.py \
  tests/integration/test_batch_engine.py tests/unit/test_allocation.py \
  tests/unit/test_contiguous_cache.py tinyinfer/cache/adapter.py \
  tinyinfer/cache/block.py tinyinfer/cache/tinyinfer_cache.py \
  tinyinfer/runtime/generation.py tinyinfer/scheduler/queue.py \
  tinyinfer/profiling/timer.py
git commit -m "chore: restore phase 2 quality baseline"
git status --short
```

#### Outputs

- One quality-only commit and fresh green baseline output.

#### Acceptance Criteria

- [ ] Focused allocation tests report `2 passed`.
- [ ] Full unit suite has zero failures.
- [ ] Ruff check/format and Mypy report zero errors.
- [ ] `uv.lock` is unchanged and lock check exits 0.
- [ ] Commit contains no true-batching implementation.
- [ ] Worktree is clean.

### Task 1: Correct Phase 2 Status Documentation

**Files:** create `tests/unit/test_phase2_docs.py` and `docs/PHASE2_STATUS.md`; modify `README.md` and `docs/BEGINNER_GUIDE.md`.

**Interfaces:** produces `docs/PHASE2_STATUS.md` as the current capability summary; historical G1 evidence remains immutable.

#### Inputs

- Task 0 clean commit and approved design Section 13.

#### Actions

- [ ] **Step 1.1: Write documentation RED tests.**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_readme_does_not_claim_all_phase2_is_absent() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    stale = "Runtime-owned KV cache, continuous batching, paged KV cache"
    assert stale not in text


def test_phase2_status_preserves_gate_boundary() -> None:
    text = (ROOT / "docs" / "PHASE2_STATUS.md").read_text(encoding="utf-8")
    assert "true batched decode: not verified" in text
    assert "block-native paged attention: not implemented" in text
    assert "G2: not passed" in text
```

- [ ] **Step 1.2: Run RED.**

```bash
.venv/bin/python -m pytest tests/unit/test_phase2_docs.py -q
```

Expected: stale README assertion and missing status file fail.

- [ ] **Step 1.3: Correct the documents.**

README and `PHASE2_STATUS.md` must state:

```text
contiguous KV cache: implemented prototype
paged KV cache: correctness prototype
FCFS scheduler: iteration-interleaving prototype
true batched decode: not verified
block-native paged attention: not implemented
G2: not passed
```

The status file links the design/plan and distinguishes G2-BD, G2-P, and full G2. Add the same boundary to the relevant beginner-guide section. Do not edit the G1 report, G1 JSONL, or `PROJECT_PLAN.md`.

- [ ] **Step 1.4: Run GREEN and protected-file checks.**

```bash
.venv/bin/python -m pytest tests/unit/test_phase2_docs.py -q
.venv/bin/python -m ruff check tests/unit/test_phase2_docs.py
.venv/bin/python -m ruff format --check tests/unit/test_phase2_docs.py
git diff --check
git diff --exit-code HEAD -- docs/G1_REFERENCE_RUNTIME_REPORT.md \
  docs/results/g1_reference.jsonl docs/PROJECT_PLAN.md
```

- [ ] **Step 1.5: Commit.**

```bash
git add README.md docs/BEGINNER_GUIDE.md docs/PHASE2_STATUS.md \
  tests/unit/test_phase2_docs.py
git commit -m "docs: correct phase 2 capability status"
```

#### Outputs

- Truthful pre-implementation Phase 2 status with an automated guard.

#### Acceptance Criteria

- [ ] Docs test exits 0.
- [ ] True batch and G2 remain explicitly unverified/not passed.
- [ ] Protected historical artifacts are unchanged.
- [ ] Worktree is clean.

### Task 2: Make Allocation and Release Atomic

**Files:** modify `tinyinfer/cache/block.py`, `tinyinfer/cache/paged.py`, and `tests/unit/test_allocation.py`.

**Interfaces:**

```python
BlockTable.clear(layer_idx: int, logical_block: int) -> None
BlockTable.clear_all() -> None
BlockAllocator.free(block_idx: int) -> None  # rejects unknown/double free
PagedKVCache.allocate_blocks(table: BlockTable, num_blocks: int) -> list[int]
PagedKVCache.free_blocks(table: BlockTable, num_blocks: int) -> None
```

#### Inputs

- Task 1 clean commit and current allocator APIs.

#### Actions

- [ ] **Step 2.1: Write RED tests.**

Add tests that: double-free raises `ValueError("not allocated")`; allocating 3 blocks from a 2-block pool restores free count 2 and leaves table entries `-1`; successful free clears every layer mapping; out-of-range free fails.

- [ ] **Step 2.2: Run RED.**

```bash
.venv/bin/python -m pytest tests/unit/test_allocation.py \
  -k "double_free or rolls_back or clears_every_layer or out_of_range" -q
```

- [ ] **Step 2.3: Implement ownership and table clearing.**

`BlockAllocator` maintains `_allocated: set[int]`. `allocate()` adds the index. `free()` validates range/membership, removes once, then appends to the free deque. Add `BlockTable.clear()` and `clear_all()`.

- [ ] **Step 2.4: Make pool operations atomic.**

Validate count/capacity/free space before allocation. On an unexpected mid-operation exception, return blocks acquired by that call and clear mappings written by that call before re-raising. `free_blocks()` frees each mapped physical block once and clears mappings; a repeated pool-level free is a no-op.

- [ ] **Step 2.5: Run GREEN twice and quality checks.**

```bash
.venv/bin/python -m pytest tests/unit/test_allocation.py -q
.venv/bin/python -m pytest tests/unit/test_allocation.py -q
.venv/bin/python -m ruff check tinyinfer/cache/block.py tinyinfer/cache/paged.py \
  tests/unit/test_allocation.py
.venv/bin/python -m ruff format --check tinyinfer/cache/block.py \
  tinyinfer/cache/paged.py tests/unit/test_allocation.py
.venv/bin/python -m mypy
git diff --check
```

- [ ] **Step 2.6: Commit.**

```bash
git add tinyinfer/cache/block.py tinyinfer/cache/paged.py tests/unit/test_allocation.py
git commit -m "fix: make paged block lifecycle atomic"
```

#### Outputs

- Atomic allocation failure, double-free protection, and cleared mappings.

#### Acceptance Criteria

- [ ] Exhaustion restores exact original free count.
- [ ] No mapping survives failed allocation or successful release.
- [ ] Double/out-of-range free is rejected.
- [ ] Randomized tests show no duplicate active blocks or leak.
- [ ] Focused tests and quality checks exit 0.

### Task 3: Introduce Shared Per-request Paged State

**Files:** create `tinyinfer/cache/request_state.py` and `tests/unit/test_paged_request_state.py`; modify `paged.py`, `paged_tinyinfer_cache.py`, cache exports, and `runtime/generation.py`.

**Interfaces:**

```python
@dataclass(slots=True)
class PagedRequestState:
    request_id: str
    block_table: BlockTable
    seq_length: int
    blocks_allocated: int
    released: bool = False

    def require_active(self) -> None: ...
    def commit_tokens(self, count: int) -> None: ...
    def mark_released(self) -> None: ...


PagedTinyinferCache(paged: PagedKVCache, state: PagedRequestState)
PagedKVCache.release_request(state: PagedRequestState) -> None
```

#### Inputs

- Task 2 atomic lifecycle and existing single-request paged generation.

#### Actions

- [ ] **Step 3.1: Write RED state/protocol tests.**

Create tests for blank ID, negative length/block count, bool-as-int rejection, active positive commit, idempotent release, released access rejection, and:

```python
state = make_state(seq_length=7)
cache = PagedTinyinferCache(make_paged(), state)
assert cache.get_seq_length() == 7
assert cache.get_query_offset() == 7
assert cache.get_mask_sizes(1, 0) == (8, 0)
```

- [ ] **Step 3.2: Run RED.**

```bash
.venv/bin/python -m pytest tests/unit/test_paged_request_state.py -q
```

Expected: module missing and old query offset equals 0.

- [ ] **Step 3.3: Implement the state contract.**

`request_id` is nonblank; `seq_length` and `blocks_allocated` are nonnegative real ints; commit count is a positive real int. Released state methods raise `RuntimeError` containing request ID. `mark_released()` is idempotent.

- [ ] **Step 3.4: Refactor the single adapter.**

Replace private table/length ownership with `self._state`. Expose `state`, `block_table`, and `length`. Keep prefill/decode commit after final layer. Make `get_query_offset()` return committed length.

Update `generate_with_paged_cache()` to create one state:

```python
state = PagedRequestState(
    request_id=request.request_id,
    block_table=block_table,
    seq_length=0,
    blocks_allocated=total_blocks,
)
cache = PagedTinyinferCache(paged, state)
```

Use `try/finally` and `paged.release_request(state)` so the single path also cleans up.

- [ ] **Step 3.5: Run GREEN and compatibility checks.**

```bash
.venv/bin/python -m pytest tests/unit/test_paged_request_state.py \
  tests/unit/test_generation.py -q
.venv/bin/python -m ruff check tinyinfer/cache tinyinfer/runtime/generation.py \
  tests/unit/test_paged_request_state.py
.venv/bin/python -m ruff format --check tinyinfer/cache \
  tinyinfer/runtime/generation.py tests/unit/test_paged_request_state.py
.venv/bin/python -m mypy
```

- [ ] **Step 3.6: Commit.**

```bash
git add tinyinfer/cache/request_state.py tinyinfer/cache/paged.py \
  tinyinfer/cache/paged_tinyinfer_cache.py tinyinfer/cache/__init__.py \
  tinyinfer/runtime/generation.py tests/unit/test_paged_request_state.py
git commit -m "refactor: share per-request paged cache state"
```

#### Outputs

- One source of truth for request length, table, allocation count, and release state.

#### Acceptance Criteria

- [ ] Scheduler and adapters share the same state instance.
- [ ] Length advances only after final model layer.
- [ ] Released state cannot be used or committed.
- [ ] Single Cache offset/mask contract matches Transformers 5.14.1.
- [ ] Existing single paged generation remains compatible.

### Task 4: Add Batched Paged KV Write and Left-padded Gather

**Files:** modify `tinyinfer/cache/paged.py`; create `tests/unit/test_paged_batch_io.py`.

**Interfaces:**

```python
def write_decode_batch(
    self,
    key: Any,
    value: Any,
    layer_idx: int,
    block_tables: Sequence[BlockTable],
    positions: Sequence[int],
) -> None: ...

def gather_batch(
    self,
    layer_idx: int,
    block_tables: Sequence[BlockTable],
    lengths: Sequence[int],
) -> tuple[Any, Any]: ...
```

`gather_batch()` returns zero-left-padded `[B, Hkv, max(lengths), D]`.

#### Inputs

- Task 3 state contract and real CPU Torch.

#### Actions

- [ ] **Step 4.1: Write real-tensor RED tests.**

Use CPU `torch.bfloat16`, three tables, lengths `(2, 3, 5)`, two KV heads, head dim four, block size two, and distinct values per request/layer/token. Assert:

```python
assert key_out.shape == (3, 2, 5, 4)
torch.testing.assert_close(key_out[0, :, -2:, :], expected_a)
torch.testing.assert_close(key_out[1, :, -3:, :], expected_b)
torch.testing.assert_close(key_out[2, :, -5:, :], expected_c)
assert torch.count_nonzero(key_out[0, :, :-2, :]) == 0
assert torch.count_nonzero(key_out[1, :, :-3, :]) == 0
```

Also test B=1, isolation, empty batch, key/value mismatch, query length not 1, wrong heads/head dim, table/count mismatch, negative position, unmapped block, layer range, dtype, and device where supported.

For every validation case, clone the physical buffer before the call and assert it is byte-for-byte unchanged after the exception. This is the evidence that validation finishes before the first row mutates storage.

- [ ] **Step 4.2: Run RED.**

```bash
.venv/bin/python -m pytest tests/unit/test_paged_batch_io.py -q
```

Expected: methods absent.

- [ ] **Step 4.3: Validate all rows before mutation.**

Add private validation helpers. Errors name batch row, layer, position, or actual shape. No `.copy_()` occurs until the entire batch validates.

- [ ] **Step 4.4: Implement batched writes.**

For row/position compute logical block and offset, resolve that row's table, and copy `key[row, :, 0, :]`/`value[row, :, 0, :]`. A Python row loop is allowed.

- [ ] **Step 4.5: Implement left-padded gather.**

Allocate zeros using pool dtype/device. Gather each request in logical token order and copy into `destination_start = max_length - length`; padding remains zero.

- [ ] **Step 4.6: Run GREEN twice and commit.**

```bash
.venv/bin/python -m pytest tests/unit/test_paged_batch_io.py -q
.venv/bin/python -m pytest tests/unit/test_paged_batch_io.py -q
.venv/bin/python -m ruff check tinyinfer/cache/paged.py \
  tests/unit/test_paged_batch_io.py
.venv/bin/python -m ruff format --check tinyinfer/cache/paged.py \
  tests/unit/test_paged_batch_io.py
.venv/bin/python -m mypy
git add tinyinfer/cache/paged.py tests/unit/test_paged_batch_io.py
git commit -m "feat: add batched paged KV gather"
```

#### Outputs

- Content-verified batch I/O over independent tables and lengths.

#### Acceptance Criteria

- [ ] B=1/B=3 shapes and values match exactly.
- [ ] No cross-request contamination.
- [ ] Padding is zero and valid tokens are right-aligned.
- [ ] Every listed invalid input fails before mutation.
- [ ] Tests use real tensors, not the no-op fake tensor.

### Task 5: Implement the Ephemeral Batched HF Cache Adapter

**Files:** create `tinyinfer/cache/batched_paged_tinyinfer_cache.py` and `tests/unit/test_batched_paged_cache.py`; modify cache exports.

**Interfaces:**

```python
class BatchedPagedTinyinferCache:
    def __init__(self, paged: PagedKVCache,
                 states: Sequence[PagedRequestState]) -> None: ...
    def update(self, key_states: Any, value_states: Any, layer_idx: int,
               cache_kwargs: Any = None) -> tuple[Any, Any]: ...
    def get_seq_length(self) -> int: ...
    def get_query_offset(self, layer_idx: int = 0) -> int: ...
    def get_mask_sizes(self, q_length: int,
                       layer_idx: int = 0) -> tuple[int, int]: ...
    @property
    def request_ids(self) -> tuple[str, ...]: ...
    @property
    def old_lengths(self) -> tuple[int, ...]: ...
```

#### Inputs

- Task 4 batch I/O and three active states of different lengths.

#### Actions

- [ ] **Step 5.1: Write protocol/layer RED tests.**

For lengths `(2, 3, 5)` assert request order, old lengths, seq/query offset 5, and mask sizes `(6, 0)`. Prove layer 0 leaves committed lengths unchanged and final layer changes them to `(3, 4, 6)`. Test output `[3, Hkv, 6, D]`, empty states, released/duplicate IDs, repeated/out-of-order layers, and query length not 1.

- [ ] **Step 5.2: Run RED.**

```bash
.venv/bin/python -m pytest tests/unit/test_batched_paged_cache.py -q
```

- [ ] **Step 5.3: Freeze iteration membership.**

Convert states to a tuple; reject empty, duplicate IDs, released states, and geometry mismatch. Capture old lengths once.

- [ ] **Step 5.4: Implement layer transaction.**

Require layer order `0..num_layers-1`. Each update writes at old positions and gathers lengths `old+1`. Commit one token to all states only after final-layer gather succeeds. Maintain HF-compatible `key_cache`/`value_cache` without extra tensor copies.

- [ ] **Step 5.5: Run GREEN and commit.**

```bash
.venv/bin/python -m pytest tests/unit/test_batched_paged_cache.py \
  tests/unit/test_paged_batch_io.py tests/unit/test_paged_request_state.py -q
.venv/bin/python -m ruff check tinyinfer/cache \
  tests/unit/test_batched_paged_cache.py
.venv/bin/python -m ruff format --check tinyinfer/cache \
  tests/unit/test_batched_paged_cache.py
.venv/bin/python -m mypy
git add tinyinfer/cache/batched_paged_tinyinfer_cache.py \
  tinyinfer/cache/__init__.py tests/unit/test_batched_paged_cache.py
git commit -m "feat: add heterogeneous batched cache adapter"
```

#### Outputs

- One HF-compatible cache view per decode iteration.

#### Acceptance Criteria

- [ ] Membership/order is immutable.
- [ ] Different lengths produce one right-aligned dense KV result.
- [ ] No committed length changes before final-layer success.
- [ ] Duplicate/out-of-order update is rejected.
- [ ] Focused tests and quality checks exit 0.

### Task 6: Replace Serial Decode Calls with One Batched Model Forward

**Files:** create `tinyinfer/scheduler/state.py` and `tests/unit/test_batch_engine.py`; modify engine, scheduler exports, and generation token selection.

**Interfaces:**

```python
@dataclass(slots=True)
class EngineRequestState:
    request: GenerationRequest
    cache: PagedRequestState
    prompt_token_ids: tuple[int, ...]
    output_token_ids: list[int]


@dataclass(frozen=True, slots=True)
class IterationStats:
    iteration_id: int
    admitted_count: int
    decode_batch_size: int
    decode_forward_count: int
    running_count_after_step: int
    finished_count: int
    cancelled_count: int
    request_ids: tuple[str, ...]
    old_cache_lengths: tuple[int, ...]


BatchEngine.last_iteration -> IterationStats | None
BatchEngine.iteration_history -> tuple[IterationStats, ...]
```

`BatchEngine.step() -> int` remains compatible.

#### Inputs

- Task 5 batched adapter and current engine public API.

#### Actions

- [ ] **Step 6.1: Write a fake-model RED path test.**

The tokenizer/model double records call input IDs, masks, position IDs, and cache type and returns row-distinct logits. To exercise cache semantics instead of bypassing them, the fake model invokes `past_key_values.update()` once for each configured layer using synthetic KV shaped `[B, Hkv, query_length, D]`. After three different-length prefills, clear prefill calls and assert one decode call:

```python
assert len(model.decode_calls) == 1
call = model.decode_calls[0]
assert tuple(call.input_ids.shape) == (3, 1)
assert call.position_ids.tolist() == [[length_a], [length_b], [length_c]]
assert isinstance(call.past_key_values, BatchedPagedTinyinferCache)
```

Current expected RED is three calls of `(1, 1)`.

- [ ] **Step 6.2: Write batch token-selection RED tests.**

Define `_select_next_tokens(logits, expected_batch) -> tuple[int, ...]`; test one integer per row and rejection of non-3D/wrong-batch logits. Keep `_select_next_token()` for single paths.

- [ ] **Step 6.3: Run RED.**

```bash
.venv/bin/python -m pytest tests/unit/test_batch_engine.py -q
.venv/bin/python -m pytest tests/unit/test_generation.py -k select_next -q
```

- [ ] **Step 6.4: Replace `_StateEntry` with dataclasses.**

Serial prefill creates a local `PagedTinyinferCache` over the shared state, stores prompt IDs and first output token, then discards the adapter.

- [ ] **Step 6.5: Build exact batched tensors in stable FCFS order.**

```python
request_ids = tuple(state.request.request_id for state in active)
old_lengths = tuple(state.cache.seq_length for state in active)
input_ids = torch.tensor(
    [[state.output_token_ids[-1]] for state in active],
    device=model.device,
    dtype=torch.long,
)
position_ids = torch.tensor(
    [[length] for length in old_lengths],
    device=model.device,
    dtype=torch.long,
)
```

Build `attention_mask` shape `[B, max(old_lengths)+1]`; row i has exactly `old_lengths[i]+1` trailing ones and leading zeros.

- [ ] **Step 6.6: Make exactly one decode call.**

For nonempty active batch:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=BatchedPagedTinyinferCache(paged, cache_states),
    use_cache=True,
    return_dict=True,
)
```

Do not pass `cache_position`. Map selected logits rows back using the frozen order. Empty active batch makes no model call.

- [ ] **Step 6.7: Record immutable path evidence.**

Append one `IterationStats` per `step()` with `decode_forward_count` exactly 0 or 1. Never retain logits/KV tensors in history.

- [ ] **Step 6.8: Run GREEN and commit.**

```bash
.venv/bin/python -m pytest tests/unit/test_batch_engine.py \
  tests/unit/test_generation.py -q
.venv/bin/python -m ruff check tinyinfer/scheduler tinyinfer/runtime/generation.py \
  tests/unit/test_batch_engine.py
.venv/bin/python -m ruff format --check tinyinfer/scheduler \
  tinyinfer/runtime/generation.py tests/unit/test_batch_engine.py
.venv/bin/python -m mypy
git add tinyinfer/scheduler/state.py tinyinfer/scheduler/engine.py \
  tinyinfer/scheduler/__init__.py tinyinfer/runtime/generation.py \
  tests/unit/test_batch_engine.py tests/unit/test_generation.py
git commit -m "feat: run one heterogeneous batched decode forward"
```

#### Outputs

- Actual one-forward batched decode, typed engine state, and path metrics.

#### Acceptance Criteria

- [ ] B=3 produces one call with input shape `(3,1)`.
- [ ] Position IDs equal each row's old cache length.
- [ ] Mask rows contain exactly `old_length+1` trailing valid slots.
- [ ] Row-distinct logits return to correct requests.
- [ ] B=0 makes no call; B=1 works.
- [ ] Every iteration records forward count 0/1 without tensors.

### Task 7: Verify Cancellation and Dynamic Continuous Admission

**Files:** create `tests/unit/test_continuous_admission.py`; modify engine, queue, and scheduler tests.

**Interfaces:**

```python
BatchEngine.cancel(request_id: str) -> None
RequestQueue.state_of(request_id: str) -> RequestState
```

All terminal/error paths use one engine release helper.

#### Inputs

- Task 6 one-forward engine and Task 2 release safety.

#### Actions

- [ ] **Step 7.1: Write dynamic-admission RED tests.**

Use `max_batch=2`, A with `max_new_tokens=1`, longer B, and waiting C. Freeze the expected history: A completes after prefill without a decode row; while B continues, C enters the next available slot; every iteration has at most one decode forward.

- [ ] **Step 7.2: Write lifecycle RED tests.**

Cover waiting/running/repeated/unknown cancellation, prefill exception, and batched decode exception. After every terminal path, initial and final free counts match, block tables are clear, and no failed token/length is committed.

- [ ] **Step 7.3: Run RED.**

```bash
.venv/bin/python -m pytest tests/unit/test_continuous_admission.py -q
```

- [ ] **Step 7.4: Centralize release/cancel.**

Implement `_release_state(request_id)` using `paged.release_request()`, then remove engine state. `BatchEngine.cancel()` transitions queue and releases any allocated state. Add `RequestQueue.state_of()` instead of inspecting `_entries`.

- [ ] **Step 7.5: Finalize prefill completions before decode.**

After prefill, immediately finalize EOS/token-limit requests and release them. Never include them in the batched decode view and never exceed `max_batch`.

- [ ] **Step 7.6: Clean up failed forwards.**

Prefill failure cancels/releases that request and re-raises. Batched decode failure cancels/releases every request in the frozen active batch, records no generated token or length, and re-raises the original exception.

- [ ] **Step 7.7: Run GREEN and commit.**

```bash
.venv/bin/python -m pytest tests/unit/test_continuous_admission.py \
  tests/unit/test_scheduler.py tests/unit/test_allocation.py \
  tests/unit/test_batch_engine.py -q
.venv/bin/python -m ruff check tinyinfer/scheduler \
  tests/unit/test_continuous_admission.py tests/unit/test_scheduler.py
.venv/bin/python -m ruff format --check tinyinfer/scheduler \
  tests/unit/test_continuous_admission.py tests/unit/test_scheduler.py
.venv/bin/python -m mypy
git add tinyinfer/scheduler/engine.py tinyinfer/scheduler/queue.py \
  tests/unit/test_continuous_admission.py tests/unit/test_scheduler.py
git commit -m "feat: add continuous admission and cache cleanup"
```

#### Outputs

- Dynamic membership and one release path for completion/cancel/error.

#### Acceptance Criteria

- [ ] C enters after A frees a slot while B continues.
- [ ] No iteration exceeds max batch or one decode call.
- [ ] Cancellation and forward failures leak zero blocks.
- [ ] Failed decode commits no token/length.
- [ ] Queue and engine state remain consistent.

### Task 8: Prove Real Qwen Path Identity and Token Agreement on GPU

**Files:** rename `tests/integration/test_batch_engine.py` to `tests/integration/test_batch_decode.py` and strengthen it.

**Interfaces:** `test_batch_decode.py` becomes the canonical G2-BD GPU proof.

#### Inputs

- Task 7 clean commit, cached fixed model, CUDA/BF16 available.

#### Actions

- [ ] **Step 8.1: Rename and freeze three heterogeneous prompts.**

```bash
git mv tests/integration/test_batch_engine.py tests/integration/test_batch_decode.py
```

Assert three tokenized prompt lengths are distinct. Use deterministic greedy generation with `max_new_tokens=8` unless EOS ends earlier.

- [ ] **Step 8.2: Add a real model forward pre-hook.**

Use `with_kwargs=True`; record only input shape, position IDs, and cache class. Classify decode calls by `BatchedPagedTinyinferCache` and always remove the hook in `finally`.

- [ ] **Step 8.3: Assert alignment and path identity separately.**

```python
assert any(record.input_shape == (3, 1) for record in batched_records)
assert any(
    stat.decode_batch_size == 3 and stat.decode_forward_count == 1
    for stat in engine.iteration_history
)
assert any(
    len(set(stat.old_cache_lengths)) == 3
    for stat in engine.iteration_history
    if stat.decode_batch_size == 3
)
```

Each output tuple must equal its own `reference_generate()` result.

- [ ] **Step 8.4: Strengthen pool/memory stability.**

Across three runs, initial/final free counts match, no engine request state remains, hooks are removed, and post-warm-up peak allocated memory does not exceed the first measured peak by more than 8%.

- [ ] **Step 8.5: Record RED path evidence safely.**

Use a disposable controller-created review worktree at the parent of Task 6 or inject a serial engine test double. Record three `(1,1)` calls. Do not rewrite feature-branch history.

- [ ] **Step 8.6: Run current-head GREEN and commit.**

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/integration/test_batch_decode.py -m "gpu and model" -vv -s
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/integration -m "gpu and model" -q
.venv/bin/python -m ruff check tests/integration/test_batch_decode.py
.venv/bin/python -m ruff format --check tests/integration/test_batch_decode.py
.venv/bin/python -m mypy
git diff --check
git add tests/integration/test_batch_decode.py
git commit -m "test: prove real heterogeneous batch decode on GPU"
```

#### Outputs

- Direct real-model `[3,1]` proof, exact token agreement, and repeated lifecycle evidence.

#### Acceptance Criteria

- [ ] Three prompt lengths differ.
- [ ] Hook observes real Qwen `[3,1]` with batched cache.
- [ ] Matching iteration has one decode forward and three different old lengths.
- [ ] All three output tuples equal HF references.
- [ ] Pool fully recovers after every run.
- [ ] Memory stays within 8% tolerance.

### Task 9: Add the Verifier and Close G2-BD Only

**Files:** create `scripts/verify_g2_bd.py`, `tests/unit/test_verify_g2_bd.py`, and `docs/G2_BD_REPORT.md`; update current status docs/tests.

**Interfaces:** verifier order is fixed:

```text
lock-check
pytest-unit
pytest-gpu-batch-decode
pytest-gpu-integration
ruff-check
ruff-format
mypy
git-diff-check
```

It fails fast, prints command/exit code, installs/downloads nothing, and mutates no Git state.

#### Inputs

- Task 8 clean commit and reviewed Task 0–8 history.

#### Actions

- [ ] **Step 9.1: Write verifier RED tests.**

Use an injected runner to assert exact order, fail-fast, exit propagation, timeouts, offline GPU environment, and final line `G2-BD VERIFICATION: PASS`. Assert source contains no `pip install`, `uv sync`, `git add`, `git commit`, hard-coded `/mnt/d/`, or download command.

- [ ] **Step 9.2: Run RED.**

```bash
.venv/bin/python -m pytest tests/unit/test_verify_g2_bd.py -q
```

- [ ] **Step 9.3: Implement the typed verifier.**

Use `Path(__file__).resolve().parents[1]`, `sys.executable`, and explicit timeouts: 120 seconds for lock/static/diff, 900 for unit/focused GPU, 1800 for all GPU integration. Timeout output includes the step name.

- [ ] **Step 9.4: Run GREEN and commit verifier code.**

```bash
.venv/bin/python -m pytest tests/unit/test_verify_g2_bd.py -q
.venv/bin/python -m ruff check scripts/verify_g2_bd.py \
  tests/unit/test_verify_g2_bd.py
.venv/bin/python -m ruff format --check scripts/verify_g2_bd.py \
  tests/unit/test_verify_g2_bd.py
.venv/bin/python -m mypy
git add scripts/verify_g2_bd.py tests/unit/test_verify_g2_bd.py
git commit -m "test: add G2-BD verification gate"
git status --short
```

Record this clean SHA as the tested implementation revision.

- [ ] **Step 9.5: Run the complete verifier.**

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_g2_bd.py
```

Expected exit 0 and final PASS line. Record every actual exit code, test count, prompt length, `[3,1]` hook record, token agreement, pool count, and memory value.

- [ ] **Step 9.6: Write evidence/status from actual output.**

`docs/G2_BD_REPORT.md` records tested implementation SHA, environment/model revision, command table, actual counts, heterogeneous lengths, model hook shape, forward count, token results, pool/memory results, dense-gather limitation, and exactly:

```text
G2-BD: verified locally
G2-P: not started
full G2: not passed
```

Update README, beginner guide, `PHASE2_STATUS.md`, and docs test to the same boundary. Never invent the report commit's own SHA; after commit, report the evidence commit in the handoff rather than self-reference.

- [ ] **Step 9.7: Verify protected files and commit evidence.**

```bash
.venv/bin/python -m pytest tests/unit/test_phase2_docs.py -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
uv lock --check
git diff --check
git diff --exit-code HEAD -- docs/G1_REFERENCE_RUNTIME_REPORT.md \
  docs/results/g1_reference.jsonl docs/PROJECT_PLAN.md uv.lock
git add README.md docs/BEGINNER_GUIDE.md docs/PHASE2_STATUS.md \
  docs/G2_BD_REPORT.md tests/unit/test_phase2_docs.py
git commit -m "docs: record verified G2-BD evidence"
```

- [ ] **Step 9.8: Verify current evidence HEAD.**

```bash
.venv/bin/python -m pytest tests/unit -q
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/integration/test_batch_decode.py -m "gpu and model" -vv -s
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
uv lock --check
git diff --check
git status --short
```

- [ ] **Step 9.9: Hand off for controller push/CI.**

Report final SHA. Do not push/merge/tag. Until GitHub Actions passes on that SHA, status is `local candidate`, not verified.

#### Outputs

- Reusable verifier, auditable report, truthful status, and clean review branch.

#### Acceptance Criteria

- [ ] Verifier tests prove order/fail-fast/timeouts/no mutation.
- [ ] Full verifier exits 0 on tested implementation SHA.
- [ ] Current evidence SHA passes full CPU quality and focused GPU test.
- [ ] Report uses actual values.
- [ ] Status keeps G2-P not started and full G2 not passed.
- [ ] Protected artifacts/lock are unchanged.
- [ ] Worktree is clean and no tag exists.

## 8. Gate / Exit Criteria

The controller accepts G2-BD only when all evidence applies to the final branch history:

1. `git merge-base --is-ancestor 1c1333a HEAD` exits 0.
2. Lock check exits 0 and `uv.lock` is unchanged.
3. Full unit suite has zero failures.
4. Focused real-Qwen batch test exits 0.
5. A hook record is exactly `[3,1]` for three different old cache lengths.
6. The same iteration records one decode forward.
7. Three outputs have 100% agreement with their own references.
8. Randomized allocation/dynamic admission show no duplicate blocks or leak.
9. Pool count recovers after completion, cancellation, errors, and repeated GPU runs.
10. Ruff check/format, Mypy, diff check, and remote CI all exit 0.
11. `docs/G2_BD_REPORT.md` names the tested implementation SHA.
12. Status retains `G2-P: not started` and `full G2: not passed`.
13. Worktree is clean and no G2 tag exists.

Without green remote CI, status is `local candidate`.

## 9. Definition of Done

- **Functionality:** each nonempty decode iteration executes at most one model forward for all eligible requests.
- **Correctness:** heterogeneous GPU outputs match HF token-for-token with direct shape/count evidence.
- **Ownership:** allocation, release, cancellation, error cleanup, and per-request lengths are verified.
- **Continuous behavior:** waiting C enters after A finishes while B remains running.
- **Quality:** tests, Ruff, format, Mypy, lock/diff checks, and remote CI are green.
- **Documentation:** README, guide, status, and report agree.
- **Safety:** no secrets, weights, caches, logs, or machine paths are committed.
- **Reproducibility:** versions, revisions, commands, counts, lengths, shapes, and memory/pool evidence are recorded.
- **Version control:** one reviewed commit per task; clean branch; no unapproved push/merge/tag.
- **Boundary:** G2-P/full G2 remain incomplete; dense gather is not block-native attention.

## 10. Verification Commands

Run from the WSL worktree:

```bash
.venv/bin/python - <<'PY'
import torch
import transformers
print(torch.__version__)
print(transformers.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
PY

uv lock --check
.venv/bin/python -m pytest tests/unit -q
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/integration/test_batch_decode.py -m "gpu and model" -vv -s
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/integration -m "gpu and model" -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
git diff --check
HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_g2_bd.py
git diff --exit-code 1c1333a -- docs/G1_REFERENCE_RUNTIME_REPORT.md \
  docs/results/g1_reference.jsonl docs/PROJECT_PLAN.md uv.lock
git status --porcelain
```

Expected: exact environment identity, all exits 0, verifier final PASS line, no protected diff, empty status.

## 11. Stop Conditions

Stop and report evidence when:

- branch/path/HEAD differs from controller handoff or worktree is unexpectedly dirty;
- HEAD does not descend from `1c1333a`;
- lockfile/version/CUDA/GPU/BF16/model revision differs;
- a RED test already passes and does not prove missing behavior;
- the same deterministic root cause fails three times;
- left-padding contradicts actual Transformers behavior;
- cross-request KV contamination or token mismatch occurs;
- a claimed batch iteration makes more than one decode call;
- any terminal path changes final free count;
- memory exceeds capacity/tolerance;
- implementation requires custom attention, a kernel, or dependency change;
- protected evidence/project plan changes;
- push/merge/tag/reset/worktree mutation is required without controller approval.

## 12. Failure Handling

1. Capture command, exit code, first assertion, traceback, HEAD, and status.
2. GPU failures also record versions, GPU, prompt lengths, request IDs, old lengths, input/mask shapes, and stats.
3. Token mismatch records token IDs and first mismatch only; do not dump weights/KV.
4. Allocation failure records initial/final free count, mappings, and released flags.
5. Retry only a transient CUDA initialization failure; deterministic failures require a hypothesis/change.
6. Do not destructive-reset, discard, stash, or overwrite other changes.
7. Return failed review to its implementer with evidence.
8. If dense gather is invalid, return to brainstorming; do not switch architecture autonomously.

## 13. Git / Worktree Rules

- **Branch:** `agent/true-batched-decode`.
- **Worktree:** controller-created native WSL path stated above.
- **Commits:** one after each Task 0–9.
- **Style:** `chore:`, `docs:`, `fix:`, `refactor:`, `feat:`, or `test:`.
- **Review:** controller reruns focused tests and reviews scope between tasks.
- **Merge/push:** later branch-finishing workflow; not authorized here.
- **Tags/force push/destructive reset:** prohibited.
- **Worktree create/remove:** controller only.

## 14. Agent Handoff Contract

### Start Report

Read the approved design, this whole plan, Global Constraints, and complete task files. Report:

```text
task number/name
branch and HEAD SHA
origin/main SHA
design ancestor check
absolute worktree path and dirty files
runtime identity when relevant
previous task SHA
confirmation that G2-P/full G2 remain incomplete
```

### End Report

Report:

```text
every file/change type
RED command/exit/reason
GREEN commands/exits/pass-fail-skip counts
quality/lock/diff results required by task
each Acceptance Criterion mapped to fresh evidence
actual shapes/counts/lengths/free blocks when relevant
risks/concerns
commit SHA/subject and final status
confirmation of no push/merge/tag
```

The controller independently checks the diff and focused test before the next task. Task 8/9 require fresh GPU evidence; screenshots or old agent reports are insufficient.

## 15. Status Vocabulary

| Status | Definition |
|---|---|
| `not started` | Preconditions/task inputs are not freshly confirmed. |
| `in progress` | Actions started but Acceptance Criteria are incomplete. |
| `blocked` | A listed Stop Condition prevents progress with evidence. |
| `local candidate` | Local G2-BD passes; remote CI/controller review is incomplete. |
| `verified` | Every G2-BD criterion and remote CI pass on declared history and controller accepts evidence. |

`verified` here means G2-BD only, never G2-P or full G2.

## 16. Plan Boundary and Follow-up

After acceptance:

1. use `superpowers:finishing-a-development-branch` for integration;
2. write a separate G2-P plan for serial versus B=1/2/4/8 evidence;
3. profile dense gather before selecting a Triton kernel;
4. mark full G2 only after G2-BD, G2-P, and original project-plan conditions pass.
