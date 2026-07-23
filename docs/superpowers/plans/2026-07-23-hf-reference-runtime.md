# TinyInfer HF Reference Runtime Implementation Plan

> Required execution skill: use `superpowers:executing-plans` for sequential execution or
> `superpowers:subagent-driven-development` when the owner explicitly delegates independent tasks.
> Check each box only after the stated command and expected result have been observed.

**Goal:** Build a fixed-revision Qwen2.5-0.5B single-request greedy runtime with explicit prefill and
one-token decode, exact Hugging Face token alignment, and a reproducible 20-record G1 benchmark.

**Approved boundary:** This plan delivers R4-R7 and G1 only. It does not implement a custom KV cache,
continuous batching, paged KV, an API server, sampling, quantization, multi-GPU support, or a new Triton
kernel.

**Execution environment:** WSL2 Ubuntu-22.04, Python 3.11.15, PyTorch 2.13.0+cu130, Transformers 5.14.1,
Triton 3.7.0, NVIDIA RTX 4080 SUPER.

**Fixed model:** `Qwen/Qwen2.5-0.5B` at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, BF16, CUDA, SDPA,
`trust_remote_code=False`.

## Shared implementation contracts

The following names and signatures are authoritative across all tasks:

```python
StopReason = Literal["eos", "length"]

@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int

@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    text: str
    stop_reason: StopReason

@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    model_id: str
    model_revision: str
    eos_token_id: int
    pad_token_id: int

def load_reference_model(config: EngineConfig) -> LoadedModel:
    raise NotImplementedError
def prefill(model: Any, input_ids: Any, attention_mask: Any) -> PrefillOutput:
    raise NotImplementedError
def decode_one(model: Any, last_token_id: int, attention_mask: Any, cache: Any) -> DecodeOutput:
    raise NotImplementedError
def generate_greedy(
    loaded: LoadedModel,
    request: GenerationRequest,
    observer: GenerationObserver | None = None,
) -> GenerationResult:
    raise NotImplementedError
def reference_generate(loaded: LoadedModel, request: GenerationRequest) -> tuple[int, ...]:
    raise NotImplementedError
```

The `NotImplementedError` bodies document interfaces only. Each owning task supplies the production function's
executable body and full strict-Mypy annotations.

`GenerationObserver` has these callbacks in this order: `request_started`, `prefill_started`,
`prefill_finished`, `first_token_ready`, `decode_started`, `decode_finished(output_tokens)`, and
`request_finished`. The immediate-EOS path still emits all seven callbacks; decode duration is zero.

## Task 0: Create the native WSL worktree and prove the baseline

**Files:** no tracked changes.

- [ ] From `/home/administrator/projects/TinyInfer`, run:

```bash
git fetch origin main --tags
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor v0.1.0-g0.5 HEAD
git worktree add .worktrees/hf-reference-runtime -b agent/hf-reference-runtime
cd .worktrees/hf-reference-runtime
uv sync --locked --extra dev --extra inference --extra kernels
python scripts/verify_g0_5.py
```

**Acceptance:** the source checkout is clean; `HEAD` equals `origin/main`; the ancestor check exits 0; the
new worktree is on `agent/hf-reference-runtime`; the verifier ends with `G0.5 VERIFICATION: PASS`.

**Stop condition:** do not change runtime code when any baseline assertion fails.

## Task 1: Add request and result contracts

**Create:** `tinyinfer/runtime/__init__.py`, `tinyinfer/runtime/request.py`,
`tinyinfer/runtime/result.py`, `tests/unit/test_request.py`, `tests/unit/test_result.py`.

- [ ] Write failing tests for blank `request_id`, blank `prompt`, and `max_new_tokens` values 0 and -1.
- [ ] Write passing-value tests proving dataclass immutability and tuple token storage.
- [ ] Implement `GenerationRequest` and `GenerationResult` exactly as specified above.
- [ ] Validate `GenerationResult`: nonblank request ID, nonempty prompt token IDs, nonempty output token IDs,
  and stop reason restricted to `eos` or `length`.
- [ ] Export only the public runtime contract names from `tinyinfer/runtime/__init__.py`.

Run RED before implementation and then run GREEN:

```bash
python -m pytest tests/unit/test_request.py tests/unit/test_result.py -v
python -m pytest -m "not gpu and not model and not slow"
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
git add tinyinfer/runtime tests/unit/test_request.py tests/unit/test_result.py
git commit -m "feat: add generation request and result contracts"
```

**Acceptance:** all commands exit 0 and the new tests execute, rather than skip.

## Task 2: Add the fixed-revision Hugging Face loader

**Create:** `tinyinfer/runtime/hf_loader.py`, `tests/unit/test_hf_loader.py`,
`tests/integration/test_hf_loader.py`. **Modify:** `tinyinfer/runtime/__init__.py`, `pyproject.toml`.

- [ ] Register strict pytest markers `gpu`, `model`, and `slow` in `pyproject.toml`.
- [ ] Unit-test the loader with injected fake `AutoTokenizer`, `AutoModelForCausalLM`, and torch modules.
- [ ] Assert exact loader arguments: fixed revision, `torch_dtype=torch.bfloat16`,
  `attn_implementation="sdpa"`, `trust_remote_code=False`, and no `device_map="auto"`.
- [ ] Move the loaded model explicitly to `cuda`, call `eval()`, preserve the revision in `LoadedModel`, and
  use EOS as PAD only when the tokenizer has no PAD ID.
- [ ] Keep torch and transformers imports inside the loader boundary so CPU-only CI can collect unit tests.
- [ ] Add a real-model integration test asserting CUDA placement, BF16 parameters, eval mode, and resolved
  EOS/PAD IDs.

```bash
python -m pytest tests/unit/test_hf_loader.py -v
python -m pytest tests/integration/test_hf_loader.py -m "gpu and model" -v
python -m pytest -m "not gpu and not model and not slow"
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
git add pyproject.toml tinyinfer/runtime tests/unit/test_hf_loader.py tests/integration/test_hf_loader.py
git commit -m "feat: add fixed revision hugging face loader"
```

**Acceptance:** fake tests verify every loading argument; the real test loads the fixed model once and passes.

## Task 3: Implement explicit prefill, decode, and greedy generation

**Create:** `tinyinfer/runtime/generation.py`, `tests/unit/test_generation.py`,
`tests/integration/test_hf_reference_runtime.py`. **Modify:** `tinyinfer/runtime/__init__.py`.

- [ ] Unit-test prefill using a fake model output. It must pass the complete prompt, `use_cache=True`, and
  return argmax of the last-position logits plus the public cache object.
- [ ] Unit-test decode using a public fake cache exposing `get_seq_length()`. It must pass shape `[1, 1]`,
  append exactly one attention-mask position, supply monotonic `cache_position`, and increase cache length by 1.
- [ ] Reject a cache whose length does not advance exactly once.
- [ ] Unit-test greedy generation for length stop, immediate EOS, EOS after decode, and `max_new_tokens=1`.
- [ ] Implement tokenization once, one prefill call, repeated one-token decode, and deterministic argmax only.
- [ ] Decode text only after token generation; retain EOS in `output_token_ids` while using
  `skip_special_tokens=True` for text.
- [ ] Add real-model cases with `max_new_tokens` 1, 2, 8, and 32.

```bash
python -m pytest tests/unit/test_generation.py -v
python -m pytest tests/integration/test_hf_reference_runtime.py -m "gpu and model" -v
python -m pytest -m "not gpu and not model and not slow"
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
git add tinyinfer/runtime tests/unit/test_generation.py tests/integration/test_hf_reference_runtime.py
git commit -m "feat: add explicit greedy prefill and decode loop"
```

**Acceptance:** fake tests prove cache/mask invariants and all four real-model lengths complete without OOM.

## Task 4: Freeze workloads and prove token alignment

**Create:** `benchmarks/prompts.json`, `tests/unit/test_prompts.py`.
**Modify:** `tinyinfer/runtime/generation.py`, `tinyinfer/runtime/__init__.py`,
`tests/integration/test_hf_reference_runtime.py`.

The JSON file has schema version 1 and exactly these ordered cases, all with `max_new_tokens: 32`:

| case_id | category | exact prompt |
|---|---|---|
| `short_en` | `short English` | `Explain KV cache in one concise paragraph.` |
| `short_zh` | `short Chinese` | `请用一段简洁的话解释大语言模型中的 KV Cache。` |
| `structured` | `structured output` | `Return exactly three numbered points explaining prefill, decode, and token latency.` |
| `long_context` | `long context` | Twenty-four newline-separated copies of `A reference runtime separates prompt prefill from one-token decode so that cache growth, latency, and correctness can be measured independently. Accurate measurements require fixed inputs, deterministic decoding, explicit synchronization, and recorded software revisions.` followed by ` Summarize the engineering lesson in two sentences.` |

- [ ] Unit-test exact case order, key set, nonblank prompts, 32-token limit, and twenty-four occurrences of the
  long-context paragraph. The real tokenizer must place this prompt in the 512-768 token interval; failure is a
  design stop, not permission to mutate the prompt during a benchmark.
- [ ] Implement `reference_generate` by calling `loaded.model.generate` with the same tokenized prompt,
  `do_sample=False`, `max_new_tokens=request.max_new_tokens`, fixed EOS/PAD IDs, and `use_cache=True`.
- [ ] Slice only the generated suffix; never compare prompt tokens with generated tokens.
- [ ] Add a parametrized integration test comparing the `output_token_ids` returned by `generate_greedy` to
  the tuple returned by `reference_generate` for all four cases and reporting the first mismatch index.

```bash
python -m pytest tests/unit/test_prompts.py -v
python -m pytest tests/integration/test_hf_reference_runtime.py -m "gpu and model" -v
python -m pytest -m "not gpu and not model and not slow"
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
git add benchmarks/prompts.json tinyinfer/runtime tests/unit/test_prompts.py tests/integration/test_hf_reference_runtime.py
git commit -m "test: prove reference token alignment"
```

**Acceptance:** four of four prompts have exact token-sequence equality, including sequence length.

## Task 5: Add observer-based Host/CUDA profiling

**Create:** `tinyinfer/profiling/__init__.py`, `tinyinfer/profiling/timer.py`,
`tests/unit/test_timer.py`. **Modify:** `tinyinfer/runtime/generation.py`,
`tests/unit/test_generation.py`.

`PhaseTimings` contains: `ttft_host_ms`, `prefill_host_ms`, `prefill_device_ms`, `decode_host_ms`,
`decode_device_ms`, `tpot_host_ms`, `tpot_device_ms`, `output_tokens_per_s`, `peak_allocated_bytes`, and
`peak_reserved_bytes`.

- [ ] Define `GenerationObserver` with the seven callbacks listed in Shared implementation contracts.
- [ ] Add a no-op observer and make `generate_greedy` emit the exact event order without duplicating the
  generation loop or allowing observer state to change tokens/cache.
- [ ] Implement `InferenceTimer` with injected monotonic clock and injected CUDA facade for CPU tests.
- [ ] Reset peak memory at request start; use CUDA events around prefill and decode; synchronize at first token
  and request completion; sample both allocated and reserved peak bytes.
- [ ] Compute TTFT from request start through first token, TPOT over tokens after the first, and throughput over
  all output tokens. For one output token, both TPOT values and decode durations are exactly `0.0`.
- [ ] Unit-test callback misuse: duplicate start, finish before completion, negative duration, and missing event
  must raise `RuntimeError` or `ValueError` with a specific phase name.

```bash
python -m pytest tests/unit/test_timer.py tests/unit/test_generation.py -v
python -m pytest -m "not gpu and not model and not slow"
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
git add tinyinfer/runtime/generation.py tinyinfer/profiling tests/unit/test_timer.py tests/unit/test_generation.py
git commit -m "feat: add phase and memory profiling"
```

**Acceptance:** fixed fake clock/event tests reproduce exact numeric values; immediate EOS has zero decode/TPOT.

## Task 6: Add agreement and schema-v1 records

**Create:** `tinyinfer/profiling/record.py`, `tests/unit/test_record.py`.
**Modify:** `tinyinfer/profiling/__init__.py`.

- [ ] Implement `compare_token_ids(actual, reference)` with denominator `max(len(actual), len(reference))`.
  Empty-versus-empty is invalid. A length mismatch sets the first mismatch to the shared-length boundary.
- [ ] Implement immutable `TokenComparison(agreement, first_mismatch_index)`.
- [ ] Implement immutable `BenchmarkRecord` and `to_dict()` with exactly the schema-v1 keys defined in
  section 9 of the design specification, including `output_token_ids`.
- [ ] Reject non-finite or negative timings, negative counters, non-40-character Git SHA, inconsistent output
  counts, and any canonical record whose agreement is not exactly 1.0.
- [ ] Test exact match, token mismatch, both length-mismatch directions, empty inputs, schema key equality,
  JSON-compatible stop reason, and every validation branch.

```bash
python -m pytest tests/unit/test_record.py -v
python -m pytest -m "not gpu and not model and not slow"
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
git add tinyinfer/profiling tests/unit/test_record.py
git commit -m "feat: add reference benchmark record schema"
```

**Acceptance:** serialized keys equal the specification exactly and no NaN/Infinity can enter JSONL.

## Task 7: Build the benchmark CLI and canonical evidence

**Create:** `benchmarks/__init__.py`, `benchmarks/bench_reference.py`,
`tests/unit/test_bench_reference.py`, `docs/results/g1_reference.jsonl`.

- [ ] Implement `PromptCase`, strict schema/case-order validation, and an atomic JSONL writer using a sibling
  `.tmp` file plus `os.replace`. Refuse existing output unless `--overwrite` is present.
- [ ] Implement `BenchmarkDependencies` as a frozen collection of loader, TinyInfer runner, HF reference
  runner, timer factory, record factory, environment identity, and Git-SHA provider callables.
- [ ] Implement `run_benchmark`: load once; compute one HF reference per case; perform three unrecorded warmups;
  record five repetitions per case; require exact agreement and one deterministic output sequence per case.
- [ ] Implement CLI options `--config`, `--prompts`, `--output`, `--warmup-runs`, `--repetitions`, and
  `--overwrite`. Defaults are the repository config/prompts, `docs/results/g1_reference.jsonl`, 3, and 5.
- [ ] On alignment failure, print case ID and first mismatch, exit 1, and leave canonical output untouched.
  Write no partially valid canonical file. Model-loading and OOM failures retain a nonzero traceback.
- [ ] Unit-test 3 warmups plus 5 measured calls, 20-record ordering, mismatch abort, nondeterminism abort,
  existing-output refusal, temporary-file cleanup, and atomic replacement.

```bash
python -m pytest tests/unit/test_bench_reference.py tests/unit/test_timer.py tests/unit/test_record.py -v
python -m pytest -m "not gpu and not model and not slow"
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
python benchmarks/bench_reference.py \
  --config configs/qwen2_5_0_5b.yaml \
  --prompts benchmarks/prompts.json \
  --output docs/results/g1_reference.jsonl \
  --warmup-runs 3 \
  --repetitions 5 \
  --overwrite
```

**Acceptance:** CLI ends with `G1 REFERENCE RUNTIME: PASS`; JSONL has exactly 20 lines, four case IDs,
five repetitions per case, agreement 1.0, deterministic token IDs per case, finite nonnegative timings, positive
TTFT/throughput, and positive peak memory values.

```bash
git add benchmarks tinyinfer docs/results/g1_reference.jsonl tests/unit/test_bench_reference.py
git commit -m "feat: add reproducible reference benchmark"
```

## Task 8: Complete the G1 report and release gate

**Create:** `docs/G1_REFERENCE_RUNTIME_REPORT.md`.
**Modify:** `README.md`, `docs/PHASE1_WORKFLOW.md`.

- [ ] Summarize environment identity, fixed model revision, workload definitions, correctness result, median
  and p95 TTFT, TPOT, throughput, and peak memory from the committed JSONL. Do not transcribe invented numbers.
- [ ] Explain prefill versus decode and clearly state that the cache is Transformers' public reference cache,
  not a TinyInfer custom-cache optimization.
- [ ] Link README commands to the config, prompts, benchmark CLI, canonical JSONL, and G1 report.
- [ ] Mark R4-R7/G1 complete in `docs/PHASE1_WORKFLOW.md` only after every gate below passes.

Run the final gate in a fresh shell from the WSL worktree:

```bash
git status --short --branch
uv sync --locked --extra dev --extra inference --extra kernels
python scripts/verify_g0_5.py
python -m pytest -m "not gpu and not model and not slow"
python -m pytest tests/integration -m "gpu and model" -v
python -m ruff check .
python -m ruff format --check .
python -m mypy
git diff --check
python benchmarks/bench_reference.py \
  --config configs/qwen2_5_0_5b.yaml \
  --prompts benchmarks/prompts.json \
  --output /tmp/tinyinfer-g1-evidence/reference.jsonl \
  --warmup-runs 3 \
  --repetitions 5 \
  --overwrite
python - <<'PY'
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

def read(path: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]

candidate = read("/tmp/tinyinfer-g1-evidence/reference.jsonl")
canonical = read("docs/results/g1_reference.jsonl")
assert len(candidate) == len(canonical) == 20
assert set(Counter(row["case_id"] for row in candidate).values()) == {5}
assert all(row["token_agreement"] == 1.0 for row in candidate)
assert all(math.isfinite(row[key]) and row[key] >= 0 for row in candidate for key in (
    "ttft_host_ms", "prefill_host_ms", "prefill_device_ms", "decode_host_ms",
    "decode_device_ms", "tpot_host_ms", "tpot_device_ms", "output_tokens_per_s",
))
def outputs(rows: list[dict[str, object]]) -> dict[str, set[tuple[int, ...]]]:
    grouped: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for row in rows:
        grouped[str(row["case_id"])].add(tuple(row["output_token_ids"]))
    return grouped
assert all(len(values) == 1 for values in outputs(candidate).values())
assert outputs(candidate) == outputs(canonical)
print("G1 VERIFICATION: PASS")
PY
```

```bash
git add README.md docs/PHASE1_WORKFLOW.md docs/G1_REFERENCE_RUNTIME_REPORT.md
git commit -m "docs: record verified G1 reference runtime"
git status --short --branch
```

**Definition of done:** the branch is clean; all CPU, GPU/model, lint, format, type, baseline, benchmark, and G1
verification commands pass; evidence is committed; scope exclusions remain absent; the report contains only
metrics derived from evidence.

## Execution handoff

Recommended execution is task-by-task in one WSL session with a review checkpoint after Tasks 2, 4, 7, and 8.
Each worker must read the design specification and `docs/R4_R7_G1_WORKFLOW.md`, preserve task order, show RED
before GREEN for code behavior, commit only the files listed for that task, and stop at the documented failure
boundary. No worker may silently widen scope or waive an acceptance check.
