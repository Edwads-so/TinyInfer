"""Native CUDA proof for the G2-P1 shared-pool benchmark path."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from benchmarks.bench_phase2 import assert_token_alignment, run_shared_paged_workload
from tinyinfer.config import EngineConfig
from tinyinfer.profiling.phase2 import Phase2Request, Phase2Workload
from tinyinfer.runtime.generation import reference_generate
from tinyinfer.runtime.hf_loader import load_reference_model
from tinyinfer.runtime.request import GenerationRequest

pytestmark = [pytest.mark.gpu, pytest.mark.model]


@pytest.mark.parametrize("mode", ["sequential", "static_batch", "continuous_batch"])
def test_phase2_modes_align_and_recover_pool(mode: str) -> None:
    loaded = load_reference_model(EngineConfig())
    workload = Phase2Workload(
        workload_id="gpu_smoke",
        family="fixed",
        requests=(
            Phase2Request("r0", "short", "Explain KV cache.", 4, 0),
            Phase2Request("r1", "short_zh", "解释 KV Cache。", 4, 0),
        ),
    )
    execution = run_shared_paged_workload(loaded, EngineConfig(), workload, mode=mode)
    references = {
        item.request_id: reference_generate(
            loaded, GenerationRequest(item.request_id, item.prompt, item.max_new_tokens)
        )
        for item in workload.requests
    }
    assert_token_alignment(execution.results, references.__getitem__)
    assert execution.paged.free_block_count == execution.paged.num_physical_blocks
