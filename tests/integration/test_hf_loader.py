"""Real CUDA/model checks for the fixed Hugging Face loader."""

import pytest

from tinyinfer.config import EngineConfig
from tinyinfer.runtime.hf_loader import LoadedModel, load_reference_model

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = [pytest.mark.gpu, pytest.mark.model]


@pytest.fixture(scope="module")
def loaded_model() -> LoadedModel:
    return load_reference_model(EngineConfig())


def test_loaded_model_has_fixed_identity(loaded_model: LoadedModel) -> None:
    assert loaded_model.model_id == "Qwen/Qwen2.5-0.5B"
    assert loaded_model.model_revision == "060db6499f32faf8b98477b0a26969ef7d8b9987"


def test_loaded_model_is_cuda_bf16_eval_with_cache(loaded_model: LoadedModel) -> None:
    parameter = next(loaded_model.model.parameters())
    assert parameter.device.type == "cuda"
    assert parameter.dtype == torch.bfloat16
    assert loaded_model.model.training is False
    assert loaded_model.model.config.use_cache is True


def test_loaded_model_has_resolved_token_ids(loaded_model: LoadedModel) -> None:
    assert type(loaded_model.eos_token_id) is int
    assert type(loaded_model.pad_token_id) is int
    assert loaded_model.eos_token_id >= 0
    assert loaded_model.pad_token_id >= 0
