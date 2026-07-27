"""Fixed-revision Hugging Face loader with lazy optional dependencies."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from tinyinfer.config import EngineConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


class ModelLoadError(RuntimeError):
    """Raised when the fixed reference model cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class LoaderDependencies:
    """Optional libraries used by the loader, injectable in CPU tests."""

    torch: Any
    auto_tokenizer: Any
    auto_model: Any


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Resolved model, tokenizer, and generation token identities."""

    model: Any
    tokenizer: Any
    model_id: str
    model_revision: str
    eos_token_id: int
    pad_token_id: int


def _failure(stage: str, error: BaseException) -> ModelLoadError:
    return ModelLoadError(
        f"{stage} for {MODEL_ID} at revision {MODEL_REVISION} failed ({type(error).__name__})"
    )


def _load_dependencies() -> LoaderDependencies:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise _failure("importing optional dependencies", error) from error
    return LoaderDependencies(
        torch=torch,
        auto_tokenizer=transformers.AutoTokenizer,
        auto_model=transformers.AutoModelForCausalLM,
    )


def _validate_config(config: EngineConfig) -> None:
    if config.model_id != MODEL_ID:
        raise ModelLoadError(f"unsupported model_id: {config.model_id!r}")
    if config.dtype != "bfloat16":
        raise ModelLoadError(f"unsupported dtype: {config.dtype!r}")
    if not config.device.startswith("cuda"):
        raise ModelLoadError(f"unsupported device: {config.device!r}")


def _resolve_token_ids(tokenizer: Any) -> tuple[int, int]:
    eos_token_id = tokenizer.eos_token_id
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ModelLoadError("token IDs require a non-negative integer EOS ID")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ModelLoadError("token IDs require a non-negative integer PAD ID")
    return eos_token_id, pad_token_id


def load_reference_model(config: EngineConfig) -> LoadedModel:
    """Load the approved Qwen revision on an explicit CUDA device."""

    _validate_config(config)
    dependencies = _load_dependencies()
    try:
        tokenizer = dependencies.auto_tokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
        )
    except Exception as error:
        raise _failure("loading tokenizer", error) from error
    try:
        model = dependencies.auto_model.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=dependencies.torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
    except Exception as error:
        raise _failure("loading model", error) from error
    try:
        placed_model = model.to(config.device)
        if placed_model is not None:
            model = placed_model
        model.eval()
        model.config.use_cache = True
    except Exception as error:
        raise _failure(f"placing model on {config.device}", error) from error
    eos_token_id, pad_token_id = _resolve_token_ids(tokenizer)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
