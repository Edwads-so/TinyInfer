from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tinyinfer.config import EngineConfig
from tinyinfer.runtime import hf_loader
from tinyinfer.runtime.hf_loader import LoadedModel, ModelLoadError


def fake_dependencies(
    *,
    eos_token_id: object = 151643,
    pad_token_id: object = 151643,
) -> tuple[hf_loader.LoaderDependencies, MagicMock, MagicMock]:
    tokenizer = MagicMock()
    tokenizer.eos_token_id = eos_token_id
    tokenizer.pad_token_id = pad_token_id
    model = MagicMock()
    model.to.return_value = model
    model.config = SimpleNamespace(use_cache=False)
    dependencies = hf_loader.LoaderDependencies(
        torch=SimpleNamespace(bfloat16=object()),
        auto_tokenizer=MagicMock(from_pretrained=MagicMock(return_value=tokenizer)),
        auto_model=MagicMock(from_pretrained=MagicMock(return_value=model)),
    )
    return dependencies, tokenizer, model


def install_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    dependencies: hf_loader.LoaderDependencies,
) -> None:
    monkeypatch.setattr(hf_loader, "_load_dependencies", lambda: dependencies)


def test_loader_uses_fixed_identity_and_safe_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies, _, _ = fake_dependencies()
    install_dependencies(monkeypatch, dependencies)

    hf_loader.load_reference_model(EngineConfig())

    dependencies.auto_tokenizer.from_pretrained.assert_called_once_with(
        "Qwen/Qwen2.5-0.5B",
        revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        trust_remote_code=False,
    )
    dependencies.auto_model.from_pretrained.assert_called_once_with(
        "Qwen/Qwen2.5-0.5B",
        revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        torch_dtype=dependencies.torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    assert "device_map" not in dependencies.auto_model.from_pretrained.call_args.kwargs


def test_loader_places_model_and_returns_resolved_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies, tokenizer, model = fake_dependencies()
    install_dependencies(monkeypatch, dependencies)

    loaded = hf_loader.load_reference_model(EngineConfig(device="cuda:0"))

    assert isinstance(loaded, LoadedModel)
    assert loaded.model is model
    assert loaded.tokenizer is tokenizer
    assert loaded.model_id == "Qwen/Qwen2.5-0.5B"
    assert loaded.model_revision == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert loaded.eos_token_id == 151643
    assert loaded.pad_token_id == 151643
    model.to.assert_called_once_with("cuda:0")
    model.eval.assert_called_once_with()
    assert model.config.use_cache is True


def test_loader_sets_missing_pad_id_to_eos(monkeypatch: pytest.MonkeyPatch) -> None:
    dependencies, tokenizer, _ = fake_dependencies(eos_token_id=42, pad_token_id=None)
    install_dependencies(monkeypatch, dependencies)

    loaded = hf_loader.load_reference_model(EngineConfig())

    assert tokenizer.pad_token_id == 42
    assert loaded.pad_token_id == 42


@pytest.mark.parametrize(
    "config",
    [
        EngineConfig(model_id="other/model"),
        EngineConfig(dtype="float16"),
        EngineConfig(device="cpu"),
    ],
)
def test_loader_rejects_unsupported_config_before_importing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    config: EngineConfig,
) -> None:
    dependency_loader = MagicMock()
    monkeypatch.setattr(hf_loader, "_load_dependencies", dependency_loader)

    with pytest.raises(ModelLoadError):
        hf_loader.load_reference_model(config)

    dependency_loader.assert_not_called()


@pytest.mark.parametrize(
    ("eos_token_id", "pad_token_id"),
    [(None, 1), (-1, 1), (1.5, 1), (1, -1), (1, True)],
)
def test_loader_rejects_invalid_resolved_token_ids(
    monkeypatch: pytest.MonkeyPatch,
    eos_token_id: object,
    pad_token_id: object,
) -> None:
    dependencies, _, _ = fake_dependencies(
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    install_dependencies(monkeypatch, dependencies)

    with pytest.raises(ModelLoadError, match="token IDs"):
        hf_loader.load_reference_model(EngineConfig())


def test_loader_wraps_and_sanitizes_model_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies, _, _ = fake_dependencies()
    dependencies.auto_model.from_pretrained.side_effect = OSError(
        "secret path C:/Users/name/.cache/model"
    )
    install_dependencies(monkeypatch, dependencies)

    with pytest.raises(ModelLoadError) as captured:
        hf_loader.load_reference_model(EngineConfig())

    message = str(captured.value)
    assert "Qwen/Qwen2.5-0.5B" in message
    assert "060db6499f32faf8b98477b0a26969ef7d8b9987" in message
    assert "OSError" in message
    assert "secret path" not in message


def test_loader_wraps_device_placement_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    dependencies, _, model = fake_dependencies()
    model.to.side_effect = RuntimeError("CUDA unavailable at C:/private/path")
    install_dependencies(monkeypatch, dependencies)

    with pytest.raises(ModelLoadError) as captured:
        hf_loader.load_reference_model(EngineConfig())

    assert "placing model on cuda" in str(captured.value)
    assert "RuntimeError" in str(captured.value)
    assert "private/path" not in str(captured.value)


def test_optional_dependency_failure_is_wrapped_without_leaking_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_: str) -> Any:
        raise ImportError("token=do-not-leak")

    monkeypatch.setattr(hf_loader.importlib, "import_module", fail_import)

    with pytest.raises(ModelLoadError) as captured:
        hf_loader._load_dependencies()

    assert "ImportError" in str(captured.value)
    assert "do-not-leak" not in str(captured.value)
