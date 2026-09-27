import pytest
import torch

from tinyinfer.model.kv import ContiguousKV


def _kv(max_tokens: int = 8) -> ContiguousKV:
    return ContiguousKV(
        num_layers=2,
        batch=2,
        num_kv_heads=2,
        head_dim=4,
        max_tokens=max_tokens,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _step(kv: ContiguousKV, tokens: int, fill: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
    outputs = []
    for layer in range(2):
        key = torch.full((2, 2, tokens, 4), fill + layer)
        outputs.append(kv.update(layer, key, -key))
    kv.commit(tokens)
    return outputs


def test_update_returns_history_and_commit_advances_length() -> None:
    kv = _kv()
    _step(kv, 3, 1.0)
    outputs = _step(kv, 1, 5.0)

    assert kv.length == 4
    key, value = outputs[1]
    assert key.shape == (2, 2, 4, 4)
    assert torch.equal(key[:, :, :3], torch.full((2, 2, 3, 4), 2.0))
    assert torch.equal(key[:, :, 3], torch.full((2, 2, 4), 6.0))
    assert torch.equal(value, -key)


def test_length_does_not_move_before_commit() -> None:
    kv = _kv()
    kv.update(0, torch.zeros(2, 2, 3, 4), torch.zeros(2, 2, 3, 4))

    assert kv.length == 0


def test_layers_must_update_in_order() -> None:
    kv = _kv()

    with pytest.raises(RuntimeError, match="order"):
        kv.update(1, torch.zeros(2, 2, 1, 4), torch.zeros(2, 2, 1, 4))


def test_commit_requires_every_layer_with_the_same_token_count() -> None:
    kv = _kv()
    kv.update(0, torch.zeros(2, 2, 2, 4), torch.zeros(2, 2, 2, 4))

    with pytest.raises(RuntimeError, match="layer"):
        kv.commit(2)
    with pytest.raises(RuntimeError, match="tokens"):
        kv.update(1, torch.zeros(2, 2, 1, 4), torch.zeros(2, 2, 1, 4))


def test_capacity_and_shape_are_checked_before_writing() -> None:
    kv = _kv(max_tokens=2)

    with pytest.raises(ValueError, match="capacity"):
        kv.update(0, torch.zeros(2, 2, 3, 4), torch.zeros(2, 2, 3, 4))
    with pytest.raises(ValueError, match="shape"):
        kv.update(0, torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))
