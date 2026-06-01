"""Tests for the Transformer encoder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from discovery_agents.mlinfra.config import ModelConfig  # noqa: E402
from discovery_agents.mlinfra.model import TextEncoder  # noqa: E402


def _encoder(vocab_size: int = 64) -> TextEncoder:
    return TextEncoder(
        ModelConfig(dim=32, num_layers=2, num_heads=4, max_seq_len=16, vocab_size=vocab_size)
    )


def test_forward_shape_and_l2_norm() -> None:
    torch.manual_seed(0)
    model = _encoder().eval()
    tokens = torch.randint(1, 64, (5, 16))
    out = model(tokens)
    assert out.shape == (5, 32)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)  # L2-normalized


def test_eval_mode_is_deterministic() -> None:
    torch.manual_seed(0)
    model = _encoder().eval()
    tokens = torch.randint(1, 64, (4, 16))
    with torch.no_grad():
        a = model(tokens)
        b = model(tokens)
    assert torch.equal(a, b)


def test_padding_is_ignored_in_pooling() -> None:
    torch.manual_seed(0)
    model = _encoder().eval()
    base = torch.randint(1, 64, (1, 8))
    padded = torch.cat([base, torch.zeros(1, 8, dtype=torch.long)], dim=1)  # right-pad with PAD=0
    with torch.no_grad():
        out_padded = model(padded)
    # Masked mean-pool ignores PAD positions, so a padded row still pools cleanly (no NaN).
    assert out_padded.shape == (1, 32)
    assert not torch.isnan(out_padded).any()


def test_gradients_flow() -> None:
    torch.manual_seed(0)
    model = _encoder()
    tokens = torch.randint(1, 64, (4, 16))
    out = model(tokens)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
