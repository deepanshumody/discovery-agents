"""Tests for the SimCSE InfoNCE loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from discovery_agents.mlinfra.model.losses import contrastive_accuracy, info_nce  # noqa: E402


def _normalized(b: int = 8, d: int = 16) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.nn.functional.normalize(torch.randn(b, d), dim=-1)


def test_aligned_views_have_low_loss_and_perfect_accuracy() -> None:
    z = _normalized()
    loss = info_nce(z, z.clone(), temperature=0.05)
    assert loss.item() >= 0.0
    assert contrastive_accuracy(z, z.clone()) == 1.0


def test_misaligned_views_have_higher_loss() -> None:
    z = _normalized()
    aligned = info_nce(z, z.clone())
    perm = torch.randperm(z.shape[0])
    misaligned = info_nce(z, z[perm].clone())
    assert misaligned.item() > aligned.item()


def test_loss_is_scalar() -> None:
    z = _normalized()
    loss = info_nce(z, z.clone())
    assert loss.dim() == 0
