"""SimCSE-style in-batch contrastive loss (symmetric InfoNCE)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Symmetric InfoNCE over two views (z1[i], z2[i]) of the same item.

    z1, z2 are (B, dim) L2-normalized embeddings; the positive for row i is the
    i-th row of the other view, negatives are the rest of the batch.
    """
    logits = (z1 @ z2.t()) / temperature  # (B, B) cosine similarities / temp
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def contrastive_accuracy(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Fraction of rows whose nearest neighbor in z2 is the correct positive."""
    logits = z1 @ z2.t()
    labels = torch.arange(z1.shape[0], device=z1.device)
    return float((logits.argmax(dim=1) == labels).float().mean().item())
