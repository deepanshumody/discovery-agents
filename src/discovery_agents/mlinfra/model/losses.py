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


def supervised_contrastive(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020).

    For L2-normalized `embeddings` (B, dim) and integer `labels` (B,), each anchor's
    positives are the other same-label rows in the batch. Anchors with no in-batch
    positive are skipped; returns a 0 (graph-connected) loss if none exist.
    """
    device = embeddings.device
    batch = embeddings.shape[0]
    logits = embeddings @ embeddings.t() / temperature  # (B, B)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # numerical stability

    self_mask = torch.eye(batch, dtype=torch.bool, device=device)
    label_col = labels.view(-1, 1)
    positive_mask = ((label_col == label_col.t()) & ~self_mask).float()  # same label, not self

    # log p(j|i) over all non-self j (zero self out of the denominator, never use -inf
    # so a 0*-inf NaN can't arise).
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positives_per_anchor = positive_mask.sum(dim=1)
    valid = positives_per_anchor > 0
    if not bool(valid.any()):
        return (embeddings * 0.0).sum()  # keep graph connected; the sampler prevents this
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positives_per_anchor[valid]
    return -mean_log_prob_pos.mean()
