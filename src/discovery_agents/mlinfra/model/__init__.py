"""Model: Transformer sentence encoder + contrastive loss."""

from __future__ import annotations

from .encoder import TextEncoder
from .losses import contrastive_accuracy, info_nce

__all__ = ["TextEncoder", "contrastive_accuracy", "info_nce"]
