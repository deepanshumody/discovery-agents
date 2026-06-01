"""Data layer: archive-backed Dataset + GPU-native DataLoader."""

from __future__ import annotations

from .dataset import ArrayStoreDataset
from .loader import build_dataloader, collate, to_device

__all__ = ["ArrayStoreDataset", "build_dataloader", "collate", "to_device"]
