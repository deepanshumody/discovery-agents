"""A torch Dataset over the tensor archive (backend-agnostic)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..store.base import ArrayStore


class ArrayStoreDataset(Dataset[torch.Tensor]):
    """Map-style dataset yielding one token row (LongTensor) per index."""

    def __init__(self, store: ArrayStore, name: str = "tokens") -> None:
        self.store = store
        self.name = name
        self._length = store.length(name)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.store.read(self.name, index, index + 1)[0]
        return torch.as_tensor(row, dtype=torch.long)
