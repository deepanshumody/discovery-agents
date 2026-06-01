"""ZarrStore: chunked, resizable, on-disk arrays (scalable backend)."""

from __future__ import annotations

import numpy as np


class ZarrStore:
    def __init__(self, path: str, mode: str = "r", chunk_rows: int = 1024) -> None:
        import zarr  # lazy import

        self.path = path
        self.mode = mode
        self.chunk_rows = chunk_rows
        # 'w' overwrites, 'r' read-only, 'a' read/write.
        self._group = zarr.open_group(path, mode=mode)

    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        cols = shape[1:]
        self._group.zeros(
            name,
            shape=(0, *cols),
            chunks=(self.chunk_rows, *cols),
            dtype=dtype,
            overwrite=True,
        )

    def append(self, name: str, rows: np.ndarray) -> None:
        self._group[name].append(np.asarray(rows), axis=0)

    def read(self, name: str, start: int, end: int) -> np.ndarray:
        return np.asarray(self._group[name][start:end])

    def length(self, name: str) -> int:
        return int(self._group[name].shape[0])

    def names(self) -> list[str]:
        return sorted(self._group.array_keys())

    def close(self) -> None:
        # zarr flushes on write; nothing to close for the DirectoryStore.
        return None
