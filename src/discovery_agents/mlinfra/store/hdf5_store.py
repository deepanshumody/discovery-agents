"""HDF5Store: chunked, resizable HDF5 datasets (scalable backend)."""

from __future__ import annotations

from typing import Any

import numpy as np


class HDF5Store:
    def __init__(self, path: str, mode: str = "r", chunk_rows: int = 1024) -> None:
        from pathlib import Path

        import h5py  # lazy import

        if mode != "r":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # 'w' truncates, 'r' read-only, 'a' read/write-append.
        self._file: Any = h5py.File(path, mode)
        self.chunk_rows = chunk_rows

    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        cols = shape[1:]
        if name in self._file:
            del self._file[name]
        self._file.create_dataset(
            name,
            shape=(0, *cols),
            maxshape=(None, *cols),
            chunks=(self.chunk_rows, *cols),
            dtype=dtype,
        )

    def append(self, name: str, rows: np.ndarray) -> None:
        dataset = self._file[name]
        rows = np.asarray(rows)
        n = dataset.shape[0]
        dataset.resize(n + rows.shape[0], axis=0)
        dataset[n : n + rows.shape[0]] = rows

    def read(self, name: str, start: int, end: int) -> np.ndarray:
        return np.asarray(self._file[name][start:end])

    def length(self, name: str) -> int:
        return int(self._file[name].shape[0])

    def names(self) -> list[str]:
        return sorted(self._file.keys())

    def close(self) -> None:
        self._file.close()
