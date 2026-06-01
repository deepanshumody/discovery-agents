"""NumpyStore: the light, dependency-minimal default backend (.npz on disk).

Holds arrays in memory and persists to a single ``.npz`` on close; reads load it
back. Fine for the small/default path; Zarr/HDF5 are the scalable, chunked backends.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyStore:
    def __init__(self, path: str, mode: str = "r") -> None:
        self.path = Path(path).with_suffix(".npz")
        self.mode = mode
        self._arrays: dict[str, np.ndarray] = {}
        if mode == "r":
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            with np.load(self.path) as data:
                self._arrays = {k: data[k] for k in data.files}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        self._arrays[name] = np.empty((0, *shape[1:]), dtype=dtype)

    def append(self, name: str, rows: np.ndarray) -> None:
        existing = self._arrays[name]
        self._arrays[name] = np.concatenate([existing, np.asarray(rows, dtype=existing.dtype)])

    def read(self, name: str, start: int, end: int) -> np.ndarray:
        return self._arrays[name][start:end]

    def length(self, name: str) -> int:
        return int(self._arrays[name].shape[0])

    def names(self) -> list[str]:
        return sorted(self._arrays)

    def close(self) -> None:
        if self.mode != "r":
            np.savez(self.path, **self._arrays)
