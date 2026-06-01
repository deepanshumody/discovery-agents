"""NumpyStore: the light default backend (a directory of .npy arrays).

Writes one ``<name>.npy`` per array and reads via ``mmap`` so reads perform real
(memory-mapped) file access — keeping it comparable to Zarr/HDF5 in the I/O benchmark
rather than being a pure in-RAM slice. Modes: ``w`` (fresh), ``a`` (load + append),
``r`` (mmap reads).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyStore:
    def __init__(self, path: str, mode: str = "r") -> None:
        self.dir = Path(path)
        self.mode = mode
        self._buffers: dict[str, np.ndarray] = {}  # in-memory arrays while writing/appending
        self._memmaps: dict[str, np.ndarray] = {}  # lazily-opened mmaps while reading
        if mode in ("w", "a"):
            self.dir.mkdir(parents=True, exist_ok=True)
            if mode == "a":  # load existing arrays so appends continue (resume-friendly)
                for npy in self.dir.glob("*.npy"):
                    self._buffers[npy.stem] = np.load(npy)
        elif mode == "r":
            if not self.dir.exists():
                raise FileNotFoundError(self.dir)
        else:
            raise ValueError(f"unknown mode: {mode!r} (expected r|w|a)")

    def _path(self, name: str) -> Path:
        return self.dir / f"{name}.npy"

    def _mmap(self, name: str) -> np.ndarray:
        if name not in self._memmaps:
            self._memmaps[name] = np.load(self._path(name), mmap_mode="r")
        return self._memmaps[name]

    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        self._buffers[name] = np.empty((0, *shape[1:]), dtype=dtype)

    def append(self, name: str, rows: np.ndarray) -> None:
        existing = self._buffers[name]
        self._buffers[name] = np.concatenate([existing, np.asarray(rows, dtype=existing.dtype)])

    def read(self, name: str, start: int, end: int) -> np.ndarray:
        if self.mode == "r":
            return np.array(self._mmap(name)[start:end])  # copy out of the memmap
        return self._buffers[name][start:end]

    def length(self, name: str) -> int:
        if self.mode == "r":
            return int(self._mmap(name).shape[0])
        return int(self._buffers[name].shape[0])

    def names(self) -> list[str]:
        if self.mode == "r":
            return sorted(p.stem for p in self.dir.glob("*.npy"))
        return sorted(self._buffers)

    def close(self) -> None:
        if self.mode != "r":
            for name, array in self._buffers.items():
                np.save(self._path(name), array)
