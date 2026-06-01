"""The ArrayStore protocol: a chunked, appendable multi-dimensional tensor archive.

Row-extensible: ``shape[0]`` grows via ``append``; the trailing dims are fixed (e.g.
``(N, seq_len)`` token rows, or ``(N, C, H, W)`` image tensors). Backends: Numpy
(default), Zarr, HDF5 — they share this interface so curation/training/benchmarks are
backend-agnostic ("abstractions others depend on").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class ArrayStore(Protocol):
    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        """Create a row-extensible array; ``shape[0]`` is the initial row count (often 0)."""
        ...

    def append(self, name: str, rows: np.ndarray) -> None:
        """Append rows along axis 0; ``rows.shape[1:]`` must match the array's trailing dims."""
        ...

    def read(self, name: str, start: int, end: int) -> np.ndarray:
        """Return rows ``[start:end]`` along axis 0."""
        ...

    def length(self, name: str) -> int:
        """Number of rows currently stored in ``name``."""
        ...

    def names(self) -> list[str]: ...

    def close(self) -> None: ...
