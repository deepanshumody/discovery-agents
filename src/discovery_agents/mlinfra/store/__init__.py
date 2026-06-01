"""Multi-backend tensor archive (Numpy default; Zarr/HDF5 scalable backends)."""

from __future__ import annotations

from .base import ArrayStore
from .factory import open_store

__all__ = ["ArrayStore", "open_store"]
