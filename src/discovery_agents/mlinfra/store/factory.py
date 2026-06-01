"""Open a tensor archive by backend name (numpy | zarr | hdf5)."""

from __future__ import annotations

from .base import ArrayStore


def open_store(backend: str, path: str, mode: str = "r") -> ArrayStore:
    backend = backend.lower()
    if backend == "numpy":
        from .numpy_store import NumpyStore

        return NumpyStore(path, mode=mode)
    if backend == "zarr":
        from .zarr_store import ZarrStore

        return ZarrStore(path, mode=mode)
    if backend == "hdf5":
        from .hdf5_store import HDF5Store

        return HDF5Store(path, mode=mode)
    raise ValueError(f"unknown store backend: {backend!r} (expected numpy|zarr|hdf5)")
