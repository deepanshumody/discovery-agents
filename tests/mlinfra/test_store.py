"""Tests for the multi-backend tensor archive (numpy always; zarr/hdf5 if present)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from discovery_agents.mlinfra.store import open_store  # noqa: E402


def _roundtrip(backend: str, path: str) -> None:
    data = np.arange(20, dtype="int64").reshape(5, 4)
    store = open_store(backend, path, mode="w")
    store.create("tokens", (0, 4), "int64")
    store.append("tokens", data[:3])
    store.append("tokens", data[3:])
    assert store.length("tokens") == 5
    np.testing.assert_array_equal(store.read("tokens", 1, 4), data[1:4])
    assert store.names() == ["tokens"]
    store.close()

    reopened = open_store(backend, path, mode="r")
    np.testing.assert_array_equal(reopened.read("tokens", 0, 5), data)
    reopened.close()


def test_numpy_store_roundtrip(tmp_path) -> None:
    _roundtrip("numpy", str(tmp_path / "corpus"))


def test_zarr_store_roundtrip(tmp_path) -> None:
    pytest.importorskip("zarr")
    _roundtrip("zarr", str(tmp_path / "corpus.zarr"))


def test_hdf5_store_roundtrip(tmp_path) -> None:
    pytest.importorskip("h5py")
    _roundtrip("hdf5", str(tmp_path / "corpus.h5"))


def test_backends_agree(tmp_path) -> None:
    pytest.importorskip("zarr")
    pytest.importorskip("h5py")
    data = np.arange(24, dtype="int64").reshape(6, 4)
    reads = {}
    for backend, path in [
        ("numpy", str(tmp_path / "n")),
        ("zarr", str(tmp_path / "z.zarr")),
        ("hdf5", str(tmp_path / "h.h5")),
    ]:
        store = open_store(backend, path, mode="w")
        store.create("tokens", (0, 4), "int64")
        store.append("tokens", data)
        reads[backend] = store.read("tokens", 0, 6)
        store.close()
    np.testing.assert_array_equal(reads["numpy"], reads["zarr"])
    np.testing.assert_array_equal(reads["numpy"], reads["hdf5"])


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown store backend"):
        open_store("spark", "/tmp/x", mode="w")
