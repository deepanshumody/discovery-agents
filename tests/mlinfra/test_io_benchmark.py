"""Tests for the I/O benchmark + resource profiler."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from discovery_agents.mlinfra.bench import ResourceSampler, run  # noqa: E402

_KEYS = {"backend", "write_s", "read_samples_per_s", "read_mb_per_s", "p50_ms", "p95_ms"}


def test_numpy_benchmark_returns_metrics(tmp_path) -> None:
    results = run(["numpy"], str(tmp_path), rows=256, cols=32, batch_size=16, num_batches=8)
    assert len(results) == 1
    result = results[0]
    assert set(result) >= _KEYS
    assert result["read_samples_per_s"] > 0
    assert result["read_mb_per_s"] > 0


def test_benchmark_sweeps_available_backends(tmp_path) -> None:
    backends = ["numpy"]
    try:
        import zarr  # noqa: F401

        backends.append("zarr")
    except ImportError:  # pragma: no cover
        pass
    try:
        import h5py  # noqa: F401

        backends.append("hdf5")
    except ImportError:  # pragma: no cover
        pass
    results = run(backends, str(tmp_path), rows=256, cols=32, batch_size=16, num_batches=8)
    assert {r["backend"] for r in results} == set(backends)
    assert all("error" not in r for r in results)


def test_resource_sampler_records_wall_time() -> None:
    with ResourceSampler() as sampler:
        sum(i * i for i in range(10_000))
    assert sampler.stats["wall_s"] >= 0.0
    assert sampler.stats["rss_mb"] > 0.0
